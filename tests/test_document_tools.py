"""Tests for the document reading tools.

Everything here runs offline.  The optional parsers (pypdf, python-docx,
openpyxl) are skipped when absent, so the stdlib-only paths - CSV, TSV, plain
text, path jailing and error handling - are always exercised.  The PDF fixture
is a hand-built minimal file so no PDF writer library is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.security import safe_path
from app.tools import registry
from app.tools.base import InvalidInput, PermanentToolError, ToolContext
from app.tools.document_tools import (
    document_info,
    document_read,
    document_search,
    spreadsheet_query,
)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _write(rel: str, content: str) -> Path:
    target = safe_path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _write_bytes(rel: str, content: bytes) -> Path:
    target = safe_path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _minimal_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid PDF with one Helvetica text line per page.

    Written by hand (with a correct xref table) so the test suite does not need
    a PDF writing dependency just to prove the reader works.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_id = add(b"")          # placeholder, patched below
    pages_id = add(b"")            # placeholder
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    for text in pages:
        escaped = text.replace("\\", "").replace("(", "").replace(")", "")
        stream = (
            "BT /F1 18 Tf 72 720 Td (" + escaped + ") Tj ET"
        ).encode("ascii", errors="replace")
        content_id = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add(
            b"<< /Type /Page /Parent " + str(pages_id).encode() + b" 0 R "
            b"/MediaBox [0 0 612 792] /Resources << /Font << /F1 "
            + str(font_id).encode()
            + b" 0 R >> >> /Contents "
            + str(content_id).encode()
            + b" 0 R >>"
        )
        page_ids.append(page_id)

    kids = b" ".join(str(pid).encode() + b" 0 R" for pid in page_ids)
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode() + b" >>"
    )
    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += str(offset).zfill(10).encode() + b" 00000 n \n"
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_id).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


CSV_TEXT = "name,qty,price\nwidget,2,9.50\ngadget,1,120.00\nbolt,40,0.25\n"
TSV_TEXT = "name\tqty\tprice\nwidget\t2\t9.50\ngadget\t1\t120.00\n"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
async def test_tools_are_registered(environment):
    names = registry.names()
    for expected in ("document_read", "document_info", "spreadsheet_query", "document_search"):
        assert expected in names, f"missing tool: {expected}"
    assert registry.get("document_read").permission.value == "READ"


async def test_runs_through_the_tool_framework(environment):
    _write("temp/sample.csv", CSV_TEXT)
    result = await registry.get("document_read").run({"path": "temp/sample.csv"}, ToolContext())
    assert result.ok, result.error
    assert result.data["format"] == "csv"
    assert "widget" in result.data["text"]


async def test_framework_rejects_unknown_argument(environment):
    _write("temp/sample.csv", CSV_TEXT)
    result = await registry.get("document_read").run(
        {"path": "temp/sample.csv", "bogus": 1}, ToolContext()
    )
    assert not result.ok and "unknown argument" in result.error


# --------------------------------------------------------------------------- #
# CSV / TSV
# --------------------------------------------------------------------------- #
async def test_document_read_csv(environment):
    _write("temp/sample.csv", CSV_TEXT)
    out = document_read("temp/sample.csv")
    assert out["format"] == "csv"
    assert out["pages"] == 1
    assert out["truncated"] is False
    lines = out["text"].splitlines()
    assert lines[0] == "name\tqty\tprice"
    assert lines[1] == "widget\t2\t9.50"
    assert len(lines) == 4


async def test_document_read_csv_with_semicolon_delimiter(environment):
    _write("temp/euro.csv", "name;qty\nwidget;2\n")
    out = document_read("temp/euro.csv")
    assert out["text"].splitlines()[0] == "name\tqty"


async def test_document_read_tsv(environment):
    _write("temp/sample.tsv", TSV_TEXT)
    out = document_read("temp/sample.tsv")
    assert out["format"] == "tsv"
    assert out["text"].splitlines()[0] == "name\tqty\tprice"
    assert "gadget\t1\t120.00" in out["text"]


async def test_document_info_csv_counts_rows_and_headers(environment):
    target = _write("temp/sample.csv", CSV_TEXT)
    out = document_info("temp/sample.csv")
    assert out["format"] == "csv"
    assert out["rows"] == 4                      # header + 3 data rows
    assert out["metadata"]["data_rows"] == 3
    assert out["metadata"]["headers"] == ["name", "qty", "price"]
    # Compare against bytes on disk: Windows rewrites "\n" as CRLF on text writes.
    assert out["size_bytes"] == target.stat().st_size
    assert out["path"].endswith("sample.csv")


# --------------------------------------------------------------------------- #
# Plain text formats
# --------------------------------------------------------------------------- #
async def test_document_read_txt_and_md(environment):
    _write("temp/notes.txt", "plain text body")
    _write("temp/readme.md", "# Title\n\nsome markdown")

    assert document_read("temp/notes.txt")["text"] == "plain text body"
    md = document_read("temp/readme.md")
    assert md["format"] == "text"
    assert "# Title" in md["text"]


async def test_document_read_json(environment):
    payload = {"invoice": 42, "total": 99.5}
    _write("temp/data.json", json.dumps(payload))
    out = document_read("temp/data.json")
    assert out["format"] == "text"
    assert json.loads(out["text"]) == payload


async def test_document_read_strips_utf8_bom(environment):
    _write_bytes("temp/bom.txt", "\ufeffhello".encode("utf-8"))
    assert document_read("temp/bom.txt")["text"] == "hello"


async def test_document_info_text_counts_lines(environment):
    _write("temp/app.log", "line one\nline two\nline three\n")
    out = document_info("temp/app.log")
    assert out["format"] == "text"
    assert out["rows"] == 3
    assert out["metadata"]["extension"] == ".log"


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
async def test_max_chars_truncates(environment):
    _write("temp/big.txt", "x" * 5000)
    out = document_read("temp/big.txt", max_chars=100)
    assert out["truncated"] is True
    assert len(out["text"]) == 100


async def test_max_chars_not_truncated_when_short(environment):
    _write("temp/small.txt", "short")
    out = document_read("temp/small.txt", max_chars=100)
    assert out["truncated"] is False
    assert out["text"] == "short"


async def test_max_chars_must_be_positive(environment):
    _write("temp/small.txt", "short")
    with pytest.raises(InvalidInput):
        document_read("temp/small.txt", max_chars=0)


# --------------------------------------------------------------------------- #
# Security and error handling
# --------------------------------------------------------------------------- #
async def test_path_traversal_is_blocked(environment):
    with pytest.raises(InvalidInput):
        document_read("../../etc/passwd")
    with pytest.raises(InvalidInput):
        document_info("../../../etc/passwd")
    with pytest.raises(InvalidInput):
        document_search("../secrets.txt", "token")
    with pytest.raises(InvalidInput):
        spreadsheet_query("../../escape.csv")


async def test_missing_file_raises_invalid_input(environment):
    with pytest.raises(InvalidInput):
        document_read("temp/does_not_exist.pdf")
    with pytest.raises(InvalidInput):
        document_info("temp/does_not_exist.csv")


async def test_directory_raises_invalid_input(environment):
    folder = safe_path("temp/adir")
    folder.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidInput):
        document_read("temp/adir")


async def test_unsupported_extension_raises_permanent_error(environment):
    _write_bytes("temp/malware.exe", b"MZ\x00\x00binary")
    with pytest.raises(PermanentToolError) as excinfo:
        document_read("temp/malware.exe")
    message = str(excinfo.value)
    assert ".exe" in message
    assert "unsupported document type" in message


async def test_spreadsheet_query_rejects_non_tabular(environment):
    _write("temp/notes.txt", "not a table")
    with pytest.raises(PermanentToolError):
        spreadsheet_query("temp/notes.txt")


async def test_empty_query_rejected(environment):
    _write("temp/notes.txt", "body")
    with pytest.raises(InvalidInput):
        document_search("temp/notes.txt", "   ")


# --------------------------------------------------------------------------- #
# document_search
# --------------------------------------------------------------------------- #
async def test_document_search_finds_term_with_context(environment):
    body = ("filler " * 60) + "The grand Total due is 4242 EUR " + ("tail " * 60)
    _write("temp/invoice.txt", body)

    out = document_search("temp/invoice.txt", "total", context_chars=40)
    assert out["count"] == 1
    assert len(out["matches"]) == 1
    match = out["matches"][0]
    assert match["page"] == 1
    assert "total" in match["excerpt"].lower()
    assert "4242" in match["excerpt"]
    assert match["excerpt"].startswith("...")


async def test_document_search_counts_all_occurrences(environment):
    _write("temp/repeat.txt", "alpha beta alpha beta ALPHA")
    out = document_search("temp/repeat.txt", "alpha")
    assert out["count"] == 3
    assert len(out["matches"]) == 3


async def test_document_search_respects_max_matches(environment):
    _write("temp/repeat.txt", "hit " * 20)
    out = document_search("temp/repeat.txt", "hit", max_matches=5)
    assert out["count"] == 20
    assert len(out["matches"]) == 5
    assert out["truncated"] is True


async def test_document_search_no_match(environment):
    _write("temp/notes.txt", "nothing relevant here")
    out = document_search("temp/notes.txt", "zebra")
    assert out["count"] == 0 and out["matches"] == []


async def test_document_search_inside_csv(environment):
    _write("temp/sample.csv", CSV_TEXT)
    out = document_search("temp/sample.csv", "gadget")
    assert out["count"] == 1
    assert "gadget" in out["matches"][0]["excerpt"]


# --------------------------------------------------------------------------- #
# spreadsheet_query on CSV
# --------------------------------------------------------------------------- #
async def test_spreadsheet_query_csv(environment):
    _write("temp/sample.csv", CSV_TEXT)
    out = spreadsheet_query("temp/sample.csv")
    assert out["headers"] == ["name", "qty", "price"]
    assert out["row_count"] == 3
    assert out["rows"][0] == ["widget", "2", "9.50"]
    assert out["rows"][2] == ["bolt", "40", "0.25"]
    assert out["truncated"] is False


async def test_spreadsheet_query_csv_max_rows(environment):
    _write("temp/sample.csv", CSV_TEXT)
    out = spreadsheet_query("temp/sample.csv", max_rows=2)
    assert out["row_count"] == 2
    assert out["truncated"] is True


async def test_spreadsheet_query_tsv(environment):
    _write("temp/sample.tsv", TSV_TEXT)
    out = spreadsheet_query("temp/sample.tsv")
    assert out["headers"] == ["name", "qty", "price"]
    assert out["row_count"] == 2


async def test_spreadsheet_query_rejects_bad_max_rows(environment):
    _write("temp/sample.csv", CSV_TEXT)
    with pytest.raises(InvalidInput):
        spreadsheet_query("temp/sample.csv", max_rows=0)


# --------------------------------------------------------------------------- #
# XLSX (openpyxl)
# --------------------------------------------------------------------------- #
def _make_xlsx(rel: str) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    target = safe_path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    for row in (["region", "amount"], ["north", 100], ["south", 250]):
        sheet.append(row)
    second = workbook.create_sheet("Notes")
    second.append(["comment"])
    second.append(["quarterly review"])
    workbook.save(str(target))
    return target


async def test_xlsx_round_trip(environment):
    pytest.importorskip("openpyxl")
    _make_xlsx("temp/book.xlsx")

    out = document_read("temp/book.xlsx")
    assert out["format"] == "xlsx"
    assert out["pages"] == 2
    assert "## Sales" in out["text"]
    assert "## Notes" in out["text"]
    assert "region\tamount" in out["text"]
    assert "south\t250" in out["text"]


async def test_xlsx_info_lists_sheets(environment):
    pytest.importorskip("openpyxl")
    _make_xlsx("temp/book.xlsx")

    out = document_info("temp/book.xlsx")
    assert out["format"] == "xlsx"
    assert [sheet["name"] for sheet in out["sheets"]] == ["Sales", "Notes"]
    assert out["metadata"]["sheet_count"] == 2
    assert out["size_bytes"] > 0


async def test_spreadsheet_query_xlsx_default_and_named_sheet(environment):
    pytest.importorskip("openpyxl")
    _make_xlsx("temp/book.xlsx")

    first = spreadsheet_query("temp/book.xlsx")
    assert first["sheet"] == "Sales"
    assert first["headers"] == ["region", "amount"]
    assert first["row_count"] == 2
    assert first["rows"] == [["north", "100"], ["south", "250"]]

    named = spreadsheet_query("temp/book.xlsx", sheet="Notes")
    assert named["sheet"] == "Notes"
    assert named["headers"] == ["comment"]
    assert named["row_count"] == 1


async def test_spreadsheet_query_unknown_sheet(environment):
    pytest.importorskip("openpyxl")
    _make_xlsx("temp/book.xlsx")
    with pytest.raises(InvalidInput):
        spreadsheet_query("temp/book.xlsx", sheet="Missing")


async def test_document_search_xlsx_reports_sheet(environment):
    pytest.importorskip("openpyxl")
    _make_xlsx("temp/book.xlsx")

    out = document_search("temp/book.xlsx", "quarterly")
    assert out["count"] == 1
    assert out["matches"][0]["label"] == "Notes"
    assert "quarterly" in out["matches"][0]["excerpt"].lower()


# --------------------------------------------------------------------------- #
# DOCX (python-docx)
# --------------------------------------------------------------------------- #
def _make_docx(rel: str) -> Path:
    docx = pytest.importorskip("docx")
    target = safe_path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)

    document = docx.Document()
    document.add_paragraph("Contract heading")
    document.add_paragraph("The total amount is 1234 USD.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "item"
    table.cell(0, 1).text = "cost"
    table.cell(1, 0).text = "license"
    table.cell(1, 1).text = "500"
    document.save(str(target))
    return target


async def test_docx_round_trip(environment):
    pytest.importorskip("docx")
    _make_docx("temp/contract.docx")

    out = document_read("temp/contract.docx")
    assert out["format"] == "docx"
    assert out["pages"] == 1
    assert "Contract heading" in out["text"]
    assert "The total amount is 1234 USD." in out["text"]
    assert "license\t500" in out["text"]


async def test_docx_info(environment):
    pytest.importorskip("docx")
    _make_docx("temp/contract.docx")

    out = document_info("temp/contract.docx")
    assert out["format"] == "docx"
    assert out["metadata"]["tables"] == 1
    assert out["metadata"]["paragraphs"] >= 2


async def test_docx_search(environment):
    pytest.importorskip("docx")
    _make_docx("temp/contract.docx")

    out = document_search("temp/contract.docx", "TOTAL AMOUNT")
    assert out["count"] == 1
    assert "1234" in out["matches"][0]["excerpt"]


# --------------------------------------------------------------------------- #
# PDF (pypdf)
# --------------------------------------------------------------------------- #
async def test_pdf_round_trip(environment):
    pytest.importorskip("pypdf")
    _write_bytes(
        "temp/report.pdf",
        _minimal_pdf(["Page one Invoice Total 4242", "Page two appendix", "Page three notes"]),
    )

    out = document_read("temp/report.pdf")
    assert out["format"] == "pdf"
    assert out["pages"] == 3
    assert out["total_pages"] == 3
    assert "Invoice Total 4242" in out["text"]
    assert "appendix" in out["text"]
    assert "[page 2]" in out["text"]


async def test_pdf_page_range(environment):
    pytest.importorskip("pypdf")
    _write_bytes(
        "temp/report.pdf",
        _minimal_pdf(["alpha page", "beta page", "gamma page", "delta page"]),
    )

    single = document_read("temp/report.pdf", page_range="2")
    assert single["pages"] == 1
    assert single["total_pages"] == 4
    assert "beta" in single["text"]
    assert "alpha" not in single["text"]

    span = document_read("temp/report.pdf", page_range="1-3")
    assert span["pages"] == 3
    assert "gamma" in span["text"]
    assert "delta" not in span["text"]


async def test_pdf_invalid_page_range(environment):
    pytest.importorskip("pypdf")
    _write_bytes("temp/report.pdf", _minimal_pdf(["only page"]))
    with pytest.raises(InvalidInput):
        document_read("temp/report.pdf", page_range="abc")
    with pytest.raises(InvalidInput):
        document_read("temp/report.pdf", page_range="0-2")


async def test_pdf_info(environment):
    pytest.importorskip("pypdf")
    _write_bytes("temp/report.pdf", _minimal_pdf(["one", "two"]))

    out = document_info("temp/report.pdf")
    assert out["format"] == "pdf"
    assert out["pages"] == 2
    assert out["metadata"]["encrypted"] is False


async def test_pdf_search_reports_page_number(environment):
    pytest.importorskip("pypdf")
    _write_bytes(
        "temp/report.pdf",
        _minimal_pdf(["intro section", "the Grand Total is 987", "closing"]),
    )

    out = document_search("temp/report.pdf", "grand total")
    assert out["count"] == 1
    assert out["matches"][0]["page"] == 2
    assert "987" in out["matches"][0]["excerpt"]


# --------------------------------------------------------------------------- #
# Missing optional dependency
# --------------------------------------------------------------------------- #
async def test_missing_library_raises_named_permanent_error(environment, monkeypatch):
    """A missing wheel must name the pip package, never leak a bare ImportError."""
    import app.tools.document_tools as module

    real_import = module.importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    _write_bytes("temp/report.pdf", b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(PermanentToolError) as excinfo:
        document_read("temp/report.pdf")
    assert "pip install pypdf" in str(excinfo.value)
