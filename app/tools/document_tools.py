"""Document reading tools: PDF, Word, Excel, CSV and plain text.

WHY: the agent receives real documents - invoices, contracts, statements,
exports - but could previously only read plain UTF-8 text.  These tools turn a
binary document into text or structured rows the model can reason about, while
keeping three properties the rest of the toolbox relies on:

* every path is jailed inside the workspace through ``safe_path``;
* output is bounded (``max_chars`` / ``max_rows``) so a 200 page PDF can never
  blow up the prompt - ``document_search`` exists precisely so the agent can
  pull the one paragraph it needs instead of reading everything;
* the heavy parsers (pypdf, python-docx, openpyxl) are optional and imported
  lazily inside the handlers, so a missing dependency surfaces as an actionable
  PermanentToolError naming the pip package instead of breaking startup.

Handlers are plain (non-async) functions on purpose: parsing is blocking CPU
work and the tool framework already runs sync handlers in a thread executor,
which keeps the event loop free.
"""

from __future__ import annotations

import csv
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.logging_conf import get_logger
from app.security import Permission, UnsafePath, rel_path, safe_path
from app.tools.base import Arg, InvalidInput, PermanentToolError
from app.tools.registry import tool

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}
_XLSX_EXTS = {".xlsx", ".xlsm"}
_CSV_EXTS = {".csv"}
_TSV_EXTS = {".tsv"}
_TEXT_EXTS = {".txt", ".md", ".json", ".log", ".yaml", ".yml", ".xml", ".html", ".htm"}

_EXT_FORMAT: dict[str, str] = {
    **{ext: "pdf" for ext in _PDF_EXTS},
    **{ext: "docx" for ext in _DOCX_EXTS},
    **{ext: "xlsx" for ext in _XLSX_EXTS},
    **{ext: "csv" for ext in _CSV_EXTS},
    **{ext: "tsv" for ext in _TSV_EXTS},
    **{ext: "text" for ext in _TEXT_EXTS},
}

_TABULAR_FORMATS = {"csv", "tsv", "xlsx"}
_SUPPORTED = ", ".join(sorted(_EXT_FORMAT))

# pip package required for each format, surfaced verbatim in error messages.
_PACKAGES = {"pdf": "pypdf", "docx": "python-docx", "xlsx": "openpyxl"}


@dataclass(slots=True)
class _Unit:
    """One addressable chunk of a document: a PDF page, a sheet, or the whole file."""

    number: int
    label: str
    text: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve(path: str, *, must_exist: bool = True) -> Path:
    try:
        target = safe_path(path, must_exist=must_exist)
    except UnsafePath as exc:
        raise InvalidInput(str(exc)) from exc
    if target.is_dir():
        raise InvalidInput(f"{rel_path(target)} is a directory, not a document")
    return target


def _detect_format(target: Path) -> str:
    fmt = _EXT_FORMAT.get(target.suffix.lower())
    if fmt is None:
        suffix = target.suffix.lower() or "(no extension)"
        raise PermanentToolError(
            f"unsupported document type '{suffix}'; supported extensions: {_SUPPORTED}"
        )
    return fmt


def _check_size(target: Path) -> int:
    size = target.stat().st_size
    limit = get_settings().max_file_bytes
    if size > limit:
        raise PermanentToolError(
            f"{rel_path(target)} is {size} bytes, above the {limit} byte document limit"
        )
    return size


def _import_optional(module: str, package: str) -> Any:
    """Import a parser lazily; a missing wheel must never look like a crash."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise PermanentToolError(
            f"reading this format requires the '{package}' package; "
            f"install it with: pip install {package}"
        ) from exc


def _cell(value: Any) -> str:
    """Render a spreadsheet cell as a single-line string (TSV rows must stay one line)."""
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _tsv_row(values: Any) -> str:
    # Trailing empty cells are noise: read_only sheets often report phantom columns.
    return "\t".join(_cell(value) for value in values).rstrip("\t")


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _delimiter_for(fmt: str, first_line: str) -> str:
    if fmt == "tsv":
        return "\t"
    # Excel in several locales writes ";" separated files with a .csv suffix.
    if "," not in first_line and ";" in first_line:
        return ";"
    return ","


def _parse_page_range(spec: str, total: int) -> list[int]:
    """Turn a 1-indexed spec ('2', '1-3', '1-3,7') into 0-indexed page numbers."""
    if not spec or not spec.strip():
        return list(range(total))

    selected: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part:
                first_text, _, last_text = part.partition("-")
                first, last = int(first_text), int(last_text)
            else:
                first = last = int(part)
        except ValueError as exc:
            raise InvalidInput(
                f"invalid page_range '{spec}'; use forms like '2', '1-3' or '1-3,7'"
            ) from exc
        if first < 1 or last < first:
            raise InvalidInput(f"invalid page_range '{spec}'; pages are 1-indexed ranges")
        for page in range(first, min(last, total) + 1):
            index = page - 1
            if index not in selected:
                selected.append(index)

    if not selected:
        raise InvalidInput(f"page_range '{spec}' selects no pages; document has {total}")
    return selected


# --------------------------------------------------------------------------- #
# Per format extraction
# --------------------------------------------------------------------------- #
def _read_text_file(target: Path) -> str:
    # utf-8-sig transparently drops the BOM Windows tooling likes to add.
    return target.read_text(encoding="utf-8-sig", errors="replace")


def _delimited_records(target: Path, fmt: str, limit: int | None = None) -> list[list[str]]:
    """Read csv/tsv records, optionally stopping after ``limit`` records."""
    with target.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        reader = csv.reader(handle, delimiter=_delimiter_for(fmt, first_line))
        records: list[list[str]] = []
        for row in reader:
            records.append([_cell(value) for value in row])
            if limit is not None and len(records) >= limit:
                break
    return records


def _pdf_units(target: Path, page_range: str) -> tuple[list[_Unit], int]:
    pypdf = _import_optional("pypdf", _PACKAGES["pdf"])
    try:
        reader = pypdf.PdfReader(str(target))
        total = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - malformed/encrypted PDFs are permanent
        raise PermanentToolError(
            f"could not open PDF {rel_path(target)}: {type(exc).__name__}: {exc}"
        ) from exc

    indexes = _parse_page_range(page_range, total)
    units: list[_Unit] = []
    for index in indexes:
        try:
            text = reader.pages[index].extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill the read
            log.warning(
                "pdf_page_failed",
                extra={"tool": "document_read", "page": index + 1, "error": str(exc)[:200]},
            )
            text = ""
        units.append(_Unit(number=index + 1, label=f"page {index + 1}", text=text.strip()))
    return units, total


def _docx_units(target: Path) -> tuple[list[_Unit], int]:
    docx = _import_optional("docx", _PACKAGES["docx"])
    try:
        document = docx.Document(str(target))
    except Exception as exc:  # noqa: BLE001
        raise PermanentToolError(
            f"could not open Word document {rel_path(target)}: {type(exc).__name__}: {exc}"
        ) from exc

    parts = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    # Tables carry the numbers people actually ask about (invoices, statements).
    for table in document.tables:
        for row in table.rows:
            cells = [_cell(cell.text) for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return [_Unit(number=1, label="document", text="\n".join(parts))], 1


def _xlsx_units(target: Path) -> tuple[list[_Unit], int]:
    openpyxl = _import_optional("openpyxl", _PACKAGES["xlsx"])
    try:
        workbook = openpyxl.load_workbook(filename=str(target), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise PermanentToolError(
            f"could not open workbook {rel_path(target)}: {type(exc).__name__}: {exc}"
        ) from exc

    units: list[_Unit] = []
    try:
        for position, name in enumerate(workbook.sheetnames, start=1):
            sheet = workbook[name]
            rows = [_tsv_row(row) for row in sheet.iter_rows(values_only=True)]
            while rows and not rows[-1]:
                rows.pop()
            units.append(_Unit(number=position, label=name, text="\n".join(rows)))
    finally:
        workbook.close()
    return units, len(units)


def _extract_units(target: Path, fmt: str, page_range: str = "") -> tuple[list[_Unit], int]:
    """Return the addressable units of a document plus the document's total unit count."""
    if fmt == "pdf":
        return _pdf_units(target, page_range)
    if fmt == "docx":
        return _docx_units(target)
    if fmt == "xlsx":
        return _xlsx_units(target)
    if fmt in {"csv", "tsv"}:
        rows = _delimited_records(target, fmt)
        text = "\n".join("\t".join(row) for row in rows)
        return [_Unit(number=1, label="document", text=text)], 1
    return [_Unit(number=1, label="document", text=_read_text_file(target))], 1


def _render(units: list[_Unit], fmt: str) -> str:
    blocks: list[str] = []
    for unit in units:
        if fmt == "pdf":
            # Keep the marker even for empty pages: it tells the agent the page
            # was scanned images, not that the document ended.
            blocks.append("[page " + str(unit.number) + "]" + "\n" + unit.text)
        elif fmt == "xlsx":
            blocks.append("## " + unit.label + "\n" + unit.text)
        else:
            blocks.append(unit.text)
    return "\n\n".join(block for block in blocks if block.strip())


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool(
    "document_read",
    description=(
        "Read the text of a document: PDF, Word (.docx), Excel (.xlsx/.xlsm), CSV/TSV or "
        "plain text. Spreadsheets render as '## Sheet' plus TSV rows; page_range like "
        "'1-3' selects PDF pages. Use document_search instead for a long document."
    ),
    permission=Permission.READ,
    args={
        "path": Arg("string", True, "Workspace-relative file path, e.g. downloads/invoice.pdf"),
        "max_chars": Arg("integer", False, "Maximum characters of text to return", default=8000),
        "page_range": Arg("string", False, "PDF pages, 1-indexed, e.g. '2' or '1-3'", default=""),
    },
    timeout_s=120,
    max_retries=0,
)
def document_read(path: str, max_chars: int = 8000, page_range: str = "") -> dict[str, Any]:
    target = _resolve(path)
    fmt = _detect_format(target)
    _check_size(target)
    if max_chars <= 0:
        raise InvalidInput("max_chars must be a positive integer")

    units, total = _extract_units(target, fmt, page_range)
    text = _render(units, fmt)
    truncated = len(text) > max_chars

    log.info(
        "document_read",
        extra={"tool": "document_read", "path": rel_path(target), "format": fmt, "units": len(units)},
    )
    return {
        "path": rel_path(target),
        "format": fmt,
        "pages": len(units),
        "total_pages": total,
        "text": text[:max_chars],
        "truncated": truncated,
    }


@tool(
    "document_info",
    description=(
        "Cheap structural summary of a document (page count, sheet names, row count, "
        "author/title metadata) without extracting all of its text. Call this first to "
        "decide whether to read the whole file or search it."
    ),
    permission=Permission.READ,
    args={"path": Arg("string", True, "Workspace-relative file path")},
    timeout_s=60,
    max_retries=0,
)
def document_info(path: str) -> dict[str, Any]:
    target = _resolve(path)
    fmt = _detect_format(target)
    size = target.stat().st_size

    pages: int | None = None
    sheets: list[dict[str, Any]] | None = None
    rows: int | None = None
    metadata: dict[str, Any] = {}

    if fmt == "pdf":
        pypdf = _import_optional("pypdf", _PACKAGES["pdf"])
        try:
            reader = pypdf.PdfReader(str(target))
            pages = len(reader.pages)
            metadata["encrypted"] = bool(reader.is_encrypted)
            for key, value in dict(reader.metadata or {}).items():
                metadata[str(key).lstrip("/").lower()] = str(value)[:300]
        except PermanentToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PermanentToolError(
                f"could not open PDF {rel_path(target)}: {type(exc).__name__}: {exc}"
            ) from exc

    elif fmt == "docx":
        docx = _import_optional("docx", _PACKAGES["docx"])
        try:
            document = docx.Document(str(target))
        except Exception as exc:  # noqa: BLE001
            raise PermanentToolError(
                f"could not open Word document {rel_path(target)}: {type(exc).__name__}: {exc}"
            ) from exc
        rows = len(document.paragraphs)
        metadata["paragraphs"] = rows
        metadata["tables"] = len(document.tables)
        props = document.core_properties
        for name in ("title", "author", "subject", "last_modified_by"):
            value = getattr(props, name, None)
            if value:
                metadata[name] = str(value)[:300]
        for name in ("created", "modified"):
            value = getattr(props, name, None)
            if value:
                metadata[name] = str(value)

    elif fmt == "xlsx":
        openpyxl = _import_optional("openpyxl", _PACKAGES["xlsx"])
        try:
            workbook = openpyxl.load_workbook(filename=str(target), read_only=True, data_only=True)
        except Exception as exc:  # noqa: BLE001
            raise PermanentToolError(
                f"could not open workbook {rel_path(target)}: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            sheets = []
            for name in workbook.sheetnames:
                sheet = workbook[name]
                max_row = sheet.max_row if isinstance(sheet.max_row, int) else None
                max_col = sheet.max_column if isinstance(sheet.max_column, int) else None
                sheets.append({"name": name, "rows": max_row, "columns": max_col})
            props = getattr(workbook, "properties", None)
            for name in ("title", "creator"):
                value = getattr(props, name, None)
                if value:
                    metadata[name] = str(value)[:300]
        finally:
            workbook.close()
        metadata["sheet_count"] = len(sheets)

    elif fmt in {"csv", "tsv"}:
        records = _delimited_records(target, fmt)
        rows = len(records)
        headers = records[0] if records else []
        metadata["headers"] = headers
        metadata["columns"] = len(headers)
        # ``rows`` counts every record including the header row.
        metadata["data_rows"] = max(rows - 1, 0)

    else:
        with target.open("r", encoding="utf-8-sig", errors="replace") as handle:
            rows = sum(1 for _ in handle)
        metadata["extension"] = target.suffix.lower()

    return {
        "path": rel_path(target),
        "format": fmt,
        "size_bytes": size,
        "pages": pages,
        "sheets": sheets,
        "rows": rows,
        "metadata": metadata,
    }


@tool(
    "spreadsheet_query",
    description=(
        "Return structured rows (list of lists) from an Excel sheet or CSV/TSV file so the "
        "agent can compute with the data instead of reading prose. The first row is treated "
        "as headers. Use the sheet argument to pick a worksheet by name."
    ),
    permission=Permission.READ,
    args={
        "path": Arg("string", True, "Workspace-relative .xlsx/.xlsm/.csv/.tsv file"),
        "sheet": Arg("string", False, "Worksheet name; defaults to the first sheet", default=""),
        "max_rows": Arg("integer", False, "Maximum data rows to return", default=200),
    },
    timeout_s=120,
    max_retries=0,
)
def spreadsheet_query(path: str, sheet: str = "", max_rows: int = 200) -> dict[str, Any]:
    target = _resolve(path)
    fmt = _detect_format(target)
    if fmt not in _TABULAR_FORMATS:
        raise PermanentToolError(
            f"spreadsheet_query needs a csv, tsv or xlsx file but got '{fmt}'; "
            "use document_read for this format"
        )
    if max_rows <= 0:
        raise InvalidInput("max_rows must be a positive integer")
    _check_size(target)

    if fmt == "xlsx":
        name, headers, rows, truncated = _xlsx_rows(target, sheet, max_rows)
    else:
        # Read one extra record so truncation is detected without loading the file twice.
        records = _delimited_records(target, fmt, limit=max_rows + 2)
        headers = records[0] if records else []
        data = records[1:]
        truncated = len(data) > max_rows
        rows = data[:max_rows]
        name = target.name

    return {
        "sheet": name,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def _xlsx_rows(
    target: Path, sheet: str, max_rows: int
) -> tuple[str, list[str], list[list[str]], bool]:
    openpyxl = _import_optional("openpyxl", _PACKAGES["xlsx"])
    try:
        workbook = openpyxl.load_workbook(filename=str(target), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise PermanentToolError(
            f"could not open workbook {rel_path(target)}: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        names = list(workbook.sheetnames)
        if not names:
            raise PermanentToolError(f"{rel_path(target)} contains no worksheets")
        if sheet:
            if sheet not in names:
                raise InvalidInput(
                    f"sheet '{sheet}' not found in {rel_path(target)}; "
                    f"available sheets: {', '.join(names)}"
                )
            name = sheet
        else:
            name = names[0]

        worksheet = workbook[name]
        stream = worksheet.iter_rows(values_only=True)
        headers = [_cell(value) for value in next(stream, ())]
        while headers and not headers[-1]:
            headers.pop()

        rows: list[list[str]] = []
        truncated = False
        for raw in stream:
            values = [_cell(value) for value in raw]
            if not any(values):
                continue  # phantom rows are common in exported workbooks
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append(values)
    finally:
        workbook.close()
    return name, headers, rows, truncated


@tool(
    "document_search",
    description=(
        "Case-insensitive search inside a document (PDF, Word, Excel, CSV or text), "
        "returning short excerpts with surrounding context and the page or sheet they came "
        "from. Use this to answer questions about a long file without reading all of it."
    ),
    permission=Permission.READ,
    args={
        "path": Arg("string", True, "Workspace-relative file path"),
        "query": Arg("string", True, "Text to look for, case-insensitive"),
        "context_chars": Arg("integer", False, "Characters of context per side", default=200),
        "max_matches": Arg("integer", False, "Maximum excerpts to return", default=20),
    },
    timeout_s=180,
    max_retries=0,
)
def document_search(
    path: str, query: str, context_chars: int = 200, max_matches: int = 20
) -> dict[str, Any]:
    target = _resolve(path)
    fmt = _detect_format(target)
    needle = (query or "").strip()
    if not needle:
        raise InvalidInput("query must not be empty")
    if max_matches <= 0:
        raise InvalidInput("max_matches must be a positive integer")
    context = max(0, min(int(context_chars), 2000))
    _check_size(target)

    units, _ = _extract_units(target, fmt, "")
    lowered = needle.lower()
    matches: list[dict[str, Any]] = []
    count = 0

    for unit in units:
        text = unit.text
        haystack = text.lower()
        position = 0
        while True:
            found = haystack.find(lowered, position)
            if found < 0:
                break
            count += 1
            if len(matches) < max_matches:
                start = max(0, found - context)
                end = min(len(text), found + len(needle) + context)
                excerpt = _collapse(text[start:end])
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(text):
                    excerpt = excerpt + "..."
                matches.append({"page": unit.number, "label": unit.label, "excerpt": excerpt})
            position = found + len(lowered)

    return {
        "path": rel_path(target),
        "format": fmt,
        "query": needle,
        "matches": matches,
        "count": count,
        "truncated": count > len(matches),
    }
