# --- build ------------------------------------------------------------------
# whatsmeow requires a recent Go toolchain (>= 1.26).
FROM golang:1.26-bookworm AS build

WORKDIR /src

# CGO is required by go-sqlite3 (whatsmeow session store).
ENV CGO_ENABLED=1 \
    GOFLAGS=-mod=mod \
    GOTOOLCHAIN=auto

# go.mod + go.sum are committed, so the dependency graph is reproducible.
COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./
# tidy resolves any dependency added since go.sum was captured
RUN go mod tidy && go build -trimpath -ldflags="-s -w" -o /out/whatsapp-bridge .

# --- runtime ----------------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini wget \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10002 --create-home --shell /usr/sbin/nologin bridge

COPY --from=build /out/whatsapp-bridge /usr/local/bin/whatsapp-bridge

RUN mkdir -p /session /data && chown -R bridge:bridge /session /data

USER bridge
EXPOSE 8081

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/whatsapp-bridge"]
