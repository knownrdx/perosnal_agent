# --- build ------------------------------------------------------------------
# Standard library only: no CGO, no external modules, tiny static binary.
FROM golang:1.24-alpine AS build

WORKDIR /src
ENV CGO_ENABLED=0

COPY go.mod ./
COPY main.go ./

RUN go build -trimpath -ldflags="-s -w" -o /out/teams-bridge .

# --- runtime ----------------------------------------------------------------
FROM alpine:3.20

RUN apk add --no-cache ca-certificates tini wget \
    && adduser -D -u 10003 bridge \
    && mkdir -p /data && chown bridge:bridge /data

COPY --from=build /out/teams-bridge /usr/local/bin/teams-bridge

USER bridge
EXPOSE 8082

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["/usr/local/bin/teams-bridge"]
