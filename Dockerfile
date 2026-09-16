FROM python:3.12-slim

# Set to "true" at build time to bake Playwright + Chromium into the image.
ARG INSTALL_BROWSER=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

WORKDIR /app

# Runtime deps only (asyncpg/psutil ship manylinux wheels, no compiler needed).
# gcc is only needed transiently: tgcrypto (Pyrogram's speed-up) has no
# manylinux wheel for every python/arch combo and compiles from source.
# Installed, used, then purged so the final image stays lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$INSTALL_BROWSER" = "true" ]; then \
        pip install --no-cache-dir "playwright>=1.47" && \
        playwright install --with-deps chromium && \
        chmod -R a+rx /opt/playwright ; \
    fi

COPY app ./app
COPY tests ./tests

# Non-root runtime user; /data is the only writable workspace.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin agent \
    && mkdir -p /data \
    && chown -R agent:agent /data /app

USER agent

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "app.main"]
