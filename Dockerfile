# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ARG INSTALL_EXTRAS=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG USE_CONSTRAINTS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    HF_HOME=/workspace/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence-transformers

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        default-jre-headless \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md requirements.txt constraints-production.txt ./
COPY ms_rag ./ms_rag

RUN if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
        python -m pip config set global.extra-index-url "$PIP_EXTRA_INDEX_URL"; \
    fi \
    && python -m pip install --upgrade --retries 10 --timeout 120 pip setuptools wheel \
    && CONSTRAINT_ARGS="" \
    && if [ "$USE_CONSTRAINTS" = "1" ]; then CONSTRAINT_ARGS="-c constraints-production.txt"; fi \
    && for attempt in 1 2 3; do \
        if [ -n "$INSTALL_EXTRAS" ]; then \
            python -m pip install --prefer-binary --retries 10 --timeout 120 $CONSTRAINT_ARGS -e ".[${INSTALL_EXTRAS}]"; \
        else \
            python -m pip install --prefer-binary --retries 10 --timeout 120 $CONSTRAINT_ARGS -e .; \
        fi && break; \
        if [ "$attempt" = "3" ]; then exit 1; fi; \
        echo "pip install failed; retrying in $((attempt * 10)) seconds..."; \
        sleep $((attempt * 10)); \
    done

RUN useradd --create-home --shell /bin/bash msrag \
    && mkdir -p /workspace \
    && chown -R msrag:msrag /workspace /app

USER msrag
WORKDIR /workspace

ENTRYPOINT ["ms-rags"]
CMD []
