# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ARG INSTALL_EXTRAS=production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence-transformers

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
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

COPY pyproject.toml README.md requirements.txt ./
COPY ms_rag ./ms_rag

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$INSTALL_EXTRAS" ]; then \
        python -m pip install -e ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install -e .; \
    fi

RUN useradd --create-home --shell /bin/bash msrag \
    && mkdir -p /workspace \
    && chown -R msrag:msrag /workspace /app

USER msrag
WORKDIR /workspace

ENTRYPOINT ["ms-rag"]
CMD []
