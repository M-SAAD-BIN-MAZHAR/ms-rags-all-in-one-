#!/bin/bash

echo ""
echo "  MS_RAG — Production-Grade RAG Framework Builder"
echo "  Installation Script (Linux/macOS)"
echo "================================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment."
    exit 1
fi

echo "[2/5] Upgrading pip..."
.venv/bin/pip install --upgrade pip --quiet

echo "[3/5] Installing core dependencies..."
.venv/bin/pip install -e . --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Core installation failed."
    exit 1
fi

echo "[4/5] Installing production extras (vector DBs + evaluators + rerankers)..."
echo "      This may take several minutes..."
.venv/bin/pip install -e ".[production]" --quiet

echo "[5/5] Verifying installation..."
.venv/bin/python -c "import ms_rag; from ms_rag.ui.banner import MS_RAG_BANNER; print('  Installation verified OK')"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo " Installation complete!"
    echo "================================================================"
    echo ""
    echo " To start MS_RAG:"
    echo ""
    echo "   source .venv/bin/activate"
    echo "   ms-rag"
    echo ""
else
    echo "WARNING: Verification failed. Some packages may be missing."
fi
