@echo off
echo.
echo  ███╗   ███╗███████╗      ██████╗  █████╗  ██████╗
echo  ████╗ ████║██╔════╝      ██╔══██╗██╔══██╗██╔════╝
echo  ██╔████╔██║███████╗      ██████╔╝███████║██║  ███╗
echo  ██║╚██╔╝██║╚════██║      ██╔══██╗██╔══██║██║   ██║
echo  ██║ ╚═╝ ██║███████║      ██║  ██║██║  ██║╚██████╔╝
echo  ╚═╝     ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
echo.
echo  MS_RAG — Production-Grade RAG Framework Builder
echo  Installation Script (Windows)
echo ================================================================
echo.

REM Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/5] Upgrading pip...
.venv\Scripts\pip.exe install --upgrade pip --quiet

echo [3/5] Installing core dependencies...
.venv\Scripts\pip.exe install -e . --quiet
if errorlevel 1 (
    echo ERROR: Core installation failed. Check error above.
    pause
    exit /b 1
)

echo [4/5] Installing production extras (vector DBs + evaluators + rerankers)...
echo       This may take several minutes...
.venv\Scripts\pip.exe install -e ".[production]" --quiet

echo [5/5] Verifying installation...
.venv\Scripts\python.exe -c "import ms_rag; from ms_rag.ui.banner import MS_RAG_BANNER; print('  Installation verified OK')"
if errorlevel 1 (
    echo WARNING: Verification failed. Some packages may be missing.
) else (
    echo.
    echo ================================================================
    echo  Installation complete!
    echo ================================================================
    echo.
    echo  To start MS_RAG:
    echo.
    echo    .venv\Scripts\activate
    echo    ms-rag
    echo.
    echo  Or directly:
    echo    .venv\Scripts\ms-rag.exe
    echo.
)
pause
