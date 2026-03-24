@echo off
echo Building PolyCouncil EXE...
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing dependencies...
    pip install -r requirements-dev.txt
)

echo.
echo Starting build process...
echo.

REM Build the EXE using the spec file
python -m PyInstaller build_exe.spec --clean --noconfirm

echo.
echo Build complete!
echo.
echo The EXE can be found in: dist\PolyCouncil.exe
echo.
pause




