@echo off
echo Building PolyCouncil EXE...
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting build process...
echo.

REM Build the EXE using the spec file
pyinstaller build_exe.spec --clean

echo.
echo Build complete!
echo.
echo The EXE can be found in: dist\PolyCouncil.exe
echo.
pause




