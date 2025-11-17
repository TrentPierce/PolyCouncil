@echo off
echo Building PolyCouncil EXE with proper icon embedding...
echo.

REM Step 1: Generate the icon
echo Step 1: Generating icon...
python generate_icon_simple.py
if errorlevel 1 (
    echo Failed to generate icon!
    pause
    exit /b 1
)

REM Step 2: Build the EXE
echo.
echo Step 2: Building EXE...
python -m PyInstaller build_exe.spec --clean
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

REM Step 3: Embed icon using rcedit (if available)
echo.
echo Step 3: Embedding icon...
where rcedit >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: rcedit not found!
    echo.
    echo To properly embed the icon, install rcedit:
    echo   npm install -g rcedit
    echo.
    echo Then run this command manually:
    echo   rcedit dist\PolyCouncil.exe --set-icon PolyCouncil.ico
    echo.
    echo For now, the EXE has been built but the icon may not display correctly.
    echo PyInstaller's icon embedding is unreliable on Windows.
    echo.
) else (
    echo Using rcedit to embed icon...
    rcedit dist\PolyCouncil.exe --set-icon PolyCouncilIco.ico
    if errorlevel 1 (
        echo rcedit failed, but EXE was built
    ) else (
        echo Icon embedded successfully!
    )
)

echo.
echo Build complete!
echo EXE location: dist\PolyCouncil.exe
echo.
pause




