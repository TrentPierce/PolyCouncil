@echo off
echo Refreshing EXE icon display...
echo.

REM Method 1: Clear icon cache and restart Explorer
echo Step 1: Clearing icon cache...
taskkill /f /im explorer.exe >nul 2>&1
timeout /t 2 /nobreak >nul

del /a /q /f "%localappdata%\IconCache.db" >nul 2>&1
del /a /q /f "%localappdata%\Microsoft\Windows\Explorer\iconcache*" >nul 2>&1

echo Step 2: Restarting Windows Explorer...
start explorer.exe
timeout /t 2 /nobreak >nul

echo.
echo Step 3: Refreshing folder...
REM Open the dist folder and refresh
if exist "dist\PolyCouncil.exe" (
    echo Opening dist folder...
    start "" explorer.exe /select,"%~dp0dist\PolyCouncil.exe"
    timeout /t 1 /nobreak >nul
    REM Send F5 to refresh
    powershell -Command "$wshell = New-Object -ComObject wscript.shell; $wshell.SendKeys('{F5}')"
)

echo.
echo Icon cache cleared and folder refreshed!
echo.
echo If the icon still doesn't show correctly:
echo 1. Close and reopen File Explorer
echo 2. Right-click the EXE > Properties - check if icon appears there
echo 3. Try moving the EXE to a different location
echo.
pause




