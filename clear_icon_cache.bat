@echo off
echo Clearing Windows icon cache...
echo.

REM Stop Windows Explorer
taskkill /f /im explorer.exe

REM Delete icon cache files
del /a /q /f "%localappdata%\IconCache.db" 2>nul
del /a /q /f "%localappdata%\Microsoft\Windows\Explorer\iconcache*" 2>nul

REM Restart Windows Explorer
start explorer.exe

echo.
echo Icon cache cleared! Please refresh the folder (F5) or restart File Explorer.
echo.
pause




