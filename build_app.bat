@echo off
title Build APP.exe
cd /d "%~dp0"

echo Checking PyInstaller...
python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller
)

if not exist "logo emsi.png" (
    echo WARNING: "logo emsi.png" not found in this folder. EXE will use a generated logo.
)

echo.
echo Building APP.exe (this may take a few minutes)...
python -m PyInstaller --noconfirm --clean APP.spec

if exist "dist\APP.exe" (
    echo.
    echo SUCCESS: dist\APP.exe
    copy /Y "dist\APP.exe" "APP.exe" >nul 2>&1
    if exist "APP.exe" echo Also copied to: %cd%\APP.exe
) else (
    echo.
    echo BUILD FAILED. Check the output above.
    exit /b 1
)

pause
