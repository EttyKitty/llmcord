@echo off
REM https://github.com/coffin399/llmcord-JP-plana/blob/advanced-bot-utilities/startPLANA.bat

chcp 65001 >nul
title llmcord

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

echo ================================
echo llmcord
echo ================================
echo.

REM Check/Create virtual enviroment
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment in '%VENV_DIR%' folder...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo [ERROR] Please check if Python is installed correctly.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully.
    echo.
) else (
    echo [INFO] Virtual environment already exists.
    echo.
)

REM Activate virtual enviroment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
	echo.
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated.
echo.

REM Verify we are using the correct Python
echo [INFO] Verifying Python location...
"%PYTHON_EXE%" -c "import sys; print('Using Python at: ' + sys.executable)"
echo.

REM Install/Update required packages
echo [INFO] Installing/Updating required packages...
python -m pip install --upgrade pip
python -m pip install -U -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages.
    echo [ERROR] Please check requirements.txt file.
    pause
    exit /b 1
)
echo [SUCCESS] All packages installed successfully.
echo.

REM Start the bot loop
:start_bot
echo ================================
echo llmcord is starting...
echo ================================
echo.

python main.py

REM Check exit code for reload (2)
if %errorlevel% equ 2 (
    echo.
    echo [INFO] Reloading bot...
    echo.
    goto start_bot
)

REM Stop
echo.
echo ================================
echo llmcord has stopped.
echo ================================
pause
