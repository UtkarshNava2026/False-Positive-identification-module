@echo off
echo ===================================================
echo   False Positive Identification Agent Bootstrapper
echo ===================================================
echo.

:: Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on this computer.
    echo.
    echo Please download and install Python from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, you MUST check the box that says:
    echo            "Add Python to PATH" (at the bottom of the installer window).
    echo.
    pause
    exit /b
)

:: Create Virtual Environment if not present
if not exist venv (
    echo [INFO] Setting up virtual environment... (This only happens on the first run)
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)

:: Activate environment and install dependencies
call venv\Scripts\activate
if not exist venv\.setup_complete (
    echo [INFO] Installing required dependencies... (This may take a couple of minutes)
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-openvino.txt
    if exist YOLOX (
        echo [INFO] Installing YOLOX package...
        pip install -e YOLOX
    )
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause
        exit /b
    )
    echo. > venv\.setup_complete
)

:: Run the Application
echo [INFO] Launching Application...
python detection.py
if %errorlevel% neq 0 (
    echo [WARNING] Application closed with an error code.
    pause
)
