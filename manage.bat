@echo off
TITLE Black Hole Sim Manager
CLS

ECHO ======================================================
ECHO       EVENT HORIZON SIMULATION - CONTROL PANEL
ECHO ======================================================
ECHO.
ECHO  [1] Start Infrastructure (Docker Redis)
ECHO  [2] Start Worker (Celery - The Chef)
ECHO  [3] Run Connection Test (Check Setup)
ECHO  [4] Stop Infrastructure (Docker Down)
ECHO  [5] Activate Venv (Instruction Only)
ECHO.
ECHO ======================================================
SET /P opt="Select an option: "

IF "%opt%"=="1" GOTO START_DOCKER
IF "%opt%"=="2" GOTO START_WORKER
IF "%opt%"=="3" GOTO RUN_TEST
IF "%opt%"=="4" GOTO STOP_DOCKER
IF "%opt%"=="5" GOTO VENV_INFO

GOTO END

:START_DOCKER
ECHO.
ECHO Starting Redis Container...
docker-compose up -d
ECHO.
ECHO Docker status:
docker ps
PAUSE
GOTO END

:START_WORKER
ECHO.
ECHO Starting Celery Worker...
ECHO (Press Ctrl+C to stop the worker)
ECHO.
REM Check if venv exists, if so use it, else use global
IF EXIST "venv\Scripts\activate.bat" (
    CALL venv\Scripts\activate.bat
    celery -A workers.celery_app worker --pool=solo --loglevel=info
) ELSE (
    ECHO Using Global Python...
    celery -A workers.celery_app worker --pool=solo --loglevel=info
)
GOTO END

:RUN_TEST
ECHO.
ECHO Running Connectivity Test...
IF EXIST "venv\Scripts\activate.bat" CALL venv\Scripts\activate.bat
python check_setup.py
PAUSE
GOTO END

:STOP_DOCKER
ECHO.
ECHO Stopping Containers...
docker-compose down
PAUSE
GOTO END

:VENV_INFO
ECHO.
ECHO To activate virtual environment in your terminal:
ECHO.
ECHO    .\venv\Scripts\activate
ECHO.
ECHO (Use 'deactivate' to exit)
PAUSE
GOTO END

:END