@echo off
rem One-click launch: builds the web app if stale, serves it, opens the browser.
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Creating the Python environment ^(first run only^) ...
  python -m venv .venv || goto :error
  "%PY%" -m pip install -e ".[dev,model]" || goto :error
)

"%PY%" -m fedci serve %*
goto :eof

:error
echo.
echo Startup failed. See the messages above.
pause
