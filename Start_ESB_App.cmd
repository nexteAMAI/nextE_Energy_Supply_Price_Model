@echo off
rem nextE Energy Supply Bid Management Tool - starts the application in the browser (no command line needed).
rem Requires the repository's .venv with the [app] extra installed (pip install -e ".[dev,app]").
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment .venv was not found next to this file.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
