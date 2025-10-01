cd %~dp0

rem deactivate virtualenv, if it's active
call deactivate 2>/nul >nul

call venv\Scripts\activate

start python expert_pi\app\ipython_console.py
