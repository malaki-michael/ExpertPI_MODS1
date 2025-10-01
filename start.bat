cd %~dp0

rem deactivate virtualenv, if it's active
call deactivate 2>/nul >nul

call venv\Scripts\activate

start pythonw -m expert_pi
