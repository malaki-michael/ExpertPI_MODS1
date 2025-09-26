set PROJECT_DIR=%~dp0\..\

@echo on

cd %PROJECT_DIR%
if not exist venv (
    echo "Trying: py -m venv"
    call py -3.12 -m venv venv && goto:INSTALL
    echo "Trying: python -m venv"
    call python -m venv venv && goto:INSTALL

    goto:END
)

:INSTALL

rem deactivate virtualenv, if it's active
call deactivate 2>/nul >nul

call venv\Scripts\activate

call python -m pip install -U pip setuptools wheel uv
call %PROJECT_DIR%scripts\install_torch.bat
call uv pip install -r %PROJECT_DIR%requirements.txt

:END
