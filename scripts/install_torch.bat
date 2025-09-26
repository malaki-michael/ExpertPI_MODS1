set PROJECT_DIR=%~dp0\..\
set CUDA_VERSION=cu118

rem deactivate virtualenv, if it's active
call deactivate 2>/nul >nul

call %PROJECT_DIR%venv\Scripts\activate

call python %PROJECT_DIR%scripts\install_torch.py --cuda-version=%CUDA_VERSION%