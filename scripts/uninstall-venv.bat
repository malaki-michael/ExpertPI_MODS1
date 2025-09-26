rem @echo off

rem get parent directory of current file
set PYSTEM_DIR=%~dsp0

rem remove trailing slash
set PYSTEM_DIR=%PYSTEM_DIR:~0,-1%\..\

rem enter PySTEM directory to be able use relative paths
cd %PYSTEM_DIR%


set VENV_DIR=venv
rd /q/s %VENV_DIR%

