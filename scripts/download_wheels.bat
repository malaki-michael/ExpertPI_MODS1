set PROJECT_DIR=%~dp0..\

mkdir %PROJECT_DIR%wheels

py -3.12 -m pip download pip setuptools wheel -d %PROJECT_DIR%wheels
py -3.12 -m pip download -r %PROJECT_DIR%requirements.txt -d %PROJECT_DIR%wheels
