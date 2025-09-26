import os
import shutil
import sys
import zipfile
from collections.abc import Iterable

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from expert_pi import version  # noqa: E402


def filter(folder: str, items: list[str]) -> Iterable[str]:
    return ("__pycache__",)


def pack():
    if not os.path.exists(os.path.join(root, "dist")):
        os.mkdir(os.path.join(root, "dist"))

    output_path = os.path.join(root, f"dist/ExpertPI-{version.VERSION}")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    os.mkdir(os.path.join(output_path, "data"))

    shutil.copytree(os.path.join(root, "expert_pi"), os.path.join(output_path, "expert_pi"), ignore=filter)

    shutil.copytree(os.path.join(root, "serving_manager"), os.path.join(output_path, "serving_manager"), ignore=filter)
    shutil.copytree(
        os.path.join(root, "stem_measurements"), os.path.join(output_path, "stem_measurements"), ignore=filter
    )

    os.mkdir(os.path.join(output_path, "config"))
    shutil.copyfile(
        os.path.join(root, "config/config_default.yml"), os.path.join(output_path, "config/config_default.yml")
    )

    shutil.copytree(os.path.join(root, "doc"), os.path.join(output_path, "doc"))
    shutil.copytree(os.path.join(root, "examples"), os.path.join(output_path, "examples"))
    shutil.copytree(os.path.join(root, "scripts"), os.path.join(output_path, "scripts"))

    shutil.copyfile(os.path.join(root, "LICENSE"), os.path.join(output_path, "LICENSE"))
    shutil.copyfile(os.path.join(root, "README.md"), os.path.join(output_path, "README.md"))
    shutil.copyfile(os.path.join(root, "requirements.txt"), os.path.join(output_path, "requirements.txt"))
    shutil.copyfile(os.path.join(root, "start.bat"), os.path.join(output_path, "start.bat"))
    shutil.copyfile(os.path.join(root, "start_with_console.bat"), os.path.join(output_path, "start_with_console.bat"))

    base_dir = os.path.dirname(output_path)
    zip_file = os.path.join(base_dir, f"ExpertPI-{version.VERSION}.zip")

    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for folder, _, files in os.walk(output_path):
            for file in files:
                zipf.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder, file), base_dir))


if __name__ == "__main__":
    pack()
