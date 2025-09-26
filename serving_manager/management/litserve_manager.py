import datetime
import json
import multiprocessing as mp
from pathlib import Path
import re
import zipfile


class LitserveManager(mp.Process):

    def __init__(self, executable_path: str | Path, port: str | int = "8079"):
        self.executable_path = Path(executable_path).resolve()
        self.port = str(port)

    def infer(self, image, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def clean_up(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.clean_up()


def _is_python_file(file: str | Path) -> bool:
    if isinstance(file, str):
        file = Path(file).resolve()
    return file.suffix == ".py"


def _extract_classes_from_python_file(file: str | Path) -> list[str]:
    if not _is_python_file(file):
        return ["None"]
    if isinstance(file, str):
        file = Path(file).resolve()
    with open(file, "r") as f:
        content = f.read()
    return re.findall(r"class\s+(\w+)\(", content)


def make_executable_model(model_name, litserve_api_file: str | Path, model: str | Path, files_to_include: list[str | Path] | None = None, output_dir: str | Path | None = None):
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir).resolve()

    if files_to_include is None:
        files_to_include = []

    if isinstance(litserve_api_file, str):
        litserve_api_file = Path(litserve_api_file).resolve()
    if isinstance(model, str):
        model = Path(model).resolve()
    files_to_include = [Path(file).resolve() for file in files_to_include]

    model_description = {
        "model_name": model_name,
        "handler": litserve_api_file.name,
        "handler_class": _extract_classes_from_python_file(litserve_api_file)[0],
        "model": model.name,
        "files_to_include": [file.name for file in files_to_include],
        "created": datetime.datetime.now().isoformat(),
        "version": "1.0.0"
    }

    with open(litserve_api_file.parent / "model_description.json", "w") as f:
        json.dump(model_description, f, indent=4)

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_dir / f"{model_name}.zip", "w") as f:
        f.write(litserve_api_file.parent / "model_description.json", "model_description.json")
        f.write(model, model.name)
        f.write(litserve_api_file, litserve_api_file.name)
        for file in files_to_include:
            f.write(file, file.name)
    (litserve_api_file.parent / "model_description.json").unlink()
    return output_dir / f"{model_name}.zip"


if __name__ == "__main__":
    make_executable_model(
        "test_focus",
        Path("/Users/brani/tescan/serving-manager/test_make_executable/litserve_handler.py"),
        Path("/Users/brani/tescan/serving-manager/test_make_executable/hiera_focus.pt"),
        None,
        Path("/Users/brani/tescan/serving-manager/test_make_executable/output")
    )