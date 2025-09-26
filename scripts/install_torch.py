import argparse
import functools
import platform
import subprocess
import sys


def install(package, extra_index):
    if isinstance(package, str):
        package = [
            package,
        ]
    subprocess.check_call([
        # sys.executable,
        "uv",
        "pip",
        "install",
        *package,
        "--extra-index-url",
        extra_index,
        "--no-cache-dir",
        "--upgrade",
    ])


def _has_nvidia_gpu():
    nvidia_smi = "nvidia-smi.exe" if "Windows" in platform.system() else "nvidia-smi"
    try:
        subprocess.check_output([nvidia_smi, "--query-gpu=gpu_name", "--format=csv"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def print_index(fn):
    functools.wraps(fn)

    def wrapper(*args, **kwargs):
        index = fn(*args, **kwargs)
        print(f"Using index: {index}")
        return index

    return wrapper


@print_index
def get_extra_index(cuda_version: str, use_cuda: bool = True):
    if use_cuda and _has_nvidia_gpu():
        return f"https://download.pytorch.org/whl/{cuda_version}"
    return "https://download.pytorch.org/whl/cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-version", type=str, default="cu118")
    parser.add_argument("--package", type=str, default="torch,torchvision")

    args = parser.parse_args()
    install(args.package.split(","), get_extra_index(args.cuda_version, use_cuda=args.cuda_version.lower() != "cpu"))


if __name__ == "__main__":
    main()
