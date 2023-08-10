import PyInstaller.__main__
from argparse import ArgumentParser

def create_executable(site_packages_path=None, exe_type="onefile"):
    cmds = [
        "main.py",
        "--add-data", "externals;externals",
        "--hidden-import", "seaborn",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.signal",
        "--exclude-module", "test", # Yolov7 imports a local test which conflicts with the default test
        "--onedir",
        "--noconfirm"
    ]

    if site_packages_path is not None:
        cmds += ["--paths", site_packages_path]

    cmds += [f"--{exe_type}"]

    PyInstaller.__main__.run(cmds)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--site-packages", type=str, default=None)
    parser.add_argument("--exe_type", type=str, default="onefile")
    args = parser.parse_args()
    create_executable(site_packages_path=args.site_packages, exe_type=args.exe_type)