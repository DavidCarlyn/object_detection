import PyInstaller.__main__
from argparse import ArgumentParser

def create_executable(site_packages_path="virtual_environments/test/Lib/site-packages"):
    PyInstaller.__main__.run([
        "main.py",
        "--add-data", "externals;externals",
        "--paths", site_packages_path,
        "--hidden-import", "seaborn",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.signal",
        "--onedir",
        "--console"
    ])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--site-packages", type=str, default="virtual_environments/test/Lib/site-packages")
    args = parser.parse_args()
    create_executable(site_packages_path=args.site_packages)