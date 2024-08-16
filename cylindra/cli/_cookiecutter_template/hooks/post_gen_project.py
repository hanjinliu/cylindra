import shutil
from pathlib import Path

PROJECT_DIRECTORY = Path.cwd()
ALL_TEMP_FOLDERS = ["licenses"]


def remove_temp_folders():
    for folder in ALL_TEMP_FOLDERS:
        folder_path = PROJECT_DIRECTORY / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)


if __name__ == "__main__":
    remove_temp_folders()
