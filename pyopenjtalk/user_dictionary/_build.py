from pathlib import Path

import pyopenjtalk


def is_csv_file(file: Path) -> bool:
    supported_extensions = [".csv"]
    return file.suffix.lower() in supported_extensions


input_dir_path = Path("pyopenjtalk/user_dict_src")
csv_files = [file for file in input_dir_path.rglob("*") if is_csv_file(file)]

for file in csv_files:
    pyopenjtalk.mecab_dict_index(str(file), f"./pyopenjtalk/user_dictionary/{file.stem}.dic")
