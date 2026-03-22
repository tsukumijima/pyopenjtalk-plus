#!/usr/bin/env python3
"""
辞書 CSV ファイルを surface（第1カラム）で昇順ソートするスクリプト。

MeCab 辞書の CSV ファイルは、差分管理の観点から surface の文字コード順にソートされていることが望ましい。
このスクリプトは pyopenjtalk/dictionary/ 以下の全 CSV ファイルを surface でソートし、上書き保存する。

Usage:
    uv run python scripts/sort_dictionary_csv.py
"""

import csv
import io
import sys
from pathlib import Path


def _write_csv_no_trailing_newline(csv_path: Path, rows: list[list[str]]) -> None:
    """
    CSV を末尾改行なしで書き出す。

    csv.writer は各行末に lineterminator を付与するため、最終行にも改行が付く。
    MeCab の辞書 CSV は末尾改行なしが望ましいため、rstrip で除去する。

    Args:
        csv_path (Path): 書き出し先ファイルパス
        rows (list[list[str]]): CSV の各行データ
    """

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    content = buf.getvalue().rstrip("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def sort_csv_by_surface(csv_path: Path) -> int:
    """
    CSV ファイルを surface（第1カラム）で昇順ソートして上書き保存する。

    Args:
        csv_path (Path): ソート対象の CSV ファイルパス

    Returns:
        int: ソートした行数
    """

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        return 0

    # surface（第1カラム）で安定ソート
    rows.sort(key=lambda row: row[0])

    _write_csv_no_trailing_newline(csv_path, rows)

    return len(rows)


def main() -> None:
    """
    pyopenjtalk/dictionary/ 以下の全 CSV ファイルをソートする。
    """

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dictionary_dir = project_root / "pyopenjtalk" / "dictionary"

    if not dictionary_dir.exists():
        print(f"Error: Dictionary directory not found: {dictionary_dir}")
        sys.exit(1)

    csv_files = sorted(dictionary_dir.glob("*.csv"))

    if len(csv_files) == 0:
        print("No CSV files found in dictionary directory.")
        sys.exit(1)

    for csv_path in csv_files:
        row_count = sort_csv_by_surface(csv_path)
        print(f"Sorted {csv_path.name}: {row_count} rows")

    print(f"\nDone. Sorted {len(csv_files)} CSV files.")


if __name__ == "__main__":
    main()
