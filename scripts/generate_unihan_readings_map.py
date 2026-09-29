#!/usr/bin/env python3
"""Unihan (ユニコード漢字データベース) の kJapanese から未知漢字用の読み表を生成する。"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path


# 公式配布は zip で、読み表の生成に使うのは中の Unihan_Readings.txt だけである
UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
KATAKANA_RE = re.compile(r"[ァ-ヴー]+$")
HIRAGANA_RE = re.compile(r"[ぁ-ゖー]+$")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "pyopenjtalk" / "_unihan_readings_map.py"


def _to_katakana(text: str) -> str:
    """ひらがなの読みを同じコードポイント配置のカタカナへ変換する。"""

    return "".join(chr(ord(char) + 0x60) if "ぁ" <= char <= "ゖ" else char for char in text)


def _write_python_map(entries: dict[str, str], output_path: Path) -> None:
    """Unihan 読み表を _known_symbols.py と同型の Python 定数へ書き出す。"""

    lines = [
        "# このファイルは scripts/generate_unihan_readings_map.py により自動生成されています",
        "# 出典: Unihan Database (kJapanese). Unicode License v3: https://www.unicode.org/license.txt",
        "",
        "UNIHAN_READINGS: dict[str, str] = {",
    ]
    # 文字列リテラルをダブルクォートで統一するリポジトリ規約に、生成物のエントリ行も合わせる
    for character, reading in sorted(entries.items()):
        lines.append(f'    "{character}": "{reading}",')
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """公式の Unihan.zip を取得し、kJapanese から Python 定数を生成する。"""

    # zip 自体は数 MB なので、作業ディレクトリへ置かずメモリ上で Unihan_Readings.txt だけを取り出す
    with urllib.request.urlopen(UNIHAN_ZIP_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        readings_text = archive.read("Unihan_Readings.txt").decode("utf-8")

    values_by_character: dict[str, list[str]] = {}
    for line in readings_text.splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        codepoint, field, value = line.split("\t", 2)
        if field == "kJapanese":
            # Unihan の U+ 接頭辞を外し、公式ファイルの16進コードポイントを解釈する
            values_by_character[chr(int(codepoint.removeprefix("U+"), 16))] = value.split()

    entries: dict[str, str] = {}
    for character, values in values_by_character.items():
        on_readings = [value for value in values if KATAKANA_RE.fullmatch(value)]
        kun_readings = [value for value in values if HIRAGANA_RE.fullmatch(value)]
        # 辞書に無い漢字は漢語の複合語で現れることが多いので、カタカナの音読みを先に採る
        if len(on_readings) > 0:
            entries[character] = on_readings[0]
        elif len(kun_readings) > 0:
            entries[character] = _to_katakana(kun_readings[0])

    _write_python_map(entries, OUTPUT_PATH)
    print(f"{len(entries):,} characters -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
