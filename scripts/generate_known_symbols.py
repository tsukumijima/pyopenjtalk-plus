from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_CSV_PATH = PROJECT_ROOT / "pyopenjtalk" / "dictionary" / "symbols.csv"
OUTPUT_PATH = PROJECT_ROOT / "pyopenjtalk" / "_known_symbols.py"


def main() -> None:
    """同梱辞書から1文字の既知記号と MeCab feature の対応表を生成する。"""

    known_symbols: dict[str, tuple[int, str]] = {}

    # 同じ表層に複数の読みがある場合は、辞書コンパイル時と同じく単語コストが最小の行を採用
    with SYMBOLS_CSV_PATH.open(encoding="utf-8", newline="") as symbols_file:
        for row in csv.reader(symbols_file):
            surface = row[0]
            word_cost = int(row[3])
            if len(surface) != 1:
                continue

            feature = ",".join(row[4:])
            current = known_symbols.get(surface)
            if current is None or word_cost < current[0]:
                known_symbols[surface] = (word_cost, feature)

    # feature と単語コストは Cython から通常の Python 辞書として参照できる形に固定
    output_lines = [
        "# このファイルは scripts/generate_known_symbols.py により自動生成されています",
        "",
        "KNOWN_SYMBOL_FEATURES: dict[str, tuple[str, int]] = {",
    ]
    for surface, (word_cost, feature) in sorted(known_symbols.items()):
        encoded_surface = json.dumps(surface, ensure_ascii=False)
        encoded_feature = json.dumps(feature, ensure_ascii=False)
        output_lines.append(f"    {encoded_surface}: ({encoded_feature}, {word_cost}),")
    output_lines.extend(["}", ""])
    OUTPUT_PATH.write_text("\n".join(output_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
