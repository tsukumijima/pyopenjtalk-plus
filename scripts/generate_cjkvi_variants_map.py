#!/usr/bin/env python3
"""CJKVI 異体字データベースの日本語の字体表から、辞書にない異体字を通用字体へ置き換えるための対応表を生成する。"""

import hashlib
import urllib.request
from collections import defaultdict
from pathlib import Path


# 入力は cjkvi/cjkvi-variants の特定のコミットに固定し、取得したファイルを SHA-256 で照合する
CJKVI_COMMIT = "e4f1da248c9737a243f9930b5dc497cef5d5ae16"
CJKVI_BASE_URL = f"https://raw.githubusercontent.com/cjkvi/cjkvi-variants/{CJKVI_COMMIT}"
# 常用漢字・人名用漢字・表外漢字・戸籍統一文字の順に読み、先の表で決めた対応を優先する
CJKVI_SOURCES = {
    "joyo-variants.txt": "da0666d333c330c07efa2184cdc596bb0de245f65b775c0c805b2b11e6b13c4b",
    "jinmei-variants.txt": "a4dbe76597ca64e6eb2e2b3ec41bab037cb9f443151d02d97a8edfbd9befc6bb",
    "hyogai-variants.txt": "1fb13dfcd96ca212096621cdef5bd27f270ebf8f1fc5834425033be248ce6b3e",
    "koseki-variants.txt": "c33703056001b7b3a01ff860695017ef9cdad0407ee68217fffe538ad68ca032",
}
# 字体の対応があっても、置き換えると別の語の読みになってしまう組
## 「崗」の「コウ」は「岡」を含む地名の「オカ」に、部首名の「巛」の「セン」は「川」を含む人名の「カワ」になる
EXCLUDED_PAIRS = frozenset({("崗", "岡"), ("巛", "川")})

# MIT License は著作権表示と許諾文の同梱を条件にしているので、生成物の先頭コメントに全文を載せる
CJKVI_LICENSE = """MIT License

Copyright (c) 2014 CJKVI Database

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "pyopenjtalk" / "_cjkvi_variants_map.py"


def main() -> None:
    """固定したコミットの字体表を取得し、1字から1字への対応を Python 定数へ書き出す。"""

    mapping: dict[str, str] = {}
    preferred: set[str] = set()
    for filename, digest in CJKVI_SOURCES.items():
        with urllib.request.urlopen(f"{CJKVI_BASE_URL}/{filename}", timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"{filename} の SHA-256 が {CJKVI_COMMIT} の版と一致しません")
        targets_by_variant: dict[str, set[str]] = defaultdict(set)
        for line in payload.decode("utf-8").splitlines():
            fields = line.split(",")
            if len(fields) != 3:
                continue
            proper, relation, variant = fields
            # 1字から1字への字体の対応だけを使い、異体字セレクタを伴う列や関連字・借用字の行は使わない
            if len(proper) != 1 or len(variant) != 1 or relation.endswith("/variant") is False:
                continue
            # 表外漢字の表は、簡易慣用字体を後ろの列に置いているので向きを入れ替える
            if filename == "hyogai-variants.txt":
                proper, variant = variant, proper
            if (variant, proper) in EXCLUDED_PAIRS:
                continue
            targets_by_variant[variant].add(proper)
        # 対応先が複数ある字は除き、先の表で正字に選んだ字を後の表で別の字へ変換しない
        accepted = {
            variant: next(iter(targets))
            for variant, targets in targets_by_variant.items()
            if len(targets) == 1 and variant not in mapping and variant not in preferred
        }
        mapping.update(accepted)
        preferred.update(accepted.values())

    # 対応を辿った先の字へまとめ、循環する対応は除く
    resolved: dict[str, str] = {}
    for variant in mapping:
        seen = {variant}
        proper = mapping[variant]
        while proper in mapping and proper not in seen:
            seen.add(proper)
            proper = mapping[proper]
        if proper not in seen:
            resolved[variant] = proper

    lines = [
        "# このファイルは scripts/generate_cjkvi_variants_map.py により自動生成されています",
        f"# 出典: CJKVI Database (cjkvi/cjkvi-variants {CJKVI_COMMIT})",
        "#",
        *(f"# {line}".rstrip() for line in CJKVI_LICENSE.splitlines()),
        "",
        "CJKVI_VARIANTS: dict[str, str] = {",
    ]
    # 文字列リテラルをダブルクォートで統一するリポジトリ規約に、生成物のエントリ行も合わせる
    for variant, proper in sorted(resolved.items()):
        lines.append(f'    "{variant}": "{proper}",')
    lines.append("}")
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(resolved)} 字: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
