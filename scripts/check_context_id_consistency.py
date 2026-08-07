#!/usr/bin/env python3
"""
辞書 CSV の left-id / right-id と品詞素性列の整合を全行検査する。

naist-jdic の文脈 ID は品詞素性列 (品詞, 細分類1〜3, 活用型, 活用形) の組で決まるため、
手書きエントリで ID と素性列が食い違うと、見た目の品詞と異なる連接コストで解析される。
left-id.def と right-id.def は同一内容 (両方向で同じ分類) であることを前提に検査する。
(なお、naist-jdic に含まれる unidic-csj.csv は、UniDic 2.2.0 をベースに naist-jdic と
被らないエントリだけを、文脈 ID や品詞を naist-jdic の品詞体系に合わせて変換したものである)

Usage:
    uv run python scripts/check_context_id_consistency.py
"""

import csv
import sys
from pathlib import Path


DICTIONARY_DIR = Path(__file__).resolve().parents[1] / "pyopenjtalk" / "dictionary"


def load_context_id_table(definition_path: Path) -> dict[tuple[str, ...], set[int]]:
    """
    left-id.def を読み、6要素の品詞素性タプルから許容文脈 ID 集合への対応表を作る。

    定義の7要素目は '*' のほか頻出語の語彙化 (例: 動詞,非自立,…,連用形,合う) を取り、
    CSV 側は表記違い (ひらがな基本形が漢字語彙化 ID を参照する等) が正当に存在するため、
    同じ6素性に属する全 ID を許容集合として返す。素性分類そのものの食い違いだけを違反にする。

    Args:
        definition_path (Path): left-id.def のパス

    Returns:
        dict[tuple[str, ...], set[int]]: 6素性タプルから許容 ID 集合への対応
    """

    table: dict[tuple[str, ...], set[int]] = {}
    for line in definition_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        id_text, feature_text = line.split(" ", 1)
        features = tuple(feature_text.split(","))
        table.setdefault(features[:6], set()).add(int(id_text))
    return table


def main() -> int:
    """
    全辞書 CSV を検査し、違反行を表示して件数を返す。

    Returns:
        int: 違反が1件でもあれば1、なければ0
    """

    left_definition_path = DICTIONARY_DIR / "left-id.def"
    right_definition_path = DICTIONARY_DIR / "right-id.def"
    # 前提の検査: left と right の分類が同一でなければ「同じ値」の前提自体が崩れている
    if left_definition_path.read_bytes() != right_definition_path.read_bytes():
        print("FATAL: left-id.def and right-id.def differ; the same-id assumption is broken")
        return 1
    context_table = load_context_id_table(left_definition_path)

    violation_count = 0
    for csv_path in sorted(DICTIONARY_DIR.glob("*.csv")):
        csv_name = csv_path.name
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            for line_number, row in enumerate(csv.reader(csv_file), start=1):
                if len(row) < 11:
                    continue
                surface, left_id_text, right_id_text = row[0], row[1], row[2]
                base_features = tuple(row[4:10])
                allowed_ids = context_table.get(base_features)
                if allowed_ids is None:
                    violation_count += 1
                    print(
                        f"{csv_name}:{line_number}: {surface}: "
                        f"no context id for features {','.join(base_features)} (base={row[10]})"
                    )
                    continue
                if int(left_id_text) not in allowed_ids or int(right_id_text) not in allowed_ids:
                    violation_count += 1
                    print(
                        f"{csv_name}:{line_number}: {surface}: id {left_id_text}/{right_id_text} "
                        f"but features {','.join(base_features)} expect {sorted(allowed_ids)}"
                    )
    print(f"violations: {violation_count}")
    return 1 if violation_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
