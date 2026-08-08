#!/usr/bin/env python3
"""
naist-jdic.csv の文脈 ID と品詞素性列の不整合を修正する。

check_context_id_consistency.py が検出する violation を解消する。
原則は left-id.def が指す6素性へ ID と品詞列を揃えるが、辞書内先例から
ID 側が誤っている地名・接尾辞等は明示オーバーライドで ID を修正する。

Usage:
    uv run python scripts/fix_context_id_violations.py
"""

import csv
import sys
from pathlib import Path


DICTIONARY_DIR = Path(__file__).resolve().parents[1] / "pyopenjtalk" / "dictionary"
NAIST_JDIC_PATH = DICTIONARY_DIR / "naist-jdic.csv"


def load_id_to_features(definition_path: Path) -> dict[int, tuple[str, ...]]:
    """
    left-id.def から文脈 ID → 6素性タプルへの対応表を作る。

    Args:
        definition_path (Path): left-id.def のパス

    Returns:
        dict[int, tuple[str, ...]]: 文脈 ID から6素性タプルへの対応
    """

    id_to_features: dict[int, tuple[str, ...]] = {}
    for line in definition_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        id_text, feature_text = line.split(" ", 1)
        features = tuple(feature_text.split(","))
        id_to_features[int(id_text)] = features[:6]
    return id_to_features


def load_features_to_ids(definition_path: Path) -> dict[tuple[str, ...], list[int]]:
    """
    left-id.def から6素性タプル → 文脈 ID リストへの対応表を作る。

    Args:
        definition_path (Path): left-id.def のパス

    Returns:
        dict[tuple[str, ...], list[int]]: 6素性タプルから文脈 ID リストへの対応
    """

    features_to_ids: dict[tuple[str, ...], list[int]] = {}
    for line in definition_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        id_text, feature_text = line.split(" ", 1)
        features = tuple(feature_text.split(","))
        features_to_ids.setdefault(features[:6], []).append(int(id_text))
    for feature_key in features_to_ids:
        features_to_ids[feature_key].sort()
    return features_to_ids


def entry_match_key(row: list[str]) -> tuple[str, ...]:
    """
    オーバーライド照合用のエントリキーを作る。

    Args:
        row (list[str]): CSV 行

    Returns:
        tuple[str, ...]: surface, left_id, right_id, orig, read
    """

    return (row[0], row[1], row[2], row[10], row[11])


def explicit_match_key(row: list[str]) -> tuple[str, ...]:
    """
    明示オーバーライド照合用キー。同 surface の自立/非自立等は pos2 を含める。

    Args:
        row (list[str]): CSV 行

    Returns:
        tuple[str, ...]: entry_match_key + pos2
    """

    return (*entry_match_key(row), row[5])


def delete_match_key(row: list[str]) -> tuple[str, ...]:
    """
    削除対象照合用のエントリキーを作る (品詞列まで含む)。

    Args:
        row (list[str]): CSV 行

    Returns:
        tuple[str, ...]: surface, left_id, right_id, orig, read, pos1, pos2, pos3
    """

    return (row[0], row[1], row[2], row[10], row[11], row[4], row[5], row[6])


def build_explicit_overrides() -> dict[tuple[str, ...], dict[str, str | tuple[str, ...]]]:
    """
    品詞列だけでは直せない明示修正 (主に ID 修正) を定義する。

    Returns:
        dict[tuple[str, ...], dict[str, str | tuple[str, ...]]]: エントリキーから修正内容への対応
    """

    # 値: left_id, right_id, features (6tuple) のいずれかを指定
    overrides: dict[tuple[str, ...], dict[str, str | tuple[str, ...]]] = {}

    def set_id(
        surface: str,
        left_id: str,
        right_id: str,
        orig: str,
        read: str,
        new_id: int,
    ) -> None:
        overrides[(surface, left_id, right_id, orig, read)] = {
            "left_id": str(new_id),
            "right_id": str(new_id),
        }

    def set_id_and_features(
        surface: str,
        left_id: str,
        right_id: str,
        orig: str,
        read: str,
        new_id: int,
        features: tuple[str, ...],
    ) -> None:
        overrides[(surface, left_id, right_id, orig, read)] = {
            "left_id": str(new_id),
            "right_id": str(new_id),
            "features": features,
        }

    # --- 連接コストが壊れている単発 ---
    set_id("ちゅう", "187", "855", "ちゅう", "チュウ", 855)

    # --- 地名: 1354(国) → 1353(地域,一般)。品詞列はそのまま ---
    place_name_fixes = [
        ("巴理", "1354", "1354", "巴理", "ニンポウ"),  # typo in read? パリ - check
        ("寧波", "1354", "1354", "寧波", "ニンポウ"),
        ("猶太", "1354", "1354", "猶太", "ユダヤ"),
        ("恵土呂府", "1354", "1354", "恵土呂府", "エトロフ"),
        ("羅馬", "1354", "1354", "羅馬", "ローマ"),
        ("三方ヶ原", "1354", "1354", "三方ヶ原", "ミカタガハラ"),
        ("石川島", "1354", "1354", "石川島", "イシカワジマ"),
        ("鹿ケ谷", "1354", "1354", "鹿ケ谷", "シシガタニ"),
        ("鹿ヶ谷", "1354", "1354", "鹿ヶ谷", "シシガタニ"),
        ("六合", "1350", "1350", "六合", "リクゴウ"),
    ]
    for surface, lid, rid, orig, read in place_name_fixes:
        set_id(surface, lid, rid, orig, read, 1353)

    # 巴理 read is パリ not ニンポウ - fix key
    overrides.pop(("巴理", "1354", "1354", "巴理", "ニンポウ"), None)
    set_id("巴理", "1354", "1354", "巴理", "パリ", 1353)

    # --- 名詞細分類・接尾 ---
    set_id_and_features(
        "初",
        "1358",
        "1358",
        "初",
        "ハツ",
        1347,
        ("名詞", "形容動詞語幹", "*", "*", "*", "*"),
    )
    set_id("爆", "1358", "1358", "爆", "バク", 585)
    set_id("ネイチャー", "1348", "1348", "ネイチャー", "ネイチャー", 1345)
    set_id("ヘソ", "1348", "1348", "ヘソ", "ヘソ", 1345)
    set_id("のりば", "1345", "1345", "のりば", "ノリバ", 1358)
    set_id("研究所", "1345", "1345", "研究所", "ケンキュウジョ", 1358)
    set_id("納め", "1345", "1345", "納め", "オサメ", 1358)
    set_id_and_features(
        "人",
        "1358",
        "1358",
        "人",
        "ジン",
        1363,
        ("名詞", "接尾", "地域", "*", "*", "*"),
    )
    set_id_and_features(
        "ん家",
        "1358",
        "1358",
        "ん家",
        "ンチ",
        1362,
        ("名詞", "接尾", "人名", "*", "*", "*"),
    )
    set_id_and_features(
        "王",
        "1358",
        "1358",
        "王",
        "オウ",
        1362,
        ("名詞", "接尾", "人名", "*", "*", "*"),
    )
    set_id_and_features(
        "売り場",
        "1345",
        "1345",
        "売り場",
        "ウリバ",
        1358,
        ("名詞", "接尾", "一般", "*", "*", "*"),
    )
    set_id_and_features(
        "満点",
        "1345",
        "1345",
        "満点",
        "マンテン",
        1347,
        ("名詞", "形容動詞語幹", "*", "*", "*", "*"),
    )

    # --- 非自立動詞の ID 修正 (自立形と同キーなので pos2 で区別) ---
    overrides[("おける", "645", "645", "おける", "オケル", "非自立")] = {
        "left_id": "958",
        "right_id": "958",
        "features": ("動詞", "非自立", "*", "*", "一段", "基本形"),
    }

    # --- 品詞列のみ特殊 (ID は 1345 のまま) ---
    overrides[("周辺", "1345", "1345", "周辺", "シュウヘン")] = {
        "features": ("名詞", "一般", "*", "*", "*", "*"),
    }
    overrides[("近隣", "1345", "1345", "近隣", "キンリン")] = {
        "features": ("名詞", "一般", "*", "*", "*", "*"),
    }
    overrides[("あと", "1371", "1371", "あと", "アト")] = {
        "features": ("名詞", "非自立", "一般", "*", "*", "*"),
    }
    overrides[("そう", "1369", "1369", "そう", "ソウ")] = {
        "features": ("名詞", "特殊", "助動詞語幹", "*", "*", "*"),
    }
    overrides[("よう", "1373", "1373", "よう", "ヨウ")] = {
        "features": ("名詞", "非自立", "助動詞語幹", "*", "*", "*"),
    }
    overrides[("する", "624", "624", "する", "スル")] = {
        "features": ("動詞", "自立", "*", "*", "サ変・スル", "基本形"),
    }
    overrides[("よる", "1152", "1152", "よる", "ヨル")] = {
        "features": ("動詞", "非自立", "*", "*", "五段・ラ行", "基本形"),
    }
    overrides[("致す", "761", "761", "する", "イタス")] = {
        "features": ("動詞", "自立", "*", "*", "五段・サ行", "基本形"),
    }

    return overrides


DELETE_KEYS: set[tuple[str, ...]] = {
    ("美術館", "1345", "1345", "美術館", "ビジュツカン", "名詞", "地域", "*"),
    ("伝説", "1358", "1358", "伝説", "デンセツ", "名詞", "接尾", "*"),
}


def apply_override(row: list[str], override: dict[str, str | tuple[str, ...]]) -> bool:
    """
    1行へ明示オーバーライドを適用する。

    Args:
        row (list[str]): CSV 行
        override (dict[str, str | tuple[str, ...]]): 適用する修正

    Returns:
        bool: 変更があった場合 True
    """

    changed = False
    if "left_id" in override and row[1] != override["left_id"]:
        row[1] = str(override["left_id"])
        changed = True
    if "right_id" in override and row[2] != override["right_id"]:
        row[2] = str(override["right_id"])
        changed = True
    if "features" in override:
        new_features = override["features"]
        assert isinstance(new_features, tuple)
        if tuple(row[4:10]) != new_features:
            row[4:10] = list(new_features)
            changed = True
    return changed


def sync_features_from_id(row: list[str], id_to_features: dict[int, tuple[str, ...]]) -> bool:
    """
    品詞6列を left-id が指す6素性へ揃える。

    Args:
        row (list[str]): CSV 行
        id_to_features (dict[int, tuple[str, ...]]): 文脈 ID から6素性への対応

    Returns:
        bool: 変更があった場合 True
    """

    left_id = int(row[1])
    id_features = id_to_features.get(left_id)
    if id_features is None:
        return False
    if tuple(row[4:10]) == id_features:
        return False
    row[4:10] = list(id_features)
    return True


def is_violation(
    row: list[str],
    features_to_ids: dict[tuple[str, ...], list[int]],
) -> bool:
    """
    行が文脈 ID 整合チェックの violation か判定する。

    Args:
        row (list[str]): CSV 行
        features_to_ids (dict[tuple[str, ...], list[int]]): 6素性から ID リストへの対応

    Returns:
        bool: violation なら True
    """

    if len(row) < 11:
        return False
    features = tuple(row[4:10])
    allowed_ids = features_to_ids.get(features)
    if allowed_ids is None:
        return True
    left_id = int(row[1])
    right_id = int(row[2])
    return left_id not in allowed_ids or right_id not in allowed_ids


def main() -> int:
    """
    naist-jdic.csv の violation を修正する。

    Returns:
        int: 修正後も violation が残れば 1、なければ 0
    """

    id_to_features = load_id_to_features(DICTIONARY_DIR / "left-id.def")
    features_to_ids = load_features_to_ids(DICTIONARY_DIR / "left-id.def")
    explicit_overrides = build_explicit_overrides()

    with NAIST_JDIC_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    input_row_count = len(rows)

    explicit_count = 0
    sync_count = 0
    deleted_count = 0
    kept_rows: list[list[str]] = []

    for row in rows:
        if len(row) < 11:
            kept_rows.append(row)
            continue

        if delete_match_key(row) in DELETE_KEYS:
            deleted_count += 1
            continue

        match_key = explicit_match_key(row)
        if match_key in explicit_overrides:
            if apply_override(row, explicit_overrides[match_key]):
                explicit_count += 1
        elif entry_match_key(row) in explicit_overrides:
            if apply_override(row, explicit_overrides[entry_match_key(row)]):
                explicit_count += 1

        if is_violation(row, features_to_ids):
            if sync_features_from_id(row, id_to_features):
                sync_count += 1

        kept_rows.append(row)

    rows = kept_rows
    output_row_count = len(rows)
    if output_row_count < input_row_count - deleted_count:
        print(
            f"FATAL: row count dropped unexpectedly "
            f"({input_row_count} -> {output_row_count}, deleted={deleted_count})"
        )
        return 1

    rows.sort(key=lambda row: row[0])

    with NAIST_JDIC_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(rows)

    remaining = 0
    for row in rows:
        if is_violation(row, features_to_ids):
            remaining += 1
            print(
                f"remaining: {row[0]} id={row[1]}/{row[2]} "
                f"features={','.join(row[4:10])} base={row[10]}"
            )

    print(
        f"rows: {input_row_count} -> {output_row_count} "
        f"(explicit overrides: {explicit_count}, sync from id: {sync_count}, "
        f"deleted: {deleted_count}, remaining violations: {remaining})"
    )
    return 1 if remaining > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
