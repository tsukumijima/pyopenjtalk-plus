"""MeCab lattice のコスト変更と辞書由来情報の受け渡しを確認する。"""

import math
from collections.abc import Sequence
from ctypes import c_long, sizeof
from pathlib import Path
from typing import Any, cast

import pytest

import pyopenjtalk
from pyopenjtalk.types import MeCabCostCandidate


def _feature_from_morphs(morphs: Sequence[Any]) -> list[str]:
    """既存の run_mecab() と同じ条件で feature 列を復元する"""

    return [
        ",".join(cast(list[str], morph["features"]))
        for morph in morphs
        if morph["is_ignored"] is False
    ]


def _zero_adjuster(candidates: list[MeCabCostCandidate]) -> list[float]:
    """全候補ノードを補正しない Δc=0 の cost_adjuster"""

    return [0.0 for _ in candidates]


def _reading(morph: Any) -> str:
    """反転テストで読みだけを比較する"""

    features = cast(list[str], morph["features"])
    assert isinstance(features, list)
    return features[8]


def test_mecab_morphs_report_dictionary_load_order(tmp_path: Path):
    """system は0、複数の OpenJTalk 用ユーザー辞書は読込順の1..Nとして由来を返す"""

    user_dictionary_paths: list[Path] = []
    for dictionary_number, (surface, pronunciation) in enumerate(
        (("辞書由来一", "ジショユライイチ"), ("辞書由来二", "ジショユライニ")),
        start=1,
    ):
        csv_path = tmp_path / f"user-{dictionary_number}.csv"
        dictionary_path = tmp_path / f"user-{dictionary_number}.dic"
        csv_path.write_text(
            f"{surface},,,1000,名詞,一般,*,*,*,*,{surface},{pronunciation},{pronunciation},0/8,C1\n",
            encoding="utf-8",
        )
        pyopenjtalk.mecab_dict_index(str(csv_path), str(dictionary_path))
        user_dictionary_paths.append(dictionary_path)

    jtalk = pyopenjtalk.OpenJTalk(
        dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR,
        userdic=",".join(str(path) for path in user_dictionary_paths).encode("utf-8"),
        userdic_reading_protection=[False, True],
    )
    system_morph = next(
        morph for morph in jtalk.run_mecab_detailed("学生") if morph["surface"] == "学生"
    )
    first_user_morph = jtalk.run_mecab_detailed("辞書由来一")[0]
    second_user_morph = jtalk.run_mecab_detailed("辞書由来二")[0]
    unknown_morph = jtalk.run_mecab_detailed("𰻞𰻞麺")[0]
    seen_candidates: list[MeCabCostCandidate] = []

    def capture_candidates(candidates: list[MeCabCostCandidate]) -> list[float]:
        """辞書由来情報を記録し、コストを変更せず返す。"""

        seen_candidates.extend(candidates)
        return [0.0 for _ in candidates]

    jtalk.run_mecab_with_cost_adjustments("辞書由来一", capture_candidates)
    selected = jtalk.run_mecab_with_cost_adjustments("辞書由来二", capture_candidates)

    assert system_morph["dictionary_index"] == 0
    assert first_user_morph["dictionary_index"] == 1
    assert second_user_morph["dictionary_index"] == 2
    assert unknown_morph["is_unknown"] is True
    assert unknown_morph["dictionary_index"] == 255
    assert selected["morphs"][0]["dictionary_index"] == 2
    assert (
        any(
            candidate["surface"] == "辞書由来一" and candidate["is_reading_protected"] is False
            for candidate in seen_candidates
        )
        is True
    )
    assert any(
        candidate["surface"] == "辞書由来二" and candidate["is_reading_protected"] is True
        for candidate in seen_candidates
    )


def test_openjtalk_rejects_mismatched_user_dictionary_protection(tmp_path: Path):
    """OpenJTalk 用ユーザー辞書数と読み保護フラグ数の不一致を初期化時に拒否する"""

    user_csv = tmp_path / "user.csv"
    user_dictionary = tmp_path / "user.dic"
    user_csv.write_text(
        "辞書由来,,,1000,名詞,一般,*,*,*,*,辞書由来,ジショユライ,ジショユライ,0/6,C1\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dictionary))
    with pytest.raises(ValueError, match="same number"):
        pyopenjtalk.OpenJTalk(
            dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR,
            userdic=str(user_dictionary).encode("utf-8"),
            userdic_reading_protection=[],
        )


def test_run_mecab_with_cost_adjustments_zero_delta_matches_existing_one_best():
    """Δc=0 では既存 one-best と feature、morph、合計コストが完全一致する"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    texts = [
        "こんにちは世界",
        "その方が良い",
        "東京　大阪",
        "大分県にもう大分長いこと住んでいる",
        "今日は2112年9月3日です",
    ]

    for text in texts:
        expected_features = jtalk.run_mecab(text)
        expected_morphs = jtalk.run_mecab_detailed(text)
        captured_candidates: list[MeCabCostCandidate] = []

        def capture_zero(candidates: list[MeCabCostCandidate]) -> list[float]:
            """候補列を記録し、全候補のコスト差をゼロにする。"""

            captured_candidates.extend(candidates)
            return _zero_adjuster(candidates)

        result = jtalk.run_mecab_with_cost_adjustments(text, capture_zero)

        assert len(captured_candidates) > 0
        assert result["features"] == expected_features
        assert result["morphs"] == expected_morphs
        assert result["path_cost"] == result["base_path_cost"]
        assert result["clipped_node_count"] == 0


def test_complete_path_cost_matches_forced_base_path_cost_for_every_candidate():
    """各候補を通る完全経路費用が強制選択した補正前経路費用と一致する"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    captured_candidates: list[MeCabCostCandidate] = []

    def capture_zero(candidates: list[MeCabCostCandidate]) -> list[float]:
        """完全経路費用を含む候補列を記録し、既定経路を維持する。"""

        captured_candidates.extend(candidates)
        return _zero_adjuster(candidates)

    jtalk.run_mecab_with_cost_adjustments("その方が良い", capture_zero)

    # 制御ノードと補正対象外ノードを除く全候補について、十分な負の補正で当該ノードを経路へ含める
    for candidate in captured_candidates:
        if candidate["surface"] == "" or candidate["is_ignored"] is True:
            continue
        target_node_index = candidate["node_index"]

        def force_candidate(candidates: list[MeCabCostCandidate]) -> list[float]:
            """検証対象の1ノードだけを short 下限まで優先する。"""

            return [-100.0 if row["node_index"] == target_node_index else 0.0 for row in candidates]

        forced = jtalk.run_mecab_with_cost_adjustments("その方が良い", force_candidate)

        assert target_node_index in forced["node_indices"]
        assert forced["base_path_cost"] == candidate["complete_path_cost"]


def test_complete_path_cost_decomposes_into_forward_and_backward_costs():
    """完全経路費用が全候補で前向き費用と後向き費用の和になる"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    captured_candidates: list[MeCabCostCandidate] = []

    def capture_zero(candidates: list[MeCabCostCandidate]) -> list[float]:
        """候補ごとの前向き・後向き費用を記録する。"""

        captured_candidates.extend(candidates)
        return _zero_adjuster(candidates)

    result = jtalk.run_mecab_with_cost_adjustments("国立に行く", capture_zero)

    assert len(captured_candidates) > 0
    assert all(
        candidate["complete_path_cost"]
        == candidate["forward_path_cost"] + candidate["backward_path_cost"]
        for candidate in captured_candidates
    )
    assert (
        min(candidate["complete_path_cost"] for candidate in captured_candidates)
        == result["base_path_cost"]
    )


@pytest.mark.parametrize(("delta", "expected_unit_cost"), [(0.0005, 1), (-0.0005, -1)])
def test_run_mecab_with_cost_adjustments_rounds_half_away_from_zero(
    delta: float,
    expected_unit_cost: int,
) -> None:
    """±0.5 コスト単位の補正を C llround() と同じ方向へ丸める。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    baseline = jtalk.run_mecab_with_cost_adjustments("こんにちは世界", _zero_adjuster)

    def adjust_every_candidate(candidates: list[MeCabCostCandidate]) -> list[float]:
        """無視対象を除く全候補に丸め境界の補正を加える。"""

        return [delta if candidate["is_ignored"] is False else 0.0 for candidate in candidates]

    adjusted = jtalk.run_mecab_with_cost_adjustments("こんにちは世界", adjust_every_candidate)
    adjusted_node_count = sum(morph["is_ignored"] is False for morph in adjusted["morphs"])

    assert adjusted["node_indices"] == baseline["node_indices"]
    assert adjusted["base_path_cost"] == baseline["base_path_cost"]
    assert adjusted["path_cost"] == baseline["path_cost"] + (
        expected_unit_cost * adjusted_node_count
    )


def test_run_mecab_with_cost_adjustments_preserves_existing_path_on_exact_tie():
    """別経路を既定経路と同額にしても元の one-best を維持する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    captured_candidates: list[MeCabCostCandidate] = []

    def capture_zero(candidates: list[MeCabCostCandidate]) -> list[float]:
        """候補列を記録し、既定経路を取得する。"""

        captured_candidates.extend(candidates)
        return _zero_adjuster(candidates)

    baseline = jtalk.run_mecab_with_cost_adjustments("その方が良い", capture_zero)
    alternative = next(
        candidate for candidate in captured_candidates if candidate["surface"] == "その方"
    )
    tie_delta = (baseline["base_path_cost"] - alternative["complete_path_cost"]) / 1000.0

    def tie_alternative(candidates: list[MeCabCostCandidate]) -> list[float]:
        """「その方」を1ノードで通る別経路を既定経路と同額にする。"""

        return [
            tie_delta if candidate["node_index"] == alternative["node_index"] else 0.0
            for candidate in candidates
        ]

    tied = jtalk.run_mecab_with_cost_adjustments("その方が良い", tie_alternative)

    assert tied["path_cost"] == baseline["path_cost"]
    assert tied["node_indices"] == baseline["node_indices"]


def test_run_mecab_with_cost_adjustments_keeps_consecutive_symbol_chunks_unsplit():
    """run_mecab_detailed() の既知記号復元を cost 補正経路では行わないことを確認"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    text = "マジ！？！？！？！？！？！？！？！？"

    detailed_morphs = jtalk.run_mecab_detailed(text)
    adjusted = jtalk.run_mecab_with_cost_adjustments(text, _zero_adjuster)

    # run_mecab_detailed() は未知語へ連結された記号列を1文字ずつ復元する
    assert [morph["surface"] for morph in detailed_morphs] == ["マジ", *list("！？" * 8)]
    # run_mecab_with_cost_adjustments() は lattice の one-best 表層をそのまま返す
    assert [morph["surface"] for morph in adjusted["morphs"]] == [
        "マジ",
        "！？！？！？！？！？！？！？！？",
    ]
    assert adjusted["morphs"] != detailed_morphs


def test_run_mecab_with_cost_adjustments_can_flip_specific_candidate_path():
    """特定候補の Δc だけで one-best が反転し、features と morphs が同じ path を指す"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    text = "その方が良い"

    def prefer_kata(candidates: list[MeCabCostCandidate]) -> list[float]:
        """「方」をカタと読む候補だけを優先する。"""

        deltas = [0.0 for _ in candidates]
        for index, candidate in enumerate(candidates):
            features = candidate["features"]
            assert isinstance(features, list)
            # 「方/カタ」の候補だけを十分に安くし、通常の「方/ホウ」から one-best を反転する
            if candidate["surface"] == "方" and features[8] == "カタ":
                deltas[index] = -10.0
        return deltas

    def penalize_kata(candidates: list[MeCabCostCandidate]) -> list[float]:
        """「方」をカタと読む候補だけを不利にする。"""

        deltas = [0.0 for _ in candidates]
        for index, candidate in enumerate(candidates):
            features = candidate["features"]
            assert isinstance(features, list)
            # 「方/カタ」をさらに高くし、通常の one-best が維持されることを確認する
            if candidate["surface"] == "方" and features[8] == "カタ":
                deltas[index] = 10.0
        return deltas

    flipped = jtalk.run_mecab_with_cost_adjustments(text, prefer_kata)
    retained = jtalk.run_mecab_with_cost_adjustments(text, penalize_kata)

    flipped_morph = next(morph for morph in flipped["morphs"] if morph["surface"] == "方")
    retained_morph = next(morph for morph in retained["morphs"] if morph["surface"] == "方")

    assert _reading(flipped_morph) == "カタ"
    assert _reading(retained_morph) == "ホウ"
    assert _feature_from_morphs(flipped["morphs"]) == flipped["features"]
    assert _feature_from_morphs(retained["morphs"]) == retained["features"]


def test_run_mecab_with_cost_adjustments_uses_current_lattice_only():
    """直前に別文を解析しても、新 API の候補や結果に前文の feature が混入しない"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    seen_surfaces: list[str] = []

    # 直前の lattice にだけ現れる語を作ってから、別文で新 API を実行する
    assert any(morph["surface"] == "大分" for morph in jtalk.run_mecab_detailed("大分県に行く"))

    def capture_candidates(candidates: list[MeCabCostCandidate]) -> list[float]:
        """現在の lattice 表層を記録し、コストを変更せず返す。"""

        seen_surfaces.extend(str(candidate["surface"]) for candidate in candidates)
        return [0.0 for _ in candidates]

    result = jtalk.run_mecab_with_cost_adjustments("こんにちは世界", capture_candidates)

    assert result["features"] == jtalk.run_mecab("こんにちは世界")
    assert "大分" not in seen_surfaces
    assert "こんにちは" in seen_surfaces


def test_run_mecab_with_cost_adjustments_clips_wcost():
    """巨大な Δc でも wcost を short の範囲へ収める"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    def huge_positive(candidates: list[MeCabCostCandidate]) -> list[float]:
        """通常ノードへ short の範囲を超えるコスト差を返す。"""

        return [100000.0 if candidate["is_ignored"] is False else 0.0 for candidate in candidates]

    result = jtalk.run_mecab_with_cost_adjustments("こんにちは世界", huge_positive)

    assert isinstance(result["features"], list)
    assert isinstance(result["morphs"], list)
    assert all(
        morph["word_cost"] == 32767 for morph in result["morphs"] if morph["is_ignored"] is False
    )


@pytest.mark.parametrize("non_finite_delta", [float("nan"), float("inf"), float("-inf")])
def test_run_mecab_with_cost_adjustments_rejects_non_finite_delta(
    non_finite_delta: float,
) -> None:
    """NaN と無限大の Δc は C の整数変換へ渡す前に拒否する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    def return_non_finite(candidates: list[MeCabCostCandidate]) -> list[float]:
        """全候補へ検証対象の非有限値を返す。"""

        return [non_finite_delta for _ in candidates]

    with pytest.raises(ValueError, match="cost_adjuster deltas must be finite"):
        jtalk.run_mecab_with_cost_adjustments("こんにちは世界", return_non_finite)


@pytest.mark.parametrize("excessive_delta", [1e308, -1e308])
def test_run_mecab_with_cost_adjustments_rejects_delta_exceeding_c_long(
    excessive_delta: float,
) -> None:
    """1000倍後に C long を超える Δc は整数変換前に拒否する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    def return_excessive(candidates: list[MeCabCostCandidate]) -> list[float]:
        """全候補へ C long の安全範囲を超える値を返す。"""

        return [excessive_delta for _ in candidates]

    with pytest.raises(ValueError, match="cost_adjuster deltas are too large"):
        jtalk.run_mecab_with_cost_adjustments("こんにちは世界", return_excessive)


def test_run_mecab_with_cost_adjustments_clips_before_c_long_addition_overflows() -> None:
    """C long 上限直下の Δc も符号反転させず short 上限へクリップする"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    long_max = (1 << (sizeof(c_long) * 8 - 1)) - 1
    near_long_max_delta = math.nextafter(long_max / 1000.0, 0.0)

    def return_near_long_max(candidates: list[MeCabCostCandidate]) -> list[float]:
        """通常ノードへ C long 上限直下のコスト差を返す。"""

        return [
            near_long_max_delta if candidate["is_ignored"] is False else 0.0
            for candidate in candidates
        ]

    result = jtalk.run_mecab_with_cost_adjustments(
        "こんにちは世界",
        return_near_long_max,
    )

    assert all(
        morph["word_cost"] == 32767 for morph in result["morphs"] if morph["is_ignored"] is False
    )
