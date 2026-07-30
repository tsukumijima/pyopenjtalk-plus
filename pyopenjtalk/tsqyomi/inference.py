"""ロード済み tsqyomi モデルのコスト補正を MeCab lattice へ接続する。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..types import MeCabCostAdjustedPath, MeCabCostCandidate
from .model import get_loaded_model


if TYPE_CHECKING:
    # 実行時の循環インポートを避けつつ、公開されている低レベル API の型を参照する
    from ..openjtalk import OpenJTalk


def run_mecab_with_tsqyomi(text: str, jtalk: OpenJTalk) -> MeCabCostAdjustedPath:
    """
    ロード済みの tsqyomi モデルで読み候補を選んだ MeCab path を返す。

    高レベル API から同じ lattice 介入処理を呼び出すための内部関数である。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        jtalk (OpenJTalk): コスト補正を適用する OpenJTalk インスタンス

    Returns:
        MeCabCostAdjustedPath: tsqyomi のコスト補正後に採用した形態素 path
    """

    return jtalk.run_mecab_with_cost_adjustments(text, make_cost_adjuster())


def make_cost_adjuster() -> Callable[[list[MeCabCostCandidate]], list[float]]:
    """
    ロード済みモデルを保持し、MeCab 候補へ加算するコスト差を返す関数を作る。

    Returns:
        Callable[[list[MeCabCostCandidate]], list[float]]: 候補順のコスト差を返す関数
    """

    model = get_loaded_model()

    def cost_adjuster(candidates: list[MeCabCostCandidate]) -> list[float]:
        """
        1回の lattice 候補列をモデル採点し、候補順のコスト差を返す。

        Args:
            candidates (list[MeCabCostCandidate]): Cython から渡された lattice 候補列

        Returns:
            list[float]: 各候補の `wcost` へ加算するコスト差
        """

        deltas = [0.0] * len(candidates)
        # 対象表層がなければ本文復元とトークン化を含む推論処理をすべて省く
        if not any(
            candidate["surface"] in model.metadata.model_scored_surfaces for candidate in candidates
        ):
            return deltas

        # lattice の文字位置と表層から、モデルへ渡す正規化済み本文を復元する
        sentence_length = max((candidate["char_span"][1] for candidate in candidates), default=0)
        sentence_chars = [""] * sentence_length
        for candidate in candidates:
            char_start, char_end = candidate["char_span"]
            surface = candidate["surface"]
            # Cython 境界の座標が壊れている場合は、誤った対象位置でモデル介入を行わない
            if (
                char_start < 0
                or char_end < char_start
                or char_end > sentence_length
                or char_end - char_start != len(surface)
            ):
                return deltas
            for offset, character in enumerate(surface):
                character_index = char_start + offset
                existing_character = sentence_chars[character_index]
                # 重なったノードの表層が矛盾する lattice は、本文を一意に復元できない
                if existing_character != "" and existing_character != character:
                    return deltas
                sentence_chars[character_index] = character
        # 空白位置を詰めると後続候補の char_span がずれるため、不完全な復元は採点しない
        if any(character == "" for character in sentence_chars):
            return deltas
        lattice_text = "".join(sentence_chars)

        # 同じ文字範囲と表層を持つ候補を1つの読み選択問題として束ねる
        candidate_indices_by_group: dict[tuple[int, int, str], list[int]] = {}
        protected_groups: set[tuple[int, int, str]] = set()
        for candidate_index, candidate in enumerate(candidates):
            if candidate["surface"] not in model.metadata.model_scored_surfaces:
                continue
            if candidate["is_ignored"] is True or candidate["surface"] == "":
                continue
            char_start, char_end = candidate["char_span"]
            group_key = (char_start, char_end, candidate["surface"])
            # 読みを持たない候補も同じ lattice 群の一員なので、辞書保護だけは発音抽出より先に記録する
            if candidate["is_reading_protected"] is True:
                protected_groups.add(group_key)
            features = candidate["features"]
            pronunciation = features[9] if len(features) > 9 else ""
            if pronunciation == "":
                continue
            candidate_indices_by_group.setdefault(group_key, []).append(candidate_index)

        # 保護対象と単一読みの候補群を除き、モデルの相対コストを元の候補位置へ戻す
        for group_key, candidate_indices in candidate_indices_by_group.items():
            char_start, char_end, _surface = group_key
            # 保護辞書が同じ候補群に1件でも含まれる場合は、群全体を辞書側のコストへ委ねる
            if group_key in protected_groups:
                continue
            pronunciations = [candidates[index]["features"][9] for index in candidate_indices]
            unique_pronunciations = tuple(dict.fromkeys(pronunciations))
            if len(unique_pronunciations) < 2:
                continue
            scores = model.score_candidates(
                lattice_text,
                (char_start, char_end),
                unique_pronunciations,
            )
            relative_cost_by_pronunciation = {
                score["pronunciation"]: score["relative_cost"] for score in scores
            }
            for candidate_index, pronunciation in zip(
                candidate_indices,
                pronunciations,
            ):
                deltas[candidate_index] = relative_cost_by_pronunciation[pronunciation]
        return deltas

    return cost_adjuster
