"""tsqyomi v2: MeCab 候補グラフ上で読みを選び、NJD 入力用 feature 列を返す。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from ..openjtalk import OpenJTalk
from ..types import CandidateNode, CandidatePath, MeCabMorph, ReadingAnalysis
from .model import ReadingTarget, get_loaded_model


_CLOSING_DELIMITER_BY_OPENING = {
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "［": "］",
    "[": "]",
    "【": "】",
    "〈": "〉",
    "《": "》",
    "“": "”",
    "‘": "’",
}


@dataclass(frozen=True)
class _ResolvedTarget:
    """
    候補グラフ上で確定した1対象の位置情報と、モデルが選んだ読み。

    Attributes:
        char_span (tuple[int, int]): 正規化本文上の対象表層の半開区間
        morph_range (tuple[int, int]): 差し替え対象の形態素列における半開添字区間
        surface (str): 対象表層
        span_paths (tuple[CandidatePath, ...]): 差し替え可能な候補経路
        pronunciations (tuple[str, ...]): 候補グラフ上で到達可能な発音 (重複なし)
        selected_pronunciation (str | None): モデルが選んだ発音。推論前は None
    """

    char_span: tuple[int, int]
    morph_range: tuple[int, int]
    surface: str
    span_paths: tuple[CandidatePath, ...]
    pronunciations: tuple[str, ...]
    selected_pronunciation: str | None = None

    def to_reading_target(self) -> ReadingTarget:
        """
        ONNX 推論へ渡す公開型へ投影する。

        Returns:
            ReadingTarget: `predict()` へ渡す1対象
        """

        return ReadingTarget(
            char_span=self.char_span,
            surface=self.surface,
            pronunciations=self.pronunciations,
        )

    @property
    def selected_paths(self) -> tuple[CandidatePath, ...]:
        """
        選択済み発音を実現する候補経路を返す。

        Returns:
            tuple[CandidatePath, ...]: `selected_pronunciation` に一致する候補経路
        """

        if self.selected_pronunciation is None:
            return ()
        return tuple(
            path for path in self.span_paths if path.pronunciation == self.selected_pronunciation
        )


def select_mecab_features_with_tsqyomi(
    text: str,
    jtalk: OpenJTalk,
    *,
    include_morphs: bool = True,
) -> tuple[list[str], list[MeCabMorph]]:
    """
    ロード済みの tsqyomi モデルで選んだ読みの MeCab feature 列を返す。
    MeCab の解析ロックは候補グラフのコピー時点で解放され、モデル推論中は保持されない。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        jtalk (OpenJTalk): 候補解析に使う OpenJTalk インスタンス
        include_morphs (bool): 呼び出し元が詳細形態素列を必要とする場合は True

    Returns:
        tuple[list[str], list[MeCabMorph]]: NJD 入力用 MeCab feature 列と差し替え後の形態素列
    """

    model = get_loaded_model()
    normalized_text = jtalk.normalize_for_mecab(text)
    target_spans = _find_target_spans(
        normalized_text,
        model.metadata.surfaces_by_first_character,
    )

    # 対象表層が本文にない多数の呼び出しでは候補グラフもモデル推論も省く
    if len(target_spans) == 0:
        mecab_features = jtalk.run_mecab(normalized_text)
        if include_morphs is False:
            return mecab_features, []
        return mecab_features, jtalk.run_mecab_detailed(normalized_text)

    processing_segments = _split_target_processing_segments(normalized_text, target_spans)
    if len(processing_segments) > 1:
        combined_features: list[str] = []
        combined_morphs: list[MeCabMorph] = []
        for segment_start, segment_end in processing_segments:
            segment_features, segment_morphs = select_mecab_features_with_tsqyomi(
                normalized_text[segment_start:segment_end],
                jtalk,
                include_morphs=include_morphs,
            )
            combined_features.extend(segment_features)
            for morph in segment_morphs:
                # 分割入力の char_span は区間先頭からの相対位置なので、全文位置へ加算する
                shifted_morph = morph.copy()
                shifted_morph["char_span"] = (
                    morph["char_span"][0] + segment_start,
                    morph["char_span"][1] + segment_start,
                )
                combined_morphs.append(shifted_morph)
        return combined_features, combined_morphs

    analysis = jtalk.analyze_mecab_candidates(normalized_text, target_spans)
    nodes_by_id = {node.node_id: node for node in analysis.nodes}
    selected_features = list(analysis.features)
    selected_morphs = [morph.copy() for morph in analysis.morphs]
    resolved_targets: list[_ResolvedTarget] = []

    # メタデータ上の最長一致と既定形態素境界の両方を満たす出現だけをモデルへ渡す
    for char_span in target_spans:
        surface = analysis.normalized_text[char_span[0] : char_span[1]]
        allowed_readings = frozenset(
            model.metadata.reading_class_ids_by_surface_and_pronunciation.get(surface, {})
        )
        if len(allowed_readings) < 2:
            continue
        morph_range = _find_exact_morph_range(analysis.morphs, char_span)
        if morph_range is None:
            continue
        span_paths = _eligible_span_paths(
            analysis,
            char_span,
            surface,
            allowed_readings,
            nodes_by_id,
        )
        pronunciations = tuple(dict.fromkeys(path.pronunciation for path in span_paths))
        # 候補グラフ上で読み候補が2件未満なら、辞書の最良経路をそのまま維持する
        if len(pronunciations) < 2:
            continue
        resolved_targets.append(
            _ResolvedTarget(
                char_span=char_span,
                morph_range=morph_range,
                surface=surface,
                span_paths=tuple(span_paths),
                pronunciations=pronunciations,
            )
        )

    if len(resolved_targets) > 0:
        predictions = model.predict(
            analysis.normalized_text,
            tuple(item.to_reading_target() for item in resolved_targets),
        )
        resolved_targets = [
            replace(item, selected_pronunciation=prediction.pronunciation)
            for item, prediction in zip(resolved_targets, predictions, strict=True)
        ]

        # 隣接対象は接続費用をまとめて比較し、形態素を挟む対象だけを独立に選ぶ
        selected_paths: list[tuple[_ResolvedTarget, CandidatePath]] = []
        for target_group in _group_adjacent_targets(resolved_targets):
            selected_paths.extend(_select_joint_paths(analysis, target_group))

        # 後方から差し替えれば、複数形態素を1形態素へ畳んでも前方の添字が変わらない
        for target, path in reversed(selected_paths):
            start, end = target.morph_range
            node = nodes_by_id[path.node_ids[0]]
            replacement_morph = _replace_morph(selected_morphs[start], node)
            selected_morphs[start:end] = [replacement_morph]
            feature_start = sum(morph["is_ignored"] is False for morph in analysis.morphs[:start])
            feature_end = feature_start + sum(
                morph["is_ignored"] is False for morph in analysis.morphs[start:end]
            )
            selected_features[feature_start:feature_end] = list(path.features)

    return selected_features, selected_morphs if include_morphs is True else []


def _find_target_spans(
    text: str,
    surfaces_by_first_character: dict[str, tuple[str, ...]],
) -> tuple[tuple[int, int], ...]:
    """
    推論対象表層を最長一致させ、内側の短い対象を除いた範囲を返す。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        surfaces_by_first_character (dict[str, tuple[str, ...]]): 先頭文字別・長さ降順の表層索引

    Returns:
        tuple[tuple[int, int], ...]: 重なりのない対象表層の半開区間 (文字位置昇順)
    """

    occurrences: list[tuple[int, int]] = []
    for start, character in enumerate(text):
        # 同じ開始位置では長い表層から確認し、最長の1件だけを対象候補にする
        for surface in surfaces_by_first_character.get(character, ()):
            if text.startswith(surface, start):
                occurrences.append((start, start + len(surface)))
                break
    selected: list[tuple[int, int]] = []
    for occurrence in occurrences:
        if any(occurrence[0] < end and start < occurrence[1] for start, end in selected):
            continue
        selected.append(occurrence)
    return tuple(sorted(selected))


def _split_target_processing_segments(
    text: str,
    target_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """
    対象を含む文だけを独立させ、連続する対象なし文をまとめた範囲を返す。
    引用符・括弧の入れ子を考慮して句点分割し、長文の前置きで MeCab 呼び出しが増えないようにする。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        target_spans (tuple[tuple[int, int], ...]): 推論対象表層の半開区間

    Returns:
        tuple[tuple[int, int], ...]: `text` 上の処理区間 (半開区間) の列。
            要素が1件のときは全文を1区間として返す
    """

    matched_openings, matched_closings = _matched_delimiter_indices(text)
    sentence_ranges: list[tuple[int, int]] = []
    sentence_start = 0
    delimiter_depth = 0
    for index, character in enumerate(text):
        # 対応が取れた括弧だけ深さへ加え、不均衡な開き括弧で残り全文を抱え込まない
        if index in matched_openings:
            delimiter_depth += 1
        if character in "。！？!?\n" and delimiter_depth == 0:
            sentence_ranges.append((sentence_start, index + 1))
            sentence_start = index + 1
        if index in matched_closings:
            delimiter_depth -= 1
    if sentence_start < len(text):
        sentence_ranges.append((sentence_start, len(text)))
    if len(sentence_ranges) <= 1:
        return ((0, len(text)),)

    segments: list[tuple[int, int, bool]] = []
    for range_start, range_end in sentence_ranges:
        has_target = any(
            range_start <= target_start and target_end <= range_end
            for target_start, target_end in target_spans
        )
        # 対象なし文だけは連結し、長い前置きで MeCab 呼び出しが文数分に増えないようにする
        if segments and has_target is False and segments[-1][2] is False:
            segments[-1] = (segments[-1][0], range_end, False)
        else:
            segments.append((range_start, range_end, has_target))
    return tuple((start, end) for start, end, _ in segments)


def _matched_delimiter_indices(text: str) -> tuple[frozenset[int], frozenset[int]]:
    """
    入れ子順まで対応した括弧と引用符の文字位置を返す。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト

    Returns:
        tuple[frozenset[int], frozenset[int]]: 対応が取れた開き括弧位置と閉じ括弧位置
    """

    expected_closings: list[tuple[str, int]] = []
    matched_openings: set[int] = set()
    matched_closings: set[int] = set()
    for index, character in enumerate(text):
        expected_closing = _CLOSING_DELIMITER_BY_OPENING.get(character)
        if expected_closing is not None:
            expected_closings.append((expected_closing, index))
            continue
        # 入れ子の途中に異なる閉じ括弧があっても、外側との誤対応は作らない
        if len(expected_closings) == 0 or expected_closings[-1][0] != character:
            continue
        _, opening_index = expected_closings.pop()
        matched_openings.add(opening_index)
        matched_closings.add(index)
    return frozenset(matched_openings), frozenset(matched_closings)


def _find_exact_morph_range(
    morphs: tuple[MeCabMorph, ...], char_span: tuple[int, int]
) -> tuple[int, int] | None:
    """
    対象範囲を隙間なく覆う既定形態素列の半開区間を返す。

    Args:
        morphs (tuple[MeCabMorph, ...]): MeCab の既定最良経路の形態素列
        char_span (tuple[int, int]): 対象表層の半開区間

    Returns:
        tuple[int, int] | None: 形態素添字の半開区間。境界が一意に定まらない場合は None
    """

    start_indices = [
        index for index, morph in enumerate(morphs) if morph["char_span"][0] == char_span[0]
    ]
    if len(start_indices) != 1:
        return None
    start = start_indices[0]
    cursor = char_span[0]
    for end in range(start, len(morphs)):
        morph_span = morphs[end]["char_span"]
        if morph_span[0] != cursor or morph_span[1] > char_span[1]:
            return None
        cursor = morph_span[1]
        if cursor == char_span[1]:
            return start, end + 1
    return None


def _eligible_span_paths(
    analysis: ReadingAnalysis,
    char_span: tuple[int, int],
    surface: str,
    allowed_readings: frozenset[str],
    nodes_by_id: dict[int, CandidateNode],
) -> list[CandidatePath]:
    """
    最良経路の形態素範囲で差し替え可能な辞書候補を返す。

    Args:
        analysis (ReadingAnalysis): `analyze_mecab_candidates()` の結果
        char_span (tuple[int, int]): 対象表層の半開区間
        surface (str): 対象表層
        allowed_readings (frozenset[str]): メタデータが許可する発音
        nodes_by_id (dict[int, CandidateNode]): 候補ノード ID からノードへの索引

    Returns:
        list[CandidatePath]: 保護候補を含まず、許可読みを実現する候補経路
    """

    paths = [
        path for path in analysis.paths if path.char_span == char_span and path.surface == surface
    ]
    # ユーザー辞書の保護候補やメタデータ外の読みが混在する範囲では tsqyomi による差し替えを止める
    if any(nodes_by_id[path.node_ids[0]].is_reading_protected is True for path in paths):
        return []
    return [
        path
        for path in paths
        if path.pronunciation in allowed_readings
        and nodes_by_id[path.node_ids[0]].is_ignored is False
    ]


def _group_adjacent_targets(
    targets: list[_ResolvedTarget],
) -> list[tuple[_ResolvedTarget, ...]]:
    """
    形態素を挟まない対象を同じ候補経路選択へまとめる。

    Args:
        targets (list[_ResolvedTarget]): モデルが選んだ読みと形態素範囲

    Returns:
        list[tuple[_ResolvedTarget, ...]]: 隣接グループごとの対象列
    """

    groups: list[list[_ResolvedTarget]] = []
    for target in sorted(targets, key=lambda item: item.morph_range):
        if groups and groups[-1][-1].morph_range[1] == target.morph_range[0]:
            groups[-1].append(target)
        else:
            groups.append([target])
    return [tuple(group) for group in groups]


def _select_joint_paths(
    analysis: ReadingAnalysis,
    targets: tuple[_ResolvedTarget, ...],
) -> list[tuple[_ResolvedTarget, CandidatePath]]:
    """
    選択済み読みを実現する候補経路の組を、固定した外側経路の費用で決める。

    Args:
        analysis (ReadingAnalysis): 接続辺を含む候補解析結果
        targets (tuple[_ResolvedTarget, ...]): 同一グループ内の隣接対象

    Returns:
        list[tuple[_ResolvedTarget, CandidatePath]]: 各対象と選ばれた候補経路の対
    """

    connection_costs = {
        (connection.left_node_id, connection.right_node_id): connection.cost
        for connection in analysis.connections
    }
    best_paths: tuple[CandidatePath, ...] | None = None
    best_cost: int | None = None
    for candidate_paths in product(*(target.selected_paths for target in targets)):
        if len(candidate_paths) == 1:
            cost = candidate_paths[0].boundary_cost
        else:
            adjacent_costs = [
                connection_costs.get((left.node_ids[-1], right.node_ids[0]))
                for left, right in zip(candidate_paths, candidate_paths[1:])
            ]
            if any(cost is None for cost in adjacent_costs):
                continue
            # any() 通過後も Pyright は None 除去を推論しないため、型上は明示する
            cost = (
                candidate_paths[0].left_boundary_cost
                + sum(cost for cost in adjacent_costs if cost is not None)
                + candidate_paths[-1].right_boundary_cost
            )
        if best_cost is None or (cost, tuple(path.path_id for path in candidate_paths)) < (
            best_cost,
            tuple(path.path_id for path in best_paths or ()),
        ):
            best_cost = cost
            best_paths = candidate_paths
    if best_paths is None:
        return []
    return list(zip(targets, best_paths))


def _replace_morph(base_morph: MeCabMorph, node: CandidateNode) -> MeCabMorph:
    """
    選択した辞書ノードの feature を NJD 入力と詳細形態素へ反映する。

    Args:
        base_morph (MeCabMorph): 差し替え前の形態素 (未使用フィールドは引き継ぐ)
        node (CandidateNode): 採用する辞書候補ノード

    Returns:
        MeCabMorph: 候補ノードの surface・feature・コスト情報を反映した形態素
    """

    return MeCabMorph(
        surface=node.surface,
        features=node.feature.split(","),
        pos_id=node.pos_id,
        left_id=node.left_id,
        right_id=node.right_id,
        word_cost=node.word_cost,
        link_cost=base_morph["link_cost"],
        node_cost=base_morph["node_cost"],
        char_span=node.char_span,
        is_unknown=node.is_unknown,
        is_ignored=node.is_ignored,
        dictionary_index=node.dictionary_index,
    )
