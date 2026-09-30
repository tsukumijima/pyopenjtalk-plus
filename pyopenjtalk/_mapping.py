"""
形態素-音素マッピングと、NJD・MeCab 形態素のアライメントの実装。

NJD の数詞変換・踊り字展開・長音吸収によって MeCab 形態素と NJD ノードの対応がずれるため、
表層・入力文字位置・音素列の対応付けはこのモジュールへ集約している。
Haqumei (Rust 実装) の open_jtalk/mapping.rs に対応する。
"""

from collections.abc import Callable

from .openjtalk import OpenJTalk
from .types import (
    JPCommonMappingEntry,
    MeCabMorph,
    NJDFeature,
    NormalizeMode,
    SurfacePhonemeMapping,
)
from .utils import normalize_itaiji, normalize_text


# 括弧と引用符は読み上げで間を置かないため、既定では短ポーズを割り当てずに音素なしで保持する
_DEFAULT_NON_PAUSE_SYMBOLS = frozenset(
    (
        "「",
        "」",
        "『",
        "』",
        "（",
        "）",
        "(",
        ")",
        "【",
        "】",
        "［",
        "］",
        "[",
        "]",
        "〈",
        "〉",
        "《",
        "》",
        "〔",
        "〕",
        "｛",
        "｝",
        "{",
        "}",
        '"',
        "'",
        "”",
        "“",
        "’",
        "‘",
    )
)
# 踊り字展開 (process_odori_features()) で morph/NJD のずれを検出するための文字集合
_ODORI_CHARS = frozenset("々ゝゞヽヾ")
# 数字正規化後の NJD ノードと MeCab morph を局所的に対応させるための文字集合
_DIGIT_MORPH_SURFACES = frozenset("０１２３４５６７８９0123456789")
# Unicode NFKC で1符号位置から展開される最大文字数を上限にし、入力不一致時の二乗探索を避ける
_MAX_CALLER_TEXT_CHUNK_LENGTH = 18
# 数詞ブロックの編集距離表が入力長の二乗で増えないよう、通常の数値表記を十分に上回る上限を設ける
_MAX_NUMBER_ALIGNMENT_BLOCK_LENGTH = 128
# njd_set_digit_rule_numeral_list1 と同じ異表記を、アライメント比較用の漢数字へ変換する
_NJD_NUMBER_MORPH_SURFACE_KEYS = {
    "○": "〇",
    "〇": "〇",
    "０": "〇",
    "0": "〇",
    "１": "一",
    "1": "一",
    "一": "一",
    "いち": "一",
    "壱": "一",
    "２": "二",
    "2": "二",
    "二": "二",
    "に": "二",
    "弐": "二",
    "貳": "二",
    "ニ": "二",
    "３": "三",
    "3": "三",
    "三": "三",
    "さん": "三",
    "参": "三",
    "４": "四",
    "4": "四",
    "四": "四",
    "よん": "四",
    "し": "四",
    "５": "五",
    "5": "五",
    "五": "五",
    "ご": "五",
    "６": "六",
    "6": "六",
    "六": "六",
    "ろく": "六",
    "７": "七",
    "7": "七",
    "七": "七",
    "なな": "七",
    "しち": "七",
    "８": "八",
    "8": "八",
    "八": "八",
    "はち": "八",
    "９": "九",
    "9": "九",
    "九": "九",
    "きゅう": "九",
    "く": "九",
}
_KANJI_NUMBER_SURFACES = frozenset("一二三四五六七八九十百千万億兆〇零")
_NJD_NUMBER_ALIGNMENT_TRANSLATION = str.maketrans(
    "0123456789０１２３４５６７８９零",
    "〇一二三四五六七八九〇一二三四五六七八九〇",
)


def default_is_non_pause_symbol(surface: str) -> bool:
    """
    記号へ短ポーズを割り当てず、音素なしで保持するかを判定する。

    Args:
        surface (str): 判定対象の形態素表層

    Returns:
        bool: 括弧・引用符として短ポーズを割り当てない場合は True
    """

    return surface in _DEFAULT_NON_PAUSE_SYMBOLS


def make_phoneme_mapping(
    njd_features: list[NJDFeature],
    morphs: list[MeCabMorph] | None,
    inference_jtalk: OpenJTalk,
    *,
    caller_text: str | None = None,
    normalize_mode: NormalizeMode = "None",
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
) -> list[SurfacePhonemeMapping]:
    """
    NJD features から各形態素に対応する音素列のマッピングを返す。
    グローバルインスタンスの借り出しを含む公開入口は pyopenjtalk.make_phoneme_mapping() で、
    この実装は呼び出し側が借り出したインスタンスをそのまま使う。
    Cython 側の OpenJTalk.make_phoneme_mapping() で基本マッピングを取得し、
    morphs が渡された場合は MeCab morphs とアライメントして is_unknown / is_ignored を付与する。

    morphs を省略した場合は is_unknown=False 、is_ignored は音素列の空判定から推定される。
    このとき `char_span` は NJD 後処理後の surface を連結した座標系であり、
    `g2p_mapping(text=...)` が返す呼び出し元入力文上の半開区間とは一致しない。
    morphs を渡す場合、踊り字展開や数字正規化により NJD と MeCab の粒度がずれることがあるが、
    アライメントロジックが自動的に補正する。音素列自体は常に正しい値が得られる。
    pause-like な記号は surface として保持されるが、
    JPCommon が実際に短ポーズを生成しない場合は phonemes は空のまま返る。
    morphs 付きで対応する morph を特定できないエントリの `char_span` は、未特定を表す `(0, 0)` になる。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        morphs (list[MeCabMorph] | None): MeCab の形態素解析結果 (pyopenjtalk.run_frontend_detailed() の戻り値)
            None の場合は is_unknown / is_ignored の推定精度が下がる
        inference_jtalk (OpenJTalk): 借り出し済みの OpenJTalk インスタンス
        caller_text (str | None): `char_span` の座標系に使う正規化前の入力文
            None の場合は MeCab 正規化本文上の座標を使う
        normalize_mode (NormalizeMode): caller_text に適用した Unicode 正規化方式 (デフォルト: `"None"`)
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング

    Raises:
        ValueError: caller_text と MeCab 正規化本文の対応付けに失敗した場合。
            `g2p_mapping()` 経由でも `_build_caller_text_spans_by_mecab_character()` からそのまま伝播する
    """

    # Cython レベルで基本マッピングと長音吸収マージを取得し、呼び出し元座標への変換まで同じインスタンスで行う
    base_mapping = inference_jtalk.make_phoneme_mapping(njd_features)

    # Cython 側の既定マッピングを呼び出し側の記号判定で上書きし、通常音素の対応付けは維持する
    for entry in base_mapping:
        if entry["pron"] not in ("、", "？", "！"):
            continue
        if is_non_pause_symbol(entry["surface"]) is True:
            entry["phonemes"] = []
        elif len(entry["phonemes"]) == 0:
            entry["phonemes"] = ["pau"]

    mecab_text = "" if morphs is None else "".join(morph["surface"] for morph in morphs)
    reference_text = caller_text if caller_text is not None else mecab_text
    if mecab_text == reference_text:
        caller_text_spans = [(index, index + 1) for index in range(len(mecab_text))]
    else:
        caller_text_spans = _build_caller_text_spans_by_mecab_character(
            inference_jtalk,
            mecab_text,
            reference_text,
            normalize_mode,
        )

    # morphs が渡されていない場合: NJDFeature ベースで is_unknown を推定
    # njd_set_pronunciation が mora_size=0 のノードの読みを補完し、pos を "フィラー" に上書きする
    # この判定は辞書に元からフィラーとして登録された既知語（ゔぁ等）でも True になるため、
    # MeCab の is_unknown より範囲が広い（偽陽性がある）ため、正確な判定には morphs が必要
    if morphs is None:
        sequential_start = 0
        entries_without_morphs: list[SurfacePhonemeMapping] = []
        for entry in base_mapping:
            char_span = (sequential_start, sequential_start + len(entry["surface"]))
            sequential_start += len(entry["surface"])
            entries_without_morphs.append(
                _base_to_detail(
                    entry,
                    entry["phonemes"],
                    char_span=char_span,
                    is_unknown=(entry["pos"] == "フィラー" and entry["chain_rule"] == "*"),
                    is_ignored=len(entry["phonemes"]) == 0,
                )
            )
        return entries_without_morphs

    # base_mapping と morphs のアライメント: is_unknown / is_ignored を付与する

    result: list[SurfacePhonemeMapping] = []
    morph_ranges: list[tuple[int, int]] = []
    mecab_char_span_overrides: list[tuple[int, int] | None] = []

    def _append_aligned(
        entry: SurfacePhonemeMapping,
        morph_range: tuple[int, int],
        *,
        mecab_char_span: tuple[int, int] | None = None,
    ) -> None:
        """
        アライメント結果を morph_range 付きで result へ積む。

        Args:
            entry (SurfacePhonemeMapping): char_span は後段で付与する mapping
            morph_range (tuple[int, int]): 対応する morph 添字半開区間
            mecab_char_span (tuple[int, int] | None): morph 内部の部分範囲を直接指定する場合の MeCab 座標
        """

        result.append(entry)
        morph_ranges.append(morph_range)
        mecab_char_span_overrides.append(mecab_char_span)

    # 全 morphs が ignored の場合は全て sp として返す
    has_valid_morph = any(morph["is_ignored"] is False for morph in morphs)
    if has_valid_morph is False:
        ignored_entries = [
            _sp_entry(morph["surface"], char_span=(0, 0), is_unknown=morph["is_unknown"])
            for morph in morphs
        ]
        ignored_ranges = [(index, index + 1) for index in range(len(morphs))]
        ignored_entries = _assign_char_spans_from_morph_ranges(
            ignored_entries,
            ignored_ranges,
            morphs,
            caller_text_spans,
        )
        return _restore_caller_itaiji_surfaces(ignored_entries, caller_text, normalize_mode)

    morph_idx = 0
    number_block_end_base_idx = 0
    # 連語辞書エントリ (orig が「四捨:五入」のようにコロン区切り) は mecab2njd が NJD ノードを
    # 表層ごとに分割するため、1 morph が複数の NJD feature に対応する。分割消費中の残り表層を保持する
    split_remaining_surface = ""
    for base_idx, base_entry in enumerate(base_mapping):
        # 数詞ブロックは先頭ノードでまとめて出力済みなので、後続ノードの通常アライメントを省く
        if base_idx < number_block_end_base_idx:
            continue

        current_surface = base_entry["surface"]
        current_phonemes = base_entry["phonemes"]

        # 連語分割の継続: 同じ morph の残り表層を順に消費し、全断片へ同じ morph 範囲を割り当てる
        if split_remaining_surface != "":
            if (
                morph_idx < len(morphs)
                and split_remaining_surface.startswith(current_surface) is True
            ):
                _append_aligned(
                    _base_to_detail(
                        base_entry,
                        list(current_phonemes),
                        char_span=(0, 0),
                        is_unknown=morphs[morph_idx]["is_unknown"],
                        is_ignored=len(current_phonemes) == 0,
                    ),
                    (morph_idx, morph_idx + 1),
                    mecab_char_span=(
                        morphs[morph_idx]["char_span"][0]
                        + len(morphs[morph_idx]["surface"])
                        - len(split_remaining_surface),
                        morphs[morph_idx]["char_span"][0]
                        + len(morphs[morph_idx]["surface"])
                        - len(split_remaining_surface)
                        + len(current_surface),
                    ),
                )
                split_remaining_surface = split_remaining_surface[len(current_surface) :]
                if split_remaining_surface == "":
                    # 最後の断片を出力し終えてから、対応する morph を1つだけ消費する
                    morph_idx += 1
                continue
            # 後段の NJD 処理で断片がさらに変形した場合は、通常の不一致処理へ戻す
            split_remaining_surface = ""
            morph_idx += 1

        # is_ignored な morph を先に sp として出力
        while morph_idx < len(morphs):
            morph = morphs[morph_idx]
            if morph["is_ignored"] is True:
                _append_aligned(
                    _sp_entry(
                        morph["surface"],
                        char_span=(0, 0),
                        is_unknown=morph["is_unknown"],
                    ),
                    (morph_idx, morph_idx + 1),
                )
                morph_idx += 1
            else:
                break

        if morph_idx >= len(morphs):
            # morphs が尽きた: 後処理で feature 数が変動しうるため出力を継続
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    current_phonemes,
                    char_span=(0, 0),
                    is_ignored=len(current_phonemes) == 0,
                ),
                (0, 0),
            )
            continue

        morph = morphs[morph_idx]

        # NJD が位取り文字を挿入・吸収する数詞列は、個々のノード数から morph 消費数を決められない
        ## 入力側の数字と NJD 側の数詞をブロック単位で対応付け、各入力範囲を一度だけ割り当てる
        if _is_njd_number_surface(current_surface) is True and _is_njd_number_morph(morph) is True:
            number_block_end_base_idx = base_idx
            while (
                number_block_end_base_idx < len(base_mapping)
                and _is_njd_number_surface(base_mapping[number_block_end_base_idx]["surface"])
                is True
            ):
                number_block_end_base_idx += 1

            number_morph_indices: list[int] = []
            number_block_end_morph_idx = morph_idx
            last_number_morph_end_idx = morph_idx
            while number_block_end_morph_idx < len(morphs):
                number_morph = morphs[number_block_end_morph_idx]
                if number_morph["is_ignored"] is True:
                    number_block_end_morph_idx += 1
                    continue
                if _is_njd_number_morph(number_morph) is False:
                    break
                number_morph_indices.append(number_block_end_morph_idx)
                number_block_end_morph_idx += 1
                last_number_morph_end_idx = number_block_end_morph_idx

            # 24日 の「四日」のように次ノードが末尾数字を吸収する場合、その数字は純数詞ブロックへ渡さない
            ## NJD は「二十」「四日」と分割するため、ここで 2 と 4 の両方を「二十」へ割り当てると
            ## 後段の「四日」が 4 を再消費し、座標が重複する
            reserved_number_morph_count = 0
            if number_block_end_base_idx < len(base_mapping):
                next_surface = base_mapping[number_block_end_base_idx]["surface"]
                next_number_length = _njd_number_leading_length(next_surface)
                if 0 < next_number_length < len(next_surface):
                    next_number_key = _njd_number_alignment_key(next_surface[:next_number_length])
                    trailing_number_key = ""
                    for number_morph_index in reversed(number_morph_indices):
                        trailing_number_key = (
                            _njd_number_alignment_key(morphs[number_morph_index]["surface"])
                            + trailing_number_key
                        )
                        reserved_number_morph_count += 1
                        if trailing_number_key == next_number_key:
                            break
                        if next_number_key.endswith(trailing_number_key) is False:
                            reserved_number_morph_count = 0
                            break
                    if trailing_number_key != next_number_key:
                        reserved_number_morph_count = 0

            if reserved_number_morph_count > 0:
                first_reserved_morph_idx = number_morph_indices[-reserved_number_morph_count]
                number_morph_indices = number_morph_indices[:-reserved_number_morph_count]
                number_block_end_morph_idx = first_reserved_morph_idx
            else:
                # 数詞末尾の空白は次の通常ノードとの境界に残す
                number_block_end_morph_idx = last_number_morph_end_idx
            # 現在の base_entry が数詞の場合だけ入る分岐なので、この範囲は必ず1ノード以上になる
            number_entries = base_mapping[base_idx:number_block_end_base_idx]
            number_assignments = _align_njd_number_block(
                [entry["surface"] for entry in number_entries],
                morphs,
                number_morph_indices,
            )
            ignored_morph_indices = [
                index
                for index in range(morph_idx, number_block_end_morph_idx)
                if morphs[index]["is_ignored"] is True
            ]
            emitted_ignored_indices: set[int] = set()

            for number_entry, assigned_morph_indices in zip(
                number_entries,
                number_assignments,
            ):
                if len(assigned_morph_indices) == 0:
                    entry_morph_range = (0, 0)
                    assigned_morphs: list[MeCabMorph] = []
                else:
                    entry_morph_range = (
                        assigned_morph_indices[0],
                        assigned_morph_indices[-1] + 1,
                    )
                    assigned_morphs = [morphs[index] for index in assigned_morph_indices]

                    # 現在の数詞より前にある内部空白は、入力順を維持して先に出力する
                    for ignored_index in ignored_morph_indices:
                        if (
                            ignored_index >= entry_morph_range[0]
                            or ignored_index in emitted_ignored_indices
                        ):
                            continue
                        _append_aligned(
                            _sp_entry(
                                morphs[ignored_index]["surface"],
                                char_span=(0, 0),
                                is_unknown=morphs[ignored_index]["is_unknown"],
                            ),
                            (ignored_index, ignored_index + 1),
                        )
                        emitted_ignored_indices.add(ignored_index)

                # 表層が変わらない1対1対応だけは、従来どおり MeCab feature を引き継ぐ
                features = None
                if (
                    len(assigned_morphs) == 1
                    and assigned_morphs[0]["surface"] == number_entry["surface"]
                ):
                    features = assigned_morphs[0]["features"]
                _append_aligned(
                    _base_to_detail(
                        number_entry,
                        list(number_entry["phonemes"]),
                        char_span=(0, 0),
                        features=features,
                        is_unknown=any(
                            assigned_morph["is_unknown"] is True
                            for assigned_morph in assigned_morphs
                        ),
                        is_ignored=len(number_entry["phonemes"]) == 0,
                    ),
                    entry_morph_range,
                )

                # 1ノードが空白をまたいで複数桁を吸収した場合、空白はゼロ幅の sp として残す
                for ignored_index in ignored_morph_indices:
                    if (
                        ignored_index <= entry_morph_range[0]
                        or ignored_index >= entry_morph_range[1]
                        or ignored_index in emitted_ignored_indices
                    ):
                        continue
                    _append_aligned(
                        _sp_entry(
                            morphs[ignored_index]["surface"],
                            char_span=(0, 0),
                            is_unknown=morphs[ignored_index]["is_unknown"],
                        ),
                        (0, 0),
                    )
                    emitted_ignored_indices.add(ignored_index)

            # 最後の数詞より後ろに残った内部空白を回収する
            for ignored_index in ignored_morph_indices:
                if ignored_index in emitted_ignored_indices:
                    continue
                _append_aligned(
                    _sp_entry(
                        morphs[ignored_index]["surface"],
                        char_span=(0, 0),
                        is_unknown=morphs[ignored_index]["is_unknown"],
                    ),
                    (ignored_index, ignored_index + 1),
                )
            morph_idx = number_block_end_morph_idx
            continue

        # 完全一致: morph と NJD feature の surface が一致
        if current_surface == morph["surface"]:
            phonemes = list(current_phonemes)

            # 未知語を NJD が読点扱いした場合も、区切り記号と誤認させず unk へ戻す
            if morph["is_unknown"] is True and (len(phonemes) == 0 or phonemes == ["pau"]):
                phonemes = ["unk"]

            # is_ignored は音素列が空かで判定 (MeCab の is_ignored とは異なるセマンティクス)
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    phonemes,
                    char_span=(0, 0),
                    is_unknown=morph["is_unknown"],
                    is_ignored=len(current_phonemes) == 0,
                    features=morph["features"],
                ),
                (morph_idx, morph_idx + 1),
            )
            morph_idx += 1

        # 先頭一致: NJD が複数の morph を結合したケース
        elif current_surface.startswith(morph["surface"]):
            match_start_idx = morph_idx
            # 記号だけの NJD ノードは、発音を増やさず詳細形態素の表層粒度へ戻す
            ## MeCab の通常出力が連続記号を1ノードへまとめても、Lattice から復元した morphs は
            ## 1文字ずつ保持されるため、最初の形態素だけへ NJD のポーズ音素を割り当てる
            symbol_morphs: list[MeCabMorph] = []
            symbol_surface = ""
            symbol_morph_idx = morph_idx
            while symbol_morph_idx < len(morphs) and len(symbol_surface) < len(current_surface):
                symbol_morph = morphs[symbol_morph_idx]
                if (
                    symbol_morph["is_ignored"] is True
                    or len(symbol_morph["surface"]) != 1
                    or symbol_morph["surface"].isalnum() is True
                ):
                    break
                symbol_morphs.append(symbol_morph)
                symbol_surface += symbol_morph["surface"]
                symbol_morph_idx += 1

            is_restored_symbol_chunk = (
                len(symbol_morphs) > 1
                and symbol_surface == current_surface
                and (len(current_phonemes) == 0 or current_phonemes == ["pau"])
            )
            if is_restored_symbol_chunk is True:
                symbol_range_start = morph_idx
                for symbol_idx, symbol_morph in enumerate(symbol_morphs):
                    symbol_mapping = _base_to_detail(
                        base_entry,
                        list(current_phonemes) if symbol_idx == 0 else [],
                        char_span=(0, 0),
                        features=symbol_morph["features"],
                        is_unknown=symbol_morph["is_unknown"],
                        is_ignored=False,
                    )
                    symbol_mapping["surface"] = symbol_morph["surface"]
                    symbol_mapping["orig"] = symbol_morph["surface"]
                    _append_aligned(
                        symbol_mapping,
                        (symbol_range_start + symbol_idx, symbol_range_start + symbol_idx + 1),
                    )
                morph_idx = symbol_morph_idx
                continue

            is_unknown_word = False
            matched_len = 0
            internal_ignored_entries: list[SurfacePhonemeMapping] = []

            while morph_idx < len(morphs):
                inner_morph = morphs[morph_idx]

                # 結合語の内部にある空白は、表層の構成要素を先に出してから直後へ戻す
                ## その場で result へ追加すると、まだ未出力の結合語より空白が前へ移動してしまう
                if inner_morph["is_ignored"] is True:
                    internal_ignored_entries.append(
                        _sp_entry(
                            inner_morph["surface"],
                            char_span=(0, 0),
                            is_unknown=inner_morph["is_unknown"],
                        )
                    )
                    morph_idx += 1
                    continue

                remaining = current_surface[matched_len:]

                if remaining.startswith(inner_morph["surface"]):
                    # いずれかの構成トークンが未知語なら全体を未知語とみなす
                    is_unknown_word = is_unknown_word or inner_morph["is_unknown"]
                    matched_len += len(inner_morph["surface"])
                    morph_idx += 1

                    if matched_len == len(current_surface):
                        break
                else:
                    break

            phonemes = list(current_phonemes)

            # 結合語を構成する未知語が読点扱いされた場合も unk へ戻す
            if is_unknown_word is True and (len(phonemes) == 0 or phonemes == ["pau"]):
                phonemes = ["unk"]

            _append_aligned(
                _base_to_detail(
                    base_entry,
                    phonemes,
                    char_span=(0, 0),
                    is_unknown=is_unknown_word,
                    is_ignored=len(current_phonemes) == 0,
                ),
                (match_start_idx, morph_idx),
            )
            for ignored_entry in internal_ignored_entries:
                # 結合ノードの char_span が内部空白も覆うため、sp へ同じ実座標を重ねない
                _append_aligned(ignored_entry, (0, 0))

        # 分割一致: 連語辞書エントリで NJD が1 morph を複数ノードへ分割したケース
        # (例: morph '四捨五入' → NJD '四捨' + '五入')
        # 後続 NJD 表層の連結で morph 表層を厳密に復元できる場合だけ分割として扱い、数字展開などの偶然の前方一致は除外する
        elif (
            morph["surface"].startswith(current_surface)
            and _is_split_morph(base_mapping, base_idx, morph["surface"]) is True
        ):
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    list(current_phonemes),
                    char_span=(0, 0),
                    is_unknown=morph["is_unknown"],
                    is_ignored=len(current_phonemes) == 0,
                ),
                (morph_idx, morph_idx + 1),
                mecab_char_span=(
                    morph["char_span"][0],
                    morph["char_span"][0] + len(current_surface),
                ),
            )
            # morph は最後の断片を処理し終えるまで消費しない (継続処理が split_remaining_surface で追跡する)
            split_remaining_surface = morph["surface"][len(current_surface) :]

        # 不一致: 数字正規化・踊り字展開等で surface が変化したケース
        # 数詞列は上のブロック処理、数詞と助数詞の縮約は _njd_digit_compound_morph_range() で完結する
        # ここでは踊り字展開と、ノード数が変わらない通常の surface 変化だけを扱う
        else:
            # 不一致ブランチでは morph と NJD の surface が異なるため、
            # morph の features をこのエントリに紐づけると嘘データになる (features は空リスト)
            compound_morph_range = _njd_digit_compound_morph_range(
                morphs,
                morph_idx,
                current_surface,
            )
            if compound_morph_range != (morph_idx, morph_idx + 1):
                entry_morph_range = compound_morph_range
            else:
                entry_morph_range = (morph_idx, morph_idx + 1)
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    list(current_phonemes),
                    char_span=(0, 0),
                    is_ignored=len(current_phonemes) == 0,
                ),
                entry_morph_range,
            )

            current_morph_surface = morphs[morph_idx]["surface"]
            has_odori = any(c in _ODORI_CHARS for c in current_morph_surface)

            # digit+morph 縮約 (2人→二人) は後続の数字消費ロジックを通さずまとめて進める
            if compound_morph_range != (morph_idx, morph_idx + 1):
                morph_idx = compound_morph_range[1]
                continue

            # A) 踊り字展開: 踊り字 morph + 結合先 morph を消費
            # 踊り字展開では、単独の踊り字 morph ('々' 等) と後続の漢字 morph が
            # 結合されて 1 つの NJD feature になる (例: morphs['々','活'] → NJD '生活')
            if has_odori is True:
                odori_morph_start = morph_idx
                morph_idx += 1
                # 結合先 morph の判定: current_surface の末尾と次の morph の surface が一致
                # 結合先がないケース (例: '学生々' → NJD='生') では追加消費しない
                if morph_idx < len(morphs):
                    ahead = morphs[morph_idx]
                    if ahead["is_ignored"] is not True and current_surface.endswith(
                        ahead["surface"]
                    ):
                        morph_idx += 1
                morph_ranges[-1] = (odori_morph_start, morph_idx)

            else:
                # ノード数が変わらない通常の surface 変化は対応する morph を1つだけ消費する
                morph_idx += 1

    # morphs 末尾に残った is_ignored トークンを sp として回収
    while morph_idx < len(morphs):
        morph = morphs[morph_idx]
        if morph["is_ignored"] is True:
            _append_aligned(
                _sp_entry(morph["surface"], char_span=(0, 0), is_unknown=morph["is_unknown"]),
                (morph_idx, morph_idx + 1),
            )
        morph_idx += 1

    result = _assign_char_spans_from_morph_ranges(
        result,
        morph_ranges,
        morphs,
        caller_text_spans,
        mecab_char_span_overrides,
    )
    return _restore_caller_itaiji_surfaces(result, caller_text, normalize_mode)


def _build_caller_text_spans_by_mecab_character(
    inference_jtalk: OpenJTalk,
    mecab_text: str,
    reference_text: str,
    normalize_mode: NormalizeMode,
) -> list[tuple[int, int]]:
    """
    MeCab 正規化本文の各文字に対応する呼び出し元入力上の範囲を返す。

    Args:
        inference_jtalk (OpenJTalk): text2mecab と同じ正規化を行うインスタンス
        mecab_text (str): morph 表層を連結した MeCab 側文字列
        reference_text (str): `g2p_mapping()` に渡した入力文
        normalize_mode (NormalizeMode): reference_text に適用した Unicode 正規化方式

    Returns:
        list[tuple[int, int]]: MeCab 側の各文字に対応する入力文上の半開区間
    """

    source_spans: list[tuple[int, int]] = []
    # 素の入力と、異体字を通用字に変えた入力の2通りを MeCab 側の文字列と照合する
    mecab_text_by_source_chunk: dict[str, tuple[str, str]] = {}
    source_start = 0
    mecab_start = 0
    while source_start < len(reference_text):
        # NUL 以降は C 文字列として MeCab へ渡らないため、対応先が尽きた時点で残りを無視する
        if mecab_start == len(mecab_text):
            break
        matched_end: int | None = None
        matched_text = ""
        maximum_source_end = min(
            source_start + _MAX_CALLER_TEXT_CHUNK_LENGTH,
            len(reference_text),
        )
        for source_end in range(source_start + 1, maximum_source_end + 1):
            source_chunk = reference_text[source_start:source_end]
            candidate_text_pair = mecab_text_by_source_chunk.get(source_chunk)
            if candidate_text_pair is None:
                normalized_chunk = normalize_text(source_chunk, normalize_mode)
                candidate_text_pair = (
                    inference_jtalk.normalize_for_mecab(normalized_chunk),
                    inference_jtalk.normalize_for_mecab(normalize_itaiji(normalized_chunk)),
                )
                mecab_text_by_source_chunk[source_chunk] = candidate_text_pair

            # 制御文字など、正規化時に消える1文字は対応する MeCab 文字を持たない
            if candidate_text_pair[0] == "" and source_end == source_start + 1:
                matched_end = source_end
                break
            for candidate_text in dict.fromkeys(candidate_text_pair):
                if mecab_text.startswith(candidate_text, mecab_start) is True:
                    matched_end = source_end
                    matched_text = candidate_text
                    break
            if matched_end is not None:
                break

        if matched_end is None:
            raise ValueError("caller text normalization does not match MeCab text")
        source_spans.extend([(source_start, matched_end)] * len(matched_text))
        source_start = matched_end
        mecab_start += len(matched_text)

    if mecab_start != len(mecab_text):
        raise ValueError("caller text normalization does not cover MeCab text")
    return source_spans


def _base_to_detail(
    base: JPCommonMappingEntry,
    phonemes: list[str],
    *,
    char_span: tuple[int, int],
    features: list[str] | None = None,
    is_unknown: bool = False,
    is_ignored: bool = False,
) -> SurfacePhonemeMapping:
    """
    Cython 側 base_mapping の1エントリから SurfacePhonemeMapping を構築する。

    Args:
        base (JPCommonMappingEntry): `OpenJTalk.make_phoneme_mapping()` の1要素
        phonemes (list[str]): 割り当て済み音素列
        char_span (tuple[int, int]): 入力文上の半開区間
        features (list[str] | None): MeCab feature 列。不明な場合は空 list
        is_unknown (bool): MeCab 未知語フラグ
        is_ignored (bool): アライメント上無視対象か

    Returns:
        SurfacePhonemeMapping: 詳細 API 向けマッピング1件
    """

    return SurfacePhonemeMapping(
        surface=base["surface"],
        phonemes=phonemes,
        features=features if features is not None else [],
        char_span=char_span,
        pos=base["pos"],
        pos_group1=base["pos_group1"],
        pos_group2=base["pos_group2"],
        pos_group3=base["pos_group3"],
        ctype=base["ctype"],
        cform=base["cform"],
        orig=base["orig"],
        read=base["read"],
        pron=base["pron"],
        accent_nucleus=base["accent_nucleus"],
        mora_count=base["mora_count"],
        chain_rule=base["chain_rule"],
        chain_flag=base["chain_flag"],
        is_unknown=is_unknown,
        is_ignored=is_ignored,
    )


def _sp_entry(
    surface: str,
    *,
    char_span: tuple[int, int],
    is_unknown: bool = False,
) -> SurfacePhonemeMapping:
    """
    is_ignored な morph 向けの sp エントリを構築する。

    Args:
        surface (str): 表層形 (通常は空白)
        char_span (tuple[int, int]): 入力文上の半開区間
        is_unknown (bool): MeCab 未知語フラグ

    Returns:
        SurfacePhonemeMapping: phonemes=["sp"] のマッピング1件
    """

    return SurfacePhonemeMapping(
        surface=surface,
        phonemes=["sp"],
        features=[],
        char_span=char_span,
        pos="記号",
        pos_group1="空白",
        pos_group2="*",
        pos_group3="*",
        ctype="*",
        cform="*",
        orig=surface,
        read=surface,
        pron=surface,
        accent_nucleus=0,
        mora_count=0,
        chain_rule="*",
        chain_flag=-1,
        is_unknown=is_unknown,
        is_ignored=True,
    )


def _assign_char_spans_from_morph_ranges(
    entries: list[SurfacePhonemeMapping],
    aligned_morph_ranges: list[tuple[int, int]],
    morphs: list[MeCabMorph],
    caller_text_spans: list[tuple[int, int]],
    mecab_char_spans: list[tuple[int, int] | None] | None = None,
) -> list[SurfacePhonemeMapping]:
    """
    morph_range を char_span へ写し、必要なら呼び出し元入力座標へ射影する。

    Args:
        entries (list[SurfacePhonemeMapping]): アライメント済み mapping
        aligned_morph_ranges (list[tuple[int, int]]): 各 entry に対応する morph 添字半開区間
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        caller_text_spans (list[tuple[int, int]]): MeCab 側の各文字に対応する呼び出し元入力上の範囲
        mecab_char_spans (list[tuple[int, int] | None] | None): morph 内部の部分範囲を指定する MeCab 座標

    Returns:
        list[SurfacePhonemeMapping]: char_span を付与した mapping
    """

    if len(entries) != len(aligned_morph_ranges):
        raise ValueError("aligned entry count must match morph_range count")
    resolved_mecab_char_span_overrides: list[tuple[int, int] | None]
    if mecab_char_spans is None:
        resolved_mecab_char_span_overrides = [None] * len(entries)
    else:
        resolved_mecab_char_span_overrides = mecab_char_spans
    if len(entries) != len(resolved_mecab_char_span_overrides):
        raise ValueError("aligned entry count must match MeCab char_span count")
    resolved_mecab_char_spans = [
        mecab_char_span
        if mecab_char_span is not None
        else _char_span_from_morph_range(morphs, aligned_morph_ranges[index])
        for index, mecab_char_span in enumerate(resolved_mecab_char_span_overrides)
    ]
    # entries はこの関数内で新規生成した辞書なので、全フィールドを複製せず位置だけ確定する
    for index, entry in enumerate(entries):
        entry["char_span"] = _project_char_span(
            caller_text_spans,
            resolved_mecab_char_spans[index],
        )
    return entries


def _char_span_from_morph_range(
    morphs: list[MeCabMorph],
    morph_range: tuple[int, int],
) -> tuple[int, int]:
    """
    morph 添字半開区間から MeCab 正規化本文上の char_span を返す。

    Args:
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        morph_range (tuple[int, int]): morph 添字の半開区間

    Returns:
        tuple[int, int]: MeCab 正規化本文上の半開区間
    """

    morph_start, morph_end = morph_range
    if morph_start >= morph_end:
        return (0, 0)
    return (morphs[morph_start]["char_span"][0], morphs[morph_end - 1]["char_span"][1])


def _project_char_span(
    source_spans: list[tuple[int, int]], char_span: tuple[int, int]
) -> tuple[int, int]:
    """
    MeCab 座標の半開区間を呼び出し元入力座標へ射影する。

    Args:
        source_spans (list[tuple[int, int]]): MeCab 側の各文字に対応する入力文上の範囲
        char_span (tuple[int, int]): MeCab 正規化本文上の半開区間

    Returns:
        tuple[int, int]: 呼び出し元入力上の半開区間
    """

    char_start, char_end = char_span
    if char_start >= char_end:
        return (0, 0)
    return (source_spans[char_start][0], source_spans[char_end - 1][1])


def _restore_caller_itaiji_surfaces(
    entries: list[SurfacePhonemeMapping],
    caller_text: str | None,
    normalize_mode: NormalizeMode,
) -> list[SurfacePhonemeMapping]:
    """
    異体字を通用字に変えて解析した形態素の表層を、同じ文字位置にある呼び出し元の異体字へ戻す。

    Args:
        entries (list[SurfacePhonemeMapping]): char_span を付与した mapping
        caller_text (str | None): `char_span` の座標系に使った正規化前の入力文
        normalize_mode (NormalizeMode): caller_text に適用した Unicode 正規化方式

    Returns:
        list[SurfacePhonemeMapping]: 表層を呼び出し元の表記に戻した mapping
    """

    if caller_text is None:
        return entries
    for entry in entries:
        char_start, char_end = entry["char_span"]
        if (char_start, char_end) == (0, 0):
            continue
        source_surface = caller_text[char_start:char_end]
        normalized_source_surface = normalize_text(source_surface, normalize_mode)
        itaiji_normalized_surface = normalize_itaiji(normalized_source_surface)
        if (
            itaiji_normalized_surface != normalized_source_surface
            and itaiji_normalized_surface == entry["surface"]
        ):
            entry["surface"] = source_surface
    return entries


def _is_njd_number_surface(surface: str) -> bool:
    """
    NJD 表層が数字展開の構成要素だけであるかを返す。

    Args:
        surface (str): 判定対象の表層形

    Returns:
        bool: 漢数字または算用数字のみなら True
    """

    return surface != "" and all(
        character in _KANJI_NUMBER_SURFACES or character in _DIGIT_MORPH_SURFACES
        for character in surface
    )


def _is_njd_number_morph(morph: MeCabMorph) -> bool:
    """
    NJD が数詞列として変換する MeCab 形態素かを返す。

    Args:
        morph (MeCabMorph): 判定対象の MeCab 形態素

    Returns:
        bool: 品詞が数で、NJD の数字変換表に存在する表層なら True
    """

    return (
        len(morph["features"]) > 2
        and morph["features"][2] == "数"
        and (
            morph["surface"] in _NJD_NUMBER_MORPH_SURFACE_KEYS
            or _is_njd_number_surface(morph["surface"]) is True
        )
    )


def _njd_number_leading_length(surface: str) -> int:
    """
    表層先頭から続く漢数字文字数を返す。

    Args:
        surface (str): NJD 側の表層形

    Returns:
        int: 先頭漢数字の文字数
    """

    leading_length = 0
    for character in surface:
        if character not in _KANJI_NUMBER_SURFACES:
            break
        leading_length += 1
    return leading_length


def _njd_number_alignment_key(surface: str) -> str:
    """
    算用数字と対応する漢数字を同じ比較表現へ変換する。

    Args:
        surface (str): MeCab または NJD 側の数字表層

    Returns:
        str: 数字表記を漢数字へ寄せた比較用文字列
    """

    return _NJD_NUMBER_MORPH_SURFACE_KEYS.get(
        surface,
        surface.translate(_NJD_NUMBER_ALIGNMENT_TRANSLATION),
    )


def _align_njd_number_block(
    feature_surfaces: list[str],
    morphs: list[MeCabMorph],
    morph_indices: list[int],
) -> list[list[int]]:
    """
    NJD の数詞列へ入力側の数字形態素を重複なく対応付ける。

    NJD は位取り文字を挿入する一方、ゼロや助数詞との結合では入力形態素を吸収する。
    数詞ブロック全体の編集距離を最小化し、挿入された形態素には入力範囲を割り当てず、
    吸収された入力は直前の出力形態素へまとめる。

    Args:
        feature_surfaces (list[str]): 連続する NJD 数詞表層
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        morph_indices (list[int]): 連続する入力側数字形態素の添字

    Returns:
        list[list[int]]: 各 NJD 数詞形態素が消費する MeCab 形態素添字
    """

    source_keys = [
        _njd_number_alignment_key(morphs[morph_index]["surface"]) for morph_index in morph_indices
    ]
    target_keys = [_njd_number_alignment_key(surface) for surface in feature_surfaces]
    source_count = len(source_keys)
    target_count = len(target_keys)
    assignments: list[list[int]] = [[] for _ in feature_surfaces]

    # 異常に長い数詞では編集距離表を作らず、入力順に1対1で消費する
    ## NJD 側が少ない場合の余りは最後の出力形態素へ集約し、入力範囲を保持する
    if (
        source_count > _MAX_NUMBER_ALIGNMENT_BLOCK_LENGTH
        or target_count > _MAX_NUMBER_ALIGNMENT_BLOCK_LENGTH
    ):
        for source_index, morph_index in enumerate(morph_indices):
            assignments[min(source_index, target_count - 1)].append(morph_index)
        return assignments

    # 数字の挿入・吸収をブロック全体で決めるため、最小編集経路を表に保持する
    edit_costs = [[0] * (target_count + 1) for _ in range(source_count + 1)]
    edit_actions = [[""] * (target_count + 1) for _ in range(source_count + 1)]
    for source_index in range(1, source_count + 1):
        edit_costs[source_index][0] = source_index
        edit_actions[source_index][0] = "delete"
    for target_index in range(1, target_count + 1):
        edit_costs[0][target_index] = target_index
        edit_actions[0][target_index] = "insert"

    # 同値対応を最優先し、同じ費用なら吸収、挿入の順で安定した経路を選ぶ
    for source_index in range(1, source_count + 1):
        for target_index in range(1, target_count + 1):
            substitution_cost = int(source_keys[source_index - 1] != target_keys[target_index - 1])
            candidates = [
                (
                    edit_costs[source_index - 1][target_index - 1] + substitution_cost,
                    0,
                    "align",
                ),
                (edit_costs[source_index - 1][target_index] + 1, 1, "delete"),
                (edit_costs[source_index][target_index - 1] + 1, 2, "insert"),
            ]
            best_cost, _priority, best_action = min(candidates)
            edit_costs[source_index][target_index] = best_cost
            edit_actions[source_index][target_index] = best_action

    # 逆向きの編集経路を入力順へ戻し、各出力形態素が消費する位置を確定する
    reversed_actions: list[tuple[str, int | None, int | None]] = []
    source_index = source_count
    target_index = target_count
    while source_index > 0 or target_index > 0:
        action = edit_actions[source_index][target_index]
        if action == "align":
            reversed_actions.append((action, source_index - 1, target_index - 1))
            source_index -= 1
            target_index -= 1
        elif action == "delete":
            reversed_actions.append((action, source_index - 1, None))
            source_index -= 1
        else:
            reversed_actions.append(("insert", None, target_index - 1))
            target_index -= 1

    # 吸収された入力は直前の出力へまとめ、NJD の縮約後も元の文字範囲を保持する
    pending_morph_indices: list[int] = []
    previous_target_index: int | None = None
    for action, aligned_source_index, aligned_target_index in reversed(reversed_actions):
        if action == "delete":
            assert aligned_source_index is not None
            if previous_target_index is None:
                pending_morph_indices.append(morph_indices[aligned_source_index])
            else:
                assignments[previous_target_index].append(morph_indices[aligned_source_index])
            continue
        if action == "insert":
            continue
        assert aligned_source_index is not None
        assert aligned_target_index is not None
        assignments[aligned_target_index].extend(pending_morph_indices)
        pending_morph_indices = []
        assignments[aligned_target_index].append(morph_indices[aligned_source_index])
        previous_target_index = aligned_target_index

    # 先頭で吸収された入力だけが残った場合も、最寄りの出力形態素へ範囲を引き継ぐ
    if len(pending_morph_indices) > 0:
        fallback_target_index = previous_target_index if previous_target_index is not None else 0
        assignments[fallback_target_index].extend(pending_morph_indices)
    for assignment in assignments:
        assignment.sort()
    return assignments


def _is_split_morph(
    entries: list[JPCommonMappingEntry],
    start_idx: int,
    morph_surface: str,
) -> bool:
    """
    start_idx 以降の NJD 表層の連結が morph 表層を厳密に復元できるかを返す。

    Args:
        entries (list[JPCommonMappingEntry]): NJD 由来の基本 mapping
        start_idx (int): 復元を開始する mapping 添字
        morph_surface (str): 復元対象の morph 表層

    Returns:
        bool: 2ノード以上の連結で morph 表層と完全一致すれば True
    """

    concatenated = ""
    for entry in entries[start_idx:]:
        concatenated += entry["surface"]
        if len(concatenated) >= len(morph_surface):
            break
    # 1ノードで一致する場合は完全一致ブランチの領分なので、分割は2ノード以上に限る
    return concatenated == morph_surface and len(entries[start_idx]["surface"]) < len(morph_surface)


def _njd_digit_compound_morph_range(
    morphs: list[MeCabMorph],
    morph_index: int,
    feature_surface: str,
) -> tuple[int, int]:
    """
    数字と後続形態素が NJD で1語へ縮約された範囲を返す。

    例: morphs['２','人'] → NJD '二人'、morphs['１','日'] → NJD '一日'

    Args:
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        morph_index (int): 現在の MeCab 形態素添字
        feature_surface (str): 対応する NJD 表層

    Returns:
        tuple[int, int]: 対応する MeCab 形態素添字の半開区間
    """

    # 先頭の漢数字と接尾語が同じ NJD 形態素に入った場合だけ、縮約候補として扱う
    leading_number_length = _njd_number_leading_length(feature_surface)
    suffix = feature_surface[leading_number_length:]
    if leading_number_length == 0 or suffix == "" or morph_index >= len(morphs):
        return (morph_index, morph_index + 1)

    # 分割後の「四日」のように直前の数字と現在の接尾語が結び付く範囲を返す
    if (
        morphs[morph_index]["surface"] == suffix
        and morph_index > 0
        and _is_njd_number_morph(morphs[morph_index - 1]) is True
    ):
        return (morph_index - 1, morph_index + 1)
    if _is_njd_number_morph(morphs[morph_index]) is False:
        return (morph_index, morph_index + 1)

    # 複数桁を吸収したあと、残りの表層と一致する後続形態素までを同じ範囲へ含める
    morph_end = morph_index + 1
    while morph_end < len(morphs) and _is_njd_number_morph(morphs[morph_end]) is True:
        morph_end += 1
    consumed_suffix = ""
    while morph_end < len(morphs) and len(consumed_suffix) < len(suffix):
        candidate_morph = morphs[morph_end]
        if candidate_morph["is_ignored"] is True:
            morph_end += 1
            continue
        remaining_suffix = suffix[len(consumed_suffix) :]
        if remaining_suffix.startswith(candidate_morph["surface"]) is False:
            break
        consumed_suffix += candidate_morph["surface"]
        morph_end += 1
    if consumed_suffix == suffix:
        return (morph_index, morph_end)
    return (morph_index, morph_index + 1)
