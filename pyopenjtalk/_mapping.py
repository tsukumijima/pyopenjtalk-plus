"""
形態素-音素マッピングと、NJD・MeCab 形態素のアライメントの実装。

NJD の数詞変換・踊り字展開・長音吸収によって MeCab 形態素と NJD ノードの対応がずれるため、
表層・入力文字位置・音素列の対応付けはこのモジュールへ集約している。
Haqumei (Rust 実装) の open_jtalk/mapping.rs に対応する。
"""

from collections.abc import Callable, Sequence

from .openjtalk import OpenJTalk
from .types import (
    JPCommonMappingEntry,
    MeCabMorph,
    NJDFeature,
    NormalizeMode,
    SurfacePhonemeMapping,
)
from .utils import normalize_itaiji, normalize_roman_numerals, normalize_text, split_kana_mora


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
    # 大字の「拾」と旧字体の「萬」は、NJD の数詞処理が「十」「万」に置き換える
    "拾": "十",
    "萬": "万",
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


def mark_user_dictionary_reading_protection(
    njd_features: list[NJDFeature],
    morphs: list[MeCabMorph],
    reading_protection: tuple[bool, ...],
) -> None:
    """
    NJD 形態素へユーザー辞書の読み保護状態を位置順で対応付ける。

    Args:
        njd_features (list[NJDFeature]): NJD 処理後の形態素列
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        reading_protection (tuple[bool, ...]): ユーザー辞書ごとの読み保護フラグ
    """

    if any(reading_protection) is False:
        return

    feature_char_spans = njd_feature_char_spans(njd_features, morphs)
    morph_index = 0
    for feature, feature_char_span in zip(njd_features, feature_char_spans, strict=True):
        feature_start, feature_end = feature_char_span

        # 位取りとして挿入された NJD 形態素は元の文字を持たないため、保護対象から外す
        if feature_start == feature_end:
            continue

        # 通過済みの入力位置を戻らずに進め、長文でも形態素数に比例する範囲で候補を探す
        while morph_index < len(morphs) and morphs[morph_index]["char_span"][1] <= feature_start:
            morph_index += 1

        # 同じ文字範囲と重なる MeCab 形態素だけを候補にし、同表層の別位置を分離する
        overlapping_morphs: list[MeCabMorph] = []
        candidate_morph_index = morph_index
        while (
            candidate_morph_index < len(morphs)
            and morphs[candidate_morph_index]["char_span"][0] < feature_end
        ):
            candidate_morph = morphs[candidate_morph_index]
            if (
                candidate_morph["is_ignored"] is False
                and feature_start < candidate_morph["char_span"][1]
            ):
                overlapping_morphs.append(candidate_morph)
            candidate_morph_index += 1
        if len(overlapping_morphs) == 0:
            continue

        # 結合された NJD 形態素では、重なる入力のいずれかが保護辞書由来なら読みを保護する
        protected_morphs = [
            morph
            for morph in overlapping_morphs
            if 1 <= morph["dictionary_index"] <= len(reading_protection)
            and reading_protection[morph["dictionary_index"] - 1] is True
        ]
        feature["is_reading_protected"] = len(protected_morphs) > 0
        if len(protected_morphs) == 0:
            continue

        # 1形態素の全範囲に一致する保護語は、NJD 前処理で読み替え済みでも辞書の値へ戻す
        ## 複数アクセント句や複数形態素の結合は、NJD が分割・結合した読みをそのまま保護する
        if len(overlapping_morphs) != 1 or overlapping_morphs[0]["char_span"] != feature_char_span:
            continue
        morph_features = overlapping_morphs[0]["features"]
        if (
            len(morph_features) >= 10
            and ":" not in morph_features[8]
            and ":" not in morph_features[9]
            # 無声化記号の有無だけの差は読み替えではないため、NJD が推定した無声化ごと現在値を保持する
            ## 辞書の生値で上書きすると、登録語の全てで母音の無声化が失われて発音の自然さが下がる
            and (
                feature["read"].replace("’", "") != morph_features[8].replace("’", "")
                or feature["pron"].replace("’", "") != morph_features[9].replace("’", "")
            )
        ):
            feature["read"] = morph_features[8]
            feature["pron"] = morph_features[9]
            feature["mora_size"] = len(split_kana_mora(morph_features[9].replace("’", "")))


def make_phoneme_mapping(
    njd_features: list[NJDFeature],
    morphs: list[MeCabMorph] | None,
    *,
    caller_text: str | None = None,
    normalize_mode: NormalizeMode = "None",
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
    inference_jtalk: OpenJTalk,
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
    句読点などの記号は surface として保持し、is_non_pause_symbol() が True を返す記号の phonemes は空にする。
    False を返す記号には、JPCommon が短ポーズを生成しなかった場合も phonemes に `pau` を割り当てるので、音素を連結した結果が extract_phonemes() と一致しないことがある。
    morphs 付きで対応する morph を特定できないエントリの `char_span` は、未特定を表す `(0, 0)` になる。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        morphs (list[MeCabMorph] | None): MeCab の形態素解析結果 (pyopenjtalk.run_frontend_detailed() の戻り値)
            None の場合は is_unknown / is_ignored の推定精度が下がる
        caller_text (str | None): `char_span` の座標系に使う正規化前の入力文
            None の場合は MeCab 正規化本文上の座標を使う
        normalize_mode (NormalizeMode): caller_text に適用した Unicode 正規化方式 (デフォルト: `"None"`)
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する
        inference_jtalk (OpenJTalk): 借り出し済みの OpenJTalk インスタンス

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング

    Raises:
        ValueError: caller_text と MeCab 正規化本文の対応付けに失敗した場合。
            `g2p_mapping()` 経由でも `build_caller_text_spans_by_mecab_character()` からそのまま伝播する
    """

    # Cython レベルで基本マッピングと長音吸収マージを取得し、呼び出し元座標への変換まで同じインスタンスで行う
    base_mapping = inference_jtalk.make_phoneme_mapping(njd_features)

    # Cython 側の既定マッピングを呼び出し側の記号判定で上書きし、通常音素の対応付けは維持する
    for entry in base_mapping:
        # 「1・2年生」の数を並べる中黒は休止を置かないので、音素なしのまま保つ
        if entry["pron"] not in ("、", "？", "！") or entry["pos_group3"] == "数の区切り":
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
        caller_text_spans = build_caller_text_spans_by_mecab_character(
            mecab_text,
            reference_text,
            normalize_mode,
            inference_jtalk=inference_jtalk,
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
        ## 「　―「」のように NJD が空白を記号と1ノードにまとめたときは、空白もそのノードの表層の一部なので、下の先頭一致でノードと一緒に消費する
        while morph_idx < len(morphs):
            morph = morphs[morph_idx]
            if (
                morph["is_ignored"] is True
                and len(current_surface) > len(morph["surface"])
                and current_surface.startswith(morph["surface"]) is True
            ):
                break
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
                and (
                    number_block_end_base_idx == base_idx
                    or base_mapping[number_block_end_base_idx]["pos_group1"] != "固有名詞"
                )
                and (
                    number_block_end_base_idx == base_idx
                    or base_mapping[number_block_end_base_idx - 1]["surface"].endswith("ー")
                    is False
                )
            ):
                number_block_end_base_idx += 1

            number_entries = base_mapping[base_idx:number_block_end_base_idx]
            next_surface = (
                base_mapping[number_block_end_base_idx]["surface"]
                if number_block_end_base_idx < len(base_mapping)
                else ""
            )
            number_morph_indices, number_block_end_morph_idx = _collect_njd_number_morphs(
                morphs,
                morph_idx,
                [entry["surface"] for entry in number_entries],
                next_surface,
            )
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

        # 「二〇日」→「二十日」は先頭の「二」も一致するため、先頭一致より前に縮約範囲を確定する
        compound_morph_range = _njd_digit_compound_morph_range(morphs, morph_idx, current_surface)
        if compound_morph_range != (morph_idx, morph_idx + 1):
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    list(current_phonemes),
                    char_span=(0, 0),
                    is_ignored=len(current_phonemes) == 0,
                ),
                compound_morph_range,
            )
            morph_idx = compound_morph_range[1]
            continue

        # 完全一致: morph と NJD feature の surface が一致
        if current_surface == morph["surface"]:
            phonemes = list(current_phonemes)

            # 「070ー3224」のように NJD が読点へ変えた番号の区切りは pau を保ち、読みのない未知語だけ unk へ戻す
            if (
                morph["is_unknown"] is True
                and (len(phonemes) == 0 or phonemes == ["pau"])
                and not (current_surface == "ー" and base_entry["pron"] == "、")
            ):
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
            ## 「　―「」「㏄　＋　」のように NJD が間の空白までまとめたノードも、空白を sp として戻す
            symbol_morphs: list[MeCabMorph] = []
            symbol_surface = ""
            symbol_morph_idx = morph_idx
            while symbol_morph_idx < len(morphs) and len(symbol_surface) < len(current_surface):
                symbol_morph = morphs[symbol_morph_idx]
                if (
                    symbol_morph["is_ignored"] is False
                    and (
                        len(symbol_morph["surface"]) != 1
                        or symbol_morph["surface"].isalnum() is True
                    )
                ) or current_surface.startswith(
                    symbol_morph["surface"], len(symbol_surface)
                ) is False:
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
                first_symbol_idx = next(
                    (
                        symbol_idx
                        for symbol_idx, symbol_morph in enumerate(symbol_morphs)
                        if symbol_morph["is_ignored"] is False
                    ),
                    0,
                )
                for symbol_idx, symbol_morph in enumerate(symbol_morphs):
                    if symbol_morph["is_ignored"] is True:
                        _append_aligned(
                            _sp_entry(
                                symbol_morph["surface"],
                                char_span=(0, 0),
                                is_unknown=symbol_morph["is_unknown"],
                            ),
                            (symbol_range_start + symbol_idx, symbol_range_start + symbol_idx + 1),
                        )
                        continue
                    symbol_mapping = _base_to_detail(
                        base_entry,
                        list(current_phonemes) if symbol_idx == first_symbol_idx else [],
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
            _append_aligned(
                _base_to_detail(
                    base_entry,
                    list(current_phonemes),
                    char_span=(0, 0),
                    is_ignored=len(current_phonemes) == 0,
                ),
                (morph_idx, morph_idx + 1),
            )

            current_morph_surface = morphs[morph_idx]["surface"]
            has_odori = any(c in _ODORI_CHARS for c in current_morph_surface)

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

    _assign_reading_sources(result, morph_ranges, morphs, inference_jtalk.userdic_paths)
    result = _assign_char_spans_from_morph_ranges(
        result,
        morph_ranges,
        morphs,
        caller_text_spans,
        mecab_char_span_overrides,
    )
    return _restore_caller_itaiji_surfaces(result, caller_text, normalize_mode)


def check_caller_char_spans(
    mapping: Sequence[SurfacePhonemeMapping], text: str, function_name: str
) -> None:
    """
    マッピングの `char_span` が、呼び出し元の本文を先頭から1度ずつ覆っているかを確かめる。
    MeCab が読み飛ばす ASCII 制御文字だけの範囲は、形態素の間や文頭・文末に残っていてもよい。

    Args:
        mapping (Sequence[SurfacePhonemeMapping]): 確かめるマッピング
        text (str): 呼び出し元の入力文
        function_name (str): エラーの文言に入れる公開関数の名前

    Raises:
        ValueError: `char_span` が ASCII 制御文字を除く本文を1度ずつ覆っていない場合
    """

    expected_start = 0
    for entry in mapping:
        char_start, char_end = entry["char_span"]
        if (char_start, char_end) == (0, 0):
            continue
        # 形態素の間で消えた ASCII 制御文字は飛ばし、本文の欠落や範囲の重複はエラーにする
        if (
            char_start < expected_start
            or char_end <= char_start
            or char_end > len(text)
            or any(
                ord(character) >= 0x20 and character != "\x7f"
                for character in text[expected_start:char_start]
            )
        ):
            raise ValueError(f"{function_name} char_span must cover caller text exactly once")
        expected_start = char_end
    # 文末の制御文字も読み飛ばすため、残った範囲に本文がある場合だけエラーにする
    if any(ord(character) >= 0x20 and character != "\x7f" for character in text[expected_start:]):
        raise ValueError(f"{function_name} char_span must cover caller text exactly once")


def njd_feature_char_spans(
    njd_features: list[NJDFeature],
    morphs: list[MeCabMorph],
) -> list[tuple[int, int]]:
    """
    NJD 形態素へ解析対象文字列上の半開区間を対応付ける。

    Args:
        njd_features (list[NJDFeature]): NJD 処理後の形態素列
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列

    Returns:
        list[tuple[int, int]]: NJD 形態素ごとの文字位置
    """

    feature_char_spans = [(0, 0)] * len(njd_features)
    feature_index = 0
    morph_index = 0
    while feature_index < len(njd_features):
        feature_surface = njd_features[feature_index]["string"]
        # 空白など NJD に渡らない形態素を読み飛ばし、次の実入力位置へ進める
        while (
            morph_index < len(morphs)
            and morphs[morph_index]["is_ignored"] is True
            and feature_surface.startswith(morphs[morph_index]["surface"]) is False
        ):
            morph_index += 1

        # 入力形態素が尽きた NJD ノードは、末尾を指す空区間として記録する
        if morph_index >= len(morphs):
            text_end = morphs[-1]["char_span"][1] if len(morphs) > 0 else 0
            feature_char_spans[feature_index] = (text_end, text_end)
            feature_index += 1
            continue

        morph = morphs[morph_index]

        # 数詞列は位取りの挿入とゼロの吸収を含むため、句読点で区切られたブロックごとに対応させる
        if (
            _is_njd_number_feature(njd_features[feature_index]) is True
            and _is_njd_number_morph(morph) is True
        ):
            feature_end = feature_index
            while (
                feature_end < len(njd_features)
                and _is_njd_number_feature(njd_features[feature_end]) is True
            ):
                feature_end += 1
            next_surface = (
                njd_features[feature_end]["string"] if feature_end < len(njd_features) else ""
            )
            number_morph_indices, morph_end = _collect_njd_number_morphs(
                morphs,
                morph_index,
                [feature["string"] for feature in njd_features[feature_index:feature_end]],
                next_surface,
            )
            number_assignments = _align_njd_number_block(
                [feature["string"] for feature in njd_features[feature_index:feature_end]],
                morphs,
                number_morph_indices,
            )

            # 位取りとして挿入されたノードは直前の入力終端へ空区間を置き、隣の語と区別する
            insertion_position = morph["char_span"][0]
            for offset, assigned_morph_indices in enumerate(number_assignments):
                if len(assigned_morph_indices) == 0:
                    feature_char_spans[feature_index + offset] = (
                        insertion_position,
                        insertion_position,
                    )
                    continue
                span_start = morphs[assigned_morph_indices[0]]["char_span"][0]
                span_end = morphs[assigned_morph_indices[-1]]["char_span"][1]
                feature_char_spans[feature_index + offset] = (span_start, span_end)
                insertion_position = span_end
            feature_index = feature_end
            morph_index = morph_end
            continue

        # 数字と助数詞などが1語へ縮約された場合は、吸収された全入力を同じ文字範囲へまとめる
        compound_morph_start, compound_morph_end = _njd_digit_compound_morph_range(
            morphs,
            morph_index,
            feature_surface,
        )
        if compound_morph_start < morph_index or compound_morph_end > morph_index + 1:
            feature_char_spans[feature_index] = (
                morphs[compound_morph_start]["char_span"][0],
                morphs[compound_morph_end - 1]["char_span"][1],
            )
            morph_index = compound_morph_end
            feature_index += 1
            continue

        # 複数アクセント句は1つの辞書形態素を分割するため、表層内の各文字位置を順に割り当てる
        split_feature_end = feature_index
        concatenated_surface = ""
        while split_feature_end < len(njd_features) and len(concatenated_surface) < len(
            morph["surface"]
        ):
            concatenated_surface += njd_features[split_feature_end]["string"]
            split_feature_end += 1
        if split_feature_end > feature_index + 1 and concatenated_surface == morph["surface"]:
            split_position = morph["char_span"][0]
            for split_feature_index in range(feature_index, split_feature_end):
                split_end = split_position + len(njd_features[split_feature_index]["string"])
                feature_char_spans[split_feature_index] = (split_position, split_end)
                split_position = split_end
            feature_index = split_feature_end
            morph_index += 1
            continue

        # 通常一致、複数形態素の結合、踊り字展開の順に消費数を決める
        if feature_surface == morph["surface"]:
            consumed_morph_count = 1
        elif feature_surface.startswith(morph["surface"]):
            matched_length = 0
            consumed_morph_count = 0
            while morph_index + consumed_morph_count < len(morphs):
                candidate_morph = morphs[morph_index + consumed_morph_count]
                # 「＾＾）　（＾＾）」の内部空白は NJD 表層にも含まれるので、一致する場合は文字数を進める
                if (
                    candidate_morph["is_ignored"] is True
                    and feature_surface[matched_length:].startswith(candidate_morph["surface"])
                    is False
                ):
                    consumed_morph_count += 1
                    continue
                # 一致しなかった形態素は次の素性で使うので、消費せずに抜ける
                if feature_surface[matched_length:].startswith(candidate_morph["surface"]) is False:
                    break
                consumed_morph_count += 1
                matched_length += len(candidate_morph["surface"])
                if matched_length == len(feature_surface):
                    break
        elif any(character in _ODORI_CHARS for character in morph["surface"]):
            consumed_morph_count = 1
            next_morph_index = morph_index + 1
            while next_morph_index < len(morphs) and morphs[next_morph_index]["is_ignored"] is True:
                next_morph_index += 1
            if (
                next_morph_index < len(morphs)
                and feature_surface.endswith(morphs[next_morph_index]["surface"]) is True
            ):
                consumed_morph_count = next_morph_index - morph_index + 1
        else:
            consumed_morph_count = 1

        # 消費した入力の先頭と末尾を結び、NJD の結合後も元の文字範囲を返す
        consumed_morphs = morphs[morph_index : morph_index + consumed_morph_count]
        span_start = morph["char_span"][0]
        span_end = max(
            (consumed_morph["char_span"][1] for consumed_morph in consumed_morphs),
            default=span_start,
        )
        feature_char_spans[feature_index] = (span_start, span_end)
        morph_index += consumed_morph_count
        feature_index += 1

    return feature_char_spans


def build_caller_text_spans_by_mecab_character(
    mecab_text: str,
    reference_text: str,
    normalize_mode: NormalizeMode,
    *,
    inference_jtalk: OpenJTalk,
) -> list[tuple[int, int]]:
    """
    MeCab 正規化本文の各文字に対応する呼び出し元入力上の範囲を返す。

    Args:
        mecab_text (str): morph 表層を連結した MeCab 側文字列
        reference_text (str): `g2p_mapping()` に渡した入力文
        normalize_mode (NormalizeMode): reference_text に適用した Unicode 正規化方式
        inference_jtalk (OpenJTalk): text2mecab と同じ正規化を行うインスタンス

    Returns:
        list[tuple[int, int]]: MeCab 側の各文字に対応する入力文上の半開区間
    """

    source_spans: list[tuple[int, int]] = []
    # 素の入力と、異体字を通用字に変えてローマ数字を算用数字に変えた入力の2通りを MeCab 側の文字列と照合する
    mecab_text_by_source_chunk: dict[str, tuple[str, str]] = {}
    source_start = 0
    mecab_start = 0
    while source_start < len(reference_text):
        matched_end: int | None = None
        matched_text = ""
        maximum_source_end = min(
            source_start + _MAX_CALLER_TEXT_CHUNK_LENGTH,
            len(reference_text),
        )
        # セレクターは直前の文字と一緒に正規化し、結合先がある場合だけ取り除く C 側とそろえる
        while maximum_source_end < len(reference_text) and (
            "\ufe00" <= reference_text[maximum_source_end] <= "\ufe0f"
            or "\U000e0100" <= reference_text[maximum_source_end] <= "\U000e01ef"
        ):
            maximum_source_end += 1
        for source_end in range(source_start + 1, maximum_source_end + 1):
            # 制御文字は text2mecab が読み飛ばすため、直後に続く異体字セレクターも直前の表示文字と一緒に照合する
            selector_start = source_end
            while selector_start < len(reference_text) and (
                ord(reference_text[selector_start]) < 0x20
                or reference_text[selector_start] == "\x7f"
            ):
                selector_start += 1
            if selector_start < len(reference_text) and (
                "\ufe00" <= reference_text[selector_start] <= "\ufe0f"
                or "\U000e0100" <= reference_text[selector_start] <= "\U000e01ef"
            ):
                continue
            source_chunk = reference_text[source_start:source_end]
            candidate_text_pair = mecab_text_by_source_chunk.get(source_chunk)
            if candidate_text_pair is None:
                normalized_chunk = normalize_text(source_chunk, normalize_mode)
                candidate_text_pair = (
                    inference_jtalk.normalize_for_mecab(normalized_chunk),
                    inference_jtalk.normalize_for_mecab(
                        normalize_roman_numerals(normalize_itaiji(normalized_chunk))
                    ),
                )
                mecab_text_by_source_chunk[source_chunk] = candidate_text_pair

            # 制御文字など、正規化時に消える文字列は対応する MeCab 文字を持たない
            if candidate_text_pair[0] == "":
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
        # 「ⅩⅩ」を「２０」にしたように、文字数を変えずに置き換えた塊は1文字ずつ対応させ、数字ごとの範囲が重ならないようにする
        if len(matched_text) == matched_end - source_start:
            source_spans.extend((index, index + 1) for index in range(source_start, matched_end))
        else:
            source_spans.extend([(source_start, matched_end)] * len(matched_text))
        source_start = matched_end
        mecab_start += len(matched_text)

    if mecab_start != len(mecab_text):
        raise ValueError("caller text normalization does not cover MeCab text")
    return source_spans


def _is_njd_number_feature(feature: NJDFeature) -> bool:
    """
    NJD 形態素が数詞列に属するかを返す。

    Args:
        feature (NJDFeature): 判定対象の NJD 形態素

    Returns:
        bool: 品詞細分類が数または副詞可能で、表層が数詞の構成要素だけなら True
    """

    # 「24日」の「二十」は NJD が副詞可能へ変えるため、表層も使って数詞列を追う
    return (
        feature["pos_group1"] in ("数", "副詞可能")
        and _is_njd_number_surface(feature["string"]) is True
    )


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
        # 形態素から分かる tsqyomi とユーザー辞書の出どころは、アライメントの後で _assign_reading_sources() が決める
        reading_source="Rule" if base["is_reading_rewritten"] is True else "Dictionary",
        is_tsqyomi_target=False,
        user_dictionary_path=None,
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
        reading_source="Dictionary",
        is_tsqyomi_target=False,
        user_dictionary_path=None,
    )


def _assign_reading_sources(
    entries: list[SurfacePhonemeMapping],
    morph_ranges: list[tuple[int, int]],
    morphs: list[MeCabMorph],
    userdic_paths: tuple[str, ...],
) -> None:
    """
    アライメントした形態素から、各エントリの読みの出どころとユーザー辞書のパスを決める。
    後処理の規則が書き換えたエントリ ("Rule") はそのままにし、残りは tsqyomi の判定、ユーザー辞書の行、既定辞書の順に当てる。

    Args:
        entries (list[SurfacePhonemeMapping]): アライメント済みのマッピング。インプレースで更新される
        morph_ranges (list[tuple[int, int]]): entries と同じ順の、対応する morph の添字の半開区間
        morphs (list[MeCabMorph]): MeCab の形態素解析結果
        userdic_paths (tuple[str, ...]): ユーザー辞書ファイルのパス (読み込んだ順)
    """

    for entry, (morph_start, morph_end) in zip(entries, morph_ranges, strict=True):
        aligned_morphs = [
            morph for morph in morphs[morph_start:morph_end] if morph["is_ignored"] is False
        ]
        entry["is_tsqyomi_target"] = any(
            morph["is_tsqyomi_target"] is True for morph in aligned_morphs
        )
        # 結合された形態素では、重なる入力のいずれかがユーザー辞書の行なら、最初の行の辞書を返す
        user_dictionary_indices = [
            morph["dictionary_index"]
            for morph in aligned_morphs
            if 1 <= morph["dictionary_index"] <= len(userdic_paths)
        ]
        if len(user_dictionary_indices) > 0:
            entry["user_dictionary_path"] = userdic_paths[user_dictionary_indices[0] - 1]
        if entry["reading_source"] == "Rule":
            continue
        if any(morph["is_tsqyomi_selected"] is True for morph in aligned_morphs):
            entry["reading_source"] = "Tsqyomi"
        elif len(user_dictionary_indices) > 0:
            entry["reading_source"] = "UserDictionary"


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
    previous_mecab_end = previous_caller_end = 0
    for index, entry in enumerate(entries):
        mecab_start, mecab_end = resolved_mecab_char_spans[index]
        caller_start, caller_end = _project_char_span(
            caller_text_spans,
            resolved_mecab_char_spans[index],
        )
        # NFKC で「㎞」を「ｋ」「ｍ」に分けた場合、元の1文字は先の要素だけへ割り当てる
        ## 正規化本文上では別の範囲で、境目の両文字が同じ入力範囲へ展開された場合だけ重複を除く
        if (
            mecab_start >= previous_mecab_end > 0
            and mecab_start < mecab_end
            and caller_start < previous_caller_end
            and caller_text_spans[mecab_start] == caller_text_spans[previous_mecab_end - 1]
        ):
            caller_start = previous_caller_end
            if caller_start >= caller_end:
                caller_start = caller_end = 0
        entry["char_span"] = (caller_start, caller_end)
        if mecab_start < mecab_end:
            previous_mecab_end = mecab_end
            if caller_end > 0:
                previous_caller_end = caller_end
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
    NJD 表層が数字展開の構成要素と、吸収された末尾の長音だけであるかを返す。

    Args:
        surface (str): 判定対象の表層形

    Returns:
        bool: 漢数字または算用数字の列 (末尾の長音を含む) なら True
    """

    number_surface = surface.rstrip("ー")
    return number_surface != "" and all(
        character in _KANJI_NUMBER_SURFACES or character in _DIGIT_MORPH_SURFACES
        for character in number_surface
    )


def _is_njd_number_morph(morph: MeCabMorph, *, is_group_separator_allowed: bool = False) -> bool:
    """
    NJD が数詞列として変換する MeCab 形態素かを返す。

    Args:
        morph (MeCabMorph): 判定対象の MeCab 形態素
        is_group_separator_allowed (bool): NJD が吸収した桁区切りのカンマも数詞列に含めるか

    Returns:
        bool: 品詞が数で、変換する数字または吸収した桁区切りのカンマなら True
    """

    return (
        len(morph["features"]) > 2
        and (
            morph["features"][2] == "数"
            # 単独の「〇」は MeCab が記号として返しても、NJD が隣の数字とまとめて数へ変える
            or (morph["surface"] in ("〇", "○") and morph["features"][1] == "記号")
        )
        and (
            morph["surface"] in _NJD_NUMBER_MORPH_SURFACE_KEYS
            or _is_njd_number_surface(morph["surface"]) is True
            or (is_group_separator_allowed is True and morph["surface"] in (",", "，"))
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


def _collect_njd_number_morphs(
    morphs: list[MeCabMorph],
    morph_start: int,
    feature_surfaces: list[str],
    next_surface: str,
) -> tuple[list[int], int]:
    """
    数詞列が消費する入力形態素を、後続の日付や吸収された長音も考慮して集める。
    音素マッピングと NJD の文字位置で同じ境界を使い、顔文字と読み保護へも正しい入力位置を渡す。

    Args:
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        morph_start (int): 数詞列の入力側の開始添字
        feature_surfaces (list[str]): 連続する出力側の数詞表層
        next_surface (str): 数詞列の直後の表層 (末尾の場合は空文字列)

    Returns:
        tuple[list[int], int]: 数詞へ割り当てる形態素添字と、消費範囲の終端
    """

    number_morph_indices: list[int] = []
    number_block_end_morph_idx = morph_start
    last_number_morph_end_idx = morph_start
    is_stopped_before_spaced_compound = False
    # 「1,050円」で NJD が吸収したカンマは数字と同じ範囲へ対応させ、「1,2」の残るカンマは区切りとして保つ
    is_group_separator_allowed = next_surface not in (",", "，")
    while number_block_end_morph_idx < len(morphs):
        number_morph = morphs[number_block_end_morph_idx]
        if number_morph["is_ignored"] is True:
            # 「1 10日」の日付の桁は前の数へ含めず、空白から次の縮約語へ引き渡す
            if next_surface != "":
                following = number_block_end_morph_idx + 1
                while following < len(morphs) and morphs[following]["is_ignored"] is True:
                    following += 1
                compound_surface = next_surface
                compound_range = _njd_digit_compound_morph_range(
                    morphs, following, compound_surface
                )
                if compound_range[1] > following + 1:
                    leading_length = _njd_number_leading_length(compound_surface)
                    leading_key = _njd_number_alignment_key(compound_surface[:leading_length])
                    if compound_surface[leading_length:].startswith("日") is True:
                        leading_key = {"十": "一〇", "二十": "二〇"}.get(leading_key, leading_key)
                    number_key = ""
                    for following_morph in morphs[following : compound_range[1]]:
                        if _is_njd_number_morph(following_morph) is False:
                            break
                        number_key += _njd_number_alignment_key(following_morph["surface"])
                    # 「24日」の先頭の2は「二十」へ残し、次の「四日」へまとめて渡さない
                    if number_key.lstrip("〇") == leading_key:
                        is_stopped_before_spaced_compound = True
                        break
            number_block_end_morph_idx += 1
            continue
        if (
            _is_njd_number_morph(
                number_morph, is_group_separator_allowed=is_group_separator_allowed
            )
            is False
        ):
            break
        number_morph_indices.append(number_block_end_morph_idx)
        number_block_end_morph_idx += 1
        last_number_morph_end_idx = number_block_end_morph_idx

    # 24日 の「四日」のように次ノードが末尾数字を吸収する場合、その数字は純数詞ブロックへ渡さない
    ## NJD は「二十」「四日」と分割するため、ここで 2 と 4 の両方を「二十」へ割り当てると
    ## 後段の「四日」が 4 を再消費し、座標が重複する
    reserved_number_morph_count = 0
    if next_surface != "" and is_stopped_before_spaced_compound is False:
        next_number_length = _njd_number_leading_length(next_surface)
        # 「４　四つ」「３三振」のように、次ノードの語が数字の後の形態素としてそのまま入力にあるときは、数字を吸収していないので取り分けない
        next_morph_surface = next(
            (
                morphs[index]["surface"]
                for index in range(last_number_morph_end_idx, len(morphs))
                if morphs[index]["is_ignored"] is False
            ),
            "",
        )
        if 0 < next_number_length < len(next_surface.rstrip("ー")) and (
            next_morph_surface.startswith(next_surface) is False
        ):
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
    # 「千ー」のように JPCommon が長音を数詞へ結合した場合、その入力も数詞の範囲へ含める
    long_vowel_suffix = feature_surfaces[-1]
    long_vowel_suffix = long_vowel_suffix[len(long_vowel_suffix.rstrip("ー")) :]
    if long_vowel_suffix != "":
        long_vowel_end = number_block_end_morph_idx
        remaining_long_vowels = long_vowel_suffix
        while long_vowel_end < len(morphs) and remaining_long_vowels != "":
            long_morph = morphs[long_vowel_end]
            if long_morph["is_ignored"] is True:
                long_vowel_end += 1
                continue
            if remaining_long_vowels.startswith(long_morph["surface"]) is False:
                break
            remaining_long_vowels = remaining_long_vowels[len(long_morph["surface"]) :]
            long_vowel_end += 1
        if remaining_long_vowels == "":
            number_morph_indices.extend(
                index
                for index in range(number_block_end_morph_idx, long_vowel_end)
                if morphs[index]["is_ignored"] is False
            )
            number_block_end_morph_idx = long_vowel_end
    return number_morph_indices, number_block_end_morph_idx


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

    # 「09月」の0を日付処理が除いた場合、辞書の1語の「９月」に先頭の0も対応させる
    if (
        morph_index < len(morphs)
        and morphs[morph_index]["surface"] == "０"
        and feature_surface.endswith("月")
    ):
        following = morph_index + 1
        while following < len(morphs) and morphs[following]["is_ignored"]:
            following += 1
        if following < len(morphs) and morphs[following]["surface"] == feature_surface:
            return (morph_index, following + 1)

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
