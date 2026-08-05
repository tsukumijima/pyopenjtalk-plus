# coding: utf-8
# cython: boundscheck=True, wraparound=True
# cython: c_string_type=unicode, c_string_encoding=ascii
# cython: language_level=3
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingTypeArgument=false
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportWildcardImportFromLibrary=false

import numpy as np
from collections.abc import Callable, Sequence
from functools import wraps
from threading import Lock
from typing import Any, Concatenate, Iterable, ParamSpec, TypeVar

from .types import (
    JPCommonMappingEntry,
    MeCabLatticeCandidate,
    MeCabMorph,
    MeCabNBestPath,
    NJDFeature,
)
from .tsqyomi.types import (
    CandidateConnection,
    CandidateNode,
    CandidatePath,
    ReadingAnalysis,
)

cimport numpy as np
np.import_array()

from ._known_symbols import KNOWN_SYMBOL_FEATURES

from libc.limits cimport LONG_MAX
from libc.stdlib cimport calloc
from libc.string cimport strlen
from libc.stdint cimport *

from .openjtalk.mecab cimport Mecab, Mecab_initialize, Mecab_load, Mecab_analysis
from .openjtalk.mecab cimport Mecab_get_feature, Mecab_get_size, Mecab_refresh, Mecab_clear
from .openjtalk.mecab cimport createModel, Model, Tagger, Lattice
from .openjtalk.mecab cimport mecab_dict_index as _mecab_dict_index
from .openjtalk.mecab cimport (
    mecab_node_t,
    mecab_path_t,
    mecab_t,
    mecab_lattice_t,
    mecab_lattice_get_bos_node,
    mecab_lattice_get_begin_nodes,
    mecab_lattice_get_request_type,
    mecab_lattice_get_sentence,
    mecab_lattice_get_size,
    mecab_lattice_set_request_type,
    mecab_lattice_set_sentence,
    mecab_parse_lattice,
    mecab_nbest_init2,
    mecab_nbest_next_tonode,
    MECAB_NBEST,
)
from .openjtalk.njd cimport NJD, NJD_initialize, NJD_refresh, NJD_clear
from .openjtalk cimport njd as _njd
from .openjtalk.jpcommon cimport JPCommon, JPCommon_initialize, JPCommon_make_label
from .openjtalk.jpcommon cimport JPCommon_get_label_size, JPCommon_get_label_feature
from .openjtalk.jpcommon cimport JPCommon_refresh, JPCommon_clear
from .openjtalk.jpcommon cimport JPCommonLabel, JPCommonLabelWord, JPCommonLabelMora, JPCommonLabelPhoneme
from .openjtalk.jpcommon cimport JPCommonLabel_initialize, JPCommonLabel_push_word
from .openjtalk.jpcommon cimport JPCommonLabel_clear
from .openjtalk.jpcommon cimport JPCommonNode
from .openjtalk.jpcommon cimport (
    JPCommonNode_get_pron,
    JPCommonNode_get_pos,
    JPCommonNode_get_ctype,
    JPCommonNode_get_cform,
    JPCommonNode_get_acc,
    JPCommonNode_get_chain_flag,
)
from .openjtalk.text2mecab cimport (
    TEXT2MECAB_RESULT_INVALID_ARGUMENT,
    TEXT2MECAB_RESULT_RANGE_ERROR,
    text2mecab,
)
from .openjtalk.mecab2njd cimport mecab2njd
from .openjtalk.njd2jpcommon cimport njd2jpcommon

DEF TEXT2MECAB_BUFFER_SIZE = 16384

_NON_PAUSE_SYMBOLS = frozenset((
    "「", "」", "『", "』", "（", "）", "(", ")",
    "【", "】", "［", "］", "[", "]", "〈", "〉",
    "《", "》", "〔", "〕", "｛", "｝", "{", "}",
    "\"", "'", "”", "“", "’", "‘",
))

cdef inline str _decode_utf8_or_empty(const char* value):
    """
    C 文字列ポインタを UTF-8 の Python str へデコードする。

    Args:
        value (const char*): null 終端 C 文字列。NULL の場合は空文字列として扱う

    Returns:
        str: デコード結果
    """
    if value == NULL:
        return ""
    return (<bytes>value).decode("utf-8")

cdef inline bytes _validate_and_encode_njd_field(feature_node, str field_name) except *:
    """
    NJDFeature dict の文字列フィールドを検証し UTF-8 bytes へエンコードする。

    Args:
        feature_node: NJDFeature 相当 dict
        field_name (str): 読み取るフィールド名

    Returns:
        bytes: UTF-8 エンコード済みフィールド値

    Raises:
        TypeError: フィールドが str でない場合
        ValueError: null 文字が含まれる場合
    """
    cdef object field_value
    cdef bytes encoded_value

    field_value = feature_node[field_name]
    if isinstance(field_value, str) is False:
        raise TypeError(f"NJD feature field must be str: {field_name}")
    if "\x00" in field_value:
        raise ValueError(f"NJD feature field contains null character: {field_name}")
    encoded_value = field_value.encode("utf-8")
    return encoded_value

cdef inline object _validate_int_njd_field(feature_node, str field_name) except *:
    """
    NJDFeature dict の整数フィールドを検証する。

    Args:
        feature_node: NJDFeature 相当 dict
        field_name (str): 読み取るフィールド名

    Returns:
        object: 検証済み int 値 (Python オブジェクト)

    Raises:
        TypeError: フィールドが bool または int 以外の場合
    """
    cdef object field_value

    field_value = feature_node[field_name]
    if isinstance(field_value, bool) is True or isinstance(field_value, int) is False:
        raise TypeError(f"NJD feature field must be int: {field_name}")
    return field_value

cdef njd_node_get_string(_njd.NJDNode* node):
    """NJDNode の表層形 string を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_string(node))

cdef njd_node_get_pos(_njd.NJDNode* node):
    """NJDNode の品詞 pos を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos(node))

cdef njd_node_get_pos_group1(_njd.NJDNode* node):
    """NJDNode の pos_group1 を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group1(node))

cdef njd_node_get_pos_group2(_njd.NJDNode* node):
    """NJDNode の pos_group2 を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group2(node))

cdef njd_node_get_pos_group3(_njd.NJDNode* node):
    """NJDNode の pos_group3 を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group3(node))

cdef njd_node_get_ctype(_njd.NJDNode* node):
    """NJDNode の ctype を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_ctype(node))

cdef njd_node_get_cform(_njd.NJDNode* node):
    """NJDNode の cform を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_cform(node))

cdef njd_node_get_orig(_njd.NJDNode* node):
    """NJDNode の orig を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_orig(node))

cdef njd_node_get_read(_njd.NJDNode* node):
    """NJDNode の read を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_read(node))

cdef njd_node_get_pron(_njd.NJDNode* node):
    """NJDNode の pron を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_pron(node))

cdef int njd_node_get_acc(_njd.NJDNode* node) noexcept:
    """NJDNode の acc (アクセント核位置) を返す。"""
    return _njd.NJDNode_get_acc(node)

cdef int njd_node_get_mora_size(_njd.NJDNode* node) noexcept:
    """NJDNode の mora_size (モーラ数) を返す。"""
    return _njd.NJDNode_get_mora_size(node)

cdef njd_node_get_chain_rule(_njd.NJDNode* node):
    """NJDNode の chain_rule を UTF-8 str として返す。"""
    return _decode_utf8_or_empty(_njd.NJDNode_get_chain_rule(node))

cdef int njd_node_get_chain_flag(_njd.NJDNode* node) noexcept:
    """NJDNode の chain_flag (アクセント句連結フラグ) を返す。"""
    return _njd.NJDNode_get_chain_flag(node)


cdef node2feature(_njd.NJDNode* node):
    """
    NJD 連結リストの1ノードを NJDFeature 相当の dict へ変換する。

    Args:
        node (_njd.NJDNode*): 読み取る NJD ノード

    Returns:
        NJDFeature: `string` / `pos` / `pron` / `acc` 等を含む NJDFeature
    """
    return NJDFeature(
        string=njd_node_get_string(node),
        pos=njd_node_get_pos(node),
        pos_group1=njd_node_get_pos_group1(node),
        pos_group2=njd_node_get_pos_group2(node),
        pos_group3=njd_node_get_pos_group3(node),
        ctype=njd_node_get_ctype(node),
        cform=njd_node_get_cform(node),
        orig=njd_node_get_orig(node),
        read=njd_node_get_read(node),
        pron=njd_node_get_pron(node),
        acc=njd_node_get_acc(node),
        mora_size=njd_node_get_mora_size(node),
        chain_rule=njd_node_get_chain_rule(node),
        chain_flag=njd_node_get_chain_flag(node),
    )


cdef njd2feature(_njd.NJD* njd):
    """
    NJD 連結リスト全体を NJDFeature dict の list へ変換する。

    Args:
        njd (_njd.NJD*): 走査する NJD 構造体

    Returns:
        list[NJDFeature]: 先頭から末尾までの NJDFeature の list
    """
    cdef _njd.NJDNode* node = njd.head
    features = []
    while node is not NULL:
        features.append(node2feature(node))
        node = node.next
    return features


cdef void feature2njd(_njd.NJD* njd, features) except *:
    """
    NJDFeature dict の list から NJD 連結リストを再構築する。

    Args:
        njd (_njd.NJD*): 書き込み先 NJD 構造体 (呼び出し前に `NJD_refresh()` 済みであること)
        features (list): NJDFeature 相当 dict の list

    Raises:
        TypeError: 文字列フィールドが str でない、または整数フィールドが int でない場合
        ValueError: 文字列フィールドに null 文字が含まれる場合
        MemoryError: NJD ノードの確保に失敗した場合

    NOTE:
        例外時は `NJD_refresh(njd)` で部分構築済みノードを解放する。
    """
    cdef _njd.NJDNode* node
    cdef bytes string_bytes
    cdef bytes pos_bytes
    cdef bytes pos_group1_bytes
    cdef bytes pos_group2_bytes
    cdef bytes pos_group3_bytes
    cdef bytes ctype_bytes
    cdef bytes cform_bytes
    cdef bytes orig_bytes
    cdef bytes read_bytes
    cdef bytes pron_bytes
    cdef bytes chain_rule_bytes
    cdef int acc_value
    cdef int mora_size_value
    cdef int chain_flag_value

    try:
        for feature_node in features:
            string_bytes = _validate_and_encode_njd_field(feature_node, "string")
            pos_bytes = _validate_and_encode_njd_field(feature_node, "pos")
            pos_group1_bytes = _validate_and_encode_njd_field(feature_node, "pos_group1")
            pos_group2_bytes = _validate_and_encode_njd_field(feature_node, "pos_group2")
            pos_group3_bytes = _validate_and_encode_njd_field(feature_node, "pos_group3")
            ctype_bytes = _validate_and_encode_njd_field(feature_node, "ctype")
            cform_bytes = _validate_and_encode_njd_field(feature_node, "cform")
            orig_bytes = _validate_and_encode_njd_field(feature_node, "orig")
            read_bytes = _validate_and_encode_njd_field(feature_node, "read")
            pron_bytes = _validate_and_encode_njd_field(feature_node, "pron")
            chain_rule_bytes = _validate_and_encode_njd_field(feature_node, "chain_rule")
            acc_value = <int> _validate_int_njd_field(feature_node, "acc")
            mora_size_value = <int> _validate_int_njd_field(feature_node, "mora_size")
            chain_flag_value = <int> _validate_int_njd_field(feature_node, "chain_flag")

            node = <_njd.NJDNode *> calloc(1, sizeof(_njd.NJDNode))
            if node == NULL:
                raise MemoryError("Failed to allocate memory for NJD node")
            _njd.NJDNode_initialize(node)
            # set values
            _njd.NJDNode_set_string(node, string_bytes)
            _njd.NJDNode_set_pos(node, pos_bytes)
            _njd.NJDNode_set_pos_group1(node, pos_group1_bytes)
            _njd.NJDNode_set_pos_group2(node, pos_group2_bytes)
            _njd.NJDNode_set_pos_group3(node, pos_group3_bytes)
            _njd.NJDNode_set_ctype(node, ctype_bytes)
            _njd.NJDNode_set_cform(node, cform_bytes)
            _njd.NJDNode_set_orig(node, orig_bytes)
            _njd.NJDNode_set_read(node, read_bytes)
            _njd.NJDNode_set_pron(node, pron_bytes)
            _njd.NJDNode_set_acc(node, acc_value)
            _njd.NJDNode_set_mora_size(node, mora_size_value)
            _njd.NJDNode_set_chain_rule(node, chain_rule_bytes)
            _njd.NJDNode_set_chain_flag(node, chain_flag_value)
            _njd.NJD_push_node(njd, node)
    except Exception:
        NJD_refresh(njd)
        raise


cdef long _selected_mecab_link_cost(mecab_node_t* node, bint can_use_node_cost_fallback) except *:
    """
    最良経路または n-best 候補パス上の MeCab ノードについて、直前ノードとの接続コストを返す。

    Args:
        node (mecab_node_t*): 接続コストを求める lattice ノード
        can_use_node_cost_fallback (bint): one-best 解析向けに累積コスト差から復元してよいか

    Returns:
        long: 直前ノードとの接続コスト

    Raises:
        RuntimeError: n-best 候補で対応する接続辺が見つからない場合
    """

    cdef mecab_path_t* path

    # n-best 生成後の node.cost は MeCab 側で再計算されないため、候補パスの prev リンクから局所コストを復元する
    ## Path.cost は MeCab の connector->cost() 由来で、直前ノードとの連接コストと現在ノードの単語コストを含む
    if node == NULL or node.prev == NULL:
        return 0

    path = node.lpath
    while path != NULL:
        if path.lnode == node.prev and path.rnode == node:
            return path.cost
        path = path.lnext

    # one-best 解析では lpath が残らない場合があるため、累積コスト差から局所コストを復元する
    ## n-best の node.cost は候補パス向けに再計算されないので、このフォールバックは one-best 用に限定する
    if can_use_node_cost_fallback is True:
        return node.cost - node.prev.cost

    # n-best で対応する接続が見つからない場合は、候補パスの内部リンクが壊れている
    ## 0 として扱うと path_cost が過小評価されるため、呼び出し側に明示的に失敗を返す
    raise RuntimeError("Failed to resolve selected MeCab link cost")


cdef list _build_byte_to_char_offsets(bytes sentence_bytes):
    """
    MeCab 文の UTF-8 バイト位置を Python 文字位置へ変換する対応表を構築する。

    Args:
        sentence_bytes (bytes): MeCab lattice の sentence バッファ

    Returns:
        list: 長さ `len(sentence_bytes) + 1` の対応表。インデックス i はバイト i の直前までの文字数
    """

    cdef Py_ssize_t byte_index
    cdef Py_ssize_t character_index = 0
    cdef unsigned char current_byte
    cdef list byte_to_char_offsets = [0] * (len(sentence_bytes) + 1)

    # MeCab の位置は UTF-8 のバイト単位なので、文ごとに Python の文字位置との対応表を作る
    ## 各ノードで先頭から decode すると候補数に応じて同じ接頭辞を繰り返し走査することになる
    for byte_index in range(len(sentence_bytes)):
        current_byte = sentence_bytes[byte_index]
        if (current_byte & 0xC0) != 0x80:
            character_index += 1
        byte_to_char_offsets[byte_index + 1] = character_index
    return byte_to_char_offsets


cdef tuple _mecab_node_to_common_fields(
    mecab_node_t* node,
    const char* sentence,
    list byte_to_char_offsets,
) except *:
    """
    MeCab ノードから surface・feature・文字位置などの共通フィールドを抽出する。

    Args:
        node (mecab_node_t*): 読み取る lattice ノード
        sentence (const char*): MeCab が解析した sentence バッファ
        byte_to_char_offsets (list): `_build_byte_to_char_offsets()` が構築したバイト→文字対応表

    Returns:
        tuple: `(surface_str, feature_columns, is_unknown, is_ignored, char_span)`
    """

    cdef Py_ssize_t byte_begin
    cdef Py_ssize_t byte_end
    cdef uintptr_t byte_offset
    cdef bytes surface_bytes
    cdef bytes feature_bytes
    cdef str surface_str
    cdef str feature_str
    cdef list feature_columns
    cdef bint is_unknown
    cdef bint is_ignored
    cdef object char_span

    # surface は null 終端ではないため、MeCab が示すバイト長だけを UTF-8 として読む
    ## c_string_encoding=ascii の影響を受けると非 ASCII 文字が壊れるため、明示的に bytes 化する
    if node.surface != NULL and node.length > 0:
        surface_bytes = (<char*>node.surface)[:node.length]
        surface_str = surface_bytes.decode("utf-8", errors="replace")
    else:
        surface_str = ""

    # feature は MeCab 側で null 終端済みの文字列として保持される
    ## 未知語や異常系でも Python 側の検証コードが扱えるよう、NULL は空文字列として返す
    if node.feature != NULL:
        feature_bytes = <bytes> node.feature
        feature_str = feature_bytes.decode("utf-8", errors="replace")
    else:
        feature_str = ""

    feature_columns = (surface_str + "," + feature_str).split(",")
    is_unknown = node.stat == 1  # MECAB_UNK_NODE
    is_ignored = "記号,空白" in feature_str

    # sentence と同じバッファを指すノードだけ、Python の半開区間へ変換する
    ## n-best や外部確保されたノードの surface は sentence の範囲外を指す場合がある
    if (
        sentence != NULL
        and node.surface != NULL
        and <uintptr_t> node.surface >= <uintptr_t> sentence
    ):
        byte_offset = <uintptr_t> node.surface - <uintptr_t> sentence
        if (
            byte_offset < <uintptr_t> len(byte_to_char_offsets)
            and node.length < <uintptr_t> len(byte_to_char_offsets) - byte_offset
        ):
            byte_begin = <Py_ssize_t> byte_offset
            byte_end = byte_begin + node.length
            char_span = (
                byte_to_char_offsets[byte_begin],
                byte_to_char_offsets[byte_end],
            )
        else:
            char_span = (0, 0)
    else:
        char_span = (0, 0)

    return surface_str, feature_columns, is_unknown, is_ignored, char_span


cdef object _mecab_node_to_morph(
    mecab_node_t* node,
    bint can_use_node_cost_fallback,
    const char* sentence,
    list byte_to_char_offsets,
) except *:
    """
    MeCab ノードを `run_mecab_detailed()` 互換の形態素 dict へ変換する。

    Args:
        node (mecab_node_t*): 読み取る lattice ノード
        can_use_node_cost_fallback (bint): 接続コスト復元に one-best 向けフォールバックを許可するか
        sentence (const char*): MeCab が解析した sentence バッファ
        byte_to_char_offsets (list): `_build_byte_to_char_offsets()` が構築したバイト→文字対応表

    Returns:
        MeCabMorph: 詳細形態素 API 向けの surface・feature・コスト・文字位置情報
    """

    cdef long link_cost
    cdef str surface_str
    cdef list feature_columns
    cdef bint is_unknown
    cdef bint is_ignored
    cdef object char_span

    link_cost = _selected_mecab_link_cost(node, can_use_node_cost_fallback)
    surface_str, feature_columns, is_unknown, is_ignored, char_span = (
        _mecab_node_to_common_fields(node, sentence, byte_to_char_offsets)
    )
    return MeCabMorph(
        surface=surface_str,
        features=feature_columns,
        pos_id=node.posid,
        left_id=node.lcAttr,
        right_id=node.rcAttr,
        word_cost=node.wcost,
        link_cost=link_cost,
        node_cost=node.cost,
        char_span=char_span,
        is_unknown=is_unknown,
        is_ignored=is_ignored,
        dictionary_index=node.dictionary_index,
    )


cdef list _expand_symbol_morphs(
    mecab_node_t* node,
    object node_morph,
) except *:
    """
    未知語連結記号を1文字ずつ既知記号へ復元した MeCabMorph 列を返す。
    分割不要なら `node_morph` 1件だけを返す。
    """

    cdef list expanded_morphs = []
    cdef str surface_str
    cdef str morph_feature_str
    cdef bint is_unknown
    cdef bint should_split_symbol_chunk
    cdef Py_ssize_t character_index
    cdef str character
    cdef object known_symbol
    cdef int split_left_id
    cdef int split_right_id
    cdef long split_word_cost
    cdef str split_feature
    cdef long split_link_cost
    cdef int split_char_start

    surface_str = node_morph["surface"]
    morph_feature_str = ",".join(node_morph["features"][1:])
    is_unknown = node_morph["is_unknown"]
    should_split_symbol_chunk = (
        is_unknown is True
        and len(surface_str) > 1
        and all(character.isalnum() is False for character in surface_str)
    )
    if should_split_symbol_chunk is False:
        expanded_morphs.append(node_morph)
        return expanded_morphs

    for character_index, character in enumerate(surface_str):
        known_symbol = KNOWN_SYMBOL_FEATURES.get(character)
        if known_symbol is not None:
            split_left_id = known_symbol[0]
            split_right_id = known_symbol[1]
            split_word_cost = known_symbol[2]
            split_feature = known_symbol[3]
        else:
            split_feature = morph_feature_str
            split_word_cost = node.wcost
            split_left_id = node.lcAttr
            split_right_id = node.rcAttr

        split_link_cost = node_morph["link_cost"] if character_index == 0 else 0
        split_char_start = node_morph["char_span"][0] + character_index
        expanded_morphs.append(MeCabMorph(
            surface=character,
            features=(character + "," + split_feature).split(","),
            pos_id=node.posid,
            left_id=split_left_id,
            right_id=split_right_id,
            word_cost=split_word_cost,
            link_cost=split_link_cost,
            node_cost=node_morph["node_cost"],
            char_span=(split_char_start, split_char_start + 1),
            is_unknown=known_symbol is None,
            is_ignored="記号,空白" in split_feature,
            dictionary_index=0 if known_symbol is not None else node_morph["dictionary_index"],
        ))
    return expanded_morphs


cdef object _mecab_node_to_cost_candidate(
    mecab_node_t* node,
    const char* sentence,
    list byte_to_char_offsets,
    tuple userdic_reading_protection,
) except *:
    """
    tsqyomi 候補解析向けに MeCab ノードを辞書候補 dict へ変換する。

    Args:
        node (mecab_node_t*): 読み取る lattice ノード
        sentence (const char*): MeCab が解析した sentence バッファ
        byte_to_char_offsets (list): `_build_byte_to_char_offsets()` が構築したバイト→文字対応表
        userdic_reading_protection (tuple): ユーザー辞書ごとの読み保護フラグ

    Returns:
        MeCabLatticeCandidate: 候補ノードの surface・feature・文字位置・保護状態・MeCab コスト情報
    """

    cdef str surface_str
    cdef list feature_columns
    cdef bint is_unknown
    cdef bint is_ignored
    cdef bint is_reading_protected
    cdef object char_span

    surface_str, feature_columns, is_unknown, is_ignored, char_span = (
        _mecab_node_to_common_fields(node, sentence, byte_to_char_offsets)
    )
    # システム辞書と未知語は保護対象外とし、ユーザー辞書だけを読み込み順のフラグへ対応させる
    is_reading_protected = (
        node.dictionary_index >= 1
        and node.dictionary_index <= len(userdic_reading_protection)
        and userdic_reading_protection[node.dictionary_index - 1] is True
    )

    return MeCabLatticeCandidate(
        surface=surface_str,
        features=feature_columns,
        char_span=char_span,
        pos_id=node.posid,
        left_id=node.lcAttr,
        right_id=node.rcAttr,
        word_cost=node.wcost,
        is_unknown=is_unknown,
        is_ignored=is_ignored,
        is_reading_protected=is_reading_protected,
        dictionary_index=node.dictionary_index,
        local_replacement_cost=None,
        left_boundary_cost=None,
        right_boundary_cost=None,
    )


# based on Mecab_load in impl. from mecab.cpp
cdef inline int Mecab_load_with_userdic(Mecab *m, char* dicdir, char* userdic) noexcept nogil:
    """
    システム辞書とユーザー辞書 (カンマ区切り複数可) を読み込み、MeCab Model / Tagger / Lattice を初期化する。

    Args:
        m (Mecab*): 初期化対象の OpenJTalk MeCab ラッパ
        dicdir (char*): システム辞書ディレクトリ
        userdic (char*): ユーザー辞書パス。空文字列なら `Mecab_load()` のみ実行

    Returns:
        int: 成功時 1、失敗時 0
    """
    if userdic == NULL or strlen(userdic) == 0:
        return Mecab_load(m, dicdir)

    if m == NULL or dicdir == NULL or strlen(dicdir) == 0:
        return 0

    Mecab_clear(m)

    cdef char* argv[5]
    argv[0] = "mecab"
    argv[1] = "-d"
    argv[2] = dicdir
    argv[3] = "-u"
    argv[4] = userdic
    cdef Model *model = createModel(5, argv)

    if model == NULL:
        return 0
    m.model = model

    cdef Tagger *tagger = model.createTagger()
    if tagger == NULL:
        Mecab_clear(m)
        return 0
    m.tagger = tagger

    cdef Lattice *lattice = model.createLattice()
    if lattice == NULL:
        Mecab_clear(m)
        return 0
    m.lattice = lattice

    return 1

P = ParamSpec("P")
R = TypeVar("R")
Self = TypeVar("Self")


def _lock_manager() -> Callable[[Callable[Concatenate[Self, P], R]], Callable[Concatenate[Self, P], R]]:
    """
    OpenJTalk インスタンスの公開メソッドを直列化するデコレータを返す。

    Returns:
        Callable: `OpenJTalk` の公開メソッドをラップし、`self._lock` で排他するデコレータ

    NOTE:
        `Mecab` / `NJD` / `JPCommon` はインスタンス内で共有されるため、同一インスタンスへの同時呼び出しは安全でない。
        ロックはインスタンスごとに分離され、別インスタンスの `nogil` 区間は並行実行できる。
    """

    def decorator(method: Callable[Concatenate[Self, P], R]) -> Callable[Concatenate[Self, P], R]:
        @wraps(method)
        def wrapped(self: Self, *args: P.args, **kwargs: P.kwargs) -> R:
            with self._lock:
                return method(self, *args, **kwargs)

        return wrapped

    return decorator


cdef class OpenJTalk:
    """
    OpenJTalk のテキスト処理フロントエンドの Cython 実装。
    通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス
        userdic (bytes): OpenJTalk 用ユーザー辞書 (.dic) のパス。空バイト列の場合は無視される。複数指定時はカンマ区切り。デフォルト: 空
        userdic_reading_protection (Sequence[bool] | None): 各ユーザー辞書の読み候補を tsqyomi による MeCab feature 差し替えから保護するか
            None の場合は全辞書を未保護として扱う。デフォルト: None

    Raises:
        ValueError: `userdic_reading_protection` の要素数が辞書数と一致しない場合
        TypeError: 保護フラグに bool 以外が含まれる場合
        RuntimeError: MeCab 初期化に失敗した場合

    NOTE:
        公開メソッドは `@_lock_manager()` で直列化される。`Mecab` / `NJD` / `JPCommon` はインスタンス内で共有される。
        `Mecab_refresh()` は Python 側の `try/finally` から呼び出し、lattice ノード走査後に MeCab 内部状態を解放する。
    """
    cdef Mecab* mecab
    cdef NJD* njd
    cdef JPCommon* jpcommon
    cdef tuple userdic_reading_protection
    cdef readonly object _lock

    def __cinit__(
        self,
        dn_mecab: bytes = b"/usr/local/dic",
        userdic: bytes = b"",
        userdic_reading_protection: Sequence[bool] | None = None,
    ):
        """
        OpenJTalk のテキスト処理フロントエンドの Cython 実装。
        通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

        Args:
            dn_mecab (bytes): MeCab システム辞書のディレクトリパス
            userdic (bytes): OpenJTalk 用ユーザー辞書 (.dic) のパス。空バイト列の場合は無視される。複数指定時はカンマ区切り。デフォルト: 空
            userdic_reading_protection (Sequence[bool] | None): 各ユーザー辞書の読み候補を tsqyomi による MeCab feature 差し替えから保護するか
                None の場合は全辞書を未保護として扱う。デフォルト: None

        Raises:
            ValueError: `userdic_reading_protection` の要素数が辞書数と一致しない場合
            TypeError: 保護フラグに bool 以外が含まれる場合
            RuntimeError: MeCab 初期化に失敗した場合

        NOTE:
            公開メソッドは `@_lock_manager()` で直列化される。`Mecab` / `NJD` / `JPCommon` はインスタンス内で共有される。
            `Mecab_refresh()` は Python 側の `try/finally` から呼び出し、lattice ノード走査後に MeCab 内部状態を解放する。
        """
        cdef char* _dn_mecab = dn_mecab
        cdef char* _userdic = userdic
        cdef tuple protection_flags
        cdef Py_ssize_t userdic_count

        # 引数検証で例外になっても __dealloc__() が未初期化ポインタを解放しないよう先に NULL を設定する
        self.mecab = NULL
        self.njd = NULL
        self.jpcommon = NULL

        # 低レベル API のカンマ区切り辞書と同じ順序で保護状態を固定する
        userdic_count = 0 if len(userdic) == 0 else len(userdic.split(b","))
        if userdic_reading_protection is None:
            protection_flags = (False,) * userdic_count
        else:
            protection_flags = tuple(userdic_reading_protection)
            if len(protection_flags) != userdic_count:
                raise ValueError(
                    "userdic_reading_protection must have the same number of entries as userdic"
                )
            if any(type(flag) is not bool for flag in protection_flags):
                raise TypeError("userdic_reading_protection entries must be bool")
        self.userdic_reading_protection = protection_flags

        # 排他範囲をインスタンス内へ限定し、異なる辞書を使う処理同士も並行実行できるようにする
        self._lock = Lock()
        self.mecab = new Mecab()
        self.njd = new NJD()
        self.jpcommon = new JPCommon()

        with nogil:
            Mecab_initialize(self.mecab)
            NJD_initialize(self.njd)
            JPCommon_initialize(self.jpcommon)

            r = self._load(_dn_mecab, _userdic)
            if r != 1:
                self._clear()
        if r != 1:
            raise RuntimeError("Failed to initialize Mecab")

    cdef void _clear(self) noexcept nogil:
        """
        インスタンスが保持する MeCab / NJD / JPCommon の内部バッファをクリアする。
        C Wrapper 自体は解放しない。
        """
        if self.mecab != NULL:
            Mecab_clear(self.mecab)
        if self.njd != NULL:
            NJD_clear(self.njd)
        if self.jpcommon != NULL:
            JPCommon_clear(self.jpcommon)

    cdef int _load(self, char* dn_mecab, char* userdic) noexcept nogil:
        """
        `Mecab_load_with_userdic()` へ委譲して辞書をロードする。

        Returns:
            int: 成功時 1、失敗時 0
        """
        return Mecab_load_with_userdic(self.mecab, dn_mecab, userdic)

    @_lock_manager()
    def normalize_for_mecab(self, text: str | bytes | bytearray) -> str:
        """
        OpenJTalk の MeCab 入力と同じ規則で本文を正規化する。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            str: 正規化されたテキスト
        """
        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        cdef int text2mecab_result
        if isinstance(text, str):
            text = text.encode("utf-8")
        cdef const char* _text = text
        with nogil:
            text2mecab_result = text2mecab(buff, TEXT2MECAB_BUFFER_SIZE, _text)
        if text2mecab_result != 0:
            if text2mecab_result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if text2mecab_result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(text2mecab_result))
        return (<bytes> buff).decode("utf-8")

    def _run_mecab(self, text: str | bytes | bytearray) -> list[str]:
        """
        MeCab で形態素解析し、NJD 入力用の feature 文字列列を返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く)

        NOTE:
            pyopenjtalk-plus 独自の "記号,空白" フィルタを適用する。
            `text2mecab` が半角スペースを全角スペースへ変換し MeCab が "記号,空白" としてトークン化すると、
            NJD 経由で `pau` が挿入されるため、通常の G2P 経路では除外する。
            全トークンが必要な場合は `_run_mecab_detailed()` を使う。
        """
        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        cdef int result
        with nogil:
            result = text2mecab(buff, TEXT2MECAB_BUFFER_SIZE, _text)
        if result != 0:
            if result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(result))

        cdef int morph_size
        cdef char** mecab_morphs
        cdef int analysis_result
        with nogil:
            analysis_result = Mecab_analysis(self.mecab, buff)

            morph_size = Mecab_get_size(self.mecab)
            mecab_morphs = Mecab_get_feature(self.mecab)
        try:
            if analysis_result != 1:
                raise RuntimeError("Failed to run MeCab analysis")
            if morph_size > 0 and mecab_morphs == NULL:
                raise RuntimeError("MeCab returned invalid feature buffer")
            if morph_size < 0:
                raise RuntimeError("MeCab returned invalid morph size")

            # "記号,空白" を NJD 入力から除外する (NOTE は _run_mecab() の Docstring を参照)
            morphs = []
            for i in range(morph_size):
                if mecab_morphs[i] == NULL:
                    raise RuntimeError("MeCab returned null morph entry")
                m = (<bytes>(mecab_morphs[i])).decode("utf-8")
                if "記号,空白" not in m:
                    morphs.append(m)
            return morphs
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab(self, text: str | bytes | bytearray) -> list[str]:
        """
        MeCab で形態素解析を実行する。"記号,空白" は除外される。
        全トークン (未知語フラグ・コスト情報含む) が必要な場合は代わりに run_mecab_detailed() を使うこと。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く)
        """
        return self._run_mecab(text)

    def _run_mecab_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[str], list[MeCabMorph]]:
        """
        MeCab で形態素解析し、フィルタ済み features と全 morphs を同時に返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            tuple[list[str], list[MeCabMorph]]: (フィルタ済み features, 全 morphs)
                features は `_run_mecab()` と同等 ("記号,空白" を除く)
                morphs は lattice 走査で構築した詳細形態素列 ("記号,空白" も含む)

        NOTE:
            `Mecab_analysis()` 後に lattice ノードを走査し、未知語フラグ・コスト・文字位置を取得する。
            未知語に連結された連続記号は、既知記号辞書を使って1文字ずつ morph へ分割する。
        """

        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        # cdef 宣言は関数スコープの先頭でなければならないため、ここで事前宣言する
        cdef mecab_lattice_t* lattice = NULL
        cdef mecab_node_t* node
        cdef int morph_size
        cdef char** mecab_feature_array
        cdef int analysis_result
        cdef bytes sentence_bytes
        cdef list byte_to_char_offsets

        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        cdef int result
        with nogil:
            result = text2mecab(buff, TEXT2MECAB_BUFFER_SIZE, _text)
        if result != 0:
            if result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(result))

        sentence_bytes = <bytes> buff
        byte_to_char_offsets = _build_byte_to_char_offsets(sentence_bytes)

        # Mecab_analysis() で解析を実行
        with nogil:
            analysis_result = Mecab_analysis(self.mecab, buff)
            morph_size = Mecab_get_size(self.mecab)
            mecab_feature_array = Mecab_get_feature(self.mecab)
        try:
            if analysis_result != 1:
                raise RuntimeError("Failed to run MeCab analysis")
            if morph_size > 0 and mecab_feature_array == NULL:
                raise RuntimeError("MeCab returned invalid feature buffer")
            if morph_size < 0:
                raise RuntimeError("MeCab returned invalid morph size")

            # NJD へ渡す features は通常経路と同じ MeCab の解析結果から構築
            ## 詳細情報で既知記号を1文字ずつ復元しても、発音解析まで記号単位へ変化させない
            features = []
            for i in range(morph_size):
                if mecab_feature_array[i] == NULL:
                    raise RuntimeError("MeCab returned null morph entry")
                mecab_feature = (<bytes>(mecab_feature_array[i])).decode("utf-8")
                if "記号,空白" not in mecab_feature:
                    features.append(mecab_feature)

            # lattice ノードを走査して MeCabMorph リストを構築
            ## 未知語へ連結された既知記号は、NJD 入力から独立した詳細情報として1文字ずつ復元
            if self.mecab.lattice == NULL:
                raise RuntimeError("Failed to access MeCab lattice")
            lattice = <mecab_lattice_t*> self.mecab.lattice
            node = mecab_lattice_get_bos_node(lattice)

            morphs = []
            while node != NULL:
                stat = node.stat
                # BOS (stat=2), EOS (stat=3) ノードはスキップ
                if stat != 2 and stat != 3:
                    # 通常ノードと記号分割の両方で lattice 由来のコスト・位置情報を共通化する
                    node_morph = _mecab_node_to_morph(
                        node,
                        True,
                        buff,
                        byte_to_char_offsets,
                    )
                    for split_morph in _expand_symbol_morphs(node, node_morph):
                        morphs.append(split_morph)
                node = node.next

            return features, morphs
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[str], list[MeCabMorph]]:
        """
        MeCab を1回だけ実行し、run_mecab() 互換の features と詳細 morphs を返す。
        詳細 morphs には "記号,空白" も含まれ、未知語フラグ・コスト情報を保持する。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            tuple[list[str], list[MeCabMorph]]: (フィルタ済み features, 全 morphs)
                features は run_mecab() と同等 ("記号,空白" を除く)
                morphs は lattice 走査で構築した詳細形態素列 ("記号,空白" も含む)

        NOTE:
            `Mecab_analysis()` 後に lattice ノードを走査し、未知語フラグ・コスト・文字位置を取得する。
            未知語に連結された連続記号は、既知記号辞書を使って1文字ずつ morph へ分割する。
        """

        return self._run_mecab_detailed(text)

    def _run_mecab_nbest_features(
        self, text: str | bytes | bytearray, max_paths: int = 5
    ) -> list[MeCabNBestPath]:
        """
        MeCab の n-best 候補を、NJD に渡せる features と詳細 morphs の組として返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)

        Returns:
            list[MeCabNBestPath]: 各候補パスの features / morphs / path_cost

        NOTE:
            `parseNBestInit()` は Tagger 内部の可変 lattice を使う。終了時は `Mecab_refresh()` で OpenJTalk 側状態も初期化する。
        """

        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        cdef int result
        cdef int init_result
        cdef int stat
        cdef long path_cost
        cdef mecab_t* tagger = NULL
        cdef const mecab_node_t* const_node
        cdef mecab_node_t* node
        cdef bytes sentence_bytes
        cdef list byte_to_char_offsets

        if isinstance(max_paths, bool) is True or isinstance(max_paths, int) is False:
            raise TypeError("max_paths must be int")
        if max_paths < 1 or max_paths > 512:
            raise ValueError("max_paths must be between 1 and 512")

        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        with nogil:
            result = text2mecab(buff, TEXT2MECAB_BUFFER_SIZE, _text)
        if result != 0:
            if result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(result))

        sentence_bytes = <bytes> buff
        byte_to_char_offsets = _build_byte_to_char_offsets(sentence_bytes)

        if self.mecab.tagger == NULL:
            raise RuntimeError("Failed to access MeCab tagger")
        tagger = <mecab_t*> self.mecab.tagger

        # parseNBestInit() は Tagger 内部の可変ラティスを使う
        ## 既存の Mecab_analysis() 用 lattice とは別領域なので、最後は Mecab_refresh() で OpenJTalk 側の状態も初期化する
        with nogil:
            init_result = mecab_nbest_init2(tagger, buff, strlen(buff))
        try:
            if init_result != 1:
                raise RuntimeError("Failed to initialize MeCab n-best analysis")

            paths = []
            for _path_index in range(max_paths):
                with nogil:
                    const_node = mecab_nbest_next_tonode(tagger)
                if const_node == NULL:
                    break

                node = <mecab_node_t*> const_node
                features = []
                morphs = []
                path_cost = 0
                while node != NULL:
                    stat = node.stat
                    if stat != 2:
                        path_cost += _selected_mecab_link_cost(node, False)

                    # BOS/EOS/EON は制御用ノードなので、形態素候補としては返さない
                    if stat != 2 and stat != 3 and stat != 4:
                        morph = _mecab_node_to_morph(
                            node,
                            False,
                            buff,
                            byte_to_char_offsets,
                        )
                        morphs.append(morph)
                        if morph["is_ignored"] is False:
                            features.append(",".join(morph["features"]))
                    node = node.next

                paths.append(MeCabNBestPath(
                    features=features,
                    morphs=morphs,
                    path_cost=path_cost,
                ))
            return paths
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab_nbest_features(
        self, text: str | bytes | bytearray, max_paths: int = 5
    ) -> list[MeCabNBestPath]:
        """
        MeCab の n-best 候補を features / morphs / path_cost 付きで返す。
        features は run_njd_from_mecab() に渡せる形式で、morphs は run_mecab_detailed() と同じ詳細形式を持つ。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)

        Returns:
            list[MeCabNBestPath]: MeCab n-best 候補パスのリスト
        """
        return self._run_mecab_nbest_features(text, max_paths)

    @_lock_manager()
    def analyze_mecab_candidates(
        self,
        text: str | bytes | bytearray,
        target_spans: Sequence[tuple[int, int]],
    ) -> ReadingAnalysis:
        """
        MeCab の補正前最良経路と全候補ノードをコピーして返す。
        戻り値は MeCab の生ポインタを含まず、呼び出し完了後にモデル推論へ安全に渡せる。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            target_spans (Sequence[tuple[int, int]]): 候補経路を列挙する正規化本文上の半開区間

        Returns:
            ReadingAnalysis: 正規化本文、最良経路、候補グラフのコピー

        NOTE:
            MeCab を NBEST モードで解析し、候補ノード間の接続辺を取得する。コスト変更や最良経路の再計算は行わない。
            戻り値は lattice ノードの Python コピーのみで、呼び出し完了後は `Mecab_refresh()` で C 側 lattice を解放する。
            tsqyomi はこの戻り値をロック外でモデル推論へ渡せる。
        """

        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        cdef int text2mecab_result
        cdef int parse_result
        cdef int previous_request_type
        cdef int stat
        cdef size_t lattice_size
        cdef size_t pos
        cdef Py_ssize_t node_index
        cdef mecab_t* tagger = NULL
        cdef mecab_lattice_t* lattice = NULL
        cdef mecab_node_t* node = NULL
        cdef mecab_node_t* bos_node = NULL
        cdef const char* sentence = NULL
        cdef bytes sentence_bytes
        cdef list byte_to_char_offsets
        cdef list candidates
        cdef list node_addresses
        cdef dict node_index_by_address
        cdef tuple normalized_target_spans
        cdef dict best_neighbors_by_span
        cdef dict previous_neighbor_by_start
        cdef dict next_neighbor_by_end
        cdef dict candidate
        cdef object candidate_span
        cdef object morph
        cdef list features
        cdef list morphs
        cdef set public_node_ids
        cdef int public_node_id
        cdef list public_nodes
        cdef list public_paths
        cdef list public_connections
        cdef long left_boundary_cost
        cdef long right_boundary_cost
        cdef mecab_path_t* candidate_path
        cdef uintptr_t previous_node_address
        cdef uintptr_t next_node_address
        cdef uintptr_t right_node_address

        normalized_target_spans = tuple(target_spans)
        if len(normalized_target_spans) == 0:
            raise ValueError("target_spans must not be empty")

        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        with nogil:
            text2mecab_result = text2mecab(buff, TEXT2MECAB_BUFFER_SIZE, _text)
        if text2mecab_result != 0:
            if text2mecab_result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if text2mecab_result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(text2mecab_result))

        if self.mecab.tagger == NULL or self.mecab.lattice == NULL:
            raise RuntimeError("Failed to access MeCab internals")
        tagger = <mecab_t*> self.mecab.tagger
        lattice = <mecab_lattice_t*> self.mecab.lattice

        # 候補の接続辺を取得するため NBEST で解析するが、費用変更と最良経路の再計算は行わない
        previous_request_type = mecab_lattice_get_request_type(lattice)
        with nogil:
            mecab_lattice_set_sentence(lattice, buff)
            mecab_lattice_set_request_type(lattice, MECAB_NBEST)
            parse_result = mecab_parse_lattice(tagger, lattice)
        try:
            if parse_result != 1:
                raise RuntimeError("Failed to run MeCab lattice analysis")
            sentence = mecab_lattice_get_sentence(lattice)
            if sentence == NULL:
                raise RuntimeError("Failed to access MeCab lattice sentence")
            sentence_bytes = <bytes> sentence
            byte_to_char_offsets = _build_byte_to_char_offsets(sentence_bytes)
            lattice_size = mecab_lattice_get_size(lattice)

            candidates = []
            node_addresses = []
            node_index_by_address = {}
            node_index = 0

            # BOS も費用計算の境界として必要なので候補列へ保持する
            bos_node = mecab_lattice_get_bos_node(lattice)
            if bos_node == NULL:
                raise RuntimeError("Failed to access MeCab BOS node")
            candidate = _mecab_node_to_cost_candidate(
                bos_node,
                sentence,
                byte_to_char_offsets,
                self.userdic_reading_protection,
            )
            candidates.append(candidate)
            node_addresses.append(<uintptr_t> bos_node)
            node_index_by_address[<uintptr_t> bos_node] = node_index
            node_index += 1

            # 候補の列挙順を固定し、同額候補の選択診断を再現可能にする
            for pos in range(lattice_size + 1):
                node = mecab_lattice_get_begin_nodes(lattice, pos)
                while node != NULL:
                    candidate = _mecab_node_to_cost_candidate(
                        node,
                        sentence,
                        byte_to_char_offsets,
                        self.userdic_reading_protection,
                    )
                    candidates.append(candidate)
                    node_addresses.append(<uintptr_t> node)
                    node_index_by_address[<uintptr_t> node] = node_index
                    node_index += 1
                    node = node.bnext

            # 固定した外側経路と接続する費用だけを読み実現候補へ残す
            best_neighbors_by_span = {}
            previous_neighbor_by_start = {}
            next_neighbor_by_end = {}
            node = mecab_lattice_get_bos_node(lattice)
            while node != NULL:
                if node.stat != 2 and node.stat != 3:
                    candidate = candidates[node_index_by_address[<uintptr_t> node]]
                    best_neighbors_by_span[candidate["char_span"]] = (
                        <uintptr_t> node.prev,
                        <uintptr_t> node.next,
                    )
                    previous_neighbor_by_start[candidate["char_span"][0]] = <uintptr_t> node.prev
                    next_neighbor_by_end[candidate["char_span"][1]] = <uintptr_t> node.next
                node = node.next

            for node_index in range(len(candidates)):
                node = <mecab_node_t*> <uintptr_t> node_addresses[node_index]
                candidate = candidates[node_index]
                candidate["local_replacement_cost"] = None
                candidate["left_boundary_cost"] = None
                candidate["right_boundary_cost"] = None
                candidate_span = candidate["char_span"]
                # 複数形態素を覆う1ノード候補も、固定外側経路の両境界が一致すれば交換対象にできる
                if (
                    candidate_span[0] in previous_neighbor_by_start
                    and candidate_span[1] in next_neighbor_by_end
                ):
                    best_neighbors_by_span[candidate_span] = (
                        previous_neighbor_by_start[candidate_span[0]],
                        next_neighbor_by_end[candidate_span[1]],
                    )
                if candidate_span in best_neighbors_by_span and node.stat != 2 and node.stat != 3:
                    previous_node_address, next_node_address = best_neighbors_by_span[candidate_span]
                    left_boundary_cost = LONG_MAX
                    candidate_path = node.lpath
                    while candidate_path != NULL:
                        if <uintptr_t> candidate_path.lnode == previous_node_address:
                            left_boundary_cost = candidate_path.cost
                            break
                        candidate_path = candidate_path.lnext
                    right_boundary_cost = LONG_MAX
                    candidate_path = node.rpath
                    while candidate_path != NULL:
                        if <uintptr_t> candidate_path.rnode == next_node_address:
                            right_boundary_cost = candidate_path.cost
                            break
                        candidate_path = candidate_path.rnext
                    if left_boundary_cost != LONG_MAX and right_boundary_cost != LONG_MAX:
                        candidate["left_boundary_cost"] = left_boundary_cost
                        candidate["right_boundary_cost"] = (
                            right_boundary_cost - (<mecab_node_t*> next_node_address).wcost
                        )
                        candidate["local_replacement_cost"] = (
                            candidate["left_boundary_cost"] + candidate["right_boundary_cost"]
                        )

            # 補正前の node.next が示す最良経路をコピーし、モデル選択後も外側経路を固定できるようにする
            features = []
            morphs = []
            node = mecab_lattice_get_bos_node(lattice)
            while node != NULL:
                stat = node.stat
                if stat != 2 and stat != 3:
                    node_morph = _mecab_node_to_morph(
                        node,
                        True,
                        sentence,
                        byte_to_char_offsets,
                    )
                    for split_morph in _expand_symbol_morphs(node, node_morph):
                        morphs.append(split_morph)
                    if node_morph["is_ignored"] is False:
                        features.append(",".join(node_morph["features"]))
                node = node.next

            public_nodes = []
            public_paths = []
            for node_index in range(len(candidates)):
                candidate = candidates[node_index]
                if candidate["local_replacement_cost"] is None:
                    continue
                if candidate["char_span"] not in normalized_target_spans:
                    continue
                if len(candidate["features"]) <= 9:
                    continue
                public_nodes.append(CandidateNode(
                    node_id=node_index,
                    surface=candidate["surface"],
                    feature=",".join(candidate["features"]),
                    pronunciation=candidate["features"][9],
                    char_span=candidate["char_span"],
                    pos_id=candidate["pos_id"],
                    left_id=candidate["left_id"],
                    right_id=candidate["right_id"],
                    word_cost=candidate["word_cost"],
                    dictionary_index=candidate["dictionary_index"],
                    is_unknown=candidate["is_unknown"],
                    is_ignored=candidate["is_ignored"],
                    is_reading_protected=candidate["is_reading_protected"],
                ))
                public_paths.append(CandidatePath(
                    path_id=len(public_paths),
                    node_ids=(node_index,),
                    char_span=candidate["char_span"],
                    surface=candidate["surface"],
                    pronunciation=candidate["features"][9],
                    features=(",".join(candidate["features"]),),
                    left_boundary_cost=candidate["left_boundary_cost"],
                    right_boundary_cost=candidate["right_boundary_cost"],
                    boundary_cost=candidate["local_replacement_cost"],
                ))

            public_node_ids = set()
            for public_node in public_nodes:
                public_node_ids.add(public_node["node_id"])

            # 公開候補ノード同士を結ぶ辺だけをコピーする
            public_connections = []
            for public_node_id in public_node_ids:
                node = <mecab_node_t*> <uintptr_t> node_addresses[public_node_id]
                candidate_path = node.rpath
                while candidate_path != NULL:
                    right_node_address = <uintptr_t> candidate_path.rnode
                    if right_node_address in node_index_by_address:
                        right_node_id = node_index_by_address[right_node_address]
                        if right_node_id in public_node_ids:
                            public_connections.append(CandidateConnection(
                                left_node_id=public_node_id,
                                right_node_id=right_node_id,
                                cost=candidate_path.cost,
                            ))
                    candidate_path = candidate_path.rnext

            return ReadingAnalysis(
                normalized_text=sentence_bytes.decode("utf-8"),
                features=tuple(features),
                morphs=tuple(morphs),
                nodes=tuple(public_nodes),
                paths=tuple(public_paths),
                connections=tuple(public_connections),
            )
        finally:
            mecab_lattice_set_request_type(lattice, previous_request_type)
            Mecab_refresh(self.mecab)

    def _run_njd_from_mecab(self, mecab_features: list[str]) -> list[NJDFeature]:
        """
        MeCab feature 列から NJD 処理を実行し、Python 側のアクセント結合規則を挟んで NJDFeature 列を返す。

        Args:
            mecab_features (list[str]): MeCab の feature 文字列のリスト

        Returns:
            list[NJDFeature]: NJD 処理後の features

        NOTE:
            `mecab2njd` → Python dict → `apply_original_rule_before_chaining()` → NJD 再構築 → digit/accent 等
            という二重変換を行う。Python dict を直接操作して chaining 前ルールを適用するため、この構造が必要。
            処理完了後は `NJD_refresh()` で C 側メモリを解放する。
        """
        # if empty list, return empty list
        new_size = len(mecab_features)
        if new_size == 0:
            return []

        for mecab_feature in mecab_features:
            if isinstance(mecab_feature, str) is False:
                raise TypeError("Each MeCab feature must be str")
            if "\x00" in mecab_feature:
                raise ValueError("MeCab feature must not contain null characters")

        byte_morphs = [m.encode("utf-8") + b"\x00" for m in mecab_features]
        int_morphs = np.zeros(len(byte_morphs), dtype=np.uint64)
        for i in range(new_size):
            int_morphs[i] = <uint64_t>(<char *>byte_morphs[i])

        cdef uint64_t[:] cint_morphs = int_morphs
        cdef char** new_mecab_morphs = <char**>&cint_morphs[0]
        with nogil:
            mecab2njd(self.njd, new_mecab_morphs, new_size)

            _njd.njd_set_pronunciation(self.njd)

        feature = njd2feature(self.njd)
        feature = apply_original_rule_before_chaining(feature)
        NJD_refresh(self.njd)
        feature2njd(self.njd, feature)

        with nogil:
            _njd.njd_set_digit(self.njd)
            _njd.njd_set_accent_phrase(self.njd)
            _njd.njd_set_accent_type(self.njd)
            _njd.njd_set_unvoiced_vowel(self.njd)
            _njd.njd_set_long_vowel(self.njd)
        feature = njd2feature(self.njd)

        # Note that this will release memory for njd feature
        NJD_refresh(self.njd)

        return feature

    @_lock_manager()
    def run_njd_from_mecab(self, mecab_features: list[str]) -> list[NJDFeature]:
        """
        MeCab の feature 文字列のリストから NJD 処理を実行する。
        run_mecab() の戻り値をそのまま渡す想定。
        数字正規化・アクセント句設定・長音処理などの NJD ルールが適用される。

        Args:
            mecab_features (list[str]): MeCab の feature 文字列のリスト

        Returns:
            list[NJDFeature]: NJDNode 用 features
        """
        return self._run_njd_from_mecab(mecab_features)

    @_lock_manager()
    def run_frontend(self, text: str | bytes | bytearray) -> list[NJDFeature]:
        """
        OpenJTalk のテキスト処理フロントエンドを実行する。
        MeCab 形態素詳細を構築せず、NJD features のみを返す軽量経路。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[NJDFeature]: NJDNode 用 features
        """
        features = self._run_mecab(text)
        njd_features = self._run_njd_from_mecab(features)
        return njd_features

    @_lock_manager()
    def run_frontend_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[NJDFeature], list[MeCabMorph]]:
        """
        OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
        MeCab 解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
                NJD features は run_frontend() と、MeCab morphs は run_mecab_detailed() と同一の結果
        """
        features, morphs = self._run_mecab_detailed(text)
        njd_features = self._run_njd_from_mecab(features)
        return njd_features, morphs

    @_lock_manager()
    def extract_phonemes(self, features: Iterable[NJDFeature]) -> list[str]:
        """
        NJD features からフラットな音素列を直接抽出する。
        HTS フルコンテキストラベル文字列は生成せず、JPCommonLabel の音素連結リストをそのまま走査する。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フラットな音素列

        NOTE:
            `try/finally` で `JPCommon_refresh()` と `NJD_refresh()` を呼び、インスタンス共有バッファを解放する。
        """

        cdef JPCommonLabelPhoneme* phoneme_node
        cdef JPCommonNode* node

        features = list(features)
        if not features:
            return []

        try:
            feature2njd(self.njd, features)
            with nogil:
                njd2jpcommon(self.jpcommon, self.njd)

            if self.jpcommon.label != NULL:
                JPCommonLabel_clear(self.jpcommon.label)
            else:
                self.jpcommon.label = <JPCommonLabel*> calloc(1, sizeof(JPCommonLabel))
                if self.jpcommon.label == NULL:
                    raise MemoryError("Failed to allocate JPCommonLabel")
            JPCommonLabel_initialize(self.jpcommon.label)

            node = self.jpcommon.head
            while node != NULL:
                JPCommonLabel_push_word(
                    self.jpcommon.label,
                    JPCommonNode_get_pron(node),
                    JPCommonNode_get_pos(node),
                    JPCommonNode_get_ctype(node),
                    JPCommonNode_get_cform(node),
                    JPCommonNode_get_acc(node),
                    JPCommonNode_get_chain_flag(node),
                )
                node = <JPCommonNode*> node.next

            if self.jpcommon.label.is_valid == 0:
                raise RuntimeError("JPCommonLabel internal allocation failure (is_valid=0)")

            phonemes = []
            phoneme_node = self.jpcommon.label.phoneme_head
            while phoneme_node != NULL:
                if phoneme_node.phoneme != NULL:
                    phonemes.append((<bytes> phoneme_node.phoneme).decode("utf-8"))
                phoneme_node = phoneme_node.next

            return phonemes
        finally:
            JPCommon_refresh(self.jpcommon)
            NJD_refresh(self.njd)

    @_lock_manager()
    def make_label(self, features: Iterable[NJDFeature]) -> list[str]:
        """
        HTS 音声合成用のフルコンテキストラベルを返す。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フルコンテキストラベル文字列のリスト

        NOTE:
            `try/finally` で `JPCommon_refresh()` と `NJD_refresh()` を呼び、ラベル文字列と中間バッファを解放する。
        """
        try:
            feature2njd(self.njd, features)
            with nogil:
                njd2jpcommon(self.jpcommon, self.njd)

                JPCommon_make_label(self.jpcommon)

                label_size = JPCommon_get_label_size(self.jpcommon)
                label_feature = JPCommon_get_label_feature(self.jpcommon)
            if label_size > 0 and label_feature == NULL:
                raise RuntimeError("Failed to create full-context labels")
            if label_size < 0:
                raise RuntimeError("OpenJTalk returned invalid label size")

            labels = []
            for i in range(label_size):
                if label_feature[i] == NULL:
                    raise RuntimeError("OpenJTalk returned null label entry")
                # This will create a copy of c string
                # http://cython.readthedocs.io/en/latest/src/tutorial/strings.html
                labels.append(<unicode>label_feature[i])
            return labels
        finally:
            # Note that this will release memory for label feature
            JPCommon_refresh(self.jpcommon)
            NJD_refresh(self.njd)

    @_lock_manager()
    def make_phoneme_mapping(self, features: Iterable[NJDFeature]) -> list[dict[str, Any]]:
        """
        NJD features から各形態素に対応する音素列のマッピングを生成する。
        JPCommon の Word-Mora-Phoneme 階層を構築し、各 feature に音素を割り当てる。
        NJD の pron が短ポーズを表す記号へ ["pau"] を割り当て、括弧類は空の音素列で保持する。
        長音吸収マージにより、戻り値の長さが入力と異なる場合がある。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[dict[str, Any]]: NJDFeature の全フィールド + phonemes を含む辞書のリスト。
                MeCab の未知語情報や features が必要な場合は pyopenjtalk.make_phoneme_mapping() を使用すること

        Raises:
            RuntimeError: JPCommonLabel の内部アロケーション失敗時

        NOTE:
            `JPCommon_make_label()` は呼ばず、`JPCommonLabel_push_word()` で Word-Mora-Phoneme 階層だけ構築する。
            ポーズ形態素 ("、"/"？"/"！") や長音吸収された 'ー' では Word が生成されず、対応 feature の音素は空のままになる。
            長音吸収で隣接 feature がマージされ、戻り値の要素数が入力より少なくなる場合がある。
        """

        # cdef 宣言は関数スコープの先頭でなければならないため、ここで事前宣言する
        cdef JPCommonLabelWord* prev_word_tail
        cdef JPCommonLabelWord* curr_word_tail
        cdef JPCommonLabelPhoneme* phoneme_node
        cdef JPCommonLabelMora* mora_ptr
        cdef JPCommonLabelWord* word_ptr
        cdef JPCommonNode* node

        # features を複数回イテレーションする (feature2njd, pause 割当, メインループ) ため、
        # Iterable (ジェネレータ等) が渡された場合に備えて list に変換する
        features = list(features)

        if not features:
            return []

        try:
            feature2njd(self.njd, features)
            with nogil:
                njd2jpcommon(self.jpcommon, self.njd)

            # JPCommonLabel_push_word() を個別に呼び出し、
            # Word が新規生成されたかを word_tail の変化で追跡する
            if self.jpcommon.label != NULL:
                JPCommonLabel_clear(self.jpcommon.label)
            else:
                self.jpcommon.label = <JPCommonLabel*> calloc(1, sizeof(JPCommonLabel))
                if self.jpcommon.label == NULL:
                    raise MemoryError("Failed to allocate JPCommonLabel")
            JPCommonLabel_initialize(self.jpcommon.label)

            # Word ポインタ → feature index のマッピングを構築
            # JPCommonLabel_push_word() は以下の場合に新しい Word を生成しない:
            #   - pause-like な記号で、Word を生成せず短ポーズフラグのみ設定した場合
            #   - 長音 'ー' で先行 Word に吸収された場合
            # これらの feature は ptr_to_idx に含まれず、音素が空のままになる
            ptr_to_idx = {}
            node = self.jpcommon.head
            f_idx = 0
            while node != NULL:
                prev_word_tail = self.jpcommon.label.word_tail

                JPCommonLabel_push_word(
                    self.jpcommon.label,
                    JPCommonNode_get_pron(node),
                    JPCommonNode_get_pos(node),
                    JPCommonNode_get_ctype(node),
                    JPCommonNode_get_cform(node),
                    JPCommonNode_get_acc(node),
                    JPCommonNode_get_chain_flag(node),
                )

                # push_word 後に word_tail が変化していれば新しい Word が生成された
                curr_word_tail = self.jpcommon.label.word_tail
                if prev_word_tail != curr_word_tail and curr_word_tail != NULL:
                    ptr_to_idx[<uintptr_t> curr_word_tail] = f_idx

                node = <JPCommonNode*> node.next
                f_idx += 1

            # JPCommonLabel_make() は不要 (push_word() で階層は構築済み)
            # is_valid=0 は内部アロケーション失敗を示す
            if self.jpcommon.label.is_valid == 0:
                raise RuntimeError("JPCommonLabel internal allocation failure (is_valid=0)")

            # --- 全 feature のマッピングを初期化 ---
            # NJDFeature の全フィールドを転写する
            mapping = []
            for feat in features:
                phonemes = []
                is_pause_pron = feat["pron"] in ("、", "？", "！")
                if is_pause_pron is True and feat["string"] not in _NON_PAUSE_SYMBOLS:
                    phonemes.append("pau")

                mapping.append(JPCommonMappingEntry(
                    surface=feat["string"],
                    phonemes=phonemes,
                    pos=feat["pos"],
                    pos_group1=feat["pos_group1"],
                    pos_group2=feat["pos_group2"],
                    pos_group3=feat["pos_group3"],
                    ctype=feat["ctype"],
                    cform=feat["cform"],
                    orig=feat["orig"],
                    read=feat["read"],
                    pron=feat["pron"],
                    accent_nucleus=feat["acc"],
                    mora_count=feat["mora_size"],
                    chain_rule=feat["chain_rule"],
                    chain_flag=feat["chain_flag"],
                ))

            # 通常音素を Phoneme → Mora → Word の階層から対応する feature へ割り当てる
            ## pau は Word を持たず位置を逆引きできないため、NJD の pron から上で静的に割り当て済み
            phoneme_node = self.jpcommon.label.phoneme_head
            while phoneme_node != NULL:
                if phoneme_node.phoneme != NULL:
                    phoneme_str = (<bytes> phoneme_node.phoneme).decode("utf-8")

                    if phoneme_str != "pau":
                        # phoneme → Mora → Word の階層を辿って feature index を取得
                        mora_ptr = phoneme_node.up
                        if mora_ptr != NULL:
                            word_ptr = mora_ptr.up
                            if word_ptr != NULL:
                                word_addr = <uintptr_t> word_ptr
                                if word_addr in ptr_to_idx:
                                    target_idx = ptr_to_idx[word_addr]
                                    mapping[target_idx]["phonemes"].append(phoneme_str)

                phoneme_node = phoneme_node.next

            # 長音吸収マージ: 長音処理で先行 Word に吸収されたトークンは音素が空のまま残る
            # 記号由来の空音素まで誤って吸収しないよう、pron が長音記号のみの要素だけ前方に結合する
            merged = []
            for entry in mapping:
                is_absorbed_long_vowel = (
                    len(entry["phonemes"]) == 0
                    and len(entry["pron"]) > 0
                    and set(entry["pron"]) == {"ー"}
                    and len(merged) > 0
                )
                if is_absorbed_long_vowel is True:
                    prev = merged[-1]
                    # 前方が ["pau"] や空音素の場合は結合しない
                    is_prev_pause = (len(prev["phonemes"]) == 1 and prev["phonemes"][0] == "pau")
                    if is_prev_pause is False and len(prev["phonemes"]) > 0:
                        prev["surface"] += entry["surface"]
                        prev["mora_count"] += entry["mora_count"]
                        # orig は辞書の原形を表すため、活用形の吸収 (食べよ+う→食べよう) では連結しない
                        # ただしリテラルの長音記号 (ー) が吸収された場合は入力テキストを保持するため連結する
                        if set(entry["orig"]) == {"ー"}:
                            prev["orig"] += entry["orig"]
                        prev["read"] += entry["read"]
                        prev["pron"] += entry["pron"]
                        continue
                merged.append(entry)
            mapping = merged

            return mapping
        finally:
            # Note that this will release memory for label feature
            JPCommon_refresh(self.jpcommon)
            NJD_refresh(self.njd)

    def g2p(
        self, text: str | bytes | bytearray, kana: bool = False, join: bool = True
    ) -> list[str] | str:
        """
        文字から音素への変換 (G2P) 。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式。デフォルト: False
            join (bool): True の場合、音素またはカタカナを単一の文字列に連結する。デフォルト: True

        Returns:
            str | list[str]: kana と join の組み合わせにより、str または list[str] を返す
        """
        njd_features = self.run_frontend(text)

        if not kana:
            prons = self.extract_phonemes(njd_features)
            if join:
                prons = " ".join(prons)
            return prons

        # kana
        prons = []
        for n in njd_features:
            if n["pos"] == "記号":
                p = n["string"]
            else:
                p = n["pron"]
            # remove special chars
            for c in "’":
                p = p.replace(c,"")
            prons.append(p)
        if join:
            prons = "".join(prons)
        return prons

    def __dealloc__(self) -> None:
        """
        MeCab / NJD / JPCommon の C Wrapper を解放する。
        """
        self._clear()
        if self.mecab != NULL:
            del self.mecab
        if self.njd != NULL:
            del self.njd
        if self.jpcommon != NULL:
            del self.jpcommon

def mecab_dict_index(dn_mecab: bytes, path: bytes, out_path: bytes) -> int:
    """
    OpenJTalk 用のユーザー辞書を CSV からビルドする。低レベル API 。
    通常は pyopenjtalk.mecab_dict_index() を使用すること。
    CSV は naist-jdic 互換の品詞体系で記述する必要がある。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス
        path (bytes): ユーザー CSV ファイルのパス
        out_path (bytes): 出力辞書ファイルのパス

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗)
    """
    cdef char* argv[10]
    argv[0] = "mecab-dict-index"
    argv[1] = "-d"
    argv[2] = dn_mecab
    argv[3] = "-u"
    argv[4] = out_path
    argv[5] = "-f"
    argv[6] = "utf-8"
    argv[7] = "-t"
    argv[8] = "utf-8"
    argv[9] = path
    cdef int ret
    with nogil:
        ret = _mecab_dict_index(10, argv)
    return ret

def build_mecab_dictionary(dn_mecab: bytes) -> int:
    """
    OpenJTalk 用のシステム辞書を再ビルドする。低レベル API 。
    通常は pyopenjtalk.build_mecab_dictionary() を使用すること。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗)
    """
    cdef char* argv[9]
    argv[0] = "mecab-dict-index"
    argv[1] = "-d"
    argv[2] = dn_mecab
    argv[3] = "-o"
    argv[4] = dn_mecab
    argv[5] = "-f"
    argv[6] = "utf-8"
    argv[7] = "-t"
    argv[8] = "utf-8"
    cdef int ret
    with nogil:
        ret = _mecab_dict_index(9, argv)
    return ret

def apply_original_rule_before_chaining(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    NJD features に chaining 前の独自ルールを適用する。内部用。
    サ変接続・接頭語・動詞連続・連用形・助動詞などのアクセント結合規則を適用する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features 。インプレースで更新される

    Returns:
        list[NJDFeature]: 更新後の njd_features（同一オブジェクト）
    """
    for i, njd in enumerate(njd_features[:-1]):
        # サ変動詞(スル)の前にサ変接続や名詞が来た場合は、一つのアクセント句に纏める
        if (njd["pos_group1"] in ["サ変接続", "格助詞", "接続助詞"] or (njd["pos"] == "名詞" and njd["pos_group1"] == "一般") or njd["pos"] == "副詞" ) and njd_features[i+1]["ctype"] == "サ変・スル":
            njd_features[i+1]["chain_flag"] = 1
        # ご遠慮、ご配慮のような接頭語がつく場合にその後に続く単語の結合則を変更する
        if (njd["string"] in ["お","御","ご"] and njd["chain_rule"] == "P1"):
            if njd_features[i+1]["acc"] == 0 or njd_features[i+1]["acc"] == njd_features[i+1]["mora_size"]:
                njd_features[i+1]["chain_rule"] = "C4"
                njd_features[i+1]["acc"] = 0
            else:
                njd_features[i+1]["chain_rule"] = "C1"
        # 動詞(自立)が連続する場合(ex 推し量る、刺し貫く)、後ろの動詞のアクセント核が採用される
        if njd["pos"] == "動詞"  and njd_features[i+1]["pos"] == "動詞" :
            njd_features[i+1]["chain_rule"] = "C1" if njd_features[i+1]["acc"] != 0 else "C4"
        # 連用形のアクセント核の登録を修正する
        if njd["cform"] in ["連用形","連用タ接続","連用ゴザイ接続","連用テ接続"] and njd["acc"] == njd["mora_size"] > 1 :
            njd["acc"] -= 1
        # 「らる、られる」＋「た」の組み合わせで「た」の助動詞/F2@0を上書きしてアクセントを下げないようにする
        if njd["orig"] in ["れる", "られる","せる", "させる","ちゃう"]  and njd_features[i+1]["string"] in ["た"] :
            njd_features[i+1]["chain_rule"] = "F2@1"

        # 形容詞＋「なる、する」は一つのアクセント句に纏める
        if njd["pos"] == "形容詞" and njd_features[i+1]["orig"] in ["なる", "する"]:
            njd_features[i+1]["chain_flag"] = 1

    return njd_features
