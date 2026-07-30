# coding: utf-8
# cython: boundscheck=True, wraparound=True
# cython: c_string_type=unicode, c_string_encoding=ascii
# cython: language_level=3
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownParameterType=false
# pyright: reportWildcardImportFromLibrary=false

import numpy as np
from functools import wraps
from threading import Lock

cimport numpy as np
np.import_array()

from ._known_symbols import KNOWN_SYMBOL_FEATURES

from libc.math cimport isfinite, llround
from libc.limits cimport LONG_MAX, LONG_MIN, SHRT_MAX, SHRT_MIN
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
    mecab_lattice_rebuild_best,
    mecab_lattice_set_request_type,
    mecab_lattice_set_sentence,
    mecab_parse_lattice,
    mecab_nbest_init2,
    mecab_nbest_next_tonode,
    MECAB_ONE_BEST,
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
    if value == NULL:
        return ""
    return (<bytes>value).decode("utf-8")

cdef inline bytes _validate_and_encode_njd_field(feature_node, str field_name) except *:
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
    cdef object field_value

    field_value = feature_node[field_name]
    if isinstance(field_value, bool) is True or isinstance(field_value, int) is False:
        raise TypeError(f"NJD feature field must be int: {field_name}")
    return field_value

cdef njd_node_get_string(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_string(node))

cdef njd_node_get_pos(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos(node))

cdef njd_node_get_pos_group1(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group1(node))

cdef njd_node_get_pos_group2(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group2(node))

cdef njd_node_get_pos_group3(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_pos_group3(node))

cdef njd_node_get_ctype(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_ctype(node))

cdef njd_node_get_cform(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_cform(node))

cdef njd_node_get_orig(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_orig(node))

cdef njd_node_get_read(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_read(node))

cdef njd_node_get_pron(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_pron(node))

cdef int njd_node_get_acc(_njd.NJDNode* node) noexcept:
    return _njd.NJDNode_get_acc(node)

cdef int njd_node_get_mora_size(_njd.NJDNode* node) noexcept:
    return _njd.NJDNode_get_mora_size(node)

cdef njd_node_get_chain_rule(_njd.NJDNode* node):
    return _decode_utf8_or_empty(_njd.NJDNode_get_chain_rule(node))

cdef int njd_node_get_chain_flag(_njd.NJDNode* node) noexcept:
    return _njd.NJDNode_get_chain_flag(node)


cdef node2feature(_njd.NJDNode* node):
    return {
        "string": njd_node_get_string(node),
        "pos": njd_node_get_pos(node),
        "pos_group1": njd_node_get_pos_group1(node),
        "pos_group2": njd_node_get_pos_group2(node),
        "pos_group3": njd_node_get_pos_group3(node),
        "ctype": njd_node_get_ctype(node),
        "cform": njd_node_get_cform(node),
        "orig": njd_node_get_orig(node),
        "read": njd_node_get_read(node),
        "pron": njd_node_get_pron(node),
        "acc": njd_node_get_acc(node),
        "mora_size": njd_node_get_mora_size(node),
        "chain_rule": njd_node_get_chain_rule(node),
        "chain_flag": njd_node_get_chain_flag(node),
    }


cdef njd2feature(_njd.NJD* njd):
    cdef _njd.NJDNode* node = njd.head
    features = []
    while node is not NULL:
        features.append(node2feature(node))
        node = node.next
    return features


cdef void feature2njd(_njd.NJD* njd, features) except *:
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
    return {
        "surface": surface_str,
        "features": feature_columns,
        "pos_id": node.posid,
        "left_id": node.lcAttr,
        "right_id": node.rcAttr,
        "word_cost": node.wcost,
        "link_cost": link_cost,
        "node_cost": node.cost,
        "char_span": char_span,
        "is_unknown": is_unknown,
        "is_ignored": is_ignored,
        "dictionary_index": node.dictionary_index,
    }


cdef object _mecab_node_to_cost_candidate(
    mecab_node_t* node,
    Py_ssize_t node_index,
    const char* sentence,
    list byte_to_char_offsets,
    tuple userdic_reading_protection,
) except *:
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

    return {
        "surface": surface_str,
        "features": feature_columns,
        "char_span": char_span,
        "pos_id": node.posid,
        "left_id": node.lcAttr,
        "right_id": node.rcAttr,
        "word_cost": node.wcost,
        "node_cost": node.cost,
        "is_unknown": is_unknown,
        "is_ignored": is_ignored,
        "is_reading_protected": is_reading_protected,
        "dictionary_index": node.dictionary_index,
        "node_index": node_index,
        "node_id": node.id,
    }


# based on Mecab_load in impl. from mecab.cpp
cdef inline int Mecab_load_with_userdic(Mecab *m, char* dicdir, char* userdic) noexcept nogil:
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

def _lock_manager():
    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            # OpenJTalk の C 構造体はインスタンス内で共有されるため、公開メソッド単位で直列化する
            ## ロックをインスタンスごとに分けることで、別インスタンスの nogil 区間は並行実行できる
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
        userdic (bytes): OpenJTalk 用のユーザー辞書のパス (空バイト列の場合は無視される、デフォルトは空)
        userdic_reading_protection (Sequence[bool] | None): 各ユーザー辞書の読み候補をコスト補正から保護するか
            None の場合は全辞書を未保護として扱う。デフォルト: None
    """
    cdef Mecab* mecab
    cdef NJD* njd
    cdef JPCommon* jpcommon
    cdef tuple userdic_reading_protection
    cdef readonly object _lock

    def __cinit__(
        self,
        bytes dn_mecab=b"/usr/local/dic",
        bytes userdic=b"",
        userdic_reading_protection=None,
    ):
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
        if self.mecab != NULL:
            Mecab_clear(self.mecab)
        if self.njd != NULL:
            NJD_clear(self.njd)
        if self.jpcommon != NULL:
            JPCommon_clear(self.jpcommon)

    cdef int _load(self, char* dn_mecab, char* userdic) noexcept nogil:
        return Mecab_load_with_userdic(self.mecab, dn_mecab, userdic)

    def _run_mecab(self, text):
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

            # seperating word with space
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
    def run_mecab(self, text):
        """
        MeCab で形態素解析を実行する。"記号,空白" は除外される。
        全トークン (未知語フラグ・コスト情報含む) が必要な場合は代わりに run_mecab_detailed() を使うこと。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く)
        """
        return self._run_mecab(text)

    def _run_mecab_detailed(self, text):
        """
        MeCab で形態素解析を実行し、フィルタ済み features と全 morphs を同時に返す。
        Mecab_analysis() を呼んだ後、lattice ノードを走査して NJD 用フィルタ済み features と
        未知語フラグ・コスト情報付きの全 morphs を同じ粒度で取得する。
        Haqumei (https://github.com/stellanomia/haqumei) の run_mecab_detailed() に相当する。

        Returns:
            tuple[list[str], list[dict]]: (フィルタ済み features, 全 morphs)
                - features: 記号,空白 を除外した MeCab feature 文字列のリスト (_run_mecab() と同等)
                - morphs: MeCab の形態素解析結果のリスト (各要素は surface, features, pos_id, left_id, right_id, word_cost, is_unknown, is_ignored)
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
                    surface_str = node_morph["surface"]
                    morph_feature_str = ",".join(node_morph["features"][1:])
                    is_unknown = node_morph["is_unknown"]

                    # MeCab が非英数字の連続を未知語へまとめた場合は、辞書由来の記号情報を1文字ずつ復元
                    ## 既知記号を未知語の feature のまま分割すると NJD で通常語として扱われるため、
                    ## 同梱辞書から生成した feature と単語コストを使って既知・未知の判定も戻す
                    should_split_symbol_chunk = (
                        is_unknown is True
                        and len(surface_str) > 1
                        and all(character.isalnum() is False for character in surface_str)
                    )
                    if should_split_symbol_chunk is True:
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

                            # 元ノードの局所コストは先頭文字だけへ割り当て、分割後も合計値を維持する
                            split_link_cost = (
                                node_morph["link_cost"] if character_index == 0 else 0
                            )
                            split_char_start = (
                                node_morph["char_span"][0] + character_index
                            )
                            morphs.append({
                                "surface": character,
                                "features": (character + "," + split_feature).split(","),
                                "pos_id": node.posid,
                                "left_id": split_left_id,
                                "right_id": split_right_id,
                                "word_cost": split_word_cost,
                                "link_cost": split_link_cost,
                                "node_cost": node_morph["node_cost"],
                                "char_span": (split_char_start, split_char_start + 1),
                                "is_unknown": known_symbol is None,
                                "is_ignored": "記号,空白" in split_feature,
                                "dictionary_index": (
                                    0
                                    if known_symbol is not None
                                    else node_morph["dictionary_index"]
                                ),
                            })
                    else:
                        morphs.append(node_morph)
                node = node.next

            return features, morphs
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab_detailed(self, text):
        """
        MeCab の形態素解析結果を未知語フラグ・コスト情報付きで返す。
        通常の run_mecab() と異なり、"記号,空白" もフィルタせずに全トークンを返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[MeCabMorph]: MeCab の形態素解析結果のリスト
        """
        _, morphs = self._run_mecab_detailed(text)
        return morphs

    @_lock_manager()
    def run_mecab_with_cost_adjustments(self, text, cost_adjuster):
        """
        MeCab 候補ノードへ外部モデルの補正コストを加算して one-best の features / morphs を返す。
        cost_adjuster は候補ノード情報の list[MeCabCostCandidate] を受け取り、同じ長さの list[float] を返す呼び出し可能オブジェクト
        candidates には BOS / EOS と無視対象の空白・記号も含まれ、それらに対応する Δc は適用されない。
        適用対象外の候補を含め、cost_adjuster は candidates 全体と同じ長さのリストを返す必要がある。
        Δc は MeCab コスト単位 / 1000 として扱われ、llround(delta * 1000.0) で wcost に加算される
        cost_adjuster 内から同じ OpenJTalk インスタンスの公開メソッドを呼ぶと、非リエントラントなロックでデッドロックする

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            cost_adjuster (Callable[[list[MeCabCostCandidate]], list[float]]): 候補ノードごとの Δc を返す関数

        Returns:
            MeCabCostAdjustedPath: コスト補正後の one-best 解析結果
        """

        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        cdef int text2mecab_result
        cdef int parse_result
        cdef int rebuild_result
        cdef int previous_request_type
        cdef int clipped_node_count = 0
        cdef int stat
        cdef long rounded_delta
        cdef long original_wcost
        cdef long adjusted_wcost
        cdef long base_link_cost
        cdef size_t lattice_size
        cdef size_t pos
        cdef Py_ssize_t node_index
        cdef Py_ssize_t delta_index
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
        cdef list ignored_flags
        cdef list deltas
        cdef list applied_cost_deltas
        cdef object delta
        cdef object candidate
        cdef object morph
        cdef list features
        cdef list morphs
        cdef list selected_node_indices
        cdef list base_link_costs
        cdef long eos_link_cost
        cdef long path_cost = 0
        cdef long base_path_cost = 0

        if callable(cost_adjuster) is False:
            raise TypeError("cost_adjuster must be callable")

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

        if self.mecab.tagger == NULL:
            raise RuntimeError("Failed to access MeCab tagger")
        if self.mecab.lattice == NULL:
            raise RuntimeError("Failed to access MeCab lattice")

        tagger = <mecab_t*> self.mecab.tagger
        lattice = <mecab_lattice_t*> self.mecab.lattice

        # 共有 lattice を one-best 用に切り替え、終了時に呼び出し前の解析モードへ戻す
        previous_request_type = mecab_lattice_get_request_type(lattice)
        with nogil:
            mecab_lattice_set_sentence(lattice, buff)
            mecab_lattice_set_request_type(lattice, MECAB_ONE_BEST)
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
            # 補正除外の判定は cost_adjuster に渡した dict でなく、列挙時に確定した内部リストで行う
            ## コールバックが candidate["is_ignored"] を書き換えても除外条件を崩せない
            ignored_flags = []
            node_index = 0

            # BOS は begin_nodes には含まれないため、候補列の先頭へ明示的に入れる
            ## cost_adjuster には見せるが、制御用ノードなので Δc の加算対象から外す
            bos_node = mecab_lattice_get_bos_node(lattice)
            if bos_node == NULL:
                raise RuntimeError("Failed to access MeCab BOS node")
            candidate = _mecab_node_to_cost_candidate(
                bos_node,
                node_index,
                sentence,
                byte_to_char_offsets,
                self.userdic_reading_protection,
            )
            candidates.append(candidate)
            node_addresses.append(<uintptr_t> bos_node)
            node_index_by_address[<uintptr_t> bos_node] = node_index
            ignored_flags.append(candidate["is_ignored"])
            node_index += 1

            # begin_nodes(pos) を pos 昇順・bnext 順に走査し、タイブレークに関わる列挙順を固定する
            for pos in range(lattice_size + 1):
                node = mecab_lattice_get_begin_nodes(lattice, pos)
                while node != NULL:
                    candidate = _mecab_node_to_cost_candidate(
                        node,
                        node_index,
                        sentence,
                        byte_to_char_offsets,
                        self.userdic_reading_protection,
                    )
                    candidates.append(candidate)
                    node_addresses.append(<uintptr_t> node)
                    node_index_by_address[<uintptr_t> node] = node_index
                    ignored_flags.append(candidate["is_ignored"])
                    node_index += 1
                    node = node.bnext

            deltas = list(cost_adjuster(candidates))
            if len(deltas) != len(candidates):
                raise ValueError("cost_adjuster must return the same number of deltas as candidates")
            applied_cost_deltas = [0] * len(candidates)

            for delta_index in range(len(deltas)):
                node = <mecab_node_t*> <uintptr_t> node_addresses[delta_index]
                stat = node.stat

                # BOS/EOS と OpenJTalk 側で無視する空白は、外部補正で path を変えない
                ## 除外判定は列挙時に確定した ignored_flags を使う (渡した dict の改変に依存しない)
                if stat == 2 or stat == 3 or ignored_flags[delta_index] is True:
                    continue

                delta = deltas[delta_index]
                if isinstance(delta, bool) is True or isinstance(delta, (int, float)) is False:
                    raise TypeError("cost_adjuster deltas must be float")
                # C の llround() は NaN と無限大を整数へ変換できないため、Python 側の値を先に拒否する
                if isfinite(float(delta)) == 0:
                    raise ValueError("cost_adjuster deltas must be finite")
                # 1000倍後の値が C long を超える場合も llround() の結果が未定義になるため拒否する
                if float(delta) >= LONG_MAX / 1000.0 or float(delta) <= LONG_MIN / 1000.0:
                    raise ValueError("cost_adjuster deltas are too large")

                rounded_delta = llround(float(delta) * 1000.0)
                original_wcost = <long> node.wcost

                # 加算前に short の境界と比較し、符号付き long のオーバーフローを起こさず飽和させる
                if rounded_delta > SHRT_MAX - original_wcost:
                    adjusted_wcost = SHRT_MAX
                    clipped_node_count += 1
                elif rounded_delta < SHRT_MIN - original_wcost:
                    adjusted_wcost = SHRT_MIN
                    clipped_node_count += 1
                else:
                    adjusted_wcost = original_wcost + rounded_delta
                # クリップ後に実際に加わった整数コストを残し、選択 path の補正前コストを正確に復元する
                applied_cost_deltas[delta_index] = adjusted_wcost - original_wcost
                node.wcost = <short> adjusted_wcost

            with nogil:
                rebuild_result = mecab_lattice_rebuild_best(tagger, lattice)
            if rebuild_result != 1:
                raise RuntimeError("Failed to rebuild MeCab best path")

            features = []
            morphs = []
            selected_node_indices = []
            base_link_costs = []
            node = mecab_lattice_get_bos_node(lattice)
            if node == NULL:
                raise RuntimeError("Failed to access rebuilt MeCab BOS node")

            while node != NULL:
                stat = node.stat

                # EOS 自体は返却しないが、最終形態素から EOS への遷移は総コストへ含める
                if stat == 3:
                    eos_link_cost = _selected_mecab_link_cost(node, True)
                    path_cost += eos_link_cost
                    base_path_cost += eos_link_cost

                # BOS/EOS は制御用ノードなので、返却する morphs から外す
                if stat != 2 and stat != 3:
                    node_index = node_index_by_address[<uintptr_t> node]
                    selected_node_indices.append(node_index)
                    morph = _mecab_node_to_morph(
                        node,
                        True,
                        sentence,
                        byte_to_char_offsets,
                    )
                    morphs.append(morph)
                    path_cost += morph["link_cost"]
                    # 接続コストは不変なので、link_cost から実適用 wcost 差分を引けば補正前値へ戻せる
                    base_link_cost = morph["link_cost"] - applied_cost_deltas[node_index]
                    base_link_costs.append(base_link_cost)
                    base_path_cost += base_link_cost
                    if morph["is_ignored"] is False:
                        features.append(",".join(morph["features"]))
                node = node.next

            return {
                "features": features,
                "morphs": morphs,
                "node_indices": selected_node_indices,
                "path_cost": path_cost,
                "base_link_costs": base_link_costs,
                "base_path_cost": base_path_cost,
                "clipped_node_count": clipped_node_count,
            }
        finally:
            mecab_lattice_set_request_type(lattice, previous_request_type)
            Mecab_refresh(self.mecab)

    def _run_mecab_nbest_features(self, text, max_paths=5):
        """
        MeCab の n-best 候補を、NJD に渡せる features と詳細 morphs の組として返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)

        Returns:
            list[MeCabNBestPath]: 各候補パスの features / morphs / path_cost
        """

        cdef char buff[TEXT2MECAB_BUFFER_SIZE]
        cdef int result
        cdef int init_result
        cdef int path_index
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
            for path_index in range(max_paths):
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

                paths.append({
                    "features": features,
                    "morphs": morphs,
                    "path_cost": path_cost,
                })
            return paths
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab_nbest_features(self, text, max_paths=5):
        """
        MeCab の n-best 候補を features / morphs / path_cost 付きで返す。
        features は run_njd_from_mecab() に渡せる形式で、morphs は run_mecab_detailed() と同じ詳細形式を持つ

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)

        Returns:
            list[MeCabNBestPath]: MeCab n-best 候補パスのリスト
        """
        return self._run_mecab_nbest_features(text, max_paths)

    def _run_njd_from_mecab(self, mecab_features):
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
    def run_njd_from_mecab(self, mecab_features):
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
    def run_frontend(self, text):
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
    def run_frontend_detailed(self, text):
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
    def extract_phonemes(self, features):
        """
        NJD features からフラットな音素列を直接抽出する。
        HTS フルコンテキストラベル文字列は生成せず、JPCommonLabel の音素連結リストをそのまま走査する。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フラットな音素列
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
    def make_label(self, features):
        """
        HTS 音声合成用のフルコンテキストラベルを返す。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フルコンテキストラベル文字列のリスト
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
    def make_phoneme_mapping(self, features):
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

                mapping.append({
                    "surface": feat["string"],
                    "phonemes": phonemes,
                    "pos": feat["pos"],
                    "pos_group1": feat["pos_group1"],
                    "pos_group2": feat["pos_group2"],
                    "pos_group3": feat["pos_group3"],
                    "ctype": feat["ctype"],
                    "cform": feat["cform"],
                    "orig": feat["orig"],
                    "read": feat["read"],
                    "pron": feat["pron"],
                    "accent_nucleus": feat["acc"],
                    "mora_count": feat["mora_size"],
                    "chain_rule": feat["chain_rule"],
                    "chain_flag": feat["chain_flag"],
                })

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

    def g2p(self, text, kana=False, join=True):
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

    def __dealloc__(self):
        self._clear()
        if self.mecab != NULL:
            del self.mecab
        if self.njd != NULL:
            del self.njd
        if self.jpcommon != NULL:
            del self.jpcommon

def mecab_dict_index(bytes dn_mecab, bytes path, bytes out_path):
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

def build_mecab_dictionary(bytes dn_mecab):
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

def apply_original_rule_before_chaining(njd_features):
    """
    NJD features に chaining 前の独自ルールを適用する。内部用。
    サ変接続・接頭語・動詞連続・連用形・助動詞などのアクセント結合規則を適用する。

    Args:
        njd_features (list[dict]): NJDNode 用 features 。インプレースで更新される

    Returns:
        list[dict]: 更新後の njd_features（同一オブジェクト）
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
