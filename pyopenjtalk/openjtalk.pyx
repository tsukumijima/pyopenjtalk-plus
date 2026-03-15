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
from contextlib import contextmanager
from threading import Lock

cimport numpy as np
np.import_array()

from libc.stdlib cimport calloc
from libc.string cimport strlen
from libc.stdint cimport *

from .openjtalk.mecab cimport Mecab, Mecab_initialize, Mecab_load, Mecab_analysis
from .openjtalk.mecab cimport Mecab_get_feature, Mecab_get_size, Mecab_refresh, Mecab_clear
from .openjtalk.mecab cimport createModel, Model, Tagger, Lattice
from .openjtalk.mecab cimport mecab_dict_index as _mecab_dict_index
from .openjtalk.mecab cimport (
    mecab_node_t,
    mecab_lattice_t,
    mecab_lattice_get_bos_node,
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

def _generate_lock_manager():
    lock = Lock()

    @contextmanager
    def f():
        with lock:
            yield

    return f


cdef class OpenJTalk:
    """
    OpenJTalk のテキスト処理フロントエンドの Cython 実装。
    通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス。
        userdic (bytes): MeCab ユーザー辞書のパス。空バイト列の場合は無視される。デフォルトは空。
    """
    cdef Mecab* mecab
    cdef NJD* njd
    cdef JPCommon* jpcommon
    _lock_manager = _generate_lock_manager()

    def __cinit__(self, bytes dn_mecab=b"/usr/local/dic", bytes userdic=b""):
        cdef char* _dn_mecab = dn_mecab
        cdef char* _userdic = userdic

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
        Mecab_clear(self.mecab)
        NJD_clear(self.njd)
        JPCommon_clear(self.jpcommon)

    cdef int _load(self, char* dn_mecab, char* userdic) noexcept nogil:
        return Mecab_load_with_userdic(self.mecab, dn_mecab, userdic)

    def _run_mecab(self, text):
        cdef char buff[8192]
        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        cdef int result
        with nogil:
            result = text2mecab(buff, 8192, _text)
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
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く) 。
        """
        return self._run_mecab(text)

    def _run_mecab_detailed(self, text):
        """
        MeCab で形態素解析を実行し、フィルタ済み features と全 morphs を同時に返す。
        Mecab_analysis() を呼んだ後、Mecab_get_feature() で NJD 用フィルタ済み features を取得し、
        同時に lattice ノードを走査して未知語フラグ・コスト情報付きの全 morphs も取得する。
        Haqumei (https://github.com/stellanomia/haqumei) の run_mecab_detailed() に相当する。

        Returns:
            tuple[list[str], list[dict]]: (フィルタ済み features, 全 morphs)
                - features: 記号,空白 を除外した MeCab feature 文字列のリスト (_run_mecab() と同等)
                - morphs: MeCab の形態素解析結果のリスト (各要素は surface, feature, is_unknown, is_ignored, pos_id, word_cost, cost)
        """

        cdef char buff[8192]
        # cdef 宣言は関数スコープの先頭でなければならないため、ここで事前宣言する
        cdef mecab_lattice_t* lattice = NULL
        cdef mecab_node_t* node
        cdef int morph_size
        cdef char** mecab_feature_array
        cdef int analysis_result

        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        cdef int result
        with nogil:
            result = text2mecab(buff, 8192, _text)
        if result != 0:
            if result == TEXT2MECAB_RESULT_INVALID_ARGUMENT:
                raise RuntimeError("Invalid arguments for text2mecab")
            if result == TEXT2MECAB_RESULT_RANGE_ERROR:
                raise RuntimeError("Input text is too long after normalization")
            raise RuntimeError("Unknown text2mecab error: " + str(result))

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

            # 1) Mecab_get_feature() から NJD 用のフィルタ済み features を構築
            #    (_run_mecab() と同等のロジック)
            features = []
            for i in range(morph_size):
                if mecab_feature_array[i] == NULL:
                    raise RuntimeError("MeCab returned null morph entry")
                feature_str = (<bytes>(mecab_feature_array[i])).decode("utf-8")
                if "記号,空白" not in feature_str:
                    features.append(feature_str)

            # 2) lattice ノードを走査して MeCabMorph リストを構築
            #    Mecab_analysis() 後、lattice ノードは Mecab_refresh() まで有効
            if self.mecab.lattice == NULL:
                raise RuntimeError("Failed to access MeCab lattice")
            lattice = <mecab_lattice_t*> self.mecab.lattice
            node = mecab_lattice_get_bos_node(lattice)

            morphs = []
            while node != NULL:
                stat = node.stat
                # BOS (stat=2), EOS (stat=3) ノードはスキップ
                if stat != 2 and stat != 3:
                    # surface は null 終端ではないので length 分だけ読む
                    # c_string_encoding=ascii ディレクティブの影響を避けるため明示的にスライスしてから decode する
                    # NULL チェック + length > 0 を確認
                    if node.surface != NULL and node.length > 0:
                        surface_bytes = (<char*>node.surface)[:node.length]
                        surface_str = surface_bytes.decode("utf-8", errors="replace")
                    else:
                        surface_str = ""

                    # feature の NULL チェック
                    if node.feature != NULL:
                        feature_bytes = <bytes> node.feature
                        morph_feature_str = feature_bytes.decode("utf-8", errors="replace")
                    else:
                        morph_feature_str = ""

                    is_unknown = (stat == 1)  # MECAB_UNK_NODE
                    is_ignored = "記号,空白" in morph_feature_str

                    # 既存 run_mecab() と同じ "surface,feature" フォーマットで feature を構築
                    morphs.append({
                        "surface": surface_str,
                        "feature": surface_str + "," + morph_feature_str,
                        "pos_id": node.posid,
                        "left_id": node.lcAttr,
                        "right_id": node.rcAttr,
                        "word_cost": node.wcost,
                        "is_unknown": is_unknown,
                        "is_ignored": is_ignored,
                    })
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
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[MeCabMorph]: MeCab の形態素解析結果のリスト。
        """
        _, morphs = self._run_mecab_detailed(text)
        return morphs

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
            mecab_features (list[str]): MeCab の feature 文字列のリスト。

        Returns:
            list[NJDFeature]: NJDNode 用 features 。
        """
        return self._run_njd_from_mecab(mecab_features)

    def run_frontend(self, text):
        """
        OpenJTalk のテキスト処理フロントエンドを実行する。
        run_frontend_detailed() に委譲し、NJD features のみを返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[NJDFeature]: NJDNode 用 features 。
        """
        njd_features, _ = self.run_frontend_detailed(text)
        return njd_features

    @_lock_manager()
    def run_frontend_detailed(self, text):
        """
        OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
        MeCab 解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
                NJD features は run_frontend() と、MeCab morphs は run_mecab_detailed() と同一の結果。
        """
        features, morphs = self._run_mecab_detailed(text)
        njd_features = self._run_njd_from_mecab(features)
        return njd_features, morphs

    @_lock_manager()
    def make_label(self, features):
        """
        HTS 音声合成用のフルコンテキストラベルを返す。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値) 。

        Returns:
            list[str]: フルコンテキストラベル文字列のリスト。
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
        ポーズ記号 ("、"/"？"/"！") は常に ['pau'] として保持する。
        長音吸収マージにより、戻り値の長さが入力と異なる場合がある。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値) 。

        Returns:
            list[dict[str, Any]]: 各要素は {'word': str, 'phonemes': list[str]} の辞書。
                is_unknown / is_ignored が必要な場合は pyopenjtalk.make_phoneme_mapping() を使用すること。

        Raises:
            RuntimeError: JPCommonLabel の内部アロケーション失敗時 。
        """

        # cdef 宣言は関数スコープの先頭でなければならないため、ここで事前宣言する
        cdef JPCommonLabelWord* prev_word_tail
        cdef JPCommonLabelWord* curr_word_tail
        cdef JPCommonLabelPhoneme* phoneme_node
        cdef JPCommonLabelMora* mora_ptr
        cdef JPCommonLabelWord* word_ptr
        cdef JPCommonNode* node

        # features を複数回イテレーションする (feature2njd, ポーズカウント, メインループ) ため、
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
            #   - ポーズ形態素 (pron が "、"/"？"/"！"): フラグのみ設定
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
            mapping = []
            for feat in features:
                mapping.append({"word": feat["string"], "phonemes": []})

            # ポーズ形態素 (pron が "、"/"？"/"！") に ["pau"] をアサイン
            pause_count = 0
            for f_idx in range(len(features)):
                pron = features[f_idx]["pron"]
                if pron in ("、", "？", "！"):
                    mapping[f_idx]["phonemes"] = ["pau"]
                    pause_count += 1

            # phoneme を走査し、Phoneme → Mora → Word の階層から feature index を特定
            phoneme_node = self.jpcommon.label.phoneme_head
            while phoneme_node != NULL:
                if phoneme_node.phoneme != NULL:
                    phoneme_str = (<bytes> phoneme_node.phoneme).decode("utf-8")

                    # "pau" はポーズ形態素で既にアサイン済みのためスキップ
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
            # これらを前方の Word に word テキストを結合する
            needs_merge = len(features) > len(ptr_to_idx) + pause_count
            if needs_merge is True:
                merged = []
                for entry in mapping:
                    # 空音素の要素を前方に結合する
                    if len(merged) > 0 and len(entry["phonemes"]) == 0:
                        prev = merged[-1]
                        # 前方が ["pau"] の場合は結合しない
                        is_prev_pause = (len(prev["phonemes"]) == 1 and prev["phonemes"][0] == "pau")
                        if is_prev_pause is False:
                            prev["word"] += entry["word"]
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
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。
            kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式。デフォルトは False 。
            join (bool): True の場合、音素またはカタカナを単一の文字列に連結する。デフォルトは True 。

        Returns:
            str | list[str]: kana と join の組み合わせにより、str または list[str] を返す。
        """
        njd_features = self.run_frontend(text)

        if not kana:
            labels = self.make_label(njd_features)
            prons = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))
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
        del self.mecab
        del self.njd
        del self.jpcommon

def mecab_dict_index(bytes dn_mecab, bytes path, bytes out_path):
    """
    OpenJTalk 用のユーザー辞書を CSV からビルドする。低レベル API 。
    通常は pyopenjtalk.mecab_dict_index() を使用すること。
    CSV は naist-jdic 互換の品詞体系で記述する必要がある。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス。
        path (bytes): ユーザー CSV ファイルのパス。
        out_path (bytes): 出力辞書ファイルのパス。

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗) 。
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
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス。

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗) 。
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
        njd_features (list[dict]): NJDNode 用 features 。インプレースで更新される。

    Returns:
        list[dict]: 更新後の njd_features（同一オブジェクト）。
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
