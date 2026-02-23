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
from .openjtalk.njd cimport NJD, NJD_initialize, NJD_refresh, NJD_clear
from .openjtalk cimport njd as _njd
from .openjtalk.jpcommon cimport JPCommon, JPCommon_initialize,JPCommon_make_label
from .openjtalk.jpcommon cimport JPCommon_get_label_size, JPCommon_get_label_feature
from .openjtalk.jpcommon cimport JPCommon_refresh, JPCommon_clear
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
    """OpenJTalk

    Args:
        dn_mecab (bytes): Dictionary path for MeCab.
        userdic (bytes): Dictionary path for MeCab userdic.
            This option is ignored when empty bytestring is given.
            Default is empty.
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
                m = (<bytes>(mecab_morphs[i])).decode('utf-8')
                if '記号,空白' not in m:
                    morphs.append(m)
            return morphs
        finally:
            Mecab_refresh(self.mecab)

    @_lock_manager()
    def run_mecab(self, text):
        """Run MeCab analysis and return features

        Args:
            text: Input text

        Returns:
            list[str]: List of MeCab features
        """
        return self._run_mecab(text)

    def _run_njd_from_mecab(self, mecab_features):
        # if empty list, return empty list
        new_size = len(mecab_features)
        if new_size == 0:
            return []

        for mecab_feature in mecab_features:
            if isinstance(mecab_feature, str) is False:
                raise TypeError("Each MeCab feature must be str")
            if '\x00' in mecab_feature:
                raise ValueError("MeCab feature must not contain null characters")

        byte_morphs = [m.encode('utf-8')+b'\x00' for m in mecab_features]
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
        """Run NJD processing from MeCab features

        Args:
            mecab_features: List of MeCab features (strings)

        Returns:
            list[dict]: List of NJD features
        """
        return self._run_njd_from_mecab(mecab_features)

    @_lock_manager()
    def run_frontend(self, text):
        """Run OpenJTalk's text processing frontend

        Args:
            text: Input text

        Returns:
            List[dict]: List of NJD features
        """
        mecab_features = self._run_mecab(text)
        return self._run_njd_from_mecab(mecab_features)

    @_lock_manager()
    def make_label(self, features):
        """Make full-context label
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

    def g2p(self, text, kana=False, join=True):
        """Grapheme-to-phoeneme (G2P) conversion (Cython implementation)
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
    for i, njd in enumerate(njd_features[:-1]):
        # サ変動詞(スル)の前にサ変接続や名詞が来た場合は、一つのアクセント句に纏める
        if (njd["pos_group1"] in ["サ変接続", "格助詞", "接続助詞"] or (njd["pos"] == "名詞" and njd["pos_group1"] == "一般") or njd["pos"] == "副詞" ) and njd_features[i+1]["ctype"] == "サ変・スル":
            njd_features[i+1]["chain_flag"] = 1
        # ご遠慮、ご配慮のような接頭語がつく場合にその後に続く単語の結合則を変更する
        if (njd["string"] in ["お","御","ご"] and njd["chain_rule"] == "P1"):
            if njd_features[i+1]["acc"] == 0 or njd_features[i+1]["acc"] == njd_features[i+1]["mora_size"]:
                njd_features[i+1]['chain_rule'] = "C4"
                njd_features[i+1]["acc"] = 0
            else:
                njd_features[i+1]['chain_rule'] = "C1"
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
