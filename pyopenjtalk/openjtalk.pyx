# coding: utf-8
# cython: boundscheck=True, wraparound=True
# cython: c_string_type=unicode, c_string_encoding=ascii
# cython: language_level=3

import errno
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
from .openjtalk.text2mecab cimport text2mecab
from .openjtalk.mecab2njd cimport mecab2njd
from .openjtalk.njd2jpcommon cimport njd2jpcommon

cdef njd_node_get_string(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_string(node))).decode("utf-8")

cdef njd_node_get_pos(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_pos(node))).decode("utf-8")

cdef njd_node_get_pos_group1(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_pos_group1(node))).decode("utf-8")

cdef njd_node_get_pos_group2(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_pos_group2(node))).decode("utf-8")

cdef njd_node_get_pos_group3(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_pos_group3(node))).decode("utf-8")

cdef njd_node_get_ctype(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_ctype(node))).decode("utf-8")

cdef njd_node_get_cform(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_cform(node))).decode("utf-8")

cdef njd_node_get_orig(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_orig(node))).decode("utf-8")

cdef njd_node_get_read(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_read(node))).decode("utf-8")

cdef njd_node_get_pron(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_pron(node))).decode("utf-8")

cdef int njd_node_get_acc(_njd.NJDNode* node) noexcept:
    return _njd.NJDNode_get_acc(node)

cdef int njd_node_get_mora_size(_njd.NJDNode* node) noexcept:
    return _njd.NJDNode_get_mora_size(node)

cdef njd_node_get_chain_rule(_njd.NJDNode* node):
    return (<bytes>(_njd.NJDNode_get_chain_rule(node))).decode("utf-8")

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


cdef void feature2njd(_njd.NJD* njd, features):
    cdef _njd.NJDNode* node

    for feature_node in features:
        node = <_njd.NJDNode *> calloc(1, sizeof(_njd.NJDNode))
        _njd.NJDNode_initialize(node)
        # set values
        _njd.NJDNode_set_string(node, feature_node["string"].encode("utf-8"))
        _njd.NJDNode_set_pos(node, feature_node["pos"].encode("utf-8"))
        _njd.NJDNode_set_pos_group1(node, feature_node["pos_group1"].encode("utf-8"))
        _njd.NJDNode_set_pos_group2(node, feature_node["pos_group2"].encode("utf-8"))
        _njd.NJDNode_set_pos_group3(node, feature_node["pos_group3"].encode("utf-8"))
        _njd.NJDNode_set_ctype(node, feature_node["ctype"].encode("utf-8"))
        _njd.NJDNode_set_cform(node, feature_node["cform"].encode("utf-8"))
        _njd.NJDNode_set_orig(node, feature_node["orig"].encode("utf-8"))
        _njd.NJDNode_set_read(node, feature_node["read"].encode("utf-8"))
        _njd.NJDNode_set_pron(node, feature_node["pron"].encode("utf-8"))
        _njd.NJDNode_set_acc(node, feature_node["acc"])
        _njd.NJDNode_set_mora_size(node, feature_node["mora_size"])
        _njd.NJDNode_set_chain_rule(node, feature_node["chain_rule"].encode("utf-8"))
        _njd.NJDNode_set_chain_flag(node, feature_node["chain_flag"])
        _njd.NJD_push_node(njd, node)

# based on Mecab_load in impl. from mecab.cpp
cdef inline int Mecab_load_with_userdic(Mecab *m, char* dicdir, char* userdic) noexcept nogil:
    if userdic == NULL or strlen(userdic) == 0:
        return Mecab_load(m, dicdir)

    if m == NULL or dicdir == NULL or strlen(dicdir) == 0:
        return 0

    Mecab_clear(m)

    cdef (char*)[5] argv = ["mecab", "-d", dicdir, "-u", userdic]
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
                raise RuntimeError("Failed to initalize Mecab")

    cdef void _clear(self) noexcept nogil:
        Mecab_clear(self.mecab)
        NJD_clear(self.njd)
        JPCommon_clear(self.jpcommon)

    cdef int _load(self, char* dn_mecab, char* userdic) noexcept nogil:
        return Mecab_load_with_userdic(self.mecab, dn_mecab, userdic)

    @_lock_manager()
    def run_frontend(self, text, kansai:bool = False):
        """Run OpenJTalk's text processing frontend
        """
        cdef char buff[8192]
        if isinstance(text, str):
            text = text.encode("utf-8")

        cdef const char* _text = text
        cdef int result
        with nogil:
            result = text2mecab(buff, 8192, _text)
        if result != 0:
            if result == errno.ERANGE:
                raise RuntimeError("Text is too long")
            if result == errno.EINVAL:
                raise RuntimeError("Invalid input for text2mecab")
            raise RuntimeError("Unknown error: " + str(result))

        cdef int morph_size
        cdef char** mecab_morphs
        with nogil:
            Mecab_analysis(self.mecab, buff)

            morph_size = Mecab_get_size(self.mecab)
            mecab_morphs = Mecab_get_feature(self.mecab)

        # seperating word with space
        morphs = []
        cdef int new_size = 0
        for i in range(morph_size):
            m = (<bytes>(mecab_morphs[i])).decode('utf-8')
            if '記号,空白' not in m:
                morphs.append(m)
                new_size = new_size + 1

        # if empty string, return empty list
        if new_size == 0:
            return []

        byte_morphs = [m.encode('utf-8')+b'\x00' for m in morphs]
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

        if kansai:
            with nogil:
                _njd.njd_set_digit(self.njd)
                _njd.njd_set_unvoiced_vowel(self.njd)
                _njd.njd_set_long_vowel(self.njd)
        else:
            with nogil:
                _njd.njd_set_digit(self.njd)
                _njd.njd_set_accent_phrase(self.njd)
                _njd.njd_set_accent_type(self.njd)
                _njd.njd_set_unvoiced_vowel(self.njd)
                _njd.njd_set_long_vowel(self.njd)
        feature = njd2feature(self.njd)

        # Note that this will release memory for njd feature
        NJD_refresh(self.njd)
        Mecab_refresh(self.mecab)

        return feature

    @_lock_manager()
    def make_label(self, features):
        """Make full-context label
        """
        feature2njd(self.njd, features)
        with nogil:
            njd2jpcommon(self.jpcommon, self.njd)

            JPCommon_make_label(self.jpcommon)

            label_size = JPCommon_get_label_size(self.jpcommon)
            label_feature = JPCommon_get_label_feature(self.jpcommon)

        labels = []
        for i in range(label_size):
            # This will create a copy of c string
            # http://cython.readthedocs.io/en/latest/src/tutorial/strings.html
            labels.append(<unicode>label_feature[i])

        # Note that this will release memory for label feature
        JPCommon_refresh(self.jpcommon)
        NJD_refresh(self.njd)

        return labels

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
    cdef (char*)[10] argv = [
        "mecab-dict-index",
        "-d",
        dn_mecab,
        "-u",
        out_path,
        "-f",
        "utf-8",
        "-t",
        "utf-8",
        path
    ]
    cdef int ret
    with nogil:
        ret = _mecab_dict_index(10, argv)
    return ret

def build_mecab_dictionary(bytes dn_mecab):
    cdef (char*)[9] argv = [
        "mecab-dict-index",
        "-d",
        dn_mecab,
        "-o",
        dn_mecab,
        "-f",
        "utf-8",
        "-t",
        "utf-8",
    ]
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
