# distutils: language = c++
# cython: language_level=3

cdef extern from "mecab.h":
    cdef struct mecab_t:
        pass

    cdef cppclass Mecab:
        char **feature
        int size
        void *model
        void *tagger
        void *lattice

    # MeCab の lattice ノード構造体
    # mecab.h L139-257 に定義されている
    # Cython の cdef extern 構造体宣言では参照するフィールドのみ宣言すれば十分なため、
    # enext, bnext, rpath, lpath, id, char_type, isbest, alpha, beta, prob は省略している
    cdef struct mecab_node_t:
        mecab_node_t *prev
        mecab_node_t *next
        const char *surface    # null 終端ではない。length バイト分のみ有効
        const char *feature    # null 終端の feature 文字列
        unsigned short length  # surface のバイト長
        unsigned short rlength # surface の前の空白を含むバイト長 (rlength - length で空白量が分かる)
        unsigned short rcAttr  # 右文脈 ID (ipadic ベースの辞書では品詞 ID として機能する)
        unsigned short lcAttr  # 左文脈 ID (ipadic ベースの辞書では rcAttr と同一値)
        unsigned short posid   # 品詞 ID (pos-id.def で定義)
        unsigned char stat     # 0=NOR, 1=UNK, 2=BOS, 3=EOS, 4=EON
        short wcost            # 単語コスト (辞書に登録されたコスト。低いほど出現しやすい)
        long cost              # BOS からこのノードまでの最小累積コスト

    # mecab_lattice_t は opaque struct として宣言
    # void* ではなく struct として宣言しないと、mecab_lattice_t* が void** になってしまう
    cdef struct mecab_lattice_t:
        pass

    int mecab_parse_lattice(mecab_t *mecab, mecab_lattice_t *lattice) nogil
    void mecab_lattice_clear(mecab_lattice_t *lattice) nogil
    void mecab_lattice_set_sentence(mecab_lattice_t *lattice, const char *sentence) nogil

    # mecab.h L590: C wrapper of MeCab::Lattice::bos_node()
    mecab_node_t *mecab_lattice_get_bos_node(mecab_lattice_t *lattice) nogil

    cdef int Mecab_initialize(Mecab *m) nogil
    cdef int Mecab_load(Mecab *m, const char *dicdir) nogil
    cdef int Mecab_analysis(Mecab *m, const char *str) nogil
    cdef int Mecab_print(Mecab *m)
    int Mecab_get_size(Mecab *m) nogil
    char **Mecab_get_feature(Mecab *m) nogil
    cdef int Mecab_refresh(Mecab *m) nogil
    cdef int Mecab_clear(Mecab *m) nogil
    cdef int mecab_dict_index(int argc, char **argv) nogil

cdef extern from "mecab.h" namespace "MeCab":
    cdef cppclass Tagger:
        pass
    cdef cppclass Lattice:
        pass
    cdef cppclass Model:
        Tagger *createTagger() nogil
        Lattice *createLattice() nogil
    cdef Model *createModel(int argc, char **argv) nogil
