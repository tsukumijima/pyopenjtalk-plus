# distutils: language = c++
# cython: language_level=3

cdef extern from "mecab.h":
    cdef enum:
        MECAB_ONE_BEST

    cdef struct mecab_t:
        pass

    cdef cppclass Mecab:
        char **feature
        int size
        void *model
        void *tagger
        void *lattice

    # MeCab の Lattice ノード構造体
    # n-best では best path 以外のノードも読むため、mecab.h と同じ順序で必要フィールドを宣言する
    # Cython は node->field 形式の C コードを出すため、ここで宣言したフィールド名が mecab.h に存在する必要がある
    cdef struct mecab_node_t:
        mecab_node_t *prev
        mecab_node_t *next
        mecab_node_t *enext
        mecab_node_t *bnext
        mecab_path_t *rpath
        mecab_path_t *lpath
        const char *surface    # null 終端ではない。length バイト分のみ有効
        const char *feature    # null 終端の feature 文字列
        unsigned int id
        unsigned short length  # surface のバイト長
        unsigned short rlength # surface の前の空白を含むバイト長
        unsigned short rcAttr  # 右文脈 ID (right-id.def で定義)
        unsigned short lcAttr  # 左文脈 ID (left-id.def で定義)
        unsigned short posid   # 品詞 ID (pos-id.def で定義。文脈 ID とは別の粗い分類)
        unsigned char char_type
        unsigned char stat     # 0=NOR, 1=UNK, 2=BOS, 3=EOS, 4=EON
        unsigned char isbest
        unsigned char dictionary_index  # 0=system, 1..N=userdic 読込順, 255=unknown/control
        float alpha
        float beta
        float prob
        short wcost            # 単語コスト (辞書に登録されたコスト)
        long cost              # BOS からこのノードまでの累積コスト

    # MeCab の接続経路構造体
    cdef struct mecab_path_t:
        mecab_node_t *rnode
        mecab_path_t *rnext
        mecab_node_t *lnode
        mecab_path_t *lnext
        int cost
        float prob

    # mecab_lattice_t は opaque struct として宣言
    # void* ではなく struct として宣言しないと、mecab_lattice_t* が void** になってしまう
    cdef struct mecab_lattice_t:
        pass

    int mecab_parse_lattice(mecab_t *mecab, mecab_lattice_t *lattice) nogil
    int mecab_lattice_rebuild_best(mecab_t *mecab, mecab_lattice_t *lattice) nogil
    int mecab_nbest_init2(mecab_t *mecab, const char *str, size_t len) nogil
    const mecab_node_t *mecab_nbest_next_tonode(mecab_t *mecab) nogil
    void mecab_lattice_clear(mecab_lattice_t *lattice) nogil
    void mecab_lattice_set_sentence(mecab_lattice_t *lattice, const char *sentence) nogil
    int mecab_lattice_get_request_type(mecab_lattice_t *lattice) nogil
    void mecab_lattice_set_request_type(mecab_lattice_t *lattice, int request_type) nogil
    mecab_node_t *mecab_lattice_get_begin_nodes(mecab_lattice_t *lattice, size_t pos) nogil
    const char *mecab_lattice_get_sentence(mecab_lattice_t *lattice) nogil
    size_t mecab_lattice_get_size(mecab_lattice_t *lattice) nogil

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
