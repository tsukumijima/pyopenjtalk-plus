# distutils: language = c++
# cython: language_level=3

from libc.stdio cimport FILE

cdef extern from "jpcommon.h":
    cdef cppclass JPCommonNode:
        char *pron
        char *pos
        char *ctype
        char *cform
        int acc
        int chain_flag
        void *prev
        void *next

    # jpcommon.h に定義されている JPCommonLabel 階層構造体
    # 階層: Word → Mora → Phoneme
    # 各構造体の up ポインタは親階層を指す
    # JPCommonLabelAccentPhrase と JPCommonLabelBreathGroup は今回不要なので宣言しない
    # C ヘッダでは typedef struct _X { ... } X; パターンで定義されているため、
    # Cython では ctypedef struct を使う (cdef struct だと C++ モードで typedef 名と衝突する)
    ctypedef struct JPCommonLabelPhoneme:
        char *phoneme
        JPCommonLabelPhoneme *prev
        JPCommonLabelPhoneme *next
        JPCommonLabelMora *up  # 親 Mora への上方向ポインタ

    ctypedef struct JPCommonLabelMora:
        char *mora
        JPCommonLabelPhoneme *head
        JPCommonLabelPhoneme *tail
        JPCommonLabelMora *prev
        JPCommonLabelMora *next
        JPCommonLabelWord *up  # 親 Word への上方向ポインタ

    ctypedef struct JPCommonLabelWord:
        char *pron
        char *pos
        char *ctype
        char *cform
        JPCommonLabelMora *head
        JPCommonLabelMora *tail
        JPCommonLabelWord *prev
        JPCommonLabelWord *next
        # up (JPCommonLabelAccentPhrase*) は今回不要なので省略

    ctypedef struct JPCommonLabel:
        int size
        char **feature
        int is_valid
        JPCommonLabelWord *word_head
        JPCommonLabelWord *word_tail
        JPCommonLabelPhoneme *phoneme_head
        JPCommonLabelPhoneme *phoneme_tail
        int short_pause_flag

    cdef cppclass JPCommon:
        JPCommonNode *head
        JPCommonNode *tail
        JPCommonLabel *label  # void* → JPCommonLabel* に型を変更

    void JPCommon_initialize(JPCommon * jpcommon) nogil
    void JPCommon_push(JPCommon * jpcommon, JPCommonNode * node)
    void JPCommon_make_label(JPCommon * jpcommon) nogil
    int JPCommon_get_label_size(JPCommon * jpcommon) nogil
    char **JPCommon_get_label_feature(JPCommon * jpcommon) nogil
    void JPCommon_print(JPCommon * jpcommon)
    void JPCommon_fprint(JPCommon * jpcommon, FILE * fp)
    void JPCommon_refresh(JPCommon * jpcommon)
    void JPCommon_clear(JPCommon * jpcommon) nogil

    # JPCommonLabel 操作関数
    # JPCommon_make_label() を分解して個別に呼び出すために使用する
    # (Haqumei の prepare_jpcommon_label_internal 相当)
    void JPCommonLabel_initialize(JPCommonLabel * label)
    void JPCommonLabel_push_word(
        JPCommonLabel * label,
        const char *pron,
        const char *pos,
        const char *ctype,
        const char *cform,
        int acc,
        int chain_flag,
    )
    void JPCommonLabel_make(JPCommonLabel * label)
    void JPCommonLabel_clear(JPCommonLabel * label)

    # JPCommonNode アクセサ関数
    const char *JPCommonNode_get_pron(JPCommonNode * node)
    const char *JPCommonNode_get_pos(JPCommonNode * node)
    const char *JPCommonNode_get_ctype(JPCommonNode * node)
    const char *JPCommonNode_get_cform(JPCommonNode * node)
    int JPCommonNode_get_acc(JPCommonNode * node)
    int JPCommonNode_get_chain_flag(JPCommonNode * node)
