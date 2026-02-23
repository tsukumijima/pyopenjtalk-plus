# distutils: language = c++
# cython: language_level=3

cdef extern from "text2mecab.h":
    ctypedef enum text2mecab_result_t:
        TEXT2MECAB_RESULT_SUCCESS
        TEXT2MECAB_RESULT_INVALID_ARGUMENT
        TEXT2MECAB_RESULT_RANGE_ERROR

    text2mecab_result_t text2mecab(char *output, size_t sizeOfOutput, const char *input) nogil
