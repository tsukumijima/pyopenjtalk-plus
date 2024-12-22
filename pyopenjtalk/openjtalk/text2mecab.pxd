# distutils: language = c++
# cython: language_level=3

cdef extern from "text2mecab.h":
    int text2mecab(char *output, size_t sizeOfOutput, const char *input) nogil
