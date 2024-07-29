# distutils: language = c++

cdef extern from "text2mecab.h":
    int text2mecab(char *output, size_t sizeOfOutput, const char *input)
