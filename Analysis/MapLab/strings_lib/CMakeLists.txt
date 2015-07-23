cmake_minimum_required (VERSION 2.6)

project (strings_lib)

add_library (strings
    align_templ.h align_templ.hpp align_templ.cpp
    alphabet.h 
    alphabet_templ.h alphabet_templ.cpp
    banded_align.h   banded_align.hpp
    banded_convex_align.h banded_convex_align.hpp
    batch.h batch.cpp
    convex_align.h convex_align.hpp
    full_history_tracer.h full_history_tracer.hpp
    fundamental_preprocessing.h fundamental_preprocessing.hpp fundamental_preprocessing.cpp
    gap_cost_templ.h
    gencode.h gencode.cpp
    globals.cpp
    trace_holder_templ.h
    trace_matrix.h trace_matrix.cpp
    tracebox_templ.h
    translator.h
    translator_templ.h
    wmatrix_templ.h
    )

include_directories (strings, ../min_common_lib)