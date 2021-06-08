# Copyright (C) 2020 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.6)

project (cigar_man_lib_test)

include (../tmap_lib/tmapCommonBuild.txt)


set (PROJECT_DESCRIPTION "Test suite for cigar_man_lib")

set (SOURCES 
    cigar_man_lib_test_main.cpp
    test_cigar_lib.cpp test_cigar_lib.h
    test_align_util.cpp test_align_util.h
    test_alignment_generator.cpp test_alignment_generator.h
    swmatrix_facet.cpp swmatrix_facet.h
)

add_executable (cigar_man_lib_test ${SOURCES})

include_directories (../min_common_lib ../cigar_man_lib ${ION_STATGEN_INCLUDES} )
include_directories ( BEFORE ${PROJECT_BINARY_DIR} )

configure_file (
    "${TMAP_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h" @ONLY
)

set_target_properties (tmap_lib_test PROPERTIES COMPILE_FLAGS ${TMAP_COMPILE_FLAGS} )

find_package(Threads REQUIRED)

# TODO: need to check if CMAKE_USE_PTHREADS_INIT is true - pthreads is the only threading mechanism supported

target_link_libraries (cigar_man_lib_test min_common cigar_man ${ION_STATGEN_LIBS} ${CMAKE_THREAD_LIBS_INIT} rt )
