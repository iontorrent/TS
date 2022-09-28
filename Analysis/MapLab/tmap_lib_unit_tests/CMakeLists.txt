# Copyright (C) 2020 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (tmap_lib_test)

include (../tmap_lib/tmapCommonBuild.txt)

set (PROJECT_DESCRIPTION "Test suite for tmap_lib")

include_directories ( ../min_common_lib )
include_directories ( BEFORE ${PROJECT_BINARY_DIR} )

set (SOURCES 
    tmap_lib_unit_tests_main.cpp
    test_tmap_binary_search.cpp
    )

add_executable (tmap_lib_unit_tests ${SOURCES})
set_target_properties(tmap_lib_unit_tests PROPERTIES COMPILE_FLAGS ${TMAP_COMPILE_FLAGS})
set_target_properties(tmap_lib_unit_tests PROPERTIES LINKER_LANGUAGE CXX)

find_package(Threads REQUIRED)

# TODO: need to check if CMAKE_USE_PTHREADS_INIT is true - pthreads is the only threading mechanism supported

target_link_libraries (tmap_lib_unit_tests min_common tmapl tmap_samtools m z pthread bz2)

