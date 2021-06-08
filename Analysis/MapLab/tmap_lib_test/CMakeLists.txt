# Copyright (C) 2020 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.6)

project (tmap_lib_test)

include (../tmap_lib/tmapCommonBuild.txt)

set (PROJECT_DESCRIPTION "Test suite for tmap_lib")

set (SOURCES
    ${TMAP_SRC_DIR}/tmap_main.c
    )

#include_directories ( BEFORE ${CMAKE_CURRENT_BINARY_DIR} )
include_directories ( BEFORE ${PROJECT_BINARY_DIR} )

configure_file (
    "${TMAP_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h" @ONLY
)

add_executable (tmap_lib_test ${SOURCES})
set_target_properties(tmap_lib_test PROPERTIES COMPILE_FLAGS ${TMAP_COMPILE_FLAGS})

find_package(Threads REQUIRED)

# TODO: need to check if CMAKE_USE_PTHREADS_INIT is true - pthreads is the only threading mechanism supported

target_link_libraries (tmap_lib_test tmapl tmap_samtools  m z pthread bz2 rt)
