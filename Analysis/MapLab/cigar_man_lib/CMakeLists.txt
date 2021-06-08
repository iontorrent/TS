# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.6)

project (cigar_man_lib)

include (../tmap_lib/tmapCommonBuild.txt)

set (PROJECT_DESCRIPTION "Alignment manipulation library, CIGAR string based")

set (CIGAR_MAN_SOURCES
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_repair.h                 ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_repair.c
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_align_util.h             ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_align_util.c
    ${TMAP_SRC_UTIL_DIR}/tmap_cigar_util.h                     ${TMAP_SRC_UTIL_DIR}/tmap_cigar_util.c
    ${TMAP_SRC_UTIL_DIR}/tmap_error.h                          ${TMAP_SRC_UTIL_DIR}/tmap_error.c
    ${TMAP_SRC_UTIL_DIR}/tmap_definitions.h                    ${TMAP_SRC_UTIL_DIR}/tmap_definitions.c
    ${TMAP_SRC_UTIL_DIR}/tmap_alloc.h                          ${TMAP_SRC_UTIL_DIR}/tmap_alloc.c
    ${TMAP_SRC_IO_DIR}/tmap_file.h                             ${TMAP_SRC_IO_DIR}/tmap_file.c
    alignment_generator.h                                      alignment_generator.cpp
)

include_directories ( BEFORE ${PROJECT_BINARY_DIR} )

configure_file (
    "${TMAP_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h" @ONLY
)

add_library (cigar_man 
    ${CIGAR_MAN_SOURCES}
)


