# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (tmap_lib)

include (tmapCommonBuild.txt)

set (TMAP_UTIL_SOURCES
    ${TMAP_SRC_UTIL_DIR}/tmap_alloc.h                       ${TMAP_SRC_UTIL_DIR}/tmap_alloc.c
    ${TMAP_SRC_UTIL_DIR}/tmap_definitions.h                 ${TMAP_SRC_UTIL_DIR}/tmap_definitions.c
    ${TMAP_SRC_UTIL_DIR}/tmap_error.h                       ${TMAP_SRC_UTIL_DIR}/tmap_error.c
    ${TMAP_SRC_UTIL_DIR}/tmap_rand.h                        ${TMAP_SRC_UTIL_DIR}/tmap_rand.c
    ${TMAP_SRC_UTIL_DIR}/tmap_time.h                        ${TMAP_SRC_UTIL_DIR}/tmap_time.c
    ${TMAP_SRC_UTIL_DIR}/tmap_fibheap.h                     ${TMAP_SRC_UTIL_DIR}/tmap_fibheap.c
    ${TMAP_SRC_UTIL_DIR}/tmap_hash.h
    ${TMAP_SRC_UTIL_DIR}/tmap_progress.h                    ${TMAP_SRC_UTIL_DIR}/tmap_progress.c
    ${TMAP_SRC_UTIL_DIR}/tmap_string.h                      ${TMAP_SRC_UTIL_DIR}/tmap_string.c
    ${TMAP_SRC_UTIL_DIR}/tmap_levenshtein.h                 ${TMAP_SRC_UTIL_DIR}/tmap_levenshtein.c
    ${TMAP_SRC_UTIL_DIR}/tmap_sort.h
    ${TMAP_SRC_UTIL_DIR}/tmap_vec.h
    ${TMAP_SRC_UTIL_DIR}/tmap_bsearch.h                     ${TMAP_SRC_UTIL_DIR}/tmap_bsearch.c
    ${TMAP_SRC_UTIL_DIR}/tmap_histo.h                       ${TMAP_SRC_UTIL_DIR}/tmap_histo.c
    ${TMAP_SRC_UTIL_DIR}/tmap_sam_convert.h                 ${TMAP_SRC_UTIL_DIR}/tmap_sam_convert.c
    ${TMAP_SRC_UTIL_DIR}/tmap_cigar_util.h                  ${TMAP_SRC_UTIL_DIR}/tmap_cigar_util.c
)

set (TMAP_SEQ_SOURCES
    ${TMAP_SRC_SEQ_DIR}/tmap_fq.h                           ${TMAP_SRC_SEQ_DIR}/tmap_fq.c
    ${TMAP_SRC_SEQ_DIR}/tmap_sff.h                          ${TMAP_SRC_SEQ_DIR}/tmap_sff.c
    ${TMAP_SRC_SEQ_DIR}/tmap_sam.h                          ${TMAP_SRC_SEQ_DIR}/tmap_sam.c
    ${TMAP_SRC_SEQ_DIR}/tmap_seq.h                          ${TMAP_SRC_SEQ_DIR}/tmap_seq.c
    ${TMAP_SRC_SEQ_DIR}/tmap_seqs.h                         ${TMAP_SRC_SEQ_DIR}/tmap_seqs.c
)

set (TMAP_IO_SOURCES
    ${TMAP_SRC_IO_DIR}/tmap_file.h                          ${TMAP_SRC_IO_DIR}/tmap_file.c
    ${TMAP_SRC_IO_DIR}/tmap_fq_io.h                         ${TMAP_SRC_IO_DIR}/tmap_fq_io.c
    ${TMAP_SRC_IO_DIR}/tmap_sff_io.h                        ${TMAP_SRC_IO_DIR}/tmap_sff_io.c
    ${TMAP_SRC_IO_DIR}/tmap_sam_io.h                        ${TMAP_SRC_IO_DIR}/tmap_sam_io.c
    ${TMAP_SRC_IO_DIR}/tmap_seq_io.h                        ${TMAP_SRC_IO_DIR}/tmap_seq_io.c
    ${TMAP_SRC_IO_DIR}/tmap_seqs_io.h                       ${TMAP_SRC_IO_DIR}/tmap_seqs_io.c
)

set (TMAP_INDEX_CORE_SOURCES
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt.h                        ${TMAP_SRC_INDEX_DIR}/tmap_bwt.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_aux.h                    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_aux.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwtl.h                       ${TMAP_SRC_INDEX_DIR}/tmap_bwtl.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_gen.h                    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_gen.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_match.h                  ${TMAP_SRC_INDEX_DIR}/tmap_bwt_match.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_match_hash.h             ${TMAP_SRC_INDEX_DIR}/tmap_bwt_match_hash.c
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_smem.h                   ${TMAP_SRC_INDEX_DIR}/tmap_bwt_smem.c
    ${TMAP_SRC_INDEX_DIR}/tmap_index.h                      ${TMAP_SRC_INDEX_DIR}/tmap_index.c
    ${TMAP_SRC_INDEX_DIR}/tmap_refseq.h                     ${TMAP_SRC_INDEX_DIR}/tmap_refseq.c
    ${TMAP_SRC_INDEX_DIR}/tmap_sa.c                         ${TMAP_SRC_INDEX_DIR}/tmap_sa.h
    ${TMAP_SRC_INDEX_DIR}/tmap_sa_aux.c                     ${TMAP_SRC_INDEX_DIR}/tmap_sa_aux.h
    ${TMAP_SRC_INDEX_DIR}/tmap_index_size.h                 ${TMAP_SRC_INDEX_DIR}/tmap_index_size.c
)

set (TMAP_SW_SOURCES
    ${TMAP_SRC_SW_DIR}/tmap_sw.h                            ${TMAP_SRC_SW_DIR}/tmap_sw.c
    ${TMAP_SRC_SW_DIR}/tmap_fsw.h                           ${TMAP_SRC_SW_DIR}/tmap_fsw.c
    ${TMAP_SRC_SW_DIR}/tmap_vsw_definitions.h               ${TMAP_SRC_SW_DIR}/tmap_vsw_definitions.c
    ${TMAP_SRC_SW_DIR}/tmap_vsw.h                           ${TMAP_SRC_SW_DIR}/tmap_vsw.c
    ${TMAP_SRC_SWLIB_DIR}/vsw.h                             ${TMAP_SRC_SWLIB_DIR}/vsw.cpp
    ${TMAP_SRC_SWLIB_DIR}/vsw16.h                           ${TMAP_SRC_SWLIB_DIR}/vsw16.cpp
    ${TMAP_SRC_SWLIB_DIR}/sw-vector.h                       ${TMAP_SRC_SWLIB_DIR}/sw-vector.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution.h                        ${TMAP_SRC_SWLIB_DIR}/Solution.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution1.h                       ${TMAP_SRC_SWLIB_DIR}/Solution1.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution2.h                       ${TMAP_SRC_SWLIB_DIR}/Solution2.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution3.h                       ${TMAP_SRC_SWLIB_DIR}/Solution3.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution4.h                       ${TMAP_SRC_SWLIB_DIR}/Solution4.cpp
    #${TMAP_SRC_SWLIB_DIR}/Solution5.h                       ${TMAP_SRC_SWLIB_DIR}/Solution5.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution6.h                       ${TMAP_SRC_SWLIB_DIR}/Solution6.cpp
    #${TMAP_SRC_SWLIB_DIR}/Solution7.h                       ${TMAP_SRC_SWLIB_DIR}/Solution7.cpp
    #${TMAP_SRC_SWLIB_DIR}/Solution8.h                       ${TMAP_SRC_SWLIB_DIR}/Solution8.cpp
    #${TMAP_SRC_SWLIB_DIR}/Solution9.h                       ${TMAP_SRC_SWLIB_DIR}/Solution9.cpp
    ${TMAP_SRC_SWLIB_DIR}/Solution10.h                      ${TMAP_SRC_SWLIB_DIR}/Solution10.cpp
    ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimization.h            ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimization.cpp
    ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimizationHash.h        ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimizationHash.cpp
    ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimizationWrapper.h     ${TMAP_SRC_SWLIB_DIR}/AffineSWOptimizationWrapper.cpp
    ${TMAP_SRC_SW_DIR}/tmap_vsw_bm.c
)

set (TMAP_REALIGN_SOURCES
    ${TMAP_SRC_REALIGN_DIR}/cigar_op.h
    ${TMAP_SRC_REALIGN_DIR}/realign_cliptype.h
    ${TMAP_SRC_REALIGN_DIR}/align_common.h
    ${TMAP_SRC_REALIGN_DIR}/Realign.h                       ${TMAP_SRC_REALIGN_DIR}/Realign.cpp
    ${TMAP_SRC_REALIGN_DIR}/realign_proxy.h                 ${TMAP_SRC_REALIGN_DIR}/realign_proxy.cpp
    ${TMAP_SRC_REALIGN_DIR}/realign_wrapper.h               ${TMAP_SRC_REALIGN_DIR}/realign_wrapper.cpp
    ${TMAP_SRC_REALIGN_DIR}/realign_wrapper_imp.h           ${TMAP_SRC_REALIGN_DIR}/realign_wrapper_imp.cpp
    ${TMAP_SRC_REALIGN_DIR}/realign_c_util.h                ${TMAP_SRC_REALIGN_DIR}/realign_c_util.c
    ${TMAP_SRC_REALIGN_DIR}/realign_util.h                  ${TMAP_SRC_REALIGN_DIR}/realign_util.cpp
    ${TMAP_SRC_REALIGN_DIR}/align_batch.h                   ${TMAP_SRC_REALIGN_DIR}/align_batch.cpp
    ${TMAP_SRC_REALIGN_DIR}/sequtil.h                       ${TMAP_SRC_REALIGN_DIR}/sequtil.cpp
    ${TMAP_SRC_REALIGN_DIR}/contalign.h                     ${TMAP_SRC_REALIGN_DIR}/contalign.cpp
    ${TMAP_SRC_REALIGN_DIR}/cigar_utils.h                   ${TMAP_SRC_REALIGN_DIR}/cigar_utils.cpp
    ${TMAP_SRC_REALIGN_DIR}/realign_wrapper_context_imp.h   ${TMAP_SRC_REALIGN_DIR}/realign_wrapper_context_imp.cpp
)

set (TMAP_INDEX_SPEED_SOURCES
    ${TMAP_SRC_INDEX_DIR}/tmap_index_speed.h                ${TMAP_SRC_INDEX_DIR}/tmap_index_speed.c
)

set (TMAP_BWT_CHECK_SOURCES
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_check.c                   ${TMAP_SRC_INDEX_DIR}/tmap_bwt_check.h 
)

set (TMAP_INDEX_SOURCES
    ${TMAP_INDEX_CORE_SOURCES}
    ${TMAP_BWT_CHECK_SOURCES}
)

set (TMAP_BWT_COMPARE_SOURCES
    ${TMAP_SRC_INDEX_DIR}/tmap_bwt_compare.c                 ${TMAP_SRC_INDEX_DIR}/tmap_bwt_compare.h 
)

set (TMAP_SAM2FS_SOURCES
    ${TMAP_SRC_SAM2FS_DIR}/tmap_sam2fs_aux.h                ${TMAP_SRC_SAM2FS_DIR}/tmap_sam2fs_aux.c 
    ${TMAP_SRC_SAM2FS_DIR}/tmap_sam2fs.h                    ${TMAP_SRC_SAM2FS_DIR}/tmap_sam2fs.c 
)

set (TMAP_SERVER_SOURCES
    ${TMAP_SRC_SERVER_DIR}/tmap_server.h                    ${TMAP_SRC_SERVER_DIR}/tmap_server.c
    ${TMAP_SRC_SERVER_DIR}/tmap_shm.h                       ${TMAP_SRC_SERVER_DIR}/tmap_shm.c
)

set (TMAP_MAP_UTIL_SOURCES
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_opt.h                 ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_opt.c
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_stats.h               ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_stats.c
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_util.h                ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_util.c
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_locopt.h              ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_locopt.c
    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_align_util.h          ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_align_util.c
#    ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_repair.h              ${TMAP_SRC_MAP_UTIL_DIR}/tmap_map_repair.c
)

set (TMAP_MAP_PAIRING_SOURCES
    ${TMAP_SRC_MAP_PAIRING_DIR}/tmap_map_pairing.h          ${TMAP_SRC_MAP_PAIRING_DIR}/tmap_map_pairing.c
)

set (TMAP_MAP1_SOURCES
    ${TMAP_SRC_MAP1_DIR}/tmap_map1.h                        ${TMAP_SRC_MAP1_DIR}/tmap_map1.c 
    ${TMAP_SRC_MAP1_DIR}/tmap_map1_aux.h                    ${TMAP_SRC_MAP1_DIR}/tmap_map1_aux.c 
)

set (TMAP_MAP2_SOURCES
    ${TMAP_SRC_MAP2_DIR}/tmap_map2.h                        ${TMAP_SRC_MAP2_DIR}/tmap_map2.c
    ${TMAP_SRC_MAP2_DIR}/tmap_map2_aux.h                    ${TMAP_SRC_MAP2_DIR}/tmap_map2_aux.c
    ${TMAP_SRC_MAP2_DIR}/tmap_map2_chain.h                  ${TMAP_SRC_MAP2_DIR}/tmap_map2_chain.c
    ${TMAP_SRC_MAP2_DIR}/tmap_map2_core.h                   ${TMAP_SRC_MAP2_DIR}/tmap_map2_core.c
    ${TMAP_SRC_MAP2_DIR}/tmap_map2_mempool.h                ${TMAP_SRC_MAP2_DIR}/tmap_map2_mempool.c
)

set (TMAP_MAP3_SOURCES
    ${TMAP_SRC_MAP3_DIR}/tmap_map3.h                        ${TMAP_SRC_MAP3_DIR}/tmap_map3.c
    ${TMAP_SRC_MAP3_DIR}/tmap_map3_aux.h                    ${TMAP_SRC_MAP3_DIR}/tmap_map3_aux.c
)

set (TMAP_MAP4_SOURCES
    ${TMAP_SRC_MAP4_DIR}/tmap_map4.h                        ${TMAP_SRC_MAP4_DIR}/tmap_map4.c
    ${TMAP_SRC_MAP4_DIR}/tmap_map4_aux.h                    ${TMAP_SRC_MAP4_DIR}/tmap_map4_aux.c
)

set (TMAP_MAPVSW_SOURCES
    ${TMAP_SRC_MAPVSW_DIR}/tmap_map_vsw.h                   ${TMAP_SRC_MAPVSW_DIR}/tmap_map_vsw.c
)

set (TMAP_MAPALL_SOURCES
    ${TMAP_SRC_MAPALL_DIR}/tmap_map_all.h                   ${TMAP_SRC_MAPALL_DIR}/tmap_map_all.c
)

set (TMAP_MAP_SOURCES
    ${TMAP_MAP_UTIL_SOURCES}
    ${TMAP_MAP_PAIRING_SOURCES}
    ${TMAP_MAP1_SOURCES}
    ${TMAP_MAP2_SOURCES}
    ${TMAP_MAP3_SOURCES}
    ${TMAP_MAP4_SOURCES}
    ${TMAP_MAPVSW_SOURCES}
    ${TMAP_MAPALL_SOURCES}
    ${TMAP_SRC_MAP_DIR}/tmap_map_driver.h                   ${TMAP_SRC_MAP_DIR}/tmap_map_driver.c
)

set (TMAP_LIB_SOURCES
    ${SAMTOOLS_SOURCES}
    ${TMAP_UTIL_SOURCES}
    ${TMAP_SEQ_SOURCES}
    ${TMAP_IO_SOURCES}
    ${TMAP_INDEX_SOURCES}
    ${TMAP_SW_SOURCES}
    ${TMAP_REALIGN_SOURCES}
    ${TMAP_SAM2FS_SOURCES}
    ${TMAP_SERVER_SOURCES}
    ${TMAP_MAP_SOURCES}
)

#include_directories ( BEFORE ${CMAKE_CURRENT_BINARY_DIR} )
include_directories ( BEFORE ${PROJECT_BINARY_DIR} )

message (STATUS "PROJECT_BINARY_DIR == ${PROJECT_BINARY_DIR}")
message (STATUS "CMAKE_CURRENT_BINARY_DIR == ${CMAKE_CURRENT_BINARY_DIR}")
message (STATUS "TMAP_DIR == ${TMAP_DIR}")

configure_file (
    "${TMAP_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h" @ONLY
)

add_library (tmapl
    ${TMAP_LIB_SOURCES}
)

set_target_properties (tmapl PROPERTIES COMPILE_FLAGS ${TMAP_COMPILE_FLAGS})
