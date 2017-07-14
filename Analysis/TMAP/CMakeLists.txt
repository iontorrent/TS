# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required(VERSION 2.6.4)

project(TMAP)

set (GLOBAL_SOURCES              config.h
				 src/util/tmap_alloc.h src/util/tmap_alloc.c 
				 src/util/tmap_definitions.h src/util/tmap_definitions.c 
				 src/util/tmap_error.h src/util/tmap_error.c 
				 src/util/tmap_rand.h src/util/tmap_rand.c 
				 src/util/tmap_time.h src/util/tmap_time.c 
				 src/util/tmap_fibheap.h src/util/tmap_fibheap.c 
				 src/util/tmap_hash.h 
				 src/util/tmap_progress.h src/util/tmap_progress.c 
				 src/util/tmap_string.h src/util/tmap_string.c 
				 src/util/tmap_levenshtein.h src/util/tmap_levenshtein.c 
				 src/util/tmap_sort.h 
				 src/util/tmap_vec.h 
				 src/util/tmap_sam_convert.h src/util/tmap_sam_convert.c 
				 src/seq/tmap_fq.h src/seq/tmap_fq.c 
				 src/seq/tmap_sff.h src/seq/tmap_sff.c 
				 src/seq/tmap_sam.h src/seq/tmap_sam.c 
				 src/seq/tmap_seq.h src/seq/tmap_seq.c 
				 src/seq/tmap_seqs.h src/seq/tmap_seqs.c 
				 src/io/tmap_file.h src/io/tmap_file.c 
				 src/io/tmap_fq_io.h src/io/tmap_fq_io.c 
				 src/io/tmap_sff_io.h src/io/tmap_sff_io.c 
				 src/io/tmap_sam_io.h src/io/tmap_sam_io.c 
				 src/io/tmap_seq_io.h src/io/tmap_seq_io.c 
				 src/io/tmap_seqs_io.h src/io/tmap_seqs_io.c 
				 src/index/tmap_bwt.h src/index/tmap_bwt.c 
				 src/index/tmap_bwt_aux.h src/index/tmap_bwt_aux.c 
				 src/index/tmap_bwtl.h src/index/tmap_bwtl.c 
				 src/index/tmap_bwt_gen.h src/index/tmap_bwt_gen.c 
				 src/index/tmap_bwt_match.h src/index/tmap_bwt_match.c 
				 src/index/tmap_bwt_match_hash.h src/index/tmap_bwt_match_hash.c 
				 src/index/tmap_bwt_smem.h src/index/tmap_bwt_smem.c 
				 src/index/tmap_index.h src/index/tmap_index.c 
				 src/index/tmap_refseq.h src/index/tmap_refseq.c 
				 src/index/tmap_sa.c src/index/tmap_sa.h 
				 src/index/tmap_sa_aux.c src/index/tmap_sa_aux.h 
				 src/index/tmap_index_size.h src/index/tmap_index_size.c 
				 src/sw/tmap_sw.h src/sw/tmap_sw.c 
				 src/sw/tmap_fsw.h src/sw/tmap_fsw.c 
				 src/sw/tmap_vsw_definitions.h src/sw/tmap_vsw_definitions.c 
				 src/sw/tmap_vsw.h src/sw/tmap_vsw.c 
				 src/sw/lib/vsw.cpp src/sw/lib/vsw.h 
				 src/sw/lib/vsw16.cpp src/sw/lib/vsw16.h 
				 src/sw/lib/sw-vector.cpp src/sw/lib/sw-vector.h 
				 src/sw/lib/Solution.cpp src/sw/lib/Solution.h 
				 src/sw/lib/Solution1.cpp src/sw/lib/Solution1.h 
				 src/sw/lib/Solution2.cpp src/sw/lib/Solution2.h 
				 src/sw/lib/Solution3.cpp src/sw/lib/Solution3.h 
				 src/sw/lib/Solution4.cpp src/sw/lib/Solution4.h 
				 #src/sw/lib/Solution5.cpp src/sw/lib/Solution5.h 
				 src/sw/lib/Solution6.cpp src/sw/lib/Solution6.h 
				 #src/sw/lib/Solution7.cpp src/sw/lib/Solution7.h 
				 #src/sw/lib/Solution8.cpp src/sw/lib/Solution8.h 
				 #src/sw/lib/Solution9.cpp src/sw/lib/Solution9.h 
				 src/sw/lib/Solution10.cpp src/sw/lib/Solution10.h 
				 src/sw/lib/AffineSWOptimization.cpp src/sw/lib/AffineSWOptimization.h 
				 src/sw/lib/AffineSWOptimizationHash.cpp src/sw/lib/AffineSWOptimizationHash.h 
				 src/sw/lib/AffineSWOptimizationWrapper.cpp src/sw/lib/AffineSWOptimizationWrapper.h 
				 src/sw/tmap_vsw_bm.c
				 src/realign/cigar_op.h
				 src/realign/realign_cliptype.h
				 src/realign/Realign.cpp src/realign/Realign.h
				 src/realign/realign_proxy.cpp src/realign/realign_proxy.h
				 src/realign/realign_wrapper.cpp src/realign/realign_wrapper.h
				 src/realign/realign_wrapper_imp.cpp src/realign/realign_wrapper_imp.h
				 src/realign/realign_c_util.c src/realign/realign_c_util.h
				 src/realign/realign_util.cpp src/realign/realign_util.h
				 src/realign/align_batch.cpp src/realign/align_batch.h
				 src/realign/align_common.h
				 src/realign/sequtil.cpp src/realign/sequtil.h
				 src/realign/contalign.cpp src/realign/contalign.h
				 src/realign/cigar_utils.cpp src/realign/cigar_utils.h
				 src/realign/realign_wrapper_context_imp.cpp src/realign/realign_wrapper_context_imp.h )

set (SAMTOOLS ${ION_TS_EXTERNAL}/samtools_tmap_e783ea9/samtools)

set (samtools_SOURCES  
				   ${SAMTOOLS}/bam.c ${SAMTOOLS}/bam.h 
				   ${SAMTOOLS}/sam.c ${SAMTOOLS}/sam.h 
				   ${SAMTOOLS}/sam_header.c ${SAMTOOLS}/sam_header.h 
				   ${SAMTOOLS}/bam_import.c 
				   ${SAMTOOLS}/bam_aux.c 
				   ${SAMTOOLS}/bam_pileup.c 
#				   ${SAMTOOLS}/bam_lpileup.c 
#				   ${SAMTOOLS}/bam_md.c 
				   ${SAMTOOLS}/faidx.c ${SAMTOOLS}/faidx.h 
#				   ${SAMTOOLS}/bedidx.c 
				   ${SAMTOOLS}/razf.c ${SAMTOOLS}/razf.h 
				   ${SAMTOOLS}/bgzf.c ${SAMTOOLS}/bgzf.h 
#				   ${SAMTOOLS}/bam_endian.h 
#				   ${SAMTOOLS}/khash.h 
#				   ${SAMTOOLS}/klist.h 
#				   ${SAMTOOLS}/kseq.h 
#				   ${SAMTOOLS}/ksort.h 
				   ${SAMTOOLS}/kstring.c ${SAMTOOLS}/kstring.h 
				   ${SAMTOOLS}/knetfile.c ${SAMTOOLS}/knetfile.h 
#				   ${SAMTOOLS}/kaln.c ${SAMTOOLS}/kaln.h 
#				   ${SAMTOOLS}/kprobaln.c ${SAMTOOLS}/kprobaln.h 
#				   ${SAMTOOLS}/bam_index.c 
#				   ${SAMTOOLS}/bam_sort.c 
#				   ${SAMTOOLS}/bam_reheader.c 
#				   ${SAMTOOLS}/bam_cat.c 
#				   ${SAMTOOLS}/bam_tview.c 
#				   ${SAMTOOLS}/bam_plcmd.c 
#				   ${SAMTOOLS}/sam_view.c 
#				   ${SAMTOOLS}/bam_rmdup.c ${SAMTOOLS}/bam_rmdupse.c 
#				   ${SAMTOOLS}/bam_mate.c ${SAMTOOLS}/bam_stat.c ${SAMTOOLS}/bam_color.c 
#				   ${SAMTOOLS}/bam2bcf.c ${SAMTOOLS}/bam2bcf.h 
#				   ${SAMTOOLS}/bam2bcf_indel.c 
				   ${SAMTOOLS}/errmod.c ${SAMTOOLS}/errmod.h 
#				   ${SAMTOOLS}/sample.c ${SAMTOOLS}/sample.h 
#				   ${SAMTOOLS}/cut_target.c ${SAMTOOLS}/phase.c ${SAMTOOLS}/bam2depth.c ${SAMTOOLS}/padding.c 
#				   ${SAMTOOLS}/bcftools/bcf.c ${SAMTOOLS}/bcftools/bcf.h 
#				   ${SAMTOOLS}/bcftools/bcf2qcall.c 
#				   ${SAMTOOLS}/bcftools/bcfutils.c 
#				   ${SAMTOOLS}/bcftools/call1.c 
#				   ${SAMTOOLS}/bcftools/em.c 
#				   ${SAMTOOLS}/bcftools/fet.c 
#				   ${SAMTOOLS}/bcftools/index.c 
#				   ${SAMTOOLS}/bcftools/kfunc.c 
#				   ${SAMTOOLS}/bcftools/kmin.c 
#				   ${SAMTOOLS}/bcftools/kmin.h 
#				   ${SAMTOOLS}/bcftools/main.c 
#				   ${SAMTOOLS}/bcftools/mut.c 
#				   ${SAMTOOLS}/bcftools/prob1.c 
#				   ${SAMTOOLS}/bcftools/prob1.h 
#				   ${SAMTOOLS}/bcftools/vcf.c 
#				   ${SAMTOOLS}/bamtk.c
)

set (tmap_SOURCES  
			   ${GLOBAL_SOURCES} 
			   ${samtools_SOURCES}
			   src/server/tmap_server.h src/server/tmap_server.c 
			   src/server/tmap_shm.h src/server/tmap_shm.c 
			   src/sam2fs/tmap_sam2fs_aux.h src/sam2fs/tmap_sam2fs_aux.c 
			   src/sam2fs/tmap_sam2fs.h src/sam2fs/tmap_sam2fs.c 
			   src/index/tmap_index_speed.h src/index/tmap_index_speed.c 
			   src/index/tmap_bwt_check.c src/index/tmap_bwt_check.h 
			   src/index/tmap_bwt_compare.c src/index/tmap_bwt_compare.h 
			   src/map/util/tmap_map_opt.h src/map/util/tmap_map_opt.c 
			   src/map/util/tmap_map_stats.h src/map/util/tmap_map_stats.c 
			   src/map/util/tmap_map_util.h src/map/util/tmap_map_util.c 
			   src/map/pairing/tmap_map_pairing.h src/map/pairing/tmap_map_pairing.c 
			   src/map/map1/tmap_map1.h src/map/map1/tmap_map1.c 
			   src/map/map1/tmap_map1_aux.h src/map/map1/tmap_map1_aux.c 
			   src/map/map2/tmap_map2.h src/map/map2/tmap_map2.c 
			   src/map/map2/tmap_map2_aux.h src/map/map2/tmap_map2_aux.c 
			   src/map/map2/tmap_map2_chain.h src/map/map2/tmap_map2_chain.c 
			   src/map/map2/tmap_map2_core.h src/map/map2/tmap_map2_core.c 
			   src/map/map2/tmap_map2_mempool.h src/map/map2/tmap_map2_mempool.c 
			   src/map/map3/tmap_map3.h src/map/map3/tmap_map3.c 
			   src/map/map3/tmap_map3_aux.h src/map/map3/tmap_map3_aux.c 
			   src/map/map4/tmap_map4.h src/map/map4/tmap_map4.c 
			   src/map/map4/tmap_map4_aux.h src/map/map4/tmap_map4_aux.c 
			   src/map/mapvsw/tmap_map_vsw.h src/map/mapvsw/tmap_map_vsw.c 
			   src/map/mapall/tmap_map_all.h src/map/mapall/tmap_map_all.c 
			   src/map/tmap_map_driver.h src/map/tmap_map_driver.c 
			   src/tmap_main.h src/tmap_main.c )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
include_directories( ${ION_TS_EXTERNAL}/samtools_tmap_e783ea9 )
include_directories( ${ION_TS_EXTERNAL}/samtools_tmap_e783ea9/samtools )

configure_file (
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h" @ONLY
)

add_executable(tmap ${tmap_SOURCES})
set_target_properties(tmap PROPERTIES COMPILE_FLAGS "-D_TMAP_BWT_RUN_TYPE=0 -D_USE_KNETFILE -DSAMTOOLS_MAIN -D__STDC_LIMIT_MACROS -DTMAP_MMAP=1")

include(CheckCCompilerFlag)
check_c_compiler_flag(-fdiagnostics-color=never HAS_DIAGNOSTICS_COLOR)
if (HAS_DIAGNOSTICS_COLOR)
    target_compile_options(tmap PUBLIC -fdiagnostics-color=never)
endif()



#

target_link_libraries (tmap m z pthread bz2)
install(TARGETS tmap DESTINATION bin)


#add_executable(samtools ${samtools_SOURCES})
#target_link_libraries (samtools curses bam m z pthread)
#set_target_properties(samtools PROPERTIES
#    COMPILE_FLAGS "-pthread -msse3 -D_TMAP_BWT_RUN_TYPE=0 -D_USE_KNETFILE -D_CURSES_LIB=1 -D_FILE_OFFSET_BITS=64 -DSAMTOOLS_MAIN -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -D__S$
# -D_FILE_OFFSET_BITS=64 -D_USE_KNETFILE -DMD5SUM_MAIN -D_CURSES_LIB=1


