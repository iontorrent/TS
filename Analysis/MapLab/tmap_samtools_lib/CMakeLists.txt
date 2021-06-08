cmake_minimum_required (VERSION 2.6)

project (tmap_samtools)

include (../tmap_lib/tmapCommonBuild.txt)

set (SAMTOOLS_SOURCES
    ${SAMTOOLS}/bam.c                                       ${SAMTOOLS}/bam.h
    ${SAMTOOLS}/sam.c                                       ${SAMTOOLS}/sam.h
    ${SAMTOOLS}/sam_header.c                                ${SAMTOOLS}/sam_header.h
    ${SAMTOOLS}/bam_import.c
    ${SAMTOOLS}/bam_aux.c
    ${SAMTOOLS}/bam_pileup.c
    ${SAMTOOLS}/faidx.c                                     ${SAMTOOLS}/faidx.h
    ${SAMTOOLS}/razf.c                                      ${SAMTOOLS}/razf.h
    ${SAMTOOLS}/bgzf.c                                      ${SAMTOOLS}/bgzf.h
    ${SAMTOOLS}/kstring.c                                   ${SAMTOOLS}/kstring.h
    ${SAMTOOLS}/knetfile.c                                  ${SAMTOOLS}/knetfile.h
    ${SAMTOOLS}/errmod.c                                    ${SAMTOOLS}/errmod.h
)


add_library (tmap_samtools
    ${SAMTOOLS_SOURCES}
)

#configure_file (
#    "${TMAP_DIR}/config.h.in"
#    "${PROJECT_BINARY_DIR}/config.h" @ONLY
#)

set_target_properties (tmap_samtools PROPERTIES COMPILE_FLAGS ${TMAP_COMPILE_FLAGS})

