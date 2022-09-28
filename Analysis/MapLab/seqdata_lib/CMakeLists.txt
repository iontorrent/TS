# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (seqdata_lib)

add_library (seqdata
    align_batch.h   align_batch.cpp
    align_common.h
    cigar_utils.h   cigar_utils.cpp
    contalign.h     contalign.cpp
    MDtag.h         MDtag.cpp
    #nalign.h        nalign.cpp
    print_batches.h print_batches.cpp
    recreate_ref.h  recreate_ref.cpp
    sequtil.h       sequtil.cpp
    )

add_dependencies ( seqdata libStatGen )

#if ( NOT EXISTS ${ION_STATGEN_INCLUDES}/CigarRoller.h )
#    message ( WARNING "Can not find file ${ION_STATGEN_INCLUDES}/CigarRoller.h" )
#    find_path (ISGINC NAMES CigarRoller.h PATHS $ENV{HOME}/packages/libStatGen/include )
#    if ( NOT ISGINC )
#        message ( FATAL_ERROR "Can not find libStatGen includes" )
#    else ()
#        set (ION_STATGEN_INCLUDES ${ISGINC} )
#    endif ()
#endif ()

include_directories (../min_common_lib ${ION_STATGEN_INCLUDES})

