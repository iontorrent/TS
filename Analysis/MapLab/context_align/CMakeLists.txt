cmake_minimum_required (VERSION 2.6)

project (context_align)

set (PROJECT_DESCRIPTION "BAM Re-Aligner implementing context-dependent gap scoring")

set (SOURCES 
    context-align-main.cpp
    context-align-params.cpp context-align-params.h
    )

add_executable (context_align ${SOURCES})

if ( NOT EXISTS ${ION_STATGEN_LIBS} )
    find_library (ISGLIB name "StatGen" PATHS $ENV{HOME}/packages/libStatGen )
    if ( NOT ISGLIB )
        message ( FATAL_ERROR "Can not find libStatGen library" )
    else ()
        set (ION_STATGEN_LIBS ${ISGLIB} )
    endif ()
endif ()

if ( NOT EXISTS ${ION_STATGEN_INCLUDES}/CigarRoller.h )
    find_path (ISGINC NAMES CigarRoller.h PATHS $ENV{HOME}/packages/libStatGen/include )
    if ( NOT ISGINC )
        message ( FATAL_ERROR "Can not find libStatGen includes" )
    else ()
        set (ION_STATGEN_INCLUDES ${ISGINC} )
    endif ()
endif ()


include_directories (../min_common_lib ../seqdata_lib ../strings_lib ${ION_STATGEN_INCLUDES})

find_package( ZLIB REQUIRED )
include_directories( ${ZLIB_INCLUDE_DIRS} )

target_link_libraries (context_align min_common seqdata strings ${ION_STATGEN_LIBS} ${ZLIB_LIBRARIES} )

