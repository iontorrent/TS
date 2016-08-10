# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.6)

project (seqdata_lib_test)

set (PROJECT_DESCRIPTION "Test suite for seqdata_lib")

set (SOURCES 
    seqdata_lib_test_main.cpp
    )

add_executable (seqdata_lib_test ${SOURCES})

#if ( NOT EXISTS ${ION_STATGEN_LIBS} )
#    find_library (ISGLIB name "StatGen" PATHS $ENV{HOME}/packages/libStatGen )
#    if ( NOT ISGLIB )
#        message ( FATAL_ERROR "Can not find libStatGen library" )
#    else ()
#        set (ION_STATGEN_LIBS ${ISGLIB} )
#    endif ()
#endif ()

#if ( NOT EXISTS ${ION_STATGEN_INCLUDES}/CigarRoller.h )
#    find_path (ISGINC NAMES CigarRoller.h PATHS $ENV{HOME}/packages/libStatGen/include )
#    if ( NOT ISGINC )
#        message ( FATAL_ERROR "Can not find libStatGen includes" )
#    else ()
#        set (ION_STATGEN_INCLUDES ${ISGINC} )
#    endif ()
#endif ()

include_directories (../min_common_lib ../seqdata_lib ${ION_STATGEN_INCLUDES} )

find_package(Threads REQUIRED)

target_link_libraries ( seqdata_lib_test min_common seqdata ${ION_STATGEN_LIBS} ${CMAKE_THREAD_LIBS_INIT} rt )