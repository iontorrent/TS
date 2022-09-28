# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (context_align)

set (PROJECT_DESCRIPTION "BAM Re-Aligner implementing context-dependent gap scoring")

set (SOURCES 
    context-align-main.cpp
    context-align-params.cpp context-align-params.h
    )

#set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Wno-overloaded-virtual -Wno-error=overloaded-virtual " )

add_executable (context_align ${SOURCES})

include_directories (../min_common_lib ../seqdata_lib ${ION_STATGEN_INCLUDES})

find_package( ZLIB REQUIRED )
include_directories( ${ZLIB_INCLUDE_DIRS} )

target_link_libraries (context_align min_common seqdata  ${ION_STATGEN_LIBS} ${ZLIB_LIBRARIES} )

add_dependencies ( context_align libStatGen )
