cmake_minimum_required (VERSION 2.6)

project (strings_lib_test)

set (PROJECT_DESCRIPTION "Test suite for strings_lib")

set (SOURCES 
    strings_lib_test.cpp
    )

add_executable (strings_lib_test ${SOURCES})

include_directories (../min_common_lib ../strings_lib)

find_package(Threads REQUIRED)

target_link_libraries (strings_lib_test strings min_common ${CMAKE_THREAD_LIBS_INIT} rt)

