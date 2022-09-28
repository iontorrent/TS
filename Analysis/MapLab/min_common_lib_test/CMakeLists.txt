# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (min_common_lib_test)

set (PROJECT_DESCRIPTION "Test suite for min_common_lib")

set (SOURCES 
    min_common_lib_test_main.cpp
    test_default.h        test_default.cpp
    test_test.h           test_test.cpp
    test_benchmark.h      test_benchmark.cpp
    test_facets.h         test_facets.cpp
    test_error_handling.h test_error_handling.cpp
    test_loggers.h        test_loggers.cpp
    test_resource.h       test_resource.cpp
    test_tracer.h         test_tracer.cpp
    test_time_counter.h   test_time_counter.cpp
    test_temp_file.h      test_temp_file.cpp
    test_cmdline_parser.h test_cmdline_parser.cpp
    test_file_utils.h     test_file_utils.cpp
    )

add_executable (min_common_lib_test ${SOURCES})

include_directories (../min_common_lib)

find_package(Threads REQUIRED)

target_link_libraries (min_common_lib_test min_common ${CMAKE_THREAD_LIBS_INIT} rt)

