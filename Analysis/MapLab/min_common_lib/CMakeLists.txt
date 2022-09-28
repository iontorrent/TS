# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8.12)

project (min_common_lib)

add_library (min_common
    benchmark.h             benchmark.cpp
    clreol.h                clreol.cpp
    cmdline.h               cmdline.cpp
    cmdline_s.h             cmdline_s.cpp
    comma_locale.h          comma_locale.cpp
    common_errors.h         common_errors.cpp
    common_types.h          common_types.cpp
    common_str.h            common_str.cpp
    fileutils.h             fileutils.cpp
    nullstream.h            nullstream.cpp
    parameters.h            parameters.cpp
    parameters_section.h    parameters_section.cpp
    process_params.h        process_params.cpp
    runtime_error.h         runtime_error.cpp
    test_args.h             test_args.cpp
    test_case.h             test_case.cpp
    test_facet.h            test_facet.cpp
    test_suite.h            test_suite.cpp
    test_selection.h        test_selection.cpp
    text_files_dir_facet.h  text_files_dir_facet.cpp
    time_counter.h          time_counter.cpp
    tracer.h                tracer.cpp
    )

