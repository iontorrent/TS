cmake_minimum_required (VERSION 2.6)

project (min_common_lib)

add_library (min_common
    benchmark.h             benchmark.cpp
    bitops.h                bitops.cpp
    bitpack.h               bitpack.cpp
    circbuf.h               circbuf.cpp
    clreol.h                clreol.cpp
    cmdline.h               cmdline.cpp
    cmdline_s.h             cmdline_s.cpp
    comma_locale.h          comma_locale.cpp
    common_errors.h         common_errors.cpp
    common_str.h            common_str.cpp
    fasta.h                 fasta.cpp
    fileutils.h             fileutils.cpp
    ncbi_weight_matrix.h    ncbi_weight_matrix.cpp
    nullstream.h            nullstream.cpp
    parameters.h            parameters.cpp
    parameters_section.h    parameters_section.cpp
    platform.h              platform.cpp
    process_params.h        process_params.cpp
    rerror.h                rerror.cpp
    test_args.h             test_args.cpp
    test_case.h             test_case.cpp
    test_facet.h            test_facet.cpp
    test_suite.h            test_suite.cpp
    test_selection.h        test_selection.cpp
    text_files_dir_facet.h  text_files_dir_facet.cpp
    timer.h                 timer.cpp
    tracer.h                tracer.cpp
    weights.h               weights.cpp
    )

