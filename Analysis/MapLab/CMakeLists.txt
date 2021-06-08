# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.6)

project (MapLab)

add_subdirectory (min_common_lib) 
add_subdirectory (min_common_lib_test) 

#add_subdirectory (strings_lib)
#add_subdirectory (strings_lib_test)

add_subdirectory (seqdata_lib)
add_subdirectory (seqdata_lib_test)

add_subdirectory (context_align)

add_subdirectory (tmap_samtools_lib)
add_subdirectory (tmap_lib)
add_subdirectory (tmap_lib_test)

add_subdirectory (cigar_man_lib)
add_subdirectory (cigar_man_lib_test)


