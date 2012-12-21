# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-alignment)
set(PROJECT_DESCRIPTION "Ion Torrent alignment QC tool")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)


option(ION_USE_SYSTEM_PICARD "Use picard system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_PICARD)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/usr/local/bin" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")


add_executable(seqCoverage coverage/seqCoverage.cpp coverage/interval_tree.cpp)
add_dependencies(seqCoverage IONVERSION)
install(TARGETS seqCoverage DESTINATION .)

add_test("trimfastq.pl" "perl" "-cw" "${PROJECT_SOURCE_DIR}/trimfastq.pl")
install(PROGRAMS "${PROJECT_SOURCE_DIR}/trimfastq.pl" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/build_genome_index.pl" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/validate_reference.pl" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/alignmentQC.pl" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/merge_alignTable.py" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/merge_alignment.summary.py" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/MarkDuplicates.jar" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/getref" DESTINATION .)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/updateref.py" DESTINATION . RENAME updateref)

if (NOT ION_USE_SYSTEM_PICARD)
  set(proj_name "picard")
  set(proj_version "1.77")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}-prefix
    URL "http://updates.iontorrent.com/updates/software/external/picard-tools-1.77.tar.gz"
    #URL_MD5 
    # in source build
    BINARY_DIR "${ION_TS_EXTERNAL}/${proj_name_version}"
    SOURCE_DIR "${ION_TS_EXTERNAL}/${proj_name_version}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
endif()
install(DIRECTORY "${ION_TS_EXTERNAL}/${proj_name_version}/" DESTINATION "/opt/picard/picard-tools-current/")


include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "perl,
    unzip,
    samtools (>= 0.1.18),
    tmap (>= 2.3.5)")
set(CPACK_PACKAGE_DESCRIPTION "A set of script that aligns/maps fasta/fastq files to reference genome for QC")
include(CPack)
