# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-alignment)
set(PROJECT_DESCRIPTION "Ion Torrent alignment QC tool")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/usr/local/bin" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")

configure_file (
	"${PROJECT_SOURCE_DIR}/debian/postinst.in"
	"${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/debian/prerm.in"
	"${PROJECT_BINARY_DIR}/debian/prerm" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/alignmentQC.pl"
	"${PROJECT_BINARY_DIR}/alignmentQC.pl" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/getref"
	"${PROJECT_BINARY_DIR}/getref" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/updateref.py"
	"${PROJECT_BINARY_DIR}/updateref.py" @ONLY
)

add_executable(seqCoverage coverage/seqCoverage.cpp coverage/interval_tree.cpp)
add_dependencies(seqCoverage IONVERSION)
install(TARGETS seqCoverage DESTINATION .)

add_test("trimfastq.pl" "perl" "-cw" "${PROJECT_SOURCE_DIR}/trimfastq.pl")
install(PROGRAMS "${PROJECT_SOURCE_DIR}/trimfastq.pl" DESTINATION .)

install(PROGRAMS "${PROJECT_SOURCE_DIR}/build_genome_index.pl" DESTINATION .)

install(PROGRAMS "${PROJECT_SOURCE_DIR}/validate_reference.pl" DESTINATION .)

install(PROGRAMS "${PROJECT_BINARY_DIR}/alignmentQC.pl" DESTINATION .)

install(PROGRAMS "${PROJECT_BINARY_DIR}/getref" DESTINATION .)

install(PROGRAMS "${PROJECT_BINARY_DIR}/updateref.py" DESTINATION . RENAME updateref)

add_custom_target(picard-tools ALL
    COMMAND "unzip" "-o" "${PROJECT_SOURCE_DIR}/picard-tools-1.52.zip"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    COMMENT "Unzipping picard-tools tarball")
    
install(DIRECTORY "${PROJECT_BINARY_DIR}/picard-tools-1.52" DESTINATION "/opt/picard/")

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6,
	libgcc1, 
    libstdc++6, 
    perl,
    unzip,
    tmap (>= 0.0.19-1)")
set(CPACK_PACKAGE_DESCRIPTION "A set of script that aligns/maps fasta/fastq files to reference genome for QC")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
