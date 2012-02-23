# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-torrentr)
set(PROJECT_DESCRIPTION "Ion Torrent torrentR")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/usr/local/lib/R/site-library" CACHE PATH "Ion Install Prefix")
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

set(ENV{ANALYSIS_INC}="${PROJECT_SOURCE_DIR}/../Analysis")
set(ENV{BKGMODEL_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel")
set(ENV{PHASE_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BaseCaller")
set(ENV{ANALYSIS_LIB}="${PROJECT_BINARY_DIR}/Analysis")

include_directories( ${PROJECT_SOURCE_DIR}/../Analysis )
include_directories( ${PROJECT_SOURCE_DIR}/../external/samtools-0.1.18 )
include_directories( /usr/local/lib/R/site-library/Rcpp/lib )

add_custom_target(LIBTORRENTRCLEAN ALL
           COMMAND "../../buildTools/cleanTorrentR.sh"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_custom_target(LIBTORRENTR ALL
           COMMAND "mkdir" "-p" "rlib"
           COMMAND "BUILD_ROOT=${PROJECT_SOURCE_DIR}" "R" "CMD" "INSTALL" "-l" "rlib" "${PROJECT_SOURCE_DIR}"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_dependencies(LIBTORRENTR LIBTORRENTRCLEAN)

install(DIRECTORY ${PROJECT_BINARY_DIR}/rlib/torrentR DESTINATION .
    USE_SOURCE_PERMISSIONS
)

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "r-base (>= 2.10.0), 
	libcurl4-openssl-dev,
    libatlas3gf-base, 
    libblas3gf, 
    libc6, 
    libgcc1, 
    libgfortran3, 
    liblapack3gf, 
    libstdc++6,
    g++, 
    gfortran, 
    r-cran-rcolorbrewer, 
    r-cran-gplots")
set(CPACK_PACKAGE_DESCRIPTION "A package for the statistical program R which provides methods for working with, analyzing, and visualizing Ion Torrent data.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
