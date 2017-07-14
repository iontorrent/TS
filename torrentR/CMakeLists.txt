# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-torrentr)
set(PROJECT_DESCRIPTION "Ion Torrent torrentR")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)
include(../buildTools/cmake/CMakeLists.dependencies.txt)

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
set(ENV{ANALYSIS_IMAGE}="${PROJECT_SOURCE_DIR}/../Analysis/Image")
set(ENV{ANALYSIS_MASK}="${PROJECT_SOURCE_DIR}/../Analysis/Mask")
set(ENV{ANALYSIS_ORG_INC}="${PROJECT_SOURCE_DIR}/../Analysis/AnalysisOrg")
set(ENV{BKGMODEL_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel")
set(ENV{BKGMODEL_MATH_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/MathModel")
set(ENV{BKGMODEL_FITTERS_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Fitters")
set(ENV{BKGMODEL_WRITERS_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Writers")
set(ENV{BKGMODEL_BOOK_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Bookkeeping")
set(ENV{PHASE_INC}="${PROJECT_SOURCE_DIR}/../Analysis/BaseCaller")
set(ENV{RECALIBRATION_INC}="${PROJECT_SOURCE_DIR}/../Analysis/Calibration")
set(ENV{CLONALFILTER_INC}="${PROJECT_SOURCE_DIR}/../Analysis/ClonalFilter")
set(ENV{UTIL_INC}="${PROJECT_SOURCE_DIR}/../Analysis/Util")
set(ENV{WELLS_INC}="${PROJECT_SOURCE_DIR}/../Analysis/Wells")
set(ENV{ANALYSIS_LIB}="${PROJECT_BINARY_DIR}/Analysis")
set(ENV{JSON_INC}="${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1")

include_directories( ${PROJECT_SOURCE_DIR}/../Analysis )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/AnalysisOrg )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BaseCaller )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Calibration )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/ClonalFilter )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Image )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Mask )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Fitters )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Writers )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/MathModel )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Bookkeeping )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Util )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Wells )
#include_directories(/usr/lib/R/site-library/Rcpp/include/Rcpp )
include_directories( ${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1 )


add_custom_target(LIBTORRENTRCLEAN ALL
           COMMAND "${PROJECT_SOURCE_DIR}/../buildTools/cleanTorrentR.sh" "${PROJECT_SOURCE_DIR}"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_custom_target(LIBTORRENTR ALL
           COMMAND "mkdir" "-p" "rlib"
           COMMAND "TS_BUILD_ROOT=${PROJECT_BINARY_DIR}/.." "TS_SOURCE_ROOT=${PROJECT_SOURCE_DIR}/.." MAKE="$(MAKE) -j$(nproc)" "R" "CMD" "INSTALL" "-l" "rlib" "${PROJECT_SOURCE_DIR}"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_dependencies(LIBTORRENTR LIBTORRENTRCLEAN)

install(DIRECTORY ${PROJECT_BINARY_DIR}/rlib/torrentR DESTINATION .
    USE_SOURCE_PERMISSIONS
)

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "g++,
    libhdf5-serial-dev,
    libgcc1,
    libgfortran3,
    libstdc++6,
    gfortran,
    libcurl4-openssl-dev,
    libxml2-dev,
    r-base-core,
    r-base-dev,
    r-base-html,
    r-doc-html,
    r-bioc-annotationdbi,
    r-bioc-biobase,
    r-bioc-biocgenerics,
    r-bioc-biostrings,
    r-bioc-genomicranges,
    r-bioc-iranges,
    r-bioc-rsamtools,
    r-bioc-shortread,
    r-cran-boot,
    r-cran-class,
    r-cran-cluster,
    r-cran-codetools,
    r-cran-dbi,
    r-cran-gdata,
    r-cran-getopt,
    r-cran-ggplot2,
    r-cran-gtable,
    r-cran-foreign,
    r-cran-formula,
    r-cran-hdf5,
    r-cran-hwriter,
    r-cran-kernsmooth,
    r-cran-lattice,
    r-cran-latticeextra,
    r-cran-mass,
    r-cran-matrix,
    r-cran-mgcv,
    r-cran-nlme,
    r-cran-nnet,
    r-cran-plyr,
    r-cran-rcolorbrewer,
    r-cran-rjson,
    r-cran-rpart,
    r-cran-scales,
    r-cran-spatial,
    r-cran-stringr,
    r-cran-survival,
    r-cran-reshape2,
    r-cran-rcpp,
    r-cran-rcurl,
    r-cran-rsqlite")
# these two packages have been removed because it causes a bug in aptitude
# TS-14064 which makes the package always upgradable
#    r-cran-xml,
#    r-cran-xtable")

set(CPACK_DEBIAN_PACKAGE_CONFLICTS "r-cran-gplots")
set(CPACK_DEBIAN_PACKAGE_REPLACES  "r-cran-gplots")

set(CPACK_PACKAGE_DESCRIPTION "A package for the statistical program R which provides methods for working with, analyzing, and visualizing Ion Torrent data.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
