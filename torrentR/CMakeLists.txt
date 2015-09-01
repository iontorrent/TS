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
set(ENV{UTIL_INC}="${PROJECT_SOURCE_DIR}/../Analysis/Util")
set(ENV{WELLS_INC}="${PROJECT_SOURCE_DIR}/../Analysis/Wells")
set(ENV{ANALYSIS_LIB}="${PROJECT_BINARY_DIR}/Analysis")
set(ENV{JSON_INC}="${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1")

include_directories( ${PROJECT_SOURCE_DIR}/../Analysis )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/AnalysisOrg )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BaseCaller )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Calibration )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Image )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Mask )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Fitters )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Writers )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/MathModel )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/BkgModel/Bookkeeping )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Util )
include_directories( ${PROJECT_SOURCE_DIR}/../Analysis/Wells )
include_directories( /usr/local/lib/R/site-library/Rcpp/lib )
include_directories( ${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1 )


add_custom_target(LIBTORRENTRCLEAN ALL
           COMMAND "${PROJECT_SOURCE_DIR}/../buildTools/cleanTorrentR.sh" "${PROJECT_SOURCE_DIR}"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_custom_target(LIBTORRENTR ALL
           COMMAND "mkdir" "-p" "rlib"
           COMMAND "TS_BUILD_ROOT=${PROJECT_BINARY_DIR}/.." "TS_SOURCE_ROOT=${PROJECT_SOURCE_DIR}/.." MAKE="make -j4" "R" "CMD" "INSTALL" "-l" "rlib" "${PROJECT_SOURCE_DIR}"
           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

add_dependencies(LIBTORRENTR LIBTORRENTRCLEAN)

install(DIRECTORY ${PROJECT_BINARY_DIR}/rlib/torrentR DESTINATION .
    USE_SOURCE_PERMISSIONS
)

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "g++,
    r-base (>= 2.15.1-1lucid0),
    r-base-core (>= 2.15.1-1lucid0),
    r-base-dev (>= 2.15.1-1lucid0),
    r-base-html (>= 2.15.1-1lucid0),
    r-doc-html (>= 2.15.1-1lucid0),
    r-recommended (>= 2.15.1-1lucid0),
    r-cran-boot (>= 1.3-5-1lucid0),
    r-cran-class (>= 7.3-2-2lucid0),
    r-cran-cluster (>= 1.14.2-1lucid0),
    r-cran-codetools (>= 0.2-8-1),
    r-cran-foreign (>= 0.8.50-1lucid0),
    r-cran-kernsmooth (>= 2.23-8-1),
    r-cran-lattice (>= 0.20-6-1lucid0),
    r-cran-mass (>= 7.3-19-1lucid0),
    r-cran-matrix (>= 1.0-6-1lucid0),
    r-cran-mgcv (>= 1.7-19-1lucid0),
    r-cran-nlme (>= 3.1.104-1lucid0),
    r-cran-nnet (>= 7.3-1-1lucid0),
    r-cran-rpart (>= 3.1.54-1lucid0),
    r-cran-spatial (>= 7.3-2-1lucid0),
    r-cran-survival (>= 2.36-14-1lucid0),
    libblas3gf,
    libc6,
    libhdf5-serial-dev,
    libgcc1,
    libgfortran3,
    liblapack3gf,
    libstdc++6,
    gfortran,
    libcurl4-openssl-dev,
    libxml2-dev")

set(CPACK_DEBIAN_PACKAGE_CONFLICTS "r-cran-bitops, r-cran-catools, r-cran-gdata, r-cran-gplots, r-cran-gtools, r-cran-rocr")
set(CPACK_DEBIAN_PACKAGE_REPLACES "r-cran-bitops, r-cran-catools, r-cran-gdata, r-cran-gplots, r-cran-gtools, r-cran-rocr")

set(CPACK_PACKAGE_DESCRIPTION "A package for the statistical program R which provides methods for working with, analyzing, and visualizing Ion Torrent data.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
