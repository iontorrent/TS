# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-publishers)
set(PROJECT_DESCRIPTION "Ion Torrent Publishers")
enable_testing()
include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/results" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")


# BED
install(DIRECTORY "${PROJECT_SOURCE_DIR}/BED" DESTINATION "publishers"
    USE_SOURCE_PERMISSIONS
    PATTERN ".svn" EXCLUDE)

configure_file (
        "${PROJECT_SOURCE_DIR}/debian/preinst.in"
       "${PROJECT_BINARY_DIR}/debian/preinst" @ONLY
)

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "perl, python")
set(CPACK_PACKAGE_DESCRIPTION "Publishers for the Torrent Browser. These publishers, when present will allow users to upload files for automatic processing and access by plugins.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        ${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
)
include(CPack)
