# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-publishers)
set(PROJECT_DESCRIPTION "Ion Torrent Publishers")
enable_testing()
include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)
include(../buildTools/cmake/CMakeLists.installpath.txt)

# Where to install - override with: -DION_PUBLISHER_PREFIX
set(CMAKE_INSTALL_PREFIX "${ION_PUBLISHER_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_PUBLISHER_PREFIX})
set(CPACK_SET_DESTDIR "ON")


# BED
install(DIRECTORY "${PROJECT_SOURCE_DIR}/BED" DESTINATION "publishers"
    USE_SOURCE_PERMISSIONS )

install(DIRECTORY "${PROJECT_SOURCE_DIR}/refAnnot" DESTINATION "publishers"
    USE_SOURCE_PERMISSIONS )

configure_file (
        "${PROJECT_SOURCE_DIR}/debian/preinst.in"
       "${PROJECT_BINARY_DIR}/debian/preinst" @ONLY
)
configure_file (
    "${PROJECT_SOURCE_DIR}/debian/postinst.in"
    "${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "perl, python")
set(CPACK_PACKAGE_DESCRIPTION "Publishers for the Torrent Browser. These publishers, when present will allow users to upload files for automatic processing and access by plugins.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        ${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
        ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
)
include(CPack)
