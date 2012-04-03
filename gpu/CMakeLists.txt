# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-gpu)
set(PROJECT_DESCRIPTION "Ion Torrent GPU Libraries")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion/gpu" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")

configure_file (
    "${PROJECT_SOURCE_DIR}/debian/postinst"
    "${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)
configure_file (
    "${PROJECT_SOURCE_DIR}/debian/prerm"
    "${PROJECT_BINARY_DIR}/debian/prerm" @ONLY
)
configure_file (
    "${PROJECT_SOURCE_DIR}/debian/preinst"
    "${PROJECT_BINARY_DIR}/debian/preinst" @ONLY
)
configure_file (
    "${PROJECT_SOURCE_DIR}/debian/postrm"
    "${PROJECT_BINARY_DIR}/debian/postrm" @ONLY
)

install(PROGRAMS devdriver_4.0_linux_64_270.27.run DESTINATION .)
install(PROGRAMS deviceQuery DESTINATION .)
install(PROGRAMS bandwidthTest DESTINATION .)
install(PROGRAMS create_nvidia_files DESTINATION .)
install(PROGRAMS nvcc DESTINATION /usr/local/cuda/bin)

install(DIRECTORY ../external/cudatoolkit-4.0.0RC/src/cudatoolkit/lib DESTINATION /usr/local/cuda
	PATTERN .svn EXCLUDE)
install(DIRECTORY ../external/cudatoolkit-4.0.0RC/src/cudatoolkit/lib64 DESTINATION /usr/local/cuda
	PATTERN .svn EXCLUDE)

include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "build-essential")
set(CPACK_PACKAGE_DESCRIPTION "This package contains the Torrent Browser and related programs including the job crawler, and the archive daemon.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
	${CMAKE_CURRENT_BINARY_DIR}/debian/postrm
)
include(CPack)
