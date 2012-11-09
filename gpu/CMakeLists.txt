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

message(STATUS "BUILD with CUDA ${CUDA_VERSION}")

# wget http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/cudatoolkit_4.2.9_linux_64_ubuntu11.04.run
# MYCUDASDK=cudatoolkit_4.2.9_linux_64_ubuntu11.04
# bash $MYCUDASDK.run --nox11 --noexec --target $MYCUDASDK
# GZIP=--best tar czf $MYCUDASDK.tar.gz $MYCUDASDK
ExternalProject_add(${cuda_proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${cuda_proj_name_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/${cuda_tar_file}"
#    URL_MD5 ""
    TIMEOUT "3600"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

if (${LSB_RELEASE_CODENAME} STREQUAL "oneiric")
    install(PROGRAMS devdriver_4.2_linux_64_295.41.run DESTINATION .)
else()
    install(PROGRAMS devdriver_4.2_linux_64_295.41.run DESTINATION .)
endif()

        install(DIRECTORY ${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/lib DESTINATION /usr/local/cuda
        	PATTERN .svn EXCLUDE)
        install(DIRECTORY ${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/lib64 DESTINATION /usr/local/cuda
        	PATTERN .svn EXCLUDE)
        install(PROGRAMS ${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/bin/nvcc DESTINATION /usr/local/cuda/bin)

install(PROGRAMS deviceQuery DESTINATION .)
install(PROGRAMS bandwidthTest DESTINATION .)
install(PROGRAMS create_nvidia_files DESTINATION .)


include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_BREAKS "ion-analysis (<< 2.9)")
set(CPACK_DEBIAN_PACKAGE_REPLACES "ion-analysis (<< 2.9)")

#% nice to automate this
set(CPACK_PACKAGE_DESCRIPTION "This package contains the Torrent Browser and related programs including the job crawler, and the archive daemon.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
	${CMAKE_CURRENT_BINARY_DIR}/debian/postrm
)
include(CPack)
