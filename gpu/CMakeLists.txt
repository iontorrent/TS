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

include(../buildTools/cmake/CMakeLists.dependencies.txt)
install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_devdriver_version}/${cuda_devdriver_file} DESTINATION .)

        install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib DESTINATION /usr/local/cuda
        	PATTERN .svn EXCLUDE)
        install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64 DESTINATION /usr/local/cuda
        	PATTERN .svn EXCLUDE)
        install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/bin/nvcc DESTINATION /usr/local/cuda/bin)

if (0)
    set (CMAKE_C_FLAGS_RELEASE "")
    set (CMAKE_CXX_FLAGS_RELEASE "")
    include_directories( "${PROJECT_SOURCE_DIR}/common/inc" )
    SET( CUDA_NVCC_FLAGS "-O2" "--compiler-options=-fno-strict-aliasing" "-g" "--use_fast_math" 
                         "-gencode=arch=compute_20,code=sm_20" 
                         "-gencode=arch=compute_30,code=sm_30" 
                         "-gencode=arch=compute_35,code=sm_35" 
                         "-maxrregcount=48" )
    CUDA_ADD_EXECUTABLE(bandwidthTest bandwidthTest.cu)
    CUDA_ADD_EXECUTABLE(deviceQuery deviceQuery.cpp)
endif()

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
