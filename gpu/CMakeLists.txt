# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-gpu)
set(PROJECT_DESCRIPTION "Ion Torrent GPU Libraries")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

set (ION_TS_EXTERNAL "${PROJECT_SOURCE_DIR}/../external/")
#set (ION_TS_EXTERNAL "${CMAKE_BINARY_DIR}/external/")

#set(ION_UPDATE_SERVER "updates.iontorrent.com") # internal developer can use just "updates"

# Use the closest ION_UPDATE_SERVER we can find.
message(STATUS "Searching for an ION_UPDATE_SERVER...")
foreach ( SERVER updates updates.itw updates.iontorrent.com )
    # See if we can look up the server in question.
    execute_process( COMMAND host -W 2 ${SERVER}
       RESULT_VARIABLE NOT_FOUND OUTPUT_QUIET ERROR_QUIET )

    # If we can find it, great! We're done!
    if ( NOT ${NOT_FOUND} )
       set( ION_UPDATE_SERVER ${SERVER} )
       break()
    endif()
endforeach()

# If we find a server, let everyone know.
if (ION_UPDATE_SERVER)
    message(STATUS "Using ION_UPDATE_SERVER: ${ION_UPDATE_SERVER}")
else()
    message(WARNING "Could not find a valid ION_UPDATE_SERVER.")
endif()

#custom ExternalProject_add with timeout feature
IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} STRLESS 2.8.2)
  SET(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${ION_TS_EXTERNAL}/CMake
  )
ENDIF ()
include(ExternalProject)

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

set(gpu_devdriver_version "375.26")
set(gpu_devdriver_file "devdriver_NVIDIA-Linux-x86_64-375.26.run")

set(gpu_devdriver "gpu_devdriver")
set(gpu_devdriver_version "${gpu_devdriver}-${gpu_devdriver_version}")

ExternalProject_add(${gpu_devdriver}
    PREFIX ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}-prefix
    SOURCE_DIR ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/${gpu_devdriver_file}.tar.gz" # cmake doesn't download .run files
#    URL_MD5 ""
    TIMEOUT "3600"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

install(PROGRAMS ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}/${gpu_devdriver_file}
        DESTINATION .)

install(PROGRAMS create_nvidia_files
        DESTINATION .)

## compile and install deviceQuery and bandwidthTest programs
message(STATUS "BUILD with CUDA ${CUDA_VERSION}")

include(../buildTools/cmake/CMakeLists.dependencies.txt)
set (CMAKE_C_FLAGS_RELEASE "")
set (CMAKE_CXX_FLAGS_RELEASE "")
include_directories( "${PROJECT_SOURCE_DIR}/common/inc" )

CUDA_ADD_EXECUTABLE(bandwidthTest bandwidthTest.cu)
CUDA_ADD_EXECUTABLE(deviceQuery deviceQuery.cpp)

add_dependencies(bandwidthTest cuda_toolkit)
add_dependencies(deviceQuery cuda_toolkit)

install(PROGRAMS ${PROJECT_BINARY_DIR}/deviceQuery DESTINATION .)
install(PROGRAMS ${PROJECT_BINARY_DIR}/bandwidthTest DESTINATION .)

include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_BREAKS "ion-analysis (<< 5.1.5)")
set(CPACK_DEBIAN_PACKAGE_REPLACES "ion-analysis (<< 5.1.5)")

# Compiling GPU kernel code requires
set(CPACK_DEBIAN_PACKAGE_DEPENDS "build-essential, linux-headers, dkms")


#% nice to automate this
set(CPACK_PACKAGE_DESCRIPTION "This package contains the NVIDIA device driver for the Torrent Server and instruments.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
	${CMAKE_CURRENT_BINARY_DIR}/debian/postrm
)
include(CPack)
