# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-gpu)
set(PROJECT_DESCRIPTION "Ion Torrent GPU Libraries")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

FIND_PACKAGE(PythonInterp)
# Determine python site-packages or dist-packages directory
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib(prefix='/usr/local')"
        OUTPUT_VARIABLE PYTHON_LOCAL_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_version; print get_python_version()"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)


# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion/gpu" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")

add_custom_target(compileall ALL
    COMMAND ${PYTHON_EXECUTABLE} -m compileall -q -f "${PROJECT_SOURCE_DIR}/python/"
)

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
install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_devdriver_version}/${cuda_devdriver_file}
        DESTINATION .)

install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib
        DESTINATION /usr/local/cuda
        PATTERN .svn EXCLUDE)

install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64
        DESTINATION /usr/local/cuda
        PATTERN .svn EXCLUDE)

install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/bin/nvcc
        DESTINATION /usr/local/cuda/bin)

# from https://pypi.python.org/pypi/nvidia-ml-py/
install(FILES python/nvidia_smi.py    DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES python/nvidia_smi.pyc   DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES python/pynvml.py        DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES python/pynvml.pyc       DESTINATION ${PYTHON_SITE_PACKAGES})
install(PROGRAMS python/pynvml_test.py         DESTINATION /usr/local/bin)
install(PROGRAMS python/nvidia_smi_test.py     DESTINATION /usr/local/bin)


if (1)
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
    CUDA_ADD_EXECUTABLE(xtalk_sim
                        xtalk_sim/xtalk_sim.cpp
                        xtalk_sim/DiffEqModel_Init.cpp
                        xtalk_sim/DiffEqModel.cpp
                        xtalk_sim/DelsqCUDA.cu
                        xtalk_sim/utils.cpp
                        xtalk_sim/xtalk_sim.cpp
                        xtalk_sim/sim_incorp_flux.cpp
                        xtalk_sim/WorkerInfoQueue.cpp)

    target_link_libraries(xtalk_sim pthread)

    add_dependencies(bandwidthTest cuda_toolkit)
    add_dependencies(deviceQuery cuda_toolkit)
    add_dependencies(xtalk_sim cuda_toolkit)

    install(PROGRAMS ${PROJECT_BINARY_DIR}/deviceQuery DESTINATION .)
    install(PROGRAMS ${PROJECT_BINARY_DIR}/bandwidthTest DESTINATION .)
    install(PROGRAMS ${PROJECT_BINARY_DIR}/xtalk_sim DESTINATION .)

endif()

install(PROGRAMS create_nvidia_files DESTINATION .)


include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_BREAKS "ion-analysis (<< 2.9)")
set(CPACK_DEBIAN_PACKAGE_REPLACES "ion-analysis (<< 2.9)")

# Compiling GPU kernel code requires
set(CPACK_DEBIAN_PACKAGE_DEPENDS "build-essential, linux-headers, dkms")


#% nice to automate this
set(CPACK_PACKAGE_DESCRIPTION "This package contains the Torrent Browser and related programs including the job crawler, and the archive daemon.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
	${CMAKE_CURRENT_BINARY_DIR}/debian/postrm
)
include(CPack)
