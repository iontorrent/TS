# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
cmake_minimum_required (VERSION 2.8)

project(ion-torrentpy)

set(PROJECT_DESCRIPTION "Python interface to internal Ion Torrent Data")
enable_testing()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

#find_package for HDF5 libraries should be run before the dependencies, otherwise we get static libraries
set(HDF5_USE_STATIC_LIBRARIES False)

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)
include(../buildTools/cmake/CMakeLists.cpack.txt)
include(../buildTools/cmake/CMakeLists.dependencies.txt)

set(Boost_USE_STATIC_LIBS False)
find_package(Boost REQUIRED COMPONENTS python)


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
set(ION_INSTALL_PREFIX "/usr" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")


if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../Analysis")
message(SEND_ERROR "Analysis must be built first")
endif()
set(ion-analysis_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../Analysis")
set(ion-analysis_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Analysis")

if (PYTHON_EXECUTABLE)
    set(SETUP_PY_IN "${PROJECT_SOURCE_DIR}/torrentPy/setup.py.in")
    set(SETUP_PY    "${PROJECT_BINARY_DIR}/setup.py")
    SET(DEV_PY "${PROJECT_SOURCE_DIR}/setup_dev.py.in")

    SET(OUTPUT      "${PROJECT_BINARY_DIR}/build/timestamp")


    GET_DIRECTORY_PROPERTY( DEF_INCLUDES INCLUDE_DIRECTORIES)

    SET(ION_ANALYSIS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../Analysis" CACHE PATH "Ion Analysis Path")

    SET( ION_INCLUDES "${ION_ANALYSIS_SOURCE_DIR}/Image; ${ION_ANALYSIS_SOURCE_DIR}/; ${ION_ANALYSIS_SOURCE_DIR}/BaseCaller;
     ${ION_ANALYSIS_SOURCE_DIR}/Mask; ${ION_ANALYSIS_SOURCE_DIR}/BkgModel; ${ION_ANALYSIS_SOURCE_DIR}/BkgModel/Fitters;  ${ION_ANALYSIS_SOURCE_DIR}/BkgModel/MathModel;
     ${ION_ANALYSIS_SOURCE_DIR}/BkgModel/Bookkeeping; ${ION_ANALYSIS_SOURCE_DIR}/Util; ${ION_ANALYSIS_SOURCE_DIR}/Wells; ${HDF5_INCLUDE_DIR}; ${Boost_INCLUDE_DIRS}; ${PYTHON_INCLUDE_DIRS}; ${DEF_INCLUDES}" )


    FIND_LIBRARY(ION_ANALYSIS_LIBRARY ion-analysis HINTS ${PROJECT_BINARY_DIR}/../Analysis)
    FIND_LIBRARY(ION_IO_LIBRARY file-io HINTS ${PROJECT_BINARY_DIR}/../Analysis)
    SET(ION_LIBS ${ion-analysis_BINARY_DIR}/libion-analysis.a;${ion-analysis_BINARY_DIR}/libfile-io.a;${HDF5_LIBRARIES};${ION_BAMTOOLS_LIBS};${Boost_LIBRARIES})
    MESSAGE(STATUS "ION_LIBS: ${ION_LIBS}")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    configure_file(${DEV_PY} setup_dev.py)

    file(GLOB_RECURSE DEPS RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/torrentPy/*.py")

    foreach( fname ${DEPS} )
      configure_file( ${fname} ${fname} COPYONLY)
    endforeach(fname)


    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON_EXECUTABLE} setup.py build
                       #COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})


    include(../buildTools/cmake/CMakeLists.cpack.txt)

    install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py install --home=install)")
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/install/lib/python/ DESTINATION ${PYTHON_SITE_PACKAGES}
        USE_SOURCE_PERMISSIONS
    )


     set(CPACK_DEBIAN_PACKAGE_DEPENDS "g++,
         python-dev,
         libboost-python-dev,
         libhdf5-dev,
         python-numpy,
         python-matplotlib,
         python-scipy,
         python-pandas,
         python-pandas-lib,
         python-psycopg2,
         python-tables,
         python-dateutil,
         python-statsmodels,
         python-six,
         python-biopython,
         ipython,
         cython,
         libpq-dev,
         libgfortran3,
         liblapack-dev,
         liblapack3gf")

    set(CPACK_PACKAGE_DESCRIPTION "A package provides Python methods for working with, analyzing, and visualizing Ion Torrent data.")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
        ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
    )

    ADD_TEST(00_Install ${CMAKE_BUILD_TOOL} package) #fake target to make cmake run install first
    add_test(NAME Test_torrentPy WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests" COMMAND  "python" "-m" "unittest" "-v" "test_torrentPy")
    set_tests_properties(Test_torrentPy PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/install/lib/python/:\${PYTHONPATH}")
    include(CPack)
endif()


