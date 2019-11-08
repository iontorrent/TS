# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-django)
set(PROJECT_DESCRIPTION "Install Django for Torrent Browser")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")

configure_file (
	"${PROJECT_SOURCE_DIR}/debian/postinst.in"
	"${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)

FIND_PACKAGE(PythonInterp)
# Determine python site-packages or dist-packages directory
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_program(VIRTUALENV virtualenv)
if(NOT VIRTUALENV)
    message(FATAL_ERROR "Could not find `virtualenv` in PATH")
endif()

# Create virtual environment in the install folder to avoid path issues
add_custom_target(venv ALL
    DEPENDS venv requirements.txt
    WORKING_DIRECTORY ${ION_INSTALL_PREFIX}
    COMMAND ${VIRTUALENV} --system-site-packages venv
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/requirements.txt requirements.txt
    COMMAND ./venv/bin/pip install -I -r requirements.txt
    COMMAND patch ${ION_INSTALL_PREFIX}/venv/local/lib/python2.7/site-packages/celery/utils/sysinfo.py < ${PROJECT_SOURCE_DIR}/celery_sysinfo_patch
    COMMAND ${CMAKE_COMMAND} -E copy_directory venv ${PROJECT_BINARY_DIR}/venv
    COMMAND ${CMAKE_COMMAND} -E remove_directory venv
)

install(DIRECTORY ${PROJECT_BINARY_DIR}/venv DESTINATION ${ION_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)

# Add venv site-packages to system python path
file(WRITE ${PROJECT_BINARY_DIR}/ion_django.pth "${ION_INSTALL_PREFIX}/venv/lib/python2.7/site-packages\n")
install(FILES ${PROJECT_BINARY_DIR}/ion_django.pth DESTINATION ${PYTHON_SITE_PACKAGES})

include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_REPLACES "python-django, python-django-south, python-django-tastypie, python-celery")

set(CPACK_PACKAGE_DESCRIPTION "This package contains Django 1.6 and related packages for ion-dbreports and ion-pipeline")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
)
include(CPack)
