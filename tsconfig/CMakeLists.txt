# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-tsconfig)
set(PROJECT_DESCRIPTION "Ion Torrent Torrent Server Configuration Script")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/usr" CACHE PATH "Ion Install Prefix")
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
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSconfig"
	"${PROJECT_BINARY_DIR}/TSconfig" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSchkconfig"
	"${PROJECT_BINARY_DIR}/TSchkconfig" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/ts_functions"
	"${PROJECT_BINARY_DIR}/ts_functions" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/ts_params"
	"${PROJECT_BINARY_DIR}/ts_params" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSstaticip"
	"${PROJECT_BINARY_DIR}/TSstaticip" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSsetproxy"
	"${PROJECT_BINARY_DIR}/TSsetproxy" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSquery"
	"${PROJECT_BINARY_DIR}/TSquery" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSdns"
	"${PROJECT_BINARY_DIR}/TSdns" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSwrapper"
	"${PROJECT_BINARY_DIR}/TSwrapper" @ONLY
)


install(PROGRAMS "${PROJECT_BINARY_DIR}/TSconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSchkconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSstaticip" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSsetproxy" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSquery" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSdns" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSwrapper" DESTINATION sbin )
install(FILES "${PROJECT_BINARY_DIR}/ts_functions" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_BINARY_DIR}/ts_params" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/config/iptables.rules" DESTINATION share/ion-tsconfig )
install(DIRECTORY "${PROJECT_SOURCE_DIR}/sge-configuration" DESTINATION share/ion-tsconfig
	PATTERN ".svn" EXCLUDE)

#################################
# Install python library files
#################################
FIND_PACKAGE(PythonInterp)
# Determine python site-packages or dist-packages directory
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
#install(DIRECTORY ion_tsconfig DESTINATION ${PYTHON_SITE_PACKAGES}
#    USE_SOURCE_PERMISSIONS
#    PATTERN ".svn" EXCLUDE
#)
install(FILES "${PROJECT_SOURCE_DIR}/ion_tsconfig/__init__.py" DESTINATION ${PYTHON_SITE_PACKAGES}/ion_tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/ion_tsconfig/TSconfig.py" DESTINATION ${PYTHON_SITE_PACKAGES}/ion_tsconfig )

include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

#% nice to automate this
set(CPACK_PACKAGE_DESCRIPTION "This package contains scripts for configuring Torrent Server.")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "expect")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
