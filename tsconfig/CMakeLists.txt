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
	"${PROJECT_SOURCE_DIR}/TSconfig"
	"${PROJECT_BINARY_DIR}/TSconfig" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/TSchkconfig"
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
	"${PROJECT_SOURCE_DIR}/TSstaticip"
	"${PROJECT_BINARY_DIR}/TSstaticip" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/TSsetproxy"
	"${PROJECT_BINARY_DIR}/TSsetproxy" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/TSquery"
	"${PROJECT_BINARY_DIR}/TSquery" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/TSdns"
	"${PROJECT_BINARY_DIR}/TSdns" @ONLY
)


install(PROGRAMS "${PROJECT_BINARY_DIR}/TSconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSchkconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSstaticip" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSsetproxy" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSquery" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSdns" DESTINATION sbin )
install(FILES "${PROJECT_BINARY_DIR}/ts_functions" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_BINARY_DIR}/ts_params" DESTINATION share/ion-tsconfig )
install(DIRECTORY "${PROJECT_SOURCE_DIR}/sge-configuration" DESTINATION share/ion-tsconfig
	PATTERN ".svn" EXCLUDE)

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
