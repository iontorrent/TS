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
	"${PROJECT_SOURCE_DIR}/ts_functions"
	"${PROJECT_BINARY_DIR}/ts_functions" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSsetproxy"
	"${PROJECT_BINARY_DIR}/bin/TSsetproxy" @ONLY
)

install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSchkconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSstaticip" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/bin/TSsetproxy" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSsetnoproxy" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSquery" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSdns" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSwrapper" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSinit" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TSaddstorage" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/ion" DESTINATION bin )
install(FILES "${PROJECT_BINARY_DIR}/ts_functions" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/ts_params" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/stresstest.tgz" DESTINATION share/ion-tsconfig )

install(FILES "${PROJECT_SOURCE_DIR}/bin/grp_functions" DESTINATION share/ion-tsconfig )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_validate.sh" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_terminal" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_update.sh" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_config_nodetest" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_connect_nodetest" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_configuration_test" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_mount_test" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/grp_addnodes.sh" DESTINATION share/ion-tsconfig/tools )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/prep_node_sudo.sh" DESTINATION share/ion-tsconfig/tools )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/add_compute.sh" DESTINATION share/ion-tsconfig/tools )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/del_compute.sh" DESTINATION share/ion-tsconfig/tools )
install(FILES "${PROJECT_SOURCE_DIR}/README.clusters" DESTINATION share/ion-tsconfig )

# perl modules used by plugins for json file parsing
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-2.53.tar.gz" DESTINATION share/ion-tsconfig/perl)
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-XS-2.32.tar.gz" DESTINATION share/ion-tsconfig/perl)
# perl modules used by plugins for ???
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/IO-All-0.44.tar.gz" DESTINATION share/ion-tsconfig/perl)
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/List-MoreUtils-0.33.tar.gz" DESTINATION share/ion-tsconfig/perl)
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Readonly-1.03.tar.gz" DESTINATION share/ion-tsconfig/perl)
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Statistics-Descriptive-3.0300.tar.gz" DESTINATION share/ion-tsconfig/perl)
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Statistics-LineFit-0.07.tar.gz" DESTINATION share/ion-tsconfig/perl)

# Deploy the ansible heirarchy for internal testing
install(DIRECTORY "${PROJECT_SOURCE_DIR}/ansible" DESTINATION share/ion-tsconfig
    USE_SOURCE_PERMISSIONS )

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
set(PYTHON_SITE_PACKAGES "${PYTHON_SITE_PACKAGES}" CACHE PATH "Python Install Prefix")
install(FILES "${PROJECT_SOURCE_DIR}/ion_tsconfig/__init__.py" DESTINATION ${PYTHON_SITE_PACKAGES}/ion_tsconfig )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/ion_tsconfig/TSconfig.py" DESTINATION ${PYTHON_SITE_PACKAGES}/ion_tsconfig )

include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "whiptail")

#% nice to automate this
set(CPACK_PACKAGE_DESCRIPTION "This package contains scripts for configuring Torrent Server.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
