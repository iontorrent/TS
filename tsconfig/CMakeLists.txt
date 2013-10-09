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
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSinit"
	"${PROJECT_BINARY_DIR}/TSinit" @ONLY
)
configure_file (
	"${PROJECT_SOURCE_DIR}/bin/TSaddstorage"
	"${PROJECT_BINARY_DIR}/TSaddstorage" @ONLY
)


install(PROGRAMS "${PROJECT_BINARY_DIR}/TSconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSchkconfig" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSstaticip" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSsetproxy" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSquery" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSdns" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSwrapper" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSinit" DESTINATION sbin )
install(PROGRAMS "${PROJECT_BINARY_DIR}/TSaddstorage" DESTINATION sbin )
install(PROGRAMS "${PROJECT_SOURCE_DIR}/bin/TScreateLocalRepository" DESTINATION sbin )
install(FILES "${PROJECT_BINARY_DIR}/ts_functions" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_BINARY_DIR}/ts_params" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/config/iptables.rules" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/config/torrentsuite-packagelist.json" DESTINATION share/ion-tsconfig )
install(FILES "${PROJECT_SOURCE_DIR}/stresstest.tgz" DESTINATION share/ion-tsconfig )
install(DIRECTORY "${PROJECT_SOURCE_DIR}/sge-configuration" DESTINATION share/ion-tsconfig
    PATTERN ".svn" EXCLUDE )

# perl modules used by plugins for json file parsing
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/common-sense-3.4.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-2.53.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-XS-2.32.tar.gz" DESTINATION "/tmp/perlModIn")
# perl modules used by plugins for ???
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/autodie-2.10.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Carp-Assert-0.20.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/IO-All-0.44.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/IPC-System-Simple-1.21.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/List-MoreUtils-0.33.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Log-Log4perl-1.35.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Perl6-Slurp-0.050000.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Readonly-1.03.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Regexp-Common-2011121001.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Statistics-Descriptive-3.0300.tar.gz" DESTINATION "/tmp/perlModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Statistics-LineFit-0.07.tar.gz" DESTINATION "/tmp/perlModIn")
# Ignore pending verification that TS-5435 resolves the issue, instead of this fix
#install(FILES "${PROJECT_SOURCE_DIR}/config/20-disable-ipv6.conf" DESTINATION "/etc/sysctl.d")
# postgresql-shm config file: configures kernel memory compatible with our postgres memory settings
install(FILES "${PROJECT_SOURCE_DIR}/config/10-postgresql-shm.conf" DESTINATION "/etc/sysctl.d")
# Install prolog/epilog scripts
install(DIRECTORY "${PROJECT_SOURCE_DIR}/gridengine-scripts" DESTINATION share/ion-tsconfig
    USE_SOURCE_PERMISSIONS
    PATTERN ".svn" EXCLUDE )

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
# TODO: putty-tools is only here so that it gets installed with 3.0 from 2.2 GUI update
set(CPACK_DEBIAN_PACKAGE_DEPENDS "expect, gcc, make, python-jsonpipe, putty-tools, lsscsi, xfsprogs, lvm2")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
