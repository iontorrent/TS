# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-tsvm)
set(PROJECT_DESCRIPTION "Ion Torrent Torrent Server Virtual Machine")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
#include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/results/tsvm" CACHE PATH "Ion Install Prefix")
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
    "${PROJECT_SOURCE_DIR}/debian/postrm.in"
    "${PROJECT_BINARY_DIR}/debian/postrm" @ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/private_key_rsa ${PROJECT_BINARY_DIR}/.private_key_rsa)

install(
	FILES "${PROJECT_SOURCE_DIR}/Vagrantfile"
	DESTINATION ${ION_INSTALL_PREFIX}
)
install(
	FILES "${PROJECT_BINARY_DIR}/.private_key_rsa"
	DESTINATION ${ION_INSTALL_PREFIX}
	PERMISSIONS OWNER_READ OWNER_WRITE
)
install(
	FILES "${PROJECT_SOURCE_DIR}/tsvm-postprov.sh"
	"${PROJECT_SOURCE_DIR}/tsvm-setup"
	"${PROJECT_SOURCE_DIR}/tsvm-status"
	"${PROJECT_SOURCE_DIR}/tsvm-versions"
	"${PROJECT_SOURCE_DIR}/tsvm-ctrl"
	"${PROJECT_SOURCE_DIR}/tsvm-include"
	"${PROJECT_SOURCE_DIR}/tsvm-isvtenabled"
	"${PROJECT_SOURCE_DIR}/tsvm-prepare"
	"${PROJECT_SOURCE_DIR}/tsvm-enable-nfs-rule"
	DESTINATION ${ION_INSTALL_PREFIX}
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
install(
	FILES "${PROJECT_SOURCE_DIR}/../dbReports/iondb/bin/serialize_runs_to_json.py"
	DESTINATION ${ION_INSTALL_PREFIX}
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
install(
	FILES "${PROJECT_SOURCE_DIR}/vagrant_sudoers"
	DESTINATION /etc/sudoers.d/
	PERMISSIONS OWNER_READ GROUP_READ
)

set(CPACK_GENERATOR "DEB")

include(../buildTools/cmake/CMakeLists.cpack.txt)
set(CPACK_DEBIAN_PACKAGE_DEPENDS
       "vagrant,
        virtualbox-4.3 (>=4.3),
        msr-tools,
        dmidecode"
    )
set(CPACK_PACKAGE_DESCRIPTION "This package configures the Ion Torrent Suite to download the Early Access Software version.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
	${CMAKE_CURRENT_BINARY_DIR}/debian/postrm
)
include(CPack)
