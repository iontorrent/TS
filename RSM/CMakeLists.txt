# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-rsmts)
set(PROJECT_DESCRIPTION "Ion Torrent Remote Monitoring")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion" CACHE PATH "Ion Install Prefix")
set(CMAKE_INSTALL_PREFIX "${ION_INSTALL_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")
set(AGENT_DIR "rsmagent")

configure_file (
    "${PROJECT_SOURCE_DIR}/debian/postinst.in"
    "${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)
configure_file (
    "${PROJECT_SOURCE_DIR}/debian/prerm.in"
    "${PROJECT_BINARY_DIR}/debian/prerm" @ONLY
)

set(AGENT_RSM "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/RSMAgent_TS")
execute_process(COMMAND "make" WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/${AGENT_DIR}")

install(PROGRAMS ${AGENT_RSM} DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/RSM_Launch" DESTINATION /etc/init.d)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/RSMAgent.sh" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/location_helper.sh" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/status.py" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/queryFileServers.py" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/find_serial.py" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/updateContactInfo.py" DESTINATION RSM)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/${AGENT_DIR}/updateContactInfo.py" DESTINATION RSM)

include(../buildTools/cmake/CMakeLists.cpack.txt)
set(CPACK_PACKAGE_DESCRIPTION "ion-rsmts provides remote monitoring for the Torrent Server.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
