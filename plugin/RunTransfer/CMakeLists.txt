# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)

# derive the plugin name from the folder name
get_filename_component(PLUGIN_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
PROJECT(${PLUGIN_NAME})

# this method will search the plugin script for it's version number and insert that into the debian package
include(../cmake/CMakeLists.PluginGetVersion.txt)

# clean the plugin name of illegal characters
STRING(REPLACE "_" "" DEB_PLUGIN_NAME ${PLUGIN_NAME})
STRING(TOLOWER ${DEB_PLUGIN_NAME} DEB_PLUGIN_NAME)

SET(CPACK_PACKAGE_NAME "ion-plugin-${DEB_PLUGIN_NAME}")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_PACKAGE_VENDOR "Ion Torrent")
SET(CPACK_PACKAGE_VERSION "${VERSION}")
SET(CPACK_SET_DESTDIR "ON")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "/results/plugins")
SET(CPACK_PACKAGE_FILE_NAME "ion-plugin-${DEB_PLUGIN_NAME}-${VERSION}")
SET(CPACK_DEBIAN_PACKAGE_SECTION "ion-plugin")
SET(CPACK_DEBIAN_PACKAGE_REPLACES "ion-plugins (<< 5.1.6)")
SET(CPACK_DEBIAN_PACKAGE_BREAKS "ion-plugins (<< 5.1.6)")

# this post install script is special due to a grooming algorithm which needs to be called
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/debian/postinst.in" "${PROJECT_BINARY_DIR}/debian/postinst" @ONLY)

CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/../debian/prerm.in"    "${PROJECT_BINARY_DIR}/debian/prerm"    @ONLY)
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/../debian/preinst.in"  "${PROJECT_BINARY_DIR}/debian/preinst"  @ONLY)
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/../debian/postinst.in"  "${PROJECT_BINARY_DIR}/debian/postinst"  @ONLY)

SET(
    CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/preinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
    )

IF(EXISTS ${PROJECT_SOURCE_DIR}/debian/changelog)
    ADD_CUSTOM_TARGET(compress_changelog ALL COMMAND gzip -c -9 ${PROJECT_SOURCE_DIR}/debian/changelog > ${PROJECT_SOURCE_DIR}/debian/changelog.gz)
    INSTALL(FILES debian/changelog.gz DESTINATION /usr/share/doc/${CPACK_PACKAGE_NAME})
ENDIF()

# these are the package specific attributues which cannot be automated
SET(CPACK_PACKAGE_DESCRIPTION "This plugin assembles reads into long sequences (contigs) and allows for basic level analysis, with metrics such as number of contigs, N50, and other analysis metrics.")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "python, ion-pipeline")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Brian Bourke-Martin")

# this will include the entire directory contents into the package
include(../cmake/CMakeLists.PluginInstallEverything.txt)

INCLUDE(CPack)
