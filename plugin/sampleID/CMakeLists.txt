# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)

# derive the plugin name from the folder name
get_filename_component(PLUGIN_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
PROJECT(${PLUGIN_NAME})

# this method will search the plugin script for it's version number and insert that into the debian package
include(../cmake/CMakeLists.PluginGetVersion.txt)

# set the standard package attributes
include(../cmake/CMakeLists.PluginSetAttributes.txt)

# these are the package specific attributues which cannot be automated
SET(CPACK_PACKAGE_DESCRIPTION "This plugin assembles reads into long sequences (contigs) and allows for basic level analysis, with metrics such as number of contigs, N50, and other analysis metrics.")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "python, ion-pipeline")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Brian Bourke-Martin")

# this will include the entire directory contents into the package
# except for bin directory as well as build files

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}"
        USE_SOURCE_PERMISSIONS
        REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/build" EXCLUDE
        REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" EXCLUDE
        REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/debian" EXCLUDE
		REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/bin" EXCLUDE
       )
INSTALL(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/bin/wkhtmltopdf-amd64"
        DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}/${PLUGIN_NAME}/bin"
       )


INCLUDE(CPack)
