# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)

#this must be here for TS automated build scripts
enable_testing()

# derive the plugin name from the folder name
get_filename_component(PLUGIN_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
PROJECT(${PLUGIN_NAME})

# this method will search the plugin script for it's version number and insert that into the debian package
include(../cmake/CMakeLists.PluginGetVersion.txt)

# set the standard package attributes
include(../cmake/CMakeLists.PluginSetAttributes.txt)

# these are the package specific attributes which cannot be automated
SET(CPACK_PACKAGE_DESCRIPTION "Torrent Suite plugin which generates files for customer support.")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "python")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alex Roemer <alexander.roemer@thermofisher.com>")

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CPACK_PACKAGE_INSTALL_DIRECTORY}"
        USE_SOURCE_PERMISSIONS
        REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/README.md" EXCLUDE
        REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/fabfile.py" EXCLUDE
       )

INCLUDE(CPack)
