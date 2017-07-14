# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-plugins)

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.cpack.txt)

#this must be here for TS automated build scripts
enable_testing()

# create the debian meta package which is simply an empty package with dependencies to all of the individual plugin packages
SET(CPACK_PACKAGE_NAME "ion-plugins")
SET(CPACK_PACKAGE_VENDOR "Ion Torrent")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Brian Bourke-Martin")

# go through each of the plugins and do a manual build for each of them
SET(PLUGIN_DIRS ampliSeqRNA AssemblerSPAdes coverageAnalysis DataExport ERCC_Analysis FieldSupport FileExporter FilterDuplicates immuneResponseRNA PGxAnalysis RNASeqAnalysis RunTransfer sampleID variantCaller)

FOREACH(PLUGIN_DIR ${PLUGIN_DIRS})
    #create a clean version of the plugin directory
    STRING(REPLACE "_" "" DEB_PLUGIN_NAME ${PLUGIN_DIR})

    ADD_CUSTOM_TARGET(package${DEB_PLUGIN_NAME}
                      ALL
                      COMMAND mkdir -p ${CMAKE_BINARY_DIR}/${PLUGIN_DIR}
                      COMMAND cd ${CMAKE_BINARY_DIR}/${PLUGIN_DIR} && cmake -DCPACK_OUTPUT_FILE_PREFIX:STRING=${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/${PLUGIN_DIR} && make package
                     )

    if(CPACK_DEBIAN_PACKAGE_DEPENDS)
        SET(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, ion-plugin-${DEB_PLUGIN_NAME}")
    else()
        SET(CPACK_DEBIAN_PACKAGE_DEPENDS "ion-plugin-${DEB_PLUGIN_NAME}")
    endif()

ENDFOREACH()

STRING(TOLOWER ${CPACK_DEBIAN_PACKAGE_DEPENDS} CPACK_DEBIAN_PACKAGE_DEPENDS)
include(CPack)
