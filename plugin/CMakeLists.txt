# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-plugins)
set(PROJECT_DESCRIPTION "Ion Torrent Plugins")
enable_testing()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/results" CACHE PATH "Ion Install Prefix")
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

set(PLUGINS
Alignment
RunRecognitION
FastqCreator
SFFCreator
coverageAnalysis
sampleID
variantCaller
IonReporterUploader_V1_0
ERCC_Analysis
)

foreach(PLUGIN ${PLUGINS})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN}")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN}/CMakeLists.txt")
            message(STATUS "Plugin ${PLUGIN} has custom build script.")
            add_subdirectory(${PLUGIN})
        else()
            message("Installing Plugin ${PLUGIN}")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN}" DESTINATION "plugins"
                USE_SOURCE_PERMISSIONS
                PATTERN ".svn" EXCLUDE
                REGEX "src$" EXCLUDE
            )
        endif()
    endif()
endforeach(PLUGIN)


install(FILES "${PROJECT_SOURCE_DIR}/plugin_functions" DESTINATION "plugins")

# ZeroInstall alternate application root - set systemwide store to
# /results/plugins/implementations (or whatever is top of implementation-dirs)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/implementations)
install(DIRECTORY "${PROJECT_BINARY_DIR}/implementations" DESTINATION "plugins" 
        DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE SETGID WORLD_READ WORLD_EXECUTE)
install(FILES "${PROJECT_SOURCE_DIR}/implementation-dirs" DESTINATION "/etc/xdg/0install.net/injector/")

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "perl, python, ion-torrentr, ion-samita, ion-pipeline, ion-dbreports, ion-igv, python-pysam (>= 0.7)")
set(CPACK_PACKAGE_DESCRIPTION "Plugins for the Torrent Browser. These plugins, when present will be executed, post analysis pipeline run, to generate additional results.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
	${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
