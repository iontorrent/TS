# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required(VERSION 2.6)
project(variantCallerPlugin)


message("Custom Installation Procedure for variantCaller")
 
add_custom_target(parameter_sets ALL
                   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/pluginMedia/parameter_sets/build_parameter_sets.py" "${CMAKE_CURRENT_BINARY_DIR}/parameter_sets.json"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pluginMedia/parameter_sets/"
                   COMMENT "Combining individual parameter sets into a single parameter_sets.json")


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION "plugins"
    USE_SOURCE_PERMISSIONS
    PATTERN ".svn" EXCLUDE
    PATTERN "parameter_sets.json" EXCLUDE
    REGEX "src$" EXCLUDE
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/parameter_sets.json" DESTINATION "plugins/variantCaller/pluginMedia/parameter_sets")




