# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project(ion-samita)

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()


set(PROJECT_DESCRIPTION "LifeScope samita library for Ion Torrent")

set (ION_TS_EXTERNAL "${PROJECT_SOURCE_DIR}/../../external/")
set(ION_UPDATE_SERVER "updates.iontorrent.com") # internal developer can use just "updates"
#custom ExternalProject_add with timeout feature
IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} STRLESS 2.8.1)
  SET(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${ION_TS_EXTERNAL}/CMake
  )
ENDIF ()
include(ExternalProject)

option(ION_USE_SYSTEM_HDF5 "Use hdf5 system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_HDF5)
option(ION_USE_SYSTEM_BOOST "Use boost system libraries" ON)
mark_as_advanced(ION_USE_SYSTEM_BOOST)
option(ION_USE_SYSTEM_SAMTOOLS "Use samtools system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_SAMTOOLS)


set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Werror -Wstrict-overflow=4 -Wstrict-aliasing=2 -O0 -DDEBUG -ftracer -rdynamic -g -fPIC -Wno-write-strings -Wno-unused-parameter -pipe")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -O2 -g -fPIC -Wno-write-strings -Wno-unused-parameter -DNDEBUG -O2 -pipe -march=nocona")
set(CMAKE_CXX_FLAGS_PROFILE "-Wall -Wextra -O2 -g -fPIC -Wno-write-strings -Wno-unused-parameter -lunwind -lprofiler -ftracer -rdynamic -pipe")
set(CMAKE_SHARED_LINKER_FLAGS "-Wall -as-needed -O2")

## Lifetech utility library
set(LIBLIFETECHFILE "${PROJECT_SOURCE_DIR}/../common/c++/lib/liblifetech.so")
add_custom_command(OUTPUT ${LIBLIFETECHFILE}
    COMMAND "make"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../common/c++/" )
add_library(LIBLIFETECH SHARED IMPORTED)
set_property(TARGET LIBLIFETECH PROPERTY IMPORTED_LOCATION ${LIBLIFETECHFILE})
include_directories("${PROJECT_SOURCE_DIR}/../common/c++/include")
include_directories("${PROJECT_SOURCE_DIR}/../3rdParty/include")
include_directories("${PROJECT_SOURCE_DIR}/")

find_package(ZLIB REQUIRED)

if(NOT ION_USE_SYSTEM_HDF5)
  set(proj_name "hdf5")
  set(proj_version "1.8.8")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/hdf5-1.8.8.tar.gz"
#    URL_MD5 1196e668f5592bfb50d1de162eb16cff
    CONFIGURE_COMMAND ${ION_TS_EXTERNAL}/${proj_name_version}/src/${proj_name}/configure
    --with-pic=yes --disable-shared --with-pthread --enable-threadsafe
    --prefix=${ION_TS_EXTERNAL}/${proj_name_version}
  )
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
  set(ION_HDF5_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libhdf5.a")
else()
  set(HDF5_USE_STATIC_LIBRARIES True)
  find_package(HDF5 REQUIRED)
  set(ION_HDF5_LIBS ${HDF5_LIBRARIES})
endif()

if(NOT ION_USE_SYSTEM_BOOST)
  set(proj_name "boost")
  set(proj_version "1.49.0")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    SOURCE_DIR "${ION_TS_EXTERNAL}/${proj_name_version}/src"
    SVN_REPOSITORY "http://svn.boost.org/svn/boost/tags/release/Boost_1_49_0"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ./bootstrap.sh
    BUILD_IN_SOURCE 1
    BUILD_COMMAND     ./bjam install
    --build-dir=${ION_TS_EXTERNAL}/${proj_name_version}/boost-build
    --prefix=${ION_TS_EXTERNAL}/${proj_name_version}
    --with-serialization
    INSTALL_COMMAND ""
    )
  set(Boost_LIBRARIES "${ION_TS_EXTERNAL}/${proj_name_version}/include")
else()
    find_package(Boost REQUIRED)
endif()
include_directories(${Boost_LIBRARIES})

if (NOT ION_USE_SYSTEM_SAMTOOLS)
  set(proj_name "samtools")
  set(proj_version "0.1.18")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}-prefix
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/samtools-0.1.18.tar.gz"
    #URL_MD5 7373854ca343ebee6a52874fcfc3a4a4
    # in source build
    BINARY_DIR "${ION_TS_EXTERNAL}/${proj_name_version}"
    SOURCE_DIR "${ION_TS_EXTERNAL}/${proj_name_version}"
    PATCH_COMMAND patch -p1 -t -N < "${ION_TS_EXTERNAL}/samtools-0.1.18.patch"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "/usr/bin/make" "CFLAGS=-fPIC -O2 -g -Wall -msse3 -march=native" "<SOURCE_DIR>/Makefile" "lib"
    INSTALL_COMMAND ""
  )
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}")
  set(LIBSAMTOOLS "${ION_TS_EXTERNAL}/${proj_name_version}/libbam.a")
else()
  set(LIBSAMTOOLS libbam.a)
  include_directories("/usr/include/samtools")
endif()

set(samita_sources
src/xsq/xsq_io.cpp
src/reference/reference.cpp
src/sam/bam_metadata.cpp
src/sam/bam.cpp
src/pileup/pileup.cpp
src/align/align_writer.cpp
src/align/align_reader.cpp
)

add_definitions(-DH5_USE_16_API)
add_library(samita SHARED ${samita_sources})
set_target_properties(samita PROPERTIES OUTPUT_NAME "samita++" CLEAN_DIRECT_OUTPUT 1 VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH} SOVERSION ${VERSION_MAJOR}.${VERSION_MINOR})

add_library(samita-static STATIC ${samita_sources})
set_target_properties(samita-static PROPERTIES OUTPUT_NAME "samita++" CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(samita ${LIBSAMTOOLS} ${LIBLIFETECH} ${Boost_LIBRARIES} log4cxx ${HDF5_LIBRARIES} ${ZLIB_LIBRARIES})
target_link_libraries(samita-static ${LIBSAMTOOLS} LIBLIFETECH ${Boost_LIBRARIES} log4cxx ${HDF5_LIBRARIES} ${ZLIB_LIBRARIES})

install(TARGETS samita LIBRARY DESTINATION lib COMPONENT libraries)

## Dev package only - static lib and includes
install(TARGETS samita-static ARCHIVE DESTINATION lib
        COMPONENT dev)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/samita" 
    DESTINATION include
    COMPONENT dev
    PATTERN ".svn" EXCLUDE)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/../common/c++/include/" 
    DESTINATION include
    PATTERN ".svn" EXCLUDE)


include(InstallRequiredSystemLibraries)

set(CPACK_GENERATOR "DEB")
set(CPACK_SET_DESTDIR "ON")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Ion Torrent <torrentdev@iontorrent.com>")
set(CPACK_PACKAGE_VENDOR "Ion Torrent Systems, Inc.")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(ARCH "amd64")
else()
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(CMAKE_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${ARCH}")


set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_GROUPING "IGNORE")
set(CPACK_RPM_USE_DISPLAY_NAME_IN_FILENAME ON)
set(CPACK_DEB_USE_DISPLAY_NAME_IN_FILENAME ON)

set(CPACK_COMPONENTS_ALL libraries dev)


set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_GROUPING "IGNORE")
set(CPACK_RPM_USE_DISPLAY_NAME_IN_FILENAME ON)
set(CPACK_DEB_USE_DISPLAY_NAME_IN_FILENAME ON)

set(CPACK_COMPONENTS_ALL libraries dev)


set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_GROUPING "IGNORE")
set(CPACK_RPM_USE_DISPLAY_NAME_IN_FILENAME ON)
set(CPACK_DEB_USE_DISPLAY_NAME_IN_FILENAME ON)

set(CPACK_COMPONENTS_ALL libraries dev)

set(CPACK_SOURCE_IGNORE_FILES
"/.svn/"
"/.git/"
"Makefile\\\\.in$"
"/build/"
)

set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "libboost-filesystem1.40.0,
    libboost-system1.40.0,
    libboost-regex1.40.0,
    libboost-iostreams1.40.0,
    libboost-thread1.40.0,
    libtbb2,
    liblog4cxx10,
    libhdf5-serial-1.8.4")
# Build deps for samita
set(CPACK_DEBIAN_PACKAGE_BUILDS_DEPENDS "libboost-filesystem-dev, libboost-system-dev, libboost-iostreams-dev, libboost-thread-dev, libhdf5-serial-dev, libtbb-dev, liblog4cxx10-dev, libcppunit-dev")

include(CPack)

