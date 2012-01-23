# Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
# Copyright (C) 2008-2010 Conrad Sanderson
# 
# This file is part of the Armadillo C++ library.
# It is provided without any warranty of fitness
# for any purpose. You can redistribute this file
# and/or modify it under the terms of the GNU
# Lesser General Public License (LGPL) as published
# by the Free Software Foundation, either version 3
# of the License or (at your option) any later version.
# (see http://www.opensource.org/licenses for more info)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(ARMA_MAJOR 1)
SET(ARMA_MINOR 1)
SET(ARMA_PATCH 0)

MESSAGE(STATUS "Configuring Armadillo ${ARMA_MAJOR}.${ARMA_MINOR}.${ARMA_PATCH}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/build_aux/cmake/Modules/")

INCLUDE(CheckIncludeFileCXX)
INCLUDE(CheckLibraryExists)
INCLUDE(FindBoost)

INCLUDE(ARMA_CheckProto)
INCLUDE(ARMA_CheckMathProto)

#SET( CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE )

PROJECT(armadillo CXX)

#CMAKE_REQUIRED_FLAGS = string of compile command line flags
#CMAKE_REQUIRED_DEFINITIONS = list of macros to define (-DFOO=bar)
#CMAKE_REQUIRED_LIBRARIES = list of libraries to link


SET(ARMA_USE_LAPACK false)
SET(ARMA_USE_BLAS   false)
SET(ARMA_USE_ATLAS  false)
SET(ARMA_USE_BOOST  false)


IF(WIN32)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "WARNING:")
  MESSAGE(STATUS "Automatic installation is currently not available for this platform.")
  MESSAGE(STATUS "Please use the manual installation, as described in the README.txt file.")
  MESSAGE(STATUS "You can also use the 'include' folder directly, but you may want to edit")
  MESSAGE(STATUS "'include/armadillo_bits/config.hpp' if you have LAPACK installed.")
  MESSAGE(STATUS "")
ENDIF(WIN32)



#
# Find LAPACK and BLAS libraries, or their optimised versions
#

IF(APPLE)
  SET(ARMA_OS macos)
  
  SET(ARMA_USE_LAPACK true)
  SET(ARMA_USE_BLAS   true)
  
  # Under MacOS, the current version of ARMA_FindCLAPACK can get confused between
  # two incompatible versions of "clapack.h" (one provided by the system and one
  # provided by ATLAS).  As such, use of ATLAS under MacOS is disabled for now.
  
ELSE(APPLE)
  SET(ARMA_OS unix)
  
  INCLUDE(ARMA_FindMKL)
  INCLUDE(ARMA_FindACMLMP)
  INCLUDE(ARMA_FindACML)
  
  MESSAGE(STATUS "MKL_FOUND     = ${MKL_FOUND}")
  MESSAGE(STATUS "ACMLMP_FOUND  = ${ACMLMP_FOUND}")
  MESSAGE(STATUS "ACML_FOUND    = ${ACML_FOUND}")
  
  IF(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)

    SET(ARMA_USE_BLAS true)
    SET(ARMA_USE_LAPACK true)

    MESSAGE(STATUS "*** If the MKL or ACML libraries are installed in a non-standard location,")
    MESSAGE(STATUS "*** make sure the run-time linker can find them.")
    MESSAGE(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
    MESSAGE(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
    MESSAGE(STATUS "*** On systems with SELinux enabled (e.g. Fedora, RHEL),")
    MESSAGE(STATUS "*** you may need to change the SELinux type of all MKL/ACML libraries")
    MESSAGE(STATUS "*** to fix permission problems that may occur during run-time.")
    MESSAGE(STATUS "*** See README.txt for more information")

  ELSE(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)
  
    INCLUDE(ARMA_FindLAPACK)
    INCLUDE(ARMA_FindBLAS)
    INCLUDE(ARMA_FindCLAPACK)
    INCLUDE(ARMA_FindCBLAS)

    MESSAGE(STATUS "LAPACK_FOUND  = ${LAPACK_FOUND}")
    MESSAGE(STATUS "BLAS_FOUND    = ${BLAS_FOUND}")
    MESSAGE(STATUS "CLAPACK_FOUND = ${CLAPACK_FOUND}")
    MESSAGE(STATUS "CBLAS_FOUND   = ${CBLAS_FOUND}")
    
    IF(LAPACK_FOUND)
      SET(ARMA_USE_LAPACK true)
    ENDIF(LAPACK_FOUND)
    
    IF(BLAS_FOUND)
      SET(ARMA_USE_BLAS true)
    ENDIF(BLAS_FOUND)
    
    IF(CLAPACK_FOUND AND CBLAS_FOUND)
      MESSAGE(STATUS "CLAPACK_INCLUDE_DIR = ${CLAPACK_INCLUDE_DIR}")
      MESSAGE(STATUS "CBLAS_INCLUDE_DIR   = ${CBLAS_INCLUDE_DIR}")
      IF(${CLAPACK_INCLUDE_DIR} STREQUAL ${CBLAS_INCLUDE_DIR})
        SET(ARMA_USE_ATLAS true)
        SET(ARMA_ATLAS_INCLUDE_DIR ${CLAPACK_INCLUDE_DIR})
      ENDIF(${CLAPACK_INCLUDE_DIR} STREQUAL ${CBLAS_INCLUDE_DIR})
    ENDIF(CLAPACK_FOUND AND CBLAS_FOUND)

  ENDIF(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)
  
ENDIF(APPLE)


IF(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)

  IF(MKL_FOUND)
    SET(ARMA_LIBS ${ARMA_LIBS} ${MKL_LIBRARIES})
 
    IF(ACMLMP_FOUND OR ACML_FOUND)
      MESSAGE(STATUS "*** Intel MKL as well as AMD ACML libraries were found.")
      MESSAGE(STATUS "*** Using only the MKL library to avoid linking conflicts.")
      MESSAGE(STATUS "*** If you wish to use ACML instead, please link manually with")
      MESSAGE(STATUS "*** acml or acml_mp instead of the armadillo run-time component.")
      MESSAGE(STATUS "*** Alternatively, remove MKL from your system and rerun")
      MESSAGE(STATUS "*** Armadillo's configuration using ./configure") 
    ENDIF(ACMLMP_FOUND OR ACML_FOUND)
    
  ELSE(MKL_FOUND)
    
    IF(ACMLMP_FOUND)
      SET(ARMA_LIBS ${ARMA_LIBS} ${ACMLMP_LIBRARIES})
      
      MESSAGE(STATUS "*** Both single-core and multi-core ACML libraries were found.")
      MESSAGE(STATUS "*** Using only the multi-core library to avoid linking conflicts.")
    ELSE(ACMLMP_FOUND)
      IF(ACML_FOUND)
        SET(ARMA_LIBS ${ARMA_LIBS} ${ACML_LIBRARIES})
      ENDIF(ACML_FOUND)
    ENDIF(ACMLMP_FOUND)

  ENDIF(MKL_FOUND)

ELSE(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)

  IF(ARMA_USE_BLAS STREQUAL true)
    SET(ARMA_LIBS ${ARMA_LIBS} ${BLAS_LIBRARIES})
  ENDIF(ARMA_USE_BLAS STREQUAL true)
  
  IF(ARMA_USE_LAPACK STREQUAL true)
    SET(ARMA_LIBS ${ARMA_LIBS} ${LAPACK_LIBRARIES})
  ENDIF(ARMA_USE_LAPACK STREQUAL true)
  
  IF(ARMA_USE_ATLAS STREQUAL true)
    SET(ARMA_LIBS ${ARMA_LIBS} ${CBLAS_LIBRARIES})
    SET(ARMA_LIBS ${ARMA_LIBS} ${CLAPACK_LIBRARIES})
  ENDIF(ARMA_USE_ATLAS STREQUAL true)

ENDIF(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)


IF(APPLE)
  SET(ARMA_LIBS ${ARMA_LIBS} "-framework Accelerate")  # or "-framework accelerate" ?
  MESSAGE(STATUS "MacOS X detected. Added '-framework Accelerate' to compiler flags")
ENDIF(APPLE)

MESSAGE(STATUS "*** ")
MESSAGE(STATUS "*** The Armadillo run-time library component will be an alias for the following libraries:")
MESSAGE(STATUS "*** ARMA_LIBS = ${ARMA_LIBS}")
MESSAGE(STATUS "*** ")

FIND_PACKAGE(Boost)

IF(Boost_FOUND)
  
  MESSAGE(STATUS "Boost_MAJOR_VERSION = ${Boost_MAJOR_VERSION}")
  MESSAGE(STATUS "Boost_MINOR_VERSION = ${Boost_MINOR_VERSION}")
  
  IF(Boost_MAJOR_VERSION GREATER 0)
    IF(Boost_MINOR_VERSION GREATER 33)
      SET(ARMA_USE_BOOST true)
      MESSAGE(STATUS "Boost_INCLUDE_DIR = ${Boost_INCLUDE_DIR}")
    ENDIF(Boost_MINOR_VERSION GREATER 33)
  ENDIF(Boost_MAJOR_VERSION GREATER 0)
  
ENDIF(Boost_FOUND)

IF(ARMA_USE_BOOST STREQUAL false)
  MESSAGE(STATUS "Boost libraries either not found or their version is too low (version 1.34 or later is good).")
  MESSAGE(STATUS "( It's possible that CMake didn't find the particular version of Boost you may have. )")
  MESSAGE(STATUS "( If that's the case, please edit include/armadillo_bits/config.hpp manually. )")
ENDIF(ARMA_USE_BOOST STREQUAL false)


# If Boost libraries were found, explicitly check if Boost's date_time library is also present.
# This is due to the non-standard packaging of Boost libraries on Debian and Ubuntu systems,
# where certain individual Boost libraries are packaged separately.

IF(ARMA_USE_BOOST STREQUAL true)
  SET(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${Boost_INCLUDE_DIR})
  CHECK_INCLUDE_FILE_CXX("boost/date_time/posix_time/posix_time.hpp" ARMA_USE_BOOST_DATE)
ENDIF(ARMA_USE_BOOST STREQUAL true)

ARMA_CHECK_MATH_PROTO("isfinite" "std" "cmath"  ARMA_HAVE_STD_ISFINITE)
ARMA_CHECK_MATH_PROTO("isinf"    "std" "cmath"  ARMA_HAVE_STD_ISINF)
ARMA_CHECK_MATH_PROTO("isnan"    "std" "cmath"  ARMA_HAVE_STD_ISNAN)
ARMA_CHECK_MATH_PROTO("log1p"    ""    "cmath"  ARMA_HAVE_LOG1P)

ARMA_CHECK_PROTO("snprintf"     "std" "cstdio"     ARMA_HAVE_STD_SNPRINTF)
ARMA_CHECK_PROTO("gettimeofday" ""    "sys/time.h" ARMA_HAVE_GETTIMEOFDAY)


MESSAGE(STATUS "Generating 'include/armadillo_bits/config.hpp'")
CONFIGURE_FILE(include/armadillo_bits/config.hpp.cmake include/armadillo_bits/config.hpp)

MESSAGE(STATUS "Generating 'examples/Makefile'")
CONFIGURE_FILE(examples/Makefile.cmake examples/Makefile)


IF(ARMA_USE_BOOST STREQUAL true)
  INCLUDE_DIRECTORIES(include ${Boost_INCLUDE_DIR})
ELSE(ARMA_USE_BOOST STREQUAL true)
  INCLUDE_DIRECTORIES(include)
ENDIF(ARMA_USE_BOOST STREQUAL true)


## For any library that is not in a default location,
## embed its path into the Armadillo runtime library.
## Examples of default locations are "/lib", "/usr/lib",
## or as specified in "/etc/ld.so.conf".
##
## Path embedding is not recommended unless you know
## what you're doing.  It might be better to add the
## path to the "/etc/ld.so.conf" file and then run "ldconfig".
##
#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH  FALSE)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE)


ADD_LIBRARY( armadillo SHARED src/pull_libs )
TARGET_LINK_LIBRARIES( armadillo ${ARMA_LIBS} )

SET_TARGET_PROPERTIES(armadillo PROPERTIES VERSION ${ARMA_MAJOR}.${ARMA_MINOR}.${ARMA_PATCH} SOVERSION 1)

# As Red Hat Enterprise Linux (and related systems such as Fedora)
# does not search /usr/local/lib by default, we need to place the
# library in /usr

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX "/usr")
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

MESSAGE(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")


# Allow for the "lib" directory to be specified on the command line

IF(NOT LIB_INSTALL_DIR)
  IF(UNIX AND NOT APPLE)   # I don't know how Mac OS handles 64 bit systems
    IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
      MESSAGE(STATUS "Detected 64 bit system")
      SET(LIB_INSTALL_DIR "lib64")
    ELSE(CMAKE_SIZEOF_VOID_P EQUAL 8)
      SET(LIB_INSTALL_DIR "lib")
    ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)
  
  ELSE(UNIX AND NOT APPLE)
    SET(LIB_INSTALL_DIR "lib")
  ENDIF(UNIX AND NOT APPLE)
ENDIF(NOT LIB_INSTALL_DIR)

MESSAGE(STATUS "LIB_INSTALL_DIR = ${LIB_INSTALL_DIR}")

# Allow for the "include" directory to be specified on the command line

IF(NOT INCLUDE_INSTALL_DIR)
  SET(INCLUDE_INSTALL_DIR "include")
ENDIF(NOT INCLUDE_INSTALL_DIR)

MESSAGE(STATUS "INCLUDE_INSTALL_DIR = ${INCLUDE_INSTALL_DIR}")


# Note that the trailing / character in "include/" is critical

INSTALL(DIRECTORY include/ DESTINATION ${INCLUDE_INSTALL_DIR}
PATTERN ".svn" EXCLUDE
PATTERN "*.cmake" EXCLUDE
PATTERN "*~" EXCLUDE
PATTERN "*orig" EXCLUDE
)


INSTALL(TARGETS armadillo DESTINATION ${LIB_INSTALL_DIR})



