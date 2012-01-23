# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

include(LibFindMacros)

# Dependencies
#libfind_package(CBLAS m)

# Include dir
find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS /usr/include
)

# Finally the library itself
find_library(CBLAS_LIBRARY
  NAMES cblas
  PATHS /usr/lib /usr/lib64 /usr/lib/atlas /usr/lib64/atlas
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CBLAS_PROCESS_INCLUDES CBLAS_INCLUDE_DIR CBLAS_INCLUDE_DIRS)
set(CBLAS_PROCESS_LIBS CBLAS_LIBRARY CBLAS_LIBRARIES)
set(CBLAS_FIND_REQUIRED 1)
libfind_process(CBLAS)

