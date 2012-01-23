# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

include(LibFindMacros)

# Dependencies
#libfind_package(CLAPACK CBLAS)

# Include dir
find_path(CLAPACK_INCLUDE_DIR
  NAMES clapack.h
  PATHS /usr/include
  PATHS /usr/include/atlas
)

# Finally the library itself
find_library(CLAPACK_LIBRARY
  NAMES lapack
  PATHS /usr/lib /usr/lib64 /usr/lib/atlas /usr/lib64/atlas
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CLAPACK_PROCESS_INCLUDES CLAPACK_INCLUDE_DIR CLAPACK_INCLUDE_DIRS)
set(CLAPACK_PROCESS_LIBS CLAPACK_LIBRARY CLAPACK_LIBRARIES)
set(CLAPACK_FIND_REQUIRED 1)
libfind_process(CLAPACK)

