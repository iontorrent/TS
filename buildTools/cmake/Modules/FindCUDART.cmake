# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

include(LibFindMacros)

# Dependencies
#libfind_package(CBLAS m)

# Include dir
find_path(CUDART_INCLUDE_DIR
  NAMES cuda_runtime_api.h
  PATHS /usr/include /usr/local/cuda/include
)

# Finally the library itself
find_library(CUDART_LIBRARY
  NAMES cudart
  PATHS /usr/lib /usr/lib64 /usr/lib/atlas /usr/lib64/atlas /usr/local/cuda/lib /usr/local/cuda/lib64
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CUDART_PROCESS_INCLUDES CUDART_INCLUDE_DIR CUDART_INCLUDE_DIRS)
set(CUDART_PROCESS_LIBS CUDART_LIBRARY CUDART_LIBRARIES)
set(CUDART_FIND_REQUIRED 1)
libfind_process(CUDART)

