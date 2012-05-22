# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

include(LibFindMacros)

# Dependencies
#libfind_package(MKL m)

# Include dir
find_path(INTEL_FLOAT_INCLUDE_DIR
  NAMES float.h
  PATHS /opt/intel/include
)
find_path(MKL_BLAS_INCLUDE_DIR
  NAMES mkl_blas.h
  PATHS /opt/intel/mkl/include
)
find_path(MKL_CBLAS_INCLUDE_DIR
  NAMES mkl_cblas.h
  PATHS /opt/intel/mkl/include
)
find_path(MKL_LAPACK_INCLUDE_DIR
  NAMES mkl_lapack.h
  PATHS /opt/intel/mkl/include
)
# the libraries
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
find_library(MKL_INTEL_LP64_LIBRARY
  NAMES mkl_intel_lp64
  PATHS /opt/intel/lib/intel64 /opt/intel/mkl/lib/intel64
)

find_library(MKL_SEQUENTIAL_LIBRARY
  NAMES mkl_sequential
  PATHS /opt/intel/mkl/lib/intel64
)
find_library(MKL_CORE_LIBRARY
  NAMES mkl_core 
  PATHS /opt/intel/mkl/lib/intel64
)
find_library(MKL_INTEL_THREAD
  NAMES mkl_intel_thread
  PATHS /opt/intel/mkl/lib/intel64
)
find_library(MKL_INTEL_OPENMP
  NAMES iomp5
  PATHS /opt/intel/lib/intel64
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries that this lib depends on.
set(MKL_PROCESS_INCLUDES INTEL_FLOAT_INCLUDE_DIR MKL_BLAS_INCLUDE_DIR MKL_CBLAS_INCLUDE_DIR MKL_LAPACK_INCLUDE_DIR MKL_INCLUDE_DIRS)

set(MKL_PROCESS_LIBS MKL_INTEL_LP64_LIBRARY MKL_SEQUENTIAL_LIBRARY MKL_CORE_LIBRARY MKL_LIBRARIES)

set(MKL_FIND_REQUIRED 1)
libfind_process(MKL)

