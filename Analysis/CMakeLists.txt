# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-analysis)
set(PROJECT_DESCRIPTION "Ion Torrent Data Processing Pipeline")
enable_testing()

option(ION_DO_BACKGROUND "Build with background model" ON)
mark_as_advanced(ION_DO_BACKGROUND)

option(ION_USE_CUDA "Compile CUDA code" ON)
mark_as_advanced(ION_USE_CUDA)

option(ION_USE_SYSTEM_CUDA "Use CUDA system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_CUDA)

option(ION_USE_SYSTEM_SAMTOOLS "Use samtools system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_SAMTOOLS)

option(ION_USE_SYSTEM_BAMTOOLS "Use bamtools system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_BAMTOOLS)

option(ION_USE_SYSTEM_FFTW "Use fftw system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_FFTW)

option(ION_USE_SYSTEM_HDF5 "Use hdf5 system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_HDF5)

option(ION_USE_SYSTEM_ARMADILLO "Use armadillo system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_ARMADILLO)

option(ION_USE_SYSTEM_BOOST "Use boost system libraries" ON)
mark_as_advanced(ION_USE_SYSTEM_BOOST)

option(ION_USE_SYSTEM_UNWIND "Use unwind system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_UNWIND)

option(ION_BUILD_STRAND_BIAS "Build samita and Strand_Bias_Extractor" OFF)
mark_as_advanced(ION_BUILD_STRAND_BIAS)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ION_DO_BACKGROUND OFF CACHE BOOL "Build with background model")
endif()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion" CACHE PATH "Ion Install Prefix")

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

if(ION_USE_CUDA)
    message(STATUS "BUILD with CUDA ${CUDA_VERSION}")
    add_definitions(-DION_COMPILE_CUDA)
    if (NOT ION_USE_SYSTEM_CUDA)

        set(CUDA_TOOLKIT_ROOT_DIR "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/")
        set(CUDA_INCLUDE_DIRS "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/include")
        set(CUDA_NVCC_EXECUTABLE "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/bin/nvcc")
        set(CUDA_CUDART_LIBRARY "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/lib64/libcudart.so")
        set(CUDA_TOOLKIT_INCLUDE "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/include")
        set(CUDA_cublas_LIBRARY "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/lib64/libcublas.so")
        set(CUDA_cufft_LIBRARY "${ION_TS_EXTERNAL}/${cuda_proj_name_version}/src/cudatoolkit/lib64/libcufft.so")
        set(CUDA_VERBOSE_BUILD OFF)
        set(CUDA_64_BIT_DEVICE_CODE ON)
        include(${CMAKE_ROOT}/Modules/FindCUDA.cmake)

    else()
        find_package(CUDA REQUIRED)
    endif()
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

if (NOT ION_USE_SYSTEM_FFTW)
  set(proj_name "fftw")
  set(proj_version "3.3")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/fftw-3.3.tar.gz"
#    URL_MD5 "0728ab3ec0ebb06631ab3b80a5c3d130"
    CONFIGURE_COMMAND ${ION_TS_EXTERNAL}/${proj_name_version}/src/${proj_name}/configure
    --enable-static --with-pic
    --prefix=${ION_TS_EXTERNAL}/${proj_name_version}
  )
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
  set(ION_FFTW_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libfftw3.a")
else()
  set(ION_FFTW_LIBS libfftw3.a)
endif()

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
  set(ION_HDF5_HL_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libhdf5_hl.a")
else()
  set(HDF5_USE_STATIC_LIBRARIES True)
  find_package(HDF5 REQUIRED)
  set(ION_HDF5_LIBS ${HDF5_LIBRARIES})
endif()

if(NOT ION_USE_SYSTEM_ARMADILLO)
  set(proj_name "armadillo")
  set(proj_version "3.0.2")
  set(proj_name_version "${proj_name}-${proj_version}")

  if (${LSB_RELEASE_CODENAME} STREQUAL "oneiric" OR ${LSB_RELEASE_CODENAME} STREQUAL "precise")
        set(ARMADILLO_PATCH "${PROJECT_SOURCE_DIR}/../external/armadillo-2.4.0~precise.patch")
  else()
        set(ARMADILLO_PATCH "${PROJECT_SOURCE_DIR}/../external/armadillo-2.4.0.patch")
  endif()

  ExternalProject_add(${proj_name}_proj
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/armadillo-3.0.2.tar.gz"
    #URL_MD5 e8400bc7adb65fa0edc8c8ccb49b60eb
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ION_TS_EXTERNAL}/${proj_name_version}
    PATCH_COMMAND patch -p1 -t -N < "${ARMADILLO_PATCH}"
  )
  set(ION_ARMADILLO_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libarmadillo.so")
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
else()
  set(ION_ARMADILLO_LIBS "armadillo")
#include_directories("/home/csugnet/local/src/armadillo-3.0.2/include")
#set(ION_ARMADILLO_LIBS "/home/csugnet/local/src/armadillo-3.0.2/libarmadillo.so")
endif()

set(LIBKMEANSFILE "${PROJECT_SOURCE_DIR}/../external/kmeans-1.7/libkmeans.a")
add_custom_command(OUTPUT ${LIBKMEANSFILE}
    COMMAND "make"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/kmeans-1.7")
add_library(LIBKMEANS STATIC IMPORTED)
set_property(TARGET LIBKMEANS PROPERTY IMPORTED_LOCATION ${LIBKMEANSFILE})
include_directories("${PROJECT_SOURCE_DIR}/../external/kmeans-1.7")

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
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
  set(ION_BOOST_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libboost_serialization.a")
else()
  find_package(Boost REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS})
  set(ION_BOOST_LIBS "boost_serialization.a")
endif()

if(NOT ION_USE_SYSTEM_UNWIND)
  set(proj_name "unwind")
  set(proj_version "1.0.1")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/libunwind-1.0.1.tar.gz"
    CONFIGURE_COMMAND ${ION_TS_EXTERNAL}/${proj_name_version}/src/${proj_name}/configure
    CFLAGS=-D_FORTIFY_SOURCE=0
    --with-pic=yes --disable-shared
    --prefix=${ION_TS_EXTERNAL}/${proj_name_version}
  )
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
  set(ION_UNWIND_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libunwind.a")
else()
  set(UNWIND_USE_STATIC_LIBRARIES True)
  find_package(UNWIND REQUIRED)
  set(ION_UNWIND_LIBS ${UNWIND_LIBRARIES})
endif()

# uncomment for profiling if gprofile and unwind installed
# add_library(UNWIND STATIC IMPORTED)
# set_property(TARGET UNWIND PROPERTY IMPORTED_LOCATION "/usr/local/lib/libunwind.a")
# add_library(PROFILE STATIC IMPORTED)
# set_property(TARGET PROFILE PROPERTY IMPORTED_LOCATION "/usr/local/lib/libprofiler.a")

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../buildTools")
include_directories("${PROJECT_SOURCE_DIR}/../external/tnt-1.2.6")
#include_directories("${PROJECT_SOURCE_DIR}/../external/fstrcmp")
include_directories("${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1")


if (NOT ION_USE_SYSTEM_SAMTOOLS)
  set(proj_name "samtools")
  set(proj_version "0.1.18")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}-prefix
    URL "http://${ION_UPDATE_SERVER}/updates/software/external/samtools-0.1.18.tar.gz"
    #URL_MD5 7373854ca343ebee6a52874fcfc3a4a4
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ION_TS_EXTERNAL}/${proj_name_version} -DBUILD_SHARED_LIBS:BOOL=OFF
    PATCH_COMMAND patch -p0 -t -N < "${ION_TS_EXTERNAL}/samtools-0.1.18.patch"
  )
  include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
  set(ION_SAMTOOLS_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libbam.a")
else()
    set(ION_SAMTOOLS_LIBS libbam.a)
    include_directories("/usr/include/samtools")
endif()


if (NOT ION_USE_SYSTEM_BAMTOOLS)
    set(proj_name "bamtools")
    set(revision_tag be1727d1e5)#set date below accordingly
    set(proj_version "2.1.1.20121113+git${revision_tag}")
    set(proj_name_version "${proj_name}-${proj_version}")
    ExternalProject_add(${proj_name}
      URL "http://${ION_UPDATE_SERVER}/updates/software/external/bamtools-2.1.1.20121113+gitbe1727d1e5.tar.gz"
#      GIT_REPOSITORY "https://github.com/iontorrent/bamtools.git"
#      GIT_TAG ${revision_tag}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${ION_TS_EXTERNAL}/${proj_name_version}
      PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}-prefix
    )
    include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include/bamtools")
    set(ION_BAMTOOLS_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/bamtools/libbamtools.a")
else()
    set(ION_BAMTOOLS_LIBS libbamtools.a)
    include_directories("/usr/include/bamtools")
endif()


include_directories("${PROJECT_SOURCE_DIR}/Replay")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg/justBeadFind")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg/IO")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel")
include_directories("${PROJECT_SOURCE_DIR}/SynchDat")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/MathModel")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Bookkeeping")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/LocalTrace")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Fitters")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Fitters/Complex")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Writers")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA")
include_directories("${PROJECT_SOURCE_DIR}/BaseCaller")
include_directories("${PROJECT_SOURCE_DIR}/Separator")
include_directories("${PROJECT_SOURCE_DIR}/Util")
#this has enough files now to justify own directory
include_directories("${PROJECT_SOURCE_DIR}/Image")
include_directories("${PROJECT_SOURCE_DIR}/Mask")
include_directories("${PROJECT_SOURCE_DIR}/Wells")

# CUDA Files
if(ION_USE_CUDA)
    ## NOTE: -O2 enables strict-aliasing, and code is not clean
    SET( CUDA_NVCC_FLAGS "-O2" "--compiler-options=-fno-strict-aliasing" "-g" "--use_fast_math" "-gencode=arch=compute_20,code=compute_20" "-maxrregcount=48" )
    add_definitions(-malign-double) ## See CUDA 4.0 Release Nodes
    ## If user set CXX on command line, make sure NVCC respects it
    ## (necessary for compiling on systems where default compiler is not nvcc compatible)
    if($ENV{CXX} MATCHES ".+")
        LIST(APPEND CUDA_NVCC_FLAGS "--compiler-bindir=$ENV{CXX}")
    endif() #($ENV{CXX} MATCHES ".+")
    CUDA_COMPILE(CUDA_TEMP_FILES 
      BkgModel/CUDA/StreamManager.cu 
      BkgModel/CUDA/SingleFitStream.cu
      BkgModel/CUDA/MultiFitStream.cu
      BkgModel/CUDA/StreamingKernels.cu
      SHARED
    )
endif()

# Ion Analysis Library
add_library(ion-analysis
    ${LIBKMEANSFILE}

    ${PROJECT_BINARY_DIR}/IonVersion.cpp

    Wells/RawWells.cpp
    Wells/RawWellsV1.cpp

    Image/deInterlace.cpp
    Image/Image.cpp
    Image/TikhonovSmoother.cpp
    Image/PreDefinedTikParams.cpp
    Image/ChipIdDecoder.cpp
    Image/ImageTransformer.cpp
    Image/LSRowImageProcessor.cpp
    Image/ComparatorNoiseCorrector.cpp

    
    Mask/PinnedInFlow.cpp
    Mask/Mask.cpp
    Mask/ComplexMask.cpp
    
    Separator/DualGaussMixModel.cpp
    Separator/Separator.cpp
    Separator/DifferentialSeparator.cpp
    Separator/KeyClassifier.cpp
    Separator/KClass.cpp
    Separator/BFReference.cpp
    Separator/Traces.cpp

    Util/H5File.cpp
    Util/IonErr.cpp
    Util/Utils.cpp
    Util/HandleExpLog.cpp
    Util/OptArgs.cpp
    Util/OptionArgsBase.cpp
    Util/OptionArgs.cpp
    Util/LoggerBase.cpp
    Util/Logger.cpp
    Util/bivariate_gaussian.cpp
    Util/flow_utils.cpp
    Util/WorkerInfoQueue.cpp
    Util/StackUnwind.cpp
    Util/SeqUtils.cpp

    AnalysisOrg/IO/CommandLineOpts.cpp
    AnalysisOrg/IO/KeyContext.cpp
    AnalysisOrg/IO/FlowContext.cpp
    AnalysisOrg/IO/BkgControlOpts.cpp
    AnalysisOrg/IO/BeadfindControlOpts.cpp
    AnalysisOrg/IO/ImageControlOpts.cpp
    AnalysisOrg/IO/BkgModelHdf5.cpp
    AnalysisOrg/IO/TrackProgress.cpp
    AnalysisOrg/IO/ProgramState.cpp
    AnalysisOrg/IO/CaptureImageState.cpp
    
    AnalysisOrg/justBeadFind/SeparatorInterface.cpp
    AnalysisOrg/justBeadFind/SetUpForProcessing.cpp
    
    AnalysisOrg/SystemContext.cpp
    AnalysisOrg/SpatialContext.cpp
    AnalysisOrg/ImageSpecClass.cpp
    AnalysisOrg/ImageLoader.cpp
    AnalysisOrg/ImageLoaderQueue.cpp
    AnalysisOrg/ProcessImageToWell.cpp
    AnalysisOrg/RegionTimingCalc.cpp
    AnalysisOrg/SeqList.cpp
    AnalysisOrg/WellFileManipulation.cpp
    AnalysisOrg/GlobalDefaultsFromBkgControl.cpp
    AnalysisOrg/BkgDataPointers.cpp
    AnalysisOrg/BkgFitterTracker.cpp
    AnalysisOrg/SlicedPrequel.cpp
    AnalysisOrg/SignalProcessingFitterQueue.cpp
    AnalysisOrg/ClonalFilter.cpp
    AnalysisOrg/MaskFunctions.cpp
    AnalysisOrg/cudaWrapper.cpp
        
    Region.cpp
    LinuxCompat.cpp
    Stats.cpp

    LevMarFitterV2.cpp
    
    mixed.cpp
    # T0CalcMt.cpp
    
    BaseCaller/DPTreephaser.cpp
    
    BkgModel/MathModel/MathOptim.cpp
    BkgModel/MathModel/PoissonCdf.cpp
    BkgModel/MathModel/DNTPRiseModel.cpp
    BkgModel/MathModel/DiffEqModel.cpp
    BkgModel/MathModel/DiffEqModelVec.cpp
    BkgModel/MathModel/Hydrogen.cpp
    BkgModel/MathModel/MathUtil.cpp
    BkgModel/MathModel/MiscVec.cpp
    BkgModel/MathModel/MultiFlowModel.cpp

    BkgModel/Bookkeeping/BeadParams.cpp
    BkgModel/Bookkeeping/BeadTracker.cpp
    BkgModel/Bookkeeping/RegionParams.cpp
    BkgModel/Bookkeeping/RegionTracker.cpp
    BkgModel/Bookkeeping/FlowBuffer.cpp
    BkgModel/Bookkeeping/NucStepCache.cpp
    BkgModel/Bookkeeping/TimeCompression.cpp
    BkgModel/Bookkeeping/EmphasisVector.cpp
    BkgModel/Bookkeeping/DarkHalo.cpp
    BkgModel/Bookkeeping/CrossTalkSpec.cpp
    BkgModel/Bookkeeping/XtalkCurry.cpp
    BkgModel/Bookkeeping/TraceCurry.cpp
    BkgModel/Bookkeeping/BeadScratch.cpp

    BkgModel/LocalTrace/BkgTrace.cpp
    BkgModel/LocalTrace/EmptyTrace.cpp
    BkgModel/LocalTrace/TraceClassifier.cpp
    BkgModel/LocalTrace/EmptyTraceTracker.cpp

    BkgModel/Fitters/Complex/FitControl.cpp
    BkgModel/Fitters/Complex/MultiLevMar.cpp
    BkgModel/Fitters/Complex/LevMarState.cpp
    BkgModel/Fitters/Complex/BkgFitMatrixPacker.cpp
    BkgModel/Fitters/Complex/BkgFitStructures.cpp
    BkgModel/Fitters/Complex/BkgFitOptim.cpp
    
    BkgModel/Fitters/SingleFlowFit.cpp
    BkgModel/Fitters/RefineFit.cpp
    BkgModel/Fitters/RefineTime.cpp
    BkgModel/Fitters/SpatialCorrelator.cpp
    BkgModel/Fitters/BkgSearchAmplitude.cpp
    BkgModel/Fitters/DarkMatter.cpp
    BkgModel/Fitters/TraceCorrector.cpp

    BkgModel/Writers/DebugWriter.cpp
    BkgModel/Writers/GlobalWriter.cpp
    
    BkgModel/GlobalDefaultsForBkgModel.cpp
    BkgModel/SignalProcessingMasterFitter.cpp
    BkgModel/RegionalizedData.cpp

    BkgModel/CUDA/JobWrapper.cpp 
    BkgModel/CUDA/GpuMultiFlowFitControl.cpp
    BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp
   

    SynchDat/BitHandler.cpp
    SynchDat/HuffmanEncode.cpp
    SynchDat/SynchDatSerialize.cpp
    SynchDat/VencoLossless.cpp
    SynchDat/DeltaComp.cpp
    SynchDat/DeltaCompFst.cpp
    SynchDat/DeltaCompFstSmX.cpp
    SynchDat/SvdDatCompress.cpp
    SynchDat/DelicatoCompression.cpp
    SynchDat/compression.cpp
    SynchDat/matrixRounding.cpp
    SynchDat/unpackMatrix.cpp
    SynchDat/packMatrix.cpp
#    SynchDat/SvdDatCompressPlus.cpp 

    Replay/FileBkgReplay.cpp
    Replay/H5Replay.cpp
    Replay/PinnedInFlowReplay.cpp
    Replay/EmptyTraceReplay.cpp
    Replay/RegionTrackerReplay.cpp
    Replay/BkgModelReplay.cpp

    crop/Acq.cpp
    
    FileEquivalent.cpp
    
    #${PROJECT_SOURCE_DIR}/../external/fstrcmp/fstrcmp.cpp
    
    SFFWrapper.cpp
    
    SamUtils/types/Cigar.cpp
    SamUtils/types/Qual.cpp
    SamUtils/types/Sequence.cpp
    SamUtils/types/BAMRead.cpp
    SamUtils/types/MD.cpp
    SamUtils/BAMReader.cpp
    SamUtils/BAMUtils.cpp
    SamUtils/alignStats.cpp

    ${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
    ${CUDA_TEMP_FILES}
)
add_dependencies(ion-analysis IONVERSION samtools bamtools fftw armadillo_proj hdf5 boost unwind)
if(ION_USE_CUDA)
  if(NOT ION_USE_SYSTEM_CUDA)
    add_dependencies(ion-analysis cudatoolkit)
    # Changes J30
    #add_dependencies(ion-analysis cudatoolkit armadillo_proj)
  endif()
endif()

target_link_libraries(ion-analysis ${ION_FFTW_LIBS} ${ION_HDF5_LIBS} LIBKMEANS ${ION_BOOST_LIBS} ${ION_ARMADILLO_LIBS} ${ION_UNWIND_LIBS} blas lapack lapack_atlas atlas file-io z)
#target_link_libraries(ion-analysis ${ION_FFTW_LIBS} ${ION_HDF5_LIBS} LIBKMEANS ${ION_ARMADILLO_LIBS} blas lapack lapack_atlas atlas file-io z)

set_target_properties(ion-analysis PROPERTIES
    VERSION   "${ION_VERSION_MAJOR}.${ION_VERSION_MINOR}.${ION_VERSION_RELEASE}.${ION_VERSION_BUILDNUM}"
)
if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        set_target_properties(ion-analysis PROPERTIES COMPILE_FLAGS "-fPIC")
endif( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
install(TARGETS ion-analysis DESTINATION lib)


add_executable(Crop crop/Crop.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Crop IONVERSION)
target_link_libraries(Crop ion-analysis pthread)
install(TARGETS Crop DESTINATION bin)

add_executable(ChkDat crop/ChkDat.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(ChkDat IONVERSION)
target_link_libraries(ChkDat ion-analysis pthread)
install(TARGETS ChkDat DESTINATION bin)

add_executable(CmpDat crop/CmpDat.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CmpDat IONVERSION)
target_link_libraries(CmpDat ion-analysis pthread)
install(TARGETS CmpDat DESTINATION bin)

add_executable(Blocks crop/Blocks.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Blocks IONVERSION)
target_link_libraries(Blocks ion-analysis pthread)
install(TARGETS Blocks DESTINATION bin)

add_executable(MergeImages crop/MergeImages.cpp crop/MergeAcq.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeImages IONVERSION)
target_link_libraries(MergeImages ion-analysis pthread)
install(TARGETS MergeImages DESTINATION bin)

add_executable(MergeDats crop/MergeDats.cpp crop/MergeAcq.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeDats IONVERSION)
target_link_libraries(MergeDats ion-analysis pthread)
install(TARGETS MergeDats DESTINATION bin)

add_executable(Archive crop/Archive.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Archive IONVERSION)
target_link_libraries(Archive ion-analysis pthread)

if(ION_DO_BACKGROUND)
    add_executable(Analysis AnalysisOrg/Analysis.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
    add_dependencies(Analysis IONVERSION)

    if (ION_USE_CUDA)
        target_link_libraries(Analysis ion-analysis gsl gslcblas pthread lapack_atlas atlas  ${CUDA_LIBRARIES})
    else()
        target_link_libraries(Analysis ion-analysis gsl gslcblas pthread lapack_atlas atlas )
    endif()

    install(TARGETS Analysis DESTINATION bin)

#    add_executable(bkgFit bkgFit.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
#    add_dependencies(bkgFit IONVERSION)
#    if (ION_USE_CUDA)
#        target_link_libraries(bkgFit ion-analysis pthread ${CUDA_LIBRARIES} ${ION_ARMADILLO_LIBS})
#    else()
#	target_link_libraries(bkgFit ion-analysis ${GTEST_BOTH_LIBRARIES} gsl gslcblas pthread lapack_atlas atlas)
#    endif()
endif()

## Standalone BeadFind
add_executable(justBeadFind AnalysisOrg/justBeadFind/justBeadFind.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(justBeadFind IONVERSION)
target_link_libraries(justBeadFind ion-analysis gsl gslcblas pthread lapack_atlas atlas )
install(TARGETS justBeadFind DESTINATION bin)

## Standalone BaseCaller
add_executable(BaseCaller 
    BaseCaller/BaseCaller.cpp
    BaseCaller/BaseCallerFilters.cpp
    BaseCaller/PerBaseQual.cpp
    BaseCaller/PhaseEstimator.cpp
    BaseCaller/BaseCallerMetricSaver.cpp
    BaseCaller/TreephaserSSE.cpp
    BaseCaller/BarcodeClassifier.cpp
    BaseCaller/BarcodeDatasets.cpp
    BaseCaller/OrderedDatasetWriter.cpp
    BaseCaller/BaseCallerRecalibration.cpp
    ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BaseCaller IONVERSION bamtools)
target_link_libraries(BaseCaller ion-analysis gsl gslcblas pthread lapack_atlas atlas ${ION_BAMTOOLS_LIBS})
install(TARGETS BaseCaller DESTINATION bin)


add_executable(ionstats 
    ionstats/ionstats.cpp
    ionstats/ionstats_basecaller.cpp
    ionstats/ionstats_alignment.cpp
    ionstats/ionstats_tf.cpp
    ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(ionstats IONVERSION bamtools)
target_link_libraries(ionstats ion-analysis ${ION_BAMTOOLS_LIBS} pthread)
install(TARGETS ionstats DESTINATION bin)

add_executable(PairedEndErrorCorrection
   ${PROJECT_BINARY_DIR}/IonVersion.cpp
   SFFCorrection/Alignment.cpp
   SFFCorrection/Cell.cpp
   SFFCorrection/semaphore.cpp
   SFFCorrection/Sequence.cpp
   SFFCorrection/sffCorrection.cpp
   SFFCorrection/SmithWaterman.cpp)
add_dependencies(PairedEndErrorCorrection IONVERSION)
target_link_libraries(PairedEndErrorCorrection ion-analysis file-io pthread ${ION_BAMTOOLS_LIBS})
install(TARGETS PairedEndErrorCorrection DESTINATION bin)

IF( ION_BUILD_STRAND_BIAS )
add_custom_command( 
 TARGET PairedEndErrorCorrection POST_BUILD
 #COMMAND mkdir -p ${CMAKE_BUILD_TOOL} ${PROJECT_SOURCE_DIR}/../external/samita/build 
 COMMAND ${CMAKE_COMMAND} ${PROJECT_SOURCE_DIR}/../external/samita 
 COMMAND ${CMAKE_BUILD_TOOL} ${PROJECT_SOURCE_DIR}/../external/samita/ 
 COMMAND "make"
 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../external/samita/ 
 )
 
 include_directories (${ION_TS_EXTERNAL}/samita ${ION_TS_EXTERNAL}/common/c++/lifetech)
 link_directories ( ${ION_TS_EXTERNAL}/samita/ ${Boost_LIBRARY_DIRS} )
 add_executable (strandBiasExtractor strandBiasExtractor/semaphore.cpp  strandBiasExtractor/strandBiasExtractor.cpp)
 add_dependencies(strandBiasExtractor IONVERSION samita )
 target_link_libraries (strandBiasExtractor samita++.a boost_regex boost_iostreams boost_thread boost_filesystem boost_system ${ION_SAMTOOLS_LIBS} ${ION_HDF5_HL_LIBS} ${ION_HDF5_LIBS} log4cxx pthread )
 install(TARGETS strandBiasExtractor DESTINATION bin)
ENDIF(ION_BUILD_STRAND_BIAS )

# add_executable(WellsTimer WellsTimer.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(WellsTimer IONVERSION)
# target_link_libraries(WellsTimer ion-analysis pthread)
# install(TARGETS WellsTimer DESTINATION bin)


add_executable(MeasureXTalk MeasureXTalk.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MeasureXTalk IONVERSION)
target_link_libraries(MeasureXTalk ion-analysis pthread)

add_executable(WellCrossTalk WellCrossTalk.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(WellCrossTalk IONVERSION)
target_link_libraries(WellCrossTalk ion-analysis pthread ${ION_ARMADILLO_LIBS})

add_executable(Recall
	Recall/Recall.cpp
	../buildTools/dbgmem.cpp
	${PROJECT_BINARY_DIR}/IonVersion.cpp)
target_link_libraries(Recall ion-analysis file-io)
install(TARGETS Recall DESTINATION bin)

add_executable(SeqBoost
        Recall/SeqBoost.cpp
        ../buildTools/dbgmem.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp)
target_link_libraries(SeqBoost ion-analysis file-io ${ION_BAMTOOLS_LIBS} pthread)
install(TARGETS SeqBoost DESTINATION bin)

add_executable(sff2bam
        Converter/sff2bam.cpp
        ../buildTools/dbgmem.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp)
target_link_libraries(sff2bam ion-analysis file-io ${ION_BAMTOOLS_LIBS} pthread)
install(TARGETS sff2bam DESTINATION bin)

add_executable(bam2sff
        Converter/bam2sff.cpp
        ../buildTools/dbgmem.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp)
target_link_libraries(bam2sff ion-analysis file-io  ${ION_BAMTOOLS_LIBS} pthread)
install(TARGETS bam2sff DESTINATION bin)

add_executable(SamVsSam
        SamUtils/SamVsSam.cpp
)
target_link_libraries(SamVsSam ion-analysis ${ION_SAMTOOLS_LIBS})
install(TARGETS SamVsSam DESTINATION bin)

add_executable(SFFSummary
        SFFSummary/SFFSummary.cpp
        SFFSummary/SFFSummaryDriver.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFSummary ion-analysis file-io pthread)
add_dependencies(SFFSummary IONVERSION)
install(TARGETS SFFSummary DESTINATION bin)

add_executable(BeadmaskParse Mask/BeadmaskParse.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskParse IONVERSION)
target_link_libraries(BeadmaskParse ion-analysis pthread)
install(TARGETS BeadmaskParse DESTINATION bin)

add_executable(BeadmaskMerge Mask/BeadmaskMerge.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskMerge IONVERSION)
target_link_libraries(BeadmaskMerge ion-analysis pthread)
install(TARGETS BeadmaskMerge DESTINATION bin)

add_executable(MergeWells Wells/MergeWells.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeWells IONVERSION)
target_link_libraries(MergeWells ion-analysis pthread)
install(TARGETS MergeWells DESTINATION bin)

add_executable(RawWellsCombine Wells/RawWellsCombine.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsCombine IONVERSION)
target_link_libraries(RawWellsCombine ion-analysis pthread)
install(TARGETS RawWellsCombine DESTINATION bin)

include_directories("${PROJECT_SOURCE_DIR}/file-io")
add_library(file-io
        file-io/dat_chip.c
        file-io/dat_flow.c
        file-io/dat_frame.c
        file-io/dat_header.c
        file-io/dat_io.c
        file-io/fastq.c
        file-io/fastq_file.c
        file-io/ion_alloc.c
        file-io/ion_error.c
        file-io/ion_string.c
        file-io/ion_util.c
        file-io/sff.c
        file-io/sff_file.c
        file-io/sff_header.c
        file-io/sff_index.c
        file-io/sff_iter.c
        file-io/sff_read.c
        file-io/sff_read_header.c
        file-io/sff_sort.c
        file-io/wells_chip.c
        file-io/wells_combine.c
        file-io/wells_data.c
        file-io/wells_header.c
        file-io/wells_mask.c
        file-io/wells_mask_combine.c
)
if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        set_target_properties(file-io PROPERTIES COMPILE_FLAGS "-fPIC")
endif( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )

add_executable(SFFTrim
        SFFTrim/SFFTrim.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFTrim ion-analysis file-io pthread)
add_dependencies(SFFTrim IONVERSION)
install(TARGETS SFFTrim DESTINATION bin)

add_executable(SFFSearch
        SFFTrim/SFFSearch.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFSearch ion-analysis file-io pthread)
add_dependencies(SFFSearch IONVERSION)

add_executable(alignStats
        SamUtils/alignStats_driver.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(alignStats ion-analysis z ${ION_SAMTOOLS_LIBS} pthread)
add_dependencies(alignStats IONVERSION)
install(TARGETS alignStats DESTINATION bin)

add_executable(FlowErrTest
        SamUtils/FlowErrTest.cpp
)
target_link_libraries(FlowErrTest ion-analysis z ${ION_SAMTOOLS_LIBS} ${ION_HDF5_LIBS} pthread)

add_executable(SFFRead SFFUtils/SFFRead.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFRead IONVERSION)
target_link_libraries(SFFRead ion-analysis pthread)
install(TARGETS SFFRead DESTINATION bin)

add_executable(SFFMerge SFFMerge.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFMerge IONVERSION)
target_link_libraries(SFFMerge ion-analysis)
install(TARGETS SFFMerge DESTINATION bin)

add_executable(SFFRandom SFFUtils/SFFRandom.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFRandom IONVERSION)
target_link_libraries(SFFRandom ion-analysis pthread)
install(TARGETS SFFRandom DESTINATION bin)

add_executable(RawWellsEquivalent Wells/RawWellsEquivalent.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsEquivalent IONVERSION)
target_link_libraries(RawWellsEquivalent ion-analysis pthread)
install(TARGETS RawWellsEquivalent DESTINATION bin)

add_executable(DiffSeparator Separator/DiffSeparator.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(DiffSeparator IONVERSION)
target_link_libraries(DiffSeparator ion-analysis pthread ${ION_ARMADILLO_LIBS})
install(TARGETS DiffSeparator DESTINATION bin)

add_executable(CompressDeviation Separator/CompressDeviation.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CompressDeviation IONVERSION)
target_link_libraries(CompressDeviation ion-analysis pthread ${ION_ARMADILLO_LIBS})

add_executable(DumpSynchDat SynchDat/DumpSynchDat.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(DumpSynchDat IONVERSION)
target_link_libraries(DumpSynchDat ion-analysis pthread ${ION_ARMADILLO_LIBS})

add_executable(GrindSynchDat SynchDat/GrindSynchDat.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(GrindSynchDat IONVERSION)
target_link_libraries(GrindSynchDat ion-analysis pthread ${ION_ARMADILLO_LIBS})


add_executable(TraceDriver SynchDat/TraceDriver.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(TraceDriver IONVERSION)
target_link_libraries(TraceDriver ion-analysis pthread )

# add_executable(VFCTest SynchDat/VFCTest.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(VFCTest IONVERSION)
# target_link_libraries(VFCTest ion-analysis -lunwind pthread -lprofiler )


# add_executable(ImageT0 ImageT0.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(ImageT0 IONVERSION)
# target_link_libraries(ImageT0 ion-analysis  -lunwind pthread -lprofiler )
# install(TARGETS ImageT0 DESTINATION bin)

#add_executable(PinnedWell PinnedWell.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
#add_dependencies(PinnedWell IONVERSION)
#target_link_libraries(PinnedWell ion-analysis  pthread )
#install(TARGETS PinnedWell DESTINATION bin)

add_executable(readDat readDat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readDat IONVERSION)
#target_link_libraries(readDat ion-analysis pthread)
target_link_libraries(readDat ion-analysis ${GTEST_BOTH_LIBRARIES} gsl gslcblas pthread lapack_atlas atlas)
install(TARGETS readDat DESTINATION bin)

add_executable(readWells Wells/readWells.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readWells IONVERSION)
target_link_libraries(readWells ion-analysis pthread)
install(TARGETS readWells DESTINATION bin)

add_executable(SFFEquivalent SFFEquivalent.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFEquivalent IONVERSION)
target_link_libraries(SFFEquivalent ion-analysis pthread)
install(TARGETS SFFEquivalent DESTINATION bin)

add_executable(iontools file-io/main.c file-io/sff_check.c ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(iontools IONVERSION)
target_link_libraries(iontools ion-analysis file-io pthread)
install(TARGETS iontools DESTINATION bin)

add_executable(barcodeMaskParse Mask/barcodeMaskParse.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(barcodeMaskParse IONVERSION)
target_link_libraries(barcodeMaskParse ion-analysis)
install(TARGETS barcodeMaskParse DESTINATION bin)


# Setup for testing and gtest in particular, only needed once in file before tests.
enable_testing()
find_package(GTest QUIET)
if(GTEST_FOUND)
  
# Copy over test data...
   configure_file (
     "utest/reference.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/reference.txt"
     COPYONLY
    )
   configure_file (
     "utest/traces.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/traces.txt"
      COPYONLY
   )
   # configure_file (
   #   "utest/1.wells"
   #   "${CMAKE_CURRENT_BINARY_DIR}/1.wells"
   # )
   configure_file (
     "utest/reference.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/reference.txt"
     COPYONLY
   )
   configure_file (
     "utest/trimdata/sim.flow-TACG.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/sim.flow-TACG.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
     COPYONLY
    )

    configure_file (
     "utest/trimdata/sim.flow-TACG.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/sim.flow-TACG.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
      COPYONLY
     )
    configure_file (
     "utest/trimdata/sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
      COPYONLY
     )
    configure_file (
     "utest/trimdata/sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
     "${CMAKE_CURRENT_BINARY_DIR}/sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt"
      COPYONLY
     )


        include_directories(${GTEST_INCLUDE_DIRS})
        add_executable(AnalysisIntegrationTest itest/AnalysisIntegrationTest.cpp)
        target_link_libraries(AnalysisIntegrationTest ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
  # For each test file

        add_executable(IonErr_Test utest/IonErr_Test.cpp)
        target_link_libraries(IonErr_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(IonErrTest IonErr_Test --gtest_output=xml:./)

        add_executable(Mask_Test utest/Mask_Test.cpp)
        target_link_libraries(Mask_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(MaskTest Mask_Test --gtest_output=xml:./)

        # add_executable(Wells_Test utest/Wells_Test.cpp)
        # target_link_libraries(Wells_Test ion-analysis ${ION_HDF5_LIBS} ${GTEST_BOTH_LIBRARIES} pthread z)
        # add_test(WellsTest Wells_Test --gtest_output=xml:./)

        add_executable(DualGaussMixModel_Test utest/DualGaussMixModel_Test.cpp)
        target_link_libraries(DualGaussMixModel_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(DualGaussMixModelTest DualGaussMixModel_Test --gtest_output=xml:./)

        # add_executable(ZeromerDiff_Test utest/ZeromerDiff_Test.cpp)
        # target_link_libraries(ZeromerDiff_Test ion-analysis ${GTEST_BOTH_LIBRARIES}  )
        # add_test(ZeromerDiffTest ZeromerDiff_Test --gtest_output=xml:./)

        add_executable(FindSlopeChange_Test utest/FindSlopeChange_Test.cpp)
        target_link_libraries(FindSlopeChange_Test ion-analysis ${GTEST_BOTH_LIBRARIES} blas lapack_atlas atlas )
#        add_test(FindSlopeChangeTest FindSlopeChange_Test --gtest_output=xml:./)

        add_executable(T0Model_Test utest/T0Model_Test.cpp)
        target_link_libraries(T0Model_Test ion-analysis ${GTEST_BOTH_LIBRARIES} blas lapack_atlas atlas)
        add_test(T0ModelTest T0Model_Test --gtest_output=xml:./)

        # add_executable(KeyClassifier_Test utest/KeyClassifier_Test.cpp)
        # target_link_libraries(KeyClassifier_Test ion-analysis ${GTEST_BOTH_LIBRARIES}  )
        # add_test(KeyClassifierTest KeyClassifier_Test --gtest_output=xml:./)

        add_executable(NumericalComparison_Test utest/NumericalComparison_Test.cpp)
        target_link_libraries(NumericalComparison_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(NumericalComparisonTest NumericalComparison_Test --gtest_output=xml:./)

        add_executable(OptArgs_Test utest/OptArgs_Test.cpp)
        target_link_libraries(OptArgs_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(OptArgsTest OptArgs_Test --gtest_output=xml:./)

        add_executable(H5File_Test utest/H5File_Test.cpp)
        target_link_libraries(H5File_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        #add_test(H5FileTest H5File_Test --gtest_output=xml:./)

        add_executable(OptionArgs_Test utest/OptionArgs_Test.cpp)
        target_link_libraries(OptionArgs_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(OptionArgsTest OptionArgs_Test --gtest_output=xml:./)

        add_executable(BitHandler_Test utest/BitHandler_Test.cpp)
        target_link_libraries(BitHandler_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(BitHandlerTest BitHandler_Test --gtest_output=xml:./)

        add_executable(SFFTrim_Test utest/SFFTrim_Test.cpp)
        target_link_libraries(SFFTrim_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(SFFTrimTest SFFTrim_Test --gtest_output=xml:./)
endif()

#files used for configuration
#not actually source files at all
install(FILES "${PROJECT_SOURCE_DIR}/config/DefaultTFs.conf" DESTINATION ${ION_INSTALL_PREFIX}/config RENAME "DefaultTFs.conf.dist")
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_314.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_316.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_318.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_324.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_314" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_316" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_318" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_324" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.txt_314" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.txt_316" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.txt_318" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.txt_900" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_314.param" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_316.param" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_318.param" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_900.param" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/Analysis.1" DESTINATION share/man/man1)

if(NOT ION_USE_SYSTEM_ARMADILLO)
   install(FILES "${PROJECT_SOURCE_DIR}/../external/armadillo-3.0.2/lib/libarmadillo.so.3" DESTINATION lib)
   install(FILES "${PROJECT_SOURCE_DIR}/../external/armadillo-3.0.2/lib/libarmadillo.so.3.0.2" DESTINATION lib)
endif()

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libatlas3gf-base,
    libblas3gf,
    libc6,
    libgcc1,
    libgfortran3,
    liblapack3gf,
    libstdc++6,
    libgsl0ldbl,
    ion-gpu (>=2.3.5)")

set(CPACK_DEBIAN_PACKAGE_BREAKS "ion-gpu (<< 2.9)")
set(CPACK_DEBIAN_PACKAGE_REPLACES "ion-gpu (<< 2.9)")

set(CPACK_PACKAGE_DESCRIPTION "ion-analysis provides the command line workflow and supporting tools to process raw DAT files into SFF and fastq files.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
