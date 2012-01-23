# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-analysis)
set(PROJECT_DESCRIPTION "Ion Torrent Data Processing Pipeline")
enable_testing()

option(ION_DO_BACKGROUND "Build with background model" ON)
mark_as_advanced(ION_DO_BACKGROUND)

option(ION_USE_CUDA "Compile CUDA code" ON)
mark_as_advanced(ION_USE_CUDA)

option(ION_USE_SYSTEM_SAMTOOLS "Use samtools system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_SAMTOOLS)

option(ION_USE_SYSTEM_FFTW "Use fftw system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_FFTW)

option(ION_USE_SYSTEM_HDF5 "Use hdf5 system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_HDF5)

option(ION_USE_SYSTEM_ARMADILLO "Use armadillo system libraries" OFF)
mark_as_advanced(ION_USE_SYSTEM_ARMADILLO)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ION_DO_BACKGROUND OFF CACHE BOOL "Build with background model")
endif()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

include(ExternalProject)

if(ION_USE_CUDA)
  message(STATUS "BUILD with CUDA")
  add_definitions(-DION_COMPILE_CUDA)
endif()

# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion" CACHE PATH "Ion Install Prefix")
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

set (ION_TS_EXTERNAL "${PROJECT_SOURCE_DIR}/../external/")
#set (ION_TS_EXTERNAL "${CMAKE_BINARY_DIR}/external/")

if (NOT ION_USE_SYSTEM_FFTW)
  set(LIBFFTWFILE "${PROJECT_SOURCE_DIR}/../external/fftw-3.2.2/.libs/libfftw3.a")
  add_custom_command(OUTPUT ${LIBFFTWFILE}
    COMMAND "./configure" "--enable-static" "--with-pic"
    COMMAND "make"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/fftw-3.2.2")
  add_library(LIBFFTW STATIC IMPORTED)
  set_property(TARGET LIBFFTW PROPERTY IMPORTED_LOCATION ${LIBFFTWFILE})
  include_directories("${PROJECT_SOURCE_DIR}/../external/fftw-3.2.2/api")
  set(ION_FFTW_LIBS LIBFFTW)
else()
  set(ION_FFTW_LIBS libfftw3.a)
endif()

if(NOT ION_USE_SYSTEM_HDF5)
  set(proj_name "hdf5")
  set(proj_version "1.8.8")
  set(proj_name_version "${proj_name}-${proj_version}")
  ExternalProject_add(${proj_name}
    PREFIX ${ION_TS_EXTERNAL}/${proj_name_version}
    URL "${ION_TS_EXTERNAL}/hdf5-1.8.8.tar.gz"
#    URL_MD5 1196e668f5592bfb50d1de162eb16cff
    CONFIGURE_COMMAND ${ION_TS_EXTERNAL}/${proj_name_version}/src/${proj_name}/configure
    --with-pic=yes --disable-shared --with-pthread --enable-threadsafe
    --prefix=${ION_TS_EXTERNAL}/${proj_name_version}
  )

  set(HDF5FILE "${PROJECT_SOURCE_DIR}/../external/hdf5-1.8.7/hdf5/lib/libhdf5.a")
  add_custom_command(OUTPUT ${HDF5FILE}
    COMMAND "./configure" "--with-pic=yes" "--disable-shared" "--with-pthread" "--enable-threadsafe"
    COMMAND "make"
    COMMAND "make" "install"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/hdf5-1.8.7")

  set (HDF5Experiment 0)
  if(${HDF5Experiment})
    include_directories("${ION_TS_EXTERNAL}/${proj_name_version}/include")
    set(ION_HDF5_LIBS "${ION_TS_EXTERNAL}/${proj_name_version}/lib/libhdf5.a")
  else()
    add_library(HDF5 STATIC IMPORTED)
    set_property(TARGET HDF5 PROPERTY IMPORTED_LOCATION ${HDF5FILE})
    include_directories("${PROJECT_SOURCE_DIR}/../external/hdf5-1.8.7/src")
    set(ION_HDF5_LIBS HDF5)
  endif()
else()
  set(HDF5_USE_STATIC_LIBRARIES True)
  find_package(HDF5 REQUIRED)
  set(ION_HDF5_LIBS ${HDF5_LIBRARIES})
endif()

if(NOT ION_USE_SYSTEM_ARMADILLO)
  set(ARMAFILE "${PROJECT_SOURCE_DIR}/../external/armadillo-1.1.0/libarmadillo.so")
  add_custom_command(OUTPUT ${ARMAFILE}
    COMMAND "cmake" "."
    COMMAND "make"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/armadillo-1.1.0")
  include_directories("${PROJECT_SOURCE_DIR}/../external/armadillo-1.1.0/include")
endif()

if(ION_DO_BACKGROUND)
    set(LIBLAPACKPPFILE "${PROJECT_SOURCE_DIR}/../external/lapackpp-2.5.3/src/.libs/liblapackpp.a")
    add_custom_command(OUTPUT ${LIBLAPACKPPFILE}
        COMMAND "./configure" "--enable-static" "--with-pic"
        COMMAND "make"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/lapackpp-2.5.3")
    add_library(LIBLAPACKPP STATIC IMPORTED)
    set_property(TARGET LIBLAPACKPP PROPERTY IMPORTED_LOCATION ${LIBLAPACKPPFILE})
    include_directories("${PROJECT_SOURCE_DIR}/../external/lapackpp-2.5.3/include")
endif()

set(LIBKMEANSFILE "${PROJECT_SOURCE_DIR}/../external/kmeans-1.7/libkmeans.a")
add_custom_command(OUTPUT ${LIBKMEANSFILE}
    COMMAND "make"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../external/kmeans-1.7")
add_library(LIBKMEANS STATIC IMPORTED)
set_property(TARGET LIBKMEANS PROPERTY IMPORTED_LOCATION ${LIBKMEANSFILE})
include_directories("${PROJECT_SOURCE_DIR}/../external/kmeans-1.7")

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../buildTools")
include_directories("${PROJECT_SOURCE_DIR}/../external/tnt-1.2.6")
include_directories("${PROJECT_SOURCE_DIR}/../external/fstrcmp")
include_directories("${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1")


if (NOT ION_USE_SYSTEM_SAMTOOLS)
  ExternalProject_add(samtools
#    PREFIX samtools
    URL "${ION_TS_EXTERNAL}/samtools-0.1.18.tar.gz"
    #URL_MD5
    # in source build
    BINARY_DIR "${ION_TS_EXTERNAL}/samtools-0.1.18"
    SOURCE_DIR "${ION_TS_EXTERNAL}/samtools-0.1.18"
#    BINARY_DIR "${CMAKE_BINARY_DIR}/samtools/src/samtools"
    PATCH_COMMAND patch -p1 -t -N < "${ION_TS_EXTERNAL}/samtools-0.1.18.patch"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "/usr/bin/make" "CFLAGS=-fPIC" "<SOURCE_DIR>/Makefile" "lib"
    INSTALL_COMMAND ""
  )
  include_directories("${PROJECT_SOURCE_DIR}/../external/samtools-0.1.18")
  set(ION_SAMTOOLS_LIBS "${PROJECT_SOURCE_DIR}/../external/samtools-0.1.18/libbam.a")
#  include_directories("${CMAKE_BINARY_DIR}/samtools/src/samtools")
#  set(ION_SAMTOOLS_LIBS "${CMAKE_BINARY_DIR}/samtools/src/samtools/libbam.a")
else()
    set(ION_SAMTOOLS_LIBS libbam.a)
    include_directories("/usr/include/samtools")
endif()


# Find External Libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../buildTools/cmake/Modules")
set(EXTRA_INCLUDES)
set(EXTRA_LIBS)

set(BACKGROUND_SRC)
if(ION_DO_BACKGROUND)
    include_directories("${PROJECT_SOURCE_DIR}/BkgModel")
    include_directories("${PROJECT_SOURCE_DIR}/Phase")
    include_directories("${PROJECT_SOURCE_DIR}/Separator")
    set(BACKGROUND_SRC BkgModel/BkgModel.cpp Phase/CafieSolver.cpp)
    find_package(CLAPACK REQUIRED)
    set(EXTRA_INC ${EXTRA_INC} ${CLAPACK_INCLUDE_DIR})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CLAPACK_LIBRARIES})

    find_package(CBLAS REQUIRED)
    set(EXTRA_INC ${EXTRA_INC} ${CBLAS_INCLUDE_DIR})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CBLAS_LIBRARIES})

    set(EXTRA_LIBS ${EXTRA_LIBS} LIBLAPACKPP libblas-3gf.a)
endif()

include_directories(${EXTRA_INC})

# CUDA Files
if(ION_USE_CUDA)
  FIND_PACKAGE(CUDA)
  if(CUDA_FOUND STREQUAL "TRUE")
    SET( CUDA_NVCC_FLAGS "--use_fast_math" "-gencode=arch=compute_20,code=compute_20" )
    CUDA_COMPILE(CUDA_TEMP_FILES BkgModel/BkgModel.cu SHARED)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()

# Ion Analysis Library
add_library(ion-analysis
    ${HDF5FILE}
    ${LIBLAPACKPPFILE}
    ${ARMAFILE}
    ${LIBFFTWFILE}
    ${LIBKMEANSFILE}

    ${PROJECT_BINARY_DIR}/IonVersion.cpp

    CommandLineOpts.cpp
    BarCode.cpp
    deInterlace.cpp
    
    Separator/DualGaussMixModel.cpp
    Separator/Separator.cpp
    Separator/DifferentialSeparator.cpp
    Separator/KeyClassifier.cpp
    Separator/KClass.cpp
    H5File.cpp

    SffDiffStats.cpp
    GenomeDiffStats.cpp
    FlowDiffStats.cpp
    Histogram.cpp
    Image.cpp
    IonErr.cpp
    LinuxCompat.cpp
    Mask.cpp
    RawWells.cpp
    RawWellsV1.cpp
    Trimmer.cpp
    SFFWrapper.cpp
    Stats.cpp
    TFs.cpp
    Utils.cpp
    TrackProgress.cpp
    Flow.cpp
    ImageSpecClass.cpp
    ImageLoader.cpp
    ProcessImageToWell.cpp
    RegionTimingCalc.cpp
    WellFileManipulation.cpp
    cudaWrapper.cpp
    Zeromer.cpp
    
    BaseCaller.cpp
    BaseCallerLogFiles.cpp
    PerBaseQual.cpp
    RegionWellsReader.cpp
    OptArgs.cpp
    bivariate_gaussian.cpp
    mixed.cpp
    flow_utils.cpp
    OrderedRegionSFFWriter.cpp
    
    Phase/PhaseSim.cpp
    Phase/PhaseSolve.cpp
    Phase/PhaseFit.cpp
    Phase/PhaseFitCfIe.cpp
    Phase/DPTreephaser.cpp
    Phase/RegionAnalysis.cpp
    
    BkgModel/MathOptim.cpp
    BkgModel/DNTPRiseModel.cpp
    BkgModel/DiffEqModel.cpp
    BkgModel/MultiFlowModel.cpp
    BkgModel/BeadParams.cpp
    BkgModel/BeadTracker.cpp
    BkgModel/RegionParams.cpp
    BkgModel/RegionTracker.cpp
    BkgModel/BeadScratch.cpp
    BkgModel/LevMarState.cpp
    BkgModel/EmphasisVector.cpp
    BkgModel/CrossTalkSpec.cpp
    BkgModel/BkgFitOptim.cpp
    BkgModel/BkgTrace.cpp
    BkgModel/SingleFlowFit.cpp
    BkgModel/BkgSearchAmplitude.cpp
    BkgModel/GlobalDefaultsForBkgModel.cpp
    BkgModel/TimeCompression.cpp
    BkgModel/Vectorization.cpp
    BkgModel/BkgFitStructures.cpp
    BkgModel/FitControl.cpp
    BkgModel/MultiLevMar.cpp
    
    crop/Acq.cpp
    BFReference.cpp
    Traces.cpp

    ${BACKGROUND_SRC}
    FileEquivalent.cpp
    BubbleFilter.cpp
    ChipIdDecoder.cpp
    LSRowImageProcessor.cpp
    PinnedWellReporter.cpp
    ${PROJECT_SOURCE_DIR}/../external/fstrcmp/fstrcmp.cpp
    SamUtils/types/Cigar.cpp
    SamUtils/types/Qual.cpp
    SamUtils/types/Sequence.cpp
    SamUtils/types/BAMRead.cpp
    SamUtils/types/MD.cpp
    SamUtils/BAMReader.cpp
    SamUtils/BAMUtils.cpp
    SamUtils/alignStats.cpp

  	OptionArgsBase.cpp
  	OptionArgs.cpp
  	LoggerBase.cpp
  	Logger.cpp
    ${PROJECT_SOURCE_DIR}/../external/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
    ${CUDA_TEMP_FILES}
)

add_dependencies(ion-analysis IONVERSION samtools hdf5)
target_link_libraries(ion-analysis ${ION_FFTW_LIBS} ${ION_HDF5_LIBS} LIBKMEANS ${EXTRA_LIBS} file-io z)
set_target_properties(ion-analysis PROPERTIES
    VERSION   "${ION_VERSION_MAJOR}.${ION_VERSION_MINOR}.${ION_VERSION_RELEASE}.${ION_VERSION_BUILDNUM}"
)
if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        set_target_properties(ion-analysis PROPERTIES COMPILE_FLAGS "-fPIC")
endif( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
install(TARGETS ion-analysis DESTINATION lib)


add_executable( SFFExtract SFFExtract.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies( SFFExtract IONVERSION TCLAP)
target_link_libraries( SFFExtract ion-analysis pthread)

add_executable( SFFExtract_sam SFFExtract_sam.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies( SFFExtract_sam IONVERSION TCLAP)
target_link_libraries( SFFExtract_sam ion-analysis pthread)

add_executable(Crop crop/Crop.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Crop IONVERSION TCLAP)
target_link_libraries(Crop ion-analysis pthread)
install(TARGETS Crop DESTINATION /usr/local/bin)

add_executable(Blocks crop/Blocks.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Blocks IONVERSION)
target_link_libraries(Blocks ion-analysis pthread)
install(TARGETS Blocks DESTINATION /usr/local/bin)

add_executable(MergeImages crop/MergeImages.cpp crop/MergeAcq.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeImages IONVERSION)
target_link_libraries(MergeImages ion-analysis pthread)
install(TARGETS MergeImages DESTINATION /usr/local/bin)

add_executable(MergeDats crop/MergeDats.cpp crop/MergeAcq.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeDats IONVERSION)
target_link_libraries(MergeDats ion-analysis pthread)
install(TARGETS MergeDats DESTINATION /usr/local/bin)

add_executable( GetRawAvgTraces GetRawAvgTraces.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Crop IONVERSION TCLAP)
target_link_libraries( GetRawAvgTraces ion-analysis pthread)

add_executable(Archive crop/Archive.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Archive IONVERSION TCLAP)
target_link_libraries(Archive ion-analysis pthread)

add_executable(WellsStats WellsStats.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(WellsStats IONVERSION TCLAP)
target_link_libraries(WellsStats ion-analysis pthread)

add_executable(SFFStats SFFStats.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFStats IONVERSION TCLAP)
target_link_libraries(SFFStats ion-analysis pthread)

add_executable(KeypassMetrics KeypassMetrics.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(KeypassMetrics IONVERSION TCLAP)
target_link_libraries(KeypassMetrics ion-analysis pthread)
install(TARGETS KeypassMetrics DESTINATION /usr/local/bin)

add_executable(HeatMaps HeatMaps.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(HeatMaps IONVERSION TCLAP)
target_link_libraries(HeatMaps ion-analysis pthread)

add_executable(GetTraces GetTraces.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(GetTraces IONVERSION TCLAP)
target_link_libraries(GetTraces ion-analysis pthread)

add_executable(Coverage Coverage.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Coverage IONVERSION TCLAP)
target_link_libraries(Coverage ion-analysis pthread)

add_executable(AsciMask2Bin AsciMask2Bin.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(AsciMask2Bin IONVERSION TCLAP)
target_link_libraries(AsciMask2Bin ion-analysis pthread)

add_executable(teMasks2Bin teMasks2Bin.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(teMasks2Bin IONVERSION TCLAP)
target_link_libraries(teMasks2Bin ion-analysis pthread)

if(ION_DO_BACKGROUND)
    add_executable(Analysis Analysis.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
    add_dependencies(Analysis IONVERSION TCLAP)

    if (ION_USE_CUDA)
        target_link_libraries(Analysis ion-analysis gsl gslcblas pthread ${CUDA_LIBRARIES} lapack_atlas atlas)
    else()
        target_link_libraries(Analysis ion-analysis gsl gslcblas pthread lapack_atlas atlas)
    endif()

    install(TARGETS Analysis DESTINATION /usr/local/bin)

    add_executable(bkgFit bkgFit.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
    add_dependencies(bkgFit IONVERSION TCLAP)
    if (ION_USE_CUDA)
        target_link_libraries(bkgFit ion-analysis pthread ${CUDA_LIBRARIES})
    else()
        target_link_libraries(bkgFit ion-analysis pthread)
    endif()
endif()

add_executable(ShowWell ShowWell.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(ShowWell IONVERSION TCLAP)
target_link_libraries(ShowWell ion-analysis pthread)
install(TARGETS ShowWell DESTINATION /usr/local/bin)

add_executable(TFMapper TFMapper.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(TFMapper IONVERSION TCLAP)
target_link_libraries(TFMapper ion-analysis pthread)
install(TARGETS TFMapper DESTINATION /usr/local/bin)

add_executable(TFMapperIterative TFMapperIterative.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(TFMapperIterative IONVERSION TCLAP)
target_link_libraries(TFMapperIterative ion-analysis pthread)
install(TARGETS TFMapperIterative DESTINATION /usr/local/bin)

# add_executable(WellsTimer WellsTimer.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(WellsTimer IONVERSION TCLAP)
# target_link_libraries(WellsTimer ion-analysis pthread)
# install(TARGETS WellsTimer DESTINATION /usr/local/bin)

add_executable(AnalyzeHPErrs AnalyzeHPErrs.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(AnalyzeHPErrs IONVERSION TCLAP)
target_link_libraries(AnalyzeHPErrs ion-analysis pthread)

add_executable(MeasureXTalk MeasureXTalk.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MeasureXTalk IONVERSION TCLAP)
target_link_libraries(MeasureXTalk ion-analysis pthread)

add_executable(WellCrossTalk WellCrossTalk.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(WellCrossTalk IONVERSION TCLAP)
target_link_libraries(WellCrossTalk ion-analysis pthread)


add_executable(SFFSummary
        SFFSummary/SFFSummary.cpp
        SFFSummary/SFFSummaryDriver.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFSummary ion-analysis file-io pthread)
add_dependencies(SFFSummary IONVERSION TCLAP)
install(TARGETS SFFSummary DESTINATION /usr/local/bin)

add_executable(BeadmaskParse BeadmaskParse.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskParse IONVERSION TCLAP)
target_link_libraries(BeadmaskParse ion-analysis pthread)
install(TARGETS BeadmaskParse DESTINATION /usr/local/bin)

add_executable(BeadmaskMerge BeadmaskMerge.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskMerge IONVERSION TCLAP)
target_link_libraries(BeadmaskMerge ion-analysis pthread)
install(TARGETS BeadmaskMerge DESTINATION /usr/local/bin)

add_executable(MergeWells MergeWells.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeWells IONVERSION TCLAP)
target_link_libraries(MergeWells ion-analysis pthread)
install(TARGETS MergeWells DESTINATION /usr/local/bin)

add_executable(Mask2TE Mask2TE.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Mask2TE IONVERSION TCLAP)
target_link_libraries(Mask2TE ion-analysis pthread)

add_executable(WellsAggregator WellsAggregator.cpp ../buildTools/dbgmem.cpp)
add_dependencies(WellsAggregator IONVERSION TCLAP)
target_link_libraries(WellsAggregator ion-analysis pthread)

add_executable(fastqsim fastqsim/fastqsim.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(fastqsim IONVERSION TCLAP)
target_link_libraries(fastqsim ion-analysis pthread)

add_executable(IonFlowAlign fastqsim/IonFlowAlign.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(IonFlowAlign IONVERSION TCLAP)
target_link_libraries(IonFlowAlign ion-analysis pthread)

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
        file-io/wells_data.c
        file-io/wells_header.c
        file-io/wells_mask.c
)
if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        set_target_properties(file-io PROPERTIES COMPILE_FLAGS "-fPIC")
endif( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )

add_executable(phaseSolveTest
        phaseTests/phaseSolveTest.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(phaseSolveTest ion-analysis file-io gsl pthread)
add_dependencies(phaseSolveTest IONVERSION TCLAP)

add_executable(SFFTrim
        SFFTrim/SFFTrim.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFTrim ion-analysis file-io pthread)
add_dependencies(SFFTrim IONVERSION TCLAP)
install(TARGETS SFFTrim DESTINATION /usr/local/bin)

add_executable(SFFSearch
        SFFTrim/SFFSearch.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(SFFSearch ion-analysis file-io pthread)
add_dependencies(SFFSearch IONVERSION TCLAP)

add_executable(alignStats
        SamUtils/alignStats_driver.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(alignStats ion-analysis z ${ION_SAMTOOLS_LIBS} pthread)
add_dependencies(alignStats IONVERSION TCLAP)
install(TARGETS alignStats DESTINATION /usr/local/bin)

add_executable(SFFRead fastqsim/SFFRead.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFRead IONVERSION TCLAP)
target_link_libraries(SFFRead ion-analysis pthread)
install(TARGETS SFFRead DESTINATION /usr/local/bin)

add_executable(SFFMerge SFFMerge.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFMerge IONVERSION)
target_link_libraries(SFFMerge ion-analysis)
install(TARGETS SFFMerge DESTINATION /usr/local/bin)

add_executable(SFFFilter fastqsim/SFFFilter.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFFilter IONVERSION TCLAP)
target_link_libraries(SFFFilter ion-analysis pthread)

add_executable(SFFRandom fastqsim/SFFRandom.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFRandom IONVERSION TCLAP)
target_link_libraries(SFFRandom ion-analysis pthread)
install(TARGETS SFFRandom DESTINATION /usr/local/bin)

add_executable(SFFAggregator fastqsim/SFFAggregate.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFAggregator IONVERSION TCLAP)
target_link_libraries(SFFAggregator ion-analysis pthread)

add_executable(Mapper fastqsim/Mapper.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Mapper IONVERSION TCLAP)
target_link_libraries(Mapper ion-analysis pthread)

add_executable(FIXSFF fastqsim/FIXSFF.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(FIXSFF IONVERSION TCLAP)
target_link_libraries(FIXSFF ion-analysis pthread)

add_executable(SFFBT fastqsim/SFFBarcodeTrim.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFBT IONVERSION TCLAP)
target_link_libraries(SFFBT ion-analysis pthread)

add_executable(SFFPT fastqsim/SFFPrimerTrim.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFPT IONVERSION TCLAP)
target_link_libraries(SFFPT ion-analysis pthread)

add_executable(BFMaskAggregator fastqsim/BFMaskAggregate.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BFMaskAggregator IONVERSION TCLAP)
target_link_libraries(BFMaskAggregator ion-analysis pthread)

add_executable(RawWellsEquivalent RawWellsEquivalent.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsEquivalent IONVERSION TCLAP)
target_link_libraries(RawWellsEquivalent ion-analysis pthread)
install(TARGETS RawWellsEquivalent DESTINATION /usr/local/bin)

add_executable(DiffSeparator Separator/DiffSeparator.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(DiffSeparator IONVERSION TCLAP)
target_link_libraries(DiffSeparator ion-analysis  pthread )
install(TARGETS DiffSeparator DESTINATION /usr/local/bin)

add_executable(CountBubbles CountBubbles.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CountBubbles IONVERSION TCLAP)
target_link_libraries(CountBubbles ion-analysis  pthread )
install(TARGETS CountBubbles DESTINATION /usr/local/bin)

# add_executable(RawWellsDump RawWellsDump.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(RawWellsDump IONVERSION TCLAP)
# target_link_libraries(RawWellsDump ion-analysis  pthread )
# install(TARGETS RawWellsDump DESTINATION /usr/local/bin)

add_executable(PinnedWell PinnedWell.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(PinnedWell IONVERSION TCLAP)
target_link_libraries(PinnedWell ion-analysis  pthread )
install(TARGETS PinnedWell DESTINATION /usr/local/bin)

# add_executable(LiveNeighbors LiveNeighbors.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(LiveNeighbors IONVERSION TCLAP)
# target_link_libraries(LiveNeighbors ion-analysis  pthread )
# install(TARGETS LiveNeighbors DESTINATION /usr/local/bin)

add_executable(readDat readDat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readDat IONVERSION)
target_link_libraries(readDat ion-analysis pthread)
install(TARGETS readDat DESTINATION /usr/local/bin)

add_executable(readWells readWells.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readWells IONVERSION)
target_link_libraries(readWells ion-analysis pthread)
install(TARGETS readWells DESTINATION /usr/local/bin)

add_executable(SFFEquivalent SFFEquivalent.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SFFEquivalent IONVERSION TCLAP)
target_link_libraries(SFFEquivalent ion-analysis pthread)
install(TARGETS SFFEquivalent DESTINATION /usr/local/bin)

add_executable(iontools file-io/main.c file-io/sff_check.c ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(iontools IONVERSION TCLAP)
target_link_libraries(iontools ion-analysis file-io pthread)
install(TARGETS iontools DESTINATION /usr/local/bin)

add_executable(barcodeSplit barcodeSplit.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(barcodeSplit IONVERSION TCLAP)
target_link_libraries(barcodeSplit ion-analysis)
install(TARGETS barcodeSplit DESTINATION /usr/local/bin)

add_executable(barcodeMaskParse barcodeMaskParse.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(barcodeMaskParse IONVERSION TCLAP)
target_link_libraries(barcodeMaskParse ion-analysis)
install(TARGETS barcodeMaskParse DESTINATION /usr/local/bin)

add_executable(BruteForcePhase Phase/BruteForcePhase.cpp ../buildTools/dbgmem.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BruteForcePhase IONVERSION TCLAP)
target_link_libraries(BruteForcePhase ion-analysis z ${ION_SAMTOOLS_LIBS} pthread )

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
   #    COPYONLY
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
        add_executable(PhaseFitCfIe_Test utest/PhaseFitCfIe_Test.cpp)
        target_link_libraries(PhaseFitCfIe_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(PhaseFitCfIeTest PhaseFitCfIe_Test --gtest_output=xml:./)

        add_executable(IonErr_Test utest/IonErr_Test.cpp)
        target_link_libraries(IonErr_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(IonErrTest IonErr_Test --gtest_output=xml:./)

        add_executable(Mask_Test utest/Mask_Test.cpp)
        target_link_libraries(Mask_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(MaskTest Mask_Test --gtest_output=xml:./)

        # add_executable(Wells_Test utest/Wells_Test.cpp)
        # target_link_libraries(Wells_Test ion-analysis ${ION_HDF5_LIBS} ${GTEST_BOTH_LIBRARIES} pthread z)
        # add_test(WellsTest Wells_Test --gtest_output=xml:./)

        add_executable(PhaseSim_Test utest/PhaseSim_Test.cpp)
        target_link_libraries(PhaseSim_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(PhaseSimTest PhaseSim_Test --gtest_output=xml:./)

        add_executable(PhaseSolve_Test utest/PhaseSolve_Test.cpp)
        target_link_libraries(PhaseSolve_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(PhaseSolveTest PhaseSolve_Test --gtest_output=xml:./)

        add_executable(DualGaussMixModel_Test utest/DualGaussMixModel_Test.cpp)
        target_link_libraries(DualGaussMixModel_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(DualGaussMixModelTest DualGaussMixModel_Test --gtest_output=xml:./)

        # add_executable(ZeromerDiff_Test utest/ZeromerDiff_Test.cpp)
        # target_link_libraries(ZeromerDiff_Test ion-analysis ${GTEST_BOTH_LIBRARIES}  )
        # add_test(ZeromerDiffTest ZeromerDiff_Test --gtest_output=xml:./)

        add_executable(FindSlopeChange_Test utest/FindSlopeChange_Test.cpp)
        target_link_libraries(FindSlopeChange_Test ion-analysis ${GTEST_BOTH_LIBRARIES} )
        add_test(FindSlopeChangeTest FindSlopeChange_Test --gtest_output=xml:./)

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

        add_executable(FlowDiffStats_Test utest/FlowDiffStats_Test.cpp)
        target_link_libraries(FlowDiffStats_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(FlowDiffStatsTest FlowDiffStats_Test --gtest_output=xml:./)

        add_executable(SFFTrim_Test utest/SFFTrim_Test.cpp)
        target_link_libraries(SFFTrim_Test ion-analysis ${GTEST_BOTH_LIBRARIES})
        add_test(SFFTrimTest SFFTrim_Test --gtest_output=xml:./)
endif()

install(FILES "${PROJECT_SOURCE_DIR}/DefaultTFs.conf" DESTINATION config RENAME "DefaultTFs.conf.dist")
install(FILES "${PROJECT_SOURCE_DIR}/exclusionMask_314.bin" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/exclusionMask_316.bin" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/exclusionMask_318.bin" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/exclusionMask_324.bin" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/excludeMask_314" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/excludeMask_316" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/excludeMask_318" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/excludeMask_324" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/Analysis.1" DESTINATION /usr/share/man/man1)
install(FILES "${PROJECT_SOURCE_DIR}/phredTable.txt_314" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/phredTable.txt_316" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/phredTable.txt_318" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/gopt_314.param" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/gopt_316.param" DESTINATION config)
install(FILES "${PROJECT_SOURCE_DIR}/gopt_318.param" DESTINATION config)
# perl modules used by plugins for json file parsing
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/common-sense-3.4.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-2.53.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/JSON-XS-2.32.tar.gz" DESTINATION "/tmp/pModIn")
# perl modules used by plugins for ???
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Readonly-1.03.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Carp-Assert-0.20.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/IPC-System-Simple-1.21.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/Statistics-LineFit-0.07.tar.gz" DESTINATION "/tmp/pModIn")
install(FILES "${PROJECT_SOURCE_DIR}/../external/perl/autodie-2.10.tar.gz" DESTINATION "/tmp/pModIn")

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
#Dependency Notes:
#
# The matePairIndividualTags plugin requires these packages:
#	emboss-data,emboss-lib,libnucleus6,libajax6
# The emboss package itself is not installed because of a filename conflict with
# /usr/bin/tmap.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libatlas3gf-base,
    libblas3gf,
    libc6,
    libgcc1,
    libgfortran3,
    liblapack3gf,
    libstdc++6,
    openjdk-6-jre,
    libgsl0ldbl,
    ion-gpu,
    emboss-data,
    emboss-lib,
    libnucleus6,
    libajax6")
set(CPACK_PACKAGE_DESCRIPTION "ion-analysis provides the command line workflow and supporting tools to process raw DAT files into SFF and fastq files.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
