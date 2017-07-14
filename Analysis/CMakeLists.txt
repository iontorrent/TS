# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
cmake_minimum_required (VERSION 2.6)
project (ion-analysis)
set(PROJECT_DESCRIPTION "Ion Torrent Data Processing Pipeline")
enable_testing()

include(CheckCCompilerFlag)
check_c_compiler_flag( -malign-double HAS_ALIGN_DOUBLE )

# This inconspicuous line holds the key to IR integration of tvc, tmap, and bamrealignment:
#set(CMAKE_EXE_LINKER_FLAGS -static)


option(ION_DO_BACKGROUND "Build with background model" ON)
mark_as_advanced(ION_DO_BACKGROUND)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ION_DO_BACKGROUND OFF CACHE BOOL "Build with background model")
endif()

include(../buildTools/cmake/CMakeLists.version.txt)
include(../buildTools/cmake/CMakeLists.compiler.txt)

FIND_PACKAGE(PythonInterp)
# Determine python site-packages or dist-packages directory
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print get_python_lib(prefix='/usr/local')"
        OUTPUT_VARIABLE PYTHON_LOCAL_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_version; print get_python_version()"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)


# Where to install - override with: -DION_INSTALL_PREFIX
set(ION_INSTALL_PREFIX "/opt/ion" CACHE PATH "Ion Install Prefix")

set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_INSTALL_PREFIX})
set(CPACK_SET_DESTDIR "ON")

configure_file (
    "${PROJECT_SOURCE_DIR}/debian/postinst.in"
    "${PROJECT_BINARY_DIR}/debian/postinst" @ONLY
)

include(../buildTools/cmake/CMakeLists.dependencies.txt)

message(STATUS "BUILD with ION_USE_CUDA ${ION_USE_CUDA}")
message(STATUS "BUILD with ION_USE_SYSTEM_CUDA ${ION_USE_SYSTEM_CUDA}")

if(ION_USE_CUDA)

    if(NOT ION_USE_SYSTEM_CUDA)

        message(STATUS "BUILD with CUDA ${CUDA_VERSION}")

        #install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib
        #        DESTINATION /usr/local/cuda)

        #install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64
        #        DESTINATION /usr/local/cuda)

        install(FILES ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart.so.8.0
                      ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart.so.8.0.44
                DESTINATION
                      /usr/local/cuda/lib64)

        #install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/bin/nvcc
        #        DESTINATION /usr/local/cuda/bin)

    endif()


    # from https://pypi.python.org/pypi/nvidia-ml-py/
    install(FILES pynvml/nvidia_smi.py             DESTINATION ${PYTHON_SITE_PACKAGES})
    install(FILES pynvml/nvidia_smi.pyc            DESTINATION ${PYTHON_SITE_PACKAGES})
    install(FILES pynvml/pynvml.py                 DESTINATION ${PYTHON_SITE_PACKAGES})
    install(FILES pynvml/pynvml.pyc                DESTINATION ${PYTHON_SITE_PACKAGES})
    install(PROGRAMS pynvml/pynvml_test.py         DESTINATION /usr/local/bin)
    install(PROGRAMS pynvml/nvidia_smi_test.py     DESTINATION /usr/local/bin)

#
#     CUDA_ADD_EXECUTABLE(xtalk_sim
#                         xtalk_sim/xtalk_sim.cpp
#                         xtalk_sim/DiffEqModel_Init.cpp
#                         xtalk_sim/DiffEqModel.cpp
#                         xtalk_sim/DelsqCUDA.cu
#                         xtalk_sim/utils.cpp
#                         xtalk_sim/xtalk_sim.cpp
#                         xtalk_sim/sim_incorp_flux.cpp
#                         Util/WorkerInfoQueue.cpp)
# 
#     target_link_libraries(xtalk_sim pthread)
# 
#     add_dependencies(xtalk_sim     cuda_toolkit)
# 
#     install(PROGRAMS ${PROJECT_BINARY_DIR}/xtalk_sim     DESTINATION gpu)

endif()

add_custom_target(compileall ALL
    COMMAND ${PYTHON_EXECUTABLE} -m compileall -q -f "${PROJECT_SOURCE_DIR}/pynvml/"
)

#include_directories("${PROJECT_SOURCE_DIR}/Replay")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg/justBeadFind")
include_directories("${PROJECT_SOURCE_DIR}/AnalysisOrg/IO")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/MathModel")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Bookkeeping")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/LocalTrace")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Fitters")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Fitters/Complex")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Writers")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/HostDataWrapper")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/HosteDeviceDataCubes")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/KernelIncludes")
include_directories("${PROJECT_SOURCE_DIR}/BkgModel/Sampling")
include_directories("${PROJECT_SOURCE_DIR}/BaseCaller")
include_directories("${PROJECT_SOURCE_DIR}/Calibration")
include_directories("${PROJECT_SOURCE_DIR}/Separator")
#include_directories("${PROJECT_SOURCE_DIR}/NormWells")
include_directories("${PROJECT_SOURCE_DIR}/Util")
#this has enough files now to justify own directory
include_directories("${PROJECT_SOURCE_DIR}/Image")
include_directories("${PROJECT_SOURCE_DIR}/Mask")
include_directories("${PROJECT_SOURCE_DIR}/Wells")

# CUDA Files
if(ION_USE_CUDA)

    CUDA_COMPILE(CUDA_TEMP_FILES 
      BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu
      BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu
      BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu  
      BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu
      BkgModel/CUDA/HostDataWrapper/JobWrapper.cu 
      BkgModel/CUDA/BkgGpuPipeline.cu
      BkgModel/CUDA/SingleFitStream.cu
      BkgModel/CUDA/StreamManager.cu 
      BkgModel/CUDA/MultiFitStream.cu
      BkgModel/CUDA/StreamingKernels.cu
      BkgModel/CUDA/MasterKernel.cu      
      SHARED
    )

    SET( CUDA_CPP_FILES
      BkgModel/CUDA/HosteDeviceDataCubes/PerBeadDataCubes.cpp
      BkgModel/CUDA/HostDataWrapper/ClonalFilterWrapper.cpp
      BkgModel/CUDA/KernelIncludes/DeviceParamDefines.cpp
    )

endif()

# Ion Analysis Library
add_library(ion-analysis

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
    Image/CorrNoiseCorrector.cpp
    Image/RowSumCorrector.cpp
    Image/IonImageSem.cpp
    Image/AdvCompr.cpp
    Image/PCACompression.cpp
    Image/PcaSpline.cpp
#    Image/DfcCompr.cpp
#    Image/ParallelDFT.cpp
    Image/PairPixelXtalkCorrector.cpp
    Image/ImageNNAvg.cpp
    Image/FluidPotentialCorrector.cpp
    Image/RowSumData.cpp

    Mask/PinnedInFlow.cpp
    Mask/Mask.cpp
    Mask/ComplexMask.cpp
    
    Separator/DualGaussMixModel.cpp
    Separator/DifferentialSeparator.cpp
    Separator/NNAvg.cpp
    Separator/ChipReduction.cpp
    Separator/BFReference.cpp
    Separator/EvaluateKey.cpp
    Separator/ZeromerMatDiff.cpp
    Separator/BfMetric.cpp
    Separator/TraceSaver.cpp
    Separator/TraceStoreCol.cpp

    Util/IonH5File.cpp
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
    Util/RingBuffer.cpp
    Util/SeqUtils.cpp

    AnalysisOrg/IO/CommandLineOpts.cpp
    AnalysisOrg/IO/KeyContext.cpp
    AnalysisOrg/IO/FlowContext.cpp
    AnalysisOrg/IO/BkgControlOpts.cpp
    AnalysisOrg/IO/GpuControlOpts.cpp
    AnalysisOrg/IO/BeadfindControlOpts.cpp
    AnalysisOrg/IO/ImageControlOpts.cpp
    AnalysisOrg/IO/BkgModelHdf5.cpp
    AnalysisOrg/IO/TrackProgress.cpp
    AnalysisOrg/IO/ProgramState.cpp
    AnalysisOrg/IO/CaptureImageState.cpp
    AnalysisOrg/IO/DebugMe.cpp
    AnalysisOrg/IO/OptBase.cpp
    
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
    AnalysisOrg/BkgDataPointers.cpp
    AnalysisOrg/BkgFitterTracker.cpp
    AnalysisOrg/SlicedPrequel.cpp
    AnalysisOrg/SignalProcessingFitterQueue.cpp
    AnalysisOrg/ClonalFilter.cpp
    AnalysisOrg/MaskFunctions.cpp
    AnalysisOrg/cudaWrapper.cpp
    AnalysisOrg/FlowSequence.cpp
        
    Region.cpp
    LinuxCompat.cpp
    Stats.cpp

    BkgModel/Fitters/LevMarFitterV2.cpp
    
    ClonalFilter/mixed.cpp
    ClonalFilter/polyclonal_filter.cpp
    # T0CalcMt.cpp
    
    BaseCaller/PIDloop.cpp
    BaseCaller/DPTreephaser.cpp
    Calibration/FlowAlignment.cpp
    Calibration/LinearCalibrationModel.cpp
    
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
    BkgModel/Bookkeeping/BarcodeTracker.cpp
    BkgModel/Bookkeeping/BeadTracker.cpp
    BkgModel/Bookkeeping/RegionParamDefault.cpp
    BkgModel/Bookkeeping/RegionParams.cpp
    BkgModel/Bookkeeping/RegionTracker.cpp
    BkgModel/Bookkeeping/FlowBuffer.cpp
    BkgModel/Bookkeeping/FlowDefaults.cpp
    BkgModel/Bookkeeping/NucStepCache.cpp
    BkgModel/Bookkeeping/TimeControl.cpp
    BkgModel/Bookkeeping/TimeCompression.cpp
    BkgModel/Bookkeeping/EmphasisVector.cpp
    BkgModel/Bookkeeping/DarkHalo.cpp
    BkgModel/Bookkeeping/CrossTalkSpec.cpp
    BkgModel/Bookkeeping/XtalkCurry.cpp
    BkgModel/Bookkeeping/TraceCurry.cpp
    BkgModel/Bookkeeping/BeadScratch.cpp
    BkgModel/Bookkeeping/WellXtalk.cpp
    BkgModel/Bookkeeping/ControlSingleFlow.cpp
    BkgModel/Bookkeeping/FitterDefaults.cpp

    BkgModel/LocalTrace/BkgTrace.cpp
    BkgModel/LocalTrace/EmptyTrace.cpp
    #BkgModel/LocalTrace/TraceClassifier.cpp
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
    BkgModel/Fitters/ExpTailFitter.cpp
    BkgModel/Fitters/exptaildecayfit.cpp

    BkgModel/Writers/DebugWriter.cpp
    BkgModel/Writers/GlobalWriter.cpp
    
    BkgModel/GlobalDefaultsForBkgModel.cpp
    BkgModel/SignalProcessingMasterFitter.cpp
    BkgModel/RegionalizedData.cpp

    BkgModel/CUDA/GpuMultiFlowFitControl.cpp
    BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp
	${CUDA_CPP_FILES}
	
    BkgModel/Sampling/FitDensity.cpp
    BkgModel/Sampling/SampleClonality.cpp


    crop/Acq.cpp
    crop/DCT0Finder.cpp
    crop/PreviousFrameSubtract.cpp 

    
    SamUtils/types/Cigar.cpp
    SamUtils/types/Qual.cpp
    SamUtils/types/Sequence.cpp
    SamUtils/types/BAMRead.cpp
    SamUtils/types/MD.cpp
    SamUtils/BAMReader.cpp
    SamUtils/BAMUtils.cpp
    SamUtils/alignStats.cpp

    ${ION_TS_EXTERNAL}/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
    ${CUDA_TEMP_FILES}
)
add_dependencies(ion-analysis IONVERSION samtools bamtools armadillo_proj hdf5_proj boost_proj)
if(ION_USE_CUDA)
  if(NOT ION_USE_SYSTEM_CUDA)
    add_dependencies(ion-analysis ${cuda_toolkit})
    # Changes J30
    #add_dependencies(ion-analysis ${cuda_toolkit} armadillo_proj)
  endif()
endif()

# armadillo doesn't provide those libs?
if(ION_USE_MKL)
    set(EXTRA_LIBS m iomp5 mkl_intel_lp64)
else()
    set(EXTRA_LIBS m blas lapack)
endif()

target_link_libraries(ion-analysis ${ION_HDF5_LIBS} dl ${ION_KMEANS_LIBS} ${ION_BOOST_LIBS} ${EXTRA_LIBS} file-io z)

set_target_properties(ion-analysis PROPERTIES
    VERSION   "${ION_VERSION_MAJOR}.${ION_VERSION_MINOR}.${ION_VERSION_RELEASE}.${ION_VERSION_BUILDNUM}"
)
install(TARGETS ion-analysis DESTINATION lib)


add_executable(Crop crop/Crop.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Crop IONVERSION)
target_link_libraries(Crop ion-analysis pthread dl)
install(TARGETS Crop DESTINATION bin)

add_executable(ChkDat crop/ChkDat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(ChkDat IONVERSION)
target_link_libraries(ChkDat ion-analysis pthread dl)
install(TARGETS ChkDat DESTINATION bin)

add_executable(bin2Dat crop/bin2Dat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(bin2Dat IONVERSION)
target_link_libraries(bin2Dat ion-analysis pthread dl)
install(TARGETS bin2Dat DESTINATION bin)

add_executable(CmpDat crop/CmpDat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CmpDat IONVERSION)
target_link_libraries(CmpDat ion-analysis pthread dl)
install(TARGETS CmpDat DESTINATION bin)

add_executable(Blocks crop/Blocks.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Blocks IONVERSION)
target_link_libraries(Blocks ion-analysis pthread dl)
install(TARGETS Blocks DESTINATION bin)

add_executable(CropRegions crop/CropRegions.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CropRegions IONVERSION)
target_link_libraries(CropRegions ion-analysis pthread dl)
install(TARGETS CropRegions DESTINATION bin)

add_executable(MergeRegions crop/MergeRegions.cpp crop/MergeAcq.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeRegions IONVERSION)
target_link_libraries(MergeRegions ion-analysis pthread dl)
install(TARGETS MergeRegions DESTINATION bin)

add_executable(MergeImages crop/MergeImages.cpp crop/MergeAcq.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeImages IONVERSION)
target_link_libraries(MergeImages ion-analysis pthread dl)
install(TARGETS MergeImages DESTINATION bin)

add_executable(MergeDats crop/MergeDats.cpp crop/MergeAcq.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeDats IONVERSION)
target_link_libraries(MergeDats ion-analysis pthread dl)
install(TARGETS MergeDats DESTINATION bin)

add_executable(Thumbnail crop/Thumbnail.cpp crop/MergeAcq.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Thumbnail IONVERSION)
target_link_libraries(Thumbnail ion-analysis pthread dl)
install(TARGETS Thumbnail DESTINATION bin)

# If doing pca spline trial
#add_executable(PcaSplineExample Image/PcaSplineExample.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
#add_dependencies(PcaSplineExample IONVERSION)
# If defined while running cmake look for mkl
# e.g. -DMKL_INSTALL_PATH=/home/ionadmin/intel/composer_xe_2013_sp1.0.080/mkl
#if (DEFINED MKL_INSTALL_PATH)
#  message("Compiling PcaSplineExample with mkl")
#  target_link_libraries(PcaSplineExample ion-analysis ${MKL_LINK_LIBS} -lpthread -lm  dl z)
#else()
#  message("Compiling PcaSplineExample with eigen only")
#  target_link_libraries(PcaSplineExample ion-analysis file-io pthread m dl z)
#endif()

# If doing DfcCompr trial
#add_executable(DfcComprExample Image/DfcComprExample.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
#add_dependencies(DfcComprExample IONVERSION)
# If defined while running cmake look for mkl
# e.g. -DMKL_INSTALL_PATH=/home/ionadmin/intel/composer_xe_2013_sp1.0.080/mkl
# include_directories("/home/ionadmin/intel/composer_xe_2013_sp1.0.080/mkl/include/")
#if (DEFINED MKL_INSTALL_PATH)
#  message("Compiling DfcComprExample with mkl")
#  target_link_libraries(DfcComprExample ion-analysis ${MKL_LINK_LIBS} -lpthread -lm  dl z)
#else()
#  message("Compiling DfcComprExample with eigen only")
#  target_link_libraries(DfcComprExample ion-analysis file-io pthread m dl z)
#endif()

add_executable(Archive crop/Archive.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Archive IONVERSION)
target_link_libraries(Archive ion-analysis pthread dl)

if(ION_DO_BACKGROUND)
    add_executable(Analysis AnalysisOrg/Analysis.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
    add_dependencies(Analysis IONVERSION)

    if (ION_USE_CUDA)
        target_link_libraries(Analysis ion-analysis pthread dl ${CUDA_LIBRARIES})
    else()
        target_link_libraries(Analysis ion-analysis pthread dl)
    endif()

    install(TARGETS Analysis DESTINATION bin)

#    add_executable(bkgFit bkgFit.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
#    add_dependencies(bkgFit IONVERSION)
#    if (ION_USE_CUDA)
#        target_link_libraries(bkgFit ion-analysis pthread ${CUDA_LIBRARIES} ${ION_ARMADILLO_LIBS})
#    else()
#	target_link_libraries(bkgFit ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
#    endif()
endif()

## Standalone BeadFind
add_executable(justBeadFind AnalysisOrg/justBeadFind/justBeadFind.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(justBeadFind IONVERSION)
target_link_libraries(justBeadFind ion-analysis ${CERES_LIB} ${MINIGLOG_LIB} ${GLOG_LIB} pthread dl)
install(TARGETS justBeadFind DESTINATION bin)

add_executable(BeadfindStats Separator/BeadfindStats.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadfindStats IONVERSION)
target_link_libraries(BeadfindStats ion-analysis pthread dl)
install(TARGETS BeadfindStats DESTINATION bin)

add_executable(CompareBf Separator/CompareBf.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CompareBf IONVERSION)
target_link_libraries(CompareBf ion-analysis pthread dl)

## Standalone BaseCaller
set(BaseCallerSRCS
    BaseCaller/BaseCaller.cpp
    BaseCaller/BaseCallerFilters.cpp
    BaseCaller/BaseCallerMetricSaver.cpp
    BaseCaller/BaseCallerParameters.cpp
    BaseCaller/BarcodeClassifier.cpp
    BaseCaller/BarcodeDatasets.cpp
    BaseCaller/MolecularTagTrimmer.cpp
    BaseCaller/OrderedDatasetWriter.cpp
    BaseCaller/PhaseEstimator.cpp
    BaseCaller/PerBaseQual.cpp
    BaseCaller/WellsNormalization.cpp
    Calibration/HistogramCalibration.cpp
    Calibration/LinearCalibrationModel.cpp
    
    ${PROJECT_BINARY_DIR}/IonVersion.cpp)
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    list(APPEND BaseCallerSRCS
        BaseCaller/TreephaserSSE.cpp)
endif()

add_executable(BaseCaller ${BaseCallerSRCS})
add_dependencies(BaseCaller IONVERSION bamtools)
target_link_libraries(BaseCaller ion-analysis pthread ${ION_BAMTOOLS_LIBS} dl)
install(TARGETS BaseCaller DESTINATION bin)


## Standalone Variant Caller, named tvc
set(ION_VCFLIB_DIR    ${ION_TS_EXTERNAL}/vcflib)
set(ION_FREEBAYES_DIR ${ION_TS_EXTERNAL}/freebayes)

include_directories(${ION_VCFLIB_DIR})
include_directories(${ION_FREEBAYES_DIR}/src)
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/EnsembleEval")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/Bookkeeping")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/Reads")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/Splice")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/Filter")
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/Consensus")
include_directories("${PROJECT_SOURCE_DIR}/realignment")
include_directories("${PROJECT_SOURCE_DIR}/Calibration")

set(tvcSRCS
  VariantCaller/VariantCaller.cpp
  VariantCaller/BAMWalkerEngine.cpp
  VariantCaller/OrderedBAMWriter.cpp
  VariantCaller/SampleManager.cpp
  VariantCaller/TargetsManager.cpp
  VariantCaller/HandleVariant.cpp
  VariantCaller/HotspotReader.cpp
  VariantCaller/MetricsManager.cpp
  VariantCaller/HypothesisEvaluator.cpp
  VariantCaller/MolecularTag.cpp
  
  VariantCaller/Bookkeeping/MiscUtil.cpp
  VariantCaller/Bookkeeping/ExtendParameters.cpp 
  VariantCaller/Bookkeeping/InputStructures.cpp
  VariantCaller/Bookkeeping/VcfFormat.cpp

  VariantCaller/Reads/ExtendedReadInfo.cpp
  
  VariantCaller/IndelAssembly/IndelAssembly.cpp
  
  VariantCaller/tvcutils/unify_vcf.cpp
  VariantCaller/tvcutils/prepare_hotspots.cpp
  
  VariantCaller/Splice/ErrorMotifs.cpp
  VariantCaller/Splice/LocalContext.cpp
  VariantCaller/Splice/ClassifyVariant.cpp
  VariantCaller/Splice/SpliceVariantHypotheses.cpp

  VariantCaller/Filter/DecisionTreeData.cpp
  VariantCaller/Filter/VariantAssist.cpp

  VariantCaller/EnsembleEval/DiagnosticJSON.cpp
  VariantCaller/EnsembleEval/BiasGenerator.cpp
  VariantCaller/EnsembleEval/SigmaGenerator.cpp
  VariantCaller/EnsembleEval/SkewGenerator.cpp
  VariantCaller/EnsembleEval/PosteriorInference.cpp
  VariantCaller/EnsembleEval/ShortStack.cpp
  VariantCaller/EnsembleEval/StackEngine.cpp
  VariantCaller/EnsembleEval/CrossHypotheses.cpp

  VariantCaller/Consensus/Consensus.cpp
  VariantCaller/Consensus/FlowSpaceConsensus.cpp
  VariantCaller/Consensus/ConsensusParameters.cpp

  # TODO: Actually build vcflib as a static library and link to variant caller.
  # TODO2: Resolve bgzf.c collisions between vcflib and bamtools
  ${ION_VCFLIB_DIR}/Variant.cpp
  ${ION_VCFLIB_DIR}/split.cpp
  ${ION_VCFLIB_DIR}/tabixpp/tabix.cpp
  ${ION_VCFLIB_DIR}/tabixpp/index.c
  ${ION_VCFLIB_DIR}/tabixpp/bgzf.c
  ${ION_VCFLIB_DIR}/smithwaterman/LeftAlign.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/Repeats.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/IndelAllele.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/SmithWatermanGotoh.cpp
  ${ION_FREEBAYES_DIR}/src/AlleleParser.cpp

  BaseCaller/PIDloop.cpp
  BaseCaller/DPTreephaser.cpp
  BaseCaller/MolecularTagTrimmer.cpp
  Calibration/LinearCalibrationModel.cpp
  Calibration/FlowAlignment.cpp
  Util/OptArgs.cpp
  AnalysisOrg/IO/OptBase.cpp
  realignment/Realigner.cpp
  ${ION_TS_EXTERNAL}/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
  ${PROJECT_BINARY_DIR}/IonVersion.cpp
)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    list(APPEND tvcSRCS
        BaseCaller/TreephaserSSE.cpp)
endif()

add_executable(tvc ${tvcSRCS})
target_link_libraries(tvc ${ION_BAMTOOLS_LIBS} ${EXTRA_LIBS} z file-io pthread)
add_dependencies(tvc IONVERSION bamtools armadillo_proj htslib_proj)
install(TARGETS tvc DESTINATION bin)

## tmol executable ##
# include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/tmol")
#set(tmolSRCS
#  VariantCaller/tmol/tmol.cpp
#  ${PROJECT_BINARY_DIR}/IonVersion.cpp
#)
#add_executable(tmol ${tmolSRCS})
#target_link_libraries(tmol ${ION_BAMTOOLS_LIBS} ${EXTRA_LIBS} z pthread)
#add_dependencies(tmol IONVERSION bamtools)
#install(TARGETS tmol DESTINATION bin)

## vcfcomp executable ##
include_directories("${PROJECT_SOURCE_DIR}/VariantCaller/vcfcomp")

set(vcfcompSRCS
  VariantCaller/vcfcomp/vcfcomp.cpp
  ${PROJECT_BINARY_DIR}/IonVersion.cpp
)

add_executable(vcfcomp ${vcfcompSRCS})
target_link_libraries(vcfcomp ${ION_SAMTOOLS_LIBS} ${ION_HTSLIB_LIBS} z pthread)
add_dependencies(vcfcomp IONVERSION samtools)
install(TARGETS vcfcomp DESTINATION bin)


# bbctools
set(bbctools_src
bbctools/src/AmpliconRegionStatistics.cpp
bbctools/src/AmpliconRegionStatistics.h
bbctools/src/BaseCoverage.cpp
bbctools/src/BaseCoverage.h
bbctools/src/BbcCoarse.cpp
bbctools/src/BbcCoarse.h
bbctools/src/BbcCreate.cpp
bbctools/src/BbcCreate.h
bbctools/src/BbcDepth.cpp
bbctools/src/BbcDepth.h
bbctools/src/BbcIndex.cpp
bbctools/src/BbcIndex.h
bbctools/src/BbcMain.cpp
bbctools/src/BbcUsage.h
bbctools/src/BbcUtils.cpp
bbctools/src/BbcUtils.h
bbctools/src/BbcView.cpp
bbctools/src/BbcView.h
bbctools/src/RegionCoverage.cpp
bbctools/src/RegionCoverage.h
bbctools/src/RegionStatistics.cpp
bbctools/src/RegionStatistics.h
bbctools/src/TrackReads.cpp
bbctools/src/TrackReads.h
)

add_executable(bbctools ${bbctools_src})
target_link_libraries(bbctools ${ION_BAMTOOLS_LIBS} z pthread)
add_dependencies(bbctools IONVERSION bamtools)
install(TARGETS bbctools DESTINATION bin)


add_executable(tvcassembly
  VariantCaller/IndelAssembly/IndelAssemblyMain.cpp
  VariantCaller/IndelAssembly/IndelAssembly.cpp
  VariantCaller/TargetsManager.cpp
  VariantCaller/SampleManager.cpp

  # TODO: Actually build vcflib as a static library and link to variant caller.
  # TODO2: Resolve bgzf.c collisions between vcflib and bamtools
  ${ION_VCFLIB_DIR}/Variant.cpp
  ${ION_VCFLIB_DIR}/split.cpp
  ${ION_VCFLIB_DIR}/tabixpp/tabix.cpp
  ${ION_VCFLIB_DIR}/tabixpp/index.c
  ${ION_VCFLIB_DIR}/tabixpp/bgzf.c
  ${ION_VCFLIB_DIR}/smithwaterman/LeftAlign.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/Repeats.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/IndelAllele.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/SmithWatermanGotoh.cpp

  Util/OptArgs.cpp
  ${ION_TS_EXTERNAL}/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
  ${PROJECT_BINARY_DIR}/IonVersion.cpp
)

target_link_libraries(tvcassembly ${ION_BAMTOOLS_LIBS} ${EXTRA_LIBS} z pthread)
add_dependencies(tvcassembly IONVERSION bamtools htslib_proj)
install(TARGETS tvcassembly DESTINATION bin)



add_executable(tvcutils
  VariantCaller/tvcutils/tvcutils.cpp
  VariantCaller/tvcutils/prepare_hotspots.cpp
  VariantCaller/tvcutils/validate_bed.cpp
  VariantCaller/tvcutils/unify_vcf.cpp
  VariantCaller/tvcutils/split_vcf.cpp
  VariantCaller/TargetsManager.cpp
  realignment/Realigner.cpp
  Util/OptArgs.cpp
#  Util/Utils.cpp
  ${ION_VCFLIB_DIR}/Variant.cpp
  ${ION_VCFLIB_DIR}/split.cpp
  ${ION_VCFLIB_DIR}/tabixpp/tabix.cpp
  ${ION_VCFLIB_DIR}/tabixpp/index.c
  ${ION_VCFLIB_DIR}/tabixpp/bgzf.c
  ${ION_VCFLIB_DIR}/smithwaterman/LeftAlign.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/Repeats.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/IndelAllele.cpp
  ${ION_VCFLIB_DIR}/smithwaterman/SmithWatermanGotoh.cpp
  ${ION_TS_EXTERNAL}/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp
  ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(tvcutils  ${ION_BAMTOOLS_LIBS} z)
add_dependencies(tvcutils IONVERSION bamtools)
install(TARGETS tvcutils DESTINATION bin)


add_executable(ionstats 
    ionstats/ionstats.cpp
    ionstats/ionstats_data.cpp
    ionstats/ionstats_basecaller.cpp
    ionstats/ionstats_alignment.cpp
    ionstats/ionstats_alignment_summary.cpp
    ionstats/ionstats_tf.cpp
    ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(ionstats IONVERSION bamtools)
target_link_libraries(ionstats ion-analysis ${ION_BAMTOOLS_LIBS} pthread dl)
install(TARGETS ionstats DESTINATION bin)

## Calibration - flowspace recalibration training in c++
add_executable(Calibration 
    Calibration/Calibration.cpp 
    Calibration/CalibrationHelper.cpp
    Calibration/FlowAlignment.cpp
    Calibration/HistogramCalibration.cpp
    VariantCaller/Bookkeeping/MiscUtil.cpp
    ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(Calibration IONVERSION bamtools)
target_link_libraries(Calibration ion-analysis ${ION_BAMTOOLS_LIBS} pthread dl)
install(TARGETS Calibration DESTINATION bin)

add_executable(bamrealignment
    realignment/bamrealignment.cpp
    realignment/Realigner.h
    realignment/Realigner.cpp
    Util/Utils.cpp
    Util/OptArgs.cpp
    ${PROJECT_BINARY_DIR}/IonVersion.cpp
    )
target_link_libraries(bamrealignment ${ION_BAMTOOLS_LIBS} z pthread)
add_dependencies(bamrealignment bamtools)
install(TARGETS bamrealignment DESTINATION bin)

add_executable(SamVsSam
        SamUtils/SamVsSam.cpp
)
target_link_libraries(SamVsSam ion-analysis ${ION_SAMTOOLS_LIBS} ${ION_HTSLIB_LIBS} pthread dl)
install(TARGETS SamVsSam DESTINATION bin)

add_executable(BeadmaskParse Mask/BeadmaskParse.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskParse IONVERSION)
target_link_libraries(BeadmaskParse ion-analysis pthread dl)
install(TARGETS BeadmaskParse DESTINATION bin)

add_executable(BeadmaskMerge Mask/BeadmaskMerge.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(BeadmaskMerge IONVERSION)
target_link_libraries(BeadmaskMerge ion-analysis pthread dl)
install(TARGETS BeadmaskMerge DESTINATION bin)

add_executable(MergeWells Wells/MergeWells.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(MergeWells IONVERSION)
target_link_libraries(MergeWells ion-analysis pthread dl)
install(TARGETS MergeWells DESTINATION bin)

add_executable(RawWellsCombine Wells/RawWellsCombine.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsCombine IONVERSION)
target_link_libraries(RawWellsCombine ion-analysis pthread dl)
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
        file-io/wells_chip.c
        file-io/wells_combine.c
        file-io/wells_data.c
        file-io/wells_header.c
        file-io/wells_mask.c
        file-io/wells_mask_combine.c
)

add_executable(alignStats
        SamUtils/alignStats_driver.cpp
        ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
target_link_libraries(alignStats ion-analysis z ${ION_SAMTOOLS_LIBS} ${ION_HTSLIB_LIBS} pthread dl)
add_dependencies(alignStats IONVERSION)
install(TARGETS alignStats DESTINATION bin)

add_executable(FlowErrTest
        SamUtils/FlowErrTest.cpp
)
target_link_libraries(FlowErrTest ion-analysis z ${ION_SAMTOOLS_LIBS} ${ION_HTSLIB_LIBS} pthread dl)


add_executable(RawWellsEquivalent Wells/RawWellsEquivalent.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsEquivalent IONVERSION)
target_link_libraries(RawWellsEquivalent ion-analysis pthread dl)
install(TARGETS RawWellsEquivalent DESTINATION bin)

add_executable(RawWellsConverter Wells/RawWellsConverter.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(RawWellsConverter IONVERSION)
target_link_libraries(RawWellsConverter ion-analysis pthread dl)
install(TARGETS RawWellsConverter DESTINATION bin)

# add_executable(DiffSeparator Separator/DiffSeparator.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
# add_dependencies(DiffSeparator IONVERSION)
# target_link_libraries(DiffSeparator ion-analysis pthread ${ION_ARMADILLO_LIBS} dl)
# install(TARGETS DiffSeparator DESTINATION bin)

add_executable(SeparatorCmp Separator/SeparatorCmp.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(SeparatorCmp IONVERSION)
target_link_libraries(SeparatorCmp ion-analysis pthread ${ION_ARMADILLO_LIBS} dl)

add_executable(CompressDeviation Separator/CompressDeviation.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(CompressDeviation IONVERSION)
target_link_libraries(CompressDeviation ion-analysis pthread ${ION_ARMADILLO_LIBS} dl)


add_executable(readDat readDat.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readDat IONVERSION)
#target_link_libraries(readDat ion-analysis pthread)
target_link_libraries(readDat ion-analysis ${GTEST_BOTH_LIBRARIES} pthread dl)
install(TARGETS readDat DESTINATION bin)

add_executable(readWells Wells/readWells.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(readWells IONVERSION)
target_link_libraries(readWells ion-analysis pthread dl)
install(TARGETS readWells DESTINATION bin)

add_executable(iontools file-io/main.c  ${PROJECT_BINARY_DIR}/IonVersion.cpp)
add_dependencies(iontools IONVERSION)
target_link_libraries(iontools ion-analysis file-io pthread dl)
install(TARGETS iontools DESTINATION bin)

add_executable(barcodeMaskParse Mask/barcodeMaskParse.cpp ${PROJECT_BINARY_DIR}/IonVersion.cpp)
target_link_libraries(barcodeMaskParse ion-analysis dl)
install(TARGETS barcodeMaskParse DESTINATION bin)

add_executable(BamDuplicates
        SamUtils/bamduplicates.cpp
       ${PROJECT_BINARY_DIR}/IonVersion.cpp
)
add_dependencies(BamDuplicates IONVERSION)
target_link_libraries( BamDuplicates ${ION_BAMTOOLS_LIBS} ion-analysis z pthread )
install( TARGETS BamDuplicates DESTINATION bin )

add_executable(seqCoverage coverage/seqCoverage.cpp coverage/interval_tree.cpp)
add_dependencies(seqCoverage IONVERSION)
install(TARGETS seqCoverage DESTINATION bin)

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
          "utest/comp_clean_test_data.txt"
          "${CMAKE_CURRENT_BINARY_DIR}/comp_clean_test_data.txt"
          COPYONLY
        )

        configure_file (
          "utest/comp_clean_test_ev_3.txt"
          "${CMAKE_CURRENT_BINARY_DIR}/comp_clean_test_ev_3.txt"
          COPYONLY
        )

#        add_executable(ComparatorClean_Test utest/ComparatorClean_Test.cpp)
#        target_link_libraries(ComparatorClean_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
#        add_test(ComparatorCleanTest ComparatorClean_Test --gtest_output=xml:./)

        add_executable(IonErr_Test utest/IonErr_Test.cpp)
        target_link_libraries(IonErr_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(IonErrTest IonErr_Test --gtest_output=xml:./)

#        add_executable(Mask_Test utest/Mask_Test.cpp)
#        target_link_libraries(Mask_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
#        add_test(MaskTest Mask_Test --gtest_output=xml:./)

        add_executable(Utils_Test utest/Utils_Test.cpp)
        target_link_libraries(Utils_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
        add_test(MaskTest Utils_Test --gtest_output=xml:./)

        # add_executable(Wells_Test utest/Wells_Test.cpp)
        # target_link_libraries(Wells_Test ion-analysis ${ION_HDF5_LIBS} ${GTEST_BOTH_LIBRARIES} pthread z)
        # add_test(WellsTest Wells_Test --gtest_output=xml:./)

        add_executable(DualGaussMixModel_Test utest/DualGaussMixModel_Test.cpp)
        target_link_libraries(DualGaussMixModel_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
#         add_test(DualGaussMixModelTest DualGaussMixModel_Test --gtest_output=xml:./)

        # add_executable(ZeromerDiff_Test utest/ZeromerDiff_Test.cpp)
        # target_link_libraries(ZeromerDiff_Test ion-analysis ${GTEST_BOTH_LIBRARIES}  )
        # add_test(ZeromerDiffTest ZeromerDiff_Test --gtest_output=xml:./)

        add_executable(FindSlopeChange_Test utest/FindSlopeChange_Test.cpp)
        target_link_libraries(FindSlopeChange_Test ion-analysis ${GTEST_BOTH_LIBRARIES} blas pthread )
#        add_test(FindSlopeChangeTest FindSlopeChange_Test --gtest_output=xml:./)

        add_executable(T0Model_Test utest/T0Model_Test.cpp)
        target_link_libraries(T0Model_Test ion-analysis ${GTEST_BOTH_LIBRARIES} blas pthread)
        add_test(T0ModelTest T0Model_Test --gtest_output=xml:./)

        # add_executable(KeyClassifier_Test utest/KeyClassifier_Test.cpp)
        # target_link_libraries(KeyClassifier_Test ion-analysis ${GTEST_BOTH_LIBRARIES}  )
        # add_test(KeyClassifierTest KeyClassifier_Test --gtest_output=xml:./)

        add_executable(NumericalComparison_Test utest/NumericalComparison_Test.cpp)
        target_link_libraries(NumericalComparison_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
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

#        add_executable(BitHandler_Test utest/BitHandler_Test.cpp)
#        target_link_libraries(BitHandler_Test ion-analysis ${GTEST_BOTH_LIBRARIES} pthread)
#        add_test(BitHandlerTest BitHandler_Test --gtest_output=xml:./)

endif()


#files used for configuration
#not actually source files at all
#
#
# exclusion masks by chip name
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_314.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_316.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_316v2.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_318.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_324.bin" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_p1.1.17.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_p1.0.19.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_p1.0.20.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_PQ.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_510.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_520.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_550.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_521.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_522.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_530.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_540.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/exclusionMask_541.txt" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_314" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_316" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_318" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/excludeMask_324" DESTINATION ${ION_INSTALL_PREFIX}/config)
#
# phredtables by chip name
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.PSP4.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.PSP4.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.PSP4.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.PSP4.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.PSP4.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.PSP4.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.PSP4.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.PSP4.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.B5.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.314.B5.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.B5.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316.B5.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.B5.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.316v2.B5.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.B5.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318.B5.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318D.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.318D.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.900" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.900.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.900.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.900.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.19" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.19.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.19.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.19.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.20" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.20.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.20.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.0.20.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_ampliseq_exome" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_ampliseq_exome.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_ampliseq_exome.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_ampliseq_exome.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im1" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im1.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im1.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im1.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im2" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im2.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im2.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.17_im2.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.541" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.541.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.541.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.1.541.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18_im2" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18_im2.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18_im2.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p1.2.18_im2.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.0.1" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.0.1.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.0.1.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.0.1.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.1.1" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.1.1.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.1.1.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.1.1.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.1" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.1.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.1.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.1.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.2" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.2.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.2.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.2.2.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.3.1" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.3.1.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.3.1.Recal" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.p2.3.1.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.520.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.520.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.521.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.521.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.521_ExT.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.521_ExT.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.522.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.522.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.530.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.530.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.540.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.540.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.541.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.541.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.550.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/phredTable.550.Recal.h5" DESTINATION ${ION_INSTALL_PREFIX}/config)

install(FILES "${PROJECT_SOURCE_DIR}/config/args_314_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_314_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_316_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_316_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_316v2_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_316v2_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318D_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318D_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318select_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_318select_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_520_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_520_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_521_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_521_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_522_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_522_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_530_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_530_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_540_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_540_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_541_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_541_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_550_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_550_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_900_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_900_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.0.19_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.0.19_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.0.20_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.0.20_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.1.17_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.1.17_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.1.541_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.1.541_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.2.18_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P1.2.18_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.2.1_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.2.1_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.2.2_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.2.2_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.0.1_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.0.1_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.1.1_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.1.1_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.3.1_analysis.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/args_P2.3.1_beadfind.json" DESTINATION ${ION_INSTALL_PREFIX}/config)

#
#
#gopt parameters by chip type
#gopt parameters
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_318D.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_314v2_Hi-Q.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_316v2_Hi-Q.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_318v2_Hi-Q.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_314.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_316.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_316v2.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_318.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.0.19.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.0.20.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.1.17.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.1.17_ampliseq_exome.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.1.541.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p1.2.18.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p2.0.1.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p2.1.1.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p2.2.1.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p2.2.2.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_p2.3.1.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_520.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_520_ExT.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_521.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_521_ExT.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_522.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_530.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_530_ExT.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_540.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_540_ampliseq_exome.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_541.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/gopt_550.param.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
#
#
#
#cross talk configurations by chip type
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p1.2.well.settings.20140102.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p2.2.1.settings.20140120.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p2.2.1.null.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p2.2.2.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p2.2.2.null.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.trace.p1.0.20.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.trace.520.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.trace.521.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.trace.522.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.trace.530.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.550.well.settings.json" DESTINATION ${ION_INSTALL_PREFIX}/config)
install(FILES "${PROJECT_SOURCE_DIR}/config/xtalk.p2.3.1.well.settings.json" DESTINATION ${ION_INSTALL_PREFIX}/config)

# Recalibration Panel
install(FILES "${PROJECT_SOURCE_DIR}/config/datasets_calibration.json" DESTINATION ${ION_INSTALL_PREFIX}/config)

if(NOT ION_USE_SYSTEM_ARMADILLO)
   install(FILES "${PROJECT_BINARY_DIR}/../armadillo-6.100.0+ion0-install/lib/libarmadillo.so.6" DESTINATION lib)
   install(FILES "${PROJECT_BINARY_DIR}/../armadillo-6.100.0+ion0-install/lib/libarmadillo.so.6.100.0" DESTINATION lib)
endif()

install(PROGRAMS "${PROJECT_BINARY_DIR}/../htslib-1.2.1+ion0/tabix" DESTINATION bin)
install(PROGRAMS "${PROJECT_BINARY_DIR}/../htslib-1.2.1+ion0/bgzip" DESTINATION bin)
install(PROGRAMS "${PROJECT_BINARY_DIR}/../samtools-1.2+ion2/samtools" DESTINATION bin)

add_test("trimfastq.pl" "perl" "-cw" "${PROJECT_SOURCE_DIR}/trimfastq.pl")
install(PROGRAMS "${PROJECT_SOURCE_DIR}/trimfastq.pl" DESTINATION bin)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/build_genome_index.pl" DESTINATION bin)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/validate_reference.pl" DESTINATION bin)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/alignmentQC.pl" DESTINATION bin)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/getref" DESTINATION bin)
install(PROGRAMS "${PROJECT_SOURCE_DIR}/updateref.py" DESTINATION bin RENAME updateref)

install(DIRECTORY "${PROJECT_BINARY_DIR}/../picard-1.130/" DESTINATION "/opt/picard/picard-tools-current/")

add_subdirectory (MapLab)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  add_subdirectory(TMAP)
endif()

include(../buildTools/cmake/CMakeLists.cpack.txt)
#% nice to automate this
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6,
    libgcc1,
    libgfortran3,
    libstdc++6,
    perl,
    ion-gpu (>=2.5.0)")

set(CPACK_DEBIAN_PACKAGE_BREAKS "ion-gpu (<< 5.1.5), ion-alignment")
set(CPACK_DEBIAN_PACKAGE_REPLACES "ion-gpu (<< 5.1.5), ion-alignment")

set(CPACK_PACKAGE_DESCRIPTION "ion-analysis provides the command line workflow and supporting tools to process raw DAT files into bam and  fastq files.")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${CMAKE_CURRENT_BINARY_DIR}/debian/postinst
    ${CMAKE_CURRENT_BINARY_DIR}/debian/prerm
)
include(CPack)
