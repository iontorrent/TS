/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * BkgGpuPipeline.cu
 *
 *  Created on: Jul 7, 2014
 *      Author: jakob
 */


#include <iostream>
#include "LayoutTranslator.h"
#include "LayoutTester.h"
#include "MasterKernel.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"
#include "GpuPipelineDefines.h"
#include "BkgGpuPipeline.h"
#include "ClonalFilterWrapper.h"


using namespace std;


#define INJECT_FG_TRACES_REG_FITTING 0

#define DEBUG_SYNC 0  //forces sync after each kernel/async call to guarantee correct error catching
#define DEBUG_OUTPUT 0  //fully verbose


#define DEBUG_REGION 14 // region to print for debugging, if DEBUG_REGION_ALL is set this value is ignored
#define DEBUG_REGION_ALL 0  // if set to 1 all regions are printed for debugging if 0 only the DEBUG_REGION is printed

#define RESULTS_CHECK 0
#define RESULT_DUMP 0  // 1 = dump 0 = compare
#define INIT_CONTROL_OUPUT 1

#define SAMPLE_CONTROL 0
#define EMPTY_CONTROL 0

//use dump for the following values
#define READ_EMPTY_FROM_FILE 0
#define READ_EMPHASIS_FROM_FILE 0
#define READ_FGBUFFER_FROM_FILE 0
#define COMPARE_FG_BUFFER_FROM_FILE 0
//////////////////////////////////////////////

//Todo:
//Remove all flowIdx references and any flow index related calls. we are only working with real flow num from now on and a flowidx of 0


//#define FIRST_FLOW 20
#define POLYCLONAL_FILTER_UPDATE_FLOW 79


BkgGpuPipeline::BkgGpuPipeline(
    BkgModelWorkInfo* pbkinfo, 
    int fbSize, 
    int startingFlow,
    int numFlowBuffers,
    int deviceId)
{ 

  checkAvailableDevMem();

  startFlowNum = startingFlow; 
  flowBlockSize = fbSize;
  this->bkinfo = pbkinfo;
  const RawImage * rpt = bkinfo->img->GetImage();
  const SpatialContext * loc = &bkinfo[0].inception_state->loc_context;

  ImgP.init(rpt->cols, rpt->rows, loc->regionXSize, loc->regionYSize);
  ImgP.print();
  copySymbolsToDevice(ImgP);

  PopulateSymbolConstantImgageParams(ImgP, ConstFrmP, bkinfo);
  PopulateSymbolConstantGlobal(ConstGP,bkinfo);
  PopulateSymbolConfigParams(ConfP,bkinfo);
  ConstXTP.XtalkSymbolInit();
  copySymbolsToDevice(ConstXTP);

  //Todo: Mulit-Device support
  CreatePoissonApproxOnDevice(deviceId);

  //ToDo: exception handling for Unsuccessful allocate
  try{
    Dev = new DeviceData(ImgP,ConstFrmP);
  }catch(cudaException &e){
    e.Print();
    throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
  }

  Host = new HostData(ImgP,ConstFrmP);


  try{
    resultsHostBuf = new GPUResultsBuffer(ImgP, numFlowBuffers);
  }catch(cudaException &e){
    e.Print();
    delete Host;
    delete Dev;
    throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
  }

  SpDev = NULL;

  Host->BfMask.wrappPtr(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);

  PopulateSymbolPerFlowGlobal(GpFP, bkinfo, 0);

  PrepareInputsForSetupKernel();
  ExecuteT0AvgNumLBeadKernel();
  InitPersistentData();
}


BkgGpuPipeline::~BkgGpuPipeline()
{
  cout << "CUDA: Starting cleanup flow by flow GPU pipeline" << endl;
  checkAvailableDevMem();
  if(SpDev != NULL) delete SpDev;
  if(Dev != NULL) delete Dev;
  if(Host != NULL) delete Host;
  if(resultsHostBuf != NULL) delete resultsHostBuf;
  cout << "CUDA: Cleanup flow by flow GPU pipeline completed" << endl;
  checkAvailableDevMem();
}


bool BkgGpuPipeline::firstFlow(){
  //return(GpFP.getRealFnum() == FIRST_FLOW) ;
  return startFlowNum;
}

size_t BkgGpuPipeline::checkAvailableDevMem()
{
  int dev_id;
  size_t free_byte ;
  size_t total_byte ;
  double divMB = 1024.0*1024.0;
  cudaMemGetInfo( &free_byte, &total_byte );
  cudaGetDevice( &dev_id );
  cout << "CUDA " << dev_id << " GPU memory usage: used = " << (total_byte-free_byte)/divMB<< ", free = " << free_byte/divMB<< " MB, total = "<< total_byte/divMB<<" MB" << endl;
  return free_byte;
}


void BkgGpuPipeline::PrepareInputsForSetupKernel()
{
  Dev->BfMask.copyIn(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
  Dev->T0.copyIn( &(*(bkinfo->smooth_t0_est))[0]);

  Host->BeadStateMask.memSet(0);
  for(size_t i=0; i< ImgP.getNumRegions(); i++){
    WorkSet myJob(&bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      TranslateBeadStateMask_RegionToCube(Host->BeadStateMask, &bkinfo[i],regId);
    }
  }
  Dev->BeadStateMask.copy(Host->BeadStateMask);

}




void BkgGpuPipeline::InitPersistentData()
{

  //Temporary Host Buffers only needed once for setup
  LayoutCubeWithRegions<float>HostRegionFrameCube(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), Rf_NUM_PARAMS, HostMem);
  LayoutCubeWithRegions<int>HostRegionFramesPerPoint(ImgP.getGridParam(ConstFrmP.getMaxCompFrames()), 1, HostMem);
  LayoutCubeWithRegions<float> HostBeadParamCube(ImgP, Bp_NUM_PARAMS, HostMem);
  LayoutCubeWithRegions<float> HostPolyClonalCube(ImgP, Poly_NUM_PARAMS, HostMem);
  LayoutCubeWithRegions<int>HostNonZeroEmphasisFrames(ImgP.getGridParam(MAX_POISSON_TABLE_COL),Nz_NUM_PARAMS, HostMem);
  LayoutCubeWithRegions<PerNucParamsRegion>HostPerNucRegP(ImgP.getGridParam(),NUMNUC,HostMem);

  for(size_t i=0; i < ImgP.getNumRegions(); i++)
  {
    //setup step to guarantee all host side buffers are generated and available
    bkinfo[i].bkgObj->SetFittersIfNeeded();

    WorkSet myJob(&bkinfo[i]);
    if(myJob.DataAvailalbe()){
      //determine region Id based on start coordinates of region
      size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      int numf = myJob.getNumFrames();
      Host->NumFrames.putAtReg(myJob.getNumFrames(), regId);
      TranslateConstantRegionParams_RegionToCube(Host->ConstRegP,&bkinfo[i],regId);
      TranslateRegionFrameCube_RegionToCube(HostRegionFrameCube, &bkinfo[i], regId);
      TranslateRegionFramesPerPoint_RegionToCube(HostRegionFramesPerPoint,&bkinfo[i], regId);
      TranslateBeadParams_RegionToCube(HostBeadParamCube, &bkinfo[i], regId);
      TranslatePolyClonal_RegionToCube(HostPolyClonalCube,&bkinfo[i], regId);
      TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
      TranslatePerNucRegionParams_RegionToCube(HostPerNucRegP,&bkinfo[i],regId);

    }
  }

  //Copy temp buffers to device
  Dev->NumFrames.copy(Host->NumFrames);
  Dev->ConstRegP.copy(Host->ConstRegP);
  Dev->RegionFrameCube.copy(HostRegionFrameCube);
  Dev->RegionFramesPerPoint.copy(HostRegionFramesPerPoint);
  Dev->BeadParamCube.copy(HostBeadParamCube);
  Dev->PolyClonalCube.copy(HostPolyClonalCube);
  Dev->NonZeroEmphasisFrames.copy(HostNonZeroEmphasisFrames);
  Dev->PerNucRegP.copy(HostPerNucRegP);

#if INIT_CONTROL_OUPUT
  cout << "BkgGpuPipeline: InitPersistentData: num Time-Compressed-Frames Per Region:" << endl;
  Host->NumFrames.printRegionTable<size_t>();
#endif

}


void BkgGpuPipeline::PerFlowDataUpdate(BkgModelWorkInfo* pbkinfo,int flowInBlock)
{

  //Per FLow Inputs:
  this->bkinfo = pbkinfo;

  WorkSet myJob(&bkinfo[0]);
  if (!(myJob.performPostFitHandshake())) {
    GlobalDefaultsForBkgModel tmp = bkinfo->bkgObj->getGlobalDefaultsForBkgModel();
    for(size_t i=0; i < getParams().getNumRegions(); i++)
    {
      SignalProcessingMasterFitter * Obj = bkinfo[i].bkgObj;
      Obj->region_data->AddOneFlowToBuffer ( tmp,*(Obj->region_data_extras.my_flow), bkinfo[i].flow);
      Obj->region_data_extras.my_flow->Increment();
      Obj->region_data->my_beads.ZeroOutPins(  Obj->region_data->region,
          Obj->GetGlobalStage().bfmask,
          *Obj->GetGlobalStage().pinnedInFlow,
          Obj->region_data_extras.my_flow->flow_ndx_map[flowInBlock],
          flowInBlock);
    }
  }

  //updated per flow device Symbols and buffers
  PopulateSymbolPerFlowGlobal(GpFP, bkinfo, flowInBlock);


  if(!firstFlow()){ //  copied in constructor for first flow
    //Host->BfMask.copyIn(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);  // debugging only
    Host->BfMask.wrappPtr(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
    Dev->BfMask.copy(Host->BfMask);
  }

  //raw image
  Host->RawTraces.wrappPtr(bkinfo->img->raw->image);
  Dev->RawTraces.copy(Host->RawTraces);

}


/////////////////////////////////////////////////////////////////////////////////
// Kernels


void BkgGpuPipeline::ExecuteT0AvgNumLBeadKernel()
{

  dim3 block(32, 4);
  dim3 grid(ImgP.getGridDimX(),ImgP.getGridDimY());
  size_t smem = 2*(block.x * block.y *sizeof(int));

  Dev->NumSamples.memSet(0);

#if DEBUG_OUTPUT
  cout << "BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: executing GenerateT0AvgAndNumLBeads Kernel with: block(" << block.x << "," << block.y <<"), grid(" << grid.x << "," << grid.y  << ")  and smem: "<< smem << endl;
#endif
  GenerateT0AvgAndNumLBeads_New<<<grid, block, smem>>>(
      Dev->RegionStateMask.getPtr(),
      Dev->BfMask.getPtr(),
      Dev->BeadStateMask.getPtr(),
      Dev->T0.getPtr(),
      Dev->SampleRowPtr.getPtr(),
      Dev->NumSamples.getPtr(),
      Dev->NumLBeads.getPtr(), //numLbeads of whole region
      Dev->T0Avg.getPtr() // T0Avg per REgion //ToDo check if this is really needed of if updating the T0Est would be better
  );

#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  CUDA_ERROR_CHECK();
#endif

#if INIT_CONTROL_OUPUT || DEBUG_OUTPUT
  LayoutCubeWithRegions<float>HostT0Avg(Dev->T0Avg,HostMem);
  LayoutCubeWithRegions<int>HostNumLBeads(Dev->NumLBeads, HostMem);
  LayoutCubeWithRegions<float>HostT0(Dev->T0, HostMem);

  cout << "BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: num live beads per region: " << endl;
  HostNumLBeads.printRegionTable<int>();
  cout << "BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: T0 avg per region: " << endl;
  HostT0Avg.printRegionTable<float>();
  //LayoutCubeWithRegions<float> RegionStdDev(ImgP.getGridDimX(), ImgP.getGridDimY());
  //for(size_t i=0; i<ImgP.getNumRegions(); i++){
  //RegionStdDev[i] = HostT0.getStdDevReg<float>(i,0,HostT0Avg[i],0,&Host->BfMask,(unsigned short)MaskLive);
  //}
  //cout << "BkgGpuPipeline: BkgGpuPipeline: std deviation per region: "<< endl;
  //RegionStdDev.printRegionTable<float>();
#endif
#if INIT_CONTROL_OUPUT || DEBUG_OUTPUT || SAMPLE_CONTROL
  LayoutCubeWithRegions<int>HostSampleCount(Dev->SampleRowPtr, HostMem);
  LayoutCubeWithRegions<int>HostNumSamples(Dev->NumSamples, HostMem);
  cout << "Number of samples for regional fitting per Region:" << endl;
  HostNumSamples.printRegionTable<int>();
#endif
#if SAMPLE_CONTROL || DEBUG_OUTPUT
  HostSampleCount.setRWStrideX();
  cout << "starting offset for samples per Row (last entry is num samples)" << endl;
  for(size_t rg = 0; rg < ImgP.getNumRegions(); rg++){
    if(rg == DEBUG_REGION || DEBUG_REGION_ALL)
      cout << "regId " << rg << "," << HostSampleCount.getCSVatReg<int>(rg,0,0,0,ImgP.getRegH()) << endl;
  }
#endif


}

void BkgGpuPipeline::ExecuteGenerateBeadTrace()
{
  dim3 block(32,4);
  dim3 grid(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);

  int cacheSetting = 0;

  size_t numTBlocksPerReg = (ImgP.getRegH()+block.y-1)/block.y;

  //Special Buffers for this kernel dependent on launch configuration
  if(SpDev == NULL)
    SpDev = new SpecialDeviceData(ImgP,ConstFrmP,numTBlocksPerReg);
  //ToDo: exception Handling if alloc fails

  Dev->EmptyTraceComplete.memSet(0);
  Dev->EmptyTraceAvg.memSet(0);

  SpDev->EmptyTraceSumRegionTBlock.memSet(0);
  SpDev->EmptyTraceCountRegionTBlock.memSet(0);

  Dev->SampleRowCounter.memSet(0);
  Dev->SampleCompressedTraces.memSet(0);
  Dev->SampleParamCube.memSet(0);
  Dev->SampleStateMask.memSet(0);

  size_t smem = block.x * block.y *sizeof(int);

  switch(cacheSetting){
    case 0:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferEqual);
      cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferEqual" << endl;
      break;
    case 2:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferL1);
      cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferL1" << endl;
      break;
    case 1:
    default:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferShared);
      //#if DEBUG_OUTPUT
      cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferShared" << endl;
      //#endif

  }

#if DEBUG_OUTPUT
  cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: executing GenerateAllBeadTraceFromMeta_k Kernel with: block(" << block.x << "," << block.y <<"), grid(" << grid.x << "," << grid.y  << ")  and smem: "<< smem << endl;
#endif

  GenerateAllBeadTraceEmptyFromMeta_k<<<grid, block, smem >>> (
      Dev->RegionStateMask.getPtr(),
      Dev->RawTraces.getPtr(),  //perwell    input and output
      Dev->BfMask.getPtr(), //per well
      Dev->T0.getPtr(), //per well
      Dev->RegionFrameCube.getPtrToPlane(RfFrameNumber),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->NumFrames.getPtr(),  //frames per region
      Dev->NumLBeads.getPtr(),
      Dev->T0Avg.getPtr(),  // ToDo: try already subtract T0 after calculating the average so this would not be needed here anymore!
      Dev->ConstRegP.getPtr(),
      Dev->PerFlowRegionParams.getPtr(),
      Dev->EmptyTraceAvg.getPtr(), // has to be initialized to 0!! will contain avg of all empty trace frames for each region
      SpDev->EmptyTraceSumRegionTBlock.getPtr(), // has to be initialized to 0!! will contain avg of all empty trace frames for each region
      SpDev->EmptyTraceCountRegionTBlock.getPtr(), // has to be initialized to 0!! will contain number of empty traces summed up for each region
      Dev->EmptyTraceComplete.getPtr(), //has to be initialized to 0!! completion counter per region for final sum ToDo: figure out if we can do without it
      // for regional fit sample extraction:
      //inputs
      Dev->BeadParamCube.getPtr(),
      Dev->BeadStateMask.getPtr(),
      //meta data
      Dev->SampleRowPtr.getPtr(),
      Dev->SampleRowCounter.getPtr(),
      //outputs
      Dev->SampleStateMask.getPtr(),
      Dev->SampleCompressedTraces.getPtr(),
      Dev->SampleParamCube.getPtr(),
      Dev->SampleCoord.getPtr()


  );



#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  CUDA_ERROR_CHECK();
#endif

  dim3 blockER(32,1);
  dim3 gridER(ImgP.getGridDimX(), ImgP.getGridDimY());

  smem = (blockER.y * ConstFrmP.getUncompFrames()  + blockER.y )* sizeof(float);


  //LayoutCubeWithRegions<float> DevDcOffsetDebug(ImgP.getGridParam(),1,DeviceGlobal);

#if DEBUG_OUTPUT
  cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: executing ReduceEmptyAverage_k Kernel with: block(" << blockER.x << "," << blockER.y <<"), grid(" << gridER.x << "," << gridER.y  << ")  and smem: "<< smem << endl;
#endif

  ReduceEmptyAverage_k<<<gridER, blockER, smem>>>(
      Dev->RegionStateMask.getPtr(),
      Dev->EmptyTraceAvg.getPtr(),
      Dev->ConstRegP.getPtr(),
      Dev->PerFlowRegionParams.getPtr(),
      Dev->RegionFrameCube.getPtrToPlane(RfFrameNumber),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->NumFrames.getPtr(),  //frames p
      SpDev->EmptyTraceSumRegionTBlock.getPtr(), // has to be initialized to 0!! will contain avg of all empty trace frames for each region
      SpDev->EmptyTraceCountRegionTBlock.getPtr(), // has to be initialized to 0!! will contain number of empty traces summed up for each region
      numTBlocksPerReg
      //DevDcOffsetDebug.getPtr()
  );

  /*
  LayoutCubeWithRegions<float> HostDcOffset(DevDcOffsetDebug,HostMem);
  HostDcOffset.setRWStrideX();
  HostDcOffset.setRWPtr(0);
  cout << GpFP.getRealFnum() <<"," << GpFP.getNucId();
  for(size_t i=0; i< HostDcOffset.numElements(); i++)
    cout << "," << HostDcOffset.read();
  cout << endl;
   */

#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  CUDA_ERROR_CHECK();
#endif

  if(GpFP.getRealFnum() == 20){
    cout << "Region State: " << endl;
    LayoutCubeWithRegions<unsigned short>HostRegionMask(Dev->RegionStateMask,HostMem);
    HostRegionMask.printRegionTable<unsigned short>();
  }




#if EMPTY_CONTROL || DEBUG_OUTPUT
  //static LayoutCubeWithRegions<int>HostEmptyTraceComplete(ImgP.getGridParam(),1,HostMem);HostEmptyTraceComplete.trackMe(muT);
  static LayoutCubeWithRegions<float>HostEmptyTraceAvg(Dev->EmptyTraceAvg,HostMem);
  HostEmptyTraceAvg.copy(Dev->EmptyTraceAvg);
  HostEmptyTraceAvg.setRWStrideX();
#if EMPTY_CONTROL
  static CubePerFlowDump<float> emptyDump(  ImgP.getGridParam(ConstFrmP.getUncompFrames()), 1, 1);
  emptyDump.setFilePathPrefix("EmptyTraces");
  emptyDump.getFlowCube(GpFP.getRealFnum()).setRWStrideX();
#endif
  cout << "BkgGpuPipeline: ExecuteGenerateBeadTrace: Average Empty Traces:" << endl;
  for(size_t regId = 0; regId < ImgP.getNumRegions(); regId++){
    if(regId == DEBUG_REGION || DEBUG_REGION_ALL   || EMPTY_CONTROL){
      int nf = Host->NumFrames.getAtReg(regId);
      if(nf <= 0 || nf > ConstFrmP.getMaxCompFrames())
        nf = ConstFrmP.getMaxCompFrames();
      cout <<"BkgGpuPipeline: ExecuteGenerateBeadTrace: DEBUG GPU " << regId <<"," << nf << "," << HostEmptyTraceAvg.getCSVatReg<float>(regId,0,0,0,nf) << endl;
#if EMPTY_CONTROL
      cout <<"BkgGpuPipeline: ExecuteGenerateBeadTrace: DEBUG CPU " << regId <<"," << nf << "," << emptyDump.getFlowCube(GpFP.getRealFnum()).getCSVatReg<float>(regId,0,0,0,nf) << endl;
#endif
    }
  }


#endif
#if SAMPLE_CONTROL
  LayoutCubeWithRegions<int> HostNumSample(Dev->NumSamples,HostMem);
  LayoutCubeWithRegions<short> SampleHostCompressedTraces(Dev->SampleCompressedTraces,HostMem);
  LayoutCubeWithRegions<SampleCoordPair> HostSamplesCoords(Dev->SampleCoord, HostMem);
  SampleHostCompressedTraces.setRWStrideZ();
  HostSamplesCoords.setRWStrideX();
  for(size_t rgid =0 ; rgid < ImgP.getNumRegions(); rgid++){
    if( rgid == DEBUG_REGION || DEBUG_REGION_ALL){
      HostSamplesCoords.setRWPtrRegion(rgid);
      for(int i = 0; i < HostNumSample.getAtReg(rgid); i++){
        int nf = Host->NumFrames.getAtReg(rgid);
        if(nf <= 0 || nf > ConstFrmP.getMaxCompFrames())
          nf = ConstFrmP.getMaxCompFrames();
        SampleCoordPair loc = HostSamplesCoords.read();
        cout << "regId," << rgid <<",x,"<< loc.x << ",y,"<< loc.y << "," << SampleHostCompressedTraces.getCSVatReg<short>(rgid,i,0,0,nf) << endl;;
      }
    }
  }
#endif

}



void BkgGpuPipeline::ExecuteSingleFlowFit()
{

  int cacheSetting = 0;

  dim3 block(32,4);

  dim3 grid(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);

#if EMPTY_IN_SHARED
  size_t smem = (MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames() + ConstFrmP.getUncompFrames())  * sizeof(float);
#else
  size_t smem = (MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames())  * sizeof(float);
#endif

#if DEBUG_OUTPUT
  cout << "BkgGpuPipeline: ExecuteSingleFlowFit: executing ExecuteThreadBlockPerRegion2DBlocks Kernel (SingleFlowFit) with: block(" << block.x << "," << block.y <<"), grid(" << grid.x << "," << grid.y  << ")  and smem: "<< smem << endl;
#endif

  switch(cacheSetting){
    case 0:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferEqual);
      cout << "BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferEqual" << endl;
      break;
    case 2:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferL1);
      cout << "BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferL1" << endl;
      break;
    case 1:
    default:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferShared);
      //#if DEBUG_OUTPUT
      cout << "BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferShared" << endl;
      //#endif

  }

  Dev->ResultCube.memSet(0);
  //  Dev->RawTraces.memSet(0,(ConstFrmP.getRawFrames()-3)*ImgP.getImgSize()*sizeof(short),ImgP.getImgSize()*sizeof(short)*3);
  //   Dev->RawTraces.memSetPlane(0,ConstFrmP.getRawFrames()-3, 3);
  //TMemSegPairAlloc<int> numLBeads(sizeof(int)*36,HostPageLocked,DeviceGlobal);
  //numLBeads.memSet(0);

  // Do we need this here ?
  Dev->ResultCube.memSetPlane(0,ResultAmpl);
  //  Dev->ResultCube.memSetPlane(0,ResultAmplXTalk);

  ExecuteThreadBlockPerRegion2DBlocksDense<<<grid, block, smem >>>(
      Dev->RegionStateMask.getPtr(),
      Dev->BfMask.getPtr(),
      Dev->BeadStateMask.getPtr(),
      Dev->RawTraces.getPtr(),
      Dev->BeadParamCube.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr(),
      Dev->NucRise.getPtr(),
      Dev->ResultCube.getPtr(),
      Dev->NumFrames.getPtr(), // moce to constant per region
      Dev->NumLBeads.getPtr(),
      Dev->ConstRegP.getPtr(),
      Dev->PerFlowRegionParams.getPtr(),
      Dev->PerNucRegP.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->EmptyTraceAvg.getPtr()
      //DeviceRegionFrameCube.getPtr() + RfBkgTraces * ImgData.maxCompFrames * ImgP.getNumRegions()
      //debug buffers
      //numLBeads.getPtr()
      //                                   fgBufferFloat.getPtr()
  );


#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  cout << "BkgGpuPipeline: ExecuteSingleFlowFit: Finalize" << endl;
  CUDA_ERROR_CHECK();
#endif

  /*
  LayoutCubeWithRegions<short>HostIterCount(ImgP,2,HostMem);
 HostIterCount.copyPlanes(Dev->RawTraces,ConstFrmP.getRawFrames()-2,0,2);

  HostIterCount.setRWStrideX();
  size_t iterCounter[9] = {0};
  cout << "Iteration counter: " << endl;
  for(size_t i= 0; i<2; i++){
    HostIterCount.setRWPtr(0,0,i);
    for(size_t x = 0; x < ImgP.getImgSize(); x++){
      int val = HostIterCount.read();
      if( val < 0 || val > 8 ) val = 0;
      iterCounter[val]++;
    }
    cout << "iterPass: " << i << ",";
    for(int c=0; c<9; c++){
      cout << iterCounter[c] << ",";
      iterCounter[c] = 0;
    }
    cout << endl;
  }

 int numWarpsPerRegRow = (ImgP.getRegW()+31)/32;
 LayoutCubeWithRegions<short>HostIterPerWarp(ImgP.getGridParam(numWarpsPerRegRow,ImgP.getRegH()),1,HostMem);

 HostIterCount.copySubSet( Dev->RawTraces,  //src
                           (ConstFrmP.getRawFrames()-3)*sizeof(short)*ImgP.getImgSize(),  //srcOffset in bytes
                           0, //dstOffset in bytes
                           HostIterPerWarp.getParams().getImgSize()*sizeof(short) //copy size in bytes
                           );

 size_t numwarps = 0;
 size_t iterCounterAll[9] = {0};
 HostIterCount.setRWStrideX();
 for(size_t reg = 0; reg < ImgP.getNumRegions(); reg++){
   size_t iterCounterReg[9] = {0};
   for(size_t row=0; row < ImgP.getRegH(reg); row++){
     HostIterCount.setRWPtrRegion(reg,0,row);
     for(int w=0; w < numWarpsPerRegRow && w*32 < ImgP.getRegW(reg); w++)
     {
       int val = HostIterCount.read();
       if( val < 0 || val > 8 ) val = 0;
       iterCounterReg[val]++;
       iterCounterAll[val]++;
       numwarps++;
     }
   }
   cout << "Region: " << reg << " numWarps: " << numWarpsPerRegRow*ImgP.getRegH(reg) <<  " max iter: ";
   for(int c=0; c<9; c++){
     cout << iterCounterReg[c] << ",";
   }
   cout << endl;
 }
 cout << "Max iterations within all " <<  numwarps << " warps: ";
 for(int c=0; c<9; c++){
     cout << iterCounterAll[c] << ",";
     iterCounterAll[c] = 0;
  }
  cout << endl;

   */

}



void BkgGpuPipeline::HandleResults()
{
  WorkSet myJob(&bkinfo[0]);
  if (myJob.performPostFitHandshake()) {
    getDataForRawWells();
    ApplyClonalFilter();
  }
  else {

    getDataForPostFitStepsOnHost();

    // No transaltion is required if background thread writing to raw wells. 
    // It can take care of translation if required    
    for(size_t i=0; i< ImgP.getNumRegions(); i++){

      WorkSet myJob(&bkinfo[i]);
      size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      if(myJob.DataAvailalbe()){
        TranslateResults_CubeToRegion(Host->ResultCube,&bkinfo[i],GpFP.getFlowIdx(),regId);
        TranslateBeadStateMask_CubeToRegion(Host->BeadStateMask,&bkinfo[i],regId);
        TranslatePerFlowRegionParams_CubeToRegion(Host->PerFlowRegionParams, &bkinfo[i], regId);
      }

      myJob.setJobToPostFitStep();
      WorkerInfoQueueItem item;
      item.private_data = (void*)&bkinfo[i];
      myJob.putJobToCPU(item);
    }

    cout << "BkgGpuPipeline: Reinjecting results for flowblock containing flows "<< getFlowP().getRealFnum() - flowBlockSize << " to " << getFlowP().getRealFnum() << endl;
    cout << "waiting on CPU Q ... ";
    bkinfo->pq->GetCpuQueue()->WaitTillDone();
    cout <<" continue" << endl;
  }
}


/////////////////////////////////////////
// Temporary Functions needed to simulate region fitting

void BkgGpuPipeline::InitRegionalParamsAtFirstFlow()
{

  if(GpFP.getRealFnum() == startFlowNum){
    std::cout << "BkgGpuPipeline: Starting Flow: " << startFlowNum << std::endl;    
    Host->PerFlowRegionParams.memSet(0);
    for(size_t i=0; i < ImgP.getNumRegions(); i++)
    {
      WorkSet myJob(&bkinfo[i]);
      size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      if(myJob.DataAvailalbe()){
        //translate current reg params inot new layout
        TranslatePerFlowRegionParams_RegionToCube(Host->PerFlowRegionParams, &bkinfo[i], 0,  regId);
#if DEBUG_OUTPUT
        if(regId == DEBUG_REGION || DEBUG_REGION_ALL)
          cout << "BkgGpuPipeline: InitOldRegionalParamsAtFirstFlow: DEBUG regId " << regId << " PerFlowRegionParams,";
        Host->PerFlowRegionParams.getAtReg(regId).print();
#endif

      }
    }
    Dev->PerFlowRegionParams.copy(Host->PerFlowRegionParams); // can get moved here, see comment below
  }
}

/*
void BkgGpuPipeline::ReadRegionDataFromFileForBlockOf20()
{

  if (GpFP.getFlowIdx() == 0 ){ //only for first flow in block
    Host->RegionDump.setFilePathPrefix("RegionParams");
    Host->EmphasisVec.memSet(0);
    for(size_t i=0; i< ImgP.getNumRegions(); i++){
      WorkSet myJob(&bkinfo[i]);
      size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      if(myJob.DataAvailalbe()){
        //overwrite current bkg-model reg params with the ones read in from file
        TranslateRegionParams_CubeToRegion(Host->RegionDump.getFlowCube(myJob.getAbsoluteFlowNum()),myJob.getRegionParams(),regId);
#if DEBUG_OUTPUT
        if(i==0) cout << "BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating GPU emphasis and nucRise" << endl;
#endif
        myJob.setUpFineEmphasisVectors();
        if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
          myJob.setUpFineEmphasisVectorsForStdCompression();
        TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      }
    }
#if DEBUG_OUTPUT
    cout << "BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating Emphasis and NucRise on device for next block of " << flowBlockSize << " flows." << endl;
#endif
    Dev->EmphasisVec.copy(Host->EmphasisVec);
  }
}
 */


void BkgGpuPipeline::ExecuteRegionalFitting() {

  // Input needed
  // reg_params
  // bead params and bead state
  // Estimated amplitude
  // Emphasis
  // Nucrise...need to be done on the device for regional fitting
  // num of lev mar iterations
  // number of flows and starting flow
  // bead traces and shifted bkg traces
  // Nuc Id in case of multi flow regional fitting

#if INJECT_FG_TRACES_REG_FITTING
  static CubePerFlowDump<short> FGDump(ImgP ,ConstFrmP.getRawFrames(),1);

  FGDump.setFilePathPrefix("FgBufferDump");


  LayoutCubeWithRegions<short> HostSampleFromFG(Dev->SampleCompressedTraces,HostMem);
  LayoutCubeWithRegions<SampleCoordPair> HostSampleCoord(Dev->SampleCoord,HostMem);

  HostSampleCoord.setRWStrideX();
  for(size_t i=0; i< ImgP.getNumRegions(); i++){
    HostSampleCoord.setRWPtrRegion(i);
    for(size_t x=0; x< NUM_SAMPLES_RF; x++){
      SampleCoordPair tmp = HostSampleCoord.read();
      FGDump.getFlowCube(GpFP.getRealFnum()).setRWPtrRegion(i,tmp.x, tmp.y);
      FGDump.getFlowCube(GpFP.getRealFnum()).setRWStrideZ();
      HostSampleFromFG.setRWPtrRegion(i,x);
      HostSampleFromFG.setRWStrideZ();
      for (int frm=0; frm<ConstFrmP.getMaxCompFrames(); ++frm) {
        HostSampleFromFG.write(FGDump.getFlowCube(GpFP.getRealFnum()).read());
      }
    }
  }
  Dev->SampleCompressedTraces.copy(HostSampleFromFG);
#endif

  dim3 block(NUM_SAMPLES_RF);
  dim3 grid(ImgP.getNumRegions());

  size_t numFlows = 1;

  cout << "BkgGpuPipeline: ExecuteRegionalFitting: executing PerformMultiFlowRegionalFitting Kernel" << endl;
  PerformMultiFlowRegionalFitting<<<grid, block>>>(
      Dev->RegionStateMask.getPtr(),
      Dev->SampleCompressedTraces.getPtr(),
      Dev->SampleParamCube.getPtr(),
      Dev->SampleStateMask.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr(),
      Dev->NucRise.getPtr(),
      Dev->ResultCube.getPtr(),
      Dev->NumFrames.getPtr(), // move to constant per region
      Dev->ConstRegP.getPtr(),
      //Dev->NewPerFlowRegionParams.getPtr(),
      Dev->PerFlowRegionParams.getPtr(),
      Dev->PerNucRegP.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->EmptyTraceAvg.getPtr(),
      Dev->NumSamples.getPtr(),
      numFlows
  );

#if DEBUG_OUTPUT
  cudaDeviceSynchronize();
  cout << "BkgGpuPipeline: ExecuteRegionalFitting" << endl;
  CUDA_ERROR_CHECK();
#endif
}

void BkgGpuPipeline::PrepareForRegionalFitting()
{
  Host->EmphasisVec.memSet(0);
  LayoutCubeWithRegions<int>HostNonZeroEmphasisFrames(ImgP.getGridParam(MAX_POISSON_TABLE_COL),Nz_NUM_PARAMS, HostMem);
  for(size_t i=0; i< ImgP.getNumRegions(); i++){
    WorkSet myJob(&bkinfo[i]);
    size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
    if(myJob.DataAvailalbe()){
#if DEBUG_OUTPUT
      if(i==0) cout << "BkgGpuPipeline: PrepareForRegionalFitting: updating GPU crude emphasis" << endl;
#endif
      myJob.setUpCrudeEmphasisVectors();
      // TODO if still going ahead with recompression
      //if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
      //  myJob.setUpFineEmphasisVectorsForStdCompression();
      TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
    }
  }
  Dev->EmphasisVec.copy(Host->EmphasisVec);
  Dev->NonZeroEmphasisFrames.copy(HostNonZeroEmphasisFrames);
  //Dev->NewPerFlowRegionParams.copy(Dev->PerFlowRegionParams);
}

void BkgGpuPipeline::PrepareForSingleFlowFit()
{
  Host->EmphasisVec.memSet(0);
  LayoutCubeWithRegions<int>HostNonZeroEmphasisFrames(ImgP.getGridParam(MAX_POISSON_TABLE_COL),Nz_NUM_PARAMS, HostMem);
  for(size_t i=0; i< ImgP.getNumRegions(); i++){
    WorkSet myJob(&bkinfo[i]);
    size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
    if(myJob.DataAvailalbe()){
#if DEBUG_OUTPUT
      if(i==0) cout << "BkgGpuPipeline: PrepareForSingleFlowFit: updating GPU fine emphasis" << endl;
#endif
      myJob.setUpFineEmphasisVectors();
      if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
        myJob.setUpFineEmphasisVectorsForStdCompression();
      TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
    }
  }
  Dev->EmphasisVec.copy(Host->EmphasisVec);
  Dev->NonZeroEmphasisFrames.copy(HostNonZeroEmphasisFrames);
}

void BkgGpuPipeline::HandleRegionalFittingResults()
{
  Host->PerFlowRegionParams.copy(Dev->PerFlowRegionParams);
  for (size_t i=0; i<ImgP.getNumRegions(); ++i) {
    WorkSet myJob(&bkinfo[i]);
    size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
    if(myJob.DataAvailalbe()){
#if DEBUG_OUTPUT
      if(i==0) cout << "BkgGpuPipeline: HandleRegionalFittingResults: updating reg params on host" << endl;
#endif
      TranslatePerFlowRegionParams_CubeToRegion(Host->PerFlowRegionParams, &bkinfo[i], regId);
    }    
  }

  /*for (size_t i=0; i<ImgP.getNumRegions(); ++i) {
      WorkSet myJob(&bkinfo[i]);
      std::cout << "regCol:" << myJob.getRegCol() << ","; 
      std::cout << "regRow:" << myJob.getRegRow() << ","; 
      std::cout << "RegId:" << i << ",";
      std::cout << "tmidNuc:" << *(myJob.getRegionParams()->AccessTMidNuc()) << ","; 
      std::cout << "rdr:" << *(myJob.getRegionParams()->AccessRatioDrift()) << ","; 
      std::cout << "pdr:" << *(myJob.getRegionParams()->AccessCopyDrift()) << ","; 
      std::cout << std::endl;
  }*/


  //Dev->PerFlowRegionParams.copy(Dev->NewPerFlowRegionParams);
}

void BkgGpuPipeline::ExecuteCrudeEmphasisGeneration() {

  dim3 block(512);
  dim3 grid(ImgP.getNumRegions());

  cout << "BkgGpuPipeline: ExecuteCrudeEmphasisGeneration: executing emphasis generation Kernel" << endl;
  int smem = (MAX_POISSON_TABLE_COL)*ConstFrmP.getMaxCompFrames()*sizeof(float);
  GenerateEmphasis<<<grid, block, smem>>>(
      MAX_POISSON_TABLE_COL,
      CRUDEXEMPHASIS,
      Dev->PerFlowRegionParams.getPtr(),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->NumFrames.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr());

#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  cout << "BkgGpuPipeline: ExecuteCrudeEmphasisGeneration" << endl;
  CUDA_ERROR_CHECK();
#endif
}

void BkgGpuPipeline::ExecuteFineEmphasisGeneration() {

  dim3 block(512);

  dim3 grid(ImgP.getNumRegions());

  cout << "BkgGpuPipeline: ExecuteFineEmphasisGeneration: executing emphasis generation Kernel" << endl;
  int smem = (MAX_POISSON_TABLE_COL)*ConstFrmP.getMaxCompFrames()*sizeof(float);
  GenerateEmphasis<<<grid, block, smem>>>(
      MAX_POISSON_TABLE_COL,
      FINEXEMPHASIS,
      Dev->PerFlowRegionParams.getPtr(),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->NumFrames.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr());

#if DEBUG_OUTPUT
#if DEBUG_SYNC
  cudaDeviceSynchronize();
#endif
  cout << "BkgGpuPipeline: ExecuteFineEmphasisGeneration" << endl;
  CUDA_ERROR_CHECK();
#endif
}

void BkgGpuPipeline::ExecutePostFitSteps() {

  WorkSet myJob(&bkinfo[0]);
  if (!(myJob.performPostFitHandshake()))
    return;

  dim3 block(32,4);
  dim3 gridBlockPerRegion(ImgP.getGridDimX(),ImgP.getGridDimY());
  dim3 gridWarpPerRow(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);

  size_t smem = 0;

  if(ConfP.PerformWellsLevelXTalk()){
    smem = block.x * block.y *sizeof(float);

    cout << "BkgGpuPipeline: ExecutePostFitSteps: executing XTalk Update Signal Map Kernel" << endl;

    UpdateSignalMap_k<<<gridBlockPerRegion, block, smem>>>(
        Dev->RegionStateMask.getPtr(),
        Dev->BfMask.getPtr(),
        Dev->BeadParamCube.getPtr(),
        Dev->ResultCube.getPtr(),
        Dev->AverageSignalRegion.getPtr()
    );

#if DEBUG_OUTPUT
#if DEBUG_SYNC
    cudaDeviceSynchronize();
#endif
    cout << "BkgGpuPipeline: ExecutePostFitSteps UpdateSignalMap_k returned" << endl;
    CUDA_ERROR_CHECK();
#endif
  }

  if(ConfP.PerformWellsLevelXTalk() || ConfP.PerformPolyClonalFilter()){
    cout << "BkgGpuPipeline: ExecutePostFitSteps: executing ProtonXTalk and Polyclonal Update Kernel" << endl;
    cudaFuncSetCacheConfig(PostProcessingCorrections_k, cudaFuncCachePreferL1);

    PostProcessingCorrections_k<<<gridWarpPerRow, block, 0>>>(
        Dev->RegionStateMask.getPtr(),
        Dev->PerFlowRegionParams.getPtr(),
        Dev->PerNucRegP.getPtr(),
        Dev->BfMask.getPtr(),
        Dev->BeadParamCube.getPtr(),
        Dev->BeadStateMask.getPtr(),
        Dev->PolyClonalCube.getPtr(),
        Dev->ResultCube.getPtr(),
        Dev->AverageSignalRegion.getPtr()
    );

#if DEBUG_OUTPUT
#if DEBUG_SYNC
    cudaDeviceSynchronize();
#endif
    cout << "BkgGpuPipeline: ExecutePostFitSteps ProtonXTalk_k returned" << endl;
    CUDA_ERROR_CHECK();
#endif
  }
}


void BkgGpuPipeline::ApplyClonalFilter()
{
  //control opts
  //if clonal filter enabled
  if(ConfP.PerformPolyClonalFilter())
  {
    //if last clonal filter update complete, execute filter
    if( ConstGP.isApplyClonalFilterFlow( GpFP.getRealFnum() ) )
    {

      try{
        //copy back from device
        cout << "Applying PolyClonal Filter after Flow: " << GpFP.getRealFnum() << endl;
        //already copied in handle results
        //LayoutCubeWithRegions<unsigned short> HostBeadStateMaskTMP(Dev->BeadStateMask, HostMem );
        Host->BeadStateMask.copy(Dev->BeadStateMask);
        Host->BfMask.copy(Dev->BfMask);
        LayoutCubeWithRegions<float> HostPolyClonalCube(Dev->PolyClonalCube, HostMem);

        //ClonalFilterWrapper clonalFilter(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0],Host->BeadStateMask,HostPolyClonalCube);
        ClonalFilterWrapper clonalFilter(Host->BfMask.getPtr(),Host->BeadStateMask,HostPolyClonalCube);
        clonalFilter.DumpPPFSSQ(bkinfo->inception_state->sys_context.GetResultsFolder());
        clonalFilter.DumpPPFSSQtoH5(bkinfo->inception_state->sys_context.GetResultsFolder());
        clonalFilter.ApplyClonalFilter(bkinfo->inception_state->bkg_control.polyclonal_filter);
        clonalFilter.UpdateMask();
        //host bf mask updated so update original bfmask being written out
        Host->BfMask.copyPlanesOut(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0],0,1);

        Dev->BeadStateMask.copy(Host->BeadStateMask);
        Dev->BfMask.copy(Host->BfMask);

      }catch(exception& e){
        cerr << "NOTE: clonal filter failed."
            << e.what()
            << endl;
      }catch(...){
        cerr << "NOTE: clonal filter failed." << endl;
      }
    }
  }

}

void BkgGpuPipeline::getDataForRawWells()
{
  float *ampBuf = bkinfo->gpuAmpEstPerFlow->writeOneBuffer();
  // need copies and copydrift too for new 1.wells format
  Dev->ResultCube.copyPlanesOut(ampBuf,ResultAmpl,1);
  bkinfo->gpuAmpEstPerFlow->updateWritePos();
}

void BkgGpuPipeline::getDataForPostFitStepsOnHost()
{
  Host->ResultCube.copy(Dev->ResultCube);
  Host->BeadStateMask.copy(Dev->BeadStateMask);
  Host->PerFlowRegionParams.copy(Dev->PerFlowRegionParams);
}






