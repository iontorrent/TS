/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * BkgGpuPipeline.cu
 *
 *  Created on: Jul 7, 2014
 *      Author: jakob
 */


#include <iostream>
#include "LayoutTranslator.h"
#include "MasterKernel.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"
#include "GpuPipelineDefines.h"
#include "BkgGpuPipeline.h"
#include "ClonalFilterWrapper.h"


using namespace std;

#define DEBUG_SYNC 1  //forces sync after each kernel/async call to guarantee correct error catching
#define DEBUG_OUTPUT 0  //fully verbose

#define DEBUG_REGION 999999 // region to print for debugging, if DEBUG_REGION_ALL is set this value is ignored
#define DEBUG_REGION_ALL 0  // if set to 1 all regions are printed for debugging if 0 only the DEBUG_REGION is printed


#define RESULTS_CHECK 0
#define RESULT_DUMP 0  // 1 = dump 0 = compare
#define INIT_CONTROL_OUPUT 1

#define SAMPLE_CONTROL 0

//use dump for the following values
#define READ_EMPTY_FROM_FILE 0
#define READ_EMPHASIS_FROM_FILE 0
#define READ_FGBUFFER_FROM_FILE 0
#define COMPARE_FG_BUFFER_FROM_FILE 0
//////////////////////////////////////////////

//Todo:
//Remove all flowIdx references and any flow index related calls. we are only working with real flow num from now on and a flowidx of 0


BkgGpuPipeline::BkgGpuPipeline(
    BkgModelWorkInfo* pbkinfo, 
 //   int fbSize,
    int startingFlow,
    int deviceId,
    SampleCollection * smpCol)
{ 

  this->bkinfo = pbkinfo;

  startFlowNum = startingFlow; 
 // flowBlockSize = fbSize;
  SpDev = NULL;
  HostTLXTalkData=NULL;
  DevTLXTalkData=NULL;
  pTLXTalkConstP=NULL;
  pConstXTP = NULL;
  Dev=NULL;
  Host=NULL;
  pSmplCol = smpCol;

  devId = deviceId;

  cudaSetDevice( devId );

  cudaDeviceProp cuda_props;
  cudaGetDeviceProperties( &cuda_props, devId );

  cout << "CUDA: BkgGpuPipeline: Initiating Flow by Flow Pipeline on Device: "<< devId << "( " << cuda_props.name  << " v"<< cuda_props.major <<"."<< cuda_props.minor << ")" << endl;

  setSpatialParams();
  InitPipeline();

}


void BkgGpuPipeline::setSpatialParams()
{

  const RawImage * rpt = bkinfo->img->GetImage();
  const SpatialContext * loc = &bkinfo[0].inception_state->loc_context;
  cout << "CUDA: Chip offset x:" << loc->chip_offset_x   << " y:" <<  loc->chip_offset_y  << endl;

  ImgP.init(rpt->cols, rpt->rows, loc->regionXSize, loc->regionYSize);
  ImgP.print();

}


void BkgGpuPipeline::InitPipeline()
{
  //check memory and set context/device
    checkAvailableDevMem();

    //Todo: Mulit-Device support
    CreatePoissonApproxOnDevice(devId);


    ConstanSymbolCopier::PopulateSymbolConstantImgageParams(ImgP, ConstFrmP, bkinfo);
    ConstanSymbolCopier::PopulateSymbolConstantGlobal(ConstGP,bkinfo);
    ConstanSymbolCopier::PopulateSymbolConfigParams(ConfP,bkinfo);
    ConstanSymbolCopier::PopulateSymbolPerFlowGlobal(GpFP, bkinfo);
    copySymbolsToDevice(ImgP);


    //ToDo: exception handling for Unsuccessful allocate
    try{
      Dev = new DeviceData(ImgP,ConstFrmP);
    }catch(cudaException &e){
      e.Print();
      throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
    }

    if(Host == NULL)
      Host = new HostData(ImgP,ConstFrmP);



    PrepareSampleCollection();

    PrepareInputsForSetupKernel();
    ExecuteT0AvgNumLBeadKernel();
    InitPersistentData();
    InitXTalk();

}



bool BkgGpuPipeline::firstFlow(){
  return(GpFP.getRealFnum() == startFlowNum) ;
  //return startFlowNum;
}

size_t BkgGpuPipeline::checkAvailableDevMem()
{
  size_t free_byte ;
  size_t total_byte ;
  double divMB = 1024.0*1024.0;
  cudaMemGetInfo( &free_byte, &total_byte );
  cout << "CUDA " << devId << ": GPU memory usage: used = " << (total_byte-free_byte)/divMB<< ", free = " << free_byte/divMB<< " MB, total = "<< total_byte/divMB<<" MB" << endl;
  return free_byte;
}


void BkgGpuPipeline::PrepareInputsForSetupKernel()
{
  Host->BfMask.wrappPtr(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
  Dev->BfMask.copy(Host->BfMask);

  Dev->T0.copyIn( &(*(bkinfo->smooth_t0_est))[0]);

  Host->BeadStateMask.memSet(0);
  for(size_t i=0; i< ImgP.getNumRegions(); i++){
    WorkSet myJob(&bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      TranslatorsFlowByFlow::TranslateBeadStateMask_RegionToCube(Host->BeadStateMask, &bkinfo[i],regId);
    }
  }
  Dev->BeadStateMask.copy(Host->BeadStateMask);

}

void BkgGpuPipeline::PrepareSampleCollection()
{
  if(pSmplCol == NULL)
    pSmplCol = new SampleCollection(ImgP,10,ConstFrmP.getMaxCompFrames());

  pSmplCol->InitDeviceBuffersAndSymbol(ConstFrmP.getMaxCompFrames());

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
      TranslatorsFlowByFlow::TranslateConstantRegionParams_RegionToCube(Host->ConstRegP,&bkinfo[i],regId);
      TranslatorsFlowByFlow::TranslateRegionFrameCube_RegionToCube(HostRegionFrameCube, &bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslateRegionFramesPerPoint_RegionToCube(HostRegionFramesPerPoint,&bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslateBeadParams_RegionToCube(HostBeadParamCube, &bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslatePolyClonal_RegionToCube(HostPolyClonalCube,&bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslatePerNucRegionParams_RegionToCube(HostPerNucRegP,&bkinfo[i],regId);

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
  cout << "CUDA: BkgGpuPipeline: InitPersistentData: num Time-Compressed-Frames Per Region:" << endl;
  Host->NumFrames.printRegionTable<size_t>();
#endif

}


void BkgGpuPipeline::InitXTalk(){

  //Oh you beautiful XTalk stuff... (cannot be done before persistent data init since it needs old pipeline buffers to be initialized.

  if(ConfP.PerformTraceLevelXTalk()){
    const TraceCrossTalkSpecification & tXTPec = bkinfo->bkgObj->getTraceXTalkSpecs();
    pTLXTalkConstP = new XTalkNeighbourStatsHost(
        tXTPec.cx,
        tXTPec.cy,
        tXTPec.multiplier
    );
    pTLXTalkConstP->setHexPacked(tXTPec.hex_packed);
    pTLXTalkConstP->setInitialPhase(tXTPec.initial_phase);
    pTLXTalkConstP->setThreeSeries(tXTPec.three_series);
    copySymbolsToDevice(*pTLXTalkConstP);
    pTLXTalkConstP->print();
    HostTLXTalkData= new HostTracelevelXTalkData(ImgP,ConstFrmP);
    DevTLXTalkData= new DeviceTracelevelXTalkData(ImgP,ConstFrmP);
    HostTLXTalkData->createSampleMask();
    DevTLXTalkData->TestingGenericXTalkSampleMask.copy(HostTLXTalkData->TestingGenericXTalkSampleMask);
  }

  if(ConfP.PerformWellsLevelXTalk()){
    const WellXtalk & wellXT = bkinfo->bkgObj->getWellXTalk();
    const int xtalkSpanX = wellXT.nn_span_x;
    const int xtalkSpanY = wellXT.nn_span_y;
    const float * evenphasemap = &wellXT.nn_even_phase_map[0];
    const float * oddphasemap = &wellXT.nn_odd_phase_map[0];
    pConstXTP = new WellsLevelXTalkParamsHost(
        oddphasemap,
        evenphasemap,
        xtalkSpanX,
        xtalkSpanY
    );
    copySymbolsToDevice(*pConstXTP);
    pConstXTP->print();
  }
}




BkgGpuPipeline::~BkgGpuPipeline()
{
  cout << "CUDA: Starting cleanup flow by flow GPU pipeline" << endl;
  checkAvailableDevMem();
  if(SpDev != NULL) delete SpDev;
  if(Dev != NULL) delete Dev;
  if(Host != NULL) delete Host;
  if(pConstXTP != NULL) delete pConstXTP;
  if(pTLXTalkConstP != NULL) delete pTLXTalkConstP;
  cout << "CUDA: Cleanup flow by flow GPU pipeline completed" << endl;
  checkAvailableDevMem();
}




void BkgGpuPipeline::PerFlowDataUpdate(BkgModelWorkInfo* pbkinfo)
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
          Obj->region_data_extras.my_flow->flow_ndx_map[0],
          0);
    }
  }

  //updated per flow device Symbols and buffers
  ConstanSymbolCopier::PopulateSymbolPerFlowGlobal(GpFP, bkinfo);


  if(!firstFlow()){ //  copied in constructor for first flow
    //Host->BfMask.copyIn(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);  // debugging only
    Host->BfMask.wrappPtr(&bkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
    Dev->BfMask.copy(Host->BfMask);
  }

  //raw image
  Host->RawTraces.wrappPtr(bkinfo->img->raw->image);
  Dev->RawTraces.copy(Host->RawTraces);

}


dim3 BkgGpuPipeline::matchThreadBlocksToRegionSize(int bx, int by)
{
  int rH = ImgP.getRegH();
  int correctBy = by;
  while(rH%correctBy != 0) --correctBy;

  if(correctBy!=by)
    cout << "CUDA WARNING: requested region height of " << ImgP.getRegH() << " does not allow optimal GPU threadblock height of 4 warps! Threadblock height corrected to " << correctBy << ". For optimal performance please choose a region height of a multiple of " << by << "." << endl;
  dim3 block(bx,correctBy);
  return block;
}

/////////////////////////////////////////////////////////////////////////////////
// Kernels


void BkgGpuPipeline::ExecuteT0AvgNumLBeadKernel()
{

  //one block per region execution model, no y-dim check needed
  dim3 block(32, 4);
  dim3 grid(ImgP.getGridDimX(),ImgP.getGridDimY());
  size_t smem = 2*(block.x * block.y *sizeof(int));

  Dev->SampleRowPtr.memSet(0);
  Dev->NumSamples.memSet(0);

  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: executing GenerateT0AvgAndNumLBeads Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
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

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: GenerateT0AvgAndNumLBeads_New finalize" << endl;
#endif

#if INIT_CONTROL_OUPUT || DEBUG_OUTPUT
  LayoutCubeWithRegions<float>HostT0Avg(Dev->T0Avg,HostMem);
  LayoutCubeWithRegions<int>HostNumLBeads(Dev->NumLBeads, HostMem);
  LayoutCubeWithRegions<float>HostT0(Dev->T0, HostMem);

  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: num live beads per region: " << endl;
  HostNumLBeads.printRegionTable<int>();
  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: T0 avg per region: " << endl;
  HostT0Avg.printRegionTable<float>();
  //LayoutCubeWithRegions<float> RegionStdDev(ImgP.getGridDimX(), ImgP.getGridDimY());
  //for(size_t i=0; i<ImgP.getNumRegions(); i++){
  //RegionStdDev[i] = HostT0.getStdDevReg<float>(i,0,HostT0Avg[i],0,&Host->BfMask,(unsigned short)MaskLive);
  //}
  //cout << "CUDA: BkgGpuPipeline: BkgGpuPipeline: std deviation per region: "<< endl;
  //RegionStdDev.printRegionTable<float>();
#endif
#if INIT_CONTROL_OUPUT || DEBUG_OUTPUT || SAMPLE_CONTROL
  LayoutCubeWithRegions<int>HostSampleCount(Dev->SampleRowPtr, HostMem);
  LayoutCubeWithRegions<int>HostNumSamples(Dev->NumSamples, HostMem);
  cout << "CUDA: Number of samples for regional fitting per Region:" << endl;
  HostNumSamples.printRegionTable<int>();
#endif
#if SAMPLE_CONTROL || DEBUG_OUTPUT
  HostSampleCount.setRWStrideX();
  cout << "CUDA: starting offset for samples per Row (last entry is num samples)" << endl;
  for(size_t rg = 0; rg < ImgP.getNumRegions(); rg++){
    if(rg == DEBUG_REGION || DEBUG_REGION_ALL)
      cout << "regId " << rg << "," << HostSampleCount.getCSVatReg<int>(rg,0,0,0,ImgP.getRegH()) << endl;
  }
#endif


}

void BkgGpuPipeline::ExecuteGenerateBeadTrace()
{


  dim3 block = matchThreadBlocksToRegionSize(32,4);
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

  //Dev->SampleCompressedTraces.memSet(0);
  pSmplCol->RezeroWriteBuffer();

  Dev->SampleParamCube.memSet(0);
  Dev->SampleStateMask.memSet(0);

  size_t smem = block.x * block.y *sizeof(int);

  switch(cacheSetting){
    case 0:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferEqual);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferEqual" << endl;
#endif
      break;
    case 2:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferL1);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferL1" << endl;
#endif
      break;
    case 1:
    default:
      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferShared);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferShared" << endl;
#endif

  }


  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: executing GenerateAllBeadTraceFromMeta_k Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
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
      NULL,//Dev->SampleCompressedTraces.getPtr(),
      Dev->SampleParamCube.getPtr(),
      Dev->SampleCoord.getPtr()


  );


#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: GenerateAllBeadTraceEmptyFromMeta_k finalize" << endl;
#endif

  pSmplCol->UpdateSampleCollection(GpFP.getRealFnum());

  //one block per region execution model, no y-dim check needed
  dim3 blockER(32,1);
  dim3 gridER(ImgP.getGridDimX(), ImgP.getGridDimY());

  smem = (blockER.y * ConstFrmP.getUncompFrames()  + blockER.y )* sizeof(float);


  //LayoutCubeWithRegions<float> DevDcOffsetDebug(ImgP.getGridParam(),1,DeviceGlobal);


  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: executing ReduceEmptyAverage_k Kernel grid(" << gridER.x << "," << gridER.y  << "), block(" << blockER.x << "," << blockER.y <<"), smem("<< smem <<")" << endl;
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

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: ReduceEmptyAverage_k finalize" << endl;
#endif
  /*
  LayoutCubeWithRegions<float> HostDcOffset(DevDcOffsetDebug,HostMem);
  HostDcOffset.setRWStrideX();
  HostDcOffset.setRWPtr(0);
  cout << GpFP.getRealFnum() <<"," << GpFP.getNucId();
  for(size_t i=0; i< HostDcOffset.numElements(); i++)
    cout << "," << HostDcOffset.read();
  cout << endl;
   */



  if(GpFP.getRealFnum() == 20){
    cout << "CUDA: Region State: " << endl;
    LayoutCubeWithRegions<unsigned short>HostRegionMask(Dev->RegionStateMask,HostMem);
    HostRegionMask.printRegionTable<unsigned short>();
#if DEBUG_OUTPUT
    printRegionStateMask();
#endif
  }




#if DEBUG_OUTPUT
  //static LayoutCubeWithRegions<int>HostEmptyTraceComplete(ImgP.getGridParam(),1,HostMem);HostEmptyTraceComplete.trackMe(muT);
  static LayoutCubeWithRegions<float>HostEmptyTraceAvg(Dev->EmptyTraceAvg,HostMem);
  HostEmptyTraceAvg.copy(Dev->EmptyTraceAvg);
  HostEmptyTraceAvg.setRWStrideX();

  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: Average Empty Traces:" << endl;
  for(size_t regId = 0; regId < ImgP.getNumRegions(); regId++){
    if(regId == DEBUG_REGION || DEBUG_REGION_ALL ){
      int nf = Host->NumFrames.getAtReg(regId);
      if(nf <= 0 || nf > ConstFrmP.getMaxCompFrames())
        nf = ConstFrmP.getMaxCompFrames();
      cout <<"BkgGpuPipeline: ExecuteGenerateBeadTrace: DEBUG GPU " << regId <<"," << nf << "," << HostEmptyTraceAvg.getCSVatReg<float>(regId,0,0,0,nf) << endl;

    }
  }

#endif

#if SAMPLE_CONTROL
  LayoutCubeWithRegions<int> HostNumSample(Dev->NumSamples,HostMem);
  LayoutCubeWithRegions<short> * SampleHostCompressedTraces = pSmplCol->getLatestSampleFromDevice();

  LayoutCubeWithRegions<SampleCoordPair> HostSamplesCoords(Dev->SampleCoord, HostMem);
  SampleHostCompressedTraces SampleHostCompressedTraces->setRWStrideZ();
  HostSamplesCoords.setRWStrideX();
  for(size_t rgid =0 ; rgid < ImgP.getNumRegions(); rgid++){
    if( rgid == DEBUG_REGION || DEBUG_REGION_ALL){
      HostSamplesCoords.setRWPtrRegion(rgid);
      for(int i = 0; i < HostNumSample.getAtReg(rgid); i++){
        int nf = Host->NumFrames.getAtReg(rgid);
        if(nf <= 0 || nf > ConstFrmP.getMaxCompFrames())
          nf = ConstFrmP.getMaxCompFrames();
        SampleCoordPair loc = HostSamplesCoords.read();
        cout << "regId," << rgid <<",x,"<< loc.x << ",y,"<< loc.y << "," << SampleHostCompressedTraces->getCSVatReg<short>(rgid,i,0,0,nf) << endl;;
      }
    }
  }
#endif

}


void BkgGpuPipeline::ExecuteTraceLevelXTalk()
{

  if(ConfP.PerformTraceLevelXTalk()){

    DevTLXTalkData->BaseXTalkContribution.memSet(0);
    DevTLXTalkData->xTalkContribution.memSet(0);
    DevTLXTalkData->genericXTalkTracesRegion.memSet(0);
    DevTLXTalkData->numGenericXTalkTracesRegion.memSet(0);



    dim3 block = matchThreadBlocksToRegionSize(32,4);
    dim3 grid(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);
    size_t smem = 0;
    //smem needed: one float per thread
    //one trace with ConstFrmP.getMaxCompFrames() frames of type float per warp
    //num warps == block.y


    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing SimpleXTalkNeighbourContribution grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
    SimpleXTalkNeighbourContribution<<<grid, block, smem >>>(// Here FL stands for flows
        Dev->RegionStateMask.getPtr(),
        Dev->BfMask.getPtr(),
        Dev->BeadStateMask.getPtr(),
        DevTLXTalkData->BaseXTalkContribution.getPtr(),
        Dev->RawTraces.getPtr(),
        Dev->EmptyTraceAvg.getPtr(), //FxR
        Dev->BeadParamCube.getPtr(), //NxP
        Dev->RegionFrameCube.getPtr(), //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
        Dev->ConstRegP.getPtr(), // R
        Dev->PerFlowRegionParams.getPtr(), // R
        Dev->PerNucRegP.getPtr(), //RxNuc
        Dev->NumFrames.getPtr() // R
    );

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: SimpleXTalkNeighbourContribution finalize" << endl;
#endif
    /*
    if (GpFP.getRealFnum() == 20 ){
        cout << "CUDA: Per Bead XTalk Contribution " <<endl;
        LayoutCubeWithRegions<float>HostBeadXtalkContri(DevTLXTalkData->BaseXTalkContribution, HostMem);
        HostBeadXtalkContri.setRWStrideZ();
        Host->BfMask.copy(Dev->BfMask);
        for(size_t idx=0; idx < ImgP.getImgSize(); idx++ ){
          if(Host->BfMask[idx] & (unsigned short)MaskLive){
            size_t x = ImgP.getXFromIdx(idx) ;
            size_t y = ImgP.getYFromIdx(idx) ;

            cout << x << ", " << y << ", ";
            //float sumF = 0;
            for(size_t f = 0; f < ConstFrmP.getMaxCompFrames(); f++)
            {
              cout << HostBeadXtalkContri.getAt(x,y,f) << ", ";
              //sumF += HostBeadXtalk.getAt(x,y,f);
            }
            //cout << sumF << endl;
            cout << endl;
              //HostBeadXtalk.getCSVatReg<float>(0,x,y,0,ConstFrmP.getMaxCompFrames()) << endl;
          }
        }
      }
     */

    smem = ( (block.x*block.y) +  (block.y * ConstFrmP.getMaxCompFrames()) ) * sizeof(float) ;

    //allocate or rezero
    int threadBlocksPerRegion = (ImgP.getRegH()+block.y-1)/block.y;
    DevTLXTalkData->allocateRezeroDynamicBuffer(ImgP,ConstFrmP,threadBlocksPerRegion);


    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing GenericXTalkAndNeighbourAccumulation grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")"<< endl;
    GenericXTalkAndNeighbourAccumulation<<<grid, block, smem >>>(// Here FL stands for flows
        Dev->RegionStateMask.getPtr(),
        Dev->BfMask.getPtr(),
        Dev->BeadStateMask.getPtr(),
        DevTLXTalkData->BaseXTalkContribution.getPtr(),
        DevTLXTalkData->xTalkContribution.getPtr(),  // buffer XTalk contribution to this well NxF
        DevTLXTalkData->pDyncmaicPerBLockGenericXTalk->getPtr(), // one trace of max compressed frames per thread block
        DevTLXTalkData->numGenericXTalkTracesRegion.getPtr(), //one int per region to average after accumulation
        Dev->PerFlowRegionParams.getPtr(), // R
        Dev->NumFrames.getPtr(), // R
        DevTLXTalkData->TestingGenericXTalkSampleMask.getPtr()  //ToDo: remove when testing done
    );

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: GenericXTalkAndNeighbourAccumulation finalize" << endl;
#endif

    dim3 accumBlock(128,1);
    dim3 accumGrid(ImgP.getGridDimX(),ImgP.getGridDimY());

    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing GenericXTalkAccumulation grid(" << accumGrid.x << "," << accumGrid.y  << "), block(" << accumBlock.x << "," << accumBlock.y <<"), smem(0)" << endl;
    GenericXTalkAccumulation<<<accumGrid,accumBlock>>>(// Here FL stands for flows
        DevTLXTalkData->genericXTalkTracesRegion.getPtr(), // one trace of max compressed frames per region
        DevTLXTalkData->pDyncmaicPerBLockGenericXTalk->getPtr(), // one trace of max compressed frames per thread block
        DevTLXTalkData->numGenericXTalkTracesRegion.getPtr(), //one int per region to average after accumulation
        Dev->NumFrames.getPtr(), // R
        threadBlocksPerRegion
    );

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: GenericXTalkAccumulation finalize" << endl;
#endif
    /*

    smem = ( (block.x*block.y) +  (block.y * ConstFrmP.getMaxCompFrames()) ) * sizeof(float) ;
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing SimpleXTalkNeighbourContributionAndAccumulation with: block(" << block.x << "," << block.y <<"), grid(" << grid.x << "," << grid.y  << ")  and smem: "<< smem << endl;
#endif


    SimpleXTalkNeighbourContributionAndAccumulation_LocalMem<<<grid, block, smem >>>(
      Dev->RegionStateMask.getPtr(),
      Dev->BfMask.getPtr(),
      Dev->BeadStateMask.getPtr(),
      DevTLXTalkData->xTalkContribution.getPtr(),  // buffer XTalk contribution to this well NxF
      DevTLXTalkData->genericXTalkTracesRegion.getPtr(), // one trace of max compressed frames per thread block or per region (atomicAdd)
      DevTLXTalkData->numGenericXTalkTracesRegion.getPtr(), //one int per region to average after accumulation
      Dev->RawTraces.getPtr(),
      Dev->EmptyTraceAvg.getPtr(), //FxR
      Dev->BeadParamCube.getPtr(), //NxP
      Dev->RegionFrameCube.getPtr(), //FxRxT bkgTrace, DarkMatter, DeltaFrames, DeltaFramesStd, FrameNumber
      Dev->ConstRegP.getPtr(), // R
      Dev->PerFlowRegionParams.getPtr(), // R
      Dev->PerNucRegP.getPtr(), //RxNuc
      Dev->NumFrames.getPtr(), // R
      DevTLXTalkData->TestingGenericXTalkSampleMask.getPtr()  //ToDo: remove when testing done
      );


     */
    /*

    HostTLXTalkData->genericXTalkTracesRegion.copy(DevTLXTalkData->genericXTalkTracesRegion);
    HostTLXTalkData->numGenericXTalkTracesRegion.copy(DevTLXTalkData->numGenericXTalkTracesRegion);
    HostTLXTalkData->xTalkContribution.copy(DevTLXTalkData->xTalkContribution);


    cout << "CUDA: num GenericXTalkTraces per Region: " << endl;
    HostTLXTalkData->numGenericXTalkTracesRegion.printRegionTable<int>();

    LayoutCubeWithRegions<unsigned short>HostRegState(Dev->RegionStateMask, HostMem);
    for(size_t regId=0; regId < ImgP.getNumRegions(); regId++ ){
      if(HostRegState[regId] == RegionMaskLive){
        HostTLXTalkData->genericXTalkTracesRegion.setRWStrideX();
        cout << "regId " << regId << "," << HostTLXTalkData->genericXTalkTracesRegion.getCSVatReg<float>(regId,0,0,0,ConstFrmP.getMaxCompFrames()) << endl;
      }
    }
     */

    /*
  if (GpFP.getRealFnum() == 39 ){
    cout << "CUDA: Per Bead XTalk " <<endl;
    LayoutCubeWithRegions<float>HostBeadXtalk(DevTLXTalkData->xTalkContribution, HostMem);
    HostBeadXtalk.setRWStrideZ();
    Host->BfMask.copy(Dev->BfMask);
    for(size_t idx=0; idx < ImgP.getImgW()*4; idx++ ){
      if(Host->BfMask[idx] & (unsigned short)MaskLive){
        size_t x = ImgP.getXFromIdx(idx) ;
        size_t y = ImgP.getYFromIdx(idx) ;

        cout << x << ", " << y << ", ";
        //float sumF = 0;
        for(size_t f = 0; f < ConstFrmP.getMaxCompFrames(); f++)
        {
          cout << HostBeadXtalk.getAt(x,y,f) << ", ";
          //sumF += HostBeadXtalk.getAt(x,y,f);
        }
        //cout << sumF << endl;
        cout << endl;
          //HostBeadXtalk.getCSVatReg<float>(0,x,y,0,ConstFrmP.getMaxCompFrames()) << endl;
      }
    }
  }
     */
  }

}


void BkgGpuPipeline::ExecuteSingleFlowFit()
{

  int cacheSetting = 0;

  dim3 block=matchThreadBlocksToRegionSize(32,4);

  dim3 grid(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);

#if EMPTY_IN_SHARED
  size_t smem = (MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames() + ConstFrmP.getUncompFrames())  * sizeof(float);
#else
  size_t smem = (MAX_POISSON_TABLE_COL * ConstFrmP.getMaxCompFrames())  * sizeof(float);
#endif

  switch(cacheSetting){
    case 0:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferEqual);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferEqual" << endl;
#endif
      break;
    case 2:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferL1);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferL1" << endl;
#endif
      break;
    case 1:
    default:
      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferShared);
#if DEBUG_OUTPUT
      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferShared" << endl;
#endif

  }

  Dev->ResultCube.memSet(0);
  //  Dev->RawTraces.memSet(0,(ConstFrmP.getRawFrames()-3)*ImgP.getImgSize()*sizeof(short),ImgP.getImgSize()*sizeof(short)*3);
  //   Dev->RawTraces.memSetPlane(0,ConstFrmP.getRawFrames()-3, 3);
  //TMemSegPairAlloc<int> numLBeads(sizeof(int)*36,HostPageLocked,DeviceGlobal);
  //numLBeads.memSet(0);

  // Do we need this here ?
  Dev->ResultCube.memSetPlane(0,ResultAmpl);
  //  Dev->ResultCube.memSetPlane(0,ResultAmplXTalk);

  cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: executing ExecuteThreadBlockPerRegion2DBlocks Kernel (SingleFlowFit) grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
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
      Dev->EmptyTraceAvg.getPtr(),
      (DevTLXTalkData)?(DevTLXTalkData->xTalkContribution.getPtr()):(NULL),  // buffer XTalk contribution to this well NxF
          (DevTLXTalkData)?(DevTLXTalkData->genericXTalkTracesRegion.getPtr()):(NULL) // one trace of max compressed frames per thread block or per region (atomicAdd)
  );

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: ExecuteThreadBlockPerRegion2DBlocksDense finalize" << endl;
#endif

  /*
  LayoutCubeWithRegions<short>HostIterCount(ImgP,2,HostMem);
 HostIterCount.copyPlanes(Dev->RawTraces,ConstFrmP.getRawFrames()-2,0,2);

  HostIterCount.setRWStrideX();
  size_t iterCounter[9] = {0};
  cout << "CUDA: Iteration counter: " << endl;
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
   cout << "CUDA: Region: " << reg << " numWarps: " << numWarpsPerRegRow*ImgP.getRegH(reg) <<  " max iter: ";
   for(int c=0; c<9; c++){
     cout << iterCounterReg[c] << ",";
   }
   cout << endl;
 }
 cout << "CUDA: Max iterations within all " <<  numwarps << " warps: ";
 for(int c=0; c<9; c++){
     cout << iterCounterAll[c] << ",";
     iterCounterAll[c] = 0;
  }
  cout << endl;

   */

}



void BkgGpuPipeline::HandleResults(RingBuffer<float> * ringbuffer)
{
  WorkSet myJob(&bkinfo[0]);
  //if (myJob.performPostFitHandshake()) {
  getDataForRawWells(ringbuffer);
  ApplyClonalFilter();
  /*}
  else {

    getDataForPostFitStepsOnHost();

    // No transaltion is required if background thread writing to raw wells.
    // It can take care of translation if required
    for(size_t i=0; i< ImgP.getNumRegions(); i++){

      WorkSet myJob(&bkinfo[i]);
      size_t regId = ImgP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      if(myJob.DataAvailalbe()){
        TranslatorsFlowByFlow::TranslateResults_CubeToRegion(Host->ResultCube,&bkinfo[i],GpFP.getFlowIdx(),regId);
        TranslatorsFlowByFlow::TranslateBeadStateMask_CubeToRegion(Host->BeadStateMask,&bkinfo[i],regId);
        TranslatorsFlowByFlow::TranslatePerFlowRegionParams_CubeToRegion(Host->PerFlowRegionParams, &bkinfo[i], regId);
      }

      myJob.setJobToPostFitStep();
      WorkerInfoQueueItem item;
      item.private_data = (void*)&bkinfo[i];
      myJob.putJobToCPU(item);
    }

    cout << "CUDA: BkgGpuPipeline: Reinjecting results for flowblock containing flows "<< getFlowP().getRealFnum() - flowBlockSize << " to " << getFlowP().getRealFnum() << endl;
    cout << "CUDA: waiting on CPU Q ... ";
    bkinfo->pq->GetCpuQueue()->WaitTillDone();
    cout <<" continue" << endl;
  }*/
}


/////////////////////////////////////////
// Temporary Functions needed to simulate region fitting

void BkgGpuPipeline::InitRegionalParamsAtFirstFlow()
{

  if(GpFP.getRealFnum() == startFlowNum){
    std::cout << "CUDA: BkgGpuPipeline: Starting Flow: " << startFlowNum << std::endl;
    Host->PerFlowRegionParams.memSet(0);
    for(size_t i=0; i < ImgP.getNumRegions(); i++)
    {
      WorkSet myJob(&bkinfo[i]);
      size_t regId = ImgP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      if(myJob.DataAvailalbe()){
        //translate current reg params inot new layout
        TranslatorsFlowByFlow::TranslatePerFlowRegionParams_RegionToCube(Host->PerFlowRegionParams, &bkinfo[i], 0,  regId);
#if DEBUG_OUTPUT
        if(regId == DEBUG_REGION || DEBUG_REGION_ALL)
          cout << "CUDA: BkgGpuPipeline: InitOldRegionalParamsAtFirstFlow: DEBUG regId " << regId << " PerFlowRegionParams,";
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
        TranslatorsFlowByFlow::TranslateRegionParams_CubeToRegion(Host->RegionDump.getFlowCube(myJob.getAbsoluteFlowNum()),myJob.getRegionParams(),regId);
#if DEBUG_OUTPUT
        if(i==0) cout << "CUDA: BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating GPU emphasis and nucRise" << endl;
#endif
        myJob.setUpFineEmphasisVectors();
        if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
          myJob.setUpFineEmphasisVectorsForStdCompression();
        TranslatorsFlowByFlow::TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      }
    }
#if DEBUG_OUTPUT
    cout << "CUDA: BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating Emphasis and NucRise on device for next block of " << flowBlockSize << " flows." << endl;
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

  dim3 block(NUM_SAMPLES_RF);
  dim3 grid(ImgP.getNumRegions());

  size_t numFlows = 1;

  cout << "CUDA: BkgGpuPipeline: ExecuteRegionalFitting: executing PerformMultiFlowRegionalFitting Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem(0)" << endl;
  PerformMultiFlowRegionalFitting<<<grid, block>>>(
      Dev->RegionStateMask.getPtr(),
      NULL, //Dev->SampleCompressedTraces.getPtr(),
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
#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteRegionalFitting: PerformMultiFlowRegionalFitting finalized" << endl;
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
      if(i==0) cout << "CUDA: BkgGpuPipeline: PrepareForRegionalFitting: updating GPU crude emphasis" << endl;
#endif
      myJob.setUpCrudeEmphasisVectors();
      // TODO if still going ahead with recompression
      //if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
      //  myJob.setUpFineEmphasisVectorsForStdCompression();
      TranslatorsFlowByFlow::TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
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
      if(i==0) cout << "CUDA: BkgGpuPipeline: PrepareForSingleFlowFit: updating GPU fine emphasis" << endl;
#endif
      myJob.setUpFineEmphasisVectors();
      if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())
        myJob.setUpFineEmphasisVectorsForStdCompression();
      TranslatorsFlowByFlow::TranslateEmphasis_RegionToCube(Host->EmphasisVec, &bkinfo[i], regId);
      TranslatorsFlowByFlow::TranslateNonZeroEmphasisFrames_RegionToCube(HostNonZeroEmphasisFrames, &bkinfo[i], regId);
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
      if(i==0) cout << "CUDA: BkgGpuPipeline: HandleRegionalFittingResults: updating reg params on host" << endl;
#endif
      TranslatorsFlowByFlow::TranslatePerFlowRegionParams_CubeToRegion(Host->PerFlowRegionParams, &bkinfo[i], regId);
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
  int smem = (MAX_POISSON_TABLE_COL)*ConstFrmP.getMaxCompFrames()*sizeof(float);

  cout << "CUDA: BkgGpuPipeline: ExecuteCrudeEmphasisGeneration: executing emphasis generation Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
  GenerateEmphasis<<<grid, block, smem>>>(
      Dev->RegionStateMask.getPtr(),
      MAX_POISSON_TABLE_COL,
      CRUDEXEMPHASIS,
      Dev->PerFlowRegionParams.getPtr(),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->NumFrames.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr());
#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteCrudeEmphasisGeneration: GenerateEmphasis finalized" << endl;
#endif
}

void BkgGpuPipeline::ExecuteFineEmphasisGeneration() {

  dim3 block(512);

  dim3 grid(ImgP.getNumRegions());
  int smem = (MAX_POISSON_TABLE_COL)*ConstFrmP.getMaxCompFrames()*sizeof(float);
  cout << "CUDA: BkgGpuPipeline: ExecuteFineEmphasisGeneration: executing emphasis generation Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
  GenerateEmphasis<<<grid, block, smem>>>(
      Dev->RegionStateMask.getPtr(),
      MAX_POISSON_TABLE_COL,
      FINEXEMPHASIS,
      Dev->PerFlowRegionParams.getPtr(),
      Dev->RegionFramesPerPoint.getPtr(),
      Dev->RegionFrameCube.getPtr(),
      Dev->NumFrames.getPtr(),
      Dev->EmphasisVec.getPtr(),
      Dev->NonZeroEmphasisFrames.getPtr());
#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
  cout << "CUDA: BkgGpuPipeline: ExecuteFineEmphasisGeneration: GenerateEmphasis finalized" << endl;
#endif
}

void BkgGpuPipeline::ExecutePostFitSteps() {

  WorkSet myJob(&bkinfo[0]);
  //if (!(myJob.performPostFitHandshake()))
  //  return;

  dim3 block(32,4);
  dim3 gridBlockPerRegion(ImgP.getGridDimX(),ImgP.getGridDimY());
  dim3 gridWarpPerRow(ImgP.getGridDimX(),(ImgP.getImgH()+block.y-1)/block.y);

  size_t smem = 0;

  if(ConfP.PerformWellsLevelXTalk()){
    smem = block.x * block.y *sizeof(float);

    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: executing Wells XTalk Update Signal Map Kernel grid(" << gridBlockPerRegion.x << "," << gridBlockPerRegion.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
    UpdateSignalMap_k<<<gridBlockPerRegion, block, smem>>>(
        Dev->RegionStateMask.getPtr(),
        Dev->BfMask.getPtr(),
        Dev->BeadParamCube.getPtr(),
        Dev->ResultCube.getPtr(),
        Dev->AverageSignalRegion.getPtr()
    );

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: UpdateSignalMap_k finalized" << endl;
#endif
  }

  if(ConfP.PerformWellsLevelXTalk() || ConfP.PerformPolyClonalFilter()){
    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: executing post processing and corrections kernel grid(" << gridWarpPerRow.x << "," << gridWarpPerRow.y  << "), block(" << block.x << "," << block.y <<"), smem(0)" << endl;
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

#if DEBUG_SYNC || DEBUG_OUTPUT
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK();
#endif
#if DEBUG_OUTPUT
    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: ProtonXTalk_k finalized" << endl;
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
        cout << "CUDA: Applying PolyClonal Filter after Flow: " << GpFP.getRealFnum() << endl;
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

void BkgGpuPipeline::getDataForRawWells(RingBuffer<float> * ringbuffer)
{
  float *ampBuf = ringbuffer->writeOneBuffer();
  // need copies and copydrift too for new 1.wells format
  Dev->ResultCube.copyPlanesOut(ampBuf,ResultAmpl,1);
  ringbuffer->updateWritePos();
}

void BkgGpuPipeline::getDataForPostFitStepsOnHost()
{
  Host->ResultCube.copy(Dev->ResultCube);
  Host->BeadStateMask.copy(Dev->BeadStateMask);
  Host->PerFlowRegionParams.copy(Dev->PerFlowRegionParams);
}



//debug helper
void BkgGpuPipeline::printBkgModelMaskEnum(){
  std::cout << "CUDA: BkgModelMask flags: "<< std::endl
      <<" BkgMaskBadRead           " << BkgMaskBadRead << std::endl
      <<" BkgMaskPolyClonal        " << BkgMaskPolyClonal << std::endl
      <<" BkgMaskCorrupt           " << BkgMaskCorrupt << std::endl
      <<" BkgMaskRandomSample      " << BkgMaskRandomSample << std::endl
      <<" BkgMaskHighQaulity       " << BkgMaskHighQaulity << std::endl
      <<" BkgMaskRegionalSampled   " << BkgMaskRegionalSampled << std::endl
      <<" BkgMaskPinned            " << BkgMaskPinned << std::endl;
}



void BkgGpuPipeline::printRegionStateMask(){
  std::cout << "CUDA: BkgModelMask flags: "<< std::endl
      <<  " RegionMaskNoLiveBeads                          " <<  RegionMaskNoLiveBeads << std::endl
      <<  " RegionMaskNoT0Average                          " <<  RegionMaskNoT0Average << std::endl
      <<  " RegionMaskT0AverageBelowThreshold ("<<THRESHOLD_T0_AVERAGE<<")          " <<  RegionMaskT0AverageBelowThreshold << std::endl
      <<  " RegionMaskNoEmpties                            " <<  RegionMaskNoEmpties << std::endl
      <<  " RegionMaskNumEmptiesBelowThreshold ("<<THRESHOLD_NUM_EMPTIES<<")         " <<  RegionMaskNumEmptiesBelowThreshold << std::endl
      <<  " RegionMaskNoRegionSamples                      " <<  RegionMaskNoRegionSamples << std::endl
      <<  " RegionMaskNumRegionSamplesBelowThreshold ("<<THRESHOLD_NUM_REGION_SAMPLE<<")   " <<  RegionMaskNumRegionSamplesBelowThreshold << std::endl;
}




