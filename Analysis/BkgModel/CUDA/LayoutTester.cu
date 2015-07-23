/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 * TestNewLayout.cu
 *
 *  Created on: Feb 18, 2014
 *      Author: jakob
 */

#include <iostream>
#include "LayoutTranslator.h"
#include "LayoutTester.h"
#include "MasterKernel.h"
#include "DeviceParamDefines.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"
#include "BkgGpuPipeline.h"

using namespace std;

#define GPU_REG_FITTING 1

#define DEBUG_OUTPUT 0
#define DEBUG_REGION 57

#define RESULTS_CHECK 0
#define RESULT_DUMP 0  // 1 = dump 0 = compare
#define INIT_CONTROL_OUPUT 0


#define EMPTY_CONTROL 0


//////////////////////////////////////////////
///  test function


bool blockLevelSignalProcessing(
    BkgModelWorkInfo* bkinfo,
    int flowBlockSize,
    int deviceId)
{

  int flowInBlock = (bkinfo->flow - 20)%flowBlockSize;

  // create static GpuPipeline Object
  //allocates all permanent device and host buffers
  //initializes all persistent buffers on the device side
  //initializes the poissontables
  //runs the T0average num bead meta data kernel
  static BkgGpuPipeline GpuPipeline(bkinfo,flowBlockSize, bkinfo->flow, GPU_NUM_FLOW_BUF, deviceId);

  //Update Host Side Buffers and Pinned status
  Timer newPtime;

  // This needs to go as floworder information is available and can be copied to device memory 
  // on the first call to this pipeline
  // Probably need new pinned mask to be transferred every flow so that ZeroOutPins
  // can be avoided
  // No need to increment the flow here. Again just need starting flow on device 
  // and things can fly from there

  /*GlobalDefaultsForBkgModel tmp = bkinfo->bkgObj->getGlobalDefaultsForBkgModel();
  for(size_t i=0; i < GpuPipeline.getParams().getNumRegions(); i++)
  {
    SignalProcessingMasterFitter * Obj = bkinfo[i].bkgObj;
    Obj->region_data->AddOneFlowToBuffer ( tmp,*(Obj->region_data_extras.my_flow), bkinfo[i].flow);
    Obj->region_data_extras.my_flow->Increment();
    Obj->region_data->my_beads.ZeroOutPins(  Obj->region_data->region,
        Obj->GetGlobalStage().bfmask,
        *Obj->GetGlobalStage().pinnedInFlow,
        Obj->region_data_extras.my_flow->flow_ndx_map[flowInBlock],
        flowInBlock);
  }*/

  //update all per flow by flow data
  GpuPipeline.PerFlowDataUpdate(bkinfo,flowInBlock);

  //backup current region param state, this function only does work the very first time it gets called
  GpuPipeline.InitRegionalParamsAtFirstFlow();

  GpuPipeline.ExecuteGenerateBeadTrace();

  GpuPipeline.ExecuteCrudeEmphasisGeneration();
  GpuPipeline.ExecuteRegionalFitting();

  GpuPipeline.ExecuteFineEmphasisGeneration();
  GpuPipeline.ExecuteSingleFlowFit();
  GpuPipeline.ExecutePostFitSteps();
  GpuPipeline.HandleResults(); // copy reg_params and single flow fit results to host


  std::cout << "New pipeline time for flow " << bkinfo->flow << ": " << newPtime.elapsed() << std::endl;

  // Not required if accomplished by a background thread
/*
  if(flowInBlock == flowBlockSize-1){
    flowInBlock = 0;
    cout << "BkgGpuPipeline: Reinjecting results for flowblock containing flows "<< GpuPipeline.getFlowP().getRealFnum() - flowBlockSize << " to " << GpuPipeline.getFlowP().getRealFnum() << endl;
    cout << "waiting on CPU Q ... ";
    bkinfo->pq->GetCpuQueue()->WaitTillDone();
    cout <<" continue" << endl;
  }
*/

  return true;
}



