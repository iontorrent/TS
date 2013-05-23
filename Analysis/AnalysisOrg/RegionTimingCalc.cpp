/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include "GaussianExponentialFit.h"
#include "Utils.h"

#include "RegionTimingCalc.h"

void FindStartingParametersForBkgModel(GaussianExponentialParams &my_initial_params, float &my_fit_residual, float *avg_sig, int avg_len){
       float max_val = FLT_MIN;
     for (int i = 0; i < avg_len; i++)
      {
        if (avg_sig[i] > max_val)
          max_val = avg_sig[i];
      }
      
      GaussianExponentialFit gexp_fitter(avg_len);
      GaussianExponentialParams max_params, min_params;

      max_params.A = 300;
      max_params.sigma = 5.0;
      max_params.t_mid_nuc = 30;
      max_params.tau = 60;

      min_params.A = 0;
      min_params.sigma = 0.5;
      min_params.t_mid_nuc = 0;
      min_params.tau = 5;

      gexp_fitter.SetParamMax(max_params);
      gexp_fitter.SetParamMin(min_params);


      // decent guesses
      gexp_fitter.params.A = max_val;
      gexp_fitter.params.sigma = 2.0;
      gexp_fitter.params.t_mid_nuc = 5.0;
      gexp_fitter.params.tau = 20.0;

      // search for something close
      gexp_fitter.GridSearch(12, avg_sig);

      // do the fit
      gexp_fitter.Fit(100, avg_sig);
      
      my_initial_params.A = gexp_fitter.params.A;
      my_initial_params.sigma = gexp_fitter.params.sigma;
      my_initial_params.t_mid_nuc = gexp_fitter.params.t_mid_nuc;
      my_initial_params.tau = gexp_fitter.params.tau;
      my_fit_residual = gexp_fitter.GetResidual();
}

void FillOneRegionTimingParameters(RegionTiming *region_time, Region *regions, int r, AvgKeyIncorporation *kic){
     // Extract what we need from the incorporation trace
      float T0Start = kic->GetStart(r,regions[r].row,
                                       regions[r].row + regions[r].h,
                                       regions[r].col, regions[r].col+regions[r].w);
      
      float *avg_sig = kic->GetAvgKeySig(r,
                       regions[r].row, regions[r].row + regions[r].h,
                       regions[r].col, regions[r].col + regions[r].w);
      int avg_len = kic->GetAvgKeySigLen();
      
      GaussianExponentialParams my_initial_params;
      float my_fit_residual=0;
//
      FindStartingParametersForBkgModel(my_initial_params, my_fit_residual, avg_sig, avg_len);

     /* printf(
        "Region:(%d,%d) A:%f, sigma:%f, t_mid_nuc:%f, tau:%f, res:%f t0:%d\n",
        regions[r].col, regions[r].row,
        my_initial_params.A, my_initial_params.sigma,
        my_initial_params.t_mid_nuc, my_initial_params.tau,
        my_fit_residual,
        T0Start);*/ // now dumped directly to file
      region_time[r].t0_frame = T0Start; 
        region_time[r].t_mid_nuc = T0Start + my_initial_params.t_mid_nuc;
        region_time[r].t_sigma = my_initial_params.sigma;
}

// find regional parameters
// this can be usefully threaded again
// or simply eliminated
void FillRegionalTimingParameters(RegionTiming *region_time, Region *regions, int numRegions, AvgKeyIncorporation *kic){

    for (int r=0; r<numRegions; r++){
      FillOneRegionTimingParameters(region_time,regions,r, kic);
    }
}


extern void *TimingFitWorker(void *arg)
{
  WorkerInfoQueue *q = static_cast<WorkerInfoQueue *>(arg);
  assert(q);

  bool done = false;

  while (!done)
  {
    WorkerInfoQueueItem item = q->GetItem();

    if (item.finished == true)
    {
      // we are no longer needed...go away!
      done = true;
      q->DecrementDone();
      continue;
    }
    if (true) {
      TimingFitWorkOrder *info = (TimingFitWorkOrder *)(item.private_data);
      FillOneRegionTimingParameters(info->region_time, info->regions, info->r, info->kic);
    }
  // indicate we finished that bit of work
    q->DecrementDone();
  }

  return (NULL);
}


void threadedFillRegionalTimingParameters(std::vector<RegionTiming>& region_time, std::vector<Region>& regions, AvgKeyIncorporation *kic, int numThreads)
{
  WorkerInfoQueue *threadWorkQ;
  WorkerInfoQueueItem item;
  int numRegions = (int)regions.size();
  int numWorkers = numCores();
  if (numThreads > 0) numWorkers = numThreads; 
  if (numWorkers>4) numWorkers = 4;
  
    threadWorkQ = new WorkerInfoQueue(std::max(numRegions,numWorkers)+1);
  // spawn threads
  {
    int cworker;
    pthread_t work_thread;
    
    for (cworker=0; cworker<numWorkers; cworker++)
    {
      int t= pthread_create(&work_thread, NULL, TimingFitWorker,threadWorkQ);
      if (t)
        fprintf(stderr, "Error starting thread\n");
    }
  }
 
  TimingFitWorkOrder *timing_fit_orders = new TimingFitWorkOrder[numRegions];
  for (int r=0; r<numRegions; r++)
  {
    timing_fit_orders[r].type = 0;
    timing_fit_orders[r].region_time = &region_time[0];
    timing_fit_orders[r].regions = &regions[0];
    timing_fit_orders[r].kic = kic;
    timing_fit_orders[r].r = r;
    
    // on queue
    item.finished = false;
    item.private_data = (void *) &timing_fit_orders[r];
    threadWorkQ->PutItem(item);
  }
  threadWorkQ->WaitTillDone();
  delete[] timing_fit_orders;
  
  // tell worker threads to exit
  item.finished=true;
  item.private_data = NULL;
  for (int i=0; i<numWorkers; i++)
    threadWorkQ->PutItem(item);
  threadWorkQ->WaitTillDone();
  
  delete threadWorkQ;
}
