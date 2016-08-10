/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#include "exptaildecayfit.h"
#include "FitExpDecay.h"
#include "TraceCorrector.h"

#define MIN_VALID_FLOWS 6

ExpTailDecayFit::ExpTailDecayFit (SignalProcessingMasterFitter &_bkg) :
  bkg (_bkg)
{
}


bool ExpTailDecayFit::ComputeAverageValidTrace(float *avg_trc, float *incorporation_traces,
    BeadParams *p, int npts, float low_A, float hi_A,
    int flow_block_size
  )
{
  int navg = 0;
  memset(avg_trc,0,sizeof(float[npts]));
   for (int fnum=0;fnum < flow_block_size;fnum++)
   {
      // keep out long HPs in order to prevent pollution of data
      // points with additional generated protons
      // keep out non-incorporating flows because they don't contain
      // exponential decay information
      if ((p->Ampl[fnum] > low_A) && (p->Ampl[fnum] < hi_A))
      {
         AccumulateVector(avg_trc,&incorporation_traces[fnum*npts],npts);
          navg++;
      }
   }
   if (navg>MIN_VALID_FLOWS){
    MultiplyVectorByScalar(avg_trc,1.0f/navg,npts);
    return(true);
   }
   return(false);
}

int SetWeightVector(float *weight_vect, int npts, std::vector<float> &ftimes, float fi_end){
  int i_start = -1;
  for (int i=0;i < npts;i++)
    {
       if (ftimes[i] < fi_end)
          weight_vect[i] = 0.0f;
       else
       {
          weight_vect[i] = 1.0f;
          if (i_start < 0)
             i_start = i;
       }
    }
  return(i_start);
}

void  ExpTailDecayFit::FitTauAdj(float *incorporation_traces, float *bkg_traces, BeadParams *p, reg_params *rp, FlowBufferInfo *my_flow, TimeCompression &time_c, int flow_block_size, int flow_block_start
  )
{
  int npts = time_c.npts();
   float tau_adj = p->tau_adj;

  std::vector<float> ftimes = time_c.frameNumber;

  if ( flow_block_start == 0 )    // First block ever.
   {
      // create average trace across all flows and process it
      float avg_trc[npts];
      const float LOW_AMPLITUDE = 0.5f;
      const float HI_AMPLITUDE = 3.0f;
      const float AVG_AMPLITUDE = 1.5f;
      bool find_adj = ComputeAverageValidTrace(avg_trc, incorporation_traces, p, npts,LOW_AMPLITUDE, HI_AMPLITUDE, flow_block_size);

      if (find_adj)
      {

         float fi_end = GetModifiedIncorporationEnd(&rp->nuc_shape,my_flow->flow_ndx_map[0],0, AVG_AMPLITUDE);
         float weight_vect[npts];
         int i_start = SetWeightVector(weight_vect,npts,ftimes, fi_end);

         // this is of course inaccurate if we happen to have nuc-modify-ratios that are very different than each other
          float my_etbR = rp->AdjustEmptyToBeadRatioForFlow(p->R, p->Ampl[0],  p->Copies, p->phi, my_flow->flow_ndx_map[0], flow_block_start );
         float my_tauB = rp->ComputeTauBfromEmptyUsingRegionLinearModel(my_etbR);

         FitExpDecayParams min_params,max_params;
         min_params.Signal = 0.0f;
         min_params.tau = my_tauB*0.9f;
         min_params.dc_offset = -50.0f;
         max_params.Signal = 500.0f;
         max_params.tau = my_tauB*1.1f;
         max_params.dc_offset =  50.0f;
         FitExpDecay exp_fitter(npts,&ftimes[0]);

         exp_fitter.SetWeightVector(weight_vect);
         exp_fitter.SetLambdaStart(1.0E-20f);
         exp_fitter.SetLambdaThreshold(100.0f);
         exp_fitter.SetParamMax(max_params);
         exp_fitter.SetParamMin(min_params);
         exp_fitter.params.Signal = 20.0f;
         exp_fitter.params.tau = my_tauB;
         exp_fitter.params.dc_offset = 0.0f;
         exp_fitter.SetStartAndEndPoints(i_start,npts);
         exp_fitter.Fit(false, 200, avg_trc);

         tau_adj = exp_fitter.params.tau/my_tauB;
      }
   }

  p->tau_adj = tau_adj;
}

// note: we 'subtract background' here, and in refine-fit
// these are not redundant operations
// as when we alter buffering, we may alter background subtraction later
// although we currently do not
void ExpTailDecayFit::AdjustBufferingOneBead(int ibd, int flow_block_size, int flow_block_start){
  BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
  float block_signal_corrected[bkg.region_data->my_scratch.bead_flow_t];

  bkg.trace_bkg_adj->ReturnBackgroundCorrectedSignal (block_signal_corrected, NULL,NULL, ibd, flow_block_size, flow_block_start);

  FitTauAdj(block_signal_corrected,bkg.region_data->my_scratch.shifted_bkg,
                             p,&bkg.region_data->my_regions.rp,
                             bkg.region_data_extras.my_flow,bkg.region_data->time_c, flow_block_size, flow_block_start);
}

void ExpTailDecayFit::AdjustBufferingEveryBead(int flow_block_size, int flow_block_start){
  if (bkg.global_defaults.signal_process_control.exp_tail_fit & bkg.global_defaults.signal_process_control.exp_tail_tau_adj){

      bkg.region_data->my_regions.cache_step.CalculateNucRiseFineStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,
                                                                       *bkg.region_data_extras.my_flow); // the same for the whole region because time-shift happens per well
      bkg.region_data->my_regions.cache_step.CalculateNucRiseCoarseStep (&bkg.region_data->my_regions.rp,bkg.region_data->time_c,*bkg.region_data_extras.my_flow); // use for xtalk

      bkg.region_data->my_scratch.FillShiftedBkg (*bkg.region_data->emptytrace, bkg.region_data->my_regions.rp.tshift, bkg.region_data->time_c, true, flow_block_size);

      if (bkg.trace_xtalk_spec.simple_model)
        bkg.trace_xtalk_execute.ComputeTypicalCrossTalk(bkg.trace_xtalk_execute.my_generic_xtalk, NULL, flow_block_size, flow_block_start ); // get the generic value set

      for (int ibd = 0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
      {
        if (bkg.region_data->my_beads.params_nn[ibd].FitBeadLogic ()){
          AdjustBufferingOneBead(ibd, flow_block_size, flow_block_start);
        }
      }
    }
  }
