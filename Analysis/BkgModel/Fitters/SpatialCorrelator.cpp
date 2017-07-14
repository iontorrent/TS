/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SpatialCorrelator.h"
#include "DNTPRiseModel.h"
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include "LinuxCompat.h"
#include "SignalProcessingMasterFitter.h"
#include "RawWells.h"
#include "MathOptim.h"
//#include "ClonalFilter/mixed.h"
#include "BkgDataPointers.h"



SpatialCorrelator::SpatialCorrelator ()
{
  // do nothing special
 // data pointed at
  region = NULL;
  region_data = NULL;


  // things to do
  Defaults();

}

void SpatialCorrelator::Defaults()
{
  //region = region_data->region;

 // my_xtalk.DefaultPI(); // set this from the master!
  //TestRead();
  //TestWrite();
}

void SpatialCorrelator::SetRegionData(RegionalizedData *_region_data, SlicedChipExtras *extras ) {
  region_data = _region_data;
  region_data_extras = extras;
  region = region_data->region;
}


void SpatialCorrelator::MakeSignalMap(HplusMap &signal_map, int fnum)
{
  Region *region = region_data->region;
  signal_map.region_mean_sig = 0.0f;
  signal_map.bead_mean_sig = 0.0f;

  memset(signal_map.ampl_map,0,sizeof(float[region->w*region->h]));
  for (int ibd=0;ibd < region_data->my_beads.numLBeads;ibd++)
  {
    int row,col;
    BeadParams *tbead = &region_data->my_beads.params_nn[ibd];
    row = tbead->y;
    col = tbead->x;

    //recover h+ signal by rescaling amplitude by copies
    float bead_sig = tbead->Copies*tbead->Ampl[fnum];
    // special cases:  poly clonal beads?
    // special cases: washouts?
    // special cases: crazy values?
    // should this be multiplied by etbR?
    signal_map.ampl_map[row*region->w + col] = bead_sig;
    signal_map.region_mean_sig += bead_sig;

  }
  signal_map.bead_mean_sig = signal_map.region_mean_sig;
  if (region_data->my_beads.numLBeads>0)
    signal_map.bead_mean_sig /= region_data->my_beads.numLBeads;
  signal_map.region_mean_sig /= (region->w*region->h);

};

void SpatialCorrelator::AmplitudeCorrectAllFlows( int flow_block_size, int flow_block_start )
{
  my_hplus.Allocate(region);
  for (int fnum=0; fnum<flow_block_size; fnum++)
    if (!my_xtalk.simple_xtalk)
      NNAmplCorrect(fnum, flow_block_start);
    else{
      SimpleXtalk(fnum,flow_block_start);
      BkgDistortion(fnum,flow_block_start);
    }
  my_hplus.DeAllocate();
}

HplusMap::HplusMap(){
  ampl_map = NULL;
  NucId=0;
  region_mean_sig = 0.0f;
  bead_mean_sig = 0.0f;
}

void HplusMap::Allocate(Region * region){
  if (region==NULL)
    printf("Alert: Null region in spatial correlator");
  if (ampl_map==NULL)
    ampl_map = new float[region->w*region->h];
}

void HplusMap::DeAllocate(){
  if (ampl_map!=NULL)
    delete[] ampl_map;
  ampl_map = NULL;
}


void SpatialCorrelator::NNAmplCorrect(int fnum, int flow_block_start)
{
  // transform amplitudes into signal
   MakeSignalMap(my_hplus,fnum);

  my_hplus.NucId = region_data_extras->my_flow->flow_ndx_map[fnum];
  float flow_num = flow_block_start + fnum;
  // setup the crazy cross-talk function
  my_xtalk.my_empirical_function.SetupFlow(flow_num, my_hplus.region_mean_sig);

  reg_params *my_rp = &region_data->my_regions.rp;

  for (int ibd=0;ibd < region_data->my_beads.numLBeads;ibd++)
  {
    BeadParams *tbead = &region_data->my_beads.params_nn[ibd];
    float etbR = my_rp->AdjustEmptyToBeadRatioForFlow(tbead->R, tbead->Ampl[fnum], tbead->Copies, tbead->phi, my_hplus.NucId, flow_num);
    int phase =  tbead->x & 1;
    float bead_corrector = my_xtalk.UnweaveMap(my_hplus.ampl_map, tbead->y, tbead->x, region, my_hplus.region_mean_sig, phase);

    float hplus_corrector = my_xtalk.my_empirical_function.ComputeCorrector(etbR,bead_corrector);

    tbead->Ampl[fnum] -= hplus_corrector/tbead->Copies;

    if (tbead->Ampl[fnum]!=tbead->Ampl[fnum])
    {
      printf("NAN: corrected to zero at %d %d %d\n", tbead->y,tbead->x,fnum);
      tbead->Ampl[fnum] = 0.0f;
    }
  }
}

void SpatialCorrelator::SimpleXtalk(int fnum, int flow_block_start){
  // transform amplitudes into signal
   MakeSignalMap(my_hplus,fnum);

  my_hplus.NucId = region_data_extras->my_flow->flow_ndx_map[fnum];

  // as we fill in entries off the map with the region mean signal
  // we can generate an empty corrector by placing the estimating point completely outside the map
  const int OUTSIDE_THE_MAP = -100;
  float empty_corrector = my_xtalk.UnweaveMap(my_hplus.ampl_map, OUTSIDE_THE_MAP,OUTSIDE_THE_MAP,region, my_hplus.region_mean_sig, 0);

  for (int ibd=0;ibd < region_data->my_beads.numLBeads;ibd++)
  {
    BeadParams *tbead = &region_data->my_beads.params_nn[ibd];
    //float etbR = AdjustEmptyToBeadRatioForFlow(tbead->R,my_rp,my_hplus.NucId,flow_num);

    int phase =  tbead->x & 1;
    float bead_corrector = my_xtalk.UnweaveMap(my_hplus.ampl_map, tbead->y, tbead->x, region, my_hplus.region_mean_sig, phase);

    // the obvious corrector is the local xtalk - average crosstalk seen by empties
    // what is the correct buffering compensation here?
    float hplus_corrector = bead_corrector - empty_corrector;


    tbead->Ampl[fnum] -= hplus_corrector/tbead->Copies;

    if (tbead->Ampl[fnum]!=tbead->Ampl[fnum])
    {
      printf("NAN: corrected to zero at %d %d %d\n", tbead->y,tbead->x,fnum);
      tbead->Ampl[fnum] = 0.0f;
    }
  }

}

void SpatialCorrelator::BkgDistortion(int fnum, int flow_block_start)
{
  // further post-well correction
  // recycles values already computed from MakeSignalMap

  //float flow_num = region_data->my_flow.buff_flow[fnum];
  float flow_num = flow_block_start+fnum;
  float distortion_factor = 0.0f;
  // this is put here because we are already looping over all beads
  distortion_factor = my_xtalk.my_bkg_distortion.AdditiveDistortion(flow_num,my_hplus.region_mean_sig);
  float multiplicative_distortion =my_xtalk.my_bkg_distortion.MultiplicativeDistortion(flow_num);
  reg_params *my_rp = &region_data->my_regions.rp;

  //printf("Region: %d Flow: %d Distortion: %f\n",region_data->region->index,(int)flow_num, distortion_factor);
  for (int ibd=0;ibd < region_data->my_beads.numLBeads;ibd++)
  {
    BeadParams *tbead = &region_data->my_beads.params_nn[ibd];
    // systematic shift detected in intensities
    // mostly proportional to mean signal (or possibly to out-of-phase signal)
    // systematic shift in all intensities: fudge factor
    float etbR = my_rp->AdjustEmptyToBeadRatioForFlow(tbead->R, tbead->Ampl[fnum], tbead->Copies, tbead->phi, my_hplus.NucId, flow_num);
    //float counter_etbR = 1/(1-0.7);  // default etbr = 0.7
    //float local_distortion = counter_etbR*(1-etbR)*distortion_factor; // more like empty, less distortion?
    float local_distortion = distortion_factor;
    tbead->Ampl[fnum] += local_distortion/tbead->Copies;
    tbead->Ampl[fnum] *= multiplicative_distortion; // currently a nop
    if (tbead->Ampl[fnum]!=tbead->Ampl[fnum])
    {
      printf("NAN: corrected to zero at %d %d %d\n", tbead->y,tbead->x,fnum);
      tbead->Ampl[fnum] = 0.0f;
    }
  }

}
