/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BeadScratch.h"


BeadScratchSpace::BeadScratchSpace()
{
  scratchSpace = fval = NULL;
  ival = NULL;

  cur_xtflux_block = NULL;
  custom_emphasis = NULL;

  observed = NULL;
  shifted_bkg = NULL;
  cur_shift = 0;

  bead_flow_t = 0;
  npts = 0;

  for (int j=0; j<NUMFB; j++)
    custom_emphasis_scale[j] = 1.0; // might be so unfortunate as to divide by this
}

BeadScratchSpace::~BeadScratchSpace()
{

  if (cur_xtflux_block != NULL) delete [] cur_xtflux_block;

  if (scratchSpace != NULL)
  {
    fval = NULL; // turn off pointer into scratchspace
    delete [] scratchSpace;
  }
  if (ival!=NULL) delete[] ival;

  if (observed !=NULL) delete[] observed;
  if (shifted_bkg!=NULL) delete[] shifted_bkg;
  if (custom_emphasis!=NULL) delete [] custom_emphasis;

}

void BeadScratchSpace::Allocate (int _npts,int num_derivatives)
{
  npts = _npts;
  // now that I know how many points we have
  bead_flow_t = NUMFB*npts; // the universal unit of currency flows by time
  int scratchSpace_nElem = bead_flow_t*num_derivatives;
  int scratchSpace_len = sizeof (float) * scratchSpace_nElem;
  scratchSpace = new float[scratchSpace_nElem];
  memset (scratchSpace,0,scratchSpace_len);
  fval = scratchSpace;
  ival = new float [bead_flow_t];
  cur_xtflux_block = new float[bead_flow_t];
  observed = new float [bead_flow_t];
  shifted_bkg = new float [bead_flow_t];
  custom_emphasis = new float [bead_flow_t];
}

void BeadScratchSpace::ResetXtalkToZero()
{
  memset (cur_xtflux_block,0,sizeof (float[bead_flow_t]));  // no cross-talk for projection search
}

void BeadScratchSpace::FillObserved (BkgTrace &my_trace,int ibd)
{
  my_trace.FillSignalForBead(observed,ibd);
}

void BeadScratchSpace::FillShiftedBkg(BkgTrace &my_trace, float tshift, bool force_fill)
{
  if (tshift!=cur_shift || force_fill) // dangerous
    my_trace.GetShiftedBkg(tshift, shifted_bkg);
  cur_shift=tshift; // keep current
}

// copies per-HP emphasis vectors into a contiguous custom emphasis vector for all NUMFB flows
void MAKE_CUSTOM_EMPHASIS_VECT (float *output,float *SourceEmphasisVector[],int *MyEmphasis,int npts)
{
  for (int fnum=0;fnum < NUMFB;fnum++)
    memcpy (&output[fnum*npts],SourceEmphasisVector[MyEmphasis[fnum]],sizeof (float[npts]));
}

void BeadScratchSpace::FillEmphasis (int *my_emphasis, float *source_emphasis[],float *source_emphasis_scale)
{
  MAKE_CUSTOM_EMPHASIS_VECT (custom_emphasis,source_emphasis,my_emphasis,npts);
  for (int fnum =0; fnum<NUMFB; fnum++)
    custom_emphasis_scale[fnum] = source_emphasis_scale[my_emphasis[fnum]];
}

void BeadScratchSpace::CreateEmphasis(float *source_emphasis[], float *source_emphasis_scale)
{
  FillEmphasis(WhichEmphasis,source_emphasis,source_emphasis_scale);
}

void BeadScratchSpace::SetEmphasis(float *Ampl, int max_emphasis)
{
  ComputeEmphasisOneBead(WhichEmphasis,Ampl,max_emphasis);
}

float BeadScratchSpace::CalculateFitError (float *per_flow_output, int numfb)
{
  // filled scratch, now evaluate
  float tot_err=0.0;
  float scale = 0.0;
  for (int nflow=0;nflow < numfb;nflow++)
  {
    float flow_err = 0.0;
    
    for (int i=0;i<npts;i++)   // evaluate over actual data points
    {
      int ti = i+nflow*npts;
      float eval= (observed[ti]-fval[ti]) *custom_emphasis[ti];
      flow_err += (eval*eval);
    }

    if (per_flow_output != NULL)
      per_flow_output[nflow] = sqrt (flow_err/npts);   // scale by real data points?

    tot_err += flow_err;
    scale += custom_emphasis_scale[nflow];
  }
  return (sqrt (tot_err/scale));
}



void BeadScratchSpace::MultiFlowReturnResiduals(float *y_minus_f)
{
    for (int ti=0; ti<bead_flow_t; ti++)
      y_minus_f[ti] = observed[ti]-fval[ti];
}



void BeadScratchSpace::MultiFlowReturnFval(float *per_flow_fval, int numfb)
{
  // return the current objective function values, only used for debugging
  for (int nflow=0;nflow < numfb;nflow++)
  {
    for (int i=0;i<npts;i++)   // evaluate over actual data points
    {
      int ti = i+nflow*npts;
      per_flow_fval[ti] = fval[ti];
    }
  }
}
