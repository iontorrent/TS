/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BeadScratch.h"

using namespace std;

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

  custom_emphasis_scale = 0;
  WhichEmphasis = 0;
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

  delete [] custom_emphasis_scale;
  delete [] WhichEmphasis;

}

void BeadScratchSpace::Allocate (int _npts,int _num_derivatives, int flow_block_size)
{
  npts = _npts;
  num_derivatives = _num_derivatives;

  // now that I know how many points we have
  bead_flow_t = flow_block_size*npts; // the universal unit of currency flows by time

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

  custom_emphasis_scale = new float[ flow_block_size ];
  WhichEmphasis = new int[ flow_block_size ];

  for (int j=0; j<flow_block_size; j++)
    custom_emphasis_scale[j] = 1.0; // might be so unfortunate as to divide by this
}

void BeadScratchSpace::ResetXtalkToZero() const
{
  memset (cur_xtflux_block,0,sizeof (float[bead_flow_t]));  // no cross-talk for projection search
}

void BeadScratchSpace::FillObserved (const BkgTrace &my_trace,int ibd, int flow_block_size) const
{
  my_trace.MultiFlowFillSignalForBead(observed,ibd, flow_block_size);
}

void BeadScratchSpace::FillShiftedBkg(
    const EmptyTrace &emptytrace, 
    float tshift, 
    const TimeCompression &time_c, 
    bool force_fill, 
    int flow_block_size
  )
{
  if (tshift!=cur_shift || force_fill) // dangerous
    emptytrace.GetShiftedBkg(tshift, time_c, shifted_bkg, flow_block_size);
  cur_shift=tshift; // keep current
}

void BeadScratchSpace::FillEmphasis (int *my_emphasis, float *source_emphasis[], 
    const vector<float>& source_emphasis_scale,
    int flow_block_size
  )
{
  for (int fnum =0; fnum<flow_block_size; fnum++)
  {
    // copies per-HP emphasis vectors into a contiguous custom emphasis vector for all flows
    memcpy (&custom_emphasis[fnum*npts],source_emphasis[my_emphasis[fnum]],sizeof (float[npts]));

    custom_emphasis_scale[fnum] = source_emphasis_scale[my_emphasis[fnum]];
  }
}

void BeadScratchSpace::CreateEmphasis(float *source_emphasis[], const vector<float>& source_emphasis_scale, int flow_block_size)
{
  FillEmphasis(WhichEmphasis,source_emphasis,source_emphasis_scale, flow_block_size);
}

void BeadScratchSpace::SetEmphasis(float *Ampl, int max_emphasis, int flow_block_size)
{
  BeadParams::ComputeEmphasisOneBead(WhichEmphasis,Ampl,max_emphasis, flow_block_size);
}

float BeadScratchSpace::CalculateFitError (float *per_flow_output, int flow_block_size)
{
  // filled scratch, now evaluate
  float tot_err=0.0f;
  float scale = 0.0f;
  for ( int nflow=0 ; nflow < flow_block_size ; nflow++ )
  {
    float flow_err = 0.0f;
    
    for (int i=0;i<npts;i++)   // evaluate over actual data points
    {
      int ti = i+nflow*npts;
      float eval= (observed[ti]-fval[ti]) *custom_emphasis[ti];
      flow_err += (eval*eval);
    }

    if (per_flow_output != NULL){
      assert (npts > 0);
      assert (flow_err >= 0);
      per_flow_output[nflow] = sqrt (flow_err/npts);   // scale by real data points?
    }
    tot_err += flow_err;
    scale += custom_emphasis_scale[nflow];
  }
  assert (scale > 0);
  assert (tot_err >=0 );
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

incorporation_params_block_flows::incorporation_params_block_flows( int flow_block_size )
{
  NucID = new int[ flow_block_size ];
  SP = new float[ flow_block_size ];
  sens = new float[ flow_block_size ];
  d = new float[ flow_block_size ];
  kr = new float[ flow_block_size ];
  kmax = new float[ flow_block_size ];
  C = new float[ flow_block_size ];
  molecules_to_micromolar_conversion = new float[ flow_block_size ];
  nuc_rise_ptr = new float*[ flow_block_size ];
  ival_output = new float*[ flow_block_size ];
}

incorporation_params_block_flows::~incorporation_params_block_flows()
{
  delete [] NucID;
  delete [] SP;
  delete [] sens;
  delete [] d;
  delete [] kr;
  delete [] kmax;
  delete [] C;
  delete [] molecules_to_micromolar_conversion;
  delete [] nuc_rise_ptr;
  delete [] ival_output;
}

buffer_params_block_flows::buffer_params_block_flows( int flow_block_size )
{
  etbR = new float[ flow_block_size ];
  tauB = new float[ flow_block_size ];
}

buffer_params_block_flows::~buffer_params_block_flows()
{
  delete [] etbR;
  delete [] tauB;
}
