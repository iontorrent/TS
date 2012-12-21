/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GlobalWriter.h"
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include "LinuxCompat.h"
#include "RawWells.h"
#include "TimeCompression.h"
#include "EmphasisVector.h"
#include "BkgDataPointers.h"
#include "BkgTrace.h"
#include "RegionalizedData.h"
//#include "BkgModel.h"

void extern_links::WriteOneBeadToDataCubes ( bead_params &p, int flow, int iFlowBuffer, int last, int ibd, Region *region, BkgTrace &my_trace )
{
  int x = p.x+region->col;
  int y = p.y+region->row;

  // one of these things is not like the others, one of these things just doesn't belong
  if ( mPtrs->mBeadDC!=NULL )
    mPtrs->copyCube_element ( mPtrs->mBeadDC,x,y,flow,my_trace.GetBeadDCoffset ( ibd,iFlowBuffer ) );

  // use copyCube_element to copy DataCube element in BkgModel::WriteBeadParameterstoDataCubes
  if ( mPtrs->mAmpl!=NULL )
    mPtrs->copyCube_element ( mPtrs->mAmpl,x,y,flow,p.Ampl[iFlowBuffer] );
  if ( mPtrs->mKMult!=NULL )
    mPtrs->copyCube_element ( mPtrs->mKMult,x,y,flow,p.kmult[iFlowBuffer] );

  if ( mPtrs->mBeadInitParam!=NULL ) // should be done only once
  {
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam,x,y,0,p.Copies );
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam,x,y,1,p.R );
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam,x,y,2,p.dmult );
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam,x,y,3,p.gain );
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam, x,y,4,my_trace.t0_map[p.x+p.y*region->w] ); // one of these things is not like the others
  }

  if ( ( iFlowBuffer+1 ) ==NUMFB || last ) // per compute block, not per flow
  {
    int iBlk = CurComputeBlock ( flow );
    if ( mPtrs->mBeadFblk_avgErr!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_avgErr,x,y,iBlk,p.my_state->avg_err );
    if ( mPtrs->mBeadFblk_clonal!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_clonal,x,y,iBlk,p.my_state->clonal_read?1:0 );
    if ( mPtrs->mBeadFblk_corrupt!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_corrupt,x,y,iBlk,p.my_state->corrupt?1:0 );
  }
}


//@TODO: this is not actually a bkgmodel function but a function of my_beads?
void extern_links::WriteBeadParameterstoDataCubes ( int iFlowBuffer, bool last,Region *region, BeadTracker &my_beads, flow_buffer_info &my_flow, BkgTrace &my_trace )
{
  if ( mPtrs == NULL )
    return;


  int flow = my_flow.buff_flow[iFlowBuffer];
  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    struct bead_params &p = my_beads.params_nn[ibd];
    WriteOneBeadToDataCubes ( p, flow, iFlowBuffer,  last, ibd, region, my_trace );
  }
}




void extern_links::SendErrorVectorToHDF5 ( bead_params *p, error_track &err_t, Region *region, flow_buffer_info &my_flow )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->mResError!=NULL )
    {
      int x = p->x+region->col;
      int y = p->y+region->row;
      //for (int fnum=0; fnum<NUMFB; fnum++)
      if ( mPtrs->mResError!=NULL )
      {
        for ( int fnum=0; fnum<my_flow.flowBufferCount; fnum++ )
        {
          // use copyCube_element to copy DataCube element to mResError
          mPtrs->copyCube_element ( mPtrs->mResError,x,y,my_flow.buff_flow[fnum],err_t.mean_residual_error[fnum] );
        }
      }
    }
  }
}

void extern_links::SendPredictedToHDF5 ( int ibd, float *block_signal_predicted, RegionalizedData &my_region_data )
{
  // placeholder
  if ( mPtrs!=NULL )
  {
    if ( ( mPtrs->m_region_debug_bead_predicted!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), MAX_COMPRESSED_FRAMES );
      int reg = my_region_data.region->index;

      for ( int fnum=0; fnum<my_region_data.my_flow.flowBufferCount; fnum++ )
      {
        int flow= my_region_data.my_flow.buff_flow[fnum];

        for ( int j=0; j<npts; j++ )
        {
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_predicted,reg,j,flow,block_signal_predicted[j+fnum*npts] );
        }
        for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_predicted,reg,j,flow, 0 ); // pad 0's

      }
    }
  }
}

void extern_links::SendCorrectedToHDF5 ( int ibd, float *block_signal_corrected, RegionalizedData &my_region_data )
{
  // placeholder
  if ( mPtrs!=NULL )
  {
    if ( ( mPtrs->m_region_debug_bead_corrected!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), MAX_COMPRESSED_FRAMES );
      int reg = my_region_data.region->index;

      for ( int fnum=0; fnum<my_region_data.my_flow.flowBufferCount; fnum++ )
      {
        int flow= my_region_data.my_flow.buff_flow[fnum];

        for ( int j=0; j<npts; j++ )
        {
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_corrected,reg,j,flow,block_signal_corrected[j+fnum*npts] );
        }
        for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_corrected,reg,j,flow, 0 ); // pad 0's

      }
    }
  }
}

void extern_links::SendXtalkToHDF5 ( int ibd, float *block_signal_xtalk, RegionalizedData &my_region_data )
{
  // placeholder
  if ( mPtrs!=NULL )
  {
    if ( ( mPtrs->m_region_debug_bead_xtalk!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), MAX_COMPRESSED_FRAMES );
      int reg = my_region_data.region->index;

      for ( int fnum=0; fnum<my_region_data.my_flow.flowBufferCount; fnum++ )
      {
        int flow= my_region_data.my_flow.buff_flow[fnum];

        for ( int j=0; j<npts; j++ )
        {
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_xtalk,reg,j,flow,block_signal_xtalk[j+fnum*npts] );
        }
        for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_xtalk,reg,j,flow, 0 ); // pad 0's

      }
    }
  }
}


// this puts our answers into the data structures where they belong
// should be the only point of contact with the external world, but isn't
void extern_links::WriteAnswersToWells ( int iFlowBuffer, Region *region, RegionTracker *my_regions, BeadTracker &my_beads, flow_buffer_info &my_flow )
{
  // make absolutely sure we're upt to date
  my_regions->rp.copy_multiplier[iFlowBuffer] = CalculateCopyDrift ( my_regions->rp, my_flow.buff_flow[iFlowBuffer] );
  //Write one flow's data to 1.wells
  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    float val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_beads.params_nn[ibd].Copies * my_regions->rp.copy_multiplier[iFlowBuffer];
    int x = my_beads.params_nn[ibd].x+region->col;
    int y = my_beads.params_nn[ibd].y+region->row;
    
    if (my_beads.params_nn[ibd].my_state->pinned or my_beads.params_nn[ibd].my_state->corrupt)
      val = 0.0f; // actively suppress pinned wells in case we are so unfortunate as to have an estimate for them

    rawWells->WriteFlowgram ( my_flow.buff_flow[iFlowBuffer], x, y, val );

  }
}

void extern_links::DumpTimeAndEmphasisByRegionH5 ( int reg, TimeCompression &time_c, EmphasisClass &emphasis_data )
{
  if ( mPtrs->mEmphasisParam!=NULL )
  {

    ION_ASSERT ( emphasis_data.numEv <= MAX_POISSON_TABLE_COL, "emphasis_data.numEv > MAX_HPLEN+1" );
    //ION_ASSERT(time_c.npts <= MAX_COMPRESSED_FRAMES, "time_c.npts > MAX_COMPRESSED_FRAMES");
    int npts = std::min ( time_c.npts(), MAX_COMPRESSED_FRAMES );

    if ( mPtrs->mEmphasisParam!=NULL )
    {
      // use copyCube_element to copy DataCube element to mEmphasisParam in BkgModel::DumpTimeAndEmphasisByRegionH5
      for ( int hp=0; hp<emphasis_data.numEv; hp++ )
      {
        for ( int t=0; t< npts; t++ )
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,emphasis_data.EmphasisVectorByHomopolymer[hp][t] );
        for ( int t=npts; t< MAX_COMPRESSED_FRAMES; t++ )
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,0 ); // pad 0's, memset faster here?
      }

      for ( int hp=emphasis_data.numEv; hp<MAX_POISSON_TABLE_COL; hp++ )
      {
        for ( int t=0; t< MAX_COMPRESSED_FRAMES; t++ )
        {
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,0 ); // pad 0's, memset faster here?
        }
      }
    }
  }
}

void extern_links::DumpDarkMatterH5 ( int reg, TimeCompression &time_c, RegionTracker &my_reg_tracker )
{
  if ( mPtrs->mDarkOnceParam!=NULL )
  {
    int npts = time_c.npts();
    for ( int i=0; i<NUMNUC; i++ )
    {
      float *missingMass = my_reg_tracker.missing_mass.dark_nuc_comp[i];
      for ( int j=0; j<npts; j++ )
        mPtrs->copyCube_element ( mPtrs->mDarkOnceParam,reg,i,j,missingMass[j] );
      for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
        mPtrs->copyCube_element ( mPtrs->mDarkOnceParam,reg,i,j,0 ); // pad 0's
    }
  }
}

void extern_links::DumpDarknessH5 ( int reg, reg_params &my_rp )
{
  ///------------------------------------------------------------------------------------------------------------
  /// darkness
  // use copyMatrix_element to copy Matrix element to mDarknessParam in BkgParamH5::DumpBkgModelRegionInfoH5
  ///------------------------------------------------------------------------------------------------------------
  if ( mPtrs->mDarknessParam!=NULL )
  {
    for ( int i=0; i<NUMFB; i++ )
      mPtrs->copyCube_element ( mPtrs->mDarknessParam,reg,i,0,my_rp.darkness[i] );
  }
}

void extern_links::DumpRegionInitH5 ( int reg, RegionalizedData &my_region_data )
{
  if ( mPtrs->mRegionInitParam!=NULL )
  {
    ///------------------------------------------------------------------------------------------------------------
    /// regionInitParam, only once at flow+1=blocksOfFlow
    ///------------------------------------------------------------------------------------------------------------
    mPtrs->copyCube_element ( mPtrs->mRegionInitParam,reg,0,0,my_region_data.t_mid_nuc_start );
    mPtrs->copyCube_element ( mPtrs->mRegionInitParam,reg,1,0,my_region_data.sigma_start );
  }
}

void extern_links::DumpEmptyTraceH5 ( int reg, RegionalizedData &my_region_data )
{
  if ( ( mPtrs->mEmptyOnceParam!=NULL ) & ( mPtrs->mBeadDC_bg!=NULL ) )
  {
    for ( int fnum=0; fnum<my_region_data.my_flow.flowBufferCount; fnum++ )
    {
      int flow= my_region_data.my_flow.buff_flow[fnum];

      EmptyTrace *mt_trace = my_region_data.emptyTraceTracker->GetEmptyTrace ( *my_region_data.region ) ;
      ///------------------------------------------------------------------------------------------------------------
      /// emptyTrace
      // use copyCube_element to copy DataCube element to mEmptyOnceParam in BkgParamH5::DumpBkgModelRegionInfoH5
      ///------------------------------------------------------------------------------------------------------------

      for ( int j=0; j<mt_trace->imgFrames; j++ )
      {
        mPtrs->copyCube_element ( mPtrs->mEmptyOnceParam,reg,j,flow,mt_trace->bg_buffers[j+fnum*mt_trace->imgFrames] );
      }

      ///------------------------------------------------------------------------------------------------------------
      /// bg_bead_DC
      ///------------------------------------------------------------------------------------------------------------

      mPtrs->copyCube_element ( mPtrs->mBeadDC_bg,reg,0, flow,mt_trace->bg_dc_offset[fnum] );
    }
  }
}

void extern_links::DumpRegionOffsetH5 ( int reg, int col, int row )
{
  if ( mPtrs->mRegionOffset!=NULL )
  {

    ///------------------------------------------------------------------------------------------------------------
    /// region_size, only once at flow+1=blocksOfFlow
    ///------------------------------------------------------------------------------------------------------------
    mPtrs->copyCube_element ( mPtrs->mRegionOffset,reg, 0, 0, col );
    mPtrs->copyCube_element ( mPtrs->mRegionOffset,reg,1,0, row );

  }
}

void extern_links::DumpRegionalEnzymatics ( reg_params &rp, int region_ndx, int iBlk,int &i_param )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_enzymatics_param!=NULL )
    {  // float krate[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param, iBlk ) = rp.krate[j];
        i_param++;
      }
      // float d[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param, iBlk ) = rp.d[j];
        i_param++;
      }
      // float kmax[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param, iBlk ) = rp.kmax[j];
        i_param++;
      }
    }
  }
}

void extern_links::DumpRegionalBuffering ( reg_params &rp, int region_ndx, int iBlk,int &i_param )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_buffering_param!=NULL )
    {
      // tau_R_m
      mPtrs->m_buffering_param->At ( region_ndx, i_param, iBlk ) = rp.tau_R_m;
      i_param++;
      // tau_R_o
      mPtrs->m_buffering_param->At ( region_ndx, i_param, iBlk ) = rp.tau_R_o;
      i_param++;
      // tauE
      mPtrs->m_buffering_param->At ( region_ndx, i_param, iBlk ) = rp.tauE;
      i_param++;
      // RatioDrift
      mPtrs->m_buffering_param->At ( region_ndx, i_param, iBlk ) = rp.RatioDrift;
      i_param++;
      // float NucModifyRatio[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_buffering_param->At ( region_ndx, i_param, iBlk ) = rp.NucModifyRatio[j];
        i_param++;
      }
    }
  }
}

void extern_links::DumpRegionNucShape ( reg_params &rp, int region_ndx, int iBlk,int &i_param )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_nuc_shape_param!=NULL )
    {
      // float t_mid_nuc[NUMFB]
      for ( int j=0; j<NUMFB; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.t_mid_nuc[j];
        i_param++;
      }
      // float t_mid_nuc_delay[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.t_mid_nuc_delay[j];
        i_param++;
      }
      // sigma
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.sigma;
      i_param++;

      // float sigma_mult[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.sigma_mult[j];
        i_param++;
      }
      // float t_mid_nuc_shift_per_flow[NUMFB]
      for ( int j=0; j<NUMFB; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.t_mid_nuc_shift_per_flow[j];
        i_param++;
      }
      // float Concentration[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.C[j];
        i_param++;
      }
      //valve_open
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.valve_open;
      i_param++;
      //nuc_flow_span
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.nuc_flow_span;
      i_param++;
      //magic_divisor_for_timing
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param, iBlk ) = rp.nuc_shape.magic_divisor_for_timing;
      i_param++;
    }
  }
}

void extern_links::SpecificRegionParamDump ( reg_params &rp, int region_ndx, int iBlk )
{
  // EXPLICIT DUMP OF REGION PARAMS BY NAME
  // float casts are toxic and obscure what we're actually dumping

  int e_param = 0;
  DumpRegionalEnzymatics ( rp, region_ndx, iBlk, e_param );
  int b_param = 0;
  DumpRegionalBuffering ( rp, region_ndx, iBlk, b_param );
 int ns_param = 0;
  DumpRegionNucShape ( rp, region_ndx, iBlk, ns_param );

  // tshift
  int i_param=0;
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = rp.tshift;
  i_param++;


  // CopyDrift
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = rp.CopyDrift;
  i_param++;
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = COPYMULTIPLIER;
  i_param++;

  // float darkness[NUMFB]
  for ( int j=0; j<NUMFB; j++ )
  {
    mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = rp.darkness[j];
    i_param++;
  }
  // sens as used?
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = rp.sens;
  i_param++;
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = SENSMULTIPLIER;
  i_param++;
  // sens
  mPtrs->m_regional_param->At ( region_ndx, i_param, iBlk ) = rp.molecules_to_micromolar_conversion;
  i_param++;
 }

void extern_links::DumpRegionFitParamsH5 ( int region_ndx, int flow, reg_params &rp )
{
  if ( ( mPtrs->m_regional_param !=NULL ) & ( mPtrs->m_derived_param!=NULL ) )
  {

    int iBlk = CurComputeBlock ( flow );

    ///------------------------------------------------------------------------------------------------------------
    /// regionalParams, per compute block
    ///------------------------------------------------------------------------------------------------------------

    SpecificRegionParamDump ( rp, region_ndx, iBlk );


    ///------------------------------------------------------------------------------------------------------------
    /// regionalParamsExtra, per compute block
    ///------------------------------------------------------------------------------------------------------------
    // This is allowed here because they are "derived" parmaeters associated with the FitParams and not individual data blocks
    // although I may regret this
    mPtrs->m_derived_param->At ( region_ndx, 0 ,iBlk ) = GetModifiedMidNucTime ( &rp.nuc_shape,TNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 1 ,iBlk ) = GetModifiedMidNucTime ( &rp.nuc_shape,ANUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 2 ,iBlk ) = GetModifiedMidNucTime ( &rp.nuc_shape,CNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 3 ,iBlk ) = GetModifiedMidNucTime ( &rp.nuc_shape,GNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 4 ,iBlk ) = GetModifiedSigma ( &rp.nuc_shape,TNUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 5 ,iBlk ) = GetModifiedSigma ( &rp.nuc_shape,ANUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 6 ,iBlk ) = GetModifiedSigma ( &rp.nuc_shape,CNUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 7 ,iBlk ) = GetModifiedSigma ( &rp.nuc_shape,GNUCINDEX );

  }
}

//
void extern_links::WriteDebugBeadToRegionH5 ( RegionalizedData *my_region_data )
{
  if ( mPtrs->m_region_debug_bead!=NULL )
  {
    // find my debug-bead
    int ibd=my_region_data->my_beads.DEBUG_BEAD;
    int region_ndx = my_region_data->region->index;

    if ( ibd<my_region_data->my_beads.numLBeads )
    {
      bead_params *p;
      p = &my_region_data->my_beads.params_nn[ibd];

      // indexing by region, debug bead(s), parameter
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,0,0,p->Copies );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,1,0,p->R );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,2,0,p->dmult );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,3,0,p->gain );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead, region_ndx,4,0,my_region_data->my_trace.t0_map[p->x+p->y*my_region_data->region->w] ); // one of these things is not like the others
    }
  }
  if ( mPtrs->m_region_debug_bead_ak!=NULL )
  {
    // find my debug-bead
    int ibd=my_region_data->my_beads.DEBUG_BEAD;
    int region_ndx = my_region_data->region->index;

    if ( ibd<my_region_data->my_beads.numLBeads )
    {
      bead_params *p;
      p = &my_region_data->my_beads.params_nn[ibd];
      // I may regret this decision, but I'm putting amplitude/kmult as the axis for this bead
      // which assumes exactly one debug bead per region
      for ( int iFlowBuffer=0; iFlowBuffer<my_region_data->my_flow.flowBufferCount; iFlowBuffer++ )
      {
        int flow = my_region_data->my_flow.buff_flow[iFlowBuffer];
        mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_ak,region_ndx,0,flow,p->Ampl[iFlowBuffer] );
        mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_ak,region_ndx,1,flow,p->kmult[iFlowBuffer] );
      }
    }
  }
  if ( mPtrs->m_region_debug_bead_location!=NULL )
  {
    // find my debug-bead
    int ibd=my_region_data->my_beads.DEBUG_BEAD;
    int region_ndx = my_region_data->region->index;

    if ( ibd<my_region_data->my_beads.numLBeads )
    {
      bead_params *p;
      p = &my_region_data->my_beads.params_nn[ibd];


      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_location,region_ndx,0,0,p->x+my_region_data->region->col );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_location,region_ndx,1,0,p->y+my_region_data->region->row );

    }

  }
}

void extern_links::DumpTimeCompressionH5 ( int reg, TimeCompression &time_c )
{
  if ( mPtrs->m_time_compression!=NULL )
  {
    int npts = time_c.npts();
    //@TODO: copy/paste bad, but I have 3 different operations for 4 variables
    // frameNumber
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,0,j,time_c.frameNumber[j] );
    for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,0,j,0.0f ); // pad 0's
    //deltaFrame
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,1,j,time_c.deltaFrame[j] );
    for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,1,j,0.0f ); // pad 0's
    //frames_per_point
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,2,j, ( float ) time_c.frames_per_point[j] );
    for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,2,j,0.0f ); // pad 0's
    // npts = which time points are weighted
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,3,j,1.0f );
    for ( int j=npts; j<MAX_COMPRESSED_FRAMES; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,3,j,0.0f ); // pad 0's
  }
}

void extern_links::WriteRegionParametersToDataCubes ( RegionalizedData *my_region_data )
{
  if ( mPtrs!=NULL ) // guard against nothing at all desired
  {
    // mPtrs allows us to put the appropriate parameters directly to the data cubes
    // and then the hdf5 routine can decide when to flush
    DumpTimeAndEmphasisByRegionH5 ( my_region_data->region->index,my_region_data->time_c, my_region_data->emphasis_data );
    DumpEmptyTraceH5 ( my_region_data->region->index,*my_region_data );
    DumpRegionFitParamsH5 ( my_region_data->region->index, my_region_data->my_flow.buff_flow[my_region_data->my_flow.flowBufferCount-1], my_region_data->my_regions.rp );
    WriteDebugBeadToRegionH5 ( my_region_data );
    // should be done only at the first compute block
    DumpDarkMatterH5 ( my_region_data->region->index, my_region_data->time_c, my_region_data->my_regions );
    DumpDarknessH5 ( my_region_data->region->index, my_region_data->my_regions.rp );
    DumpRegionInitH5 ( my_region_data->region->index, *my_region_data );
    DumpRegionOffsetH5 ( my_region_data->region->index, my_region_data->region->col, my_region_data->region->row );
    DumpTimeCompressionH5 ( my_region_data->region->index, my_region_data->time_c );
  }
}


