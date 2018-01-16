/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "GlobalWriter.h"
#include <assert.h>
#include "LinuxCompat.h"
#include "RawWells.h"
#include "TimeCompression.h"
#include "EmphasisVector.h"
#include "BkgDataPointers.h"
#include "BkgTrace.h"
#include "RegionalizedData.h"
#include "SlicedChipExtras.h"

void GlobalWriter::AllocDataCubeResErr(int col, int row, int flow){
  int w = col; //= region_data.region->w + region_data.get_region_col();
  int h = row; //region_data.region->h + region_data.get_region_row();
  printf("allocating datacube  %d %d %d \n", w, h, flow);
  mPtrs->mResError->Init(w, h, flow);
  mPtrs->mResError->AllocateBuffer();
}
void GlobalWriter::WriteOneBeadToDataCubes ( BeadParams &p, int flow, int iFlowBuffer, int last, int ibd, Region *region, BkgTrace &my_trace, int flow_block_size, int flow_block_id )
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
    mPtrs->copyCube_element ( mPtrs->mBeadInitParam,x,y,4,my_trace.t0_map[p.x+p.y*region->w] ); // one of these things is not like the others
  }

  if ( ( iFlowBuffer+1 ) == flow_block_size || last ) // per compute block, not per flow
  {
    if ( mPtrs->mBeadFblk_avgErr!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_avgErr,x,y,flow_block_id,p.my_state->avg_err );
    if ( mPtrs->mBeadFblk_clonal!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_clonal,x,y,flow_block_id,p.my_state->clonal_read?1:0 );
    if ( mPtrs->mBeadFblk_corrupt!=NULL )
      mPtrs->copyCube_element ( mPtrs->mBeadFblk_corrupt,x,y,flow_block_id,p.my_state->corrupt?1:0 );
  }
}


//@TODO: this is not actually a bkgmodel function but a function of my_beads?
void GlobalWriter::WriteBeadParameterstoDataCubes ( int iFlowBuffer, bool last,Region *region, BeadTracker &my_beads, FlowBufferInfo &my_flow, BkgTrace &my_trace, int flow_block_id, int flow_block_start )
{
  if ( mPtrs == NULL )
    return;


  int flow = flow_block_start + iFlowBuffer;
  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    struct BeadParams &p = my_beads.params_nn[ibd];
    WriteOneBeadToDataCubes ( p, flow, iFlowBuffer,  last, ibd, region, my_trace, my_flow.GetMaxFlowCount(), flow_block_id );
  }
}


void GlobalWriter::SendErrorVectorToHDF5 ( BeadParams *p, error_track &err_t,
 Region *region, FlowBufferInfo &my_flow, int flow_block_start )
{   
    if ( mPtrs->mResError != NULL )
    {
      int x = p->x + region->col;
      int y = p->y + region->row;
      if ( mPtrs->mResError!=NULL )
      {
        for ( int fnum=0; fnum<my_flow.flowBufferCount; fnum++ )
        {
          mPtrs->copyCube_element ( mPtrs->mResError,x,y,fnum + flow_block_start,err_t.mean_residual_error[fnum] );
        }
      }
    }
}

void GlobalWriter::SendErrorVectorToWells( BeadParams *p, error_track &err_t,
		 Region *region, FlowBufferInfo &my_flow, int flow_block_start ){
	if ( mPtrs->mResError1 != NULL )
	    {
	      int x = p->x + region->col;
	      int y = p->y + region->row;
	      if ( mPtrs->mResError1!=NULL )
	      {
	        for ( int fnum=0; fnum<my_flow.flowBufferCount; fnum++ )
	        {
	          mPtrs->copyCube_element ( mPtrs->mResError1, x, y, fnum + flow_block_start,err_t.mean_residual_error[fnum] );
	        }
	      }
	    }
}


void GlobalWriter::SendPredictedToHDF5 ( int ibd, float *block_signal_predicted, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
    if ( ( mPtrs->m_region_debug_bead_predicted!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
      int reg = my_region_data.region->index;
      //printf("GlobalWriter::SendPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
      for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
      {
        int flow= flow_block_start + fnum;
        for ( int j=0; j<npts; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_predicted,reg,j,flow,block_signal_predicted[j+fnum*npts] );
        for ( int j=npts; j<max_frames; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_predicted,reg,j,flow,0 ); // pad 0's
      }
    }
}

void GlobalWriter::SendCorrectedToHDF5 ( int ibd, float *block_signal_corrected, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
    if ( ( mPtrs->m_region_debug_bead_corrected!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
      int reg = my_region_data.region->index;
      //printf("GlobalWriter::SendCorrectedToHDF5... (r,b)=(%d,%d) corrected[10]=%f\n",reg,ibd,block_signal_corrected[10]);
      for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
      {
        int flow= flow_block_start + fnum;
        for ( int j=0; j<npts; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_corrected,reg,j,flow,block_signal_corrected[j+fnum*npts] );
        for ( int j=npts; j<max_frames; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_corrected,reg,j,flow,0 ); // pad 0's
      }
    }
}


void GlobalWriter::SendBestRegion_PredictedToHDF5 (int ibd, float *block_signal_predicted, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_predicted!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        //printf("GlobalWriter::SendBestRegionPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_predicted,ibd,j,flow,block_signal_predicted[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_predicted,ibd,j,flow,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendBestRegion_CorrectedToHDF5 (int ibd, float *block_signal_corrected, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_corrected!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        //printf("GlobalWriter::SendBestRegionCorrectedToHDF5... (r,b)=(%d,%d) corrected[10]=%f\n",reg,ibd,block_signal_corrected[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_corrected,ibd,j,flow,block_signal_corrected[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_corrected,ibd,j,flow,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendBestRegion_OriginalToHDF5 (int ibd, float *block_signal_original, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_original!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        //printf("GlobalWriter::SendBestRegionPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_original,ibd,j,flow,block_signal_original[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_original,ibd,j,flow,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendBestRegion_SBGToHDF5 (int ibd, float *block_signal_sbg, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_sbg!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        //printf("GlobalWriter::SendBestRegionPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_sbg,ibd,j,flow,block_signal_sbg[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_sbg,ibd,j,flow,0 ); // pad 0's
        }
      }
}

void GlobalWriter::SendBestRegion_LocationToHDF5 (int ibd, RegionalizedData &my_region_data )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_location!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //int x = p->x+my_region_data.region->col;
          //int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendBestRegionLocationToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_location,ibd,0,0,p->y+my_region_data.region->row );
          mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_location,ibd,1,0,p->x+my_region_data.region->col );
      }
}


void GlobalWriter::SendBestRegion_GainSensToHDF5 (int ibd, RegionalizedData &my_region_data )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_gainSens!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
		  reg_params *reg_p =  &my_region_data.my_regions.rp;
		  float gainSens = p->gain * reg_p->sens;
          mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_gainSens,ibd,0,0,gainSens);
      }
}


void GlobalWriter::SendBestRegion_DmultToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_dmult!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_dmult,ibd,0,0,p->dmult);
      }
}


void GlobalWriter::SendBestRegion_SPToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
      if (( mPtrs->m_beads_bestRegion_SP!=NULL ) )
      {
		BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
    reg_params *reg_p =  &my_region_data.my_regions.rp;
		int flow= flow_block_start;

    float SP = (float) (COPYMULTIPLIER * p->Copies) *reg_p->CalculateCopyDrift(flow);
		mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_SP,ibd,0,0,SP );
      }
}


void GlobalWriter::SendBestRegion_RToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_R!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_R,ibd,0,0,p->R );
       }
}


void GlobalWriter::SendBestRegion_AmplitudeToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_amplitude!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            // val in 1.wells, as in GlobalWriter::WriteAnswersToWells()
            // val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_beads.params_nn[ibd].Copies * my_regions->rp.copy_multiplier[iFlowBuffer];
            //float val = p->Ampl[fnum] * p->Copies * my_region_data.my_regions.rp.copy_multiplier[fnum];
            //mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_amplitude,ibd,0,flow,p->Ampl[fnum] * p->Copies * my_region_data.my_regions.rp.copy_multiplier[fnum]);
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_amplitude,ibd,0,flow,p->Ampl[fnum]);
          }
      }
}

// note like amplitude this is a >per flow< fitted value
void GlobalWriter::SendBestRegion_KmultToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_kmult!=NULL ) )
      {
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
             mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_kmult,ibd,0,flow,p->kmult[fnum]);
          }
        }
}


void GlobalWriter::SendBestRegion_ResidualToHDF5 (int ibd, error_track &err_t, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_residual!=NULL ) )
      {
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_residual,ibd,0,flow,err_t.mean_residual_error[fnum]);
          }
      }
}


void GlobalWriter::SendBestRegion_FitType_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int fitType[], int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_fittype!=NULL ) )
      {
          //BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_fittype,ibd,0,flow,fitType[fnum] );
          }
      }
}

void GlobalWriter::SendBestRegion_Converged_ToHDF5 (int ibd,  RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t,int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_converged!=NULL ) )
      {
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_converged,ibd,0,flow,err_t.converged[fnum]?1:0 );
          }
      }
}


void GlobalWriter::SendBestRegion_TaubToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
      if (( mPtrs->m_beads_bestRegion_taub!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_taub,ibd,0,flow,err_t.tauB[fnum]);

          }
      }
}

void GlobalWriter::SendBestRegion_etbRToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
      if (( mPtrs->m_beads_bestRegion_etbR!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_etbR,ibd,0,flow,err_t.etbR[fnum]);

          }
      }
}


void GlobalWriter::SendBestRegion_InitAkToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
      if (( mPtrs->m_beads_bestRegion_initAk!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              // store A & kmult starting point
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_initAk,ibd,0,flow,err_t.initA[fnum]);
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_initAk,ibd,1,flow,err_t.initkmult[fnum]);

          }
      }
}

void GlobalWriter::SendBestRegion_TMS_ToHDF5 ( int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t,int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_tms!=NULL ) )
      {
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_tms,ibd,0,flow,err_t.t_mid_nuc_actual[fnum] );
            mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_tms,ibd,1,flow,err_t.t_sigma_actual[fnum]);
           }
      }
}


void GlobalWriter::SendBestRegion_BkgLeakageToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
      if (( mPtrs->m_beads_bestRegion_bkg_leakage!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_bkg_leakage,ibd,0,flow,err_t.bkg_leakage[fnum]);

          }
      }
}


void GlobalWriter::SendBestRegion_TimeframeToHDF5 (RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_bestRegion_timeframe!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        //printf("GlobalWriter::SendBestRegionPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_timeframe,0,j,0,my_region_data.time_c.frameNumber[j] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_bestRegion_timeframe,0,j,0,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendRegionSamples_PredictedToHDF5 (int ibd, float *block_signal_predicted, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_predicted!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        int reg = my_region_data.region->index;
        reg = reg*nSampleOut + get_sampleIndex();
        //printf("GlobalWriter::SendRegionSamplesPredictedToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_predicted,reg,j,flow,block_signal_predicted[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_predicted,reg,j,flow,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendRegionSamples_CorrectedToHDF5 (int ibd, float *block_signal_corrected, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_corrected!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        int reg = my_region_data.region->index;
        reg = reg*nSampleOut + get_sampleIndex();
        //printf("GlobalWriter::SendRegionSamplesCorrectedToHDF5... (r,b)=(%d,%d) corrected[10]=%f\n",reg,ibd,block_signal_corrected[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_corrected,reg,j,flow,block_signal_corrected[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_corrected,reg,j,flow,0 ); // pad 0's
        }
      }
}


void GlobalWriter::SendRegionSamples_OriginalToHDF5 (int ibd, float *block_signal_original, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_original!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        int reg = my_region_data.region->index;
        reg = reg*nSampleOut + get_sampleIndex();
        //printf("GlobalWriter::SendRegionSamplesCorrectedToHDF5... (r,b)=(%d,%d) corrected[10]=%f\n",reg,ibd,block_signal_corrected[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_original,reg,j,flow,block_signal_original[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_original,reg,j,flow,0 ); // pad 0's
        }
      }
}

void GlobalWriter::SendRegionSamples_SBGToHDF5 (int ibd, float *block_signal_sbg, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_sbg!=NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        int reg = my_region_data.region->index;
        reg = reg*nSampleOut + get_sampleIndex();
        //printf("GlobalWriter::SendRegionSamplesCorrectedToHDF5... (r,b)=(%d,%d) corrected[10]=%f\n",reg,ibd,block_signal_corrected[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          for ( int j=0; j<npts; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_sbg,reg,j,flow,block_signal_sbg[j+fnum*npts] );
          for ( int j=npts; j<max_frames; j++ )
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_sbg,reg,j,flow,0 ); // pad 0's
        }
      }
}


//@TODO: this is wrong: timeframe varies by region on the chip, therefore must be stored per bead sampled

void GlobalWriter::SendRegionSamples_TimeframeToHDF5 (RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_timeframe!=NULL ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
      int reg = my_region_data.region->index;
      reg = reg*nSampleOut + get_sampleIndex();
      // only first flow counts as we keep the same time compression over time
      // match beads so as to make sure the traces have an easy partner
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_timeframe,reg,j,0,my_region_data.time_c.frameNumber[j] );
    for ( int j=npts; j<max_frames; j++ )
      mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_timeframe,reg,j,0,0 ); // pad 0's
    }
}




void GlobalWriter::SendRegionSamples_LocationToHDF5 (int ibd, RegionalizedData &my_region_data)
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_location!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg0 = my_region_data.region->index;
          int reg = reg0*nSampleOut + get_sampleIndex();
          int y = p->y+my_region_data.region->row;
          int x = p->x+my_region_data.region->col;
          //printf("GlobalWriter::SendRegionSamplesLocationToHDF5... reg=%d ibd=%d y=%d x=%d\n",reg,ibd,y,x);
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_location,reg,0,0,y );
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_location,reg,1,0,x );
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_location,reg,2,0,reg0 );
      }
}


void GlobalWriter::SendRegionSamples_GainSensToHDF5 (int ibd, RegionalizedData &my_region_data )
{
    //if ( ! hasPointers() ) return;
    if (( mPtrs->m_beads_regionSamples_gainSens!=NULL ) )
    {
      BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
      reg_params *reg_p =  &my_region_data.my_regions.rp;
      int reg = my_region_data.region->index;
      reg = reg*nSampleOut + get_sampleIndex();
      float gainSens = p->gain * reg_p->sens;
      mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_gainSens,reg,0,0,gainSens);
    }
}



void GlobalWriter::SendRegionSamples_DmultToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_dmult!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
		  mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_dmult,reg,0,0,p->dmult);
      }
}


void GlobalWriter::SendRegionSamples_SPToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_SP!=NULL ) )
      {
		BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
		reg_params *reg_p =  &my_region_data.my_regions.rp;
		int reg = my_region_data.region->index;
		reg = reg*nSampleOut + get_sampleIndex();
		int flow= flow_block_start;
		//int flow= flow_block_start + fnum;
		//float SP = ( float ) ( COPYMULTIPLIER * p->Copies ) * reg_p->copy_multiplier[fnum];
		//float SP = ( float ) ( p->Copies ) * reg_p->copy_multiplier[fnum];
		float SP = (float) (COPYMULTIPLIER * p->Copies) *pow (reg_p->CopyDrift,flow); // flow=0
		mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_SP,reg,0,0,SP );
      }
}


void GlobalWriter::SendRegionSamples_RToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_R!=NULL ) )
      {
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        //reg_params *reg_p =  &my_region_data.my_regions.rp;
        int reg = my_region_data.region->index;
        reg = reg*nSampleOut + get_sampleIndex();
		mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_R,reg,0,0,p->R );
	  }
}


void GlobalWriter::SendRegionSamples_RegionParamsToHDF5 (int ibd, RegionalizedData &my_region_data )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_regionParams!=NULL ) )
      {
          reg_params *reg_p =  &my_region_data.my_regions.rp;
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          int n = 0;
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,p->gain);
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,reg_p->sens );
          mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,(float)(COPYMULTIPLIER*p->Copies) );
          for (int i=0; i<NUMNUC; i++)
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,reg_p->kmax[i] );
          for (int i=0; i<NUMNUC; i++)
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,reg_p->krate[i] );
          for (int i=0; i<NUMNUC; i++)
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_regionParams,reg,n++,0,reg_p->d[i] );
      }
}


void GlobalWriter::SendRegionSamples_AmplitudeToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_amplitude!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
             mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_amplitude,reg,0,flow,p->Ampl[fnum]);
          }
      }
}


void GlobalWriter::SendRegionSamples_KmultToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_kmult!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
             mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_kmult,reg,0,flow,p->kmult[fnum]);
          }
      }
}

void GlobalWriter::SendRegionSamples_ResidualToHDF5 (int ibd, error_track &err_t, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_residual!=NULL ) )
      {
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_residual,reg,0,flow,err_t.mean_residual_error[fnum]);
          }
      }
}


void GlobalWriter::SendRegionSamples_FitType_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int fitType[], int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_fittype!=NULL ) )
      {
          //BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_fittype,reg,0,flow,fitType[fnum] );
          }
      }
}

void GlobalWriter::SendRegionSamples_Converged_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_converged!=NULL ) )
      {
          //BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_converged,reg,0,flow,err_t.converged?1:0 );
          }
      }
}

void GlobalWriter::SendRegionSamples_BkgLeakageToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_bkg_leakage!=NULL ) )
      {
          //BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_bkg_leakage,reg,0,flow,err_t.bkg_leakage[fnum] );
          }
      }
}


void GlobalWriter::SendRegionSamples_InitAkToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_initAk!=NULL ) )
      {
          //BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          //reg_params *reg_p =  &my_region_data.my_regions.rp;
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_initAk,reg,0,flow,err_t.initA[fnum] );
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_initAk,reg,1,flow,err_t.initkmult[fnum] );
           }
      }
}


void GlobalWriter::SendRegionSamples_TMS_ToHDF5 ( RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras,int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_tms!=NULL ) )
      {

          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_tms,reg,0,flow,my_region_data.my_regions.cache_step.t_mid_nuc_actual[fnum] );
            mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_tms,reg,1,flow,my_region_data.my_regions.cache_step.t_sigma_actual[fnum]);
           }
      }
}


void GlobalWriter::SendRegionSamples_TaubToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_taub!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_taub,reg,0,flow,err_t.tauB[fnum]);

          }

      }
}

void GlobalWriter::SendRegionSamples_etbRToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_regionSamples_etbR!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int reg = my_region_data.region->index;
          reg = reg*nSampleOut + get_sampleIndex();
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
              int flow= flow_block_start + fnum;
              mPtrs->copyCube_element ( mPtrs->m_beads_regionSamples_etbR,reg,0,flow,err_t.etbR[fnum]);

          }

      }
}


///====================Xyflow=================================================================================================================================
void GlobalWriter::SendXyflow_Predicted_ToHDF5 (int ibd, float *block_signal_predicted, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_predicted != NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        int x = p->x+my_region_data.region->col;
        int y = p->y+my_region_data.region->row;
        //printf("GlobalWriter::SendPredicted_xyflow_ToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          int ibd_select = select_xyflow(x,y,flow);
          if (ibd_select>=0) {
              for ( int j=0; j<npts; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_predicted,ibd_select,j,0,block_signal_predicted[j+fnum*npts] );
              for ( int j=npts; j<max_frames; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_predicted,ibd_select,j,0,0 ); // pad 0's
          }
        }
      }
    }

	
void GlobalWriter::SendXyflow_Location_Keys_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras &my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_location_keys!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          int ibd_select = select_xy(x,y);
          if (ibd_select>=0) // only for regions selected
          {
              //printf("GlobalWriter::SendLocation_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
              for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
              {
                int flow= flow_block_start + fnum;
                //if (flow==0 || flow==2 || flow==5 || flow==7)
                if (flow==0) // only do it once out of the 4 key flows
                    {
                    //int ibd_select = select_xy(x,y);
                    //if (ibd_select>=0)
                    {
                      mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_location_keys,ibd_select,0,0,y );
                      mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_location_keys,ibd_select,1,0,x );
                      //if ((x==13 && y==11 && flow==50) || (y==13 && x==11 && flow==50))
                      //std::cerr << "SendLocation_xyflow_ToHDF5... x=" << x << " y=" << y << " flow=" << flow << " ibd_select=" << ibd_select << std::endl << std::flush;
                    }
                }
              }
          }
      }
}


void GlobalWriter::SendXyflow_Predicted_Keys_ToHDF5 (int ibd, float *block_signal_predicted, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_predicted_keys != NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        int x = p->x+my_region_data.region->col;
        int y = p->y+my_region_data.region->row;
        int ibd_select = select_xy(x,y);
        if (ibd_select>=0) // only for regions selected
        {
            //printf("GlobalWriter::SendPredicted_xyflow_ToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
            for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
            {
              int flow= flow_block_start + fnum;
              if (flow==0 || flow==2 || flow==5 || flow==7)
                  {
                  //int ibd_select = select_xy(x,y);
                  //if (ibd_select>=0)
                  {
                      int k = 3;
                      if (flow==0) k=0;
                      else if (flow==2) k=1;
                      else if (flow==5) k=2;
                      for ( int j=0; j<npts; j++ )
                          mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_predicted_keys,ibd_select,j,k,block_signal_predicted[j+fnum*npts] );
                      for ( int j=npts; j<max_frames; j++ )
                          mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_predicted_keys,ibd_select,j,k,0 ); // pad 0's
                  }
              }
            }
        }
      }
}


void GlobalWriter::SendXyflow_Corrected_Keys_ToHDF5 (int ibd, float *block_signal, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_corrected_keys != NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        int x = p->x+my_region_data.region->col;
        int y = p->y+my_region_data.region->row;
        int ibd_select = select_xy(x,y);
        if (ibd_select>=0) // only for regions selected
        {
            //printf("GlobalWriter::SendPredicted_xyflow_ToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
            for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
            {
              int flow= flow_block_start + fnum;
              if (flow==0 || flow==2 || flow==5 || flow==7)
                  {
                  //int ibd_select = select_xy(x,y);
                  //if (ibd_select>=0)
                  {
                      int k = 3;
                      if (flow==0) k=0;
                      else if (flow==2) k=1;
                      else if (flow==5) k=2;
                      for ( int j=0; j<npts; j++ )
                          mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_corrected_keys,ibd_select,j,k,block_signal[j+fnum*npts] );
                      for ( int j=npts; j<max_frames; j++ )
                          mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_corrected_keys,ibd_select,j,k,0 ); // pad 0's
                  }
              }
            }
        }
      }
}



void GlobalWriter::SendXyflow_Corrected_ToHDF5 (int ibd, float *block_signal, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_corrected != NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        int x = p->x+my_region_data.region->col;
        int y = p->y+my_region_data.region->row;
        //printf("GlobalWriter::SendCorrected_xyflow_ToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0)
            {
                for ( int j=0; j<npts; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_corrected,ibd_select,j,0,block_signal[j+fnum*npts] );
                for ( int j=npts; j<max_frames; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_corrected,ibd_select,j,0,0 ); // pad 0's
            }
        }
      }
}


void GlobalWriter::SendXyflow_Amplitude_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_amplitude!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
                // val in 1.wells, as in GlobalWriter::WriteAnswersToWells()
                // val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_beads.params_nn[ibd].Copies * my_regions->rp.copy_multiplier[iFlowBuffer];
                //float val = p->Ampl[fnum] * p->Copies * my_region_data.my_regions.rp.copy_multiplier[fnum];
                //mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_amplitude,ibd_select,0,0,p->Ampl[fnum] * p->Copies * my_region_data.my_regions.rp.copy_multiplier[fnum]);
                mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_amplitude,ibd_select,0,0,p->Ampl[fnum]);
            }
          }
      }
}


void GlobalWriter::SendXyflow_Residual_ToHDF5 (int ibd, error_track &err_t, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_residual!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
                mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_residual,ibd_select,0,0,err_t.mean_residual_error[fnum]);
            }
          }
      }
}



void GlobalWriter::SendXyflow_Kmult_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_kmult!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_kmult,ibd_select,0,0,p->kmult[fnum] );
            }
          }
      }
}



void GlobalWriter::SendXyflow_Dmult_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_dmult!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_dmult,ibd_select,0,0,p->dmult );
            }
          }
      }
}



void GlobalWriter::SendXyflow_Taub_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, error_track &err_t, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_taub!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
                //cerr << "SendXyflow_Taub_ToHDF5... x=" << x << " y=" << y << " flow=" << flow << "fnum=" << fnum << "taub=" << p->tauB[fnum] << endl << flush;
                mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_taub,ibd_select,0,0,err_t.tauB[fnum] );

            }
          }
      }
}


void GlobalWriter::SendXyflow_SP_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_SP!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          reg_params *reg_p =  &my_region_data.my_regions.rp;
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
                //float SP = ( float ) ( COPYMULTIPLIER * p->Copies ) * reg_p->copy_multiplier[fnum];
                //float SP = ( float ) ( p->Copies ) * reg_p->copy_multiplier[fnum];
                float SP = (float) (COPYMULTIPLIER * p->Copies) *pow (reg_p->CopyDrift,flow);
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_SP,ibd_select,0,0,SP );
            }
          }
      }
}



void GlobalWriter::SendXyflow_R_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_R!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_R,ibd_select,0,0,p->R );
            }
          }
      }
}



void GlobalWriter::SendXyflow_GainSens_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_gainSens!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendAmplitude_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
                reg_params *reg_p =  &my_region_data.my_regions.rp;
                float gainSens = p->gain * reg_p->sens;
                mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_gainSens,ibd_select,0,0,gainSens );
            }
          }
      }
}


void GlobalWriter::SendXyflow_Timeframe_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int max_frames, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_timeframe != NULL ) )
      {
        int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
        //int reg = my_region_data.region->index;
        BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
        int x = p->x+my_region_data.region->col;
        int y = p->y+my_region_data.region->row;
        //printf("GlobalWriter::SendPredicted_xyflow_ToHDF5... (r,b)=(%d,%d) predicted[10]=%f\n",reg,ibd,block_signal_predicted[10]);
        for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
        {
          int flow= flow_block_start + fnum;
          int ibd_select = select_xyflow(x,y,flow);
          if (ibd_select>=0) {
              for ( int j=0; j<npts; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_timeframe,ibd_select,j,0,my_region_data.time_c.frameNumber[j] );
              for ( int j=npts; j<max_frames; j++ )
                  mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_timeframe,ibd_select,j,0,0 ); // pad 0's
          }
        }
      }
}




void GlobalWriter::SendXyflow_Location_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_location!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendLocation_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_location,ibd_select,0,0,y );
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_location,ibd_select,1,0,x );
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_location,ibd_select,2,0,flow );
              //if ((x==13 && y==11 && flow==50) || (y==13 && x==11 && flow==50))
              //std::cerr << "SendLocation_xyflow_ToHDF5... x=" << x << " y=" << y << " flow=" << flow << " ibd_select=" << ibd_select << std::endl << std::flush;
            }
          }
      }
}



void GlobalWriter::SendXyflow_MM_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_mm!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              int mm = mm_xyflow(x,y,flow);
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_mm,ibd_select,0,0,mm);
            }
          }
      }
}


void GlobalWriter::SendXyflow_FitType_ToHDF5 (int ibd, RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras, int fitType[], int flow_block_start )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_fittype!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              //int mm = mm_xyflow(x,y,flow);
              mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_fittype,ibd_select,0,0,fitType[fnum]);
            }
          }
      }
}



void GlobalWriter::SendXyflow_HPlen_ToHDF5 (int ibd, RegionalizedData &my_region_data,
    SlicedChipExtras & extra,
    int flow_block_start  )
{
    //if ( ! hasPointers() ) return;
      if (( mPtrs->m_beads_xyflow_hplen!=NULL ) )
      {
          BeadParams *p= &my_region_data.my_beads.params_nn[ibd];
          int x = p->x+my_region_data.region->col;
          int y = p->y+my_region_data.region->row;
          //printf("GlobalWriter::SendHPlen_xyflow_ToHDF5... ibd=%d, region_x=%d, region_y=%d, x=%d, y=%d\n",ibd,my_region_data.region->col,my_region_data.region->row,x,y);
          for ( int fnum=0; fnum<extra.my_flow->flowBufferCount; fnum++ )
          {
            int flow= flow_block_start + fnum;
            int ibd_select = select_xyflow(x,y,flow);
            if (ibd_select>=0) {
              std::string hp = hp_xyflow(x,y,flow);
              //std::cerr << "SendHPlen_xyflow_ToHDF5...x,y,flow=" << x << "," << y << "," << flow << " ibd_select=" << ibd_select << " hp=" << hp << std::endl << std::flush;
              if (hp.length()>=2) {
                  char base = hp.at(hp.length()-1);
                  int nuc = 0;
                  switch (base) {
                      case 'A': nuc = 0; break;
                      case 'C': nuc = 1; break;
                      case 'G': nuc = 2; break;
                      case 'T': nuc = 3; break;
                      default:  nuc = -1;
                              //std::cerr << "SendHPlen_xyflow_ToHDF5 error: base " << base << " not T/A/C/G" << std::endl << std::flush;
                              //assert(nuc>=0 && nuc<4);
                              break;
                  }
                  int hplen = atoi(hp.substr(0,hp.length()-1).c_str());
                  //std::cerr << "SendHPlen_xyflow_ToHDF5...x,y,flow=" << x << "," << y << "," << flow << " ibd_select=" << ibd_select << " hp=" << hp << " nuc=" << nuc << " hplen=" << hplen << std::endl << std::flush;
                  if (nuc >=0 && nuc<=3 && hplen>=0) {
                      mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_hplen,ibd_select,0,0,nuc );
                      mPtrs->copyCube_element ( mPtrs->m_beads_xyflow_hplen,ibd_select,1,0,hplen );
              }
            }
          }
      }
    }
}


int GlobalWriter::select_xy(int x, int y)
{
    int ibd_select = -1;
    if ( mPtrs!=NULL && mPtrs->m_xyflow_hashtable!=NULL)
        ibd_select = mPtrs->id_xy(x,y,mPtrs->m_xyflow_hashtable);
    assert(ibd_select < mPtrs->m_xyflow_hashtable->size_xy());
    return ibd_select;
}


int GlobalWriter::select_xyflow(int x, int y, int flow)
{
    int ibd_select = -1;
    if ( mPtrs!=NULL && mPtrs->m_xyflow_hashtable!=NULL)
        ibd_select = mPtrs->id_xyflow(x,y,flow,mPtrs->m_xyflow_hashtable);
    assert(ibd_select < mPtrs->m_xyflow_hashtable->size());
    return ibd_select;
}


std::string GlobalWriter::hp_xyflow(int x, int y, int flow)
{
    std::string hp = "-1N";
    if ( mPtrs!=NULL && mPtrs->m_xyflow_hashtable!=NULL)
        hp = mPtrs->hp_xyflow(x,y,flow,mPtrs->m_xyflow_hashtable);
    return hp;
}


int GlobalWriter::mm_xyflow(int x, int y, int flow)
{
    int mm = -1;
    if ( mPtrs!=NULL && mPtrs->m_xyflow_hashtable!=NULL)
        mm = mPtrs->mm_xyflow(x,y,flow,mPtrs->m_xyflow_hashtable);
    return mm;
}


void GlobalWriter::SendXtalkToHDF5 ( int ibd, float *block_signal_xtalk, 
    RegionalizedData &my_region_data, SlicedChipExtras & my_region_data_extras,
    int max_frames , int flow_block_start
  )
{
    //if ( ! hasPointers() ) return;
    if ( ( mPtrs->m_region_debug_bead_xtalk!=NULL ) && ( ibd==my_region_data.my_beads.DEBUG_BEAD ) )
    {
      int npts = std::min ( my_region_data.time_c.npts(), max_frames ); // alloced max_frames
      int reg = my_region_data.region->index;
      for ( int fnum=0; fnum<my_region_data_extras.my_flow->flowBufferCount; fnum++ )
      {
        int flow = flow_block_start + fnum;
        for ( int j=0; j<npts; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_xtalk,reg,j,flow,block_signal_xtalk[j+fnum*npts] );
        for ( int j=npts; j<max_frames; j++ )
          mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_xtalk,reg,j,flow,0 ); // pad 0's
    }
  }
}


// this puts our answers into the data structures where they belong
// should be the only point of contact with the external world, but isn't
void GlobalWriter::WriteAnswersToWells ( int iFlowBuffer, Region *region, RegionTracker *my_regions, BeadTracker &my_beads, int flow_block_start )
{
  int flow = flow_block_start + iFlowBuffer;
  // make absolutely sure we're up to date
  my_regions->rp.copy_multiplier[iFlowBuffer] = my_regions->rp.CalculateCopyDrift ( flow );
  //Write one flow's data to 1.wells
  for ( int ibd=0;ibd < my_beads.numLBeads;ibd++ )
  {
    float val = my_beads.params_nn[ibd].Ampl[iFlowBuffer] * my_regions->rp.copy_multiplier[iFlowBuffer];
    if (false == rawWells->GetSaveAsUShort())
      val *= my_beads.params_nn[ibd].Copies;

	 int x = my_beads.params_nn[ibd].x+region->col;
    int y = my_beads.params_nn[ibd].y+region->row;

    if (my_beads.params_nn[ibd].my_state->pinned or my_beads.params_nn[ibd].my_state->corrupt)
      val = 0.0f; // actively suppress pinned wells in case we are so unfortunate as to have an estimate for them

    
    if(rawWells->GetSaveCopies() && rawWells->GetSaveRes())
    {
        float my_res = mPtrs->mResError1->At(x, y, flow);
        rawWells->WriteFlowgram ( flow, x, y, val, my_beads.params_nn[ibd].Copies, my_res );
    }
    else if (rawWells->GetSaveCopies())
    {
      rawWells->WriteFlowgram ( flow, x, y, val, my_beads.params_nn[ibd].Copies );
    }
    else if (rawWells->GetSaveRes())
    {
      float my_res = mPtrs->mResError1->At(x, y, flow);
      rawWells->WriteFlowgramWithRes ( flow, x, y, val, my_res );
    }
    else
    {
        rawWells->WriteFlowgram ( flow, x, y, val );
    }
  }
}

void GlobalWriter::DumpTimeAndEmphasisByRegionH5 ( int reg, TimeCompression &time_c, EmphasisClass &emphasis_data, int max_frames )
{
  if ( mPtrs->mEmphasisParam!=NULL )
  {

    ION_ASSERT ( emphasis_data.numEv <= MAX_POISSON_TABLE_COL, "emphasis_data.numEv > MAX_HPLEN+1" );
    //ION_ASSERT(time_c.npts <= max_frames, "time_c.npts > max_frames");
    int npts = std::min ( time_c.npts(), max_frames );

    if ( mPtrs->mEmphasisParam!=NULL )
    {
      // use copyCube_element to copy DataCube element to mEmphasisParam in BkgModel::DumpTimeAndEmphasisByRegionH5
      for ( int hp=0; hp<emphasis_data.numEv; hp++ )
      {
        for ( int t=0; t< npts; t++ )
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,emphasis_data.EmphasisVectorByHomopolymer[hp][t] );
        for ( int t=npts; t< max_frames; t++ )
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,0 ); // pad 0's, memset faster here?
      }

      for ( int hp=emphasis_data.numEv; hp<MAX_POISSON_TABLE_COL; hp++ )
      {
        for ( int t=0; t< max_frames; t++ )
        {
          mPtrs->copyCube_element ( mPtrs->mEmphasisParam,reg,hp,t,0 ); // pad 0's, memset faster here?
        }
      }
    }
  }
}

void GlobalWriter::DumpDarkMatterH5 ( int reg, TimeCompression &time_c, RegionTracker &my_reg_tracker, int max_frames )
{
  if ( mPtrs->mDarkOnceParam!=NULL )
  {
    int npts = time_c.npts();
    for ( int i=0; i<NUMNUC; i++ )
    {
      float *missingMass = my_reg_tracker.missing_mass.dark_nuc_comp[i];
      for ( int j=0; j<npts; j++ )
        mPtrs->copyCube_element ( mPtrs->mDarkOnceParam,reg,i,j,missingMass[j] );
      for ( int j=npts; j<max_frames; j++ )
        mPtrs->copyCube_element ( mPtrs->mDarkOnceParam,reg,i,j,0 ); // pad 0's
    }
  }
}

void GlobalWriter::DumpDarknessH5 ( int reg, reg_params &my_rp, int flow_block_size )
{
  ///------------------------------------------------------------------------------------------------------------
  /// darkness
  // use copyMatrix_element to copy Matrix element to mDarknessParam in BkgParamH5::DumpBkgModelRegionInfoH5
  ///------------------------------------------------------------------------------------------------------------
  if ( mPtrs->mDarknessParam!=NULL )
  {
    for ( int i=0; i<flow_block_size; i++ )
      mPtrs->copyCube_element ( mPtrs->mDarknessParam,reg,i,0,my_rp.darkness[i] );
  }
}

void GlobalWriter::DumpRegionInitH5 ( int reg, RegionalizedData &my_region_data )
{
  if ( mPtrs->mRegionInitParam!=NULL )
  {
    ///------------------------------------------------------------------------------------------------------------
    /// regionInitParam, only once at flow+1=flow_block_size
    ///------------------------------------------------------------------------------------------------------------
    mPtrs->copyCube_element ( mPtrs->mRegionInitParam,reg,0,0,my_region_data.t_mid_nuc_start );
    mPtrs->copyCube_element ( mPtrs->mRegionInitParam,reg,1,0,my_region_data.sigma_start );
  }
}

void GlobalWriter::DumpEmptyTraceH5 ( 
    int reg, 
    RegionalizedData &my_region_data,
    SlicedChipExtras *my_region_data_extras,
    int flow_block_start,
    bool last,
    int last_flow
  )
{
  if ( ( mPtrs->mEmptyOnceParam!=NULL ) & ( mPtrs->mBeadDC_bg!=NULL ) )
  {
    for ( int fnum=0; fnum<my_region_data_extras->my_flow->flowBufferCount; fnum++ )
    {
      int flow = flow_block_start + fnum;

      // Careful; we don't want to go past the last flow that we're supposed to write.
      if ( last && flow > last_flow ) continue;

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

      mPtrs->copyCube_element ( mPtrs->mBeadDC_bg,reg,0,flow,mt_trace->bg_dc_offset[fnum] );
    }
  }
}

void GlobalWriter::DumpRegionOffsetH5 ( int reg, int col, int row )
{
  if ( mPtrs->mRegionOffset!=NULL )
  {

    ///------------------------------------------------------------------------------------------------------------
    /// region_size, only once at flow+1=flow_block_size
    ///------------------------------------------------------------------------------------------------------------
    mPtrs->copyCube_element ( mPtrs->mRegionOffset,reg,0,0,col );
    mPtrs->copyCube_element ( mPtrs->mRegionOffset,reg,1,0,row );
  }
}

void GlobalWriter::DumpRegionalEnzymatics ( reg_params &rp, int region_ndx, int iBlk,int &i_param )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_enzymatics_param!=NULL )
    {  // float krate[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param++, iBlk ) = rp.krate[j];
      }
      // float d[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param++, iBlk ) = rp.d[j];
      }
      // float kmax[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_enzymatics_param->At ( region_ndx, i_param++, iBlk ) = rp.kmax[j];
      }
    }
  }
}

void GlobalWriter::DumpRegionalBuffering ( reg_params &rp, int region_ndx, int iBlk,int &i_param )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_buffering_param!=NULL )
    {
      mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.tau_R_m; // tau_R_m
      mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.tau_R_o; // tau_R_o
      mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.tauE;    // tauE
      mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.RatioDrift; // RatioDrift
      // float NucModifyRatio[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.NucModifyRatio[j];
      }
      mPtrs->m_buffering_param->At ( region_ndx, i_param++, iBlk ) = rp.reg_error; // reg_error
    }
  }
}

void GlobalWriter::DumpRegionNucShape ( reg_params &rp, int region_ndx, int iBlk,int &i_param, int flow_block_size )
{
  if ( mPtrs!=NULL )
  {
    if ( mPtrs->m_nuc_shape_param!=NULL )
    {
      for ( int j=0; j<flow_block_size; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.AccessTMidNuc()[j];
      }
      // float t_mid_nuc_delay[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.t_mid_nuc_delay[j];
      }
      // sigma
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.sigma;

      // float sigma_mult[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.sigma_mult[j];
      }
      for ( int j=0; j<flow_block_size; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.t_mid_nuc_shift_per_flow[j];
      }
      // float Concentration[NUMNUC]
      for ( int j=0; j<NUMNUC; j++ )
      {
        mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.C[j];
      }
      //valve_open
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.valve_open;
      //nuc_flow_span
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.nuc_flow_span;
       //magic_divisor_for_timing
      mPtrs->m_nuc_shape_param->At ( region_ndx, i_param++, iBlk ) = rp.nuc_shape.magic_divisor_for_timing;
    }
  }
}

void GlobalWriter::SpecificRegionParamDump ( reg_params &rp, int region_ndx, int iBlk, int flow_block_size )
{
  // EXPLICIT DUMP OF REGION PARAMS BY NAME
  // float casts are toxic and obscure what we're actually dumping

  int e_param = 0;
  DumpRegionalEnzymatics ( rp, region_ndx, iBlk, e_param );
  int b_param = 0;
  DumpRegionalBuffering ( rp, region_ndx, iBlk, b_param );
 int ns_param = 0;
  DumpRegionNucShape ( rp, region_ndx, iBlk, ns_param, flow_block_size );

  // tshift
  int i_param=0;
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = rp.tshift;


  // CopyDrift
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = rp.CopyDrift;
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = COPYMULTIPLIER;

  // float darkness[]
  for ( int j=0; j<flow_block_size; j++ )
  {
    mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = rp.darkness[j];
  }
  // sens as used?
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = rp.sens;
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = SENSMULTIPLIER;
  mPtrs->m_regional_param->At ( region_ndx, i_param++, iBlk ) = rp.molecules_to_micromolar_conversion;
 }

void GlobalWriter::DumpRegionFitParamsH5 ( int region_ndx, int flow, reg_params &rp, int flow_block_size, int flow_block_id )
{
  if ( ( mPtrs->m_regional_param !=NULL ) & ( mPtrs->m_derived_param!=NULL ) )
  {

    ///------------------------------------------------------------------------------------------------------------
    /// regionalParams, per compute block
    ///------------------------------------------------------------------------------------------------------------

    SpecificRegionParamDump ( rp, region_ndx, flow_block_id, flow_block_size );


    ///------------------------------------------------------------------------------------------------------------
    /// regionalParamsExtra, per compute block
    ///------------------------------------------------------------------------------------------------------------
    // This is allowed here because they are "derived" parmaeters associated with the FitParams and not individual data blocks
    // although I may regret this
    mPtrs->m_derived_param->At ( region_ndx, 0 ,flow_block_id ) = GetModifiedMidNucTime ( &rp.nuc_shape,TNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 1 ,flow_block_id ) = GetModifiedMidNucTime ( &rp.nuc_shape,ANUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 2 ,flow_block_id ) = GetModifiedMidNucTime ( &rp.nuc_shape,CNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 3 ,flow_block_id ) = GetModifiedMidNucTime ( &rp.nuc_shape,GNUCINDEX,0 );
    mPtrs->m_derived_param->At ( region_ndx, 4 ,flow_block_id ) = GetModifiedSigma ( &rp.nuc_shape,TNUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 5 ,flow_block_id ) = GetModifiedSigma ( &rp.nuc_shape,ANUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 6 ,flow_block_id ) = GetModifiedSigma ( &rp.nuc_shape,CNUCINDEX );
    mPtrs->m_derived_param->At ( region_ndx, 7 ,flow_block_id ) = GetModifiedSigma ( &rp.nuc_shape,GNUCINDEX );

  }
}

//
void GlobalWriter::WriteDebugBeadToRegionH5 ( 
    RegionalizedData *my_region_data,
    SlicedChipExtras *my_region_data_extras,
    int flow_block_start,
    bool last,
    int last_flow
  )
{
  if ( mPtrs->m_region_debug_bead!=NULL )
  {
    // find my debug-bead
    int ibd=my_region_data->my_beads.DEBUG_BEAD;
    int region_ndx = my_region_data->region->index;

    if ( ibd<my_region_data->my_beads.numLBeads )
    {
      BeadParams *p;
      p = &my_region_data->my_beads.params_nn[ibd];

      // indexing by region, debug bead(s), parameter
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,0,0,p->Copies );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,1,0,p->R );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,2,0,p->dmult );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,3,0,p->gain );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead,region_ndx,4,0,my_region_data->my_trace.t0_map[p->x+p->y*my_region_data->region->w] ); // one of these things is not like the others
    }
  }
  if ( mPtrs->m_region_debug_bead_ak!=NULL )
  {
    // find my debug-bead
    int ibd=my_region_data->my_beads.DEBUG_BEAD;
    int region_ndx = my_region_data->region->index;

    if ( ibd<my_region_data->my_beads.numLBeads )
    {
      BeadParams *p;
      p = &my_region_data->my_beads.params_nn[ibd];
      // I may regret this decision, but I'm putting amplitude/kmult as the axis for this bead
      // which assumes exactly one debug bead per region
      for ( int iFlowBuffer=0; iFlowBuffer<my_region_data_extras->my_flow->flowBufferCount; iFlowBuffer++ )
      {
        int flow = flow_block_start + iFlowBuffer;

        if ( last && flow > last_flow ) continue;

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
      BeadParams *p;
      p = &my_region_data->my_beads.params_nn[ibd];
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_location,region_ndx,0,0,p->x+my_region_data->region->col );
      mPtrs->copyCube_element ( mPtrs->m_region_debug_bead_location,region_ndx,1,0,p->y+my_region_data->region->row );

    }

  }
}


void GlobalWriter::DumpTimeCompressionH5 ( int reg, TimeCompression &time_c,  int max_frames )
{
  if ( mPtrs->m_time_compression!=NULL )
  {
    int npts = time_c.npts();
    //@TODO: copy/paste bad, but I have 3 different operations for 4 variables
    // frameNumber
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,0,j,time_c.frameNumber[j] );
    for ( int j=npts; j<max_frames; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,0,j,0.0f ); // pad 0's
    //deltaFrame
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,1,j,time_c.deltaFrame[j] );
    for ( int j=npts; j<max_frames; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,1,j,0.0f ); // pad 0's
    //frames_per_point
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,2,j, ( float ) time_c.frames_per_point[j] );
    for ( int j=npts; j<max_frames; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,2,j,0.0f ); // pad 0's
    // npts = which time points are weighted
    for ( int j=0; j<npts; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,3,j,1.0f );
    for ( int j=npts; j<max_frames; j++ )
      mPtrs->copyCube_element ( mPtrs->m_time_compression,reg,3,j,0.0f ); // pad 0's
  }
}

void GlobalWriter::WriteRegionParametersToDataCubes ( 
    RegionalizedData *my_region_data, 
    SlicedChipExtras *my_region_data_extras,
    int max_frames, int flow_block_size, int flow_block_id, int flow_block_start,
    bool last, int last_flow
  )
{
  if ( mPtrs!=NULL ) // guard against nothing at all desired
  {
    // mPtrs allows us to put the appropriate parameters directly to the data cubes
    // and then the hdf5 routine can decide when to flush
    DumpTimeAndEmphasisByRegionH5 ( my_region_data->region->index,my_region_data->time_c, my_region_data->emphasis_data, max_frames);
    DumpEmptyTraceH5 ( my_region_data->region->index,*my_region_data, my_region_data_extras, flow_block_start, last, last_flow );
    DumpRegionFitParamsH5 ( my_region_data->region->index, 
      flow_block_start + my_region_data_extras->my_flow->flowBufferCount-1,
      my_region_data->my_regions.rp, flow_block_size, flow_block_id );
    WriteDebugBeadToRegionH5 ( my_region_data, my_region_data_extras, flow_block_start, last, last_flow );
    // should be done only at the first compute block
    DumpDarkMatterH5 ( my_region_data->region->index, my_region_data->time_c, my_region_data->my_regions, max_frames );
    DumpDarknessH5 ( my_region_data->region->index, my_region_data->my_regions.rp, flow_block_size );
    DumpRegionInitH5 ( my_region_data->region->index, *my_region_data );
    DumpRegionOffsetH5 ( my_region_data->region->index, my_region_data->region->col, my_region_data->region->row );
    DumpTimeCompressionH5 ( my_region_data->region->index, my_region_data->time_c, max_frames);
  }
}

GlobalWriter::GlobalWriter() {
  bfmask = NULL;
  pinnedInFlow = NULL;
  rawWells = NULL;
  mPtrs = NULL;
  washout_flow = NULL;
  nSampleOut = 0;
  sampleIndex = 0;
}


void GlobalWriter::FillExternalLinks(Mask *_bfmask, const PinnedInFlow *_pinnedInFlow, 
                         RawWells *_rawWells, int16_t *_washout_flow){
  rawWells = _rawWells;
  pinnedInFlow = _pinnedInFlow;
  bfmask = _bfmask;
  washout_flow = _washout_flow;
}

void GlobalWriter::SetHdf5Pointer(BkgDataPointers *_my_hdf5)
{
  mPtrs = _my_hdf5;
}


void GlobalWriter::MakeDirName(const char *_results_folder){
  dirName = _results_folder;
}

void GlobalWriter::DeLink()
{
  bfmask = NULL;
  pinnedInFlow = NULL;
  mPtrs = NULL;
}

GlobalWriter::~GlobalWriter(){
  DeLink();
}
