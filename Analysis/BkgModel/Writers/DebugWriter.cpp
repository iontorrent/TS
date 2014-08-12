/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DebugWriter.h"
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "LinuxCompat.h"
#include "RawWells.h"
#include "SignalProcessingMasterFitter.h"

#define BKG_MODEL_DEBUG_DIR "/bkg_debug/"

debug_collection::debug_collection()
  {

    trace_dbg_file = NULL;
    data_dbg_file  = NULL;
    grid_dbg_file  = NULL;
    iter_dbg_file  = NULL;
    region_trace_file = NULL;
    region_only_trace_file = NULL;
    region_1mer_trace_file = NULL;
    region_0mer_trace_file = NULL;
  };


  void debug_collection::DebugFileClose (void)
{
#ifdef FIT_ITERATION_DEBUG_TRACE
  if (iter_dbg_file != NULL)
    fclose (iter_dbg_file);
#endif

  if (trace_dbg_file != NULL){
    fclose (trace_dbg_file);
    trace_dbg_file=NULL;
  }

  if (data_dbg_file != NULL)
    {
      fclose (data_dbg_file);
      data_dbg_file = NULL;
    }

  if (region_trace_file != NULL)
    {
      fclose (region_trace_file);
      region_trace_file = NULL;
    }
  if (region_only_trace_file != NULL)
    {
      fclose(region_only_trace_file);
      region_only_trace_file = NULL;
    }
  if (region_0mer_trace_file != NULL)
    {
      fclose (region_0mer_trace_file);
      region_0mer_trace_file = NULL;
    }

  if (region_1mer_trace_file != NULL)
    {
      fclose (region_1mer_trace_file);
      region_1mer_trace_file = NULL;
    }
}

debug_collection::~debug_collection()
{
  DebugFileClose();
}

void debug_collection::DebugFileOpen (std::string& dirNameStr, Region *region)
{
  if (region == NULL)
    return;

  char *dirName = strdup(dirNameStr.c_str());
  char *fname;
  int name_len = strlen (dirName) + strlen (BKG_MODEL_DEBUG_DIR) + 64;
  struct stat fstatus;
  int         status;

  fname = new char[name_len];

  snprintf (fname,name_len,"%s%s",dirName,BKG_MODEL_DEBUG_DIR);
  status = stat (fname,&fstatus);

  if (status != 0)
  {
    // directory does not exist yet, create it
    mkdir (fname,S_IRWXU | S_IRWXG | S_IRWXO);
  }

  snprintf (fname,name_len,"%s%sdatax%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&data_dbg_file,fname, "a");

  snprintf (fname,name_len,"%s%stracex%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&trace_dbg_file,fname, "a");
  fprintf (trace_dbg_file,"Background Fit Object Created x = %d, y = %d\n",region->col,region->row);
  fflush (trace_dbg_file);

#ifdef FIT_ITERATION_DEBUG_TRACE
  snprintf (fname,name_len,"%s%siterx%dy%d.txt",dirName,BKG_MODEL_DEBUG_DIR,region->col,region->row);
  fopen_s (&iter_dbg_file,fname,"a");
#endif

  snprintf (fname,name_len,"%s/reg_tracex%dy%d.txt",dirName,region->col,region->row);
  fopen_s (&region_trace_file,fname, "a");

  snprintf (fname,name_len,"%s/reg_only_tracex%dy%d.txt",dirName,region->col,region->row);
  fopen_s (&region_only_trace_file,fname, "a");

  snprintf (fname,name_len,"%s/reg_0mer_tracex%dy%d.txt",dirName,region->col,region->row);
  fopen_s (&region_0mer_trace_file,fname, "a");

  snprintf (fname,name_len,"%s/reg_1mer_tracex%dy%d.txt",dirName,region->col,region->row);
  fopen_s (&region_1mer_trace_file,fname, "a");

  free(dirName);
  delete [] fname;
}
//lev_mar_fit->lm_state.residual[ibd]
void debug_collection::DebugBeadIteration (BeadParams &eval_params, reg_params &eval_rp, int iter, float residual,RegionTracker *pointer_regions)
{
  fprintf (trace_dbg_file,"iter:% 3d,(% 5.3f, % 5.3f,% 6.2f, % 2.1f, % 5.3f, % 5.3f, % 5.3f) ",
           iter,eval_params.gain,eval_params.Copies,residual,eval_rp.nuc_shape.sigma,eval_params.R,pointer_regions->rp.RatioDrift,pointer_regions->rp.CopyDrift);
  fprintf (trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
           eval_params.Ampl[0],eval_params.Ampl[1],eval_params.Ampl[2],eval_params.Ampl[3]);
  fprintf (trace_dbg_file,"% 3.2f,% 3.2f,% 3.2f,% 3.2f,",
           eval_params.Ampl[4],eval_params.Ampl[5],eval_params.Ampl[6],eval_params.Ampl[7]);
  fprintf (trace_dbg_file,"% 2.1f,% 2.1f,% 2.1f,% 2.1f,",
           GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape),GetTypicalMidNucTime (&eval_rp.nuc_shape)); // wrong! should be delayed

  fprintf (trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f) ",
           eval_rp.d[0],eval_rp.d[1],eval_rp.d[2],eval_rp.d[3]);
  fprintf (trace_dbg_file,"(% 5.3f, % 5.3f, % 5.3f, % 5.3f, % 5.3f\n) ",
           eval_rp.krate[0],eval_rp.krate[1],eval_rp.krate[2],eval_rp.krate[3],eval_rp.sens);

  fflush (trace_dbg_file);
}


void debug_collection::DebugIterations(BeadTracker &my_beads, RegionTracker *pointer_regions, int flow_block_size)
{
  DumpRegionParamsCSV (iter_dbg_file,& (pointer_regions->rp));
  my_beads.DumpAllBeadsCSV (iter_dbg_file, flow_block_size);
}


// debugging functions down here in the darkness
// so I don't have to read them every time I wander through the code
void debug_collection::MultiFlowComputeTotalSignalTrace (SignalProcessingMasterFitter &bkg, float *fval,struct BeadParams *p,struct reg_params *reg_p,float *sbg, int flow_block_size, int flow_block_start)
{
  float sbg_local[bkg.region_data->my_scratch.bead_flow_t];

  // allow the background to be passed in to save processing
  if (sbg == NULL)
  {
    bkg.region_data->emptytrace->GetShiftedBkg (reg_p->tshift, bkg.region_data->time_c, sbg_local,
                                                flow_block_size);
    sbg = sbg_local;
  }
  //@TODO possibly same for nuc_rise step
  MathModel::MultiFlowComputeCumulativeIncorporationSignal (p,reg_p,
    bkg.region_data->my_scratch.ival,bkg.region_data->my_regions.cache_step,
    *bkg.region_data_extras.cur_bead_block,bkg.region_data->time_c,
    *bkg.region_data_extras.my_flow,bkg.math_poiss, flow_block_size, flow_block_start);
  MathModel::MultiFlowComputeTraceGivenIncorporationAndBackground (fval,p,reg_p,bkg.region_data->my_scratch.ival,sbg,
    bkg.region_data->my_regions,*bkg.region_data_extras.cur_buffer_block,bkg.region_data->time_c,
    *bkg.region_data_extras.my_flow,bkg.global_defaults.signal_process_control.use_vectorization, bkg.region_data->my_scratch.bead_flow_t, flow_block_size, flow_block_start);
}



#define DUMP_N_VALUES(key,format,var,n) \
{\
    fprintf(my_fp,"%s",key);\
    for (int i=0;i<n;i++) fprintf(my_fp,format,var[i]);\
    fprintf(my_fp,"\n");\
}

#define DUMP_SOME_VALUES(key,format,var,end) \
{\
    fprintf(my_fp,"%s[%d-%d]",key,0,end-1);\
    for (int i=0;i<end;i++) fprintf(my_fp,format,var[i]);\
    fprintf(my_fp,"\n");\
}

#define ONLYTRACE_DUMP_SOME_VALUES(key,format,var,flow_block_size) \
  {\
  for (int i=0;i<flow_block_size;i++) fprintf(only_trace,format,var[i]);\
  }

#define ONLYTRACE_DUMP_N_VALUES(key,format,var,n) \
  {\
  for (int i=0;i<n;i++) fprintf(only_trace,format,var[i]);\
  }

// outputs a full trace of a region
// 1.) all region-wide parameters are output along w/ text identifiers so that they can be
//     parsed by downstream tools
// 2.) All pertinent data for a subset of live wells is output.  Per-well data is put in the trace file
//     in blocks of 20 flows (just as they are processed).  All parameters are output w/ text
//     identifiers so that they can be parsed later.  If there are more than 1000 live wells, some will
//     automatically be skipped in order to limit the debug output to ~1000 wells total.  Wells are skipped
//     in a sparse fashion in order to evenly represent the region
//
#define MAX_REGION_TRACE_WELLS  1000
void debug_collection::DumpRegionTrace (SignalProcessingMasterFitter &bkg, int flow_block_size, int flow_block_start )
{
  FILE *my_fp = region_trace_file;
  FILE *only_trace = region_only_trace_file;
  int buff_flow[flow_block_size];
  for( int i = 0 ; i < flow_block_size ; ++i ) buff_flow[i] = flow_block_start + i;
  if (my_fp && only_trace)
  {
    // if this is the first time through, dump all the region-wide parameters that don't change
    if ( flow_block_start == 0)
    {
      fprintf (my_fp,"reg_row:\t%d\n",bkg.region_data->region->row);
      fprintf (my_fp,"reg_col:\t%d\n",bkg.region_data->region->col);
      fprintf (my_fp,"npts:\t%d\n",bkg.region_data->time_c.npts());
      fprintf (my_fp,"tshift:\t%f\n",bkg.region_data->my_regions.rp.tshift);
      fprintf (my_fp,"tau_R_m:\t%f\n",bkg.region_data->my_regions.rp.tau_R_m);
      fprintf (my_fp,"tau_R_o:\t%f\n",bkg.region_data->my_regions.rp.tau_R_o);
      fprintf (my_fp,"sigma:\t%f\n",bkg.region_data->my_regions.rp.nuc_shape.sigma);
      DUMP_N_VALUES ("krate:","\t%f",bkg.region_data->my_regions.rp.krate,NUMNUC);
      float tmp[NUMNUC];
      for (int i=0;i<NUMNUC;i++) tmp[i]=bkg.region_data->my_regions.rp.d[i];
      DUMP_N_VALUES ("d:","\t%f",tmp,NUMNUC);
      DUMP_N_VALUES ("kmax:","\t%f",bkg.region_data->my_regions.rp.kmax,NUMNUC);
      fprintf (my_fp,"sens:\t%f\n",bkg.region_data->my_regions.rp.sens);
      DUMP_N_VALUES ("NucModifyRatio:","\t%f",bkg.region_data->my_regions.rp.NucModifyRatio,NUMNUC);
      DUMP_N_VALUES ("ftimes:","\t%f",bkg.region_data->time_c.frameNumber,bkg.region_data->time_c.npts());
      DUMP_N_VALUES ("error_term:","\t%f",bkg.region_data->my_regions.missing_mass.dark_matter_compensator,bkg.region_data->my_regions.missing_mass.nuc_flow_t);  // one time_c.npts-long term per nuc
      fprintf (my_fp,"end_section:\n");
      // we don't output t_mid_nuc, CopyDrift, or RatioDrift here, because those can change every block of 20 flows
    }
// TODO: dump computed parameters taht are functions of apparently "basic" parameters
// because the routines to compute them are "hidden" in the code
    // now dump parameters and data that can be unique for every block of 20 flows
    DUMP_SOME_VALUES ("flows:","\t%d",buff_flow,flow_block_size);
    fprintf (my_fp,"CopyDrift:\t%f\n",bkg.region_data->my_regions.rp.CopyDrift);
    fprintf (my_fp,"RatioDrift:\t%f\n",bkg.region_data->my_regions.rp.RatioDrift);
    fprintf (my_fp,"t_mid_nuc:\t%f\n",GetTypicalMidNucTime (& (bkg.region_data->my_regions.rp.nuc_shape)));
    DUMP_SOME_VALUES ("nnum:","\t%d",bkg.region_data_extras.my_flow->flow_ndx_map,
      flow_block_size);
    fprintf (my_fp,"end_section:\n");

    float tmp[bkg.region_data->my_scratch.bead_flow_t];
    struct reg_params eval_rp = bkg.region_data->my_regions.rp;
//    float my_xtflux[my_scratch.bead_flow_t];
    float sbg[bkg.region_data->my_scratch.bead_flow_t];

    bkg.region_data->emptytrace->GetShiftedBkg (bkg.region_data->my_regions.rp.tshift, bkg.region_data->time_c, sbg, flow_block_size);
    float skip_num = 1.0;
    if (bkg.region_data->my_beads.numLBeads > MAX_REGION_TRACE_WELLS)
    {
      //skip_num = (float) (bkg.region_data->my_beads.numLBeads) /1000.0;
    }

    float skip_next = 0.0;
    for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
    {
      if ( (float) ibd >= skip_next)
        skip_next += skip_num;
      else
        continue;

      struct BeadParams *p = &bkg.region_data->my_beads.params_nn[ibd];
      fprintf (my_fp,"bead_row:%d\n",p->y);
      fprintf (my_fp,"bead_col:%d\n",p->x);
      float R_tmp[flow_block_size],tau_tmp[flow_block_size];
      for (int i=0;i < flow_block_size;i++)
	      R_tmp[i] = bkg.region_data->my_regions.rp.AdjustEmptyToBeadRatioForFlow (p->R, p->Ampl[i], 
                      p->Copies, p->phi, bkg.region_data_extras.my_flow->flow_ndx_map[i], 
                      flow_block_start + i );

      DUMP_SOME_VALUES ("R:","\t%f",R_tmp,flow_block_size);
      for (int i=0;i < flow_block_size;i++)
        tau_tmp[i] = bkg.region_data->my_regions.rp.ComputeTauBfromEmptyUsingRegionLinearModel (R_tmp[i]);
      DUMP_SOME_VALUES ("tau:","\t%f",tau_tmp,flow_block_size);
      fprintf (my_fp,"P:%f\n",p->Copies);
      fprintf (my_fp,"gain:%f\n",p->gain);

      fprintf (my_fp,"dmult:%f\n",p->dmult);
//        fprintf(my_fp,"in_cnt:%d\n",in_cnt[my_beads.params_nn[ibd].y*region->w+my_beads.params_nn[ibd].x]);
      DUMP_SOME_VALUES ("Ampl:","\t%f",p->Ampl,flow_block_size);
      DUMP_SOME_VALUES ("kmult:","\t%f",p->kmult,flow_block_size);

      // run the model
      MultiFlowComputeTotalSignalTrace (bkg,bkg.region_data->my_scratch.fval,p,& (bkg.region_data->my_regions.rp),sbg, flow_block_size, flow_block_start);

      struct BeadParams eval_params = bkg.region_data->my_beads.params_nn[ibd];
      memset (eval_params.Ampl,0,sizeof (eval_params.Ampl));

      // calculate the model with all 0-mers to get synthetic background by itself
      MultiFlowComputeTotalSignalTrace (bkg,tmp,&eval_params,&eval_rp,sbg, flow_block_size, flow_block_start);

      // calculate proton flux from neighbors
      // why did this get commented out?????...it broke the below code that relies on my_xtflux being initialized!!!
      //CalcXtalkFlux(ibd,my_xtflux);

      // output values
      float tmp_signal[bkg.region_data->my_scratch.bead_flow_t];
      bkg.region_data->my_trace.MultiFlowFillSignalForBead (tmp_signal,ibd, flow_block_size);
      DUMP_N_VALUES ("raw_data:","\t%0.1f", tmp_signal,bkg.region_data->my_scratch.bead_flow_t);
      DUMP_N_VALUES ("fit_data:","\t%.1f",bkg.region_data->my_scratch.fval,bkg.region_data->my_scratch.bead_flow_t);
      DUMP_N_VALUES ("avg_empty:","\t%.1f",sbg,bkg.region_data->my_scratch.bead_flow_t);
      DUMP_N_VALUES ("background:","\t%.1f",tmp,bkg.region_data->my_scratch.bead_flow_t);
      //  temporary comment out until I figure out why CalcXtalkFlux is no longer called above
      //     DUMP_N_VALUES ("xtalk:","\t%.1f",my_xtflux,my_scratch.bead_flow_t);

      fprintf(only_trace,"%d\t%d\t",bkg.region_data->region->row + p->y,bkg.region_data->region->col + p->x);
      fprintf(only_trace,"%s\t","raw_data");
      ONLYTRACE_DUMP_SOME_VALUES ("flows:","\t%d",buff_flow,flow_block_size);
      ONLYTRACE_DUMP_N_VALUES ("fit_data:","\t%0.1f",tmp_signal,bkg.region_data->my_scratch.bead_flow_t);
      fprintf(only_trace,"\n");

      fprintf(only_trace,"%d\t%d\t",p->y,p->x);
      fprintf(only_trace,"%s\t","fit_data");
      ONLYTRACE_DUMP_SOME_VALUES ("flows:","\t%d",buff_flow,flow_block_size);
      ONLYTRACE_DUMP_N_VALUES ("fit_data:","\t%0.1f",bkg.region_data->my_scratch.fval,bkg.region_data->my_scratch.bead_flow_t);
      fprintf(only_trace,"\n");

      fprintf(only_trace,"%d\t%d\t",p->y,p->x);
      fprintf(only_trace,"%s\t","avg_empty");
      ONLYTRACE_DUMP_SOME_VALUES ("flows:","\t%d",buff_flow,flow_block_size);
      ONLYTRACE_DUMP_N_VALUES ("avg_empty:","\t%0.1f",sbg,bkg.region_data->my_scratch.bead_flow_t);
      fprintf(only_trace,"\n");
      fprintf(only_trace,"%d\t%d\t",p->y,p->x);
      fprintf(only_trace,"%s\t","background");
      ONLYTRACE_DUMP_SOME_VALUES ("flows:","\t%d",buff_flow,flow_block_size);
      ONLYTRACE_DUMP_N_VALUES ("background:","\t%0.1f",tmp,bkg.region_data->my_scratch.bead_flow_t);
      fprintf(only_trace,"\n");
    }
    fprintf (my_fp,"end_section:\n");
    fflush (only_trace);
    fflush (my_fp);
  } // end file exists
}
