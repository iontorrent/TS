/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sstream>
#include "BkgModelHdf5.h"
#include "BkgFitterTracker.h"
#include "hdf5.h"


BkgParamH5::BkgParamH5()
{

  max_frames = 0;
  nFlowBlks = 0;
  flow_block_size = 0;
  datacube_numflows = 0;
  bead_col = 0;
  bead_row = 0;
  region_total = 0;
  region_nSamples = 0;
}



MatchedCube::MatchedCube()
{
  h5_set = NULL;
}

MatchedCubeInt::MatchedCubeInt()
{
  h5_set = NULL;
}


// set up a basic data cube + matched h5 set
void MatchedCube::InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, 
                                  const char *set_name, const char *set_description, const char *param_root )
{
  //printf ( "%s\n",set_name );
  string str;
  source.Init ( col, row, maxflows );
  source.SetRange ( 0,col, 0, row, 0, maxflows );
  source.AllocateBuffer();
  h5_set = h5_local_ref.CreateDataSet ( set_name, source, 3 );
  h5_local_ref.CreateAttribute ( h5_set->getDataSetId(),"description",set_description );
  // either we're just using the axis for nothing special
  if ( strlen ( param_root ) <1 )
  {
    h5_local_ref.makeParamNames ( "flow_",maxflows, str );
  }
  else
  {
    // or we have a specific set of variables for this axis
    str = param_root;
  }
  h5_local_ref.CreateAttribute ( h5_set->getDataSetId(),"paramNames",str.c_str() );
}

// set up a basic data cube without matching to h5 set
void MatchedCube::InitBasicCube2 ( int col, int row, int maxflows )
{
  printf ( "creating a cube for error %d, %d, %d\n",col, row, maxflows );
  source.Init ( col, row, maxflows );
  source.SetRange ( 0,col, 0, row, 0, maxflows );
  source.AllocateBuffer();
}


// set up a basic data cube + matched h5 set
//@TODO: use templates to avoid "INT" duplication
void MatchedCubeInt::InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, const char *set_name, const char *set_description, const char *param_root )
{
  //printf ( "%s\n",set_name );
  string str;
  source.Init ( col, row, maxflows );
  source.SetRange ( 0,col, 0, row, 0, maxflows );
  source.AllocateBuffer();
  h5_set = h5_local_ref.CreateDataSet ( set_name, source, 3 );
  h5_local_ref.CreateAttribute ( h5_set->getDataSetId(),"description",set_description );
  // either we're just using the axis for nothing special
  if ( strlen ( param_root ) <1 )
  {
    h5_local_ref.makeParamNames ( "ndx_",maxflows, str );
  }
  else
  {
    // or we have a specific set of variables for this axis
    str = param_root;
  }
  h5_local_ref.CreateAttribute ( h5_set->getDataSetId(),"paramNames",str.c_str() );
}


void MatchedCube::Close()
{
  if ( h5_set!=NULL )
    h5_set->Close();
  h5_set = NULL;
}

void MatchedCubeInt::Close()
{
  if ( h5_set!=NULL )
    h5_set->Close();
  h5_set = NULL;
}

void MatchedCube::SafeWrite()
{
  if ( h5_set!=NULL )
    h5_set->WriteDataCube ( source );
  // should close here and null out pointers?
}

void MatchedCubeInt::SafeWrite()
{
  if ( h5_set!=NULL )
    h5_set->WriteDataCube ( source );
  // should close here and null out pointers?
}

void close_Vector ( vector<H5DataSet * > &vec )
{
  int sz = vec.size();
  if ( sz > 0 )
  {
    for ( int i=0; i<sz; i++ )
      if ( vec[i]!=NULL )
        vec[i]->Close();
    vec.clear();
  }
}

void RotatingCube::Close()
{
  close_Vector ( h5_vec );
}


void RotatingCube::SafeWrite ( int iBlk )
{
  if ( h5_vec.size() >0 )
    h5_vec[iBlk]->WriteDataCube ( source );
}

void RotatingCube::RotateMyCube ( H5File &h5_local_ref, int total_blocks, const char *cube_name, const char *cube_description )
{
  char s[128];
  string str;

  for ( int i=0; i<total_blocks; i++ )
  {
    sprintf ( s,"%s/block_%04d",cube_name, i );
    H5DataSet *ptr = h5_local_ref.CreateDataSet ( s, source, 3 );
    sprintf ( s,"%s",cube_description );
    h5_local_ref.CreateAttribute ( ptr->getDataSetId(),"description",s );
    h5_local_ref.makeParamNames ( "ndx_",source.GetNumZ(),str );
    h5_local_ref.CreateAttribute ( ptr->getDataSetId(),"paramNames",str.c_str() );
    h5_vec.push_back ( ptr );
  }
}


void RotatingCube::InitRotatingCube ( H5File &h5_local_ref, int x, int y, int z, int total_blocks, const char *cube_name, const char *cube_description )
{
  //printf ( "%s\n", cube_name );
  source.Init ( x, y, z );
  source.SetRange ( 0,x, 0, y, 0,z );
  source.AllocateBuffer();

  RotateMyCube ( h5_local_ref, total_blocks, cube_name, cube_description );
}

void BkgParamH5::TryInitBeads ( H5File &h5_local_ref, int verbosity )
{
  ///------------------------------------------------------------------------------------------------------------
  /// bead parameters
  ///------------------------------------------------------------------------------------------------------------
  string str;
  try
  {
    bead_base_parameter.InitBasicCube ( h5_local_ref, bead_col, bead_row, 5,
                                        "/bead/bead_base_parameters","basic 5: copies, etbR, dmult, gain, deltaTime", "Copies,etbR,dmult,gain,deltaTime" );

    if( verbosity>2 ){ //per flow parameters are only included when debug flag is set
      Amplitude.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows,
                                "/bead/amplitude", "mean hydrogens per molecule per flow", "" );
      krate_multiplier.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows,
                                       "/bead/kmult", "adjustment to krate", "" );
      // less important?
      bead_dc.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows,
                              "/bead/trace_dc_offset", "additive factor for trace", "" );

      residual_error.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows,
                                     "/bead/residual_error", "residual_error", "" );
    }

    // still less important?
    average_error_flow_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks,
                                             "/bead/average_error_by_block", "avg_err_by_block", "" );
    bead_clonal_compute_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks,
                                              "/bead/clonal_status_per_block", "clonal_status_by_block", "" );
    bead_corrupt_compute_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks,
                                               "/bead/corrupt_status_per_block", "corrupt_status_by_block", "" );   // status markers for beads

  }
  catch ( char * str )
  {
    cout << "Exception raised while creating bead datasets in BkgParamH5::Init(): " << str << '\n';
  }
}



int AddOneCommaString ( string &outstr, const char *my_name )
{
  char my_char_string[256];
  sprintf ( my_char_string,"%s,",my_name );
  outstr += my_char_string;
  return ( 1 );
}

int AddOneNoCommaString ( string &outstr, const char *my_name )
{
  char my_char_string[256];
  sprintf ( my_char_string,"%s",my_name );
  outstr += my_char_string;
  return ( 1 );
}

int AddCommaStrings ( string &outstr, const char *my_name, int numval )
{
  char my_char_string[256];
  for ( int i=0; i<numval; i++ )
  {
    sprintf ( my_char_string,"%s_%d,",my_name,i );
    outstr += my_char_string;
  }
  return ( numval );
}

void RegionEnzymaticsByName(string &outstr, int &my_total_names)
{
  // enzyme and nucleotide rates
  //float krate[NUMNUC];// rate of incorporation
  my_total_names+=AddCommaStrings ( outstr,"krate",NUMNUC );
  //
  //float d[NUMNUC];    // dNTP diffusion rate
  my_total_names+=AddCommaStrings ( outstr,"d",NUMNUC );
  //float kmax[NUMNUC];  // saturation per nuc action rate
  my_total_names+=AddCommaStrings ( outstr,"kmax",NUMNUC );

}

void RegionBufferingByName(string &outstr, int &my_total_names)
{
  //float tau_R_m;  // relationship of empty to bead slope
  my_total_names+=AddOneCommaString ( outstr,"tau_R_m" );
  //float tau_R_o;  // relationship of empty to bead offset
  my_total_names+=AddOneCommaString ( outstr,"tau_R_o" );
  //float tauE;
  my_total_names+=AddOneCommaString ( outstr,"tauE" );
  //float RatioDrift;      // change over time in buffering
  my_total_names+=AddOneCommaString ( outstr,"RatioDrift" );
  //float NucModifyRatio[NUMNUC];  // buffering modifier per nuc
  my_total_names+=AddCommaStrings ( outstr,"NucModifyRatio",NUMNUC );
  my_total_names+=AddOneCommaString ( outstr,"reg_error" );
}

static void RegionNucShapeByName(string &outstr, int &my_total_names, int flow_block_size)
{
  //nuc_rise_params nuc_shape;
  // rise timing parameters
  //float t_mid_nuc[numfb];
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc",flow_block_size );
  // empirically-derived per-nuc modifiers for t_mid_nuc and sigma
  //float t_mid_nuc_delay[NUMNUC];
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc_delay",NUMNUC );
  //float sigma;
  my_total_names+=AddOneCommaString ( outstr,"sigma" );
  //float sigma_mult[NUMNUC];
  my_total_names+=AddCommaStrings ( outstr,"sigma_mult",NUMNUC );

  // refined t_mid_nuc
  //float t_mid_nuc_shift_per_flow[numfb]; // note how this is redundant(!)
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc_shift_per_flow",flow_block_size );

  // not actually modified,should be an input parameter(!)
  //float C[NUMNUC]; // dntp in uM
  my_total_names+=AddCommaStrings ( outstr,"Concentration",NUMNUC );
  //float valve_open; // timing parameter
  my_total_names+=AddOneCommaString ( outstr,"valve_open" );
  //float nuc_flow_span; // timing of nuc flow
  my_total_names+=AddOneCommaString ( outstr,"nuc_flow_span" );
  //float magic_divisor_for_timing; // timing parameter
  my_total_names+=AddOneNoCommaString ( outstr,"magic_divisor_for_timing" );
}

static int RegionParametersByName ( string &outstr, int flow_block_size )
{
  int my_total_names=0;
  // exploit knowledge of the structure
  outstr = "";

  //RegionEnzymaticsByName(outstr, my_total_names);
  //float tshift;
  my_total_names+=AddOneCommaString ( outstr,"tshift" );

  //RegionBufferingByName(outstr, my_total_names);

  // account for typical change in number of live copies per bead over time
  //float CopyDrift;
  my_total_names+=AddOneCommaString ( outstr,"CopyDrift" );
  my_total_names+=AddOneCommaString ( outstr,"COPYMULTIPLIER" );

  // strength of unexplained background "dark matter"
  //float darkness[flow_block_size];
  my_total_names+=AddCommaStrings ( outstr,"darkness",flow_block_size );
  //float sens; // conversion b/w protons generated and signal - no reason why this should vary by nuc as hydrogens are hydrogens.
  my_total_names+=AddOneCommaString ( outstr,"sens" );
  my_total_names+=AddOneCommaString ( outstr,"SENSMULTIPLIER" );
  my_total_names+=AddOneNoCommaString ( outstr,"molecules_to_micromolar" );

  //RegionNucShapeByName(outstr, my_total_names);
  
  return ( my_total_names );
}


void BkgParamH5::InitRegionalParamCube ( H5File &h5_local_ref )
{
  //printf ( "RegionParamSet\n" );

  string region_names_str;

  int my_total_names = RegionParametersByName ( region_names_str, flow_block_size );
  int numParam = my_total_names;

  regional_params.InitBasicCube ( h5_local_ref, region_total, numParam, nFlowBlks,
                                  "/region/region_param/misc", "regional parameters by compute block", region_names_str.c_str() );

  string nuc_shape_names;
  int nucshapeParam = 0;
  nuc_shape_names="";
  RegionNucShapeByName(nuc_shape_names,nucshapeParam, flow_block_size);
  nuc_shape_params.InitBasicCube ( h5_local_ref, region_total, nucshapeParam, nFlowBlks,
                                   "/region/region_param/nuc_shape", "nuc shape parameters by compute block", nuc_shape_names.c_str() );
  string enzymatics_names;
  int enzymaticsParam = 0;
  enzymatics_names="";
  RegionEnzymaticsByName(enzymatics_names,enzymaticsParam);
  enzymatics_params.InitBasicCube ( h5_local_ref, region_total, enzymaticsParam, nFlowBlks,
                                    "/region/region_param/enzymatics", "enzyme parameters by compute block", enzymatics_names.c_str() );
  string buffering_names;
  int bufferingParam = 0;
  buffering_names="";
  RegionBufferingByName(buffering_names,bufferingParam);
  buffering_params.InitBasicCube ( h5_local_ref, region_total, bufferingParam, nFlowBlks,
                                   "/region/region_param/buffering", "buffering regional models by compute block", buffering_names.c_str() );

  string derived_str = "midNucTime_0,midNucTime_1,midNucTime_2,midNucTime_3,sigma_0,sigma_1,sigma_2,sigma_3";
  derived_params.InitBasicCube ( h5_local_ref, region_total, 8, nFlowBlks,
                                 "/region/derived_param", "derived parameters by compute block", derived_str.c_str() );
}

void BkgParamH5::InitRegionDebugBead( H5File &h5_local_ref)
{
  region_debug_bead.InitBasicCube ( h5_local_ref, region_total, 5, 1,
                                    "/region/debug_bead/basic", "basic 5: copies, etbR, dmult, gain, deltaTime", "Copies,etbR,dmult,gain,deltaTime" );
  region_debug_bead_location.InitBasicCube ( h5_local_ref, region_total, 2, 1,
                                             "/region/debug_bead/location", "col, row", "col,row" );
  region_debug_bead_amplitude_krate.InitBasicCube ( h5_local_ref, region_total, 2, datacube_numflows,
                                                    "/region/debug_bead/amplitude_krate", "Amplitude/krate_multiplier by numflows", "" );

  region_debug_bead_predicted.InitBasicCube ( h5_local_ref, region_total, max_frames, datacube_numflows,
                                              "/region/debug_bead/predicted", "predicted trace for debug bead", "" );
  region_debug_bead_corrected.InitBasicCube ( h5_local_ref, region_total, max_frames, datacube_numflows,
                                              "/region/debug_bead/corrected", "background-adjusted trace for debug bead", "" );
  region_debug_bead_xtalk.InitBasicCube ( h5_local_ref, region_total, max_frames, datacube_numflows,
                                          "/region/debug_bead/xtalk", "local xtalk estimated trace for debug bead", "" );
}



void BkgParamH5::InitBeads_BestRegion( H5File &h5_local_ref, int nBeads_live, const Region *region)
{
  char buff[80];
  //cout << "BkgParamH5::InitBeads_BestRegion... bestRegion=" << bestRegion.first << "," << bestRegion.second << endl << flush;
  // only once (the first flow, not all flows)
  beads_bestRegion_timeframe.InitBasicCube (h5_local_ref,1,max_frames,1,"/bestRegion/timeframe", "Time Frame", "frameNumber");
  beads_bestRegion_location.InitBasicCube (h5_local_ref,nBeads_live,2,1,"/bestRegion/location", "y(row),x(col) for each bead", "row,col" );
  beads_bestRegion_gainSens.InitBasicCube(h5_local_ref,nBeads_live,1,1,"/bestRegion/gainSens","gain*sens","");
  beads_bestRegion_dmult.InitBasicCube(h5_local_ref,nBeads_live,1,1,"/bestRegion/dmult","dMult","");
  beads_bestRegion_SP.InitBasicCube(h5_local_ref,nBeads_live,1,1,"/bestRegion/SP","SP: copies*copy_multiplier","");
  beads_bestRegion_R.InitBasicCube(h5_local_ref,nBeads_live,1,1,"/bestRegion/R","R: ratio of bead buffering to empty buffering","");
  // all flows
  beads_bestRegion_predicted.InitBasicCube(h5_local_ref,nBeads_live,max_frames,datacube_numflows,"/bestRegion/predicted","predicted trace","");
  beads_bestRegion_corrected.InitBasicCube(h5_local_ref,nBeads_live,max_frames,datacube_numflows,"/bestRegion/corrected","background-adjusted trace","");
  beads_bestRegion_original.InitBasicCube(h5_local_ref,nBeads_live,max_frames,datacube_numflows,"/bestRegion/original","original trace","");
  beads_bestRegion_sbg.InitBasicCube(h5_local_ref,nBeads_live,max_frames,datacube_numflows,"/bestRegion/sbg","background trace","");

  beads_bestRegion_amplitude.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/amplitude","amplititude","");
  beads_bestRegion_kmult.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/kmult","kMult","");

  beads_bestRegion_residual.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/residual","residual error","");
  beads_bestRegion_fittype.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/fittype","fittype","");
  beads_bestRegion_converged.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/converged","converged","");
  beads_bestRegion_taub.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/taub","taub","");
  beads_bestRegion_etbR.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/etbR","etbR","");
  beads_bestRegion_bkg_leakage.InitBasicCube(h5_local_ref,nBeads_live,1,datacube_numflows,"/bestRegion/bkgleakage","bkgleakage","");
  // hold both initA, initkmult as they always travel together
  beads_bestRegion_initAk.InitBasicCube(h5_local_ref,nBeads_live,2,datacube_numflows,"/bestRegion/initAk","initAk","");
  beads_bestRegion_tms.InitBasicCube(h5_local_ref,nBeads_live,2,datacube_numflows,"/bestRegion/tms","t_mid_nuc,t_sigma","");

  sprintf(buff,"%d",nBeads_live);
  h5_local_ref.CreateAttribute(beads_bestRegion_location.h5_set->getDataSetId(),"nBeads_live",buff);
  sprintf(buff,"%d",region->h);
  h5_local_ref.CreateAttribute(beads_bestRegion_location.h5_set->getDataSetId(),"region_h",buff);
  sprintf(buff,"%d",region->w);
  h5_local_ref.CreateAttribute(beads_bestRegion_location.h5_set->getDataSetId(),"region_w",buff);
  sprintf(buff,"%d",region->row);
  h5_local_ref.CreateAttribute(beads_bestRegion_location.h5_set->getDataSetId(),"region_y(row)",buff);
  sprintf(buff,"%d",region->col);
  h5_local_ref.CreateAttribute(beads_bestRegion_location.h5_set->getDataSetId(),"region_x(col)",buff);
}


void BkgParamH5::InitBeads_RegionSamples( H5File &h5_local_ref, int nRegions,int nSamples)
{
  int nBeads = nRegions * nSamples;
  // only once (the first flow, not all flows)
  beads_regionSamples_location.InitBasicCube (h5_local_ref,nBeads,3,1,"/regionSamples/location", "y(row),x(col),r(reg) for each bead", "row,col,reg" );
  beads_regionSamples_gainSens.InitBasicCube(h5_local_ref,nBeads,1,1,"/regionSamples/gainSens","gain*sens","");
  beads_regionSamples_dmult.InitBasicCube(h5_local_ref,nBeads,1,1,"/regionSamples/dmult","dMult","");
  beads_regionSamples_SP.InitBasicCube(h5_local_ref,nBeads,1,1,"/regionSamples/SP","SP: copies*copy_multiplier","");
  beads_regionSamples_R.InitBasicCube(h5_local_ref,nBeads,1,1,"/regionSamples/R","R: ratio of bead buffering to empty buffering","");
  // we fill this in once, but it changes over the chip, so we need 1/bead to match traces - might be redundant in same region, but just fine.
  beads_regionSamples_timeframe.InitBasicCube (h5_local_ref,nBeads,max_frames,1,"/regionSamples/timeframe", "Time Frame", "frameNumber");
  // extra for RegionSample, not in bestRegion
  beads_regionSamples_regionParams.InitBasicCube (h5_local_ref,nBeads,3+NUMNUC*3,1,"/regionSamples/regionParams", "gain,sens,copies,kmax*4,krate*4,d*4", "" );
  // all flows
  beads_regionSamples_predicted.InitBasicCube(h5_local_ref,nBeads,max_frames,datacube_numflows,"/regionSamples/predicted","predicted trace","");
  beads_regionSamples_corrected.InitBasicCube(h5_local_ref,nBeads,max_frames,datacube_numflows,"/regionSamples/corrected","background-adjusted trace","");
  beads_regionSamples_original.InitBasicCube(h5_local_ref,nBeads,max_frames,datacube_numflows,"/regionSamples/original","original trace","");
  beads_regionSamples_sbg.InitBasicCube(h5_local_ref,nBeads,max_frames,datacube_numflows,"/regionSamples/sbg","background trace","");
  beads_regionSamples_amplitude.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/amplitude","amplititude","");
  beads_regionSamples_kmult.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/kmult","kMult","");
  beads_regionSamples_residual.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/residual","residual error","");
  beads_regionSamples_fittype.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/fittype","fittype","");
  beads_regionSamples_converged.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/converged","converged","");
  beads_regionSamples_taub.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/taub","taub","");
  beads_regionSamples_bkg_leakage.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/bkgleakage","bkg leakage fraction","");
  beads_regionSamples_etbR.InitBasicCube(h5_local_ref,nBeads,1,datacube_numflows,"/regionSamples/etbR","etbR","");
  beads_regionSamples_initAk.InitBasicCube(h5_local_ref,nBeads,2,datacube_numflows,"/regionSamples/initAk","initAk","");
  beads_regionSamples_tms.InitBasicCube(h5_local_ref,nBeads,2,datacube_numflows,"/regionSamples/tms","t_mid_nuc,t_sigma","");
}




void BkgParamH5::TryInitRegionParams ( H5File &h5_local_ref, const ImageSpecClass &my_image_spec )
{
  ///------------------------------------------------------------------------------------------------------------
  /// region parameters
  ///------------------------------------------------------------------------------------------------------------
  try
  {
    empty_trace.InitBasicCube ( h5_local_ref, region_total, my_image_spec.uncompFrames, datacube_numflows,
                                "/region/empty_trace", "empty trace per frame per flow","" );
    empty_dc.InitBasicCube ( h5_local_ref, region_total, 1, datacube_numflows,
                             "/region/empty_dc", "empty dc offset per flow","" );

    // write once items
    dark_matter_trace.InitBasicCube ( h5_local_ref, region_total, NUMNUC, max_frames,
                                      "/region/darkMatter/missingMass", "dark matter trace by nucleotide","" );
    darkness_val.InitBasicCube ( h5_local_ref, region_total, flow_block_size, 1,
                                 "/region/darkMatter/darkness","darkness per region per flowbuffer", "" );

    region_init_val.InitBasicCube ( h5_local_ref, region_total, 2, 1,
                                    "/region/region_init_param", "starting values", "t_mid_nuc_start,sigma_start" );

    time_compression.InitBasicCube ( h5_local_ref, region_total, 4, max_frames,
                                     "/region/time_compression", "Time compression", "frameNumber,deltaFrame,frames_per_point,npt" );

    region_offset_val.InitBasicCube ( h5_local_ref, region_total, 2, 1,
                                      "/region/region_location", "smallest corner of region aka offset of beads", "col,row" );

    //this is a data-cube per compute block, so need the 'rotating cube' formulation
    emphasis_val.InitRotatingCube ( h5_local_ref, region_total, MAX_POISSON_TABLE_COL,max_frames,nFlowBlks,
                                    "/region/emphasis","Emphasis vector by HPLen in compute blocks" );

    InitRegionalParamCube ( h5_local_ref );
    // store a "debug bead" as a representative for a region
    InitRegionDebugBead( h5_local_ref);

    // Note: bestRegion is Init2
  }
  catch ( char * str )
  {
    cout << "Exception raised while creating region datasets in BkgParamH5::Init(): " << str << '\n';
  }
}


void BkgParamH5::TryInitBeads_BestRegion ( H5File &h5_local_ref, int nBeads_live, Region *region )
{
  ///------------------------------------------------------------------------------------------------------------
  /// region parameters
  ///------------------------------------------------------------------------------------------------------------
  try
  {
    InitBeads_BestRegion( h5_local_ref, nBeads_live, region);
  }
  catch ( char * str )
  {
    cout << "Exception raised while creating region datasets in BkgParamH5::TryInitBeadBestRegion(): " << str << '\n';
  }
}


void BkgParamH5::ConstructOneFile ( H5File &h5_local_ref, string &hgLocalFile, string &local_results, const char *my_name )
{
  hgLocalFile = local_results;

  if ( hgLocalFile[hgLocalFile.length()-1] != '/' )
    hgLocalFile += '/';
  hgLocalFile += my_name;
  cout << "H5File for params:" << hgLocalFile << endl;

  h5_local_ref.Init();
  h5_local_ref.SetFile ( hgLocalFile );
  h5_local_ref.Open ( true );
}


void BkgParamH5::Init ( const char *results_folder, const SpatialContext &loc_context, 
                        const ImageSpecClass &my_image_spec,
                        int numFlows,
                        int write_params_flag,
                        int _max_frames,
                        int _flow_block_size,
                        int num_flow_blocks
                        )
{
  cout << "BkgParamH5::Init... _max_frames = " << _max_frames << ", MAX_COMPRESSED_FRAMES = " << MAX_COMPRESSED_FRAMES << endl;
  //max_frames = MAX_COMPRESSED_FRAMES;
  max_frames = _max_frames;
  if ( write_params_flag>0 )
  {
    local_results_directory=results_folder;

    flow_block_size = _flow_block_size;
    datacube_numflows = numFlows;
    nFlowBlks = num_flow_blocks;
    bead_col = loc_context.cols;
    bead_row = loc_context.rows;
    region_total = loc_context.numRegions;

    ConstructOneFile ( h5BeadDbg, hgBeadDbgFile,local_results_directory, "bead_param.h5" );

    TryInitBeads ( h5BeadDbg, write_params_flag ); //write_params_flag: 1-only important params, small file; 2-all params (very large file, use for debugging only)

    ConstructOneFile ( h5RegionDbg, hgRegionDbgFile,local_results_directory, "region_param.h5" );

    TryInitRegionParams ( h5RegionDbg,my_image_spec );

    ///------------------------------------------------------------------------------------------------------------
    /// savePointers to be passed to BkgModel::WriteBeadParameterstoDataCubes
    ///------------------------------------------------------------------------------------------------------------
    /// WARNING: these pointers determine >individual< behavior of data cubes
    /// just because one cube exists doesn't mean any others exists - see different logging levels
    /// if a cube exists, the >exporting< function in GlobalWriter needs to handle it correctly.
    savePointers();
  }
  else
  {
    hgBeadDbgFile = "";
    hgRegionDbgFile="";
  }
}

void BkgParamH5::AllocBeadRes ( const SpatialContext &loc_context,
                        const ImageSpecClass &my_image_spec,
                        int numFlows,
                        int _max_frames,
                        int _flow_block_size,
                        int num_flow_blocks
                        ){
	flow_block_size = _flow_block_size;
	datacube_numflows = numFlows;
	nFlowBlks = num_flow_blocks;
	bead_col = loc_context.cols;
	bead_row = loc_context.rows;
	region_total = loc_context.numRegions;
	residual_error.InitBasicCube2 ( bead_col, bead_row, datacube_numflows );
	ptrs.mResError1 = residual_error.Ptr();
}


void BkgParamH5::Init2 (int write_params_flag, int nBeads_live, const Region *region, int nRegions,int nSamples)
{
  if ( write_params_flag>1 )
  {
    ConstructOneFile ( h5TraceDbg, hgTraceDbgFile,local_results_directory, "trace.h5" );
    InitBeads_BestRegion ( h5TraceDbg,nBeads_live,region );
    saveBestRegionPointers();
    InitBeads_RegionSamples ( h5TraceDbg,nRegions,nSamples );
    saveRegionSamplesPointers();
    region_nSamples = nSamples;
  }
  else
  {
    hgTraceDbgFile = "";
    region_nSamples = 0;
  }
}


void BkgParamH5::WriteOneFlowBlock ( DataCube<float> &cube, H5DataSet *set, int flow, int chunksize )
{
  //  fprintf ( stdout, "Writing incremental H5-diagnostics at flow: %d\n", flow );
  MemUsage ( "BeforeWrite" );
  size_t starts[3];
  size_t ends[3];
  cube.SetStartsEnds ( starts, ends );
  // here's the actual write
  set->WriteRangeData ( starts, ends, cube.GetMemPtr() );
  // set for next iteration
  int nextflow = flow+1;
  int nextchunk = min ( chunksize,datacube_numflows- ( flow+1 ) );
  cube.SetRange ( 0, cube.GetNumX(), 0, cube.GetNumY(), nextflow, nextflow+nextchunk );
  MemUsage ( "AfterWrite" );
}


void BkgParamH5::WriteOneFlowBlock ( DataCube<int> &cube, H5DataSet *set, int flow, int chunksize )
{
  //  fprintf ( stdout, "Writing incremental H5-diagnostics at flow: %d\n", flow );
  MemUsage ( "BeforeWrite" );
  size_t starts[3];
  size_t ends[3];
  cube.SetStartsEnds ( starts, ends );
  // here's the actual write
  set->WriteRangeData ( starts, ends, cube.GetMemPtr() );
  // set for next iteration
  int nextflow = flow+1;
  int nextchunk = min ( chunksize,datacube_numflows- ( flow+1 ) );
  cube.SetRange ( 0, cube.GetNumX(), 0, cube.GetNumY(), nextflow, nextflow+nextchunk );
  MemUsage ( "AfterWrite" );
}


void BkgParamH5::IncrementalWriteParam ( DataCube<float> &cube, H5DataSet *set, int flow )
{
  // please do not do this(!)
  // there needs to be >one< master routine that controls when we write to files
  // not several independent implementations of the same logic

  if ( set!=NULL )
  {
    WriteOneFlowBlock ( cube,set,flow,flow_block_size );
  }
}


void BkgParamH5::IncrementalWriteParam ( DataCube<int> &cube, H5DataSet *set, int flow )
{
  // please do not do this(!)
  // there needs to be >one< master routine that controls when we write to files
  // not several independent implementations of the same logic

  if ( set!=NULL )
  {
    WriteOneFlowBlock ( cube,set,flow,flow_block_size );
  }
}


// set to write one compute block
void BkgParamH5::WriteOneBlock ( DataCube<float> &cube, H5DataSet *set, int iBlk )
{
  if ( set!=NULL )
  {
    //   fprintf ( stdout, "Writing incremental H5-diagnostics at compute block: %d\n", iBlk );
    MemUsage ( "BeforeWrite" );
    size_t starts[3];
    size_t ends[3];
    cube.SetStartsEnds ( starts, ends );
    // here's the actual write
    set->WriteRangeData ( starts, ends, cube.GetMemPtr() );
    // set for next iteration
    int nextBlk = iBlk+1;
    int nextChunk = min ( 1,nFlowBlks-nextBlk );
    cube.SetRange ( 0, cube.GetNumX(), 0, cube.GetNumY(), nextBlk, nextBlk+nextChunk );
    MemUsage ( "AfterWrite" );
  }
}

// set to write one compute block
void BkgParamH5::WriteOneBlock ( DataCube<int> &cube, H5DataSet *set, int iBlk )
{
  if ( set!=NULL )
  {
    //   fprintf ( stdout, "Writing incremental H5-diagnostics at compute block: %d\n", iBlk );
    MemUsage ( "BeforeWrite" );
    size_t starts[3];
    size_t ends[3];
    cube.SetStartsEnds ( starts, ends );
    // here's the actual write
    set->WriteRangeData ( starts, ends, cube.GetMemPtr() );
    // set for next iteration
    int nextBlk = iBlk+1;
    int nextChunk = min ( 1,nFlowBlks-nextBlk );
    cube.SetRange ( 0, cube.GetNumX(), 0, cube.GetNumY(), nextBlk, nextBlk+nextChunk );
    MemUsage ( "AfterWrite" );
  }
}

void BkgParamH5::IncrementalWriteBeads ( int flow, int iBlk )
{
  // every parameter checks itself to see if writing is useful or safe
  // every time we write, we write these
	printf("+++ IncrementalWriteBeads %d", flow);
  IncrementalWriteParam ( bead_dc.source,bead_dc.h5_set,flow );
  IncrementalWriteParam ( Amplitude.source,Amplitude.h5_set,flow );
  IncrementalWriteParam ( krate_multiplier.source,krate_multiplier.h5_set,flow );
  IncrementalWriteParam ( residual_error.source,residual_error.h5_set,flow );

  // do compute blocks
  WriteOneBlock ( average_error_flow_block.source, average_error_flow_block.h5_set,iBlk );
  WriteOneBlock ( bead_clonal_compute_block.source, bead_clonal_compute_block.h5_set,iBlk );
  WriteOneBlock ( bead_corrupt_compute_block.source, bead_corrupt_compute_block.h5_set,iBlk );

  if ( iBlk==0 ) // only do the first compute block
  {
    bead_base_parameter.SafeWrite();
  }
}

void BkgParamH5::IncrementalWriteBestRegion ( int flow, bool lastflow )
{
  // trace
  IncrementalWriteParam ( beads_bestRegion_predicted.source, beads_bestRegion_predicted.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_corrected.source, beads_bestRegion_corrected.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_original.source, beads_bestRegion_original.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_sbg.source, beads_bestRegion_sbg.h5_set, flow );
  // per bead-per flow fitted
  IncrementalWriteParam ( beads_bestRegion_amplitude.source, beads_bestRegion_amplitude.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_kmult.source, beads_bestRegion_kmult.h5_set, flow );
  // incidental data
  IncrementalWriteParam ( beads_bestRegion_residual.source, beads_bestRegion_residual.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_fittype.source, beads_bestRegion_fittype.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_converged.source, beads_bestRegion_converged.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_taub.source, beads_bestRegion_taub.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_etbR.source, beads_bestRegion_etbR.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_bkg_leakage.source, beads_bestRegion_bkg_leakage.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_initAk.source, beads_bestRegion_initAk.h5_set, flow );
  IncrementalWriteParam ( beads_bestRegion_tms.source, beads_bestRegion_tms.h5_set, flow );

  if ( lastflow ) // do only once at lastflow, for data independant of flow
  {
    beads_bestRegion_timeframe.SafeWrite(); // this only has to done once for all flows
    beads_bestRegion_location.SafeWrite(); // this only has to done once for all flows
    beads_bestRegion_gainSens.SafeWrite();
    beads_bestRegion_dmult.SafeWrite();
    beads_bestRegion_SP.SafeWrite();
    beads_bestRegion_R.SafeWrite();
  }
}


void BkgParamH5::IncrementalWriteRegionSamples ( int flow, bool lastflow )
{
  // trace
  IncrementalWriteParam ( beads_regionSamples_predicted.source, beads_regionSamples_predicted.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_corrected.source, beads_regionSamples_corrected.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_original.source, beads_regionSamples_original.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_sbg.source, beads_regionSamples_sbg.h5_set, flow );
  // per bead per flow fitted
  IncrementalWriteParam ( beads_regionSamples_amplitude.source, beads_regionSamples_amplitude.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_kmult.source, beads_regionSamples_kmult.h5_set, flow );
  // incidental data
  IncrementalWriteParam ( beads_regionSamples_residual.source, beads_regionSamples_residual.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_fittype.source, beads_regionSamples_fittype.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_converged.source, beads_regionSamples_converged.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_bkg_leakage.source, beads_regionSamples_bkg_leakage.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_initAk.source, beads_regionSamples_initAk.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_tms.source, beads_regionSamples_tms.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_taub.source, beads_regionSamples_taub.h5_set, flow );
  IncrementalWriteParam ( beads_regionSamples_etbR.source, beads_regionSamples_etbR.h5_set, flow );

  if ( lastflow ) // do only once at lastflow, for data independant of flow
  {
    beads_regionSamples_timeframe.SafeWrite(); // this only has to done once for all flows
    beads_regionSamples_location.SafeWrite(); // this only has to done once for all flows
    beads_regionSamples_gainSens.SafeWrite();

    beads_regionSamples_regionParams.SafeWrite();

    beads_regionSamples_dmult.SafeWrite();
    beads_regionSamples_SP.SafeWrite();
    beads_regionSamples_R.SafeWrite();
  }
}

void BkgParamH5::IncrementalWriteRegions ( int flow, int iBlk )
{
  // write to any live region parameters
  // every parameter checks itself to see if writing is useful or safe
  IncrementalWriteParam ( empty_trace.source,empty_trace.h5_set,flow );
  IncrementalWriteParam ( empty_dc.source, empty_dc.h5_set, flow );

  IncrementalWriteParam ( region_debug_bead_amplitude_krate.source, region_debug_bead_amplitude_krate.h5_set, flow );
  IncrementalWriteParam ( region_debug_bead_predicted.source, region_debug_bead_predicted.h5_set, flow );
  IncrementalWriteParam ( region_debug_bead_corrected.source, region_debug_bead_corrected.h5_set, flow );
  IncrementalWriteParam ( region_debug_bead_xtalk.source, region_debug_bead_xtalk.h5_set, flow );

  WriteOneBlock ( regional_params.source, regional_params.h5_set,iBlk );
  WriteOneBlock ( nuc_shape_params.source, nuc_shape_params.h5_set,iBlk );
  WriteOneBlock ( enzymatics_params.source, enzymatics_params.h5_set,iBlk );
  WriteOneBlock ( buffering_params.source, buffering_params.h5_set,iBlk );
  WriteOneBlock ( derived_params.source, derived_params.h5_set,iBlk );

  emphasis_val.SafeWrite ( iBlk ); // do every compute block

  if ( iBlk==0 ) // do only once at first compute block
  {
    region_debug_bead_location.SafeWrite();
    region_debug_bead.SafeWrite();
    dark_matter_trace.SafeWrite();
    darkness_val.SafeWrite();
    region_init_val.SafeWrite();
    region_offset_val.SafeWrite();
    time_compression.SafeWrite();
  }
}

void BkgParamH5::IncrementalWrite ( int flow, bool last_flow, FlowBlockSequence::const_iterator flow_block,
                                    int flow_block_id )
{
  // single check: are we done with a compute block?
  // try to isolate logic as much as possible - we should only see one check for writing, ever

  if ( last_flow || flow == flow_block->end() - 1 )
  {
    IncrementalWriteBeads ( flow, flow_block_id );
    IncrementalWriteRegions ( flow, flow_block_id );
    IncrementalWriteBestRegion ( flow, last_flow );
    IncrementalWriteRegionSamples ( flow, last_flow );
    IncrementalWrite_xyflow ( last_flow );
  }
}



void BkgParamH5::CloseBeads()
{
  Amplitude.Close();
  bead_dc.Close();
  krate_multiplier.Close();
  residual_error.Close();
  bead_base_parameter.Close();
  average_error_flow_block.Close();
  bead_clonal_compute_block.Close();
  bead_corrupt_compute_block.Close();

}

void BkgParamH5::CloseBestRegion(){
  beads_bestRegion_location.Close();
  beads_bestRegion_predicted.Close();
  beads_bestRegion_corrected.Close();
  beads_bestRegion_original.Close();
  beads_bestRegion_sbg.Close();
  beads_bestRegion_amplitude.Close();
  beads_bestRegion_residual.Close();
  beads_bestRegion_kmult.Close();
  beads_bestRegion_dmult.Close();
  beads_bestRegion_SP.Close();
  beads_bestRegion_R.Close();
  beads_bestRegion_gainSens.Close();
  beads_bestRegion_fittype.Close();
  beads_bestRegion_converged.Close();
  beads_bestRegion_timeframe.Close();
  beads_bestRegion_taub.Close();
  beads_bestRegion_etbR.Close();
  beads_bestRegion_bkg_leakage.Close();
  beads_bestRegion_initAk.Close();
  beads_bestRegion_tms.Close();
}

void BkgParamH5::CloseRegionSamples(){

  beads_regionSamples_location.Close();
  beads_regionSamples_predicted.Close();
  beads_regionSamples_corrected.Close();
  beads_regionSamples_amplitude.Close();
  beads_regionSamples_residual.Close();
  beads_regionSamples_kmult.Close();
  beads_regionSamples_dmult.Close();
  beads_regionSamples_SP.Close();
  beads_regionSamples_R.Close();
  beads_regionSamples_gainSens.Close();
  beads_regionSamples_fittype.Close();
  beads_regionSamples_converged.Close();
  beads_regionSamples_bkg_leakage.Close();
  beads_regionSamples_initAk.Close();
  beads_regionSamples_tms.Close();
  beads_regionSamples_timeframe.Close();
  beads_regionSamples_taub.Close();
  beads_regionSamples_etbR.Close();
  beads_regionSamples_regionParams.Close();
}

void BkgParamH5::CloseRegion()
{
  dark_matter_trace.Close();
  darkness_val.Close();
  empty_trace.Close();
  empty_dc.Close();
  region_init_val.Close();
  region_offset_val.Close();
  regional_params.Close();
  nuc_shape_params.Close();
  enzymatics_params.Close();
  buffering_params.Close();
  derived_params.Close();
  region_debug_bead.Close();
  region_debug_bead_amplitude_krate.Close();
  region_debug_bead_predicted.Close();
  region_debug_bead_corrected.Close();
  region_debug_bead_xtalk.Close();
  region_debug_bead_location.Close();
  time_compression.Close();
  emphasis_val.Close();


}


void BkgParamH5::CloseTraceXYFlow()
{
  beads_xyflow_predicted.Close();
  beads_xyflow_corrected.Close();
  beads_xyflow_amplitude.Close();
  beads_xyflow_location.Close();
  beads_xyflow_hplen.Close();
  beads_xyflow_mm.Close();
  beads_xyflow_kmult.Close();
  beads_xyflow_dmult.Close();
  beads_xyflow_SP.Close();
  beads_xyflow_R.Close();
  beads_xyflow_gainSens.Close();
  beads_xyflow_fittype.Close();
  beads_xyflow_timeframe.Close();
  beads_xyflow_residual.Close();
  beads_xyflow_taub.Close();
  // keys
  beads_xyflow_predicted_keys.Close();
  beads_xyflow_corrected_keys.Close();
  beads_xyflow_location_keys.Close();
}


void BkgParamH5::Close()
{
  //bead_param.h5
  CloseBeads();
  //region_param.h5
  CloseRegion();
  // trace.h5
  CloseBestRegion();
  CloseRegionSamples();
  CloseTraceXYFlow();
  if ( hgBeadDbgFile.length() >0 )
  {
    cout << "bgParamH5 output: " << hgBeadDbgFile << endl;
  }
  if ( hgRegionDbgFile.length() >0 )
  {
    cout << "bgParamH5 output: " << hgRegionDbgFile << endl;
  }
  if ( hgTraceDbgFile.length() >0 )
  {
    cout << "bgParamH5 output: " << hgTraceDbgFile << endl;
  }
}

void BkgParamH5::saveBeadPointers()
{
  ptrs.mAmpl = Amplitude.Ptr();
  ptrs.mBeadInitParam = bead_base_parameter.Ptr();
  ptrs.mBeadDC = bead_dc.Ptr();
  ptrs.mKMult = krate_multiplier.Ptr();
  //ptrs.mResError = residual_error.Ptr();
  ptrs.mResError1 = residual_error.Ptr();
  ptrs.mBeadFblk_avgErr= average_error_flow_block.Ptr();
  ptrs.mBeadFblk_clonal= bead_clonal_compute_block.Ptr();
  ptrs.mBeadFblk_corrupt= bead_corrupt_compute_block.Ptr();
}

void BkgParamH5::saveRegionPointers()
{
  ptrs.mDarkOnceParam = dark_matter_trace.Ptr();
  ptrs.mDarknessParam= darkness_val.Ptr();
  ptrs.mRegionInitParam = region_init_val.Ptr();
  ptrs.mEmptyOnceParam = empty_trace.Ptr();
  ptrs.mBeadDC_bg = empty_dc.Ptr();
  ptrs.mRegionOffset = region_offset_val.Ptr();
  ptrs.m_regional_param = regional_params.Ptr();
  ptrs.m_nuc_shape_param = nuc_shape_params.Ptr();
  ptrs.m_enzymatics_param = enzymatics_params.Ptr();
  ptrs.m_buffering_param = buffering_params.Ptr();
  ptrs.m_derived_param = derived_params.Ptr();
  ptrs.mEmphasisParam = emphasis_val.Ptr();
  ptrs.m_region_debug_bead = region_debug_bead.Ptr();
  ptrs.m_region_debug_bead_ak = region_debug_bead_amplitude_krate.Ptr();
  ptrs.m_region_debug_bead_predicted = region_debug_bead_predicted.Ptr();
  ptrs.m_region_debug_bead_corrected = region_debug_bead_corrected.Ptr();
  ptrs.m_region_debug_bead_xtalk = region_debug_bead_xtalk.Ptr();
  ptrs.m_region_debug_bead_location = region_debug_bead_location.Ptr();
  ptrs.m_time_compression = time_compression.Ptr();
}

void BkgParamH5::savePointers()
{
  saveBeadPointers();
  saveRegionPointers();
}


void BkgParamH5::saveBestRegionPointers()
{
  ptrs.m_beads_bestRegion_location = beads_bestRegion_location.Ptr();
  ptrs.m_beads_bestRegion_corrected = beads_bestRegion_corrected.Ptr();
  ptrs.m_beads_bestRegion_original = beads_bestRegion_original.Ptr();
  ptrs.m_beads_bestRegion_sbg = beads_bestRegion_sbg.Ptr();
  ptrs.m_beads_bestRegion_predicted = beads_bestRegion_predicted.Ptr();
  ptrs.m_beads_bestRegion_amplitude = beads_bestRegion_amplitude.Ptr();
  ptrs.m_beads_bestRegion_residual = beads_bestRegion_residual.Ptr();
  ptrs.m_beads_bestRegion_kmult = beads_bestRegion_kmult.Ptr();
  ptrs.m_beads_bestRegion_dmult = beads_bestRegion_dmult.Ptr();
  ptrs.m_beads_bestRegion_SP = beads_bestRegion_SP.Ptr();
  ptrs.m_beads_bestRegion_R = beads_bestRegion_R.Ptr();
  ptrs.m_beads_bestRegion_gainSens = beads_bestRegion_gainSens.Ptr();
  ptrs.m_beads_bestRegion_fittype = beads_bestRegion_fittype.Ptr();
  ptrs.m_beads_bestRegion_converged = beads_bestRegion_converged.Ptr();
  ptrs.m_beads_bestRegion_timeframe = beads_bestRegion_timeframe.Ptr();
  ptrs.m_beads_bestRegion_taub = beads_bestRegion_taub.Ptr();
  ptrs.m_beads_bestRegion_etbR = beads_bestRegion_etbR.Ptr();
  ptrs.m_beads_bestRegion_bkg_leakage = beads_bestRegion_bkg_leakage.Ptr();
  ptrs.m_beads_bestRegion_initAk = beads_bestRegion_initAk.Ptr();
  ptrs.m_beads_bestRegion_tms = beads_bestRegion_tms.Ptr();
}


void BkgParamH5::saveRegionSamplesPointers()
{
  ptrs.m_beads_regionSamples_location = beads_regionSamples_location.Ptr();
  ptrs.m_beads_regionSamples_corrected = beads_regionSamples_corrected.Ptr();
  ptrs.m_beads_regionSamples_original = beads_regionSamples_original.Ptr();
  ptrs.m_beads_regionSamples_sbg = beads_regionSamples_sbg.Ptr();
  ptrs.m_beads_regionSamples_predicted = beads_regionSamples_predicted.Ptr();
  ptrs.m_beads_regionSamples_amplitude = beads_regionSamples_amplitude.Ptr();
  ptrs.m_beads_regionSamples_residual = beads_regionSamples_residual.Ptr();
  ptrs.m_beads_regionSamples_kmult = beads_regionSamples_kmult.Ptr();
  ptrs.m_beads_regionSamples_dmult = beads_regionSamples_dmult.Ptr();
  ptrs.m_beads_regionSamples_SP = beads_regionSamples_SP.Ptr();
  ptrs.m_beads_regionSamples_R = beads_regionSamples_R.Ptr();
  ptrs.m_beads_regionSamples_gainSens = beads_regionSamples_gainSens.Ptr();
  ptrs.m_beads_regionSamples_fittype = beads_regionSamples_fittype.Ptr();
  ptrs.m_beads_regionSamples_converged = beads_regionSamples_converged.Ptr();
  ptrs.m_beads_regionSamples_bkg_leakage = beads_regionSamples_bkg_leakage.Ptr();
  ptrs.m_beads_regionSamples_initAk = beads_regionSamples_initAk.Ptr();
  ptrs.m_beads_regionSamples_tms = beads_regionSamples_tms.Ptr();
  ptrs.m_beads_regionSamples_timeframe = beads_regionSamples_timeframe.Ptr();
  ptrs.m_beads_regionSamples_taub = beads_regionSamples_taub.Ptr();
  ptrs.m_beads_regionSamples_etbR = beads_regionSamples_etbR.Ptr();
  ptrs.m_beads_regionSamples_regionParams = beads_regionSamples_regionParams.Ptr();
}


void BkgParamH5::InitBeads_xyflow(int write_params_flag, HashTable_xyflow &xyf_hash)
{
  if ( write_params_flag>1 )
  {
    int nBeads_xyf = xyf_hash.size(); // for xyflow traces
    int nBeads_xy = xyf_hash.size_xy(); // for key traces only
    assert(nBeads_xyf>0);
    assert(nBeads_xy>0);
    beads_xyflow_predicted.InitBasicCube(h5TraceDbg,nBeads_xyf,max_frames,1,"/xyflow/predicted","predicted trace","");
    beads_xyflow_corrected.InitBasicCube(h5TraceDbg,nBeads_xyf,max_frames,1,"/xyflow/corrected","background-adjusted trace","");
    beads_xyflow_amplitude.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/amplitude","row:col:flow amplitude","");
    beads_xyflow_residual.InitBasicCube (h5TraceDbg,nBeads_xyf,1,1,"/xyflow/residual", "residual error", "residual");
    beads_xyflow_location.InitBasicCube(h5TraceDbg,nBeads_xyf,3,1,"/xyflow/location","row:col:flow location","");
    beads_xyflow_hplen.InitBasicCube(h5TraceDbg,nBeads_xyf,2,1,"/xyflow/hplen","ACGT:len, homopolymer length of the reference sequence","");
    beads_xyflow_mm.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/mismatch","mismatch, m/mm=0/1","");
    beads_xyflow_kmult.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/kmult","kmult","");
    beads_xyflow_dmult.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/dmult","dmult","");
    beads_xyflow_SP.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/SP","SP: copies*copy_multiplier","");
    beads_xyflow_R.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/R","R: ratio of bead buffering to empty buffering","");
    beads_xyflow_gainSens.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/gainSens","gain*sens","");
    beads_xyflow_fittype.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/fittype","fittype","");
    beads_xyflow_timeframe.InitBasicCube (h5TraceDbg,nBeads_xyf,max_frames,1,"/xyflow/timeframe", "Time Frame", "frameNumber");
    beads_xyflow_taub.InitBasicCube(h5TraceDbg,nBeads_xyf,1,1,"/xyflow/taub","taub","");

    ptrs.m_beads_xyflow_predicted = beads_xyflow_predicted.Ptr();
    ptrs.m_beads_xyflow_corrected = beads_xyflow_corrected.Ptr();
    ptrs.m_beads_xyflow_amplitude = beads_xyflow_amplitude.Ptr();
    ptrs.m_beads_xyflow_location = beads_xyflow_location.Ptr();
    ptrs.m_beads_xyflow_hplen = beads_xyflow_hplen.Ptr();
    ptrs.m_beads_xyflow_mm = beads_xyflow_mm.Ptr();
    ptrs.m_beads_xyflow_kmult = beads_xyflow_kmult.Ptr();
    ptrs.m_beads_xyflow_dmult = beads_xyflow_dmult.Ptr();
    ptrs.m_beads_xyflow_SP = beads_xyflow_SP.Ptr();
    ptrs.m_beads_xyflow_R = beads_xyflow_R.Ptr();
    ptrs.m_beads_xyflow_gainSens = beads_xyflow_gainSens.Ptr();
    ptrs.m_beads_xyflow_fittype = beads_xyflow_fittype.Ptr();
    ptrs.m_beads_xyflow_timeframe = beads_xyflow_timeframe.Ptr();
    ptrs.m_beads_xyflow_residual = beads_xyflow_residual.Ptr();
    ptrs.m_beads_xyflow_taub = beads_xyflow_taub.Ptr();

    ptrs.m_xyflow_hashtable = &xyf_hash;

    // key traces corresponding the the (x,y) locations of xyflow
    beads_xyflow_location_keys.InitBasicCube(h5TraceDbg,nBeads_xy,2,1,"/xyflow/keys_location","row:col:flow location","");
    beads_xyflow_predicted_keys.InitBasicCube(h5TraceDbg,nBeads_xy,max_frames,4,"/xyflow/keys_predicted","predicted trace for keys","");
    beads_xyflow_corrected_keys.InitBasicCube(h5TraceDbg,nBeads_xy,max_frames,4,"/xyflow/keys_corrected","background-adjusted trace for keys","");
    ptrs.m_beads_xyflow_location_keys = beads_xyflow_location_keys.Ptr();
    ptrs.m_beads_xyflow_predicted_keys = beads_xyflow_predicted_keys.Ptr();
    ptrs.m_beads_xyflow_corrected_keys = beads_xyflow_corrected_keys.Ptr();

  }
}


void BkgParamH5::IncrementalWrite_xyflow ( bool lastflow )
{
  if ( lastflow ) // do only once at the last flow
  {
    beads_xyflow_corrected.SafeWrite();
    beads_xyflow_predicted.SafeWrite();
    beads_xyflow_amplitude.SafeWrite();
    beads_xyflow_location.SafeWrite();
    beads_xyflow_hplen.SafeWrite();
    beads_xyflow_mm.SafeWrite();
    beads_xyflow_kmult.SafeWrite();
    beads_xyflow_dmult.SafeWrite();
    beads_xyflow_SP.SafeWrite();
    beads_xyflow_R.SafeWrite();
    beads_xyflow_gainSens.SafeWrite();
    beads_xyflow_fittype.SafeWrite();
    beads_xyflow_timeframe.SafeWrite();
    beads_xyflow_residual.SafeWrite();
    beads_xyflow_taub.SafeWrite();
    beads_xyflow_location_keys.SafeWrite();
    beads_xyflow_corrected_keys.SafeWrite();
    beads_xyflow_predicted_keys.SafeWrite();
  }
}


