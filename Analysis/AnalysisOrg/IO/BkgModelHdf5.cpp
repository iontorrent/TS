/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sstream>
#include "BkgModelHdf5.h"
#include "BkgFitterTracker.h"



BkgParamH5::BkgParamH5()
{


  nFlowBlks = 0;
  blocksOfFlow = 0;
  datacube_numflows = 0;
  bead_col = 0;
  bead_row = 0;
  region_total = 0;
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
void MatchedCube::InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, int chunk_size,
                                  char *set_name, char *set_description, const char *param_root )
{
  //printf ( "%s\n",set_name );
  string str;
  source.Init ( col, row, maxflows );
  source.SetRange ( 0,col, 0, row, 0, chunk_size );
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


// set up a basic data cube + matched h5 set
//@TODO: use templates to avoid "INT" duplication
void MatchedCubeInt::InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, int chunk_size, char *set_name, char *set_description,char *param_root )
{
  //printf ( "%s\n",set_name );
  string str;
  source.Init ( col, row, maxflows );
  source.SetRange ( 0,col, 0, row, 0, chunk_size );
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

void RotatingCube::RotateMyCube ( H5File &h5_local_ref, int total_blocks, char *cube_name, char *cube_description )
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


void RotatingCube::InitRotatingCube ( H5File &h5_local_ref, int x, int y, int z, int total_blocks, char *cube_name, char *cube_description )
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
    bead_base_parameter.InitBasicCube ( h5_local_ref, bead_col, bead_row, 5, 5,
                                        "/bead/bead_base_parameters","basic 5: copies, etbR, dmult, gain, deltaTime", "Copies,etbR,dmult,gain,deltaTime" );

    if( verbosity>1 ){ //per flow parameters are only included when debug flag is set
        Amplitude.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows, blocksOfFlow,
                                  "/bead/amplitude", "mean hydrogens per molecule per flow", "" );
        krate_multiplier.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows, blocksOfFlow,
                                         "/bead/kmult", "adjustment to krate", "" );
        // less important?
        bead_dc.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows, blocksOfFlow,
                                "/bead/trace_dc_offset", "additive factor for trace", "" );

        residual_error.InitBasicCube ( h5_local_ref, bead_col, bead_row, datacube_numflows, blocksOfFlow,
                                       "/bead/residual_error", "residual_error", "" );
    }

    // still less important?
    average_error_flow_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks, 1,
        "/bead/average_error_by_block", "avg_err_by_block", "" );
    bead_clonal_compute_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks, 1,
        "/bead/clonal_status_per_block", "clonal_status_by_block", "" );
    bead_corrupt_compute_block.InitBasicCube ( h5_local_ref, bead_col, bead_row, nFlowBlks, 1,
        "/bead/corrupt_status_per_block", "corrupt_status_by_block", "" );   // status markers for beads

  }
  catch ( char * str )
  {
    cout << "Exception raised while creating bead datasets in BkgParamH5::Init(): " << str << '\n';
  }
}



int AddOneCommaString ( string &outstr, char *my_name )
{
  char my_char_string[256];
  sprintf ( my_char_string,"%s,",my_name );
  outstr += my_char_string;
  return ( 1 );
}

int AddOneNoCommaString ( string &outstr, char *my_name )
{
  char my_char_string[256];
  sprintf ( my_char_string,"%s",my_name );
  outstr += my_char_string;
  return ( 1 );
}

int AddCommaStrings ( string &outstr, char *my_name, int numval )
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
}

void RegionNucShapeByName(string &outstr, int &my_total_names)
{
    //nuc_rise_params nuc_shape;
// rise timing parameters
  //float t_mid_nuc[NUMFB];
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc",NUMFB );
// empirically-derived per-nuc modifiers for t_mid_nuc and sigma
  //float t_mid_nuc_delay[NUMNUC];
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc_delay",NUMNUC );
  //float sigma;
  my_total_names+=AddOneCommaString ( outstr,"sigma" );
  //float sigma_mult[NUMNUC];
  my_total_names+=AddCommaStrings ( outstr,"sigma_mult",NUMNUC );

  // refined t_mid_nuc
  //float t_mid_nuc_shift_per_flow[NUMFB]; // note how this is redundant(!)
  my_total_names+=AddCommaStrings ( outstr,"t_mid_nuc_shift_per_flow",NUMFB );

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

int RegionParametersByName ( string &outstr )
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
  //float darkness[blocksOfFlow];
  my_total_names+=AddCommaStrings ( outstr,"darkness",NUMFB );
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

  int my_total_names = RegionParametersByName ( region_names_str );
  int numParam = my_total_names;

  regional_params.InitBasicCube ( h5_local_ref, region_total, numParam, nFlowBlks, 1,
                                  "/region/region_param/misc", "regional parameters by compute block", region_names_str.c_str() );
                                  
  string nuc_shape_names;
  int nucshapeParam = 0;
  nuc_shape_names="";
  RegionNucShapeByName(nuc_shape_names,nucshapeParam);
  nuc_shape_params.InitBasicCube ( h5_local_ref, region_total, nucshapeParam, nFlowBlks, 1,
                                  "/region/region_param/nuc_shape", "nuc shape parameters by compute block", nuc_shape_names.c_str() );
  string enzymatics_names;
  int enzymaticsParam = 0;
  enzymatics_names="";
  RegionEnzymaticsByName(enzymatics_names,enzymaticsParam);
  enzymatics_params.InitBasicCube ( h5_local_ref, region_total, enzymaticsParam, nFlowBlks, 1,
                                  "/region/region_param/enzymatics", "enzyme parameters by compute block", enzymatics_names.c_str() );
  string buffering_names;
  int bufferingParam = 0;
  buffering_names="";
  RegionBufferingByName(buffering_names,bufferingParam);
  buffering_params.InitBasicCube ( h5_local_ref, region_total, bufferingParam, nFlowBlks, 1,
                                  "/region/region_param/buffering", "buffering regional models by compute block", buffering_names.c_str() );

  string derived_str = "midNucTime_0,midNucTime_1,midNucTime_2,midNucTime_3,sigma_0,sigma_1,sigma_2,sigma_3";
  derived_params.InitBasicCube ( h5_local_ref, region_total, 8, nFlowBlks, 1,
                                 "/region/derived_param", "derived parameters by compute block", derived_str.c_str() );
}

void BkgParamH5::InitRegionDebugBead( H5File &h5_local_ref)
{
      region_debug_bead.InitBasicCube ( h5_local_ref, region_total, 5, 1, 1,
                                      "/region/debug_bead/basic", "basic 5: copies, etbR, dmult, gain, deltaTime", "Copies,etbR,dmult,gain,deltaTime" );
    region_debug_bead_location.InitBasicCube ( h5_local_ref, region_total, 2, 1, 1,
        "/region/debug_bead/location", "col, row", "col,row" );
    region_debug_bead_amplitude_krate.InitBasicCube ( h5_local_ref, region_total, 2, datacube_numflows, blocksOfFlow,
        "/region/debug_bead/amplitude_krate", "Amplitude/krate_multiplier by numflows", "" );

    region_debug_bead_predicted.InitBasicCube ( h5_local_ref, region_total, MAX_COMPRESSED_FRAMES, datacube_numflows, blocksOfFlow,
        "/region/debug_bead/predicted", "predicted trace for debug bead", "" );
    region_debug_bead_corrected.InitBasicCube ( h5_local_ref, region_total, MAX_COMPRESSED_FRAMES, datacube_numflows, blocksOfFlow,
       "/region/debug_bead/corrected", "background-adjusted trace for debug bead", "" );
    region_debug_bead_xtalk.InitBasicCube ( h5_local_ref, region_total, MAX_COMPRESSED_FRAMES, datacube_numflows, blocksOfFlow,
       "/region/debug_bead/xtalk", "local xtalk estimated trace for debug bead", "" );
}



void BkgParamH5::TryInitRegionParams ( H5File &h5_local_ref, ImageSpecClass &my_image_spec )
{
  ///------------------------------------------------------------------------------------------------------------
  /// region parameters
  ///------------------------------------------------------------------------------------------------------------


  try
  {

    empty_trace.InitBasicCube ( h5_local_ref, region_total, my_image_spec.uncompFrames, datacube_numflows, blocksOfFlow,
                                "/region/empty_trace", "empty trace per frame per flow","" );
    empty_dc.InitBasicCube ( h5_local_ref, region_total, 1, datacube_numflows, blocksOfFlow,
                             "/region/empty_dc", "empty dc offset per flow","" );

    // write once items
    dark_matter_trace.InitBasicCube ( h5_local_ref, region_total, NUMNUC, MAX_COMPRESSED_FRAMES, MAX_COMPRESSED_FRAMES,
                                      "/region/darkMatter/missingMass", "dark matter trace by nucleotide","" );
    darkness_val.InitBasicCube ( h5_local_ref, region_total, blocksOfFlow, 1,1,
                                 "/region/darkMatter/darkness","darkness per region per flowbuffer", "" );

    region_init_val.InitBasicCube ( h5_local_ref, region_total, 2, 1, 1,
                                    "/region/region_init_param", "starting values", "t_mid_nuc_start,sigma_start" );

    time_compression.InitBasicCube ( h5_local_ref, region_total, 4, MAX_COMPRESSED_FRAMES, MAX_COMPRESSED_FRAMES,
                                     "/region/time_compression", "Time compression", "frameNumber,deltaFrame,frames_per_point,npt" );

    region_offset_val.InitBasicCube ( h5_local_ref, region_total, 2, 1, 1,
                                      "/region/region_location", "smallest corner of region aka offset of beads", "col,row" );

    //this is a data-cube per compute block, so need the 'rotating cube' formulation
    emphasis_val.InitRotatingCube ( h5_local_ref, region_total, MAX_POISSON_TABLE_COL,MAX_COMPRESSED_FRAMES,nFlowBlks,
                                    "/region/emphasis","Emphasis vector by HPLen in compute blocks" );

    InitRegionalParamCube ( h5_local_ref );
    InitRegionDebugBead( h5_local_ref);
    // store a "debug bead" as a representative for a region

  }
  catch ( char * str )
  {
    cout << "Exception raised while creating region datasets in BkgParamH5::Init(): " << str << '\n';
  }
}

void BkgParamH5::ConstructOneFile ( H5File &h5_local_ref, string &hgLocalFile, string &local_results,char *my_name )
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


void BkgParamH5::Init ( char *results_folder, SpatialContext &loc_context, ImageSpecClass &my_image_spec, int numFlows, int write_params_flag )
{
  if ( write_params_flag>0 )
  {
    local_results_directory=results_folder;

    blocksOfFlow = NUMFB;
    datacube_numflows = numFlows;
    nFlowBlks = ceil ( float ( datacube_numflows ) /blocksOfFlow );
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


void BkgParamH5::IncrementalWriteParam ( DataCube<float> &cube, H5DataSet *set, int flow )
{
  // please do not do this(!)
  // there needs to be >one< master routine that controls when we write to files
  // not several independent implementations of the same logic

  if ( set!=NULL )
  {
    WriteOneFlowBlock ( cube,set,flow,blocksOfFlow );
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

void BkgParamH5::IncrementalWrite ( int flow, bool last_flow )
{
  // single check: are we done with a compute block?
  // try to isolate logic as much as possible - we should only see one check for writing, ever

  if ( CheckFlowForWrite ( flow,last_flow ) )
  {
    // which compute block are we?
    int iBlk = CurComputeBlock ( flow );
    IncrementalWriteBeads ( flow,iBlk );
    IncrementalWriteRegions ( flow,iBlk );
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


void BkgParamH5::Close()
{
  CloseBeads();
  CloseRegion();
  if ( hgBeadDbgFile.length() >0 )
  {
    cout << "bgParamH5 output: " << hgBeadDbgFile << endl;
  }
  if ( hgRegionDbgFile.length() >0 )
  {
    cout << "bgParamH5 output: " << hgRegionDbgFile << endl;
  }
}

void BkgParamH5::saveBeadPointers()
{
  ptrs.mAmpl = Amplitude.Ptr();
  ptrs.mBeadInitParam = bead_base_parameter.Ptr();
  ptrs.mBeadDC = bead_dc.Ptr();
  ptrs.mKMult = krate_multiplier.Ptr();
  ptrs.mResError = residual_error.Ptr();
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





