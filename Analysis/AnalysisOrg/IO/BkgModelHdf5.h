/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODELHDF5_H
#define BKGMODELHDF5_H

#include <vector>
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "DataCube.h"
#include "H5File.h"
#include "BkgDataPointers.h"


using namespace std;

// datacube is written to by the source as a transfer mechanism
// then h5_set is used to dump it to disk
class MatchedCube
{
  public:
    DataCube<float> source;
    H5DataSet *h5_set;

    // build a basic matched cube
    void InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, int chunk_size, char *set_name, char *set_description,const char *param_root );
    void Close();
    DataCube<float> *Ptr()
    {
      if ( h5_set!=NULL )
        return ( &source );
      else
        return ( NULL );
    };
    void SafeWrite();
    MatchedCube();
};

//@TODO: template
class MatchedCubeInt
{
  public:
    DataCube<int> source;
    H5DataSet *h5_set;

    // build a basic matched cube
    void InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, int chunk_size, char *set_name, char *set_description,char *param_root );
    void Close();
    DataCube<int> *Ptr()
    {
      if ( h5_set!=NULL )
        return ( &source );
      else
        return ( NULL );
    };
    void SafeWrite();
    MatchedCubeInt();
};

// handle 1 cube per compute block smoothly
class RotatingCube{
  public:
    DataCube<float> source;
    vector<H5DataSet * > h5_vec;
    void RotateMyCube(H5File &h5_local_ref, int total_blocks, char *cube_name, char *cube_description);
    void InitRotatingCube(H5File &h5_local_ref, int x, int y, int z, int total_blocks, char *cube_name, char *cube_description);
    void Close();
    DataCube<float> *Ptr()
    {
      if ( h5_vec.size()>0 )
        return ( &source );
      else
        return ( NULL );
    };
    void SafeWrite(int iBlk);
};

class BkgParamH5
{
  public:
    BkgParamH5();
    ~BkgParamH5()
    {
      Close();
    }

    void Init ( char *results_folder,SpatialContext &loc_context,   ImageSpecClass &my_image_spec,int numFlows, int write_params_flag );

    void IncrementalWriteParam ( DataCube<float> &cube, H5DataSet *set, int flow );
    void WriteOneBlock ( DataCube<float> &cube, H5DataSet *set, int iBlk );
    void WriteOneBlock ( DataCube<int> &cube, H5DataSet *set, int iBlk );
    void WriteOneFlowBlock ( DataCube<float> &cube, H5DataSet *set, int flow, int chunksize );
    void IncrementalWriteBeads ( int flow,int iBlk );
    void IncrementalWriteRegions(int flow, int iBlk);
    
    void IncrementalWrite (  int flow, bool last_flow ); // this is the interface to trigger a write

    void Close();
    void CloseBeads();
    void CloseRegion();


    void TryInitBeads ( H5File &h5_local_ref, int verbosity );

    void InitRegionEmphasisVector ( H5File &h5_local_ref );

    void InitRegionalParamCube(H5File &h5_local_ref);
    void InitRegionDebugBead (H5File &h5_local_ref);
    void TryInitRegionParams ( H5File &h5_local_ref, ImageSpecClass &my_image_spec );

  public: // should be private eventually

    // the idea here is to "componentize" the data sets so we can
    // easily add or remove data as we're tuning up the structures.
    
    // these are the components for beads
    MatchedCube Amplitude;
    MatchedCube bead_dc;
    MatchedCube krate_multiplier;
    MatchedCube residual_error;
    MatchedCube bead_base_parameter;
    MatchedCube average_error_flow_block;

    MatchedCubeInt bead_clonal_compute_block;
    MatchedCubeInt bead_corrupt_compute_block;


    //regional params stored in data cubes for writing to files
    MatchedCube dark_matter_trace;
    MatchedCube darkness_val;
    MatchedCube empty_trace;
    MatchedCube empty_dc;
    MatchedCube region_init_val;
    MatchedCubeInt region_offset_val;
    
    MatchedCube regional_params;
    MatchedCube nuc_shape_params;
    MatchedCube enzymatics_params;
    MatchedCube buffering_params;
    MatchedCube derived_params;
    MatchedCube time_compression;
    
    MatchedCube region_debug_bead;
    MatchedCube region_debug_bead_amplitude_krate;
    MatchedCube region_debug_bead_predicted;
    MatchedCube region_debug_bead_corrected;
    MatchedCube region_debug_bead_xtalk;
    
    MatchedCubeInt region_debug_bead_location;
    
    RotatingCube emphasis_val;

// interface to external world using datacubes linked to hdf5 sets
    BkgDataPointers ptrs;

    
    void savePointers();
    void saveRegionPointers();
    void saveBeadPointers();

    std::string getBeadFilename()
    {
      return hgBeadDbgFile;
    }
    void ConstructOneFile ( H5File &h5_local_ref, string &hgLocalFile, string &local_results, char *my_name );
  private:
    H5File h5BeadDbg;
    std::string hgBeadDbgFile;
    // two files to control size
    H5File h5RegionDbg;
    std::string hgRegionDbgFile;

    std::string local_results_directory;

    // derived parameters controlling the file
    int blocksOfFlow;
    int nFlowBlks;
    //beads
    int datacube_numflows;
    // usually the whole matrix, may be by region if we're sampling
    int bead_col;
    int bead_row;
    // region
    int region_total;
};

#define SAVE_DATA_EVERY_BLOCK 1


#endif // BKGMODELHDF5_H
