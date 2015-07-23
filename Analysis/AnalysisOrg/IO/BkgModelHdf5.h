/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODELHDF5_H
#define BKGMODELHDF5_H

#include <vector>
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "DataCube.h"
#include "IonH5File.h"
#include "BkgDataPointers.h"


//using namespace std;

// datacube is written to by the source as a transfer mechanism
// then h5_set is used to dump it to disk
class MatchedCube
{
  public:
    DataCube<float> source;
    H5DataSet *h5_set;

    // build a basic matched cube
    void InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, const char *set_name, const char *set_description, const char *param_root );
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
    //hid_t attribute_id;
    //hid_t get_AttributeId() {return attribute_id;}
};

//@TODO: template
class MatchedCubeInt
{
  public:
    DataCube<int> source;
    H5DataSet *h5_set;

    // build a basic matched cube
    void InitBasicCube ( H5File &h5_local_ref, int col, int row, int maxflows, const char *set_name, const char *set_description, const char *param_root );
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
    //hid_t attribute_id;
    //hid_t get_AttributeId() {return attribute_id;}
};

// handle 1 cube per compute block smoothly
class RotatingCube{
  public:
    DataCube<float> source;
    std::vector<H5DataSet * > h5_vec;
    void RotateMyCube(H5File &h5_local_ref, int total_blocks, const char *cube_name, const char *cube_description);
    void InitRotatingCube(H5File &h5_local_ref, int x, int y, int z, int total_blocks, const char *cube_name, const char *cube_description);
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

    void Init ( const char *results_folder, 
        const SpatialContext &loc_context, 
        const ImageSpecClass &my_image_spec,
        int numFlows,         // Total number of flows in the universe.
        int write_params_flag, 
        int _max_frames,
        int flow_block_size,         // Maximum number of flows in a flow block.
        int num_flow_blocks   // Total number of flow blocks. (used to be ceil( numFlows/flow_max).
      );

    void IncrementalWriteParam ( DataCube<float> &cube, H5DataSet *set, int flow );
    void IncrementalWriteParam ( DataCube<int> &cube, H5DataSet *set, int flow );
    void WriteOneBlock ( DataCube<float> &cube, H5DataSet *set, int iBlk );
    void WriteOneBlock ( DataCube<int> &cube, H5DataSet *set, int iBlk );
    void WriteOneFlowBlock ( DataCube<float> &cube, H5DataSet *set, int flow, int chunksize );
    void WriteOneFlowBlock ( DataCube<int> &cube, H5DataSet *set, int flow, int chunksize );
    void IncrementalWriteBeads ( int flow,int iBlk );
    void IncrementalWriteRegions(int flow, int iBlk);

    // this is the interface to trigger a write
    void IncrementalWrite ( int flow, bool last_flow, FlowBlockSequence::const_iterator flow_block, 
                            int flow_block_id );

    void Close();
    void CloseBeads();
    void CloseRegion();
    void CloseTraceXYFlow();


    void TryInitBeads ( H5File &h5_local_ref, int verbosity );
    void InitRegionEmphasisVector ( H5File &h5_local_ref );
    void InitRegionalParamCube(H5File &h5_local_ref);
    void InitRegionDebugBead (H5File &h5_local_ref);
    void TryInitRegionParams ( H5File &h5_local_ref, const ImageSpecClass &my_image_spec );

    // all beads in the best region
    void Init2 (int write_params_flag,int nBeads_live,const Region *,int nRegionCenters);
    void TryInitBeads_BestRegion ( H5File &h5_local_ref, int nBeads_live,Region *);
    void InitBeads_BestRegion (H5File &h5_local_ref, int nBeads_live, const Region *);
    void InitBeads_RegionCenter (H5File &h5_local_ref, int nBeads_live);
    void IncrementalWriteBestRegion(int flow, bool lastflow);
    void IncrementalWriteRegionCenter(int flow, bool lastflow);

  public: // should be private eventually

    int max_frames;
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
    //By the way, if you are working on hdf5 parameters.
    //Is it possible to split parameters "derived_param", "enzymatics", 'nuc_shape', misc, buffering into separate parameters.
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

//  beads in the bestRegion
    MatchedCubeInt beads_bestRegion_location;
    MatchedCubeInt beads_bestRegion_fittype;
    MatchedCube beads_bestRegion_predicted;
    MatchedCube beads_bestRegion_corrected;
    MatchedCube beads_bestRegion_amplitude;
    MatchedCube beads_bestRegion_residual;
    MatchedCube beads_bestRegion_kmult;
    MatchedCube beads_bestRegion_dmult;
    MatchedCube beads_bestRegion_SP;
    MatchedCube beads_bestRegion_R;
    MatchedCube beads_bestRegion_gainSens;
    MatchedCube beads_bestRegion_timeframe;
    MatchedCube beads_bestRegion_taub;


    //  beads in the regionCenter
    MatchedCubeInt beads_regionCenter_location;
    MatchedCubeInt beads_regionCenter_fittype;
    MatchedCube beads_regionCenter_predicted;
    MatchedCube beads_regionCenter_corrected;
    MatchedCube beads_regionCenter_amplitude;
    MatchedCube beads_regionCenter_residual;
    MatchedCube beads_regionCenter_kmult;
    MatchedCube beads_regionCenter_dmult;
    MatchedCube beads_regionCenter_SP;
    MatchedCube beads_regionCenter_R;
    MatchedCube beads_regionCenter_gainSens;
    MatchedCube beads_regionCenter_timeframe;
    MatchedCube beads_regionCenter_taub;
    MatchedCube beads_regionCenter_regionParams;

//  beads specified in the sse/xyflow/rcflow file
    void InitBeads_xyflow(int write_params_flag, HashTable_xyflow &xyf_hash);
    void saveBeads_xyflowPointers();
    void IncrementalWrite_xyflow(bool lastflow);
    MatchedCubeInt beads_xyflow_fittype;
    MatchedCubeInt beads_xyflow_location;
    MatchedCubeInt beads_xyflow_hplen;
    MatchedCubeInt beads_xyflow_mm;
    MatchedCube beads_xyflow_predicted;
    MatchedCube beads_xyflow_corrected;
    MatchedCube beads_xyflow_amplitude;
    MatchedCube beads_xyflow_kmult;
    MatchedCube beads_xyflow_dmult;
    MatchedCube beads_xyflow_SP;
    MatchedCube beads_xyflow_R;
    MatchedCube beads_xyflow_gainSens;
    MatchedCube beads_xyflow_timeframe;
    MatchedCube beads_xyflow_residual;
    MatchedCube beads_xyflow_taub;
    // key traces corresponding to xyflow
    MatchedCube beads_xyflow_predicted_keys;
    MatchedCube beads_xyflow_corrected_keys;
    MatchedCubeInt beads_xyflow_location_keys;

    RotatingCube emphasis_val;

// interface to external world using datacubes linked to hdf5 sets
    BkgDataPointers ptrs;
    
    void savePointers();
    void saveRegionPointers();
    void saveBeadPointers();
    void saveBestRegionPointers();
    void saveRegionCenterPointers();
    std::string getBeadFilename() {return hgBeadDbgFile;}
    void ConstructOneFile ( H5File &h5_local_ref, std::string &hgLocalFile, std::string &local_results, const char *my_name );

  private:
    H5File h5BeadDbg;
    std::string hgBeadDbgFile;
    // two files to control size
    H5File h5RegionDbg;
    std::string hgRegionDbgFile;
    H5File h5TraceDbg;
    std::string hgTraceDbgFile;

    std::string local_results_directory;

    // derived parameters controlling the file
    int flow_block_size;       // Otherwise known as flow_max.
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
