/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BKGDATAPOINTERS_H
#define BKGDATAPOINTERS_H

#include <cstddef>  // NULL defined here
//#include <armadillo>
#include "DataCube.h"
#include "BkgControlOpts.h"


class BkgDataPointers {
 public:
  BkgDataPointers()
  {
      mAmpl = NULL;
      mBeadDC = NULL;
      mBeadDC_bg = NULL;
      mKMult = NULL;
      mResError = NULL;
      mBeadInitParam = NULL;
      mBeadFblk_avgErr = NULL;
      mBeadFblk_clonal = NULL;
      mBeadFblk_corrupt = NULL;
      
      // region
      mDarkOnceParam = NULL;
      mDarknessParam = NULL;
      mEmphasisParam = NULL;
      
      mEmptyOnceParam = NULL;
      mRegionInitParam = NULL;
      mRegionOffset = NULL;
      m_regional_param = NULL;
      m_nuc_shape_param = NULL;
      m_enzymatics_param = NULL;
      m_buffering_param = NULL;
      m_derived_param = NULL;
      m_region_debug_bead = NULL;
      m_region_debug_bead_ak = NULL;
      m_region_debug_bead_location = NULL;
      m_region_debug_bead_predicted = NULL;
      m_region_debug_bead_corrected = NULL;
      m_region_debug_bead_xtalk = NULL;
      m_time_compression = NULL;

      m_beads_bestRegion_location = NULL;
      m_beads_bestRegion_predicted = NULL;
      m_beads_bestRegion_corrected = NULL;
      m_beads_bestRegion_amplitude = NULL;
      m_beads_bestRegion_residual = NULL;
      m_beads_bestRegion_kmult = NULL;
      m_beads_bestRegion_dmult = NULL;
      m_beads_bestRegion_SP = NULL;
      m_beads_bestRegion_R = NULL;
      m_beads_bestRegion_gainSens = NULL;
      m_beads_bestRegion_fittype = NULL;
      m_beads_bestRegion_timeframe = NULL;
      m_beads_bestRegion_taub = NULL;

      m_beads_regionCenter_location = NULL;
      m_beads_regionCenter_predicted = NULL;
      m_beads_regionCenter_corrected = NULL;
      m_beads_regionCenter_amplitude = NULL;
      m_beads_regionCenter_residual = NULL;
      m_beads_regionCenter_kmult = NULL;
      m_beads_regionCenter_dmult = NULL;
      m_beads_regionCenter_SP = NULL;
      m_beads_regionCenter_R = NULL;
      m_beads_regionCenter_gainSens = NULL;
      m_beads_regionCenter_fittype = NULL;
      m_beads_regionCenter_timeframe = NULL;
      m_beads_regionCenter_taub = NULL;
      m_beads_regionCenter_regionParams = NULL;

      m_beads_xyflow_predicted = NULL;
      m_beads_xyflow_corrected = NULL;
      m_beads_xyflow_amplitude = NULL;
      m_beads_xyflow_residual = NULL;
      m_beads_xyflow_kmult = NULL;
      m_beads_xyflow_dmult = NULL;
      m_beads_xyflow_SP = NULL;
      m_beads_xyflow_R = NULL;
      m_beads_xyflow_gainSens = NULL;
      m_beads_xyflow_fittype = NULL;
      m_beads_xyflow_location = NULL;
      m_beads_xyflow_hplen = NULL;
      m_beads_xyflow_mm = NULL;
      m_beads_xyflow_timeframe = NULL;
      m_beads_xyflow_taub = NULL;

      m_xyflow_hashtable = NULL;

      // key signals
      m_beads_xyflow_predicted_keys = NULL;
      m_beads_xyflow_corrected_keys = NULL;
      m_beads_xyflow_location_keys = NULL;

  }

  ~BkgDataPointers() {};

public: // functions to set the values
   void copyCube_element(DataCube<int> *ptr, int x, int y, int j, int v);
   void copyCube_element(DataCube<float> *ptr, int x, int y, int j, float v);
//  void copyMatrix_element(arma::Mat<float> *ptr, int x, int j, float v);
//  void copyMatrix_element(arma::Mat<int> *ptr, int x, int j, int v);

public: // should be private eventually and use set/get to access them
  // beads(!)
  DataCube<float> *mAmpl;
  DataCube<float> *mBeadDC;
  DataCube<float> *mKMult;
  DataCube<float> *mResError;
  DataCube<float> *mBeadInitParam;
  DataCube<float> *mBeadFblk_avgErr;
  DataCube<int> *mBeadFblk_clonal;
  DataCube<int> *mBeadFblk_corrupt;
  
// region (!)
  DataCube<float> *mDarkOnceParam;
  DataCube<float> *mDarknessParam;
  DataCube<float> *mEmptyOnceParam;
  DataCube<float> *mBeadDC_bg;
  DataCube<float> *m_time_compression;
  
  DataCube<float> *m_regional_param;
  DataCube<float> *m_nuc_shape_param;
  DataCube<float> *m_enzymatics_param;
  DataCube<float> *m_buffering_param;
  DataCube<float> *m_derived_param;
  DataCube<float> *m_region_debug_bead;
  DataCube<float> *m_region_debug_bead_ak;
  DataCube<float> *m_region_debug_bead_predicted;
  DataCube<float> *m_region_debug_bead_corrected;
  DataCube<float> *m_region_debug_bead_xtalk;
  DataCube<int> *m_region_debug_bead_location;
  
  DataCube<float> *mRegionInitParam;
  
  DataCube<float> *mEmphasisParam;
 
  DataCube<int> *mRegionOffset;

  // bestRegion beads
  DataCube<int> *m_beads_bestRegion_location;
  DataCube<int> *m_beads_bestRegion_fittype;
  DataCube<float> *m_beads_bestRegion_corrected;
  DataCube<float> *m_beads_bestRegion_predicted;
  DataCube<float> *m_beads_bestRegion_amplitude;
  DataCube<float> *m_beads_bestRegion_residual;
  DataCube<float> *m_beads_bestRegion_kmult;
  DataCube<float> *m_beads_bestRegion_dmult;
  DataCube<float> *m_beads_bestRegion_SP;
  DataCube<float> *m_beads_bestRegion_R;
  DataCube<float> *m_beads_bestRegion_gainSens;
  DataCube<float> *m_beads_bestRegion_timeframe;
  DataCube<float> *m_beads_bestRegion_taub;

  // regionCenter beads
  DataCube<int> *m_beads_regionCenter_location;
  DataCube<int> *m_beads_regionCenter_fittype;
  DataCube<float> *m_beads_regionCenter_corrected;
  DataCube<float> *m_beads_regionCenter_predicted;
  DataCube<float> *m_beads_regionCenter_amplitude;
  DataCube<float> *m_beads_regionCenter_residual;
  DataCube<float> *m_beads_regionCenter_kmult;
  DataCube<float> *m_beads_regionCenter_dmult;
  DataCube<float> *m_beads_regionCenter_SP;
  DataCube<float> *m_beads_regionCenter_R;
  DataCube<float> *m_beads_regionCenter_gainSens;
  DataCube<float> *m_beads_regionCenter_timeframe;
  DataCube<float> *m_beads_regionCenter_taub;
  DataCube<float> *m_beads_regionCenter_regionParams;

  // traceOutput for positions specified in sse/xyf/rcf files
  DataCube<float> *m_beads_xyflow_predicted;
  DataCube<float> *m_beads_xyflow_corrected;
  DataCube<float> *m_beads_xyflow_amplitude;
  DataCube<float> *m_beads_xyflow_kmult;
  DataCube<float> *m_beads_xyflow_dmult;
  DataCube<float> *m_beads_xyflow_SP;
  DataCube<float> *m_beads_xyflow_R;
  DataCube<float> *m_beads_xyflow_gainSens;
  DataCube<float> *m_beads_xyflow_timeframe;
  DataCube<float> *m_beads_xyflow_residual;
  DataCube<int> *m_beads_xyflow_location;
  DataCube<int> *m_beads_xyflow_hplen;
  DataCube<int> *m_beads_xyflow_mm;
  DataCube<int> *m_beads_xyflow_fittype;
  DataCube<float> *m_beads_xyflow_taub;

  HashTable_xyflow *m_xyflow_hashtable;
  int id_xy(int x, int y, HashTable_xyflow *xyf_hash) {return (xyf_hash->id_xy(x,y));};
  int id_rc(int r, int c, HashTable_xyflow *xyf_hash) {return (xyf_hash->id_rc(r,c));};
  int id_xyflow(int x, int y, int flow, HashTable_xyflow *xyf_hash) {return (xyf_hash->id_xyflow(x,y,flow));};
  int id_rcflow(int r, int c, int flow, HashTable_xyflow *xyf_hash) {return (xyf_hash->id_rcflow(r,c,flow));};
  int mm_xyflow(int x, int y, int flow, HashTable_xyflow *xyf_hash) {return (xyf_hash->mm_xyflow(x,y,flow));};
  std::string hp_xyflow(int x, int y, int flow, HashTable_xyflow *xyf_hash) {return (xyf_hash->hp_xyflow(x,y,flow));};

  // key signals
  DataCube<float> *m_beads_xyflow_predicted_keys;
  DataCube<float> *m_beads_xyflow_corrected_keys;
  DataCube<int> *m_beads_xyflow_location_keys;
};




#endif // BKGDATAPOINTERS_H
