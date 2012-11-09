/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BKGDATAPOINTERS_H
#define BKGDATAPOINTERS_H

#include <cstddef>  // NULL defined here
//#include <armadillo>
#include "DataCube.h"


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

};




#endif // BKGDATAPOINTERS_H
