/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BKGDATAPOINTERS_H
#define BKGDATAPOINTERS_H

#include <cstddef>  // NULL defined here
#include <armadillo>
#include "DataCube.h"

#include "BkgMagicDefines.h"


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
      mDarkOnceParam = NULL;
      mDarknessParam = NULL;
      mEmptyOnceParam = NULL;
      mRegionInitParam = NULL;
      mEmphasisParam = NULL;
      mBeadFblk_avgErr = NULL;
      mBeadFblk_clonal = NULL;
      mBeadFblk_corrupt = NULL;
  }

  ~BkgDataPointers() {};

public: // functions to set the values
  void copyCube_element(DataCube<float> *ptr, int x, int y, int j, float v);
  void copyMatrix_element(arma::Mat<float> *ptr, int x, int j, float v);
  void copyMatrix_element(arma::Mat<int> *ptr, int x, int j, int v);

public: // should be private eventually and use set/get to access them
  DataCube<float> *mAmpl;
  DataCube<float> *mBeadDC;
  DataCube<float> *mKMult;
  DataCube<float> *mResError;
  DataCube<float> *mBeadInitParam;
  DataCube<float> *mDarkOnceParam;
  DataCube<float> *mEmptyOnceParam;
  DataCube<float> *mEmphasisParam;
  arma::Mat<float> *mDarknessParam;
  arma::Mat<float> *mRegionInitParam;

  arma::Mat<float> *mBeadFblk_avgErr;
  arma::Mat<int> *mBeadFblk_clonal;
  arma::Mat<int> *mBeadFblk_corrupt;
  arma::Mat<int> *mRegionOffset;
  arma::Mat<float> *mBeadDC_bg;

};


#endif // BKGDATAPOINTERS_H
