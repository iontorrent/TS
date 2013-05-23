/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     RecalibrationModel.cpp
//! @ingroup  BaseCaller
//! @brief    RecalibrationModel. Model estimation between simulated predictions and observed measurements

#ifndef RECALIBRATIONMODEL_H
#define RECALIBRATIONMODEL_H

#include <vector>
#include <stdint.h>
#include "OptArgs.h"
#include "BaseCallerUtils.h"

using namespace std;

struct RegionStratification{
  int xMin;
  int xMax;
  int xSpan;
  int xCuts;
  int yMin;
  int yMax;
  int ySpan;
  int yCuts;
  RegionStratification(int xi, int xx, int xs, int yi, int yx, int ys):xMin(xi), xMax(xx), xSpan(xs), yMin(yi), yMax(yx), ySpan(ys){
      xCuts = (xMax - xMin + 2) / xSpan;
      yCuts = (yMax - yMin + 2) / ySpan;
  }
};

class RecalibrationModel {
public:
  RecalibrationModel();
  ~RecalibrationModel();
  void Initialize(OptArgs& opts);
  vector<vector<vector<float> > > * getAs(int x, int y);
  vector<vector<vector<float> > > * getBs(int x, int y);
  bool is_enabled() const { return is_enabled_; }

protected:
  bool is_enabled_;
  int max_hp_calibrated_;
  vector<vector< vector< vector<float> > > > stratifiedAs;
  vector<vector< vector< vector<float> > > > stratifiedBs;
  RegionStratification* stratification_;

};

#endif // RECALIBRATIONMODEL_H
