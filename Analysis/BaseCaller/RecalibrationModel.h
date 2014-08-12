/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     RecalibrationModel.cpp
//! @ingroup  BaseCaller
//! @brief    RecalibrationModel. Model estimation between simulated predictions and observed measurements

#ifndef RECALIBRATIONMODEL_H
#define RECALIBRATIONMODEL_H

#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <SystemMagicDefines.h>

#include "OptArgs.h"
#include "json/json.h"


using namespace std;

class RegionStratification{
  public:
  int xMin;
  int xMax;
  int xSpan;
  int xCuts;
  int yMin;
  int yMax;
  int ySpan;
  int yCuts;
  RegionStratification(){
    xMin = xMax = yMin = yMax = 0;
    xSpan = ySpan = 1;
    yCuts = xCuts = 0;
  };
  RegionStratification(int xi, int xx, int xs, int yi, int yx, int ys):xMin(xi), xMax(xx), xSpan(xs), yMin(yi), yMax(yx), ySpan(ys){
      xCuts = (xMax - xMin + 2) / xSpan;
      yCuts = (yMax - yMin + 2) / ySpan;
  };
  void SetupRegion(int _xMin, int _xMax, int _xSpan, int _yMin, int _yMax, int _ySpan){
    xMin = _xMin;
    xMax = _xMax;
    xSpan = _xSpan;
    yMin = _yMin;
    yMax = _yMax;
    ySpan = _ySpan;
      xCuts = (xMax - xMin + 2) / xSpan;
      yCuts = (yMax - yMin + 2) / ySpan;    
  };
  int OffsetRegion(int x, int y) const {
    int offsetRegion = (y - yMin)/ySpan + (x -xMin)/xSpan * yCuts;
    return(offsetRegion);
  };
};

class MultiAB{
  public:
    const vector<vector<vector<float> > > * aPtr;
    const vector<vector<vector<float> > > * bPtr;

    MultiAB(){aPtr=0; bPtr = 0;};
    void Null(){aPtr=0; bPtr = 0;};
    bool Valid(){return (aPtr != 0 && bPtr != 0);};
};

class RecalibrationModel {
public:
  
  RecalibrationModel();
  ~RecalibrationModel();

  const vector<vector<vector<float> > > * getAs(int x, int y) const;
  const vector<vector<vector<float> > > * getBs(int x, int y) const;

  // Read command line arguments and then call InitializeModel
  void Initialize(OptArgs& opts);

  // Model text file and hp threshold supplied as input variables
  void InitializeModel(string model_file_name, int model_threshold);

  void getAB(MultiAB &multi_ab, int x, int y) const;

  bool is_enabled() const { return is_enabled_; };

  void suppressEnabled() { is_enabled_ = false;};

  void InitializeFromJSON(Json::Value &recal_param, string &my_block_key, bool spam_enabled, int over_flow_protect);

  void SetupStratification(int flowStart, int flowEnd, int flowSpan, 
                                             int xMin, int xMax, int xSpan, 
                                             int yMin, int yMax, int ySpan, int max_hp_calibrated);

  void FillIndexes(int offsetRegion, int nucInd, int refHP, int flowStart, int flowEnd, float paramA, float paramB);


//protected:
  bool is_enabled_;
  int max_hp_calibrated_;
  int recalModelHPThres;
  vector<vector< vector< vector<float> > > > stratifiedAs;
  vector<vector< vector< vector<float> > > > stratifiedBs;
  RegionStratification stratification;

};

#endif // RECALIBRATIONMODEL_H
