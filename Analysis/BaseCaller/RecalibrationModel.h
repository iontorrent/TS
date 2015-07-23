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
#include "BaseCallerUtils.h"

#include "OptArgs.h"
#include "json/json.h"


using namespace std;

// -------------------------------------------------------------------
// Unified region and offset handling fort our two calibration schemes

class RegionStratification{

  public:
  int xMin;     // inclusive
  int xMax;     // inclusive - closed interval
  int xSpan;    // calculated as [xSpan = (xMax - xMin) / xCuts +1;]
  int xCuts;    // inferred through xMax, xMin, and XCuts
  int yMin;     // inclusive
  int yMax;     // inclusive - closed interval
  int ySpan;    // calculated as [ySpan = (yMax - yMin) / yCuts +1;]
  int yCuts;    // inferred through xMax, yMin, and XSpan

  int flowMin;  // inclusive
  int flowMax;  // inclusive - closed interval
  int flowSpan; // calculated as [flowSpan = (numFlows-1) / flowCuts +1;]
  int flowCuts; // here inferred through xMax, yMin, and XSpan

  RegionStratification(){
    xMin = xMax = yMin = yMax = flowMin = flowMax = 0;
    xSpan = ySpan = 1;
    yCuts = xCuts = flowCuts = flowSpan = 1;
    regions_set_ = flows_set_ = false;
  };

  RegionStratification(int xi, int xx, int xs, int yi, int yx, int ys):xMin(xi), xMax(xx), xSpan(xs), yMin(yi), yMax(yx), ySpan(ys){
    xCuts = (xMax - xMin) / xSpan +1;
    yCuts = (yMax - yMin) / ySpan +1;
    regions_set_ = true;
    flowMin = flowMax = 0;
    flowCuts = flowSpan = 1;
    flows_set_ = false;
  };

  void SetupChipRegions(int _xMin, int _xMax, int _xSpan, int _yMin, int _yMax, int _ySpan){
    xMin  = _xMin;
    xMax  = _xMax;
    xSpan = _xSpan;
    yMin  = _yMin;
    yMax  = _yMax;
    ySpan = _ySpan;
    xCuts = (xMax - xMin) / xSpan +1;
    yCuts = (yMax - yMin) / ySpan +1;
    regions_set_ = true;
  };

  void SetupFlowRegions(int _flowMin,int _flowMax, int _flowSpan){
    flowMin  = _flowMin;
    flowMax  = _flowMax;
	flowSpan = _flowSpan;
    flowCuts = (flowMax - flowMin) / flowSpan +1;
    flows_set_ = true;
  }

  int OffsetRegion(const int &x,const int &y) const {

    if (not regions_set_)
      return -1;
	if (x<xMin or x>xMax or y<yMin or y>yMax)
	  return -1;

    return ((y - yMin)/ySpan + ((x -xMin)/xSpan * yCuts));
  };

  int GetRegionIndex(const int &nuc_idx,const int &flow, const int &offset_region) const {
    if (not flows_set_)
	  return -1;
    if (offset_region < 0 or nuc_idx<0 or nuc_idx>3 or flow < 0 or flow > flowMax)
      return -1;

    return (nuc_idx + 4 * (flow/flowSpan + offset_region*flowCuts));
  }

  int GetRegionIndex(const int &nuc, const int &flow,const int &x, const int &y) const {
    return GetRegionIndex(nuc, flow, OffsetRegion(x, y));
  }


  private:

  bool regions_set_;
  bool flows_set_;
};


// -------------------------------------------------------------------

class MultiAB{
  public:
    const vector<vector<vector<float> > > * aPtr;
    const vector<vector<vector<float> > > * bPtr;

    MultiAB(){aPtr=0; bPtr = 0;};
    void Null(){aPtr=0; bPtr = 0;};
    bool Valid(){return (aPtr != 0 && bPtr != 0);};
};

// -------------------------------------------------------------------

class RecalibrationModel {
public:
  
  RecalibrationModel();
  ~RecalibrationModel();

  const vector<vector<vector<float> > > * getAs(int x, int y) const;
  const vector<vector<vector<float> > > * getBs(int x, int y) const;

  // Read command line arguments and then call InitializeModel
  void Initialize(OptArgs& opts, vector<string> &bam_comments, const string & run_id, const ion::ChipSubset & chip_subset);

  // Model text file and hp threshold supplied as input variables
  bool InitializeModel(string model_file_name, int model_threshold);

  void getAB(MultiAB &multi_ab, int x, int y) const;

  const vector<unsigned int> CheckArraySize() const;

  bool is_enabled() const { return is_enabled_; };

  void suppressEnabled() { is_enabled_ = false;};

  void InitializeFromJSON(Json::Value &recal_param, string &my_block_key, bool spam_enabled, int over_flow_protect);

  void SetupStratification(int flowStart, int flowEnd, int flowSpan, 
                                             int xMin, int xMax, int xSpan, 
                                             int yMin, int yMax, int ySpan, int max_hp_calibrated);

  void FillIndexes(int offsetRegion, int nucInd, int refHP, int flowStart, int flowEnd, float paramA, float paramB);

  void SaveModelFileToBamComments(string model_file_name, vector<string> &comments, const string &run_id, int block_col_offset, int block_row_offset);


//protected:
  bool is_enabled_;
  int max_hp_calibrated_;
  int recalModelHPThres;
  vector<vector< vector< vector<float> > > > stratifiedAs;
  vector<vector< vector< vector<float> > > > stratifiedBs;
  RegionStratification stratification;

private:
  bool verbose_;

};

#endif // RECALIBRATIONMODEL_H
