/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerRecalibration.h
//! @ingroup  BaseCaller
//! @brief    BaseCallerRecalibration. Algorithms for adjusting signal and base calls using calibration tables

#ifndef BASECALLERRECALIBRATION_H
#define BASECALLERRECALIBRATION_H

#include <vector>
#include <stdint.h>
#include "OptArgs.h"
#include "BaseCallerUtils.h"
#include "RecalibrationModel.h"

using namespace std;


class BaseCallerRecalibration {
public:
  BaseCallerRecalibration();
  ~BaseCallerRecalibration();

  void Initialize(OptArgs& opts, const ion::FlowOrder& flow_order);

  bool InitializeModelFromFile(string calibration_file_name);

  void CalibrateRead(int x, int y, vector<char>& sequence, vector<float>& normalized_measurements,
      const vector<float>& prediction, const vector<float>& state_inphase);

  bool is_enabled() const { return is_enabled_; }

protected:

  bool is_enabled_;

  ion::FlowOrder flow_order_;

  int max_hp_calibrated_;
  int max_warnings_;
  int num_warnings_;

  vector< vector< vector<int> > > calibrated_table_hp_;
  vector< vector< vector<float> > > calibrated_table_delta_;
  RegionStratification stratification;
  int recal_model_hp_thres_;

};


#endif // BASECALLERRECALIBRATION_H
