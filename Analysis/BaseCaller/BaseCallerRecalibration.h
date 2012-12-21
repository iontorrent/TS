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

using namespace std;

struct Stratification{
  int flowStart;
  int flowEnd;
  int flowSpan;
  int flowCuts;
  int xMin;
  int xMax;
  int xSpan;
  int xCuts;
  int yMin;
  int yMax;
  int ySpan;
  int yCuts;
  Stratification(int fs, int fe, int f, int xi, int xx, int xs, int yi, int yx, int ys):flowStart(fs), flowEnd(fe), flowSpan(f), xMin(xi), xMax(xx), xSpan(xs), yMin(yi), yMax(yx), ySpan(ys){
      flowCuts = (flowEnd - flowStart + 1) / flowSpan;
      xCuts = (xMax - xMin + 1) / xSpan;
      yCuts = (yMax - yMin + 1) / ySpan;
  }
};

class BaseCallerRecalibration {
public:
  BaseCallerRecalibration();
  ~BaseCallerRecalibration();

  void Initialize(OptArgs& opts, const ion::FlowOrder& flow_order);

  void CalibrateRead(int x, int y, vector<char>& sequence, vector<float>& normalized_measurements,
      const vector<float>& prediction, const vector<float>& state_inphase) const;

  bool is_enabled() const { return is_enabled_; }

protected:

  bool is_enabled_;

  ion::FlowOrder flow_order_;

  int max_hp_calibrated_;

  vector< vector< vector<int> > > calibrated_table_hp_;
  vector< vector< vector<float> > > calibrated_table_delta_;
  Stratification* stratification_;

};


#endif // BASECALLERRECALIBRATION_H
