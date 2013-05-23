/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerRecalibration.cpp
//! @ingroup  BaseCaller
//! @brief    BaseCallerRecalibration. Algorithms for adjusting signal and base calls using calibration tables

#include "BaseCallerRecalibration.h"

#include <string>
#include <fstream>
#include <stdio.h>
#include <SystemMagicDefines.h>

uint toInt(char nuc) {
  switch(nuc) {
  case 'A': return 0;
  case 'C': return 1;
  case 'G': return 2;
  default: return 3;
  }
}


BaseCallerRecalibration::BaseCallerRecalibration():stratification_(NULL)
{
  is_enabled_ = false;
  max_hp_calibrated_ = 0;
  recal_model_hp_thres_ = MAX_HPXLEN + 1;
}


BaseCallerRecalibration::~BaseCallerRecalibration()
{
  if(stratification_!=NULL) delete stratification_;
}


void BaseCallerRecalibration::Initialize(OptArgs& opts, const ion::FlowOrder& flow_order)
{
  is_enabled_ = false;
  flow_order_ = flow_order;

  string calibration_file_name = opts.GetFirstString ('s', "calibration-file", "");
  if(calibration_file_name.empty()) {
    printf("Recalibration: disabled\n\n");
    return;
  }

  ifstream calibration_file;
  calibration_file.open(calibration_file_name.c_str());
  if (calibration_file.fail()) {
    printf("Recalibration: disabled (cannot open %s)\n\n", calibration_file_name.c_str());
    calibration_file.close();
    return;
  }

  recal_model_hp_thres_ = opts.GetFirstInt('-', "recal-model-hp-thres", 4);
  printf("Recalibration HP threshold: %d\n", recal_model_hp_thres_);

  string comment_line;
  getline(calibration_file, comment_line);

  int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, called_hp,  max_hp_calibrated;
  calibration_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;
  stratification_ = new Stratification(flowStart,flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan);
  max_hp_calibrated_ = max_hp_calibrated;

  vector< vector<int> > hp_pertubation;
  //hard-coded for now: 99
  hp_pertubation.assign(max_hp_calibrated + 1, vector<int>(99));
  calibrated_table_hp_.assign(4*stratification_->flowCuts*stratification_->xCuts*stratification_->yCuts,hp_pertubation);

  vector< vector<float> > hp_delta;
  hp_delta.assign(max_hp_calibrated + 1, vector<float>(99));
  calibrated_table_delta_.assign(4*stratification_->flowCuts*stratification_->xCuts*stratification_->yCuts,hp_delta);

  char flowBase;
  int flowBaseInt = 0;
  while(calibration_file.good()){
      calibration_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> called_hp;
      flowBaseInt = toInt(flowBase);

      int offsetRegion = (yMin - stratification_->yMin)/stratification_->ySpan + (xMin - stratification_->xMin)/stratification_->xSpan * stratification_->yCuts;
      if(offsetRegion >= 0 && offsetRegion < stratification_->xCuts * stratification_->yCuts){
        offsetRegion = offsetRegion * stratification_->flowCuts * 4;
      } else {
        offsetRegion = 0;
      }

      int offsetFlow = (flowStart-stratification_->flowStart)/stratification_->flowSpan;
      if(offsetFlow < 0 || offsetFlow >= stratification_->flowCuts){
          offsetFlow = 0;
      } else {
          offsetFlow = offsetFlow * 4;
      }
      flowBaseInt += offsetRegion + offsetFlow;

      int pertubation =0;
      int calibrated_hp = 0;
      float delta = 0.0;
      for(int i=0; i<=98; i++){
          calibration_file >> pertubation >> calibrated_hp >> delta;
          calibrated_table_hp_[flowBaseInt][called_hp][pertubation] = calibrated_hp;
          calibrated_table_delta_[flowBaseInt][called_hp][pertubation] = delta;
      }

  }

  calibration_file.close();

  printf("Recalibration: enabled (using calibration file %s)\n\n", calibration_file_name.c_str());
  is_enabled_ = true;
}

void BaseCallerRecalibration::CalibrateRead(int x, int y, vector<char>& sequence, vector<float>& normalized_measurements,
    const vector<float>& prediction, const vector<float>& state_inphase) const
{
  if (!is_enabled_)
    return;

  vector<char> new_sequence;
  new_sequence.reserve(2*sequence.size());


  int offsetRegion = (y - stratification_->yMin)/stratification_->ySpan + (x - stratification_->xMin)/stratification_->xSpan * stratification_->yCuts;
  if(offsetRegion >= 0 && offsetRegion < stratification_->xCuts * stratification_->yCuts){
    offsetRegion = offsetRegion * stratification_->flowCuts * 4;
    } else {
      offsetRegion = 0;
  }


  for (int flow = 0, base = 0; flow < flow_order_.num_flows(); ++flow) {
    int old_hp_length = 0;
    while (base < (int)sequence.size() and sequence[base] == flow_order_[flow]) {
      base++;
      old_hp_length++;
    }

    float scaled_residual = (normalized_measurements[flow] - prediction[flow]) / state_inphase[flow];
    float adjustment = min(0.49f, max(-0.49f, scaled_residual));

    int flowBaseInt = flow_order_.int_at(flow);
    int offsetFlow = (flow-stratification_->flowStart)/stratification_->flowSpan;
    if(offsetFlow < 0 || offsetFlow >= stratification_->flowCuts){
        offsetFlow = 0;
    } else {
        offsetFlow = offsetFlow * 4;
    }
    flowBaseInt += offsetRegion + offsetFlow;

    int new_hp_length = old_hp_length;
    //inclusive
    if(old_hp_length <= max_hp_calibrated_ && old_hp_length <= recal_model_hp_thres_) new_hp_length = std::abs(calibrated_table_hp_[flowBaseInt][old_hp_length][(int)(adjustment*100)+49]);

    if (old_hp_length == 0 or new_hp_length == 0) {
      for (int idx = 0; idx < old_hp_length; ++idx)
        new_sequence.push_back(flow_order_[flow]);
      continue;
    }

    for (int idx = 0; idx < new_hp_length; ++idx)
      new_sequence.push_back(flow_order_[flow]);

    if(old_hp_length <= max_hp_calibrated_ && old_hp_length <= recal_model_hp_thres_) normalized_measurements[flow] += calibrated_table_delta_[flowBaseInt][old_hp_length][(int)(adjustment*100)+49]*state_inphase[flow]/100;

  }

  sequence.swap(new_sequence);

}








