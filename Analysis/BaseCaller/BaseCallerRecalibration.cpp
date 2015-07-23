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


BaseCallerRecalibration::BaseCallerRecalibration()
{
  is_enabled_ = false;
  max_hp_calibrated_ = 0;
  recal_model_hp_thres_ = 0;
  max_warnings_ = 5;
  num_warnings_ = 0;
}


BaseCallerRecalibration::~BaseCallerRecalibration()
{
}


void BaseCallerRecalibration::Initialize(OptArgs& opts, const ion::FlowOrder& flow_order)
{
  flow_order_                  = flow_order;
  recal_model_hp_thres_        = opts.GetFirstInt    ('-', "recal-model-hp-thres", 4);
  bool diagonal_state_prog     = opts.GetFirstBoolean('-', "diagonal-state-prog", false);
  string calibration_file_name = opts.GetFirstString ('s', "calibration-file", "");

  if (diagonal_state_prog)
    calibration_file_name.clear();

  InitializeModelFromFile(calibration_file_name);
}

bool BaseCallerRecalibration::InitializeModelFromFile(string calibration_file_name)
{
  is_enabled_ = false;
  if(calibration_file_name.empty()) {
    printf("Recalibration: disabled\n\n");
    return false;
  }

  ifstream calibration_file;
  calibration_file.open(calibration_file_name.c_str());
  if (calibration_file.fail()) {
    printf("Recalibration: disabled (cannot open %s)\n\n", calibration_file_name.c_str());
    calibration_file.close();
    return false;
  }

  string comment_line;
  getline(calibration_file, comment_line);

  int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, called_hp,  max_hp_calibrated;
  calibration_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;

  // Setup region management from global header
  stratification.SetupChipRegions(xMin, xMax, xSpan, yMin, yMax, ySpan);
  stratification.SetupFlowRegions(flowStart,flowEnd, flowSpan);

  max_hp_calibrated_ = max_hp_calibrated;

  vector< vector<int> > hp_pertubation;
  //hard-coded for now: 99
  hp_pertubation.assign(max_hp_calibrated + 1, vector<int>(99));
  calibrated_table_hp_.assign(4*stratification.flowCuts*stratification.xCuts*stratification.yCuts,hp_pertubation);

  vector< vector<float> > hp_delta;
  hp_delta.assign(max_hp_calibrated + 1, vector<float>(99));
  calibrated_table_delta_.assign(4*stratification.flowCuts*stratification.xCuts*stratification.yCuts,hp_delta);

  char flowBase;
  while(calibration_file.good()){
	  // Read in header line for a region
      calibration_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> called_hp;

      // We use the point with minimum coordinates to identify a region
      int region_index = stratification.GetRegionIndex(toInt(flowBase), flowStart, xMin, yMin);
      if (region_index < 0){
        cerr << "Error in Recalibration: invalid region index computed from " << calibration_file_name
             << " Base: " << flowBase << " start flow: " << flowStart << " xMin: " << xMin << " xMax: " << xMax << endl;
        cout << "Recalibration: disabled; File index ERROR." << endl << endl;
        return false;
      }

      int pertubation =0;
      int calibrated_hp = 0;
      float delta = 0.0;
      for(int i=0; i<=98; i++){
          calibration_file >> pertubation >> calibrated_hp >> delta;
          calibrated_table_hp_.at(region_index).at(called_hp).at(pertubation) = calibrated_hp;
          calibrated_table_delta_.at(region_index).at(called_hp).at(pertubation) = delta;
      }

  }

  calibration_file.close();

  cout << "Recalibration: enabled (using calibration file " << calibration_file_name << ")" << endl;
  cout << " - using table calibration for HPs up to (not including)  " << recal_model_hp_thres_ << " in a "
       << stratification.xCuts << 'x' << stratification.yCuts << 'x' << stratification.flowCuts << " grid." << endl << endl;
  is_enabled_ = true;
  return is_enabled_;
}


void BaseCallerRecalibration::CalibrateRead(int x, int y, vector<char>& sequence, vector<float>& normalized_measurements,
    const vector<float>& prediction, const vector<float>& state_inphase)
{
  if (!is_enabled_)
    return;

  vector<char> new_sequence;
  new_sequence.reserve(2*sequence.size());

  int offset_region = stratification.OffsetRegion(x,y);
  if (offset_region < 0){
    // Make sure not to spam stderr with too many messages
	if (num_warnings_ < max_warnings_){
      cerr << "Recalibration ERROR: Could not resolve calibration region for well x=" << x << " y=" << y << endl;
      ++num_warnings_;
	}
    return;
  }

  for (int flow = 0, base = 0; flow < flow_order_.num_flows(); ++flow) {

    // Guard against too small of a state_inphase population
    if (state_inphase.at(flow) < 0.01f)
      continue;

    int region_index = stratification.GetRegionIndex(flow_order_.int_at(flow), flow, offset_region);
    if (region_index < 0){
      // Make sure not to spam stderr with too many messages
      if (num_warnings_ < max_warnings_){
        cerr << "Recalibration ERROR: Could not resolve calibration region for well x="
             << x << " y=" << y << " flow=" << flow << " nuc=" << flow_order_[flow] << endl;
        ++num_warnings_;
      }
      continue;
    }

    int old_hp_length = 0;
    while (base < (int)sequence.size() and sequence[base] == flow_order_[flow]) {
      base++;
      old_hp_length++;
    }
    int new_hp_length = old_hp_length;

    float scaled_residual = (normalized_measurements.at(flow) - prediction.at(flow)) / state_inphase.at(flow);
    float adjustment = min(0.49f, max(-0.49f, scaled_residual));

    // we are either using model or this method - no double recalibration of HPs
    if(old_hp_length <= max_hp_calibrated_ && old_hp_length < recal_model_hp_thres_)
      new_hp_length = std::abs(calibrated_table_hp_.at(region_index).at(old_hp_length).at((int)(adjustment*100)+49));

    if (old_hp_length == 0 or new_hp_length == 0) {
      for (int idx = 0; idx < old_hp_length; ++idx)
        new_sequence.push_back(flow_order_[flow]);
      continue;
    }

    for (int idx = 0; idx < new_hp_length; ++idx)
      new_sequence.push_back(flow_order_[flow]);

    if(old_hp_length <= max_hp_calibrated_ && old_hp_length < recal_model_hp_thres_)
      normalized_measurements[flow] += calibrated_table_delta_.at(region_index).at(old_hp_length).at((int)(adjustment*100)+49)*state_inphase.at(flow)/100;

  }

  sequence.swap(new_sequence);

}








