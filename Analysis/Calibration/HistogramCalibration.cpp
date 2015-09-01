/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     HistogramCalibration.cpp
//! @ingroup  Calibration
//! @brief    HistogramCalibration. Algorithms for adjusting signal intensity and base calls using calibration tables
//! @brief    During model training we create scaled residual histograms for aligned homopolymer lengths
//! @brief    and determine the a-posteriori signal boundaries for the homopolymers.
//! @brief    During model application we adjust signal and base call based on the scaled residual of a
//! @brief    flow by comparing it to the boundaries obtained from training.

#include "HistogramCalibration.h"
#include "FlowAlignment.h"
#include "DPTreephaser.h"

#include <string>
#include <fstream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <SystemMagicDefines.h>


// ========================================================================

void HistogramCalibration::PrintHelp_Training()
{
  cout << "HistogramCalibration Training Options:" << endl;
  cout << "     --histogram-calibration     BOOL      Turn training for this module on/off       [true]"  << endl;
  cout << "     --calibration-hp-thres      INT       Upper bound for calibration (exclusive)    [4]"     << endl;
  cout << "     --histogram-num-bins        INT       Number of histogram bins per homopolymer   [100]"   << endl;
  cout << "     --histogram-min-samples     INT       Min required number of samples per bin     [40]"    << endl;
  cout << "     --histogram-zero-mers       BOOL      Calibrate zero-mer flows (to & from)       [true]" << endl;
  cout << "     --histogram-clip-residuals  BOOL      Limit scaled residuals to ]-0.5,0.5[       [false]"  << endl;
  cout << "     --histogram-min-inphase     FLOAT     Minimum fraction of in-phase molecules     [0.1]"   << endl;
  cout << "     --histogram-training-stats  BOOL      Switch to output training statistics       [true]" << endl;
  // If you update these; don't forget to adjust help Basecaller/BaseCallerParameters.cpp XXX
}

// ------------------------------------------------------------------------

// store values where the default is shared between basecaller & histograms
// so that we are automatically consistent
void HistogramCalibration::Defaults(){
  do_training_                   = false;
  training_mode_                 = -1;
  is_enabled_                    = false;
  output_training_stats_         = false;
  num_hps_ = 4;
  min_state_inphase_  = 0.1f;
  num_bins_ = 100;
  min_observations_per_bin_ = 40;

  // these two are especially important to be consistent
  // currently setting to values for higher RRA rather than higher throughput
  process_zero_mers_      = true;
  threshold_residuals_    = false;
}

// Constructor for calibration training
HistogramCalibration::HistogramCalibration(OptArgs& opts, const CalibrationContext& calib_context) :
chip_subset_(calib_context.chip_subset)
{
  Defaults();

  flow_window_size_ = calib_context.flow_window_size;
  num_flow_windows_ = (calib_context.max_num_flows + flow_window_size_ -1) / flow_window_size_;

  // read in command line arguments for this module
  do_training_              = opts.GetFirstBoolean('-', "histogram-calibration",    true);
  num_hps_                  = opts.GetFirstInt    ('-', "calibration-hp-thres",     num_hps_);
  num_bins_                 = opts.GetFirstInt    ('-', "histogram-num-bins",       num_bins_);
  process_zero_mers_        = opts.GetFirstBoolean('-', "histogram-zero-mers",      process_zero_mers_);
  threshold_residuals_      = opts.GetFirstBoolean('-', "histogram-clip-residuals", threshold_residuals_);
  min_observations_per_bin_ = opts.GetFirstInt    ('-', "histogram-min-samples",   min_observations_per_bin_); // this value gets decreased when smoothing
  min_state_inphase_        = opts.GetFirstDouble ('-', "histogram-min-inphase",    min_state_inphase_);
  output_training_stats_    = opts.GetFirstBoolean('-', "histogram-training-stats", true);
  fractional_smooth_        = opts.GetFirstBoolean('-', "histogram-fractional-smooth", true);
  training_method_          = opts.GetFirstString ('-', "histogram-train-method", "smooth-histogram");

  if (training_method_ == "histogram")
    training_mode_ = 0;
  else if (training_method_ == "smooth-histogram")
    training_mode_ = 1;
  //else if (training_method_ == "distribution")
  //  training_mode_ = 2;
  else {
    cerr << "HistogramCalibration ERROR: unknown training method " << training_method_ << endl;
    exit(EXIT_FAILURE);
  }

  // Size and reset quantities
  hist_element_.resize(chip_subset_.NumRegions());
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){
    hist_element_.at(iRegion).resize(num_flow_windows_);
    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow)
    	hist_element_.at(iRegion).at(iWindow).SetTrainingModeAndSize(training_mode_, num_hps_, num_bins_);
  }

  num_high_residual_reads_ = 0;
  is_enabled_              = false;
  debug_                   = false;

  if (do_training_ and (calib_context.verbose_level > 0)) {
    cout << "HistogramCalibration Training Options:" << endl;
    cout << "   calibration-hp-thres    : " << num_hps_                  << endl;
    cout << "   histogram-num-bins      : " << num_bins_                 << endl;
    cout << "   histogram-min-samples   : " << min_observations_per_bin_ << endl;
    cout << "   histogram-zero-mers     : " << process_zero_mers_        << endl;
    cout << "   histogram-clip-residuals: " << threshold_residuals_      << endl;
    cout << "   histogram-min-inphase   : " << min_state_inphase_        << endl;
    cout << "   histogram-training-stats: " << output_training_stats_    << endl;
    cout << "   histogram-train-method  : " << training_method_          << endl;
  }

}

// ------------------------------------------------------------------------
// Constructor for calibration model application

HistogramCalibration::HistogramCalibration(OptArgs& opts, const ion::FlowOrder& flow_order)
{
  Defaults();

  bool   diagonal_state_prog     = opts.GetFirstBoolean('-', "diagonal-state-prog", false);
  if (diagonal_state_prog)
    return;

  num_hps_                       = opts.GetFirstInt    ('-', "calibration-hp-thres", num_hps_);
  process_zero_mers_             = opts.GetFirstBoolean('-', "histogram-zero-mers", process_zero_mers_);
  threshold_residuals_           = opts.GetFirstBoolean('-', "histogram-clip-residuals", threshold_residuals_);
  min_state_inphase_             = opts.GetFirstDouble ('-', "histogram-min-inphase", min_state_inphase_);
  string legacy_file_name        = opts.GetFirstString ('s', "calibration-file", "");
  string calibration_file_name   = opts.GetFirstString ('s', "calibration-json", "");

  // Make flow order jump table
  vector<int> current_idx(4, -1);
  jump_index.resize(flow_order.num_flows());
  for (int flow=flow_order.num_flows()-1; flow>=0; --flow) {
    current_idx.at( flow_order.int_at(flow) ) = flow;
    jump_index.at(flow) = current_idx;
  }

  // Preferentially load json if both options are provided
  if (not calibration_file_name.empty()) {

    ifstream calibration_file(calibration_file_name.c_str(), ifstream::in);
    if (not calibration_file.good()){
      cerr << "HistogramCalibration WARNING: Cannot open file " << calibration_file_name << endl;
    }
    else {
      Json::Value temp_calibraiton_file;
      calibration_file >> temp_calibraiton_file;
      if (temp_calibraiton_file.isMember("HPHistogram")){
        InitializeModelFromJson(temp_calibraiton_file["HPHistogram"]);
      } else {
        cerr << "HistogramCalibration WARNING: Cannot find json member <HPHistogram>" << endl;
      }
    }
    calibration_file.close();

    if (not is_enabled_)
      cerr << "HistogramCalibration WARNING: Unable to load calibration model from json file " << calibration_file_name << endl;
    else
      cout << "HistogramCalibration: enabled from json file " << calibration_file_name << endl;
      cout << "   histogram-zero-mers      : " << process_zero_mers_   << endl;
      cout << "   histogram-clip-residuals : " << threshold_residuals_ << endl;
      cout << "   histogram-min-inphase    : " << min_state_inphase_   << endl;
  }

  // load legacy model if provided
  if ((not is_enabled_) and (not legacy_file_name.empty())) {
    InitializeModelFromLegacyFile(legacy_file_name);
  }

  if (not is_enabled_){
    cout << "HistogramCalibration: Disabled." << endl;
  }
}

// ------------------------------------------------------------------------


bool HistogramCalibration::InitializeModelFromJson(Json::Value &json)
{
  is_enabled_ = false;

  // Check if we have a json object corresponding to a histogram calibration model.
  if ((not json.isMember("MagicCode")) or (json["MagicCode"].asString() != "5131cef78deb965eca8bba4bc517319b")){
    cerr << "HistogramCalibration WARNING: Cannot find appropriate magic code." << endl;
    return false;
  }

  // Now assume that the json object is correctly formatted
  num_hps_  = min(num_hps_, json["num_hps"].asInt());
  num_bins_ = json["num_bins"].asInt();
  flow_window_size_ = json["flow_window_size"].asInt();
  num_flow_windows_ = json["num_flow_windows"].asInt();

  chip_subset_.InitializeCalibrationRegions(json["block_offset_x"].asInt(),
                                            json["block_offset_y"].asInt(),
                                            json["block_size_x"].asInt(),
                                            json["block_size_y"].asInt(),
                                            json["num_regions_x"].asInt(),
                                            json["num_regions_y"].asInt());

  // Check size of json array
  int json_size      = json["RegionData"].size();
  if (json_size != (num_flow_windows_ * chip_subset_.NumRegions())) {
    cerr << "ERROR in HistogramCalibration: Json value RegionData has " << json_size
         << " entries; expected to see " << (num_flow_windows_ * chip_subset_.NumRegions()) << endl;
    return false;
  }

  // Load elements from json
  hist_element_.resize(chip_subset_.NumRegions());
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion)
    hist_element_.at(iRegion).resize(num_flow_windows_);

  for (int iElement=0; iElement<json_size; ++iElement){

    // Check that this elements actually has the right coordinates
    int json_start_x  = json["RegionData"][iElement]["element_start_xyflow"][0].asInt();
    int json_start_y  = json["RegionData"][iElement]["element_start_xyflow"][1].asInt();
    int json_start_fl = json["RegionData"][iElement]["element_start_xyflow"][2].asInt();

    int my_region      = chip_subset_.CoordinatesToRegionIdx(json_start_x, json_start_y);
    int my_flow_window = (json_start_fl / flow_window_size_);

    hist_element_.at(my_region).at(my_flow_window).FromJson(json["RegionData"][iElement], num_hps_);
  }

  num_high_residual_reads_ = 0;
  is_enabled_ = true;
  return true;
}

// ------------------------------------------------------------------------

bool HistogramCalibration::InitializeModelFromLegacyFile(string legacy_file_name)
{
  is_enabled_ = false;

  ifstream calibration_file;
  calibration_file.open(legacy_file_name.c_str());
  if (not calibration_file.good()) {
    cerr << "HistogramCalibration WARNING: Cannot open legacy model in file " << legacy_file_name << endl;
    calibration_file.close();
    return false;
  }

  string comment_line;
  getline(calibration_file, comment_line);

  // Load chip sizes and offset; Legacy data structures
  // xMin-xMax;         - inclusive, closed interval of block coordinates
  // xSpan/             - Size of calibration region in x direction
  // yMin-yMax;         - inclusive, closed interval of block coordinates
  // ySpan              - Size of calibration region in y direction
  // flowStart-flowEnd; - inclusive, closed interval of number of flows
  // flowSpan           - Size of flow window in calibration

  int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, called_hp,  max_hp_calibrated;
  calibration_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;

  num_bins_         = 99; // Hard coded in legacy format and erroneously set as 100-1
  flow_window_size_ = flowSpan;
  num_flow_windows_ = (flowEnd+flowSpan) / flowSpan;
  num_hps_          = min(num_hps_, max_hp_calibrated);

  if (flowEnd+1 != (int)jump_index.size()){
    cerr << "HistogramCalibration WARNING: Legacy model is for "<< flowEnd+1 << " of flows." << endl;
    calibration_file.close();
    return false;
  }

  int num_regions_x = (xMax+xSpan) / xSpan;
  int num_regions_y = (yMax+ySpan) / ySpan;
  chip_subset_.InitializeCalibrationRegions(xMin, yMin, (xMax+1-xMin), (yMax+1-yMin), num_regions_x, num_regions_y);

  // Initialize region sizes
  hist_element_.resize(chip_subset_.NumRegions());
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){
    hist_element_.at(iRegion).resize(num_flow_windows_);
    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow) {
      hist_element_.at(iRegion).at(iWindow).bin_boundaries.resize(4);
      for (int nuc=0; nuc<4; nuc++)
        hist_element_.at(iRegion).at(iWindow).bin_boundaries.at(nuc).assign(num_hps_+2, 0);
    }
  }

  // Read in the individual regions of the text file
  while(calibration_file.good()){

    // Read in the region header
    char flowBase;
    calibration_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> called_hp;

    // Skip HPs that are beyond threshold
    int   pertubation   = 0;
    int   calibrated_hp = 0;
    float delta         = 0.0;
    if (called_hp >=num_hps_){
      for (int bin=0; bin<num_bins_;++bin)
        calibration_file >> pertubation >> calibrated_hp >> delta;
      continue;
    }

    // Read in region information
    int my_region = chip_subset_.CoordinatesToRegionIdx(xMin, yMin);
    if (my_region < 0){
      cerr << "HistogramCalibration WARNING: Error reading legacy model - invalid region index for coordinates "<< xMin << "," << yMin << endl;
      calibration_file.close();
      return false;
    }

    int my_flow_window = flowStart / flow_window_size_;
    int nuc_idx = ion::FlowOrder::NucToInt(flowBase);

    // Read first data line
    int   iBin          = 0;
    calibration_file >> pertubation >> calibrated_hp >> delta;

    // Unfortunately we cannot rely on the legacy file to be sensibly formatted
    // We'll just try to make the best out of what we're given

    // Step 1: get lower bound for this HP if we haven't found one yet
    if (hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(nuc_idx).at(called_hp) == 0) {
      while ((iBin<num_bins_) and (calibrated_hp != called_hp)) {
        ++iBin;
        if (iBin < num_bins_) {
          calibration_file >> pertubation >> calibrated_hp >> delta;
          //cout << "A Read body line " << pertubation << "\t" << calibrated_hp << "\t" << delta << endl;
        }
      }
      if (iBin < num_bins_)
        hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(nuc_idx).at(called_hp) = iBin;
    }

    // Step 2: Look for a lower bound for the next higher HP
    if ((called_hp < num_hps_-1) and (iBin < num_bins_)) {
      while ((iBin<num_bins_) and (calibrated_hp != called_hp+1)) {
        ++iBin;
        if (iBin < num_bins_)
          calibration_file >> pertubation >> calibrated_hp >> delta;
      }
	  if (iBin < num_bins_)
        hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(nuc_idx).at(called_hp+1) = iBin-num_bins_;
	}

    // Advance to next header file
    while (iBin < num_bins_-1){
      ++iBin;
      calibration_file >> pertubation >> calibrated_hp >> delta;
    }

  }

  calibration_file.close();
  cout << "HistogramCalibration: enabled from legacy file " << legacy_file_name << endl;
  is_enabled_ = true;
  return is_enabled_;
}


// ------------------------------------------------------------------------

void HistogramCalibration::CleanSlate()
{
  num_high_residual_reads_ = 0;
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){
    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow)
      hist_element_.at(iRegion).at(iWindow).SetTrainingModeAndSize(training_mode_, num_hps_, num_bins_);
  }
}

// ------------------------------------------------------------------------

void HistogramCalibration::AccumulateHistData(const HistogramCalibration& other)
{
  if (not do_training_)
    return;

  num_high_residual_reads_ += other.num_high_residual_reads_;
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){
    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow)
      hist_element_.at(iRegion).at(iWindow).AccumulateTrainingData(other.hist_element_.at(iRegion).at(iWindow));
	}
}

// ------------------------------------------------------------------------

bool  HistogramCalibration::GetHistogramBin(float measurement, float prediction, float state_inphase, int& bin, double& scaled_residual)
{
  if (state_inphase < min_state_inphase_)
    return false;

  // Residual scaled by inphase state population and (ideally) transformed to [0,1] interval
  // We intentionally do not restrict the returned bin to the range of valid bins.
  scaled_residual = (measurement - prediction) / state_inphase;
  bin = (int)((0.5+scaled_residual)*(double)num_bins_);

  // unsless desired by command line argument
  if (threshold_residuals_) {
     scaled_residual = max(-0.499, min(scaled_residual, 0.499));
     //bin = (int)((0.5+scaled_residual)*(double)num_bins_);
     bin = max(0, min(bin, num_bins_-1));
  }


  return true;
}

// ------------------------------------------------------------------------
// Add the information from an aligned read to the histogram data

bool HistogramCalibration::AddTrainingRead(const ReadAlignmentInfo& read_alignment)
{
  if (read_alignment.is_filtered or (not do_training_))
    return false;
  bool high_residual_read = false;

  int my_nuc_idx, my_flow_idx, my_bin, bin_idx, my_flow_window, hp_adjustment;
  int bin_center = num_bins_/2;
  double scaled_residual;
  int counter_correct , counter_offbyone, counter_other;
  counter_correct = counter_offbyone = counter_other = 0;

  int my_region = chip_subset_.CoordinatesToRegionIdx(read_alignment.well_xy.at(0), read_alignment.well_xy.at(1));
  if (my_region < 0){
    if (debug_)
      cout << "Ignoring read " << read_alignment.alignment->Name << ": coordinates of bounds; region idx " << my_region << endl;
    return false;
  }

  // Step through flow alignment and extract info
  for (unsigned int iHP=0; iHP < read_alignment.pretty_flow_align.size(); ++iHP){

    // Ignore Flow InDels
    if (IsInDelAlignSymbol(read_alignment.pretty_flow_align[iHP])) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": Flow alignment symbol is InDel." << endl;
      continue;
    }

    // Ignore HPs that are too large
    int called_hp = read_alignment.aligned_qHPs.at(iHP);
    if (called_hp >= num_hps_) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": HP size out of bounds, " << called_hp << endl;
      continue;
    }

    my_nuc_idx = ion::FlowOrder::NucToInt(read_alignment.aln_flow_order.at(iHP));
    if (my_nuc_idx < 0){
      if (debug_)
        cout << "Ignoring HP " << iHP << ": nuc idx out of bounds, " << my_nuc_idx << endl;
      continue;
    }

    // Get flow window and hp difference between called and reference
    my_flow_idx = read_alignment.align_flow_index.at(iHP);
    my_flow_window = my_flow_idx / flow_window_size_;

    // Get binned & scaled residual values
    if (not GetHistogramBin(read_alignment.measurements.at(my_flow_idx),
                            read_alignment.predictions_as_called.at(my_flow_idx),
                            read_alignment.state_inphase.at(my_flow_idx),
                            my_bin, scaled_residual)) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": state_inphase too small " << read_alignment.state_inphase.at(my_flow_idx) << endl;
      ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).ignored_samples.at(called_hp);
      continue;
    }

    // An adjustment of -1 means we over-called and need to correct the called hp down.
    hp_adjustment = read_alignment.aligned_tHPs.at(iHP) - called_hp;

    // Compute bin index and look for outliers at the extreme ends
    bin_idx = num_bins_*called_hp + my_bin;
    if ((bin_idx < 0) or (bin_idx >= num_bins_*num_hps_) or (my_bin < bin_center-num_bins_) or (my_bin >= num_bins_+bin_center)) {
      high_residual_read = true;
      if (debug_)
    	cout << "Ignoring HP " << iHP << ": my_bin out of bounds: " << my_bin  << " adjustment is " << hp_adjustment << endl;
      ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).ignored_samples.at(called_hp);
      continue;
    }

    // We log information about every residual +- 1 (greedy) base and threat the rest as outliers
    // This results in 4 different call & residual regions, spanning the centers of 3 HP bin collections
    // ...|---x-1-|-2-x-3-|-4-x---|...

    if ((bin_center-num_bins_ <= my_bin) and (my_bin < 0)){
      if (hp_adjustment == -1)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_correct.at(bin_idx);
      else if (hp_adjustment == 0)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_offbyone.at(bin_idx);
      else
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_other.at(bin_idx);

    } else if ((0 <= my_bin) and (my_bin < bin_center)) {
      if (hp_adjustment == 0)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_correct.at(bin_idx);
      else if (hp_adjustment == -1)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_offbyone.at(bin_idx);
      else
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_other.at(bin_idx);

    } else if ((bin_center <= my_bin) and (my_bin < num_bins_)) {
      if (hp_adjustment == 0)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_correct.at(bin_idx);
      else if (hp_adjustment == 1)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_offbyone.at(bin_idx);
      else
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_other.at(bin_idx);

    } else if ((num_bins_ <= my_bin) and (my_bin < num_bins_+bin_center)) {
      if (hp_adjustment == 1)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_correct.at(bin_idx);
      else if (hp_adjustment == 0)
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_offbyone.at(bin_idx);
      else
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).seen_other.at(bin_idx);
    }

    // And currently also create distribution estimates from every data point
    hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).AddDataPoint(called_hp, scaled_residual);

  }

  if (high_residual_read)
    ++num_high_residual_reads_;
  if (debug_)
    cout << "Added " << counter_correct << " correct, " << counter_offbyone << " off-by-one, and " << counter_other << " others" << endl;

  return true;
}

// ------------------------------------------------------------------------------
// This function turns the aggregates information about the data into a calibration model

bool  HistogramCalibration::CreateCalibrationModel()
{
  if (not do_training_)
    return false;

  int min_hp = process_zero_mers_ ? 0 : 1;
  unsigned int local_min_obs = min_observations_per_bin_;
  if (training_mode_ == 1)
    local_min_obs = (min_observations_per_bin_/10); // effect of smoothing is need fewer observations per individual bin

  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){

    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow) {

      if (training_mode_ == 1) {
        hist_element_.at(iRegion).at(iWindow).SmoothHistograms(threshold_residuals_, fractional_smooth_, min_observations_per_bin_);

      }
      string missing_data = hist_element_.at(iRegion).at(iWindow).GetBinBoundaries(local_min_obs, min_hp);

      if (missing_data.length() > 0){
        cerr << "HistogramCalibration WARNING: Not enough observations for region " << iRegion << " window " << iWindow << " HPs: " << missing_data << endl;
      }

    }
  }

  is_enabled_ = true;
  return (is_enabled_);
}

// ------------------------------------------------------------------------------

void HistogramCalibration::ExportModelToJson(Json::Value &json)
{
  if (not is_enabled_)
    return;

  // Top level information about regional setup
  json["block_offset_x"]   =  (Json::UInt64)chip_subset_.GetOffsetX();
  json["block_offset_y"]   =  (Json::UInt64)chip_subset_.GetOffsetY();
  json["block_size_x"]     =  (Json::UInt64)chip_subset_.GetChipSizeX();
  json["block_size_y"]     =  (Json::UInt64)chip_subset_.GetChipSizeY();
  json["num_regions_x"]    =  (Json::UInt64)chip_subset_.GetNumRegionsX();
  json["num_regions_y"]    =  (Json::UInt64)chip_subset_.GetNumRegionsY();
  json["num_hps"]          =  (Json::UInt64)num_hps_;
  json["num_bins"]         =  (Json::UInt64)num_bins_;
  json["flow_window_size"] =  (Json::UInt64)flow_window_size_;
  json["num_flow_windows"] =  (Json::UInt64)num_flow_windows_;
  json["num_high_residual_reads"] = (Json::UInt64)num_high_residual_reads_;

  // Adds a hash code to identify json objects corresponding to histogram calibration models.
  // MD5 hash of "This is a histogram calibration model."
  json["MagicCode"]        =  "5131cef78deb965eca8bba4bc517319b";

  // Model information per region and flow window
  json["RegionData"]       =  Json::arrayValue;
  json["HistogramData"]    =   Json::arrayValue;
  int element_idx = 0;

  for (int iRegion=0; iRegion<(int)hist_element_.size(); ++iRegion){
    for (int iWindow=0; iWindow<(int)hist_element_.at(iRegion).size(); ++iWindow){

      json["RegionData"][element_idx] = hist_element_.at(iRegion).at(iWindow).ExportBinsToJson();

      // Add start coordinates (and flow) of element to make it uniquely identifyable
      int start_x, start_y;
      chip_subset_.GetRegionStart(iRegion, start_x, start_y);

      json["RegionData"][element_idx]["element_start_xyflow"]    = Json::arrayValue;
      json["RegionData"][element_idx]["element_start_xyflow"][0] = (Json::Int)start_x;
      json["RegionData"][element_idx]["element_start_xyflow"][1] = (Json::Int)start_y;
      json["RegionData"][element_idx]["element_start_xyflow"][2] = (Json::Int)(iWindow * flow_window_size_);

      if (output_training_stats_) {
        json["HistogramData"][element_idx] = hist_element_.at(iRegion).at(iWindow).ExportHistogramsToJson();
        json["HistogramData"][element_idx]["element_start_xyflow"] = json["RegionData"][element_idx]["element_start_xyflow"];
      }

      element_idx++;
    }
  }

}

// ------------------------------------------------------------------------------


void HistogramCalibration::CalibrateRead(const ion::FlowOrder& flow_order, int well_x, int well_y, BasecallerRead & read)
{
  if (not is_enabled_)
    return;

  int     my_flow_window, my_bin;
  double  scaled_residual;
  vector<char> new_sequence;
  new_sequence.reserve(2*read.sequence.size());

  int my_region = chip_subset_.CoordinatesToRegionIdx(well_x, well_y);
  if (my_region < 0) {
    cerr << "HistogramCalibration::CalibrateRead: Unable to resolve region for well coordinates " << well_x << "," << well_y << endl;
    return;
  }

  // *** Step through all the flows and the complete sequence to recalibrate HPs

  int  previous_hp_flow = 0;
  int  next_hp_flow = -1;
  int hp_adjustment, called_hp_length, calibrated_hp_length;

  for (int flow = 0, base = 0; flow < (int)read.normalized_measurements.size(); ++flow) {

    bool do_calibration = true;
    my_flow_window = flow / flow_window_size_;

    called_hp_length = 0;
    while (base < (int)read.sequence.size() and read.sequence[base] == flow_order.nuc_at(flow)) {
      base++;
      called_hp_length++;
    }

    // Determine if an HP is eligible for downward correction (avoid invalid flow sequences)
    bool downgradable = called_hp_length > 0;

    if (called_hp_length == 1) {
      if (process_zero_mers_){
        // determine flow of next incorporating HP
        next_hp_flow = flow+1;
        while (base < (int)read.sequence.size() and next_hp_flow < flow_order.num_flows() and flow_order.nuc_at(next_hp_flow) != read.sequence[base])
          ++next_hp_flow;

        if (next_hp_flow < flow_order.num_flows())
          downgradable = jump_index.at(previous_hp_flow).at(flow_order.int_at(next_hp_flow)) == next_hp_flow;
      }
      else {
        downgradable = false;
      }
    }

    // Ignore HPs that are at or above threshold
    if (called_hp_length >= num_hps_)
      do_calibration = false;
    else if (called_hp_length  == 0 and not process_zero_mers_)
      do_calibration = false;
    else if (not GetHistogramBin(read.normalized_measurements.at(flow),
                            read.prediction.at(flow),
                            read.state_inphase.at(flow),
                            my_bin, scaled_residual))
      do_calibration = false;

    // Calibrate the homopolymer if desired and possible
    calibrated_hp_length = called_hp_length;
    hp_adjustment = 0;

    if (do_calibration) {

      // Check if we should do an HP length adjustment
      if (downgradable and (my_bin < hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(called_hp_length))) {
        calibrated_hp_length -= 1;
        hp_adjustment = -1;
      }
      else if (my_bin >= num_bins_ + hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(called_hp_length+1)) {
        calibrated_hp_length += 1;
        hp_adjustment = 1;
      }

      // Compute where in the new scheme this signal point lies
      int calib_bin = my_bin - hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(calibrated_hp_length)
                             - (hp_adjustment * num_bins_);
      int new_bin_size = num_bins_ -hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(calibrated_hp_length)
                                   +hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(calibrated_hp_length+1);

      // Correct measured signal with outlier check, i.e., signal points violating bin boundaries after adjustment
      if (calib_bin < 0) {
        read.normalized_measurements.at(flow) = read.prediction.at(flow) + (((float)hp_adjustment-0.499) * read.state_inphase.at(flow));
      } else if (calib_bin >= new_bin_size){
        read.normalized_measurements.at(flow) = read.prediction.at(flow) + (((float)hp_adjustment+0.499) * read.state_inphase.at(flow));
      } else {
        // This is the implementation of the "calibrate" module with one additive delta value per bin.
        // However this is not a monotonous signal transformation.
    	//double delta = ((double)(num_bins_-1) * (double)calib_bin / (double)(new_bin_size-1) - my_bin + (hp_adjustment*num_bins_)) / (double)num_bins_;
        //read.normalized_measurements.at(flow) += read.state_inphase.at(flow)*delta;

        // A piecewise linear continuous scaling of the residual
        read.normalized_measurements.at(flow) += read.state_inphase.at(flow) * ( (float)(new_bin_size - num_bins_)*((float)hp_adjustment-scaled_residual-0.5) -
            (float)hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(flow_order.int_at(flow)).at(calibrated_hp_length) ) / (float)new_bin_size;
      }
    }

    if (calibrated_hp_length > 0)
      previous_hp_flow = flow;

    // Update calibrated base sequence
    for (int iBase=0; iBase < calibrated_hp_length; ++iBase){
      new_sequence.push_back(flow_order.nuc_at(flow));
    }

  } // end looping over all flows

  read.sequence.swap(new_sequence);

}

// ==============================================================================

bool  HistogramCalibration::HistogramTrainingElement::AccumulateStatistics(const HistogramTrainingElement & other)
{

  // Add structures for histogram training
  for (unsigned int iBin=0; iBin<seen_correct.size(); ++iBin){
    seen_correct.at(iBin)  += other.seen_correct.at(iBin);
    seen_offbyone.at(iBin) += other.seen_offbyone.at(iBin);
    seen_other.at(iBin)    += other.seen_other.at(iBin);
  }

  // Add structures for distribution estimation
  for (unsigned int iHP=0; iHP<means.size(); ++iHP){
    ignored_samples.at(iHP) += other.ignored_samples.at(iHP);

    if (num_samples.at(iHP) == 0) {
      means.at(iHP)       = other.means.at(iHP);
      sum_squares.at(iHP) = other.sum_squares.at(iHP);
      num_samples.at(iHP) = other.num_samples.at(iHP);
    }

    else if (other.num_samples.at(iHP) > 0) {
      unsigned long total_samples = num_samples.at(iHP) + other.num_samples.at(iHP);

      double delta_mean  = other.means.at(iHP) - means.at(iHP);
      means.at(iHP) += (delta_mean * (double)other.num_samples.at(iHP)) / (double)total_samples;

      sum_squares.at(iHP) += other.sum_squares.at(iHP) + (delta_mean * (double)num_samples.at(iHP) / (double)total_samples
                             * delta_mean * (double)other.num_samples.at(iHP));
      num_samples.at(iHP) = total_samples;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------------

void  HistogramCalibration::HistogramTrainingElement::AddDataPoint(int iHP, double sclaled_residual)
{
  ++num_samples.at(iHP);

  double delta =  sclaled_residual - means.at(iHP);
  means.at(iHP) += delta / (double)num_samples.at(iHP);

  sum_squares.at(iHP) += delta * (sclaled_residual - means.at(iHP));
}


// ==============================================================================
// Private Helper Class HistogramElement

void HistogramCalibration::HistogramElement::SetTrainingModeAndSize(int training_mode, int num_hps, int num_bins) {

  training_mode_ = training_mode;

  // Clear data and resize vectors to number of nucleotides
  histogram_data.clear();
  bin_boundaries.clear();
  histogram_data.resize(4);
  bin_boundaries.resize(4);

  for (int nuc=0; nuc<4; nuc++){

    bin_boundaries.at(nuc).assign(num_hps+2, 0);
    histogram_data.at(nuc).ignored_samples.assign(num_hps, 0);

    histogram_data.at(nuc).seen_correct.assign((num_bins*num_hps), 0);
    histogram_data.at(nuc).seen_offbyone.assign((num_bins*num_hps), 0);
    histogram_data.at(nuc).seen_other.assign((num_bins*num_hps), 0);

    // Structures for distribution estimation training & smoothing
    histogram_data.at(nuc).means.assign(num_hps, 0.0);
    histogram_data.at(nuc).sum_squares.assign(num_hps, 0.0);
    histogram_data.at(nuc).num_samples.assign(num_hps, 0);

  }
}

// ------------------------------------------------------------------

void HistogramCalibration::HistogramElement::ClearTrainingData() {
  histogram_data.clear();
}

// ------------------------------------------------------------------

// Combine data aggregated by different instances of object
bool HistogramCalibration::HistogramElement::AccumulateTrainingData(const HistogramElement & other){

  // Size check
  bool size_matches = (histogram_data.size() == other.histogram_data.size());
  unsigned int iNuc=0;
  while (size_matches and (iNuc < histogram_data.size())) {
    size_matches = size_matches and (histogram_data.at(iNuc).seen_correct.size() == other.histogram_data.at(iNuc).seen_correct.size());
    size_matches = size_matches and (histogram_data.at(iNuc).seen_offbyone.size() == other.histogram_data.at(iNuc).seen_offbyone.size());
    size_matches = size_matches and (histogram_data.at(iNuc).seen_other.size() == other.histogram_data.at(iNuc).seen_other.size());
    ++iNuc;
  }

  if (not size_matches)
   return false;

  for (unsigned int iNuc=0; iNuc<histogram_data.size(); ++iNuc)
    histogram_data[iNuc].AccumulateStatistics(other.histogram_data.at(iNuc));

  return true;
};

// --------------------------------------------------------------------------------

// Smooth histogram using kernel density to make up for decreased sampling in tails
void HistogramCalibration::HistogramElement::SmoothHistograms(bool clip_mask, bool fractional_smooth, float min_smooth){
  unsigned int num_hps = bin_boundaries.at(0).size()-2;
  unsigned int num_bins = histogram_data.at(0).seen_correct.size()/num_hps;
  int total_bins = histogram_data.at(0).seen_correct.size();
  vector <float> seen_correct;
  vector <float> seen_offbyone;
  vector <bool> mask_truncated_bins;
  seen_correct.resize(total_bins);
  seen_offbyone.resize(total_bins);
  mask_truncated_bins.resize(total_bins);

  int nobs, lobs;
  float mean_bin,sqr_bin;
  float sigma_inflation = 1.2f;
  float sigma_add = 1.0f;
  //printf("Smoothing\n");
  for (unsigned int iNuc=0; iNuc<4; ++iNuc){
    // reset counters
    for (unsigned int aBin=0; aBin<seen_correct.size(); aBin++){
      seen_correct[aBin]=0.0f;
      seen_offbyone[aBin]=0.0f;
      mask_truncated_bins[aBin] = false;

    }
    if (clip_mask){
      for (unsigned int ihp=0; ihp<num_hps; ihp++){
        // set masking for zero bins
        int my_bin = ihp*num_bins+0;
        mask_truncated_bins[my_bin]=true;
        mask_truncated_bins[my_bin+num_bins-1]=true;
      }
    }
     // every HP length
    for (unsigned int ihp=0; ihp<num_hps; ihp++){
        // get bandwidth first
        nobs=1;
        mean_bin =0.0f;
        sqr_bin = 0.0f;
        for (unsigned int iBin=0; iBin<num_bins; iBin++){
            // technically wrong...
          int my_bin = iBin+ihp*num_bins;
          if (!mask_truncated_bins[my_bin]){
            lobs = histogram_data[iNuc].seen_correct[my_bin];
            nobs += lobs;
            mean_bin += iBin*lobs;
            sqr_bin += iBin*iBin*lobs;
          }
        }
        // standard deviation
        // note our bandwidth increases by HP length as we have fewer observations and more spread
        float sigma = sqrt(sqr_bin*nobs-mean_bin*mean_bin)/nobs;
        sigma = 2.34*sigma/pow(nobs,0.2); // kernel bandwidth from standard formula
        sigma *=sigma_inflation; // increase because I am most interested in 'tails' where there are few observations
        sigma += sigma_add; // make sure the bandwidth is at least 2 standardly no matter how large nobs is.

        int sigma_int = sigma;
        //printf("Smooth: Nuc:%d HP:%d SIGMA: %f SIGMA_INT:%d\n", iNuc, ihp, sigma, sigma_int);
        // make kernel - sigma varies by hp length

        vector<double> kernel_vec;
        kernel_vec.resize(2*sigma_int+1);
        for (unsigned int isig=0; isig<kernel_vec.size(); isig++){
          float x = (1.0f*isig -1.0f*sigma_int)/sigma;
          kernel_vec[isig]= 0.75*(1-x*x); //epanechnikov kernel
        }
        int low_bound = num_bins*ihp;
        int hi_bound = num_bins*(ihp+1);
        // now smooth in my scratch space
        // who cares about efficiency with 100 bins
        for (unsigned int iBin=0; iBin<(2*num_bins); iBin++){
          int dBin = iBin-num_bins/2; // offset into neighboring HP
          int tBin = dBin + ihp*num_bins; // where am I targeting?
          if ((tBin>-1) and (tBin<total_bins)){ // target bin where we compute the value
            double xsum, wsum;
            xsum=0.0;
            wsum=0.001;  // just in case
            // safety!
            for (unsigned int isig=0; isig<kernel_vec.size(); isig++){
              int dsig = (int)isig-sigma_int;
              int xBin = tBin+dsig;
              if ((xBin>-1) and (xBin<total_bins)){
                if (!mask_truncated_bins[xBin]){
                  if ((xBin<low_bound) or (xBin>=hi_bound)){
                    xsum += kernel_vec[isig]*histogram_data[iNuc].seen_offbyone[xBin];
                    wsum += kernel_vec[isig];
                  } else {
                      xsum += kernel_vec[isig]*histogram_data[iNuc].seen_correct[xBin];
                      wsum += kernel_vec[isig];
                  }
                }
              }
            }
            xsum /= wsum;
            // now I have my estimate, put it in the right place
            if ((tBin<low_bound) or (tBin>=hi_bound)){
              seen_offbyone[tBin] = xsum;
            } else {
              seen_correct[tBin] = xsum;
            }

          }

        }
    } // hp

    // at this point the scratch space should be full of a lovely smoothed value
    // transfer back to our histogram and rewrite
    // boost min-allowed bin for checking recalibration validity
    // smoothly transition between smoothed and unsmoothed histogram
    // so that we compensate for poor training data and leave quirks of full training data alone
    for (unsigned int aBin=0; aBin<seen_correct.size(); aBin++){
      float tmp_correct = seen_correct[aBin];
      float tmp_offbyone = seen_offbyone[aBin];
      if (fractional_smooth){
        tmp_correct = (tmp_correct*min_smooth+histogram_data[iNuc].seen_correct[aBin]*histogram_data[iNuc].seen_correct[aBin])/(histogram_data[iNuc].seen_correct[aBin]+min_smooth);
        tmp_offbyone = (tmp_offbyone*min_smooth+histogram_data[iNuc].seen_offbyone[aBin]*histogram_data[iNuc].seen_offbyone[aBin])/(histogram_data[iNuc].seen_offbyone[aBin]+min_smooth);

      }
      if (!mask_truncated_bins[aBin]){
        histogram_data[iNuc].seen_correct[aBin] = tmp_correct;
        histogram_data[iNuc].seen_offbyone[aBin] = tmp_offbyone;
      } else {
        // truncated bin contains an excess of values due to truncation of residuals
        // in this case we leave the original value assuming it is greater than the smoothed value
        if (tmp_correct>histogram_data[iNuc].seen_correct[aBin])
          histogram_data[iNuc].seen_correct[aBin] = tmp_correct;
        if (tmp_offbyone>histogram_data[iNuc].seen_offbyone[aBin])
          histogram_data[iNuc].seen_offbyone[aBin] = tmp_offbyone;
      }
    }

  }
}

// ------------------------------------------------------------------

// Determine the post-calibration HP-bin boundaries from Histogram
string  HistogramCalibration::HistogramElement::GetBinBoundaries(unsigned int min_observations, int min_hp){

  int num_hps  = bin_boundaries.at(0).size()-2;
  int num_bins = histogram_data.at(0).seen_correct.size() / num_hps;
  int bin_center = num_bins/2;
  //cout << "Getting bin boundaries; num_hps=" << num_hps << " num_bins=" << num_bins << " bin_center=" << bin_center << endl;
  std::stringstream missing_data;
  bool have_missing_data = false;

  for (unsigned int iNuc=0; iNuc<4; ++iNuc){

	// Starting with the bins corresponding to the highest HP we step to the left and
    // search for a bin where the (aligned) lower HP is more frequent.
	// This bin then becomes the new boundary for the HP histograms.
    // This is the poor man's substitute for fitting a distribution.
    // In contrast to original approach this one avoids possible no-mans-land in between bins
    // as well as criss-crossing back and forth between hp lengths

    bin_boundaries.at(iNuc).assign(num_hps+2, 0);
    int get_lower_bound = num_hps;
    int def_boundary = get_lower_bound * num_bins;
    int iBin = def_boundary -1;
    int end_bin = iBin - bin_center + 2;
    int last_valid_bin = def_boundary;

    while (get_lower_bound > min_hp) {

      // Nothing found, keep default boundaries and move on along
      if (iBin == end_bin) {
        // Produce warning if we don't have enough observations to make a decision
        if (get_lower_bound < num_hps) {
          if (have_missing_data)
            missing_data << ",";
          else
            have_missing_data = true;
          missing_data << get_lower_bound << ion::FlowOrder::IntToNuc(iNuc);
        }

        --get_lower_bound;
        def_boundary = get_lower_bound * num_bins;
        iBin = get_lower_bound * num_bins + bin_center -1;
        end_bin = iBin - num_bins + 1;
        last_valid_bin = def_boundary;
        continue;
      }

      // Obmit bin if there are not enough observations (ignoring outliers)

      if (histogram_data.at(iNuc).seen_correct.at(iBin) + histogram_data.at(iNuc).seen_offbyone.at(iBin) < min_observations) {

        //if (get_lower_bound == num_hps)
        //  cout << " - Not enough ("<< min_observations << ") observations for bin " << iBin << ": "
        //     << histogram_data.at(iNuc).seen_correct.at(iBin) << ","
        //     << histogram_data.at(iNuc).seen_offbyone.at(iBin) << ","
        //     << histogram_data.at(iNuc).seen_other.at(iBin) <<  endl;

        --iBin;
        continue;
      }

      // Search for event where lower HP is more frequent in bin.
      bool have_change = false;
      if (iBin < def_boundary)
        have_change = histogram_data[iNuc].seen_correct[iBin] > histogram_data[iNuc].seen_offbyone[iBin];
      else
        have_change = histogram_data[iNuc].seen_offbyone[iBin] > histogram_data[iNuc].seen_correct[iBin];

      if (have_change){
    	// Check if we have an information gap across default bin boundary and stick with default if we do
    	if ((iBin < def_boundary-1) and (last_valid_bin >= def_boundary)) {
    	  bin_boundaries.at(iNuc).at(get_lower_bound) = 0;

    	  // Produce warning if we don't have enough observations to make a decision
    	  if (get_lower_bound < num_hps) {
    	    if (have_missing_data)
    	      missing_data << ",";
    	    else
    	      have_missing_data = true;
    	    missing_data << get_lower_bound << ion::FlowOrder::IntToNuc(iNuc);
    	  }
    	}
    	else {
          // Record shift from default and go to next HP
    	  bin_boundaries.at(iNuc).at(get_lower_bound) = iBin - def_boundary +1;
    	  //if (get_lower_bound == num_hps)
    	  //  cout << "---Found lower bound in bin " << iBin << " offset " << bin_boundaries.at(iNuc).at(get_lower_bound) << endl;
    	}
        --get_lower_bound;
    	iBin = get_lower_bound * num_bins + bin_center -1;
    	def_boundary -= num_bins;
    	end_bin -= num_bins;
      }
      else {
        last_valid_bin = iBin--;
      }
    }
  }
  return missing_data.str();
}

// ------------------------------------------------------------------

Json::Value HistogramCalibration::HistogramElement::ExportBinsToJson()
{
  Json::Value my_value(Json::objectValue);

  for (int nuc=0; nuc<4; nuc++){
    string nuc_str = ion::FlowOrder::IntToNucStr(nuc);

    // Default model output
    my_value[nuc_str] = Json::arrayValue;
    for (int iBin=0; iBin<(int)bin_boundaries.at(nuc).size(); ++iBin) {
      my_value[nuc_str][iBin] = (Json::Int)bin_boundaries.at(nuc).at(iBin);
    }
  }
  return my_value;
}

// ------------------------------------------------------------------

Json::Value HistogramCalibration::HistogramElement::ExportHistogramsToJson()
{
  Json::Value my_value(Json::objectValue);

  my_value["SeenCorrect"] = Json::objectValue;
  my_value["OffByOne"]    = Json::objectValue;
  my_value["Other"]       = Json::objectValue;
  my_value["Means"]       = Json::objectValue;
  my_value["Variances"]   = Json::objectValue;
  my_value["Ignored"]     = Json::objectValue;

  for (int nuc=0; nuc<4; nuc++) {

    string nuc_str = ion::FlowOrder::IntToNucStr(nuc);

    my_value["SeenCorrect"][nuc_str] = Json::arrayValue;
    my_value["OffByOne"][nuc_str]    = Json::arrayValue;
    my_value["Other"][nuc_str]       = Json::arrayValue;
    my_value["Means"][nuc_str]       = Json::arrayValue;
    my_value["Variances"][nuc_str]   = Json::arrayValue;
    my_value["NumSamples"][nuc_str]  = Json::arrayValue;
    my_value["Ignored"][nuc_str]     = Json::arrayValue;

    for (int iHP=0; iHP<(int)histogram_data.at(nuc).means.size(); ++iHP) {
      my_value["Means"][nuc_str][iHP]      = histogram_data.at(nuc).means.at(iHP);
      my_value["NumSamples"][nuc_str][iHP] = (Json::UInt64)histogram_data.at(nuc).num_samples.at(iHP);
      my_value["Ignored"][nuc_str][iHP]    = (Json::UInt64)histogram_data.at(nuc).ignored_samples.at(iHP);
      double my_variance = histogram_data.at(nuc).sum_squares.at(iHP);
      if (histogram_data.at(nuc).num_samples.at(iHP) > 1)
        my_variance = my_variance / (double)(histogram_data.at(nuc).num_samples.at(iHP)-1);
      my_value["Variances"][nuc_str][iHP]  = my_variance;
    }

    for (int iBin=0; iBin<(int)histogram_data.at(nuc).seen_correct.size(); ++iBin) {
      my_value["SeenCorrect"][nuc_str][iBin] = (Json::UInt64)histogram_data.at(nuc).seen_correct.at(iBin);
      my_value["OffByOne"][nuc_str][iBin]    = (Json::UInt64)histogram_data.at(nuc).seen_offbyone.at(iBin);
      my_value["Other"][nuc_str][iBin]       = (Json::UInt64)histogram_data.at(nuc).seen_other.at(iBin);
    }
  }

  return my_value;
}


// ------------------------------------------------------------------

void HistogramCalibration::HistogramElement::FromJson(Json::Value& json, int num_hps)
{
  ClearTrainingData();
  bin_boundaries.resize(4);
  for (int iNuc=0; iNuc<4; iNuc++){
    bin_boundaries.at(iNuc).assign(num_hps+2, 0);
    // keep upmost boundary at zero to ensure smooth transition to other model
    for (int iBin=0; iBin<(int)bin_boundaries.at(iNuc).size()-1; ++iBin){
      string nuc_str = ion::FlowOrder::IntToNucStr(iNuc);
      bin_boundaries.at(iNuc).at(iBin) = json[nuc_str][iBin].asInt();
    }
  }
}

// ------------------------------------------------------------------

