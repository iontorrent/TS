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
#include "LinearCalibrationModel.h"

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
  cout << "     --histogram-calibration        BOOL      Turn training for this module on/off        [true]"  << endl;
  cout << "     --histogram-modify-measured    BOOL      Modify output measured values.              [true]"  << endl;
  cout << "     --histogram-max-hp             INT       Upper bound for calibration (exclusive)     [4]"     << endl;
  cout << "     --histogram-num-bins           INT       Number of histogram bins per homopolymer    [100]"   << endl;
  cout << "     --histogram-min-samples        INT       Min required number of samples per bin      [40]"    << endl;
  cout << "     --histogram-zero-mers          BOOL      Calibrate zero-mer flows (to & from)        [true]"  << endl;
  cout << "     --histogram-min-inphase        FLOAT     Minimum fraction of in-phase molecules      [0.1]"   << endl;
  cout << "     --histogram-fractional-smooth  BOOL      interpolate between fully-smoothed data and original bin values [true]" << endl;
  cout << "     --histogram-careful-boundaries BOOL      find best threshold to minimize integrated error                [true]" << endl;
  cout << "     --histogram-old-bins           BOOL      remove data points too far outside normal histogram range       [true]" << endl;
  cout << "     --histogram-training-stats     BOOL      Output training statistics in json          [true]" << endl;
  // If you update these; don't forget to adjust help Basecaller/BaseCallerParameters.cpp XXX
}

// ------------------------------------------------------------------------

// store values where the default is shared between basecaller & histograms
// so that we are automatically consistent
void HistogramCalibration::Defaults(){
  do_training_              = false;
  training_mode_            = -1;
  output_training_stats_    = true;
  modify_measured_          = true;
  num_hps_                  = 4;
  min_state_inphase_        = 0.1f;
  num_bins_                 = 100;
  min_observations_per_bin_ = 40;
  // these next two are especially important to be consistent
  // currently setting to values for higher RRA rather than higher throughput
  process_zero_mers_        = true;
  old_style_bins_           = true;
  blind_training_           = false;
  fractional_smooth_        = true;
  careful_boundaries_       = true;
  num_high_residual_reads_  = 0;
  is_enabled_               = false;
  debug_                    = false;
}

// ------------------------------------------------------------------------

// Constructor for calibration training
HistogramCalibration::HistogramCalibration(OptArgs& opts, const CalibrationContext& calib_context) :
  chip_subset_(calib_context.chip_subset)
{
  Defaults();

  // Context dependent variables
  flow_window_size_ = calib_context.flow_window_size;
  num_flow_windows_ = (calib_context.max_num_flows + flow_window_size_ -1) / flow_window_size_;
  blind_training_   = calib_context.blind_fit;

  // read in command line arguments for this module
  num_hps_                  = opts.GetFirstInt    ('-', "histogram-max-hp",          num_hps_);
  do_training_              = opts.GetFirstBoolean('-', "histogram-calibration",     true);
  num_bins_                 = opts.GetFirstInt    ('-', "histogram-num-bins",        num_bins_);
  process_zero_mers_        = opts.GetFirstBoolean('-', "histogram-zero-mers",       process_zero_mers_);
  min_observations_per_bin_ = opts.GetFirstInt    ('-', "histogram-min-samples",     min_observations_per_bin_); // this value gets decreased when smoothing
  min_state_inphase_        = opts.GetFirstDouble ('-', "histogram-min-inphase",     min_state_inphase_);
  output_training_stats_    = opts.GetFirstBoolean('-', "histogram-training-stats",  output_training_stats_);
  fractional_smooth_        = opts.GetFirstBoolean('-', "histogram-fractional-smooth", fractional_smooth_);
  careful_boundaries_       = opts.GetFirstBoolean('-', "histogram-careful-boundaries", careful_boundaries_);
  old_style_bins_           = opts.GetFirstBoolean('-', "histogram-old-bins",        old_style_bins_);
  training_method_          = opts.GetFirstString ('-', "histogram-train-method",    "smooth-histogram");

  // Input argument consistency checks
  if (training_method_ == "histogram")
    training_mode_ = 0;
  else if (training_method_ == "smooth-histogram")
    training_mode_ = 1;
  else {
    cerr << "HistogramCalibration ERROR: unknown training method " << training_method_ << endl;
    exit(EXIT_FAILURE);
  }

  if (num_bins_ < 16) {
    cerr << "HistogramCalibration ERROR: need at least 16 bins per HP." << endl;
    exit(EXIT_FAILURE);
  }

  // Size and reset quantities
  hist_element_.resize(chip_subset_.NumRegions());
  for (unsigned int iRegion=0; iRegion<hist_element_.size(); ++iRegion){
    hist_element_.at(iRegion).resize(num_flow_windows_);
    for (unsigned int iWindow=0; iWindow<hist_element_.at(iRegion).size(); ++iWindow)
      hist_element_.at(iRegion).at(iWindow).SetTrainingModeAndSize(training_mode_, num_hps_, num_bins_);
  }

  if (do_training_ and (calib_context.verbose_level > 0)) {
    cout << "HistogramCalibration Training Options:" << endl;
    cout << "   histogram-max-hp            : " << num_hps_                  << endl;
    cout << "   histogram-num-bins          : " << num_bins_                 << endl;
    cout << "   histogram-min-samples       : " << min_observations_per_bin_ << endl;
    cout << "   histogram-min-inphase       : " << min_state_inphase_        << endl;
    cout << "   histogram-zero-mers         : " << (process_zero_mers_ ? "on" : "off") << endl;
    cout << "   histogram-fractional-smooth : " << (fractional_smooth_ ? "on" : "off") << endl;
    cout << "   histogram-careful-boundaries: " << (careful_boundaries_ ? "on" : "off") << endl;
    cout << "   histogram-old-bins          : " << (old_style_bins_ ? "on" : "off") << endl;
    cout << "   histogram-training-stats    : " << (output_training_stats_ ? "on" : "off") << endl;
    cout << "   histogram-train-method      : " << training_method_          << endl;
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

  process_zero_mers_             = opts.GetFirstBoolean('-', "histogram-zero-mers", process_zero_mers_);
  modify_measured_               = opts.GetFirstBoolean('-', "histogram-modify-measured", modify_measured_);
  min_state_inphase_             = opts.GetFirstDouble ('-', "histogram-min-inphase", min_state_inphase_);
  string legacy_file_name        = opts.GetFirstString ('s', "calibration-file", "");
  string calibration_file_name   = opts.GetFirstString ('-', "calibration-json", "");
  //bool do_null_calibration       = opts.GetFirstBoolean('-', "null-calibration", false);

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
    else {
      cout << "HistogramCalibration: enabled from json file " << calibration_file_name << endl;
      cout << "   histogram-max-hp            : " << num_hps_                  << endl;
      cout << "   histogram-zero-mers         : " << process_zero_mers_   << endl;
      cout << "   histogram-min-inphase       : " << min_state_inphase_   << endl;
      cout << "   histogram-modify-measured   : " << modify_measured_     << endl;
    }
  }

  // load legacy model if provided
  if ((not is_enabled_) and (not legacy_file_name.empty())) {
    InitializeModelFromLegacyFile(legacy_file_name);
  }

  /* / Initialize null calibration if desired
  if ((not is_enabled_) and do_null_calibration) {
    num_hps_ = MAX_HPXLEN;
    flow_window_size_ = flow_order.num_flows();
    num_flow_windows_ = 1;
    chip_subset_.InitializeCalibrationRegions(0, 0, 100000, 100000,  1, 1);  // hard coded as one giant block
    hist_element_.resize(1);
    hist_element_.at(0).resize(1);
    hist_element_.at(0).at(0).NullCalibration(num_hps_);
    is_enabled_ = true;
    cout << "HistogramCalibration: Null calibration." << endl;
  } //*/

  if (not is_enabled_){
    cout << "HistogramCalibration: Disabled." << endl;
  }
}

// ------------------------------------------------------------------------
// Synch up with provided model

bool HistogramCalibration::InitializeModelFromJson(Json::Value &json)
{
  is_enabled_ = false;

  // Check if we have a json object corresponding to a histogram calibration model.
  if ((not json.isMember("MagicCode")) or (json["MagicCode"].asString() != "5131cef78deb965eca8bba4bc517319b")){
    cerr << "HistogramCalibration WARNING: Cannot find appropriate magic code." << endl;
    return false;
  }

  // Now assume that the json object is correctly formatted
  num_hps_  = json["num_hps"].asInt();
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
  num_hps_          = min(num_hps_, max_hp_calibrated); // Rely on default

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

bool  HistogramCalibration::GetHistogramBin_1D(float measurement, float prediction, float rescale_interval,
                                               int& bin, double& residual_predictor) const
{
  if (rescale_interval < min_state_inphase_)
    return false;

  // Residual scaled by inphase state population and (ideally) transformed to [0,1] interval
  // We intentionally do not restrict the returned bin to the range of valid bins.
  residual_predictor = (measurement - prediction) / rescale_interval;
  bin = (int)((0.5+residual_predictor)*(double)num_bins_);

  return true;
}

// ------------------------------------------------------------------------
// Add the information from an aligned read to the histogram data

void HistogramCalibration::HistogramTrainingElement::UpdateIndividualHistogramsFromPredictor(
    int my_bin, int bin_idx, int hp_adjustment, int bin_center, int num_bins_){

  // We log information about every residual +- 1 (greedy) base and threat the rest as outliers
  // This results in 4 different call & residual regions, spanning the centers of 3 HP bin collections
  // ...|---x-1-|-2-x-3-|-4-x---|...

  if ((bin_center-num_bins_ <= my_bin) and (my_bin < 0)){
    if (hp_adjustment == -1)
      ++seen_correct.at(bin_idx);
    else if (hp_adjustment == 0)
      ++seen_offbyone.at(bin_idx);
    else
      ++seen_other.at(bin_idx);

  } else if ((0 <= my_bin) and (my_bin < bin_center)) {
    if (hp_adjustment == 0)
      ++seen_correct.at(bin_idx);
    else if (hp_adjustment == -1)
      ++seen_offbyone.at(bin_idx);
    else
      ++seen_other.at(bin_idx);

  } else if ((bin_center <= my_bin) and (my_bin < num_bins_)) {
    if (hp_adjustment == 0)
      ++seen_correct.at(bin_idx);
    else if (hp_adjustment == 1)
      ++seen_offbyone.at(bin_idx);
    else
      ++seen_other.at(bin_idx);

  } else if ((num_bins_ <= my_bin) and (my_bin < num_bins_+bin_center)) {
    if (hp_adjustment == 1)
      ++seen_correct.at(bin_idx);
    else if (hp_adjustment == 0)
      ++seen_offbyone.at(bin_idx);
    else
      ++seen_other.at(bin_idx);
  }

}

// ------------------------------------------------------------------------

void HistogramCalibration::HistogramTrainingElement::NewUpdateIndividualHistogramsFromPredictor(
    int bin_idx, int predicted_hp, int reference_hp, int called_hp){


  if (predicted_hp==reference_hp)
    ++seen_correct.at(bin_idx);
  if (abs(predicted_hp-reference_hp)==1)
    ++seen_offbyone.at(bin_idx);
  if (abs(predicted_hp-reference_hp)>1)
    ++seen_other.at(bin_idx);

}

// ------------------------------------------------------------------------

bool TrapUnusableStates(const ReadAlignmentInfo &read_alignment, int iHP, int my_nuc_idx, int called_hp, int num_hps_, bool debug_){

  // Ignore Flow InDels
  if (IsInDelAlignSymbol(read_alignment.pretty_flow_align[iHP])) {
    if (debug_)
      cout << "Ignoring HP " << iHP << ": Flow alignment symbol is InDel." << endl;
    return(true);
  }
  if (called_hp >= num_hps_) {
    if (debug_)
      cout << "Ignoring HP " << iHP << ": HP size out of bounds, " << called_hp << endl;
    return(true);
  }

  if (my_nuc_idx < 0){
    if (debug_)
      cout << "Ignoring HP " << iHP << ": nuc idx out of bounds, " << my_nuc_idx << endl;
    return(true);
  }
  return(false);
}

// ------------------------------------------------------------------------

int HistogramCalibration::ComputeFinishHP(int called_hp, float local_measurement, float local_prediction) const
{
  float dir_delta = local_measurement-local_prediction;
  int finish_hp = called_hp;
  // we always step in the direction of the measurement for a possible in/del that improves our objective function
  if (dir_delta>0.0f){
    finish_hp = called_hp+1;
  } else {
    finish_hp = called_hp-1;
  }
  if (finish_hp<0) finish_hp = 0;
  if (finish_hp>num_hps_) finish_hp = called_hp;
  return(finish_hp);
}

// ------------------------------------------------------------------------

bool HistogramCalibration::AddTrainingRead(const ReadAlignmentInfo& read_alignment, const CalibrationContext &calib_context)
{
  if (read_alignment.is_filtered or (not do_training_))
    return false;
  bool high_residual_read = false;

  int my_nuc_idx, my_flow_idx, my_bin, bin_idx, my_flow_window, hp_adjustment;
  int bin_center = num_bins_/2;
  int bin_low_limit = bin_center-num_bins_;
  int bin_hi_limit = bin_center+num_bins_;
  int total_bin_max = num_bins_*num_hps_;

  double residual_predictor;
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
    int called_hp = read_alignment.aligned_qHPs.at(iHP);
    int reference_hp = read_alignment.aligned_tHPs.at(iHP);
    // An adjustment of -1 means we over-called and need to correct the called hp down.
    hp_adjustment =  reference_hp - called_hp;
    my_nuc_idx = ion::FlowOrder::NucToInt(read_alignment.aln_flow_order.at(iHP));

    if (TrapUnusableStates(read_alignment,iHP,my_nuc_idx, called_hp, num_hps_, debug_))
      continue;

    // Get flow window and hp difference between called and reference
    my_flow_idx = read_alignment.align_flow_index.at(iHP);
    my_flow_window = my_flow_idx / flow_window_size_;

    float local_measurement = read_alignment.measurements.at(my_flow_idx);
    float local_prediction = read_alignment.predictions_as_called.at(my_flow_idx);
    float local_interval = read_alignment.state_inphase.at(my_flow_idx);

    //part of linear model to return 'local derivative of prediction'
    // that is, d(prediction)/d(insertion/deletion)
    // our model makes this directional
    if (calib_context.linear_model_master->is_enabled()){
      int finish_hp = ComputeFinishHP(called_hp, local_measurement, local_prediction);
      local_interval = calib_context.linear_model_master->ReturnLocalInterval(local_interval, local_prediction, called_hp, finish_hp, my_region, my_flow_window, my_nuc_idx);
      // if this is something crazy, we just filter it out when we check the bin (i.e. negative, zero, too small
    }
    // Get binned residual-predictor values
    // note that we are abusing the linearity of the predictor,
    // so that residual_predictor(start)  = 1- residual_predictor(finish)
    // we could use a more general residual_predictor(start), residual_predictor(finish) in the future if we have a better classifier

    if (not GetHistogramBin_1D(local_measurement, local_prediction, local_interval, my_bin, residual_predictor)) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": interval_scale too small " << local_interval << endl;
      ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).ignored_samples.at(called_hp);
      continue;
    }


    // Compute bin index and look for outliers at the extreme ends
    bin_idx = num_bins_*called_hp + my_bin;
    int predicted_hp = bin_idx/num_bins_;  // possibly off-by-one

    // check for failed range
    if ((bin_idx < 0) or (bin_idx >= total_bin_max) ){
      // outside_range entirely: impossible to record into a bin
      ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).outside_range.at(called_hp);
      continue; // cannot put into vectors(!)
    }

    if (old_style_bins_){

      // outside normal range of variation but might be real
      if ( (my_bin < bin_low_limit) or (my_bin >= bin_hi_limit)) {
        high_residual_read = true;
        if (debug_)
          cout << "Ignoring HP " << iHP << ": my_bin out of bounds: " << my_bin  << " adjustment is " << hp_adjustment << endl;
        ++hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).ignored_samples.at(called_hp);
        continue; // might be legit
      }

      // sort the predictor into the various houses
      hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).UpdateIndividualHistogramsFromPredictor(my_bin,bin_idx, hp_adjustment,bin_center, num_bins_);
    } else {
      hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).NewUpdateIndividualHistogramsFromPredictor(bin_idx, predicted_hp, reference_hp, called_hp);
    }
    // And currently also create distribution estimates from every data point
    hist_element_.at(my_region).at(my_flow_window).histogram_data.at(my_nuc_idx).AddSummaryDataPoint(called_hp, residual_predictor);

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
        hist_element_.at(iRegion).at(iWindow).SmoothHistograms(fractional_smooth_, min_observations_per_bin_);

      }
      string missing_data;
      if (!blind_training_){
      if (careful_boundaries_)
        missing_data = hist_element_.at(iRegion).at(iWindow).GetCarefulBinBoundaries(local_min_obs, min_hp);
      else
        missing_data = hist_element_.at(iRegion).at(iWindow).GetBinBoundaries(local_min_obs, min_hp);
      } else {
        missing_data = hist_element_.at(iRegion).at(iWindow).GetBlindBoundaries(local_min_obs, min_hp);
      }
      if (missing_data.length() > 0){
        cerr << "HistogramCalibration WARNING: Not enough observations for region " << iRegion << " window " << iWindow << " HPs: " << missing_data << endl;
      }

    }
  }

  is_enabled_ = true;
  return (is_enabled_);
}

// ------------------------------------------------------------------------------

void HistogramCalibration::ExportModelToJson(Json::Value &json) const
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
  json["HistogramData"]    =  Json::arrayValue;
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


void HistogramCalibration::PolishRead(const ion::FlowOrder& flow_order, int well_x, int well_y,
                                      BasecallerRead & read, LinearCalibrationModel *linear_cal_model) const
{
  if (not is_enabled_)
    return;

  int     my_flow_window, my_bin;
  double  residual_predictor;
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

    // double-calibration means that we need to modify this derivative to get the correct bin
    float local_measurement = read.normalized_measurements.at(flow);
    float local_prediction = read.prediction.at(flow);
    float local_interval = read.state_inphase.at(flow);
    int my_nuc_idx = flow_order.int_at(flow);

    if (linear_cal_model->is_enabled()){
      // local_interval automatically checks if we did the linear model first
      int finish_hp = ComputeFinishHP(called_hp_length, local_measurement, local_prediction);  // where might we change to?
      local_interval = linear_cal_model->ReturnLocalInterval(local_interval, local_prediction, called_hp_length, finish_hp,
                                                             my_region, my_flow_window, my_nuc_idx);
    }

    // Ignore HPs that are at or above threshold
    if (called_hp_length >= num_hps_)
      do_calibration = false;
    else if (called_hp_length  == 0 and not process_zero_mers_)
      do_calibration = false;
    else if (not GetHistogramBin_1D(local_measurement,
                                    local_prediction,
                                    local_interval,
                                    my_bin, residual_predictor))
      do_calibration = false;

    // Calibrate the homopolymer if desired and possible
    calibrated_hp_length = called_hp_length;
    hp_adjustment = 0;

    // detect double-taps and behave correctly
    if (flow>0){
      if (flow_order.nuc_at(flow-1)==flow_order.nuc_at(flow)){
        do_calibration = false; // double-tap, always zero
        calibrated_hp_length=0; // base caller may have made an error here?
      }
    }

    // detect not upgradeable for 0->1 changes that aren't double-taps
    if (flow>0){
       // check upgrades only
      if (called_hp_length==0){
        if (flow_order.nuc_at(previous_hp_flow)==flow_order.nuc_at(flow))
          do_calibration=false;
        // I assume that actual basecaller can't make an error here
        // need to check past zero flows as well
        if ( jump_index.at(previous_hp_flow).at(flow_order.int_at(flow)) != flow) do_calibration = false;
        // upgradable if next base is not
      }
    }

    if (do_calibration) {

      // Check if we should do an HP length adjustment
      if (downgradable and (my_bin < hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(called_hp_length))) {
        calibrated_hp_length -= 1;
        hp_adjustment = -1;
      }
      else if (my_bin >= num_bins_ + hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(called_hp_length+1)) {
        calibrated_hp_length += 1;
        hp_adjustment = 1;
      }

      // Switch turns modification of measured values on or off
      if (modify_measured_) {

        // Compute where in the new scheme this signal point lies
        int calib_bin = my_bin - hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(calibrated_hp_length)
            - (hp_adjustment * num_bins_);
        int new_bin_size = num_bins_ -hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(calibrated_hp_length)
            +hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(calibrated_hp_length+1);

        // Correct measured signal with outlier check, i.e., signal points violating bin boundaries after adjustment
        if (calib_bin < 0) {
          read.normalized_measurements.at(flow) = read.prediction.at(flow) + (((float)hp_adjustment-0.499) * local_interval);
        } else if (calib_bin >= new_bin_size){
          read.normalized_measurements.at(flow) = read.prediction.at(flow) + (((float)hp_adjustment+0.499) * local_interval);
        } else {
          // This is the implementation of the "calibrate" module with one additive delta value per bin.
          // However this is not a monotonous signal transformation.
          //double delta = ((double)(num_bins_-1) * (double)calib_bin / (double)(new_bin_size-1) - my_bin + (hp_adjustment*num_bins_)) / (double)num_bins_;
          //read.normalized_measurements.at(flow) += read.state_inphase.at(flow)*delta;

          // A piecewise linear continuous scaling of the residual
          read.normalized_measurements.at(flow) += local_interval * ( (float)(new_bin_size - num_bins_)*((float)hp_adjustment-residual_predictor-0.5) -
                 (float)hist_element_.at(my_region).at(my_flow_window).bin_boundaries.at(my_nuc_idx).at(calibrated_hp_length) ) / (float)new_bin_size;
        }
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
    outside_range.at(iHP) += other.outside_range.at(iHP);


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

void  HistogramCalibration::HistogramTrainingElement::AddSummaryDataPoint(int iHP, double residual_predictor)
{
  ++num_samples.at(iHP);

  double delta =  residual_predictor - means.at(iHP);
  means.at(iHP) += delta / (double)num_samples.at(iHP);

  sum_squares.at(iHP) += delta * (residual_predictor - means.at(iHP));
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
    histogram_data.at(nuc).outside_range.assign(num_hps,0);

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
void HistogramCalibration::HistogramElement::SmoothHistograms(bool fractional_smooth, float min_smooth){
  unsigned int num_hps = bin_boundaries.at(0).size()-2;
  unsigned int num_bins = histogram_data.at(0).seen_correct.size()/num_hps;
  int total_bins = histogram_data.at(0).seen_correct.size();
  vector <float> seen_correct;
  vector <float> seen_offbyone;
  vector <bool> mask_truncated_bins;
  seen_correct.resize(total_bins);
  seen_offbyone.resize(total_bins);
  mask_truncated_bins.resize(total_bins); // Not very useful after removing clipping option.

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

  int num_hps     = bin_boundaries.at(0).size()-2;
  int num_bins    = histogram_data.at(0).seen_correct.size() / num_hps;
  int quarter_bin = num_bins/4;
  std::stringstream missing_data;
  bool have_missing_data = false;

  for (unsigned int iNuc=0; iNuc<4; ++iNuc){

    // Starting with the bins corresponding to the highest HP we step to the left and
    // search for a bin where the (aligned) lower HP is more frequent.
    // This bin then becomes the new boundary for the HP histograms.

    // The starting point allows for potential up-calibration of highest investigated hp.
    bin_boundaries.at(iNuc).assign(num_hps+2, 0);
    int get_lower_bound = num_hps;
    int def_boundary = get_lower_bound * num_bins;
    int iBin = def_boundary -1;
    int end_bin = def_boundary - quarter_bin;
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
        iBin = def_boundary + quarter_bin;
        end_bin = def_boundary - quarter_bin;
        last_valid_bin = def_boundary;
        continue;
      }

      // Obmit bin if there are not enough observations (ignoring outliers)
      if (histogram_data.at(iNuc).seen_correct.at(iBin) + histogram_data.at(iNuc).seen_offbyone.at(iBin) < min_observations) {
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
        }
        --get_lower_bound;
        def_boundary = get_lower_bound * num_bins;
        iBin = def_boundary + quarter_bin;
        end_bin = def_boundary - quarter_bin;
        last_valid_bin = def_boundary;
      }
      else {
        last_valid_bin = iBin--;
      }
    }
  }
  return missing_data.str();
}

// ------------------------------------------------------------------

// Determine the post-calibration HP-bin boundaries from Histogram
string  HistogramCalibration::HistogramElement::GetCarefulBinBoundaries(unsigned int min_observations, int min_hp){

  int num_hps  = bin_boundaries.at(0).size()-2;
  int total_bins = histogram_data.at(0).seen_correct.size();
  int num_bins =  total_bins/ num_hps;

  int half_bin    = num_bins/2;
  int quarter_bin = num_bins/4;


  std::stringstream missing_data;
  bool have_missing_data = false;
  vector <unsigned int> cumulative_seen_correct;
  vector <unsigned int> cumulative_seen_offbyone;

  cumulative_seen_correct.resize(total_bins);
  cumulative_seen_offbyone.resize(total_bins);

  for (unsigned int iNuc=0; iNuc<4; ++iNuc){

    bin_boundaries.at(iNuc).assign(num_hps+2, 0);

    for (int ihp=1; ihp<num_hps; ihp++){

      int low_bin = (ihp-1)*num_bins+half_bin;  // lowest point should be max correct
      int hi_bin  = ihp*num_bins + half_bin;    // highest point (exclusive)

      // Step 1: Build cumulative observations over the whole range of observations

      int old_correct = 0;
      int old_offbyone = 0;
      for (int iBin=low_bin; iBin<hi_bin; iBin++) {
        cumulative_seen_correct[iBin] = old_correct + histogram_data.at(iNuc).seen_correct.at(iBin);
        old_correct = cumulative_seen_correct[iBin];
        cumulative_seen_offbyone[iBin] = old_offbyone + histogram_data.at(iNuc).seen_offbyone.at(iBin);
        old_offbyone = cumulative_seen_offbyone[iBin];
      }

      int all_correct = cumulative_seen_correct[hi_bin];
      int def_boundary = ihp*num_bins;
      int best_correct = all_correct;
      int best_bin = def_boundary;
      unsigned int best_pop = max( histogram_data.at(iNuc).seen_correct.at(best_bin),histogram_data.at(iNuc).seen_offbyone.at(best_bin));
      unsigned int cur_pop = best_pop;
      unsigned int bin_pop = 0;

      // Step 2: Vary boundary and maximize correct calls
      // As a trade off between the number of correct calls and systematic errors we restrict the search range.

      for (int iBin = def_boundary-quarter_bin; iBin < def_boundary+quarter_bin; iBin++){
        int delta_correct = cumulative_seen_correct[iBin]-cumulative_seen_correct[def_boundary];
        int delta_incorrect = cumulative_seen_offbyone[iBin]-cumulative_seen_offbyone[def_boundary];
        int local_correct = 0;
        if (iBin<def_boundary){
          local_correct = all_correct + delta_correct - delta_incorrect;
        } else{
          local_correct = all_correct -delta_correct + delta_incorrect;
        }
        // check that we are >going down< the slope so that we do not pathologically wipe out all of the higher nuc
        // if there are misalignments
        cur_pop = max( histogram_data.at(iNuc).seen_correct.at(iBin),histogram_data.at(iNuc).seen_offbyone.at(iBin));
        // check that the bin we are drawing the decision at has enough population to be reasonable
        bin_pop = histogram_data.at(iNuc).seen_correct.at(iBin) + histogram_data.at(iNuc).seen_offbyone.at(iBin);
        if ((local_correct > best_correct) and (cur_pop <= best_pop) and (bin_pop >= min_observations)){
          best_correct = local_correct;
          best_bin = iBin;
          best_pop = cur_pop;
        }
      }

      // best bin is now optimizing the estimated correct /incorrect breakdown
      // check: might not have any usable bins that were
      bin_pop = histogram_data.at(iNuc).seen_correct.at(best_bin) + histogram_data.at(iNuc).seen_offbyone.at(best_bin);
      if (bin_pop < min_observations){
        // don't have enough observations to justify this cutoff, keep at null
        missing_data << ihp << ion::FlowOrder::IntToNuc(iNuc) <<",";
      } else {
        bin_boundaries.at(iNuc).at(ihp) = best_bin - def_boundary ;
      }

    }
  }

  return missing_data.str();
}

// ------------------------------------------------------------------

Json::Value HistogramCalibration::HistogramElement::ExportBinsToJson() const
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

Json::Value HistogramCalibration::HistogramElement::ExportHistogramsToJson() const
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
      my_value["OutsideRange"][nuc_str][iHP] = (Json::UInt64)histogram_data.at(nuc).outside_range.at(iHP);
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

void  HistogramCalibration::HistogramElement::NullCalibration(int num_hps)
{
  ClearTrainingData();
  bin_boundaries.resize(4);
  for (int iNuc=0; iNuc<4; iNuc++)
    bin_boundaries.at(iNuc).assign(num_hps+2, 0);
}

// ------------------------------------------------------------------

//---------------blind training for classifier methodology
//----decompose raw histogram with no 'correct' call information into two distributions

void quick_regression(float &alpha, float &beta, vector<float> &y, vector<float> &x, vector<float> &w){
  // do online weighted regression
  int xlen=x.size();
  float sw,mx,my,vx,vy,cxy; // tracking values
  sw=w[0];
  mx=x[0];
  my=y[0];
  vx=0.0;
  vy=0.0;
  cxy=0.0;
  // online compute values
  for (int i=1; i<xlen; i++){
    float osw = sw;
    sw += w[i];
    float dx = x[i]-mx;
    float dy = y[i]-my;
    float dw = w[i]/sw;
    mx += dx*dw;
    my += dy*dw;
    float crossw = dw*osw;
    vx += dx*dx*crossw;
    vy += dy*dy*crossw;
    cxy += dx*dy*crossw;
  }
  // standard linear regression
  // cxy/sw /(vx/sw) cancels out
  beta = cxy/vx;
  alpha = my-mx*beta;
}

// helper for extracting information
class MiniRegression{
public:

  int regression_interval;
  vector<float> a_ndx;
  vector<float> b_ndx;

  vector<float> a_val;
  vector<float> w_val;

  // coef of current fit
  vector<float> zcoef;

  // "EM" iteration bits
  vector<float> rhat;
  vector<float> astar;
  vector<float> bstar;
  vector<float> astar_wt;
  vector<float> bstar_wt;

  // functions
  void BootUpEM(vector<int> &target_interval, vector<float> &log_target);
  void QuickFit();
  void FindInitialCoef();
  void SetExpectedValues();
  void EMIterate();
  int FindBestBin();
  void FakeCalls(int iBin, int def_boundary, vector<int> &target_interval, float log_t, int &sc, int &obo);
};

void MiniRegression::BootUpEM(vector<int> &target_interval, vector<float> &log_target){
  int alpha_start = target_interval[0];
  int alpha_end = target_interval[3];
  int beta_start = target_interval[1];
  int beta_end = target_interval[2];

  regression_interval = alpha_end-alpha_start+1;
  a_ndx.resize(regression_interval);
  b_ndx.resize(regression_interval);
  a_val.resize(regression_interval);
  w_val.resize(regression_interval);
  rhat.resize(regression_interval);
  astar.resize(regression_interval);
  bstar.resize(regression_interval);
  astar_wt.resize(regression_interval);
  bstar_wt.resize(regression_interval);

  for ( int i=0; i<regression_interval; i++){
    a_ndx[i] = alpha_start + i - beta_start;
    b_ndx[i] = alpha_start + i -beta_end;
    a_val[i] = exp(log_target[alpha_start+i]); // normal-scale (+1)

    w_val[i] = 0.2f;

    if (a_ndx[i]>=0 and b_ndx[i]<=0){
      w_val[i] = 1.0f; // emphasis middle points for fit
    }

  }

}

void MiniRegression::FindInitialCoef(){
  zcoef.resize(4);
  for (int i=0; i<regression_interval; i++){
    // set weights for initial regression, which just uses the points on either side of the interval
    astar[i] = log(a_val[i]); // log-scale
    bstar[i] = log(a_val[i]);
    if (a_ndx[i]<=0){
      astar_wt[i] = 1000.0f;
    } else {
      astar_wt[i] = 0.0000001f;
    }
    if (b_ndx[i]>=0){
      bstar_wt[i] = 1000.0f;
    } else {
      bstar_wt[i] = 0.0000001f;
    }
  }
  QuickFit();
}

void MiniRegression::QuickFit(){
  float astar_alpha, astar_beta;
  float bstar_alpha, bstar_beta;

  quick_regression(astar_alpha,astar_beta,astar, a_ndx, astar_wt );
  quick_regression(bstar_alpha, bstar_beta, bstar, b_ndx, bstar_wt);
  zcoef[0] = astar_alpha;
  zcoef[1] = astar_beta;
  zcoef[2] = bstar_alpha;
  zcoef[3] = bstar_beta;
}

void MiniRegression::SetExpectedValues(){
  // usually instantaneous
  float ahat, bhat, dhat;
  float safety_wt = 0.001;
  for (int i=0; i<regression_interval; i++){
    // predict latent contents of bin
    ahat =zcoef[0]+zcoef[1]*a_ndx[i];
    bhat = zcoef[2]+zcoef[3]*b_ndx[i];
    dhat = bhat-ahat;
    if (dhat>20.0f) dhat = 20.0f;
    if (dhat<-20.0f) dhat = -20.0f; // keep range to sensible values so we don't explode accidentally
    rhat[i] = 1.0f/(1.0f+exp(dhat));  // responsibility
    // divide bin into parts
    astar[i] = log(rhat[i]*a_val[i]); // predicted log number of entities
    bstar[i] = log((1.0f-rhat[i])*a_val[i]);
    // set weight for regression based on responsibility
    astar_wt[i] = rhat[i]*w_val[i]+safety_wt;
    bstar_wt[i] = (1-rhat[i])*w_val[i]+safety_wt;
    //printf("HAT:\t%d\t%f\t%f\t%f\n",i,astar[i],bstar[i],rhat[i]);
  }
}

void MiniRegression::FakeCalls(int iBin, int def_boundary, vector<int> &target_interval, float log_t, int &sc, int &obo){

  int la_ndx, lb_ndx;
  la_ndx = iBin-target_interval[1]; // b_start
  lb_ndx = iBin-target_interval[2]; // b_end


  float ahat, bhat, dhat,lrhat;
  float la_val = exp(log_t);

  ahat =zcoef[0]+zcoef[1]*la_ndx;
  bhat = zcoef[2]+zcoef[3]*lb_ndx;
  dhat = bhat-ahat;
  if (dhat>20.0f) dhat = 20.0f;
  if (dhat<-20.0f) dhat = -20.0f; // keep range to sensible values so we don't explode accidentally
  lrhat = 1.0f/(1.0f+exp(dhat));  // responsibility
  // divide bin into parts
  float lastar = (lrhat*la_val); // predicted number of entities
  float lbstar = (1.0f-lrhat)*la_val;
  // make sure the sum rounds correctly
  if (iBin<def_boundary){
    sc = lastar;
    obo = lbstar;
  } else {
    obo = lastar;
    sc = lbstar;
  }

}


void MiniRegression::EMIterate(){
  FindInitialCoef(); // use the intervals

  float safety_wt = 0.001; // keep from div zero errors
  // converges quick, not really worth testing?
  for (int iter=0; iter<10; iter++){
    SetExpectedValues();
    // best fit to current estimates of latent variable
    QuickFit();
    //printf("ZCOEF:\t%d\t%f\t%f\t%f\t%f\n" ,iter, zcoef[0],zcoef[1] , zcoef[2], zcoef[3]);
  }
}

int MiniRegression::FindBestBin(){
  int best_bin = 0;
  float delta=0.0f;
  for (int i=0; i<regression_interval; i++){
    if (rhat[i]>=0.5f){
      best_bin = i; // current best bin
      delta = rhat[i]-0.5f;
    } else {
      // round to most balanced
      if ((0.5f-rhat[i])<delta)
        best_bin = i;
      break; // possible offbyone?
    }
  }
  return(best_bin);
}

void FindTargetInterval(vector<int> &target_interval, vector<float> &cumulative_seen_all,int low_bin, int hi_bin){
  //@TODO: make match histogram bins
  int min_points = 8;
  int interval_size =25;

  int low_start = low_bin + min_points;
  int hi_end = hi_bin-interval_size-min_points;
  int beta_start = low_start;
  float beta_level = cumulative_seen_all[hi_bin-1]; // bigger than any sub-interval
  for (int iBin=low_start; iBin<hi_end; iBin++){
    float size_bin=cumulative_seen_all[iBin+interval_size]-cumulative_seen_all[iBin];
    if (size_bin<beta_level){
      beta_start = iBin;
      beta_level = size_bin;
    }
  }

  int beta_end = beta_start+interval_size;
  int alpha_start = beta_start-min_points;
  int alpha_end = beta_end+min_points;
  // located interval alpha_start:beta_start:beta_end:alpha:end
  target_interval[0] = alpha_start;
  target_interval[1] = beta_start;
  target_interval[2] = beta_end;
  target_interval[3] = alpha_end;
}


// blinded boundary determination in case we can't trust reference too much
string HistogramCalibration::HistogramElement::GetBlindBoundaries(unsigned int min_observations, int min_hp){

  int num_hps  = bin_boundaries.at(0).size()-2;
  int total_bins = histogram_data.at(0).seen_correct.size();
  int num_bins =  total_bins/ num_hps;

  int half_bin    = num_bins/2;
  int quarter_bin = num_bins/4;


  std::stringstream missing_data;
  bool have_missing_data = false;
  vector <float> cumulative_seen_all;
  vector <float> log_target;

  cumulative_seen_all.resize(total_bins);
  log_target.resize(total_bins);

  MiniRegression em_fit;

  for (unsigned int iNuc=0; iNuc<4; ++iNuc){

    bin_boundaries.at(iNuc).assign(num_hps+2, 0);

    for (int ihp=1; ihp<num_hps; ihp++){

      int low_bin = (ihp-1)*num_bins+half_bin;  // lowest point should be max correct
      int hi_bin  = ihp*num_bins + half_bin;    // highest point (exclusive)
      int def_boundary = ihp*num_bins;

      // generate lost-labeled observations
      float old_all = 0;
      for (int iBin=low_bin; iBin<hi_bin; iBin++) {
        log_target.at(iBin) = log(histogram_data.at(iNuc).seen_correct.at(iBin)+ histogram_data.at(iNuc).seen_offbyone.at(iBin)+1);  // synthetic target
        cumulative_seen_all[iBin] = old_all + log_target.at(iBin);
        old_all = cumulative_seen_all[iBin];
      }

      // locate middle range
      // nested target interval
      vector<int> target_interval(4); // alpha_start, beta_start, beta_end, alpha_end
      FindTargetInterval(target_interval, cumulative_seen_all, low_bin, hi_bin);
      //printf("INTERVAL:\t%d\t%d\t%d\t%d\n" ,target_interval[0],target_interval[1] ,target_interval[2], target_interval[3]);
      // em fit over target interval
      em_fit.BootUpEM(target_interval, log_target);
      em_fit.EMIterate();

      // iteration

      // could synthesize a cross-over interval 'fake calls'
      // just find crossover instead
      int best_bin = em_fit.FindBestBin()+target_interval[0]; // relative to start of regression
      //@TODO: fake bin population because we're doing heavy smoothing
      int bin_pop = histogram_data.at(iNuc).seen_correct.at(best_bin) + histogram_data.at(iNuc).seen_offbyone.at(best_bin);
      if (bin_pop < (int) min_observations){
        // don't have enough observations to justify this cutoff, keep at null
        missing_data << ihp << ion::FlowOrder::IntToNuc(iNuc) <<",";
      } else {
        bin_boundaries.at(iNuc).at(ihp) = best_bin - def_boundary ;
        for (int iBin = low_bin; iBin<hi_bin; iBin++){
          // fake calls
          int sc, obo;
          float ltotal = log_target.at(iBin);
          em_fit.FakeCalls(iBin,def_boundary, target_interval, ltotal, sc, obo);
          histogram_data.at(iNuc).seen_correct.at(iBin) = sc;
          histogram_data.at(iNuc).seen_offbyone.at(iBin) = obo;

        }
      }
    }
  }

  return missing_data.str();
}

