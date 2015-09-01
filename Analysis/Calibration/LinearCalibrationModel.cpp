/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     LinearCalibrationModel.cpp
//! @ingroup  Calibration
//! @brief    LinearCalibrationModel. Algorithms for adjusting the predicted signal by homopolymer.
//! @brief    During model training we collect pairs of measured and predicted signals by homopolymer
//! @brief    and determine offset and gain coefficients.
//! @brief    During model application we adjust the predicted signal using the model parameters.

#include <math.h>
#include <SystemMagicDefines.h>

#include "LinearCalibrationModel.h"
#include "FlowAlignment.h"


// ==========================================================================
// A general word of caution:
// The module LinearCalibrationModel uses closed intervals in its json representation.
// Expect for "max_hp_calibrated", which is exclusive.

void LinearCalibrationModel::PrintHelp_Training()
{
  cout << "LinearCalibrationModel Training Options:" << endl;
  cout << "     --model-calibration         BOOL      Turn training for this module on/off       [true]" << endl;
  cout << "     --model-calib-min-hp        INT       Min homopolymer length to be trained       [0]"    << endl;
  cout << "     --model-calib-min-samples   INT       Min required number of samples per HP      [50]"   << endl;
  cout << "     --model-calib-min-inphase   FLOAT     Minimum fraction of in-phase molecules     [0.1]"  << endl;
  cout << "     --model-calib-max-residual  FLOAT     Maximum accepted value for scaled residual [0.1]"  << endl;
  cout << "     --model-training-stats      BOOL      Switch to output training statistics       [false]" << endl;
  cout << "     --model-training-style      BOOL      Train using common-offset, old-style       [common-offset]" << endl;
  // These option only have an effect on the model training portion of the code
}


// ==========================================================================
// Structure to accumulate statistics for an atomic calibration unit

LinearFitCell::LinearFitCell()
{
  Reset();
};

// --------------------------------------------------------------------------

void LinearFitCell::Reset()
{
  mean_pred_     = 0.0;
  mean_measured_ = 0.0;
  M2pred_        = 0.0;
  M2meas_        = 0.0;
  Cn_            = 0.0;
  nsamples_      = 0;
};

// --------------------------------------------------------------------------
// Update for adding one more data point

void LinearFitCell::AddDataPoint(float prediction, float measured)
{
  ++nsamples_;

  double delta_p =  prediction - mean_pred_;
  mean_pred_    += delta_p / (double)nsamples_;

  double delta_m  = measured - mean_measured_;
  mean_measured_ += delta_m / (double)nsamples_;

  M2pred_ += delta_p * (prediction - mean_pred_);
  M2meas_ += delta_m * (measured - mean_measured_);
  Cn_     += delta_p * (measured - mean_measured_);
};

// --------------------------------------------------------------------------
// Add data from one cell to a master cell

void LinearFitCell::AccumulateTraingData(const LinearFitCell& other)
{
  if (other.nsamples_ == 0)
    return; // Nothing to do

  if (nsamples_ == 0){
    nsamples_      = other.nsamples_;
    mean_pred_     = other.mean_pred_;
    mean_measured_ = other.mean_measured_;
    M2pred_        = other.M2pred_;
    M2meas_        = other.M2meas_;
    Cn_            = other.Cn_;
    return;
  }

  // We have data in both the master as well as the input cell and have to combine them
  unsigned long total_samples = other.nsamples_ + nsamples_;

  double delta_mean_p  = other.mean_pred_ - mean_pred_;
  mean_pred_ += (delta_mean_p * (double)other.nsamples_) / (double)total_samples;

  M2pred_    += other.M2pred_ + (delta_mean_p * (double)nsamples_ / (double)total_samples
                * delta_mean_p * (double)other.nsamples_) ;

  double delta_mean_m  = other.mean_measured_ - mean_measured_;
  mean_measured_ += (delta_mean_m * (double)other.nsamples_) / (double)total_samples;

  M2meas_    += other.M2meas_ + (delta_mean_m * (double)nsamples_ / (double)total_samples
                * delta_mean_m * (double)other.nsamples_) ;

  Cn_ += other.Cn_ + (delta_mean_p * (double)nsamples_ / (double)total_samples
		 * delta_mean_m * (double)other.nsamples_);

  nsamples_ = total_samples;

};

// --------------------------------------------------------------------------

void LinearFitCell::GetSlopeOnlyFit(double &gain) const
{
  gain = (Cn_ / (double)(nsamples_-1) + (mean_pred_ * mean_measured_)) /
		 (M2pred_ / (double)(nsamples_-1) + (mean_pred_ * mean_pred_));
};

// --------------------------------------------------------------------------

void LinearFitCell::GetSlopeAndInterceptFit(double &gain, double &offset) const
{
  gain   = Cn_ / M2pred_;
  offset = mean_measured_ - (gain * mean_pred_);
};

void LinearFitCell::GetSafeSlopeAndInterceptFit(double &gain, double &offset, double safety_frac) const
{
  GetSlopeAndInterceptFit(gain, offset);
  double valid_offset = mean_measured_ * safety_frac;
  if (offset>valid_offset){
    offset = valid_offset;
  }
  if (offset<(-valid_offset)){
    offset = -valid_offset;
  }
  gain = GetSlopeOnlyFitWithOffset(offset);
};


// --------------------------------------------------------------------------

double  LinearFitCell::GetSlopeOnlyFitWithOffset(double offset) const
{
  double new_mean_pred = mean_pred_ + offset;
  // We store centered sums of squares and now we un-center them with the new mean
  return ((Cn_ / (double)(nsamples_-1) + (new_mean_pred * mean_measured_)) /
         (M2pred_ / (double)(nsamples_-1) + (new_mean_pred * new_mean_pred)));
}

// --------------------------------------------------------------------------

double  LinearFitCell::GetOffsetOnly() const
{
  return (mean_measured_ - mean_pred_);
}

// --------------------------------------------------------------------------

Json::Value  LinearFitCell::DataToJson() const
{
  Json::Value my_value(Json::objectValue);

  my_value["num_samples"]    = (Json::UInt64)nsamples_;
  my_value["mean_predicted"] = mean_pred_;
  my_value["mean_measured"]  = mean_measured_;

  if (nsamples_ > 1){
    my_value["var_predicted"] = M2pred_ / (double)(nsamples_-1);
    my_value["var_measured"]  = M2meas_ / (double)(nsamples_-1);
    my_value["covariance"]    = Cn_ / (double)(nsamples_-1);
  }
  else{
    my_value["var_predictions"] = 0.0f;
    my_value["var_measured"]    = 0.0f;
    my_value["covariance"]      = 0.0f;
  }

  return my_value;
}


// ==========================================================================
// gain and offset stratified by <flow><nucleotide><called_hp length>

void  LinCalModelRegion::Initialize(int num_flow_windows)
{
  gain_values.resize(num_flow_windows);
  offset_values.resize(num_flow_windows);

  for (int iWindow=0; iWindow < num_flow_windows; ++iWindow){
    gain_values.at(iWindow).resize(4);
    offset_values.at(iWindow).resize(4);

    for (int iNuc=0; iNuc < 4; ++iNuc){
      gain_values.at(iWindow).at(iNuc).assign(MAX_HPXLEN+1, 1.0);
      offset_values.at(iWindow).at(iNuc).assign(MAX_HPXLEN+1, 0.0);
    }
  }
}


// --------------------------------------------------------------------------
// This function exists because in the future we want to do something smarter
// than just copying the values


void  LinCalModelRegion::SetModelGainsAndOffsets(int num_flows, int flow_window_size)
{
  gains.resize(num_flows);
  offsets.resize(num_flows);
  int my_flow_window = 0;

  for (int iFlow=0; iFlow<num_flows; ++iFlow){
    gains.at(iFlow).resize(4);
	offsets.at(iFlow).resize(4);
	my_flow_window = iFlow / flow_window_size;

	for (int iNuc=0; iNuc < 4; ++iNuc) {
      gains.at(iFlow).at(iNuc).resize(MAX_HPXLEN+1);
      offsets.at(iFlow).at(iNuc).resize(MAX_HPXLEN+1);

      for (int iHP=0; iHP <= MAX_HPXLEN; ++iHP) {
        gains.at(iFlow).at(iNuc).at(iHP)   = gain_values.at(my_flow_window).at(iNuc).at(iHP);
        offsets.at(iFlow).at(iNuc).at(iHP) = offset_values.at(my_flow_window).at(iNuc).at(iHP);
      }
    }
  }
}

// --------------------------------------------------------------------------

void  LinCalModelRegion::InitializeTrainingData(int num_flow_windows)
{
  training_data.resize(num_flow_windows);

  for (int iWindow=0; iWindow < num_flow_windows; ++iWindow){
    training_data.at(iWindow).resize(4);

    for (int iNuc=0; iNuc < 4; ++iNuc) {
      training_data.at(iWindow).at(iNuc).resize(MAX_HPXLEN+1);

      for (int iHP=0; iHP <= MAX_HPXLEN; ++iHP) {
        training_data.at(iWindow).at(iNuc).at(iHP).Reset();
      }
    }
  }
}

// --------------------------------------------------------------------------

void  LinCalModelRegion::AccumulateTrainingData(const LinCalModelRegion& other)
{
  for (unsigned int iWindow=0; iWindow < training_data.size(); ++iWindow){
    for (unsigned int iNuc=0; iNuc < training_data.at(iWindow).size(); ++iNuc) {
      for (unsigned int iHP=0; iHP < training_data.at(iWindow).at(iNuc).size(); ++iHP) {

        training_data.at(iWindow).at(iNuc).at(iHP).AccumulateTraingData(other.training_data.at(iWindow).at(iNuc).at(iHP));
      }
    }
  }
}

// --------------------------------------------------------------------------
// Single line function???

void  LinCalModelRegion::AddDataPoint(int flow_window, int nuc, int hp, float prediction, float measured)
{
  training_data.at(flow_window).at(nuc).at(hp).AddDataPoint(prediction, measured);
}

// --------------------------------------------------------------------------
// We take training data and turn it into a calibration model
// This is a different approach than the one in the calibrate module

int   LinCalModelRegion::CreateCalibrationModel(int region_idx, unsigned long min_nsamples)
{
  // Initialize data structures to hold model parameters
  Initialize(training_data.size());
  double offset_zero, gain_value;
  int    max_hp_calibrated = 0;
  vector<int> max_hps_calibrated(4,0);

  // Iterate over atomic elements
  for (unsigned int iWindow=0; iWindow < training_data.size(); ++iWindow){
    max_hps_calibrated.assign(4,0);

    for (unsigned int iNuc=0; iNuc < 4; ++iNuc) {

      //cout << " -- Computing model for window " << iWindow << " nuc " << ion::FlowOrder::IntToNuc(iNuc) << endl;
      // Compute a common offset from the zero-mer values
      if (training_data.at(iWindow).at(iNuc).at(0).GetNSamples() >= min_nsamples) {
        offset_zero = training_data.at(iWindow).at(iNuc).at(0).GetOffsetOnly();
        offset_values.at(iWindow).at(iNuc).at(0) = offset_zero;
      }
      else
        offset_zero = 0;
      
      //cout << "    -> Offset: " << offset_zero << "Gains: ";

      for (int iHP=1; iHP <= MAX_HPXLEN; ++iHP) {

        offset_values.at(iWindow).at(iNuc).at(iHP) = offset_zero;

        if (training_data.at(iWindow).at(iNuc).at(iHP).GetNSamples() >= min_nsamples) {
          gain_values.at(iWindow).at(iNuc).at(iHP) = training_data.at(iWindow).at(iNuc).at(iHP).GetSlopeOnlyFitWithOffset(offset_zero);
          max_hp_calibrated = max(max_hp_calibrated, iHP);
          max_hps_calibrated.at(iNuc) = max(max_hps_calibrated.at(iNuc), iHP);


          //cout << gain_values.at(iWindow).at(iNuc).at(iHP) << ", ";
        }
        else {
          // Zero order interpolation if we don't have enough training data
          gain_values.at(iWindow).at(iNuc).at(iHP) = gain_values.at(iWindow).at(iNuc).at(iHP-1);
          //cout << gain_values.at(iWindow).at(iNuc).at(iHP) << "(" << training_data.at(iWindow).at(iNuc).at(iHP).GetNSamples()<< "), ";
        }

      }
    }

    // Print maximum calibrated hp lengths by region, window, and nucleotide to log file.
    if (iWindow == 0)
      cout << "LinearModelCalibration: Max HP calibrated, region " << region_idx;
    cout << "; window " << iWindow << ": " << max_hps_calibrated.at(0) << ion::FlowOrder::IntToNuc(0);
    for (unsigned int iNuc=1; iNuc < 4; ++iNuc) {
      cout << "," << max_hps_calibrated.at(iNuc) << ion::FlowOrder::IntToNuc(iNuc);
    }
    if (iWindow == training_data.size()-1)
      cout << endl;

  }

  return max_hp_calibrated;
}


// free-form can soak up issues with phasing
int   LinCalModelRegion::CreateOldStyleCalibrationModel(int region_idx, unsigned long min_nsamples)
{
  // Initialize data structures to hold model parameters
  Initialize(training_data.size());
  double offset_zero, gain_value;
  int    max_hp_calibrated = 0;
  vector<int> max_hps_calibrated(4,0);

  // Iterate over atomic elements
  for (unsigned int iWindow=0; iWindow < training_data.size(); ++iWindow){
    max_hps_calibrated.assign(4,0);

    for (unsigned int iNuc=0; iNuc < 4; ++iNuc) {

      //cout << " -- Computing model for window " << iWindow << " nuc " << ion::FlowOrder::IntToNuc(iNuc) << endl;
      // Compute a common offset from the zero-mer values
      if (training_data.at(iWindow).at(iNuc).at(0).GetNSamples() >= min_nsamples) {
        offset_zero = training_data.at(iWindow).at(iNuc).at(0).GetOffsetOnly();
        offset_values.at(iWindow).at(iNuc).at(0) = offset_zero;
      }
      else
        offset_zero = 0;

      //cout << "    -> Offset: " << offset_zero << "Gains: ";
      double last_good_gain = 1.0;
      for (int iHP=1; iHP <= MAX_HPXLEN; ++iHP) {

        offset_values.at(iWindow).at(iNuc).at(iHP) = offset_zero;

        if (training_data.at(iWindow).at(iNuc).at(iHP).GetNSamples() >= min_nsamples) {
          double tmp_gain, tmp_offset;
          //old-style free-form any slope/intercept combination
          // safety: not more than 30% of the mean measurement for offset
          training_data.at(iWindow).at(iNuc).at(iHP).GetSafeSlopeAndInterceptFit(tmp_gain, tmp_offset,0.3);

          gain_values.at(iWindow).at(iNuc).at(iHP) = tmp_gain;
          offset_values.at(iWindow).at(iNuc).at(iHP) = tmp_offset;

          // track what we would see with a simple model with offset
          last_good_gain = training_data.at(iWindow).at(iNuc).at(iHP).GetSlopeOnlyFitWithOffset(offset_zero);
          max_hp_calibrated = max(max_hp_calibrated, iHP);
          max_hps_calibrated.at(iNuc) = max(max_hps_calibrated.at(iNuc), iHP);


          //cout << gain_values.at(iWindow).at(iNuc).at(iHP) << ", ";
        }
        else {
          // Zero order interpolation if we don't have enough training data
          // extrapolate with the last good gain - effectively with near-zero intercept to be good
          gain_values.at(iWindow).at(iNuc).at(iHP) = last_good_gain;
          //cout << gain_values.at(iWindow).at(iNuc).at(iHP) << "(" << training_data.at(iWindow).at(iNuc).at(iHP).GetNSamples()<< "), ";
          // old-style requires explicit tables with extrapolation done beforehand

        }

      }
    }

    // Print maximum calibrated hp lengths by region, window, and nucleotide to log file.
    if (iWindow == 0)
      cout << "LinearModelCalibration: Max HP calibrated, region " << region_idx;
    cout << "; window " << iWindow << ": " << max_hps_calibrated.at(0) << ion::FlowOrder::IntToNuc(0);
    for (unsigned int iNuc=1; iNuc < 4; ++iNuc) {
      cout << "," << max_hps_calibrated.at(iNuc) << ion::FlowOrder::IntToNuc(iNuc);
    }
    if (iWindow == training_data.size()-1)
      cout << endl;

  }

  return max_hp_calibrated+1; // include a gain-only for extrapolation to work properly
}



// --------------------------------------------------------------------------

void  LinCalModelRegion::CoefficientZeroOrderHold(int hold_hp)
{
  int start_hp = hold_hp+1;
  for (unsigned int iWindow=0; iWindow < gain_values.size(); ++iWindow){
    for (unsigned int iNuc=0; iNuc < gain_values.at(iWindow).size(); ++iNuc) {
      for (unsigned int iHP=start_hp; iHP < gain_values.at(iWindow).at(iNuc).size(); ++iHP) {

        offset_values.at(iWindow).at(iNuc).at(iHP) = offset_values.at(iWindow).at(iNuc).at(hold_hp);
        gain_values.at(iWindow).at(iNuc).at(iHP) = gain_values.at(iWindow).at(iNuc).at(hold_hp);

      }
    }
  }
}


// ==========================================================================


LinearCalibrationModel::LinearCalibrationModel()
{
  do_training_           = false;
  is_enabled_            = false;
  verbose_               = false;
  debug_                 = false;
  output_training_stats_ = false;
  min_num_samples_       = 50;
  hp_threshold_          = 0;
  flow_window_size_      = 0;
  num_flow_windows_      = 0;
  num_flows_             = 0;
  min_state_inphase_     = 0.1;
  max_scaled_residual_   = 1.5;
  training_mode_              = 1;
  training_method_        = "common-offset";
  max_hp_calibrated_     = MAX_HPXLEN;
}

// --------------------------------------------------------------------------
// Constructor for use in BaseCaller

LinearCalibrationModel::LinearCalibrationModel(OptArgs& opts, vector<string> &bam_comments,
      const string & run_id, const ion::ChipSubset & chip_subset, const ion::FlowOrder * flow_order)
{
  do_training_                   = false;
  is_enabled_                    = false;
  verbose_                       = true;
  output_training_stats_         = false;
  bool   diagonal_state_prog     = opts.GetFirstBoolean('-', "diagonal-state-prog", false);
  if (diagonal_state_prog)
    return;

  hp_threshold_                  = opts.GetFirstInt    ('-', "calibration-hp-thres", 4);
  string legacy_file_name        = opts.GetFirstString ('s', "model-file", "");
  string calibration_file_name   = opts.GetFirstString ('s', "calibration-json", "");
  num_flows_                     = flow_order->num_flows();
  min_num_samples_               = 50; // Only used in model generation
  // Preferentially load json if both options are provided
  if (not calibration_file_name.empty()) {

    ifstream calibration_file(calibration_file_name.c_str(), ifstream::in);
    if (not calibration_file.good()){
      cerr << "LinearCalibrationModel WARNING: Cannot open file " << calibration_file_name << endl;
    }
    else {
      Json::Value temp_calibraiton_file;
      calibration_file >> temp_calibraiton_file;
      if (temp_calibraiton_file.isMember("LinearModel")){
        InitializeModelFromJson(temp_calibraiton_file["LinearModel"]);
      } else {
        cerr << "LinearCalibrationModel WARNING: Cannot find json member <LinearCalibrationModel>" << endl;
      }
    }
    calibration_file.close();

    if (not is_enabled_)
      cerr << "LinearCalibrationModel WARNING: Unable to load calibration model from json file " << calibration_file_name << endl;
  }

  // Load HP model from file if provided and we don't have a json model
  if ((not is_enabled_) and (not legacy_file_name.empty())) {
    InitializeModelFromTxtFile(legacy_file_name, hp_threshold_);
  }

  if (not is_enabled_){
    cout << "LinearCalibrationModel: Disabled." << endl;
  }
  else{
    // TODO: Check chip parameters like offset and and size for consistency

    // If we use a calibration model we are going to write in into the BAM header to avoid model mismatches
    SaveModelFileToBamComments(bam_comments, run_id);
  }
}

// --------------------------------------------------------------------------
// Constructor for calibration module

LinearCalibrationModel::LinearCalibrationModel(OptArgs& opts, const CalibrationContext& calib_context) :
    chip_subset_(calib_context.chip_subset)
{
  is_enabled_           = false;
  flow_window_size_     = calib_context.flow_window_size;
  num_flow_windows_     = (calib_context.max_num_flows + flow_window_size_ -1) / flow_window_size_;
  verbose_              =  (calib_context.verbose_level > 0);
  debug_                = calib_context.debug;
  num_flows_            = calib_context.max_num_flows;

  min_num_samples_      = opts.GetFirstInt    ('-', "model-calib-min-samples", 50);
  do_training_          = opts.GetFirstBoolean('-', "model-calibration", true);

  // By default we train all HP lengths for this model -- "calib-hp-threshold" cannot be redefined
  hp_threshold_         = opts.GetFirstInt    ('-', "model-calib-min-hp", 0);
  min_state_inphase_    = opts.GetFirstDouble ('-', "model-calib-min-inphase", 0.1);
  max_scaled_residual_  = opts.GetFirstDouble ('-', "model-calib-max-residual", 1.5);
  output_training_stats_= opts.GetFirstBoolean('-', "model-training-stats", false);
  max_hp_calibrated_    = 0;

  training_method_          = opts.GetFirstString ('-', "model-training-style", "common-offset");

   if (training_method_ == "common-offset")
     training_mode_ = 1;
   else if (training_method_ == "old-style")
     training_mode_ = 2;
   //else if (training_method_ == "distribution")
   //  training_mode_ = 2;
   else {
     cerr << "HistogramCalibration ERROR: unknown training method " << training_method_ << endl;
     exit(EXIT_FAILURE);
   }

  // Size calibration structures
  region_data.resize(chip_subset_.NumRegions());
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    //region_data.at(iRegion).Initialize(num_flow_windows_); // We don't need those until we create the model
    region_data.at(iRegion).InitializeTrainingData(num_flow_windows_);
  }

  if (do_training_ and (calib_context.verbose_level > 0)){
    cout << "LinearCalibrationModel Training Options:" << endl;
    cout << "   model-calib-min-hp     : " << hp_threshold_    << endl;
    cout << "   model-calib-min-samples: " << min_num_samples_ << endl;
  }
}

// --------------------------------------------------------------------------

const vector<vector<vector<float> > > * LinearCalibrationModel::getGains(int x, int y) const
{
  if (not is_enabled_)
    return NULL;

  int my_region = chip_subset_.CoordinatesToRegionIdx(x, y);
  if (my_region < 0) {
    if (verbose_)
      cerr << "LinearCalibrationModel::getGains ERROR: Cannot find region for for well x=" << x << " y=" << y << endl;
    return NULL;
  }
  else
    return region_data.at(my_region).getGains();
};

// --------------------------------------------------------------------------

const vector<vector<vector<float> > > * LinearCalibrationModel::getOffsets(int x, int y) const
{
  if (not is_enabled_)
    return NULL;

  int my_region = chip_subset_.CoordinatesToRegionIdx(x, y);
  if (my_region < 0) {
    if (verbose_)
      cerr << "LinearCalibrationModel::getGains ERROR: Cannot find region for for well x=" << x << " y=" << y << endl;
    return NULL;
  }
  else
    return region_data.at(my_region).getOffsets();
};

// --------------------------------------------------------------------------

void  LinearCalibrationModel::getAB(MultiAB &multi_ab, int x, int y) const
{
  if (not is_enabled_){
    multi_ab.Null();
    return;
  }

  int my_region = chip_subset_.CoordinatesToRegionIdx(x, y);
  if (my_region < 0){
    if (verbose_)
      cerr << "LinearCalibrationModel::getGains ERROR: Cannot find region for for well x=" << x << " y=" << y << endl;
    multi_ab.Null();
  }
  else {
    multi_ab.aPtr = region_data.at(my_region).getGains();
    multi_ab.bPtr = region_data.at(my_region).getOffsets();
 }
};

// --------------------------------------------------------------------------
// This function retains the original (rather wasteful) json structure that we put into bam headers

void  LinearCalibrationModel::ExportModelToJson(Json::Value &json, string run_id) const
{
  if (not is_enabled_)
    return;

  // Add a (legacy) hash code to identify json objects corresponding to linear calibration models.
  // MD5 hash of "This uniquely identifies json comments for recalibration."
  json["MagicCode"] = "6d5b9d29ede5f176a4711d415d769108";

  // Keep 5 character ID to not upset all the hard coded methods
  if (run_id.empty())
    run_id = "NORID";

  std::stringstream block_id_stream;
  block_id_stream << run_id << ".block_X" << chip_subset_.GetOffsetX() << "_Y" << chip_subset_.GetOffsetY();
  string block_id = block_id_stream.str();

  json["MasterKey"] = block_id;
  json["MasterCol"] = (Json::UInt64)chip_subset_.GetColOffset();
  json["MasterRow"] = (Json::UInt64)chip_subset_.GetRowOffset();

  // Global block information -- CLOSED INTERVALS

  json[block_id]["flowStart"] = 0;
  json[block_id]["flowEnd"]   = num_flows_ -1;
  json[block_id]["flowSpan"]  = flow_window_size_;
  json[block_id]["xMin"]      = chip_subset_.GetOffsetX();
  json[block_id]["xMax"]      = chip_subset_.GetOffsetX() + chip_subset_.GetChipSizeX() -1;
  json[block_id]["xSpan"]     = chip_subset_.GetRegionSizeX();
  json[block_id]["yMin"]      = chip_subset_.GetOffsetY();
  json[block_id]["yMax"]      = chip_subset_.GetOffsetY() + chip_subset_.GetChipSizeY() -1;
  json[block_id]["ySpan"]     = chip_subset_.GetRegionSizeY();
  json[block_id]["max_hp_calibrated"] = max_hp_calibrated_+1; // The json value is exclusive
  json[block_id]["min_hp_calibrated"] = hp_threshold_;

  // Now write the information of the individual regions
  json[block_id]["modelParameters"] = Json::arrayValue;
  int start_x, start_y, start_flow, element_id = 0;

  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion){

    chip_subset_.GetRegionStart(iRegion, start_x, start_y);
    for (int iWindow=0; iWindow < num_flow_windows_; ++iWindow){

      start_flow = iWindow * flow_window_size_;
      for (int iNuc = 0; iNuc < 4; ++iNuc){

        // To be compatible with the legacy BAM format we write the char value into the json as an integer
    	char base = ion::FlowOrder::IntToNuc(iNuc);
    	for (int iHP=hp_threshold_; iHP<= max_hp_calibrated_; ++iHP) {

          json[block_id]["modelParameters"][element_id]["flowBase"]  = (Json::Int)base;
          json[block_id]["modelParameters"][element_id]["flowStart"] = (Json::Int)start_flow;
          json[block_id]["modelParameters"][element_id]["flowEnd"]   = (Json::Int)(start_flow + flow_window_size_ -1);
          json[block_id]["modelParameters"][element_id]["xMin"]      = (Json::Int)start_x;
          json[block_id]["modelParameters"][element_id]["xMax"]      = (Json::Int)min((start_x+chip_subset_.GetRegionSizeX()-1), (chip_subset_.GetOffsetX()+chip_subset_.GetChipSizeX()-1));
          json[block_id]["modelParameters"][element_id]["yMin"]      = (Json::Int)start_y;
          json[block_id]["modelParameters"][element_id]["yMax"]      = (Json::Int)min((start_y+chip_subset_.GetRegionSizeY()-1), (chip_subset_.GetOffsetY()+chip_subset_.GetChipSizeY()-1));
          json[block_id]["modelParameters"][element_id]["refHP"]     = (Json::Int)iHP;

          // Here we assume that the model parameters are constant over one element
          json[block_id]["modelParameters"][element_id]["paramA"]    = region_data.at(iRegion).gain_values.at(iWindow).at(iNuc).at(iHP);
          json[block_id]["modelParameters"][element_id]["paramB"]    = region_data.at(iRegion).offset_values.at(iWindow).at(iNuc).at(iHP);

          // Optionally output training data
          if (output_training_stats_){
            json[block_id]["modelParameters"][element_id]["TrainingData"] = region_data.at(iRegion).training_data.at(iWindow).at(iNuc).at(iHP).DataToJson();
          }

          ++element_id;
    	}
      }
    }
  }
};

// --------------------------------------------------------------------------

void LinearCalibrationModel::SaveModelFileToBamComments(vector<string> &comments, const string &run_id) const
{
  if (not is_enabled_)
    return;

  Json::Value json;
  ExportModelToJson(json, run_id);

  Json::FastWriter writer;
  string comment_str = writer.write(json);
  // trim unwanted newline added by writer
  int last_char = comment_str.size()-1;
  if ((last_char>=0) and (comment_str.at(last_char) == '\n')) {
    comment_str.erase(last_char,1);
  }
  comments.push_back(comment_str);
}

// --------------------------------------------------------------------------
// We only load data for HPs that are above a desired threshold
// The internally stored data structures reflect that so that we put accurate
// information about what we did into the BAM header

bool  LinearCalibrationModel::InitializeModelFromJson(Json::Value &json, const int num_flows)
{
  is_enabled_ = false;
  if (num_flows > 0)
    num_flows_ = num_flows;

  // Check if we have a json object corresponding to a histogram calibration model.
  if ((not json.isMember("MagicCode")) or (json["MagicCode"].asString() != "6d5b9d29ede5f176a4711d415d769108")){
    cerr << "LinearCalibrationModel::InitializeModelFromJson WARNING: Cannot find appropriate magic code." << endl;
    return false;
  }

  // Json structure uses -- CLOSED INTERVALS -- (mostly)

  // We now assume that the json is correctly formatted and skip checks
  string block_id = json["MasterKey"].asString();
  int block_offset_x = json["MasterCol"].asInt();
  int block_offset_y = json["MasterRow"].asInt();

  // Check number of flows
  int num_flows_json = json[block_id]["flowEnd"].asInt() +1;
  if (num_flows_json != num_flows_){
    cerr << "LinearCalibrationModel::InitializeModelFromJson WARNING: Number of flows in json "
         << num_flows_json << " does not match number of flows in run " << num_flows_ << endl;
    return false;
  }
  flow_window_size_  = json[block_id]["flowSpan"].asInt();
  if (flow_window_size_ <= 0){
    cerr << "LinearCalibrationModel::InitializeModelFromJson WARNING: Flow window is zero." << endl;
    return false;
  }
  num_flow_windows_  = (num_flows_ + flow_window_size_ -1) / flow_window_size_;

  // Load region information

  int block_size_x   = json[block_id]["xMax"].asInt() - block_offset_x +1;
  int block_size_y   = json[block_id]["yMax"].asInt() - block_offset_y +1;
  int region_size_x  = json[block_id]["xSpan"].asInt();
  int region_size_y  = json[block_id]["ySpan"].asInt();
  int num_regions_x  = (block_size_x + region_size_x - 1) / region_size_x;
  int num_regions_y  = (block_size_y + region_size_y - 1) / region_size_y;

  chip_subset_.InitializeCalibrationRegions(block_offset_x, block_offset_y, block_size_x, block_size_y,
                                                num_regions_x, num_regions_y);

  // Check number and range of elements

  int json_size      = json[block_id]["modelParameters"].size();
  max_hp_calibrated_ = json[block_id]["max_hp_calibrated"].asInt() -1; // The json value is exclusive
  int min_hp         = 4; // BAM files created before TS 5.0 will not have this json entry
  if (json[block_id].isMember("min_hp_calibrated"))
    min_hp = json[block_id]["min_hp_calibrated"].asInt();
  hp_threshold_ = max(hp_threshold_, min_hp); // This will avoid doubly calibrating old BAM files in TVC

  // Iterate over model parameters in json and fill in class data

  region_data.resize(chip_subset_.NumRegions());
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).Initialize(num_flow_windows_);
  }

  int xMin, yMin, my_region, my_flow_window, nuc_idx, my_hp;

  for (int i_item=0; i_item<json_size; i_item++) {

    // We only load calibration data for HPs above the threshold and keep the parameters of the others at default
    my_hp          = json[block_id]["modelParameters"][i_item]["refHP"].asInt();
    if ((my_hp < hp_threshold_) or (my_hp > MAX_HPXLEN))
      continue;

    xMin = json[block_id]["modelParameters"][i_item]["xMin"].asInt();
    yMin = json[block_id]["modelParameters"][i_item]["yMin"].asInt();
    my_region = chip_subset_.CoordinatesToRegionIdx(xMin, yMin);
    if (my_region < 0){
      cerr << "LinearCalibrationModel::InitializeModelFromJson WARNING: Cannot resolve region for coordinates x=" << xMin << " y=" << yMin << endl;
      return false;
    }

    my_flow_window = json[block_id]["modelParameters"][i_item]["flowStart"].asInt() / flow_window_size_;
    char base = (char)json[block_id]["modelParameters"][i_item]["flowBase"].asInt();
    nuc_idx   = ion::FlowOrder::NucToInt(base);

    region_data.at(my_region).gain_values.at(my_flow_window).at(nuc_idx).at(my_hp) =
        json[block_id]["modelParameters"][i_item]["paramA"].asFloat();
    region_data.at(my_region).offset_values.at(my_flow_window).at(nuc_idx).at(my_hp) =
        json[block_id]["modelParameters"][i_item]["paramB"].asFloat();

  }

  // Now that we have calibration data we build the class output data structures
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).CoefficientZeroOrderHold(max_hp_calibrated_);
    region_data.at(iRegion).SetModelGainsAndOffsets(num_flows_, flow_window_size_);
  }

  if (verbose_)
    cout << "LinearCalibrationModel: enabled for HPs >=" << hp_threshold_ << " (using block id " << block_id << ") in a "
         << chip_subset_.GetNumRegionsX() << 'x' << chip_subset_.GetNumRegionsY() << 'x' << num_flow_windows_ << " grid." <<endl << endl;
  //PrintModelParameters();

  is_enabled_ = true;
  return true;
};

// --------------------------------------------------------------------------

bool  LinearCalibrationModel::InitializeModelFromTxtFile(string model_file_name, int hp_threshold, const int num_flows)
{
  is_enabled_ = false;
  if (num_flows > 0)
    num_flows_ = num_flows;

  ifstream calibration_file;
  calibration_file.open(model_file_name.c_str());
  if (not calibration_file.good()) {
    cerr << "LinearCalibrationModel WARNING: Cannot open legacy model in file " << model_file_name << endl;
    calibration_file.close();
    return false;
  }

  string comment_line;
  getline(calibration_file, comment_line);

  // Read global block info from header line
  int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, max_hp_calibrated;
  calibration_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;

  // Check number of flows
  int num_flows_json = flowEnd+1;
  if (num_flows_json != num_flows_){
    cerr << "LinearCalibrationModel::InitializeModelFromTxtFile WARNING: Number of flows in file "
         << num_flows_json << " does not match number of flows in run " << num_flows_ << endl;
    calibration_file.close();
    return false;
  }
  flow_window_size_  = flowSpan;
  num_flow_windows_  = (flowEnd + flow_window_size_) / flow_window_size_;

  // Load region information

  int block_size_x   = xMax-xMin+1;
  int block_size_y   = yMax-yMin+1;
  int num_regions_x  = (block_size_x + xSpan - 1) / xSpan;
  int num_regions_y  = (block_size_y + ySpan - 1) / ySpan;

  chip_subset_.InitializeCalibrationRegions(xMin, yMin, block_size_x, block_size_y,
                                                num_regions_x, num_regions_y);

  // Initialize region data and set defaults
  region_data.resize(chip_subset_.NumRegions());
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).Initialize(num_flow_windows_);
  }

  // Read lines of text file - one line per parameter pair
  float paramA, paramB;
  int   my_region, my_flow_window, nuc_idx, my_hp;
  char  flowBase;

  while (calibration_file.good()) {

    calibration_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> my_hp >> paramA >> paramB;
    if ((my_hp < hp_threshold) or (my_hp > MAX_HPXLEN))
      continue;

    my_region = chip_subset_.CoordinatesToRegionIdx(xMin, yMin);
    if (my_region < 0){
      cerr << "LinearCalibrationModel::InitializeModelFromTxtFile WARNING: Cannot resolve region for coordinates x=" << xMin << " y=" << yMin << endl;
      calibration_file.close();
      return false;
    }

    my_flow_window = flowStart / flow_window_size_;
    nuc_idx        = ion::FlowOrder::NucToInt(flowBase);
    region_data.at(my_region).gain_values.at(my_flow_window).at(nuc_idx).at(my_hp)   = paramA;
    region_data.at(my_region).offset_values.at(my_flow_window).at(nuc_idx).at(my_hp) = paramB;
  }

  calibration_file.close();


  // Now that we have calibration data we build the class output data structures
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).SetModelGainsAndOffsets(num_flows_, flow_window_size_);
  }

  if (verbose_)
    cout << "LinearCalibrationModel: enabled for HPs >=" << hp_threshold << " (from text file " << model_file_name << ") in a "
         << chip_subset_.GetNumRegionsX() << 'x' << chip_subset_.GetNumRegionsY() << 'x' << num_flow_windows_ << " grid." <<endl << endl;

  is_enabled_ = true;
  return is_enabled_;
}

// --------------------------------------------------------------------------

void LinearCalibrationModel::CleanSlate()
{
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).InitializeTrainingData(num_flow_windows_);
  }
}

// --------------------------------------------------------------------------

void  LinearCalibrationModel::AccumulateTrainingData(const LinearCalibrationModel& other)
{
  if (not do_training_)
    return;

  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    region_data.at(iRegion).AccumulateTrainingData(other.region_data.at(iRegion));
  }
}

// --------------------------------------------------------------------------

bool  LinearCalibrationModel::AddTrainingRead(const ReadAlignmentInfo& read_alignment)
{
  if (read_alignment.is_filtered or (not do_training_))
    return false;

  int my_nuc_idx, my_flow_idx, my_flow_window;
  double scaled_residual;

  int my_region = chip_subset_.CoordinatesToRegionIdx(read_alignment.well_xy.at(0), read_alignment.well_xy.at(1));
  if (my_region < 0){
    if (debug_)
      cout << "Ignoring read " << read_alignment.alignment->Name << ": coordinates of bounds; region idx " << my_region << endl;
    return false;
  }

  // Step through flow alignment and extract info
  for (unsigned int iHP=0; iHP < read_alignment.pretty_flow_align.size(); ++iHP){

    // Ignore Flow InDels
    if (IsInDelAlignSymbol(read_alignment.pretty_flow_align.at(iHP))) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": Flow alignment symbol is InDel." << endl;
      continue;
    }

    // Ignore HPs that are too large
    if (read_alignment.aligned_tHPs.at(iHP) > MAX_HPXLEN) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": HP size out of bounds, " << read_alignment.aligned_tHPs.at(iHP) << endl;
      continue;
    }

    my_nuc_idx = ion::FlowOrder::NucToInt(read_alignment.aln_flow_order.at(iHP));
    if (my_nuc_idx < 0){
      if (debug_)
        cout << "Ignoring HP " << iHP << ": nuc idx out of bounds, " << my_nuc_idx << endl;
      continue;
    }

    my_flow_idx = read_alignment.align_flow_index.at(iHP);
    my_flow_window = my_flow_idx / flow_window_size_;

    // Reject outlier measurements
    if (read_alignment.state_inphase.at(my_flow_idx) < min_state_inphase_){
      if (debug_)
        cout << "Ignoring HP " << iHP << ": state_inphase too small " << read_alignment.state_inphase.at(my_flow_idx) << endl;
      continue;
    }
    scaled_residual = fabs(read_alignment.measurements.at(my_flow_idx) - read_alignment.predictions_ref.at(my_flow_idx)) / read_alignment.state_inphase.at(my_flow_idx);
    if (scaled_residual > max_scaled_residual_) {
      if (debug_)
        cout << "Ignoring HP " << iHP << ": residual too high " << scaled_residual << endl;
      continue;
    }

    // And finally add the datat point to the training set
    region_data.at(my_region).AddDataPoint(my_flow_window, my_nuc_idx, min(read_alignment.aligned_tHPs.at(iHP), MAX_HPXLEN),
                  read_alignment.predictions_ref.at(my_flow_idx), read_alignment.measurements.at(my_flow_idx));

  }

  return true;
}

// --------------------------------------------------------------------------

bool  LinearCalibrationModel::CreateCalibrationModel()
{
  if (not do_training_)
    return false;

  max_hp_calibrated_ = 0;
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    if (training_mode_==1)
      max_hp_calibrated_ = max(max_hp_calibrated_, region_data.at(iRegion).CreateCalibrationModel(iRegion, min_num_samples_));
    if (training_mode_==2)
      max_hp_calibrated_ = max(max_hp_calibrated_, region_data.at(iRegion).CreateOldStyleCalibrationModel(iRegion, min_num_samples_));

  }
  is_enabled_ = true;
  //PrintModelParameters();
  return is_enabled_;
}

// --------------------------------------------------------------------------

void  LinearCalibrationModel::PrintModelParameters()
{
  cout << "Calibraiton model parameters: min_hp=" << hp_threshold_ << " max_hp_calibrated=" << max_hp_calibrated_ << endl;
  for (int iRegion=0; iRegion<chip_subset_.NumRegions(); ++iRegion) {
    cout << "Calibration Model for region " << iRegion;

    // Iterate over atomic elements
     for (unsigned int iWindow=0; iWindow < region_data.at(iRegion).gain_values.size(); ++iWindow) {
       cout << " flow window " << iWindow;

       for (unsigned int iNuc=0; iNuc < 4; ++iNuc) {
         cout << "Nucleotide " << ion::FlowOrder::IntToNuc(iNuc) << endl;

         cout << "-- Offset: ";
         for (int iHP=0; iHP <= MAX_HPXLEN; ++iHP)
           cout << region_data.at(iRegion).offset_values.at(iWindow).at(iNuc).at(iHP) << " ";
         cout << endl;

         cout << "-- Gains : ";
         for (int iHP=0; iHP <= MAX_HPXLEN; ++iHP)
           cout << region_data.at(iRegion).gain_values.at(iWindow).at(iNuc).at(iHP) << " ";
         cout << endl;

       }
     }
  }
}


