/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     HistogramCalibration.h
//! @ingroup  Calibration
//! @brief    HistogramCalibration. Algorithms for adjusting signal intensity and base calls using calibration tables


#ifndef HISTOGRAMCALIBRATION_H
#define HISTOGRAMCALIBRATION_H

#include <vector>
#include <stdint.h>

#include "CalibrationHelper.h"

using namespace std;

// ==============================================================================

class HistogramCalibration {

public:
  // Constructor for calibration training
  HistogramCalibration(OptArgs& opts, const CalibrationContext& calib_context);

  // Constructor for calibration application in BaseCaller
  HistogramCalibration(OptArgs& opts, const ion::FlowOrder& flow_order);


  // Zeros out memory elements to accumulate training data.
  void  CleanSlate();

  // Adds the data from an aligned read to the training histogram
  bool  AddTrainingRead(const ReadAlignmentInfo& read_alignment);

  bool  GetHistogramBin(float measurement, float prediction, float state_inphase, int& bin, double& scaled_residual);

  void  AccumulateHistData(const HistogramCalibration& other);

  bool  CreateCalibrationModel();

  void  ExportModelToJson(Json::Value &json);

  void  CalibrateRead(const ion::FlowOrder& flow_order, int well_x, int well_y, BasecallerRead & read);

  bool  InitializeModelFromJson(Json::Value &json);

  bool  InitializeModelFromLegacyFile(string legacy_file_name);

  bool  is_enabled() const { return is_enabled_; }

  void  SetDebug(bool debug) { debug_ = debug; };
  void  Defaults();

  static void PrintHelp_Training();

private:

  // ------------------------------
  // Hold signal histogram data for one nucleotide over all HP lengths
  struct HistogramTrainingElement {

    // Histogram bins for old school algorithm
    vector<uint32_t> seen_correct;
    vector<uint32_t> seen_offbyone;
    vector<uint32_t> seen_other;

    // Parameters to fit a distribution
    vector<double>   means;
    vector<double>   sum_squares;
    vector<uint64_t> num_samples;
    vector<uint64_t> ignored_samples;

    bool  AccumulateStatistics(const HistogramTrainingElement & other);
    void  AddDataPoint(int iHP, double sclaled_residual);

  };

  // -----------------------------
  // We make a private class so we can change the layout later
  // This class holds info for one chip region and one flow window
  class HistogramElement {
  public:

    HistogramElement() : training_mode_(-1) {};

    void         ClearTrainingData();

	void         SetTrainingModeAndSize(int training_mode, int num_hps, int num_bins);

	bool         AccumulateTrainingData(const HistogramElement & other);

    void         SmoothHistograms(bool clip_mask, bool fractional_smooth, float min_smooth);

	string       GetBinBoundaries(unsigned int min_observations, int min_hp);

	Json::Value  ExportBinsToJson();

    Json::Value  ExportHistogramsToJson();

	void         FromJson(Json::Value& json, int num_hps);

	// Model training data structures
    vector<HistogramTrainingElement> histogram_data;
    int                              training_mode_;

	// Model application data structures:
    // This structure hold our histogram calibration results by nuc and hp
    vector<vector<int> >             bin_boundaries;

  };

  // -----------------------------

  ion::ChipSubset                    chip_subset_;    //!< Chip coordinate & region handling for Basecaller
  vector<vector<HistogramElement> >  hist_element_;   //!< Structure of elements to collect model data
  vector<vector<int> >               jump_index;      //!< Flow order jump table

  int                num_flow_windows_;               //!< Number of flow windows
  int                flow_window_size_;               //!< Size of a flow window
  int                num_hps_;                        //!< Number of homopolymers to train
  int                num_bins_;                       //!< Number of histogram bins
  int                min_observations_per_bin_;       //!< Minimum number of observations in histopgram
  float              min_state_inphase_;              //!< Minimum state inphase to be included
  string             training_method_;                //!< Input string to select training method
  int                training_mode_;                  //!< Integer representation of training method

  // counter
  uint32_t           num_high_residual_reads_;        //!< Reads that have abs. scaled residuals of >0.5

  bool               output_training_stats_;          //!< Write training statistics into output json
  bool               process_zero_mers_;              //!< Switch to calibrate to and from zero-mers
  bool               threshold_residuals_;            //!< Clip scaled residuals to [-0.5,0.5]
  bool               fractional_smooth_;              //!< transition between smoothed histogram for bad data and unsmoothed for good
  bool               is_enabled_;                     //!< Ready to calibrate a read?
  bool               do_training_;                    //!< Is training on or off for this module?
  bool               debug_;   //!< Are we in debug mode?


};


#endif // BASECALLERRECALIBRATION_H
