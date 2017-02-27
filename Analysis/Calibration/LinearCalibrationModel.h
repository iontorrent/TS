/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     LinearCalibrationModel.h
//! @ingroup  Calibration
//! @brief    LinearCalibrationModel. Algorithms for adjusting the predicted signal by homopolymer.

#ifndef LINEARCALIBRATIONMODEL_H
#define LINEARCALIBRATIONMODEL_H

#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <cstdlib>

#include "CalibrationHelper.h"

using namespace std;


// ====================================================================
// Pointer collection for variant caller

class MultiAB{
  public:
    const vector<vector<vector<float> > > * aPtr;
    const vector<vector<vector<float> > > * bPtr;

    MultiAB(){aPtr=0; bPtr = 0;};
    void Null(){aPtr=0; bPtr = 0;};
    bool Valid(){return (aPtr != 0 && bPtr != 0);};
};

// ====================================================================
// Atomic collection of statistics to generate moments and a linear fit for a particular HP

class LinearFitCell {

public:
  // @brief    Constructor: Calls Reset
  LinearFitCell();

  // @brief   Resets all member variables for data collection
  void   Reset();

  void   AccumulateTrainingData(const LinearFitCell& other);

  void   GetSlopeAndInterceptFit(double &gain, double &offset) const;
  void   GetSafeSlopeAndInterceptFit(double &gain, double &offset, double safety_frac) const;
  void   Prior(float center, float var, int xamples);
  void   ReduceStrength(int xamples);
  void   GetSlopeOnlyFit(double &gain) const;

  double GetSlopeOnlyFitWithOffset(double offset) const;

  double GetOffsetOnly() const;

  // @brief   Variable update for adding one more data point
  void   AddDataPoint(float prediction, float measured);

  // Varaible access functions
  unsigned long NumSamples() const { return nsamples_; };
  unsigned long NumReads()   const { return nreads_; };
  void          SetActive(bool active) { active_ = active; };
  Json::Value   DataToJson() const;

private:

  double            mean_pred_;         //!< Online update of the mean of the predictions:  E[p]
  double            mean_measured_;     //!< Online update of the mean of the measurements: E[m]
  double            M2pred_;            //!< Centered sum of squares of predictions: sum(p_i-E[p])
  double            M2meas_;            //!< Centered sum of squares, measurements:  sum(m_i-E[m])
  double            Cn_;                //!< Centered co-moment sum: sum((p_i-E[p])*(m_i-E[m]))
  unsigned long     nsamples_;          //!< Number of data points collected
  unsigned long     nreads_;            //!< Number of reads actually used for this
  bool              active_;            //!< do I update number reads, or is this another data point  from the same read


};

// ====================================================================
// Collection of data and methods to process an individual calibration region

class LinCalModelRegion {
public:

  void  Initialize(int num_flow_windows);

  void  InitializeTrainingData(int num_flow_windows);

  void  SetModelGainsAndOffsets(int num_flows, int flow_window_size);

  void  AddDataPoint(int flow_window, int nuc, int hp, float prediction, float measured);
  void  FreshReadData();

  void  AccumulateTrainingData(const LinCalModelRegion& other);

  int   CreateCalibrationModel(int region_idx, unsigned long min_nsamples, float max_gain_shift, bool verbose);
  int   CreateOldStyleCalibrationModel(int region_idx, unsigned long min_nsamples, bool verbose);

  void  CoefficientZeroOrderHold(int start_hp);

  // Functions to return calibration coefficients
  const vector<vector<vector<float> > > * getGains()   const { return &gains; };
  const vector<vector<vector<float> > > * getOffsets() const { return &offsets; };

  const vector<vector<vector<float> > > * getAs() const { return &gains; };
  const vector<vector<vector<float> > > * getBs() const { return &offsets; };
  void  getAB(MultiAB &multi_ab) const { multi_ab.aPtr = &gains; multi_ab.bPtr = &offsets; };


  // We keep the region coefficient layout from the old RecalibrationModel code
  // We also keep the base caller hook constant: We have
  // gain and offset stratified by <flow><nucleotide><hp length>

  vector<vector<vector<float> > >             gains;
  vector<vector<vector<float> > >             offsets;

  // And the data structure below stores the individual coefficients per flow window
  // This consumes about 2-3 orders of magnitude less memory than above structures

  vector<vector<vector<float> > >             gain_values;
  vector<vector<vector<float> > >             offset_values;

  // Training data; similarly stratified by <flow window><nucleotide><hp length>
  vector<vector<vector<LinearFitCell> > >     training_data;

};


// ====================================================================
// Main Module

class LinearCalibrationModel {

public:

  // Default constructor (important for TVC)
  LinearCalibrationModel();

  // Constructor for initialization in BaseCaller
  LinearCalibrationModel(OptArgs& opts, vector<string> &bam_comments, const string & run_id,
      const ion::ChipSubset & chip_subset, const ion::FlowOrder * flow_order);

  // Constructor for calibration training
  LinearCalibrationModel(OptArgs& opts, const CalibrationContext& calib_context);

  void Defaults(); // shared default assumptions for all constructors

  bool  InitializeModelFromJson(Json::Value &json, const int num_flows=0);

  bool  InitializeModelFromTxtFile(string model_file_name, int hp_threshold, const int num_flows=0);

  void  CleanSlate();

  bool  AddTrainingRead(const ReadAlignmentInfo& read_alignment,  LinearCalibrationModel &linear_sim);
  bool  AddBlindTrainingRead(const ReadAlignmentInfo& read_alignment, LinearCalibrationModel &linear_model_cal_sim);
  bool  FilterBlindReads(const ReadAlignmentInfo& read_alignment, LinearCalibrationModel &linear_model_cal_sim,
                                                int my_region, int iHP, float &yobs, float &xobs, int &governing_hp);
  bool  FilterBlindReadsOld(const ReadAlignmentInfo& read_alignment, LinearCalibrationModel &linear_model_cal_sim,
                                                int my_region, int iHP, float &yobs, float &xobs, int &governing_hp);
  bool  FilterTrainingReads(const ReadAlignmentInfo& read_alignment, LinearCalibrationModel &linear_model_cal_sim,
                                                int my_region, int iHP, float &yobs, float &xobs, int &governing_hp);

  void  AccumulateTrainingData(const LinearCalibrationModel& other);
  void  CopyTrainingData(const LinearCalibrationModel& other);

  bool  CreateCalibrationModel(bool verbose=true);
  void  SetModelGainsAndOffsets(); // need to transfer model gains to expanded form if using trained model directly

  // helper function for polishing after linear model
  float ReturnLocalInterval(float original_step, float local_prediction, float called_hp, float finish_hp,
                                                    int my_region, int my_flow_window, int my_nuc_idx);

  // @brief   Writing model information to a json structure
  void  ExportModelToJson(Json::Value &json, string run_id) const;

  void  SaveModelFileToBamComments(vector<string> &comments, const string &run_id) const;

  void  PrintModelParameters();

  void  SetHPthreshold(int hp_threshold) { hp_threshold_ = hp_threshold; };

  // Interface functions to access calibration coefficients
  // Coefficients are for a chip region and stratified by <flow><nucleotide><hp length>
  const vector<vector<vector<float> > > * getGains(int x, int y) const;
  const vector<vector<vector<float> > > * getOffsets(int x, int y) const;
  const vector<vector<vector<float> > > * getAs(int x, int y) const { return getGains(x, y); };
  const vector<vector<vector<float> > > * getBs(int x, int y) const { return getOffsets(x, y); };

  // Variant caller interface
  void  getAB(MultiAB &multi_ab, int x, int y) const;

  bool DoTraining() const { return do_training_; };
  bool is_enabled() const { return is_enabled_; };
  void disable() { is_enabled_=false; };

  static void PrintHelp_Training();


  // Data Structure to hold calibration region data
  vector<LinCalModelRegion>      region_data;

private:

  ion::ChipSubset                chip_subset_;             //!< Chip coordinate & region handling for Basecaller
  int                            num_flows_;               //!< Number of flows the calibration model is built over
  int                            num_flow_windows_;        //!< Number of flow windows used
  int                            flow_window_size_;        //!< Size of a flow window
  int                            hp_threshold_;            //!< Starting value for HP training
  int                            max_hp_calibrated_;       //!< Maximum HP value for which we could accumulate significant amount of training data

  unsigned long                  min_num_samples_;         //!< Minimum number of required samples to compute parameters
  unsigned long                  min_num_reads_;           //!< minimum number of reads to escalate training: don't want all data from one read
  float                          min_state_inphase_;       //!<
  double                         max_scaled_residual_;     //!<

  bool                           output_training_stats_;   //!< Write training statistics into output json
  bool                           is_enabled_;
  bool                           do_training_;
  int                            training_mode_;
  bool                           multi_train_;
  int                            multi_weight_;
  float                          max_gain_shift_;          //!< control change between models as we increase, useful for partial training
  string                         training_method_;
  bool                           verbose_;
  bool                           debug_;
  bool                            spam_debug_;

};

#endif // LINEARCALIBRATIONMODEL_H
