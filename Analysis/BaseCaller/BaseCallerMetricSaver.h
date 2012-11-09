/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerMetricSaver.h
//! @ingroup  BaseCaller
//! @brief    BaseCallerMetricSaver.

#ifndef BASECALLERMETRICSAVER_H
#define BASECALLERMETRICSAVER_H

#include <string>
#include <vector>
#include "hdf5.h"
#include "OptArgs.h"

using namespace std;


class BaseCallerMetricSaver {
public:
  BaseCallerMetricSaver(OptArgs& opts, int chip_size_x, int chip_size_y, int num_flows,
      int region_size_x, int region_size_y, const string& output_directory);

  void SaveRawMeasurements          (int y, int x, const vector<float>& raw_measurements);
  void SaveAdditiveCorrection       (int y, int x, const vector<float>& additive_correction);
  void SaveMultiplicativeCorrection (int y, int x, const vector<float>& multiplicative_correction);
  void SaveNormalizedMeasurements   (int y, int x, const vector<float>& normalized_measurements);
  void SavePrediction               (int y, int x, const vector<float>& prediction);
  void SaveSolution                 (int y, int x, const vector<char>&  solution);
  void SaveStateInphase             (int y, int x, const vector<float>& state_inphase);
  void SaveStateTotal               (int y, int x, const vector<float>& state_total);
  void SavePenaltyResidual          (int y, int x, const vector<float>& penalty_residual);
  void SavePenaltyMismatch          (int y, int x, const vector<float>& penalty_mismatch);
  void SaveLocalNoise               (int y, int x, const vector<float>& local_noise);
  void SaveNoiseOverlap             (int y, int x, const vector<float>& minus_noise_overlap);
  void SaveHomopolymerRank          (int y, int x, const vector<float>& homopolymer_rank);
  void SaveNeighborhoodNoise        (int y, int x, const vector<float>& neighborhood_noise);

  void Close();

  static void PrintHelp();

  bool save_anything() const { return save_anything_; }
  bool save_subset_only() const { return save_subset_only_; }

protected:
  int     chip_size_x_;
  int     chip_size_y_;
  int     num_flows_;
  int     region_size_x_;
  int     region_size_y_;

  // Intermediate basecaller
  bool    save_anything_;
  bool    save_subset_only_;
  bool    save_raw_measurements_;
  bool    save_additive_correction_;
  bool    save_multiplicative_correction_;
  bool    save_normalized_measurements_;
  bool    save_prediction_;
  bool    save_solution_;
  bool    save_state_inphase_;
  bool    save_state_total_;
  bool    save_penalty_residual_;
  bool    save_penalty_mismatch_;

  bool    save_local_noise_;
  bool    save_neighborhood_noise_;
  bool    save_noise_overlap_;
  bool    save_homopolymer_rank_;

  // hdf5 file and dataset details
  hid_t   metric_file_;
  hid_t   dataspace_file_;
  hid_t   dataspace_memory_;

  hid_t   dataset_raw_measurements_;
  hid_t   dataset_additive_correction_;
  hid_t   dataset_multiplicative_correction_;
  hid_t   dataset_normalized_measurements_;
  hid_t   dataset_prediction_;
  hid_t   dataset_solution_;
  hid_t   dataset_state_inphase_;
  hid_t   dataset_state_total_;
  hid_t   dataset_penalty_residual_;
  hid_t   dataset_penalty_mismatch_;
  hid_t   dataset_local_noise_;
  hid_t   dataset_neighborhood_noise_;
  hid_t   dataset_noise_overlap_;
  hid_t   dataset_homopolymer_rank_;


};


#endif // BASECALLERMETRICSAVER_H
