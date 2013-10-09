/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerMetricSaver.cpp
//! @ingroup  BaseCaller
//! @brief    BaseCallerMetricSaver.

#include "BaseCallerMetricSaver.h"

#include <string>
#include <cassert>
#include <cmath>
#include "IonErr.h"


void BaseCallerMetricSaver::PrintHelp()
{
  printf ("Internal metrics output options:\n");
  printf ("     --save-metrics          STRING     lists intermediate metrics to save to BaseCallerMetrics.h5 [off]\n");
  printf ("                                        a = raw_measurements          (key normalized wells)\n");
  printf ("                                        b = additive_correction       (adaptive normalization)\n");
  printf ("                                        c = multiplicative_correction (adaptive normalization)\n");
  printf ("                                        d = normalized_measurements   (adaptive-normalized wells)\n");
  printf ("                                        e = prediction                (model-predicted signal)\n");
  printf ("                                        f = solution                  (basecalls in flowspace)\n");
  printf ("                                        g = state_inphase             (live polymerase in phase)\n");
  printf ("                                        h = state_total               (live polymerase)\n");
  printf ("                                        i = penalty_residual          (base qv predictor 1)\n");
  printf ("                                        j = penalty_mismatch          (base qv predictor 5)\n");
  printf ("                                        k = local_noise               (base qv predictor 2)\n");
  printf ("                                        l = neighborhood_noise        (base qv predictor 6)\n");
  printf ("                                        m = noise_overlap             (base qv predictor 3)\n");
  printf ("                                        n = homopolymer_rank          (base qv predictor 4)\n");
  printf ("     --save-subset-only      on/off     only save metrics for the subset of reads [off]\n");
  printf ("\n");
}


BaseCallerMetricSaver::BaseCallerMetricSaver(OptArgs& opts, int chip_size_x, int chip_size_y, int num_flows,
      int region_size_x, int region_size_y, const string& output_directory)
{
  chip_size_x_ = chip_size_x;
  chip_size_y_ = chip_size_y;
  num_flows_ = num_flows;
  region_size_x_ = region_size_x;
  region_size_y_ = region_size_y;

  save_anything_ = false;
  save_raw_measurements_ = false;
  save_additive_correction_ = false;
  save_multiplicative_correction_ = false;
  save_prediction_ = false;
  save_solution_ = false;
  save_state_inphase_ = false;
  save_state_total_ = false;
  save_penalty_residual_ = false;
  save_penalty_mismatch_ = false;
  save_local_noise_ = false;
  save_neighborhood_noise_ = false;
  save_noise_overlap_ = false;
  save_homopolymer_rank_ = false;

  save_subset_only_       = opts.GetFirstBoolean('-', "save-subset-only", false);
  string arg_save_metrics = opts.GetFirstString ('-', "save-metrics", "off");
  if (arg_save_metrics != "off") {
    for (string::iterator i = arg_save_metrics.begin(); i != arg_save_metrics.end(); ++i) {
      switch (*i) {
        case 'a': save_anything_ = save_raw_measurements_           = true; break;
        case 'b': save_anything_ = save_additive_correction_        = true; break;
        case 'c': save_anything_ = save_multiplicative_correction_  = true; break;
        case 'd': save_anything_ = save_normalized_measurements_    = true; break;
        case 'e': save_anything_ = save_prediction_                 = true; break;
        case 'f': save_anything_ = save_solution_                   = true; break;
        case 'g': save_anything_ = save_state_inphase_              = true; break;
        case 'h': save_anything_ = save_state_total_                = true; break;
        case 'i': save_anything_ = save_penalty_residual_           = true; break;
        case 'j': save_anything_ = save_penalty_mismatch_           = true; break;
        case 'k': save_anything_ = save_local_noise_                = true; break;
        case 'l': save_anything_ = save_neighborhood_noise_         = true; break;
        case 'm': save_anything_ = save_noise_overlap_              = true; break;
        case 'n': save_anything_ = save_homopolymer_rank_           = true; break;
      }
    }
  }

  if (save_anything_) {
    string metric_file_name = output_directory + "/BaseCallerMetrics.h5";
    metric_file_ = H5Fcreate(metric_file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (metric_file_ < 0)
      ION_ABORT("Could not create file: " + metric_file_name);

    hsize_t   dataset_dims[3];

    // All datasets above have size chip_size_x * chip_size_y * num_flows
    dataset_dims[0] = chip_size_y_;
    dataset_dims[1] = chip_size_x_;
    dataset_dims[2] = num_flows_;
    dataspace_file_ = H5Screate_simple (3, dataset_dims, NULL);

    dataset_dims[0] = num_flows_;
    dataspace_memory_ = H5Screate_simple (1, dataset_dims, NULL);

    hid_t dataset_properties_float = H5Pcreate(H5P_DATASET_CREATE);
    hid_t dataset_properties_char = H5Pcreate(H5P_DATASET_CREATE);

//    dataset_dims[0] = region_size_y_;
//    dataset_dims[1] = region_size_x_;
    dataset_dims[0] = 1;
    dataset_dims[1] = 1;
    dataset_dims[2] = num_flows_;
    H5Pset_chunk(dataset_properties_float, 3, dataset_dims);
    H5Pset_chunk(dataset_properties_char, 3, dataset_dims);

    float initializer_float = nan("");
    printf("Initializer: %f\n", initializer_float);
    H5Pset_fill_value(dataset_properties_float, H5T_NATIVE_FLOAT, &initializer_float);


    printf("\n");
    printf("Saving selected metrics to %s :\n", metric_file_name.c_str());

    if (save_raw_measurements_) {
      printf("    a - raw_measurements\n");
      dataset_raw_measurements_ = H5Dcreate2(metric_file_, "/raw_measurements",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_additive_correction_) {
      printf("    b - additive_correction\n");
      dataset_additive_correction_ = H5Dcreate2(metric_file_, "/additive_correction",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_multiplicative_correction_) {
      printf("    c - multiplicative_correction\n");
      dataset_multiplicative_correction_ = H5Dcreate2(metric_file_, "/multiplicative_correction",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_normalized_measurements_) {
      printf("    d - normalized_measurements\n");
      dataset_normalized_measurements_ = H5Dcreate2(metric_file_, "/normalized_measurements",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_prediction_) {
      printf("    e - prediction\n");
      dataset_prediction_ = H5Dcreate2(metric_file_, "/prediction",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_solution_) {
      printf("    f - solution\n");
      dataset_solution_ = H5Dcreate2(metric_file_, "/solution",
          H5T_NATIVE_CHAR, dataspace_file_, H5P_DEFAULT, dataset_properties_char, H5P_DEFAULT);
    }
    if (save_state_inphase_) {
      printf("    g - state_inphase\n");
      dataset_state_inphase_ = H5Dcreate2(metric_file_, "/state_inphase",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_state_total_) {
      printf("    h - state_total\n");
      dataset_state_total_ = H5Dcreate2(metric_file_, "/state_total",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_penalty_residual_) {
      printf("    i - penalty_residual\n");
      dataset_penalty_residual_ = H5Dcreate2(metric_file_, "/penalty_residual",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_penalty_mismatch_) {
      printf("    j - penalty_mismatch\n");
      dataset_penalty_mismatch_ = H5Dcreate2(metric_file_, "/penalty_mismatch",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_local_noise_) {
      printf("    k - local_noise\n");
      dataset_local_noise_ = H5Dcreate2(metric_file_, "/local_noise",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_neighborhood_noise_) {
      printf("    l - neighborhood_noise\n");
      dataset_neighborhood_noise_ = H5Dcreate2(metric_file_, "/neighborhood_noise",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_noise_overlap_) {
      printf("    m - noise_overlap\n");
      dataset_noise_overlap_ = H5Dcreate2(metric_file_, "/noise_overlap",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }
    if (save_homopolymer_rank_) {
      printf("    n - homopolymer_rank\n");
      dataset_homopolymer_rank_ = H5Dcreate2(metric_file_, "/homopolymer_rank",
          H5T_NATIVE_FLOAT, dataspace_file_, H5P_DEFAULT, dataset_properties_float, H5P_DEFAULT);
    }

    H5Pclose(dataset_properties_float);
    H5Pclose(dataset_properties_char);
  }
}

void BaseCallerMetricSaver::SaveRawMeasurements(int y, int x, const vector<float>& raw_measurements)
{
  if (!save_raw_measurements_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_raw_measurements_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &raw_measurements[0]);
}

void BaseCallerMetricSaver::SaveAdditiveCorrection(int y, int x, const vector<float>& additive_correction)
{
  if (!save_additive_correction_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_additive_correction_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &additive_correction[0]);
}

void BaseCallerMetricSaver::SaveMultiplicativeCorrection(int y, int x, const vector<float>& multiplicative_correction)
{
  if (!save_multiplicative_correction_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_multiplicative_correction_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &multiplicative_correction[0]);
}

void BaseCallerMetricSaver::SaveNormalizedMeasurements(int y, int x, const vector<float>& normalized_measurements)
{
  if (!save_normalized_measurements_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_normalized_measurements_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &normalized_measurements[0]);
}

void BaseCallerMetricSaver::SavePrediction(int y, int x, const vector<float>& prediction)
{
  if (!save_prediction_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_prediction_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &prediction[0]);
}

void BaseCallerMetricSaver::SaveSolution(int y, int x, const vector<char>&  solution)
{
  if (!save_solution_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_solution_, H5T_NATIVE_CHAR, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &solution[0]);
}

void BaseCallerMetricSaver::SaveStateInphase(int y, int x, const vector<float>& state_inphase)
{
  if (!save_state_inphase_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_state_inphase_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &state_inphase[0]);
}

void BaseCallerMetricSaver::SaveStateTotal(int y, int x, const vector<float>& state_total)
{
  if (!save_state_total_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,(unsigned int)num_flows_};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_all (dataspace_memory_);
  H5Dwrite (dataset_state_total_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &state_total[0]);
}

void BaseCallerMetricSaver::SavePenaltyResidual(int y, int x, const vector<float>& penalty_residual)
{
  if (!save_penalty_residual_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)penalty_residual.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_penalty_residual_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &penalty_residual[0]);
}

void BaseCallerMetricSaver::SavePenaltyMismatch(int y, int x, const vector<float>& penalty_mismatch)
{
  if (!save_penalty_mismatch_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)penalty_mismatch.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_penalty_mismatch_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &penalty_mismatch[0]);
}



void BaseCallerMetricSaver::SaveLocalNoise(int y, int x, const vector<float>& local_noise)
{
  if (!save_local_noise_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)local_noise.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_local_noise_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &local_noise[0]);
}


void BaseCallerMetricSaver::SaveNoiseOverlap(int y, int x, const vector<float>& minus_noise_overlap)
{
  if (!save_noise_overlap_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)minus_noise_overlap.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_noise_overlap_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &minus_noise_overlap[0]);
}


void BaseCallerMetricSaver::SaveHomopolymerRank(int y, int x, const vector<float>& homopolymer_rank)
{
  if (!save_homopolymer_rank_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)homopolymer_rank.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_homopolymer_rank_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &homopolymer_rank[0]);
}


void BaseCallerMetricSaver::SaveNeighborhoodNoise(int y, int x, const vector<float>& neighborhood_noise)
{
  if (!save_neighborhood_noise_)
    return;
  hsize_t   write_start[3] = {(unsigned int)y,(unsigned int)x,0};
  hsize_t   write_count[3] = {1,1,min((unsigned int)num_flows_,(unsigned int)neighborhood_noise.size())};
  H5Sselect_hyperslab (dataspace_file_, H5S_SELECT_SET, write_start, NULL, write_count, NULL);
  H5Sselect_hyperslab (dataspace_memory_, H5S_SELECT_SET, write_start+2, NULL, write_count+2, NULL);
  H5Dwrite (dataset_neighborhood_noise_, H5T_NATIVE_FLOAT, dataspace_memory_, dataspace_file_,
         H5P_DEFAULT, &neighborhood_noise[0]);
}





void BaseCallerMetricSaver::Close()
{
  if (!save_anything_)
    return;

  if (save_raw_measurements_)           H5Dclose(dataset_raw_measurements_);
  if (save_additive_correction_)        H5Dclose(dataset_additive_correction_);
  if (save_multiplicative_correction_)  H5Dclose(dataset_multiplicative_correction_);
  if (save_normalized_measurements_)    H5Dclose(dataset_normalized_measurements_);
  if (save_prediction_)                 H5Dclose(dataset_prediction_);
  if (save_solution_)                   H5Dclose(dataset_solution_);
  if (save_state_inphase_)              H5Dclose(dataset_state_inphase_);
  if (save_state_total_)                H5Dclose(dataset_state_total_);
  if (save_penalty_residual_)           H5Dclose(dataset_penalty_residual_);
  if (save_penalty_mismatch_)           H5Dclose(dataset_penalty_mismatch_);
  if (save_local_noise_)                H5Dclose(dataset_local_noise_);
  if (save_neighborhood_noise_)         H5Dclose(dataset_neighborhood_noise_);
  if (save_noise_overlap_)              H5Dclose(dataset_noise_overlap_);
  if (save_homopolymer_rank_)           H5Dclose(dataset_homopolymer_rank_);

  H5Sclose(dataspace_memory_);
  H5Sclose(dataspace_file_);
  H5Fclose(metric_file_);
}


