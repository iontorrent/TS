/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PerBaseQual.cpp
//! @ingroup  BaseCaller
//! @brief    PerBaseQual. Determination of base qualities from predictors

#include "PerBaseQual.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "ChipIdDecoder.h"
#include "Utils.h"
#include "IonErr.h"

using namespace std;


PerBaseQual::PerBaseQual()
{
  phred_thresholds_.resize(kNumPredictors);
  phred_thresholds_max_.resize(kNumPredictors);
  pthread_mutex_init(&predictor_mutex_, 0);
}


PerBaseQual::~PerBaseQual()
{
  if (save_predictors_)
    predictor_dump_.close();
  pthread_mutex_destroy(&predictor_mutex_);
}


void PerBaseQual::PrintHelp()
{
  printf ("Quality values generation options:\n");
  printf ("     --phred-table-file      FILE       predictor / quality value file [chip default]\n");
  printf ("     --save-predictors       on/off     dump predictors for every called base to Predictors.txt [off]\n");
  printf ("\n");
}


void PerBaseQual::Init(OptArgs& opts, const string& chip_type, const string &output_directory)
{
  string phred_table_file       = opts.GetFirstString ('-', "phred-table-file", "");
  save_predictors_              = opts.GetFirstBoolean('-', "save-predictors", false);

  // Determine the correct phred table filename to use

  if (phred_table_file.empty()) {

    ChipIdDecoder::SetGlobalChipId(chip_type.c_str());
    ChipIdEnum chip_id = ChipIdDecoder::GetGlobalChipId();
    switch(chip_id){
    case ChipId314:
      phred_table_file = "phredTable.txt_314";
      break;
    case ChipId316:
      phred_table_file = "phredTable.txt_316";
      break;
    case ChipId318:
      phred_table_file = "phredTable.txt_318";
      break;
    case ChipId900: // Proton chip
      phred_table_file = "phredTable.txt_900";
      break;
    default:
      phred_table_file = "phredTable.txt_314";
      fprintf(stderr, "PerBaseQual: No default phred table for chip_type=%s, trying %s instead\n",
          chip_type.c_str(), phred_table_file.c_str());
      break;
    }

    char* full_filename = GetIonConfigFile(phred_table_file.c_str());
    if(!full_filename)
      ION_ABORT("ERROR: Can't find phred table file " + phred_table_file);
    phred_table_file = full_filename;
    free(full_filename);
  }

  // Load the phred table

  ifstream source;
  source.open(phred_table_file.c_str());
  if (!source.is_open())
    ION_ABORT("ERROR: Cannot open file: " + phred_table_file);

  while (!source.eof()) {
    string line;
    getline(source, line);

    if (line.empty())
      break;

    if (line[0] == '#')
      continue;

    stringstream strs(line);
    float temp;
    for (int k = 0; k < kNumPredictors; ++k) {
      strs >> temp;
      phred_thresholds_[k].push_back(temp);
    }
    strs >> temp; //skip n-th entry
    strs >> temp;
    phred_quality_.push_back(temp);
  }

  for (int k = 0; k < kNumPredictors; ++k)
    phred_thresholds_max_[k] = *max_element(phred_thresholds_[k].begin(), phred_thresholds_[k].end());

  // Prepare for predictor dump here

  if (save_predictors_) {
    string predictors_filename = output_directory + "/Predictors.txt";
    cout << endl << "Saving PerBaseQual predictors to file " << predictors_filename << endl << endl;
    predictor_dump_.open(predictors_filename.c_str());
    if (!predictor_dump_.is_open())
      ION_ABORT("ERROR: Cannot open file: " + predictors_filename);
  }
}


uint8_t PerBaseQual::CalculatePerBaseScore(float* pred) const
{
  int num_phred_cuts = phred_quality_.size(); // number of rows/lines in the table

  for (int k = 0; k < kNumPredictors; k++)
    pred[k] = min(pred[k], phred_thresholds_max_[k]);

  for ( int j = 0; j < num_phred_cuts; ++j ) {
    bool valid_cut = true;

    for ( int k = 0; k < kNumPredictors; ++k ) {
      if (pred[k] > phred_thresholds_[k][j]) {
        valid_cut = false;
        break;
      }
    }

    if (valid_cut)
      return phred_quality_[j];
  }

  return kMinQuality; //minimal quality score
}




// Predictor 2 - Local noise/flowalign - 'noise' in the input base's measured val.  Noise is max[abs(val - round(val))] within +-1 BASE
void PerBaseQual::PredictorLocalNoise(vector<float>& local_noise, int max_base, const vector<int>& base_to_flow, const vector<float>& corrected_ionogram)
{
  int num_bases = base_to_flow.size();
  for (int base = 0; base < max_base; ++base) {
    int val1 = max(base - 1, 0);
    int val2 = min(base + 1, num_bases - 1);
    float noise = 0;
    for (int j = val1; j <= val2; ++j) {
      // Go from float to 100-based integer accuracy and back to float (e.g. 1.16945 -> 1.17)
      float current_flow_val = 0.01 * rint(100 * corrected_ionogram[base_to_flow[j]]);
      noise = max(noise, fabsf(current_flow_val - roundf(current_flow_val))); // This is just residual
    }
    local_noise[base] = noise;
  }
}


// Predictor 3  - Read Noise/Overlap - mean & stdev of the 0-mers & 1-mers in the read
// -(m_1 - m_0 - s_1 - s_0)/m_1
void PerBaseQual::PredictorNoiseOverlap(vector<float>& minus_noise_overlap, int max_base,
    const vector<float>& corrected_ionogram)
//float PerBaseQual::PredictorNoiseOverlap(const vector<float>& corrected_ionogram)
{
  // 0-mer and 1-mer overlap
  // define 0-mer and 1-mer interval
  float cutoff0 = 0.5;
  float cutoff1 = 1.5;
  int max_iter = 2; // adjust cutoffs once
  int num_flows_to_use = min((int)corrected_ionogram.size(), 60);
  float noise_overlap;

  for (int i = 0; i < max_iter; i++) {
    int one_counter = 0;
    int zero_counter = 0;
    float mean_zero = 0.0;
    float mean_one = 0.0;
    float stdev_zero = 0.0;
    float stdev_one = 0.0;

    for (int flow = 8; flow < num_flows_to_use; ++flow) {
      float current_flow_val = 0.01 * rint(100 * corrected_ionogram[flow]);
      if (current_flow_val < cutoff0) {
        mean_zero += current_flow_val;
        zero_counter++;

      } else if (current_flow_val < cutoff1) {
        mean_one += current_flow_val;
        one_counter++;
      }
    }

    if (zero_counter)
      mean_zero /= zero_counter;

    if (one_counter)
      mean_one /= one_counter;
    else
      mean_one = 1;

    for (int flow = 8; flow < num_flows_to_use; ++flow) {
      // Go from float to 100-based integer accuracy and back to float (e.g. 1.16945 -> 1.17)
      float current_flow_val = 0.01 * rint(100 * corrected_ionogram[flow]);
      if (current_flow_val < cutoff0 )
        stdev_zero += (current_flow_val - mean_zero) * (current_flow_val - mean_zero);
      else if (current_flow_val < cutoff1)
        stdev_one += (current_flow_val - mean_one) * (current_flow_val - mean_one);
    }

    if (zero_counter)
      stdev_zero = sqrt(stdev_zero / zero_counter);

    if (one_counter)
      stdev_one = sqrt(stdev_one / one_counter);

    noise_overlap = (mean_one - mean_zero - stdev_one - stdev_zero) / mean_one;

    // calculate new cutoffs for next iteration
    if (stdev_one or stdev_zero) {
      cutoff0 = (mean_one * stdev_zero + mean_zero * stdev_one) / (stdev_one + stdev_zero);
      cutoff1 = 2 * mean_one - cutoff0;
    } else
      break;
  }


  for (int base = 0; base < max_base; ++base)
    minus_noise_overlap[base] = -noise_overlap;
//  return -noise_overlap;
}

// Predictor 4 - Transformed homopolymer length
void PerBaseQual::PredictorHomopolymerRank(vector<float>& homopolymer_rank, int max_base, const vector< uint8_t >& flow_index)
{
  int hp_length = 1;
  for (int base = 0; base < max_base; base += hp_length) {

    for (hp_length = 1; base+hp_length < (int)flow_index.size(); ++hp_length)
      if (flow_index[base+hp_length])
        break;

    // other patterns tried in the past: HP 333, HP 1124

    // HP 1114
    for (int hp = 0; hp < hp_length and base+hp < max_base; hp++) {
      if (hp == hp_length-1)
        homopolymer_rank[base+hp] = hp_length;
      else
        homopolymer_rank[base+hp] = 1;
    }
  }
}


// Predictor 6 - Neighborhood noise - mean of 'noise' +-5 BASES around a base.  Noise is mean{abs(val - round(val))}
void PerBaseQual::PredictorNeighborhoodNoise(vector<float>& neighborhood_noise, int max_base, const vector<int>& base_to_flow,
    const vector<float>& corrected_ionogram)
{
  int num_bases = base_to_flow.size();
  for (int base = 0; base < max_base; ++base) {
    int radius = 5;
    // protect at start/end of read
    int val1 = max(base-radius, 0);
    int val2 = min(base+radius, num_bases-1);

    float noise = 0;
    int count = 0;
    for (int j = val1; j <= val2; j++) {
      // Go from float to 100-based integer accuracy and back to float (e.g. 1.16945 -> 1.17)
      float current_flow_val = 0.01 * rint(100 * corrected_ionogram[base_to_flow[j]]);
      noise += fabsf(current_flow_val - roundf(current_flow_val));
      count++;
    }
    if (count)
      noise /= count;
    neighborhood_noise[base] = noise;
  }
}


void PerBaseQual::GenerateBaseQualities(const string& read_name, int num_bases, int num_flows,
    const vector<float> &predictor1, const vector<float> &predictor2,
    const vector<float> &predictor3, const vector<float> &predictor4,
    const vector<float> &predictor5, const vector<float> &predictor6,
    const vector<int>& base_to_flow, vector<uint8_t> &quality,
    const vector<float> &candidate1,
    const vector<float> &candidate2,
    const vector<float> &candidate3)
{

  if (num_bases == 0)
    return;

  //! \todo This is a temporary fix for very long sequences that are sometimes generated by the basecaller
  int max_eligible_base = min(num_bases, (int)(0.75*num_flows) + 1);
  quality.clear();

  stringstream predictor_dump_block;

  for (int base = 0; base < max_eligible_base; base++) {

    float pred[kNumPredictors];
    pred[0] = transform_P1(predictor1[base]);
    pred[1] = predictor2[base];
    pred[2] = predictor3[base];
    pred[3] = predictor4[base];

    if (save_predictors_) {
      // pred[4] & pred[5] are not the same in new QvTables
      // the following two lines are only for predictor_dump_block
      pred[4] = transform_P5(predictor5[base]);
      pred[5] = transform_P6(predictor6[base]);

      predictor_dump_block << read_name << " " << base << " ";
      for (int k = 0; k < kNumPredictors; ++k)
        predictor_dump_block << pred[k] << " ";
      predictor_dump_block << candidate1[base_to_flow[base]] << " ";
      predictor_dump_block << candidate2[base_to_flow[base]] << " ";
      predictor_dump_block << candidate3[base_to_flow[base]] << " ";
      predictor_dump_block << base_to_flow[base] << endl;
    }

    // the real predictors used in the QvTable
    pred[4] = transform_P6(predictor6[base]);
    pred[5] = transform_P8(candidate2[base_to_flow[base]]);
    quality.push_back(CalculatePerBaseScore(pred));
  }

  for (int base = max_eligible_base; base < num_bases; base++)
    quality.push_back(kMinQuality);

  if (save_predictors_) {
    predictor_dump_block.flush();
    pthread_mutex_lock(&predictor_mutex_);
    predictor_dump_ << predictor_dump_block.str();
    predictor_dump_.flush();
    pthread_mutex_unlock(&predictor_mutex_);
  }
}



float PerBaseQual::transform_P1(float p)
{
    float peak = 0.005;
    if (p < peak)
        p = peak + fabs(p-peak);
    return p;
}


float PerBaseQual::transform_P5(float p)
{
    float peak = -0.6;
    if (p < peak)
    {
    if (p < -0.96)
        p = peak + fabs(p-peak) + fabs(p+0.96)*0.5;
    else
        p = peak + fabs(p-peak);
    }
    else if (p > -0.09)
        {
        if (p < 0.05)
            p = -0.09 - fabs(p+0.09) * 2;
        else
            p = -0.3+(p-0.05)*0.25;
    }
    return p;
}


float PerBaseQual::transform_P6(float p)
{
    float peak = 0.06;
    if (p < peak)
        p = peak + fabs(p-peak);
    peak = 0.4;
    if (p > peak)
        p = peak - fabs(p-peak);

    return p;
}


float PerBaseQual::transform_P7(float p)
{
    float peak = -0.05;
    if (p < peak)
        p = peak + fabs(p-peak);

    return p;
}


float PerBaseQual::transform_P8(float p)
{
    p = -p;
    return p;
}


float PerBaseQual::transform_P9(float p)
{
    p = -p;
    float peak = -0.9;
    if (p < peak)
        p = peak + abs(p - peak)*2;

    return p;
}


