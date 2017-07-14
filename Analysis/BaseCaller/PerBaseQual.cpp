/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PerBaseQual.cpp
//! @ingroup  BaseCaller
//! @brief    PerBaseQual. Determination of base qualities from predictors

#include "PerBaseQual.h"

#include <cstring>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>

#include "ChipIdDecoder.h"
#include "Utils.h"
#include "IonErr.h"

#include <hdf5.h>


using namespace std;


PerBaseQual::PerBaseQual()
: phred_table_(0), save_predictors_(false)
{
  phred_thresholds_.resize(kNumPredictors);
  phred_thresholds_max_.resize(kNumPredictors);
  pthread_mutex_init(&predictor_mutex_, 0);
}


PerBaseQual::~PerBaseQual()
{
  if(phred_table_)
  {
    delete [] phred_table_;
    phred_table_ = 0;
  }

  if (save_predictors_)
    predictor_dump_.close();
  pthread_mutex_destroy(&predictor_mutex_);
}


void PerBaseQual::PrintHelp()
{
  printf ("Quality values generation options:\n");
  printf ("     --phred-table-file      FILE       predictor / quality value file [chip default]\n");
  printf ("     --save-predictors       on/off     dump predictors for every called base to Predictors.txt [off]\n");
  //printf ("     --enzyme-name           STRING     name of the enzyme [off]\n");
  printf ("\n");
}


void PerBaseQual::Init(OptArgs& opts, const string& chip_type, const string &input_directory, const string &output_directory, bool recalib)
{
  if(phred_table_)
  {
    delete [] phred_table_;
    phred_table_ = 0;
  }

  string phred_table_file       = opts.GetFirstString ('-', "phred-table-file", "");
  save_predictors_              = opts.GetFirstBoolean('-', "save-predictors", false);
  //enzyme_name_                     = opts.GetFirstString ('-', "enzyme-name", "");
  // Determine the correct phred table filename to use

  bool binTable = true;
  char *full_filename = NULL;
  if (phred_table_file.empty()) { // no phred table specified via the --phred-table-file option
    full_filename = get_phred_table_name(chip_type,recalib); // default table name (not enzyme-specific)
    //if (!enzyme_name_.empty()) full_filename = get_phred_table_name(chip_type,recalib,enzyme_name_);

    if(!full_filename)
    {
      printf("WARNING: cannot find binary phred table file %s, try to use non-binary phred table\n", phred_table_file.c_str());
      //phred_table_file = phred_table_file.substr(0, phred_table_file.length() - 7); // get rid of .binary
      phred_table_file = phred_table_file.substr(0, phred_table_file.length() - 3); // get rid of .h5
      binTable = false;
      char* full_filename2 = GetIonConfigFile(phred_table_file.c_str());
      if(!full_filename2)
        ION_ABORT("ERROR: Can't find phred table file " + phred_table_file);

      phred_table_file = full_filename2;
      free(full_filename2);
    }
    else
    {
      phred_table_file = full_filename;
      free(full_filename);
    }
  }
  else if (recalib) // this is mainly to handle the phred_table_file passed by the pipeline that does not contain ".Recal"
  {
      phred_table_file = add_Recal_to_phredTableName(phred_table_file);
  }
  cout << endl << "PerBaseQual::Init... phred_table_file=" << phred_table_file << endl;
  binTable = hasBinaryExtension(phred_table_file);

  // Load the phred table
  if(binTable)
  {
    cout << endl << "PerBaseQual::Init... load binary phred_table_file=" << phred_table_file << endl;
    vector<size_t> vNumCuts(kNumPredictors, 0);

    if(H5Fis_hdf5(phred_table_file.c_str()) > 0)
    {
      hid_t root = H5Fopen(phred_table_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if(root < 0)
      {
        ION_ABORT("ERROR: cannot open HDF5 file " + phred_table_file);
      }

      hid_t grpQvTable = H5Gopen(root, "/QvTable", H5P_DEFAULT);
      if (grpQvTable < 0)
      {
        H5Fclose(root);
        ION_ABORT("ERROR: fail to open HDF5 group QvTable");
      }

      if(H5Aexists(grpQvTable, "NumPredictors") <= 0)
      {
        H5Gclose(grpQvTable);
        H5Fclose(root);
        ION_ABORT("ERROR: HDF5 attribute NumPredictors does not exist");
      }

      hid_t attrNumPreds = H5Aopen(grpQvTable, "NumPredictors", H5P_DEFAULT);
      if (attrNumPreds < 0)
      {
        H5Gclose(grpQvTable);
        H5Fclose(root);
        ION_ABORT("ERROR: fail to open HDF5 attribute NumPredictors");
      }

      unsigned int numPredictors = 0;
      herr_t ret = H5Aread(attrNumPreds, H5T_NATIVE_UINT, &numPredictors);
      H5Aclose(attrNumPreds);
      if(ret < 0 || numPredictors != (unsigned int)kNumPredictors)
      {
        H5Gclose(grpQvTable);
        H5Fclose(root);
        ION_ABORT("ERROR: HDF5 attribute NumPredictors is wrong");
      }

      char buf[100];
      for(size_t i = 0; i < (size_t)kNumPredictors; ++i)
      {
        offsets_.push_back(1);

        sprintf(buf, "ThresholdsOfPredictor%d", (int)i);

        if(H5Aexists(grpQvTable, buf) <= 0)
        {
          H5Gclose(grpQvTable);
          H5Fclose(root);
          ION_ABORT("ERROR: HDF5 attribute ThresholdsOfPredictor does not exist");
        }

        hid_t attrCuts = H5Aopen(grpQvTable, buf, H5P_DEFAULT);
        if (attrCuts < 0)
        {
          H5Gclose(grpQvTable);
          H5Fclose(root);
          ION_ABORT("ERROR: fail to open HDF5 attribute ThresholdsOfPredictor");
        }

        hsize_t size = H5Aget_storage_size(attrCuts);
        size /= sizeof(float);

        float* fcuts = new float[size];

        ret = H5Aread(attrCuts, H5T_NATIVE_FLOAT, fcuts);
        H5Aclose(attrCuts);
        if(ret < 0)
        {
          H5Gclose(grpQvTable);
          H5Fclose(root);
          ION_ABORT("ERROR: fail to read HDF5 attribute ThresholdsOfPredictor");
        }

        vector<float> vCuts(size);
        copy(fcuts, fcuts + size, vCuts.begin());

        phred_cuts_.push_back(vCuts);

        delete [] fcuts;
        fcuts = 0;
      }

      hid_t dsQvs = H5Dopen(grpQvTable, "Qvs", H5P_DEFAULT);
      if (dsQvs < 0)
      {
        H5Gclose(grpQvTable);
        H5Fclose(root);
        ION_ABORT("ERROR: fail to open HDF5 dataset Qvs");
      }

      hsize_t tbSize = H5Dget_storage_size(dsQvs);

      phred_table_ = new unsigned char[tbSize];

      ret = H5Dread(dsQvs, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, phred_table_);
      H5Dclose(dsQvs);
      H5Gclose(grpQvTable);
      H5Fclose(root);
      if (ret < 0)
      {
        delete [] phred_table_;
        phred_table_ = 0;

        ION_ABORT("ERROR: fail to read HDF5 dataset Qvs");
      }
    }
    else
    {
      printf("WARNING: binary phred table file %s is not a HDF5 file, try binary file mode.\n", phred_table_file.c_str());
      ifstream source;
      source.open(phred_table_file.c_str(), ios::in|ios::binary|ios::ate);
      if (!source.is_open())
        ION_ABORT("ERROR: Cannot open file: " + phred_table_file);

      long totalSize = source.tellg();
      char* tbBlock = new char [totalSize];

      source.seekg (0, ios::beg);
      source.read (tbBlock, totalSize);
      source.close();

      long headerSize = 0;
      char* ptr = tbBlock;
      int numPredictors = ptr[0]; //kNumPredictors
      if(numPredictors != kNumPredictors)
      {
        delete [] tbBlock;
        tbBlock = 0;
        ION_ABORT("ERROR: Wrong number of predictors load from " + phred_table_file);
      }

      ptr += 4;
      headerSize += 4;

      for(int i = 0; i < kNumPredictors; ++i)
      {
        vNumCuts[i] = ptr[0];
        ptr += 4;
        headerSize += 4;

        offsets_.push_back(1);
      }

      long tbSize = 1;
      for(int i = 0; i < kNumPredictors; ++i)
      {
        vector<float> vCuts;
        tbSize *= vNumCuts[i];
        for(size_t j = 0; j < vNumCuts[i]; ++j)
        {
          float tmp;
          memcpy(&tmp, ptr, 4);
          vCuts.push_back(tmp);
          ptr += 4;
          headerSize += 4;
        }

        phred_cuts_.push_back(vCuts);
      }

      if(tbSize != (totalSize - headerSize))
      {
        delete [] tbBlock;
        tbBlock = 0;
        ION_ABORT("ERROR: Wrong QV table size");
      }

      phred_table_ = new unsigned char[tbSize];
      memcpy(phred_table_, ptr, tbSize * sizeof(unsigned char));

      delete [] tbBlock;
      tbBlock = 0;
    }

    for(size_t i = kNumPredictors - 2; i > 0; --i)
    {
      offsets_[i] *= phred_cuts_[i + 1].size();
      offsets_[i - 1] = offsets_[i];
    }
    offsets_[0] *= phred_cuts_[1].size();
  }
  else
  {
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

    source.close();

    for (int k = 0; k < kNumPredictors; ++k)
      phred_thresholds_max_[k] = *max_element(phred_thresholds_[k].begin(), phred_thresholds_[k].end());
  }

  // Prepare for predictor dump here

  if (save_predictors_) {
    string predictors_filename = output_directory + "/Predictors.txt";
    cout << endl << "Saving PerBaseQual predictors to file " << predictors_filename << endl << endl;
    predictor_dump_.open(predictors_filename.c_str());
    if (!predictor_dump_.is_open())
      ION_ABORT("ERROR: Cannot open file: " + predictors_filename);
  }
}

inline size_t GetIndex(const float predVal, const vector<float>& thresholds)
{
  if(predVal >= thresholds.back())
  {
    return (thresholds.size() - 1);
  }

  size_t l = 0;
  size_t r = thresholds.size() - 1;
  size_t m = l;
  while(r > l)
  {
    m = (r + l) >> 1;
    if(m == l)
    {
      if(predVal <= thresholds[l])
      {
        return l;
      }
      else
      {
        return r;
      }
    }

    if(predVal == thresholds[m])
    {
      return m;
    }
    else if(predVal > thresholds[m])
    {
      l = m;
    }
    else
    {
      r = m;
    }
  }

  return r;
}

uint8_t PerBaseQual::CalculatePerBaseScore(float* pred) const
{
  if(phred_table_)
  {
    size_t index = 0;
    vector<size_t> vind;
    for(int i = 0; i < kNumPredictors; ++i)
    {
      size_t indi = GetIndex(pred[i], phred_cuts_[i]);
      vind.push_back(indi);
      index += (indi * offsets_[i]);
    }

    return phred_table_[index];
  }
  else
  {
    int num_phred_cuts = phred_quality_.size(); // number of rows/lines in the table

    for (int k = 0; k < kNumPredictors; k++)
      pred[k] = min(pred[k], phred_thresholds_max_[k]);

    for ( int j = 0; j < num_phred_cuts; ++j )
    {
      bool valid_cut = true;

      for ( int k = 0; k < kNumPredictors; ++k )
      {
        if (pred[k] > phred_thresholds_[k][j])
        {
          valid_cut = false;
          break;
        }
      }

      if (valid_cut)
        return phred_quality_[j];
    }

    return kMinQuality; //minimal quality score
  }
}

// Predictor 2 - Local noise/flowalign - Maximum residual within +-1 BASE

void PerBaseQual::PredictorLocalNoise(vector<float>& local_noise, int max_base, const vector<int>& base_to_flow,
                                      const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_)
{
  int num_bases = base_to_flow.size();
  for (int base = 0; base < max_base; ++base) {
    int val1 = max(base - 1, 0);
    //int val2 = min(base + 1, num_bases - 1);
    int val2 = flow_predictors_ ? min(base+1, max_base-1) : min(base+1, num_bases-1);
    float noise = 0;
    for (int j = val1; j <= val2; ++j) {
		int jj = flow_predictors_ ? j : base_to_flow[j];
		noise = max(noise, fabsf(normalized_measurements[jj] - prediction[jj]));
    }
    local_noise[base] = noise;
  }
}


// Predictor 3  - Read Noise/Overlap - mean & stdev of the 0-mers & 1-mers in the read
// -(m_1 - m_0 - s_1 - s_0)/m_1

void PerBaseQual::PredictorNoiseOverlap(vector<float>& minus_noise_overlap, int max_base,
                                        const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_)
{
  // 0-mer and 1-mer overlap
  // define 0-mer and 1-mer interval
  float cutoff0 = 0.5;
  float cutoff1 = 1.5;
  int max_iter = 2; // adjust cutoffs once
  int num_flows_to_use = min((int)prediction.size(), 60);
  float noise_overlap;

  for (int i = 0; i < max_iter; i++) {
    int one_counter = 0;
    int zero_counter = 0;
    float mean_zero = 0.0;
    float mean_one = 0.0;
    float stdev_zero = 0.0;
    float stdev_one = 0.0;

    for (int flow = 8; flow < num_flows_to_use; ++flow) {
      float current_flow_val = fabsf(normalized_measurements[flow] - prediction[flow]);
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
      float current_flow_val = fabsf(normalized_measurements[flow] - prediction[flow]);

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
}


// Predictor 4 - Transformed homopolymer length
void PerBaseQual::PredictorHomopolymerRank(vector<float>& homopolymer_rank, int max_base, const vector<char>& sequence, vector<float>& homopolymer_rank_flow, const vector<int>& flow_to_base, int flow_predictors_)
{
    int hp_length = 0;
    for (int base = 0; base < max_base; ++base) {
    hp_length++;
    // HP 1114
    homopolymer_rank[base] = 1;
    if (sequence[base] != sequence[base+1] or (base+2) == max_base) {
      homopolymer_rank[base] = hp_length;
      hp_length = 0;
      }
    }
    int nFlows = flow_to_base.size();
    for (int flow=0; flow<nFlows; flow++) {
        int base = flow_to_base[flow];
        homopolymer_rank_flow[flow] = (base>=0 && base<max_base) ? homopolymer_rank[base] : 0;
    }
}


// Predictor 6 - Neighborhood noise - mean of residual within +-5 BASES

void PerBaseQual::PredictorNeighborhoodNoise(vector<float>& neighborhood_noise, int max_base, const vector<int>& base_to_flow,
                                             const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_)
{
  int num_bases = base_to_flow.size();
  for (int base = 0; base < max_base; ++base) {
    int radius = 5;
    // protect at start/end of read
    int val1 = max(base-radius, 0);
    int val2 = flow_predictors_ ? min(base+radius, max_base-1) : min(base+radius, num_bases-1);

    float noise = 0;
    int count = 0;
    for (int j = val1; j <= val2; j++) {
	  int jj = flow_predictors_ ? j : base_to_flow[j];
      noise += fabsf(normalized_measurements[jj] - prediction[jj]);
      count++;
    }
    if (count)
      noise /= count;
    neighborhood_noise[base] = noise;
  }
}



// Candidate predictor based on Beverly filter

void PerBaseQual::PredictorBeverlyEvents(vector<float>& beverly_events, int max_base, const vector<int>& base_to_flow,
                                         const vector<float>& scaled_residual, const bool flow_predictors_)
{
  const static int flow_window_radius = 10;

  int window_start_flow = 0;
  int window_start_base = 0;
  int window_end_flow = 0;
  int window_end_base = 0;
  int num_beverly_events = 0;

  for (int base = 0; base < max_base; ++base) {
    int window_center_flow = flow_predictors_ ? base : base_to_flow[base];
    // Advance window start
    while (window_start_flow < window_center_flow+flow_window_radius) {
      int hp_length = 0;
      while (window_start_base < (int)base_to_flow.size() and base_to_flow[window_start_base] == window_start_flow) {
        window_start_base++;
        hp_length++;
      }
      if (hp_length == 1 and (scaled_residual[window_start_flow] <= -0.405f or scaled_residual[window_start_flow] >= 0.395f))
        num_beverly_events++;
      if (hp_length == 2 and (scaled_residual[window_start_flow] <= -0.405f or scaled_residual[window_start_flow] >= 0.395f))
        num_beverly_events++;
      window_start_flow++;
    }

    // Advance window end
    while (window_end_flow < window_center_flow-flow_window_radius) {
      int hp_length = 0;
      while (window_end_base < (int)base_to_flow.size() and base_to_flow[window_end_base] == window_end_flow) {
        window_end_base++;
        hp_length++;
      }
      if (hp_length == 1 and (scaled_residual[window_end_flow] <= -0.405f or scaled_residual[window_end_flow] >= 0.395f))
        num_beverly_events--;
      if (hp_length == 2 and (scaled_residual[window_end_flow] <= -0.405f or scaled_residual[window_end_flow] >= 0.395f))
        num_beverly_events--;
      window_end_flow++;
    }

    beverly_events[base] = num_beverly_events;
  }
}




void PerBaseQual::GenerateBaseQualities(const string& read_name, int num_bases, int num_flows,
                                        const vector<float> &predictor1, const vector<float> &predictor2, const vector<float> &predictor3,
                                        const vector<float> &predictor4, const vector<float> &predictor5, const vector<float> &predictor6,
                                        const vector<int>& base_to_flow, vector<uint8_t> &quality,
                                        const vector<float> &candidate1, const vector<float> &candidate2, const vector<float> &candidate3,
                                        const vector<float> &predictor1_flow, const vector<float> &predictor5_flow, const vector<float> &predictor4_flow,
                                        const vector<int>& flow_to_base, const bool flow_predictors_)
{

  if (num_bases == 0)
    return;

  //! \todo This is a temporary fix for very long sequences that are sometimes generated by the basecaller
  int last_base_to_flow = base_to_flow.back();
  int max_eligible_flow = (int)(0.75*num_flows) + 1;
  max_eligible_flow = min(max_eligible_flow,last_base_to_flow);
  //save_predictors_ = false; // debugging only
  int max_eligible_base = flow_predictors_ ? max_eligible_flow : min(num_bases, max_eligible_flow);
  //int max_eligible_base = min(num_bases, max_eligible_flow); // avoid out of range in debugging

  quality.clear();
  stringstream predictor_dump_block;

  for (int base = 0; base < max_eligible_base; base++) { // first 4 bases are the keys TCAG
    float pred[kNumPredictors];
    pred[1] = predictor2[base]; // P2: local noise
    pred[2] = predictor3[base]; // P3: high-residual events
    int base_or_flow = flow_predictors_ ? base : base_to_flow[base];
    /*
    if (save_predictors_) {
      // the following lines are only for predictor_dump_block
      // they are not the same in new QvTables
      pred[0] = flow_predictors_ ? predictor1_flow[base] : predictor1[base]; // P1: penalty residual
      pred[3] = flow_predictors_ ? predictor4_flow[base] : predictor4[base]; // P4: hp
      pred[4] = flow_predictors_ ? predictor5_flow[base] : predictor5[base]; // P5: penalty mismatch
      pred[5] = predictor6[base]; // P6: neighborhood noise

      predictor_dump_block << read_name << " " << base << " ";
      for (int k = 0; k < kNumPredictors; ++k)
        predictor_dump_block << pred[k] << " ";
      predictor_dump_block << candidate1[base_or_flow] << " ";
      predictor_dump_block << candidate2[base_or_flow] << " ";
      predictor_dump_block << candidate3[base_or_flow] << " ";
      if (flow_predictors_) {
          int always_base = flow_predictors_ ? flow_to_base[base] : base;
          predictor_dump_block << always_base << endl; // could be -1
      } else {
          int always_flow = flow_predictors_ ? base : base_to_flow[base];
          predictor_dump_block << always_flow << endl; // cannot get flow if base=-1
      }
      //predictor_dump_block << base_to_flow[base] << endl;
    }
    */
    // v3.4: p1,2,3,4,6,9
    // the real predictors used in the QvTable
    pred[0] = transform_P1(predictor1[base]);
    pred[3] = predictor4[base]; // P4: hp
    //pred[3] = flow_predictors_ ? predictor4_flow[base] : predictor4[base]; // P4: hp
    pred[4] = transform_P6(predictor6[base]);
    //pred[1] = transform_P2(predictor2[base]); // no transformation might help only if no Recalibration
    //pred[5] = transform_P8(candidate2[base_to_flow[base]]);
    pred[5] = candidate3[base_or_flow];
    pred[5] = transform_P9(pred[5]);

    // v3.0: p1,2,3,4,5,6
    //pred[0] = predictor1[base];
    //pred[0] = transform_P1(predictor1[base]);
    //pred[4] = predictor5[base];
    //pred[5] = predictor6[base];
    quality.push_back(CalculatePerBaseScore(pred));
  }

  for (int base = max_eligible_base; base < num_bases; base++)
    quality.push_back(kMinQuality);

  /*
  if (save_predictors_) {
    predictor_dump_block.flush();
    pthread_mutex_lock(&predictor_mutex_);
    predictor_dump_ << predictor_dump_block.str();
    predictor_dump_.flush();
    pthread_mutex_unlock(&predictor_mutex_);
  }
  */
}


void PerBaseQual::DumpPredictors(const string& read_name, int num_bases, int num_flows,
                                        const vector<float> &predictor1, const vector<float> &predictor2, const vector<float> &predictor3,
                                        const vector<float> &predictor4, const vector<float> &predictor5, const vector<float> &predictor6,
                                        const vector<int>& base_to_flow, vector<uint8_t> &quality,
                                        const vector<float> &candidate1, const vector<float> &candidate2, const vector<float> &candidate3,
                                        const vector<float> &predictor1_flow, const vector<float> &predictor5_flow, const vector<float> &predictor4_flow,
                                        const vector<int>& flow_to_base, const bool flow_predictors_)
{

  if (num_bases == 0)
    return;

  //! \todo This is a temporary fix for very long sequences that are sometimes generated by the basecaller
  int last_base_to_flow = base_to_flow.back();
  int max_eligible_flow = (int)(0.75*num_flows) + 1;
  max_eligible_flow = min(max_eligible_flow,last_base_to_flow);
  //save_predictors_ = false; // debugging only
  int max_eligible_base = flow_predictors_ ? max_eligible_flow : min(num_bases, max_eligible_flow);
  //int max_eligible_base = min(num_bases, max_eligible_flow); // avoid out of range in debugging

  stringstream predictor_dump_block;

  for (int base = 0; base < max_eligible_base; base++) { // first 4 bases are the keys TCAG
    float pred[kNumPredictors];
    pred[1] = predictor2[base]; // P2: local noise
    pred[2] = predictor3[base]; // P3: high-residual events
    int always_flow = flow_predictors_ ? base : base_to_flow[base];
    //if (save_predictors_) {
      // the following lines are only for predictor_dump_block
      // they are not the same in new QvTables
      pred[0] = flow_predictors_ ? predictor1_flow[base] : predictor1[base]; // P1: penalty residual
      pred[3] = flow_predictors_ ? predictor4_flow[base] : predictor4[base]; // P4: hp
      pred[4] = flow_predictors_ ? predictor5_flow[base] : predictor5[base]; // P5: penalty mismatch
      pred[5] = predictor6[base]; // P6: neighborhood noise

      predictor_dump_block << read_name << " " << base << " ";
      for (int k = 0; k < kNumPredictors; ++k)
        predictor_dump_block << pred[k] << " ";
      predictor_dump_block << candidate1[always_flow] << " ";
      predictor_dump_block << candidate2[always_flow] << " ";
      predictor_dump_block << candidate3[always_flow] << " ";
      if (flow_predictors_) {
          int always_base = flow_predictors_ ? flow_to_base[base] : base;
          predictor_dump_block << always_base << endl; // could be -1
      } else {
          predictor_dump_block << always_flow << endl; // cannot get flow if base=-1
      }
      //predictor_dump_block << base_to_flow[base] << endl;
    //}
  }

  //if (save_predictors_) {
    predictor_dump_block.flush();
    pthread_mutex_lock(&predictor_mutex_);
    predictor_dump_ << predictor_dump_block.str();
    predictor_dump_.flush();
    pthread_mutex_unlock(&predictor_mutex_);
  //}
}


float PerBaseQual::transform_P1(float p)
{
  float peak = 0.01;
  if (p < peak)
    p = peak + fabs(p-peak);
  return p;
}


float PerBaseQual::transform_P2(float p) // don't use it for new predictors
{
  /*
    float peak = 0.05;
    if (p < peak)
        p = peak + fabs(p-peak);
    peak = 0.15;
    if (p < peak)
        p = peak - fabs(p-peak) * 0.5;
    peak = 0.4;
    if (p > peak)
        p = peak - (p-peak) * 0.2;
    */
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


float PerBaseQual::transform_P5_v34(float p)
{
  float peak = -0.7;
  if (p < peak)
  {
    float peak2 = -0.985;
    if (p < peak2)
      p = peak + fabs(p-peak) + fabs(p-peak2)*6;
    else
      p = peak + fabs(p-peak);
  }

  peak = 0.1;
  if (p > peak)
    p = peak + (p-peak)*0.1;
  peak = -0.125;
  if (p > peak)
    p = peak - (p-peak);
  // make it linear
  peak = -0.4;
  if (p < peak)
    p = peak - (peak-p)*0.5;

  return p;
}



float PerBaseQual::transform_P6(float p)
{
  float peak = 0.06;
  if (p < peak)
    p = peak + fabs(p-peak);
  // v3.2 only, removed after v3.4 with thew new predictors
  // it came back again??
  peak = 0.4;
  if (p > peak)
    p = peak + fabs(p-peak)/3;

  return p;
}


float PerBaseQual::transform_P7(float p)
{
  float peak = -0.005; // -0.05 before v3.4
  if (p < peak)
    //p = peak + fabs(p-peak) * 0.5;
    p = peak + fabs(p-peak); // came back again

  return p;
}


float PerBaseQual::transform_P8(float p)
{
  float peak = 1.1;
  if (p > peak)
    p = peak - fabs(p-peak);
  p = -p;

  peak = 0.9;
  if (p > peak)
    p = peak + (p-peak)*0.25;

  return p;
}


float PerBaseQual::transform_P9(float p)
{
  p = -p;
  //float peak = -0.95; // -0.9 before v3.4
  float peak = -0.9; // came back again with newer predictors??
  if (p < peak)
    p = peak + abs(p - peak);
  //p = peak + abs(p - peak)*2; // before v3.4

  return p;
}


bool PerBaseQual::hasBinaryExtension(string &filename)
{
  vector<string> extensions;
  extensions.push_back(".h5");
  extensions.push_back(".binary");
  size_t len1 = filename.length();
  for (size_t n=0; n<extensions.size(); n++)
  {
    string extension = extensions[n];
    size_t len2 = extension.length();
    if (len1>=len2) {
      if (filename.substr(len1-len2).compare(extension) == 0) return true;
    }
  }
  return false;
}


char * PerBaseQual::get_KnownAlternate_PhredTable(string chip_type, bool recalib, string enzymeName, bool binTable)
{
    chip_type = get_KnownAlternate_chiptype(chip_type);
    string phred_table_file = "phredTable." + chip_type;
    if (enzymeName.length()>0)
        phred_table_file += "." + enzymeName;
    if (recalib)
      phred_table_file += ".Recal";
    if (binTable)
        phred_table_file += ".h5";
    char* full_filename = GetIonConfigFile(phred_table_file.c_str());

    if (full_filename==NULL && enzymeName.length()>0) // use the default table (no enzymeName) for the chip_type
    {
        phred_table_file = "phredTable." + chip_type;
        if (recalib)
          phred_table_file += ".Recal";
        if (binTable)
            phred_table_file += ".h5";
        full_filename = GetIonConfigFile(phred_table_file.c_str());
    }
    return (full_filename);
}


bool PerBaseQual::startswith(std::string const &fullString, std::string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        int testlen = teststring.length();
        return (0 == fullString.compare (0, testlen, teststring));
    }
    else
        return false;
}

bool PerBaseQual::endswith(std::string const &fullString, std::string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        int testlen = teststring.length();
        return (0 == fullString.compare (fullString.length() - testlen, testlen, teststring));
    }
    else
        return false;
}

bool PerBaseQual::contains(std::string const &fullString, std::string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        std::size_t found = fullString.find(teststring);
        return (found!=std::string::npos);
    }
    else
        return false;
}

char * PerBaseQual::get_phred_table_name(string chip_type, bool recalib, string enzymeName)
{
    string phred_table_file = "phredTable." + chip_type + ".h5";
    if (enzymeName.length()>0)
        phred_table_file = "phredTable." + chip_type + "." + enzymeName + ".h5";

    if (recalib)
    {
      phred_table_file = "phredTable." + chip_type + ".Recal.h5";
      if (enzymeName.length()>0)
          phred_table_file = "phredTable." + chip_type + "." + enzymeName + ".Recal.h5";
    }

    char* full_filename = GetIonConfigFile(phred_table_file.c_str());
    if (full_filename==NULL && enzymeName.length()>0) // use the default table (no enzymeName) for the chip_type
    {
        phred_table_file = "phredTable." + chip_type + ".h5";
        if (recalib)
            phred_table_file = "phredTable." + chip_type + ".Recal.h5";
        full_filename = GetIonConfigFile(phred_table_file.c_str());
    }

    // check to see if the file exist
    if (!full_filename)
    {
        full_filename = get_KnownAlternate_PhredTable(chip_type, recalib, enzymeName);
    }

    // use default table if still does not exist

   return (full_filename);
}

string PerBaseQual::add_Recal_to_phredTableName(string phred_table_file, bool recalib)
{
    // to handle the phred_table_file passed by the pipeline that does not contain ".Recal"
    if (recalib && (!contains(phred_table_file,".Recal")) && endswith(phred_table_file,".h5"))
    {
        phred_table_file = phred_table_file.substr(0,phred_table_file.length()-2) + "Recal.h5";
    }
    return (phred_table_file);
}
