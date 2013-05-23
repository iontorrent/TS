/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef IONSTATS_H
#define IONSTATS_H

#include <vector>
#include <stdint.h>
#include <math.h>
#include "json/json.h"

using namespace std;


int IonstatsBasecaller(int argc, const char *argv[]);
int IonstatsBasecallerReduce(const string& output_json_filename, const vector<string>& input_jsons);

int IonstatsAlignment(int argc, const char *argv[]);
int IonstatsAlignmentReduce(const string& output_json_filename, const vector<string>& input_jsons);

int IonstatsTestFragments(int argc, const char *argv[]);
int IonstatsTestFragmentsReduce(const string& output_json_filename, const vector<string>& input_jsons);


// ReadLengthHistogram keeps track of:
//  - the actual histogram
//  - number of reads
//  - number of bases
//  - mean read length (derived: num_bases/num_reads)
//  - max read length

class ReadLengthHistogram {

public:
  ReadLengthHistogram() : num_reads_(0), num_bases_(0), max_read_length_(0) {}
  ~ReadLengthHistogram() {}

  void Initialize(int histogram_length) { histogram_.assign(histogram_length,0); }

  void Add(unsigned int read_length) {
    num_reads_++;
    num_bases_ += read_length;
    if (read_length > max_read_length_)
      max_read_length_ = read_length;
    read_length = min(read_length,(unsigned int)histogram_.size()-1);
    histogram_[read_length]++;
  }

  void SummarizeToJson(Json::Value& json_value) {
    json_value["num_bases"] = (Json::UInt64)num_bases_;
    json_value["num_reads"] = (Json::UInt64)num_reads_;
    json_value["max_read_length"] = max_read_length_;
    json_value["mean_read_length"] = (Json::UInt64)(num_reads_ ? (num_bases_ / num_reads_) : 0);
  }

  void SaveToJson(Json::Value& json_value) {
    SummarizeToJson(json_value);
    json_value["read_length_histogram"] = Json::arrayValue;
    for (unsigned int idx = 0; idx < histogram_.size(); ++idx)
      json_value["read_length_histogram"][idx] = (Json::UInt64)histogram_[idx];
  }

  void LoadFromJson(const Json::Value& json_value) {
    num_bases_ = json_value["num_bases"].asInt64();
    num_reads_ = json_value["num_reads"].asInt64();
    max_read_length_ = json_value["max_read_length"].asInt();
    int histogram_length = json_value["read_length_histogram"].size();
    histogram_.assign(histogram_length,0);
    for (unsigned int idx = 0; idx < histogram_.size(); ++idx)
      histogram_[idx] = json_value["read_length_histogram"][idx].asUInt64();
  }

  void MergeFrom(const ReadLengthHistogram& other) {
    num_bases_ += other.num_bases_;
    num_reads_ += other.num_reads_;
    max_read_length_ = max(max_read_length_, other.max_read_length_);
    int histogram_length = max(histogram_.size(), other.histogram_.size());
    histogram_.resize(histogram_length,0);
    for (unsigned int idx = 0; idx < other.histogram_.size(); ++idx)
      histogram_[idx] += other.histogram_[idx];
  }

  uint64_t num_reads() const { return num_reads_; }

private:
  vector<uint64_t>  histogram_;
  uint64_t          num_reads_;
  uint64_t          num_bases_;
  unsigned int      max_read_length_;

};


// MetricGeneratorSNR


class MetricGeneratorSNR {
public:
  MetricGeneratorSNR() {
    for (int idx = 0; idx < 8; idx++) {
      zeromer_first_moment_[idx] = 0;
      zeromer_second_moment_[idx] = 0;
      onemer_first_moment_[idx] = 0;
      onemer_second_moment_[idx] = 0;
    }
    count_ = 0;
  }

  void Add(const vector<uint16_t>& flow_signal, const char *key, const string& flow_order)
  {
    if (flow_signal.size() < 8)
      return;
    for (int flow = 0; flow < 8; ++flow) {
      char nuc = flow_order[flow % flow_order.length()];
      if (*key == nuc) { // Onemer
        onemer_first_moment_ [nuc&7] += flow_signal[flow];
        onemer_second_moment_[nuc&7] += flow_signal[flow] * flow_signal[flow];
        key++;
      } else {  // Zeromer
        zeromer_first_moment_ [nuc&7] += flow_signal[flow];
        zeromer_second_moment_[nuc&7] += flow_signal[flow] * flow_signal[flow];
      }
    }
    count_++;
  }

  void Add(const vector<int16_t>& flow_signal, const char *key, const string& flow_order)
  {
    if (flow_signal.size() < 8)
      return;
    for (int flow = 0; flow < 8; ++flow) {
      char nuc = flow_order[flow % flow_order.length()];
      if (*key == nuc) { // Onemer
        onemer_first_moment_ [nuc&7] += flow_signal[flow];
        onemer_second_moment_[nuc&7] += flow_signal[flow] * flow_signal[flow];
        key++;
      } else {  // Zeromer
        zeromer_first_moment_ [nuc&7] += flow_signal[flow];
        zeromer_second_moment_[nuc&7] += flow_signal[flow] * flow_signal[flow];
      }
    }
    count_++;
  }

  void LoadFromJson(const Json::Value& json_value) {
    count_ = json_value["full"]["num_reads"].asInt();
    double system_snr = json_value["system_snr"].asDouble();
    double variance = (1.0/system_snr) * (1.0/system_snr);
    if (system_snr == 0)
      variance = 0;

    for (int idx = 0; idx < 8; ++idx) {
      zeromer_first_moment_ [idx] = 0;
      onemer_first_moment_  [idx] = count_;
      zeromer_second_moment_[idx] = variance * count_;
      onemer_second_moment_ [idx] = (variance + 1) * count_;
    }
  }

  void MergeFrom(const MetricGeneratorSNR& other) {
    count_ += other.count_;
    for (int idx = 0; idx < 8; ++idx) {
      zeromer_first_moment_ [idx] += other.zeromer_first_moment_ [idx];
      onemer_first_moment_  [idx] += other.onemer_first_moment_  [idx];
      zeromer_second_moment_[idx] += other.zeromer_second_moment_[idx];
      onemer_second_moment_ [idx] += other.onemer_second_moment_ [idx];
    }
  }

  void SaveToJson(Json::Value& json_value) {
    double nuc_snr[8];
    for(int idx = 0; idx < 8; idx++) { // only care about the first 3, G maybe 2-mer etc
      double zeromer_mean = zeromer_first_moment_[idx] / count_;
      double zeromer_var = zeromer_second_moment_[idx] / count_ - zeromer_mean * zeromer_mean;
      double onemer_mean = onemer_first_moment_[idx] / count_;
      double onemer_var = onemer_second_moment_[idx] / count_ - onemer_mean * onemer_mean;
      double average_stdev = (sqrt(zeromer_var) + sqrt(onemer_var)) / 2.0;
      nuc_snr[idx] = 0;
      if (average_stdev > 0.0)
        nuc_snr[idx] = (onemer_mean - zeromer_mean) / average_stdev;
    }
    json_value["system_snr"] = (nuc_snr['A'&7] + nuc_snr['C'&7] + nuc_snr['T'&7]) / 3.0;
  }

private:
  double      zeromer_first_moment_[8];
  double      zeromer_second_moment_[8];
  double      onemer_first_moment_[8];
  double      onemer_second_moment_[8];
  int         count_;
};


class BaseQVHistogram {
public:
  BaseQVHistogram() : histogram_(64,0) {}

  void Add (const string& fastq_qvs) {
    for (unsigned int idx = 0; idx < fastq_qvs.length(); ++idx)
      histogram_[max(min(fastq_qvs[idx]-33,63),0)]++;
  }

  void MergeFrom(const BaseQVHistogram& other) {
    for (int qv = 0; qv < 64; ++qv)
      histogram_[qv] += other.histogram_[qv];
  }

  void SaveToJson(Json::Value& json_value) {
    json_value["qv_histogram"] = Json::arrayValue;
    for (int qv = 0; qv < 64; ++qv)
      json_value["qv_histogram"][qv] = (Json::Int64)histogram_[qv];
  }

  void LoadFromJson(const Json::Value& json_value) {
    for (int qv = 0; qv < 64; ++qv)
      histogram_[qv] = json_value["qv_histogram"][qv].asInt64();
  }

private:
  vector<uint64_t> histogram_;
};


class SimpleHistogram {
public:

  void Initialize(int histogram_length) { histogram_.assign(histogram_length,0); }

  void Add (unsigned int value) {
    value = min(value,(unsigned int)histogram_.size()-1);
    histogram_[value]++;
  }

  void MergeFrom(const SimpleHistogram& other) {
    int histogram_length = max(histogram_.size(), other.histogram_.size());
    histogram_.resize(histogram_length,0);
    for (unsigned int idx = 0; idx < other.histogram_.size(); ++idx)
      histogram_[idx] += other.histogram_[idx];
  }

  void SaveToJson(Json::Value& json_value) {
    json_value = Json::arrayValue;
    for (unsigned int idx = 0; idx < histogram_.size(); ++idx)
      json_value[idx] = (Json::Int64)histogram_[idx];
  }

  void LoadFromJson(const Json::Value& json_value) {
    int histogram_length = json_value.size();
    histogram_.assign(histogram_length,0);
    for (unsigned int idx = 0; idx < histogram_.size(); ++idx)
      histogram_[idx] = json_value[idx].asUInt64();
  }

private:
  vector<uint64_t> histogram_;
};



class MetricGeneratorHPAccuracy {
public:
  MetricGeneratorHPAccuracy() {
    for(int hp = 0; hp < 8; hp++) {
      hp_count_[hp] = 0;
      hp_accuracy_[hp] = 0;
    }
  }

  void Add(int ref_hp, int called_hp) {
    if (ref_hp >= 8)
      return;
    hp_count_[ref_hp]++;
    if (ref_hp == called_hp)
      hp_accuracy_[ref_hp]++;
  }

  void LoadFromJson(const Json::Value& json_value) {
    for(int hp = 0; hp < 8; hp++) {
      hp_accuracy_[hp] = json_value["hp_accuracy_numerator"][hp].asUInt64();
      hp_count_[hp] = json_value["hp_accuracy_denominator"][hp].asInt64();
    }
  }

  void SaveToJson(Json::Value& json_value) {
    for(int hp = 0; hp < 8; hp++) {
      json_value["hp_accuracy_numerator"][hp] = (Json::UInt64)hp_accuracy_[hp];
      json_value["hp_accuracy_denominator"][hp] = (Json::UInt64)hp_count_[hp];
    }
  }

  void MergeFrom(const MetricGeneratorHPAccuracy& other) {
    for(int hp = 0; hp < 8; hp++) {
      hp_count_[hp] += other.hp_count_[hp];
      hp_accuracy_[hp] += other.hp_accuracy_[hp];
    }
  }

private:
  uint64_t hp_accuracy_[8];
  uint64_t hp_count_[8];
};


#endif // IONSTATS_H
