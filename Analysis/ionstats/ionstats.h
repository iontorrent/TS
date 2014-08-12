/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef IONSTATS_H
#define IONSTATS_H

#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include "json/json.h"

#define IONSTATS_BIGGEST_PHRED 47
#define TYPICAL_FLOWS_PER_BASE 2

using namespace std;

int IonstatsBasecaller(int argc, const char *argv[]);
int IonstatsBasecallerReduce(const string& output_json_filename, const vector<string>& input_jsons);
int IonstatsAlignment(int argc, const char *argv[]);
int IonstatsAlignmentReduce(const string& output_json_filename, const vector<string>& input_jsons);
int IonstatsAlignmentReduceH5(const string& output_h5_filename, const vector<string>& input_h5_filename, bool merge_proton_blocks);

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


class ReadAlignmentErrors {

public:
  ReadAlignmentErrors() : have_data_(false), len_(0), first_(0), last_(0) {}
  ~ReadAlignmentErrors() {}

  void Initialize() {have_data_=false; len_=0; first_=0; last_=0; ins_.resize(0); del_.resize(0); del_len_.resize(0); sub_.resize(0); err_.resize(0); err_len_.resize(0);}

  void SetHaveData (bool b=true) { have_data_= b; }
  void SetLen      (const uint16_t i) { len_= i; }
  void SetFirst    (const uint16_t i) { first_= i; }
  void SetLast     (const uint16_t i) { last_= i; }
  void AddIns      (const uint16_t i) { ins_.push_back(i); }
  void AddDel      (const uint16_t i, const uint16_t l=1, bool append=true) {
    if(append) {
      del_.push_back(i);
      del_len_.push_back(l);
    } else {
      del_.insert(del_.begin(),i);
      del_len_.insert(del_len_.begin(),l);
    }
  }
  void AddSub      (const uint16_t i) { sub_.push_back(i); }

  uint64_t AlignedLen(void) const {
    if(len_ > 0)
      return( (first_ < last_) ? (last_ - first_ + 1) : (first_ - last_ + 1) );
    else
      return(0);
  }

  void ConsolidateErrors(void) {
    // Convert deletion list so that multi-base deletions are represented as multiple single-base deletions
    vector<uint16_t> del_single;
    del_single.reserve(floor(del_.size() * 1.3));
    for(unsigned int i=0; i<del_.size(); ++i)
      for(unsigned int j=0; j<del_len_[i]; ++j)
        del_single.push_back(del_[i]);

    // Merge-sort insertions, deletions & substitutions to a list of single-error positions
    vector<uint16_t> indel(ins_.size() + del_single.size());
    merge(ins_.begin(),ins_.end(),del_single.begin(),del_single.end(),indel.begin());
    vector<uint16_t> err_single(indel.size() + sub_.size());
    merge(indel.begin(),indel.end(),sub_.begin(),sub_.end(),err_single.begin());

    // Collapse cases of multiple errors at same position
    err_.resize(0);
    err_.reserve(err_single.size());
    err_len_.resize(0);
    err_len_.reserve(err_single.size());
    uint16_t last_err_pos=0;
    uint16_t err_len=0;
    for(unsigned int i=0; i<err_single.size(); ++i) {
      if( (i > 0) && (err_single[i]!=last_err_pos) ) {
        // Have completed an error, store & reset
        err_.push_back(last_err_pos);
        err_len_.push_back(err_len);
        err_len=0;
      }
      last_err_pos = err_single[i];
      err_len++;
    }
    if(err_len > 0) {
      err_.push_back(last_err_pos);
      err_len_.push_back(err_len);
    }
  }

  void Print (void) {
    cout << "(Length,First,Last) = (" << len_ << ", " << first_ << ", " << last_ << ")\n";
    vector<uint16_t>::iterator it;
    cout << "     Insertions:  ";
    for(it=ins_.begin(); it != ins_.end(); ++it)
      cout << *it << ", ";
    cout << "\n";
    cout << "      Deletions:  ";
    for(unsigned int i=0; i < del_.size(); ++i)
      cout << del_[i] << " (" << del_len_[i] << "), ";
    cout << "\n";
    cout << "  Substitutions:  ";
    for(it=sub_.begin(); it != sub_.end(); ++it)
      cout << *it << ", ";
    cout << "\n";
    cout << "         Errors:  ";
    for(unsigned int i=0; i < err_.size(); ++i)
      cout << err_[i] << " (" << err_len_[i] << "), ";
    cout << "\n";
  }

  void ShiftPositions (int shift) {
    if(shift==0)
      return;
    len_   += shift;
    last_  += shift;
    vector<uint16_t>::iterator it;
    for(it=ins_.begin(); it != ins_.end(); ++it)
      *it += shift;
    for(it=del_.begin(); it != del_.end(); ++it)
      *it += shift;
    for(it=sub_.begin(); it != sub_.end(); ++it)
      *it += shift;
    for(it=err_.begin(); it != err_.end(); ++it)
      *it += shift;
  }

  bool have_data() { return have_data_; }
  uint16_t len()     { return len_; }
  uint16_t first()   { return first_; }
  uint16_t last()    { return last_; }
  const vector<uint16_t> & ins()     { return ins_; }
  const vector<uint16_t> & del()     { return del_; }
  const vector<uint16_t> & del_len() { return del_len_; }
  const vector<uint16_t> & sub()     { return sub_; }
  const vector<uint16_t> & err()     { return err_; }
  const vector<uint16_t> & err_len() { return err_len_; }

private:
  bool              have_data_; // used to indicate whether or not there were data to fill this object
  uint16_t          len_;       // Number of bases or flows
  uint16_t          first_;     // First aligned base or flow
  uint16_t          last_;      // Last aligned base or flow
  vector<uint16_t>  ins_;       // vector of positions where there are insertions
  vector<uint16_t>  del_;       // vector of positions where there are deletions
  vector<uint16_t>  del_len_;   // vector of deletion lengths
  vector<uint16_t>  sub_;       // vector of positions where there are substitutions
  vector<uint16_t>  err_;       // vector of positions where there are errors of any kind
  vector<uint16_t>  err_len_;   // vector of error lengths

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

  void Initialize(vector<uint64_t>::iterator begin, vector<uint64_t>::iterator end) { histogram_.assign(begin,end); }

  uint64_t Count (unsigned int value) { return(histogram_[value]); }
  unsigned int Size (void) { return(histogram_.size()); }

  void Add (unsigned int value, unsigned int count=1) {
    value = min(value,(unsigned int)histogram_.size()-1);
    histogram_[value] += count;
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

  void clear(void) { histogram_.clear(); }

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
