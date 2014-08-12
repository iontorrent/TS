/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef IONSTATS_DATA_H
#define IONSTATS_DATA_H

#include "ionstats.h"
#include "hdf5.h"
#include <sstream>

#define ERROR_DATA_N_ROWS 5

using namespace std;

hid_t H5CreateOrOpenGroup(hid_t &file_id, string &group_name);

class ErrorData {
public:

  ErrorData() : region_origin_(0,0), region_dim_(0,0) {}

  void Initialize(unsigned int histogram_length);
  void Initialize(unsigned int histogram_length, vector<unsigned int> &region_origin, vector<unsigned int> &region_dim);
  void Initialize(vector<unsigned int> &region_origin, vector<unsigned int> &region_dim, vector<unsigned int> &error_data_dim, vector<uint64_t> &error_data);

  void Add(ReadAlignmentErrors &e);
  void MergeFrom(ErrorData &other);
  void MergeFrom(Json::Value& json_value, bool &found);
  bool HasData(void) { return( (ins_.Size() > 0) || (del_.Size() > 0) || (sub_.Size() > 0) || (align_start_.Size() > 0) || (align_stop_.Size() > 0) ); };

  // Functions for reading & writing HDF5
  void writeH5(hid_t &file_id, string group_name);
  int readH5(hid_t group_id);

  void SaveToJson(Json::Value& json_value);

  unsigned int readCount(void) {
    unsigned int c=0;
    for(unsigned int i=0; i<align_start_.Size(); ++i)
      c += align_start_.Count(i);
    return(c);
  }

  uint64_t Size(void) {
    uint64_t size = 4*sizeof(unsigned int);
    size += (ins_.Size() + del_.Size() + sub_.Size() + align_start_.Size() + align_stop_.Size())*sizeof(uint64_t);
    return(size);
  }

  void clear(void) {
    ins_.clear();
    del_.clear();
    sub_.clear();
    align_start_.clear();
    align_stop_.clear();
  }

private:

  pair <unsigned int, unsigned int> region_origin_;
  pair <unsigned int, unsigned int> region_dim_;
  SimpleHistogram ins_;
  SimpleHistogram del_;
  SimpleHistogram sub_;
  SimpleHistogram align_start_;
  SimpleHistogram align_stop_;

  void LoadErrorDataBuffer(unsigned int n_col, unsigned int n_row, vector<uint64_t> &buf);

};



class HpData {

public:

  HpData() : max_hp_(0), origin_(0,0), dim_(0,0) {}

  void Initialize(unsigned int max_hp, vector<unsigned int> &o, vector<unsigned int> &d);
  void Initialize(unsigned int max_hp);
  void Add(vector<char> &ref_hp_nuc, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err);
  int MergeFrom(HpData &other);

  void writeH5(hid_t &file_id, string group_name);
  int readH5(hid_t group_id);

  unsigned int MaxHp(void) { return max_hp_; }
  map< char, vector< vector<uint64_t> > >::iterator HpCountBegin(void)     { return hp_count_.begin(); }
  map< char, vector< vector<uint64_t> > >::iterator HpCountEnd(void)       { return hp_count_.end();   }
  map< char, vector< vector<uint64_t> > >::iterator HpCountFind(char c)    { return hp_count_.find(c); }
  bool HasData(void) {
    bool has_data=false;
    for(map< char, vector< vector<uint64_t> > >::iterator it=hp_count_.begin(); it != hp_count_.end(); ++it) {
      if(it->second.size() > 0) {
        has_data=true;
        break;
      }
    }
    return(has_data);
  }

  uint64_t Size(void) {
    uint64_t size = 5*sizeof(unsigned int);
    for(map< char, vector< vector<uint64_t> > >::iterator it=hp_count_.begin(); it != hp_count_.end(); ++it)
      for(unsigned int i=0; i < it->second.size(); ++i)
        size += it->second[i].size()*sizeof(uint64_t);
    return(size);
  }

private:

  void LoadHpDataBuffer(unsigned int n_col, unsigned int n_row, vector<uint64_t> &buf, vector< vector<uint64_t> > &hp_table);

  unsigned int max_hp_;
  pair <unsigned int, unsigned int> origin_;
  pair <unsigned int, unsigned int> dim_;
  map< char, vector< vector<uint64_t> > > hp_count_;
};


class RegionalSummary {

public:

  RegionalSummary() : origin_(0,0), dim_(0,0), max_hp_(0), n_flow_(0), n_err_(0), n_aligned_(0) {}

  void Initialize(unsigned int max_hp, unsigned int n_flow, vector<unsigned int> &o, vector<unsigned int> &d);
  void Add(ReadAlignmentErrors &e);
  void Add(vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> &ref_hp_flow);
  int MergeFrom(RegionalSummary &other);

  void writeH5(hid_t &file_id, string group_name);
  int readH5(hid_t group_id);

  uint64_t Size(void) {
    uint64_t size = 6*sizeof(unsigned int);
    size += 2*sizeof(uint64_t);
    for(unsigned int i=0; i < hp_count_.size(); ++i)
      size += hp_count_[i].size()*sizeof(uint64_t);
    for(unsigned int i=0; i < hp_err_.size(); ++i)
      size += hp_err_[i].size()*sizeof(uint64_t);
    return(size);
  }

  unsigned int                       MaxHp()    { return max_hp_; }
  unsigned int                       nFlow()    { return n_flow_; }
  uint64_t                           nErr()     { return n_err_; }
  uint64_t                           nAligned() { return n_aligned_; }
  const vector< vector<uint64_t> > & HpCount()  { return hp_count_; }
  const vector< vector<uint64_t> > & HpErr()    { return hp_err_; }

private:
  int LoadDataBuffer(unsigned int n_col, unsigned int n_row, vector<uint64_t> &buf, const vector< vector<uint64_t> > & data);
  void AssertDims(void);

  pair <unsigned int, unsigned int> origin_;
  pair <unsigned int, unsigned int> dim_;
  unsigned int max_hp_;
  unsigned int n_flow_;
  uint64_t n_err_;
  uint64_t n_aligned_;
  vector< vector<uint64_t> > hp_count_;
  vector< vector<uint64_t> > hp_err_;
};


#endif // IONSTATS_DATA_H
