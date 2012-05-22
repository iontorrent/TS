/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef BASECALLERUTILS_H
#define BASECALLERUTILS_H

#include <string>
#include <algorithm>
#include "Utils.h"
/*
class FlowOrder {
public:
  FlowOrder(const string& flow_string)
  {

  }

  ~FlowOrder() {}

  char operator[](int pos) const { return flow_string_[pos % flow_string_length_]; }

  int operator()(int pos) const { return NucToInt(flow_string_[pos % flow_string_length_]); }

  const char *c_str() const { return flow_string_.c_str(); }

  void BasesToFlows (const string& basespace, vector<int> &flowspace) const;
  void BasesToFlows (const string& basespace, vector<int> &flowspace, int num_flows) const;

  //void FlowsToBases
  //

private:
  int NucToInt (char nuc) const {  // static?
    if (nuc=='a' or nuc=='A') return 0;
    if (nuc=='c' or nuc=='C') return 1;
    if (nuc=='g' or nuc=='G') return 2;
    if (nuc=='t' or nuc=='T') return 3;
    return -1;
  }

  std::string flow_string_;
  int         flow_string_length_;
//  int         num_flows_total_;
};
*/

#define MAX_KEY_FLOWS     64

class KeySequence {
public:
  KeySequence() : bases_length_(0), flows_length_(0) { flows_[0]=0; }
  KeySequence(const std::string& flow_order, const std::string& key_string, const std::string& key_name) {
    Set(flow_order, key_string, key_name);
  }

  void Set(const std::string& flow_order, const std::string& key_string, const std::string& key_name) {
    name_ = key_name;
    bases_ = key_string;
    transform(bases_.begin(), bases_.end(), bases_.begin(), ::toupper);
    bases_length_ = key_string.length();
    flows_length_ = seqToFlow(bases_.c_str(), bases_length_, &flows_[0], MAX_KEY_FLOWS,
          (char *)flow_order.c_str(), flow_order.length());
  }

  const std::string &   name() const { return name_; }
  const std::string &   bases() const { return bases_; }
  int                   bases_length() const { return bases_length_; }
  const int *           flows() const { return flows_; }
  int                   flows_length() const { return flows_length_; }
  int                   operator[](int i) const { return flows_[i]; }
  const char *          c_str() const { return bases_.c_str(); }

  // Feature request: keypass a basespace read
  // Feature request: keypass a flowspace read (int or float)

private:
  std::string           name_;
  std::string           bases_;
  int                   bases_length_;
  int                   flows_length_;
  int                   flows_[MAX_KEY_FLOWS];
};




#endif // BASECALLERUTILS_H
