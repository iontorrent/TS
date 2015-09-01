/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     SampleManager.h
//! @ingroup  VariantCaller
//! @brief    Manages sample and read group names


#ifndef SAMPLEMANAGER_H
#define SAMPLEMANAGER_H

#include <map>
#include <vector>
#include <string>
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

using namespace std;
using namespace BamTools;


class SampleManager {
public:
  SampleManager() : num_samples_(0), primary_sample_(0) {}
  ~SampleManager() {}

  void Initialize (const SamHeader& bam_header, string& sample_name, const string& force_sample_name);
  bool IdentifySample(const BamAlignment& alignment, int& sample_index, bool& primary_sample) const;

  int                 num_samples_;
  vector<string>      sample_names_;
  map<string,int>     read_group_to_sample_idx_;
  int                 primary_sample_;
};



#endif //SAMPLEMANAGER_H


