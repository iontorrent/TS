/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     SampleManager.h
//! @ingroup  VariantCaller
//! @brief    Manages sample and read group names


#include "SampleManager.h"

#include "BAMWalkerEngine.h"

// const string& force_sample_name,
// const string& sample_name,

// -----------------------------------------------------------------------------------------
// Function extracts the samples associated with read groups in bam_header
// primary_sample_name [in]: sets a primary sample name (default is first found in header)
//                           only the primary sample will be analyzed if we don't specify multi-sample analysis
// force_sample_name   [in]: ignores all information in the BAM and force analyzes as one sample
// multisample     [in/out]: activates multi-sample analysis for more than one sample in BAM


void SampleManager::Initialize (const SamHeader& bam_header, const string& primary_sample_name, const string& force_sample_name, bool &multisample)
{

  // Iterate through samples in BAM header to extract sample information
  num_samples_ = 0;
  primary_sample_name_ = primary_sample_name;
  
  for (SamReadGroupConstIterator read_group = bam_header.ReadGroups.Begin(); read_group < bam_header.ReadGroups.End(); ++read_group) {

    string sample_name;
    if (force_sample_name.empty()) {
      sample_name = read_group->Sample;
    }
    else {
      sample_name = force_sample_name;
    }

    if (read_group->ID.empty()) {
      cerr << "ERROR: One of BAM read groups is missing ID tag" << endl;
      exit(1);
    }
    if (sample_name.empty()) {
      cerr << "ERROR: BAM read group " << read_group->ID << " is missing SM tag" << endl;
      exit(1);
    }

    bool new_sample = true;
    for (int i = 0; i < num_samples_; ++i) {
      if (sample_name == sample_names_[i]) {
        read_group_to_sample_idx_[read_group->ID] = i;
        new_sample = false;
        break;
      }
    }

    if (new_sample) {
      sample_names_.push_back(sample_name);
      read_group_to_sample_idx_[read_group->ID] = num_samples_;
      num_samples_++;
    }

    map<string, int>::iterator s = read_group_to_sample_idx_.find(read_group->ID);
    if (s != read_group_to_sample_idx_.end()) {
      if (s->second != read_group_to_sample_idx_[read_group->ID]) {
        cerr << "ERROR: multiple samples (SM) map to the same read group (ID)";
        exit(1);
      }
      // if it's the same sample name and RG combo, no worries
      // TODO: what about other tags
    }
    //cout << "SampleManager: Read group " << read_group->ID << " is associated with sample " << sample_name << endl;

  }

  if (num_samples_ == 0) {
    cerr << "ERROR: BAM file(s) do not have any read group definitions" << endl;
    exit(1);
  }
  // Do we have a multi-sample analysis?
  else if (num_samples_ > 1) {
	  if (not multisample and primary_sample_name_.empty()){
        cerr << "ERROR: SampleManager: Multiple Samples (" << num_samples_ << ") found in BAM file/s provided. "<< endl;
        //cerr << "ERROR: But neither a primary sample was provided nor multi-sample analysis enabled." << endl;
        cerr << "ERROR: Please select primary sample name to process using the \"--sample-name\" parameter. " << endl;
        //cerr << "ERROR: AND/OR enable multi-sample analysis using  the \"--multisample-analysis\" parameter. " << endl;
        exit(EXIT_FAILURE);
	  }
  }
  else // We only have one sample
    multisample = false;

  // Search for specified primary sample or select a default primary (first available)

  bool default_sample = false;
  if (primary_sample_name_.empty()) {
    primary_sample_name_ = sample_names_[0];
    default_sample = true;
  }

  bool primary_sample_found = false;
  for (int i = 0; i < num_samples_; ++i) {
    if (sample_names_[i] == primary_sample_name_) {
      primary_sample_ = i;
      primary_sample_found = true;
    }
    // AWalt added this because of IR-19679?
    // TODO investigate why this is actually necessary?
    else {
      string test = primary_sample_name_ + ".";
      if (strncmp(sample_names_[i].c_str(), test.c_str(), test.length()) == 0) {
        primary_sample_ = i;
        primary_sample_found = true;
        primary_sample_name_ = sample_names_[primary_sample_];
      }
    }
  }

  if (!primary_sample_found) {
    cerr << "ERROR: Sample " << primary_sample_name << " provided using \"--sample-name\" option "
         << "is not associated with any read groups in BAM file(s)" << endl;
    exit(EXIT_FAILURE);
  }

  //now find the read group ID associated with this sample name
  int num_primary_read_groups = 0;
  for (map<string, int>::const_iterator p = read_group_to_sample_idx_.begin(); p != read_group_to_sample_idx_.end(); ++p)
    if (primary_sample_ == p->second)
      num_primary_read_groups++;

  if (multisample)
    cout << "SampleManager: Multi-sample analysis enabled." << endl;
  if (!force_sample_name.empty())
    cout << "SampleManager: All read groups forced to assume sample name " <<  force_sample_name << endl;

  cout << "SampleManager: Found " << read_group_to_sample_idx_.size() << " read group(s) and " << num_samples_ << " sample(s)." << endl;
  if (default_sample)
    cout << "SampleManager: Primary sample \"" << primary_sample_name_ << "\" (default) present in " << num_primary_read_groups << " read group(s)" << endl;
  else
    cout << "SampleManager: Primary sample \"" << primary_sample_name_ << "\" (set via -g) " << num_primary_read_groups << " read group(s)" << endl;

}

// --------------------------------------------------------------------
// This function populates sample_index and primary_sample flag for read alignment objects

bool SampleManager::IdentifySample(const BamAlignment& alignment, int& sample_index, bool& primary_sample) const
{

  string read_group;
  if (!alignment.GetTag("RG", read_group)) {
    cerr << "ERROR: Couldn't find read group id (@RG tag) for BAM Alignment " << alignment.Name << endl;
    exit(1);
  }

  map<string,int>::const_iterator I = read_group_to_sample_idx_.find(read_group);
  if (I == read_group_to_sample_idx_.end())
    return false;

  sample_index =I->second;
  primary_sample = (sample_index == primary_sample_);

  return true;
}






