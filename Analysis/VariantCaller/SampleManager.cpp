/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     SampleManager.h
//! @ingroup  VariantCaller
//! @brief    Manages sample and read group names


#include "SampleManager.h"

#include "BAMWalkerEngine.h"

// const string& force_sample_name,
// const string& sample_name,


void SampleManager::Initialize (const SamHeader& bam_header, string& sample_name, const string& force_sample_name)
{

  num_samples_ = 0;

  for (SamReadGroupConstIterator read_group = bam_header.ReadGroups.Begin(); read_group < bam_header.ReadGroups.End(); ++read_group) {

    string sample_name;
    if (force_sample_name.empty())
      sample_name = read_group->Sample;
    else
      sample_name = force_sample_name;


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
  }

  if (num_samples_ == 0) {
    cerr << "ERROR: BAM file(s) do not have any read group definitions" << endl;
    exit(1);
  }


  bool default_sample = false;
  //now check if there are multiple samples in the BAM file and if so user should provide a sampleName to process
  if (num_samples_ == 1 && sample_name.empty()) {
    sample_name = sample_names_[0];
    default_sample = true;

  } else if (num_samples_ > 1 && sample_name.empty())  {
    cerr << "ERROR: Multiple Samples found in BAM file/s provided. Torrent Variant Caller currently supports variant calling on only one sample. " << endl;
    cerr << "ERROR: Please select sample name to process using -g parameter. " << endl;
    exit(1);
  }

  bool primary_sample_found = false;
  for (int i = 0; i < num_samples_; ++i) {
    if (sample_names_[i] == sample_name) {
      primary_sample_ = i;
      primary_sample_found = true;
    }
  }

  if (!primary_sample_found) {
    cerr << "ERROR: Sample " << sample_name << " provided using -g option "
         << "is not associated with any read groups in BAM file(s)" << endl;
    exit(1);
  }

  //now find the read group ID associated with this sample name
  int num_primary_read_groups = 0;
  for (map<string, int>::const_iterator p = read_group_to_sample_idx_.begin(); p != read_group_to_sample_idx_.end(); ++p)
    if (primary_sample_ == p->second)
      num_primary_read_groups++;

  if (!force_sample_name.empty())
    cout << "SampleManager: All read groups forced to assume sample name " <<  force_sample_name << endl;

  cout << "SampleManager: Found " << read_group_to_sample_idx_.size() << " read group(s) and " << num_samples_ << " sample(s)." << endl;
  if (default_sample)
    cout << "SampleManager: Primary sample \"" << sample_name << "\" (default) present in " << num_primary_read_groups << " read group(s)" << endl;
  else
    cout << "SampleManager: Primary sample \"" << sample_name << "\" (set via -g) " << num_primary_read_groups << " read group(s)" << endl;

}



//bool SampleManager::IdentifySample(Alignment& ra) const
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






