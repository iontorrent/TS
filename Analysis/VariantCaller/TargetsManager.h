/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     TargetsManager.h
//! @ingroup  VariantCaller
//! @brief    BED loader


#ifndef TARGETSMANAGER_H
#define TARGETSMANAGER_H

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include "ReferenceReader.h"

struct Alignment;

struct MergedTarget {
  int         chr;
  int         begin;
  int         end;
  int         first_unmerged;
};

struct TargetStat{
	unsigned int read_coverage = 0;
	unsigned int family_coverage = 0;
	// fam_size_hist[x] = y indicates there are y families of size x.
	map<int, unsigned int> fam_size_hist;
};

class TargetsManager {
public:
  TargetsManager();
  ~TargetsManager();

  void Initialize(const ReferenceReader& ref_reader, const string& _targets, float min_cov_frac = 0.0f, bool _trim_ampliseq_primers = false);

  struct UnmergedTarget {
    int          chr    = 0;
    int          begin  = 0;
    int          end    = 0;
    string       name   = "";
    int          merged = 0;
    int          trim_left  = 0;
    int          trim_right = 0;
    int          hotspots_only = 0;
    int          read_mismatch_limit = -1;
    TargetStat   my_stat;
  };


  void LoadRawTargets(const ReferenceReader& ref_reader, const string& bed_filename, list<UnmergedTarget>& raw_targets);
  void ParseBedInfoField(UnmergedTarget& target, const string info);
  void TrimAmpliseqPrimers(Alignment *rai, int unmerged_target_hint) const;
  void GetBestTargetIndex(Alignment *rai, int unmerged_target_hint, int& best_target_idx, int& best_fit_penalty, int& best_overlap) const;
  bool FilterReadByRegion(Alignment* rai, int unmerged_target_hint) const;
  void AddCoverageToRegions(const map<int, TargetStat>& stat_of_targets);
  void WriteTargetsCoverage(const string& file_path, const ReferenceReader& ref_reader) const;
  int  ReportHotspotsOnly(const MergedTarget &merged, int chr, long pos);

  vector<UnmergedTarget>  unmerged;
  vector<MergedTarget>    merged;
  bool  trim_ampliseq_primers;

  // The following variables are just for bool FilterReadByRegion(Alignment* rai, int recent_target) use only.
  float min_coverage_fraction;
private:
  pthread_mutex_t coverage_counter_mutex_;
};



#endif //TARGETSMANAGER_H
