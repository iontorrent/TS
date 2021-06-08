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

struct TargetStats{
	unsigned int read_coverage_in_families = 0;   // Number of reads in a family that cover this target (where one read can cover multiple targets, e.g., super amplicon, overlapping amplicons).
    unsigned int read_coverage_in_families_by_best_target = 0;  // Number of reads that cover this target (where one read can cover only one target, determined by the best target assignment).
	unsigned int family_coverage = 0; // Number of families that cover this target (where one family can cover multiple targets, e.g., super amplicon, overlapping amplicons).
	unsigned int raw_read_coverage = 0;   // Number of raw reads that cover this target (where one read can cover multiple targets, e.g., super amplicon, overlapping amplicons), including reads in non-functional families.
	unsigned int raw_read_coverage_by_best_target = 0;   // Number of raw reads that cover this target (where one read can cover only one target, determined by the best target assignment), including reads in non-functional families.

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
    // Extra amplicon trimming
    int          trim_left  = 0;
    int          trim_right = 0;
    // HS_ONLY for unify_vcf
    int          hotspots_only = 0;
    // Amplicon-specific override for read filtering parameters
    bool         read_mismatch_limit_override = false;
    int          read_mismatch_limit = 0;
    bool         read_snp_limit_override = false;
    int          read_snp_limit = 0;
    bool         min_mapping_qv_override = false;
    int          min_mapping_qv = 0;
    bool         min_cov_fraction_override = false;
    float        min_cov_fraction = 0.0f;
    // Amplicon-specific override for molcular tagging parameters
    bool         min_tag_fam_size_override = false;
    int          min_tag_fam_size = 0;
    bool         min_fam_per_strand_cov_override = false;
    int          min_fam_per_strand_cov = 0;
    // Amplicon stats
    TargetStats   my_stats;
  };


  void LoadRawTargets(const ReferenceReader& ref_reader, const string& bed_filename, list<UnmergedTarget>& raw_targets);
  void ParseBedInfoField(UnmergedTarget& target, const string info) const;
  void TrimAmpliseqPrimers(Alignment *rai, int unmerged_target_hint) const;
  void GetBestTargetIndex(Alignment *rai, int unmerged_target_hint, int& best_target_idx, int& best_fit_penalty, int& best_overlap) const;
  bool FilterReadByRegion(Alignment* rai, int unmerged_target_hint) const;
  void AddCoverageToRegions(const map<int, TargetStats>& stat_of_targets);
  void AddToRawReadCoverage(const Alignment* const rai);
  void WriteTargetsCoverage(const string& file_path, const ReferenceReader& ref_reader, bool use_best_target, bool use_mol_tags) const;
  int  ReportHotspotsOnly(const MergedTarget &merged, int chr, long pos);
  bool IsCoveredByMerged(int merged_idx, int chr, long pos) const;
  bool IsFullyCoveredByMerged(int merged_idx, int chr, long pos_start, long pos_end) const;
  bool IsFullyCoveredByUnmerged(int unmerged_idx, int chr, long pos_start, long pos_end) const;
  bool IsOverlapWithUnmerged(int unmerged_idx, int chr, long pos_start, long pos_end) const;
  bool IsBreakingIntervalInMerged(int merged_idx, int chr, long pos_start, long pos_end) const;
  int FindMergedTargetIndex(int chr, long pos) const;

  vector<UnmergedTarget>  unmerged;
  vector<MergedTarget>    merged;
  vector<int>             chr_to_merged_idx;
  bool  trim_ampliseq_primers;

  // The following variables are just for bool FilterReadByRegion(Alignment* rai, int recent_target) use only.
  float min_coverage_fraction;
private:
  pthread_mutex_t coverage_counter_mutex_;
};



#endif //TARGETSMANAGER_H
