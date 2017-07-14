/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     TargetsManager.cpp
//! @ingroup  VariantCaller
//! @brief    BED loader

#include "TargetsManager.h"

#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include "BAMWalkerEngine.h"
//#include "ExtendParameters.h"


TargetsManager::TargetsManager()
{
  trim_ampliseq_primers = false;
  min_coverage_fraction = 0.0f;
  pthread_mutex_init(&coverage_counter_mutex_, NULL);
}

TargetsManager::~TargetsManager()
{
	pthread_mutex_destroy(&coverage_counter_mutex_);
}

bool CompareTargets(TargetsManager::UnmergedTarget *i, TargetsManager::UnmergedTarget *j)
{
  if (i->chr < j->chr)
    return true;
  if (i->chr > j->chr)
    return false;
  if (i->begin < j->begin)
    return true;
  if (i->begin > j->begin)
    return false;
  return i->end < j->end;
}

// Goals:
// * BED loading
//   - Support merged and unmerged BEDs (merge if needed, keep both versions)
//   - Support plain and detailed BEDs (track line, extra columns)
//   - Possibly support unsorted BEDs (low priority)
//   - If no BED provided, build a fake one covering entire reference
//   - Validation
// * Ampliseq primer trimming
//   - Detect and trim Ampliseq primers
//   - Save region name to read's tags, maybe other primer-trimming info



void TargetsManager::Initialize(const ReferenceReader& ref_reader, const string& _targets, float min_cov_frac, bool _trim_ampliseq_primers /*const ExtendParameters& parameters*/)
{
  min_coverage_fraction = min_cov_frac;
  //
  // Step 1. Retrieve raw target definitions
  //
  list<UnmergedTarget>  raw_targets;

  if (not _targets.empty()) {
    LoadRawTargets(ref_reader, _targets, raw_targets);

  } else {
    for (int chr = 0; chr < ref_reader.chr_count(); ++chr) {
      raw_targets.push_back(UnmergedTarget());
      UnmergedTarget& target = raw_targets.back();
      target.begin = 0;
      target.end = ref_reader.chr_size(chr);
      target.chr = chr;
    }
  }

  //
  // Step 2. Sort raw targets and transfer to the vector
  //

  int num_unmerged = raw_targets.size();
  vector<UnmergedTarget*> raw_sort;
  raw_sort.reserve(num_unmerged);
  for (list<UnmergedTarget>::iterator I = raw_targets.begin(); I != raw_targets.end(); ++I)
    raw_sort.push_back(&(*I));
  sort(raw_sort.begin(), raw_sort.end(), CompareTargets);

  unmerged.reserve(num_unmerged);
  bool already_sorted = true;
  list<UnmergedTarget>::iterator I = raw_targets.begin();
  for (int idx = 0; idx < num_unmerged; ++idx, ++I) {
    if (raw_sort[idx] != &(*I) and already_sorted) {
      already_sorted = false;
      cerr << "TargetsManager: BED not sorted at position " << idx;
      cerr << " replaced " << I->name << ":" << I->chr << ":" << I->begin << "-" << I->end;
      cerr << " with " << raw_sort[idx]->name << ":" << raw_sort[idx]->chr << ":" << raw_sort[idx]->begin << "-" << raw_sort[idx]->end << endl;
    }
    unmerged.push_back(*raw_sort[idx]);
  }



  //
  // Step 3. Merge targets and link merged/unmerged entries
  //

  merged.reserve(num_unmerged);
  bool already_merged = true;
  for (int idx = 0; idx < num_unmerged; ++idx) {
    if (idx and merged.back().chr == unmerged[idx].chr and merged.back().end >= unmerged[idx].begin) {
      merged.back().end = max(merged.back().end, unmerged[idx].end);
      already_merged = false;
    } else {
      merged.push_back(MergedTarget());
      merged.back().chr = unmerged[idx].chr;
      merged.back().begin = unmerged[idx].begin;
      merged.back().end = unmerged[idx].end;
      merged.back().first_unmerged = idx;
    }
    unmerged[idx].merged = merged.size();
  }

  if (_targets.empty()) {
    cout << "TargetsManager: No targets file specified, processing entire reference" << endl;

  } else  {
    cout << "TargetsManager: Loaded targets file " << _targets << endl;

    cout << "TargetsManager: " << num_unmerged << " target(s)";
    if (not already_merged)
      cout << " (" << merged.size() << " after merging)";
    cout << endl;
    if (not already_sorted)
      cout << "TargetsManager: Targets required sorting" << endl;

    trim_ampliseq_primers = _trim_ampliseq_primers;
    if (trim_ampliseq_primers)
      cout << "TargetsManager: Trimming of AmpliSeq primers is enabled" << endl;
  }


}


// -------------------------------------------------------------------------------------

void TargetsManager::LoadRawTargets(const ReferenceReader& ref_reader, const string& bed_filename, list<UnmergedTarget>& raw_targets)
{
  ifstream bedfile(bed_filename.c_str());
  if (not bedfile.is_open()){
    cerr << "ERROR: Unable to open target file " << bed_filename << " : " << strerror(errno) << endl;
    exit(1);
  }

  string line, bed_field;
  vector<string> bed_line;
  int line_number = 0;

  while(getline(bedfile, line)){

    ++line_number;
    // Skip header line(s)
    if (line.compare(0,5,"track")==0 or line.compare(0,7,"browser")==0 or line.length()==0)
      continue;

    // Split line into tab separated fields
    bed_line.clear();
    stringstream ss(line);
    while (getline(ss, bed_field, '\t'))
      bed_line.push_back(bed_field);

    // the first three columns are required in the bad format
    unsigned int num_fields = bed_line.size();
    if (num_fields < 3) {
      cerr << "ERROR: Failed to parse target file line " << line_number << endl;
      exit(1);
    }

    raw_targets.push_back(UnmergedTarget());
    UnmergedTarget& target = raw_targets.back();
    target.chr = ref_reader.chr_idx(bed_line[0].c_str());
    target.begin = strtol (bed_line[1].c_str(), NULL, 0);
    target.end = strtol (bed_line[2].c_str(), NULL, 0);

    if (num_fields > 3 and bed_line[3]!=".")
      target.name = bed_line[3];

    // Validate target
    if (target.chr < 0){
      cerr << "ERROR: Target region " << target.name << " (" << bed_line[0] << ":" << bed_line[1] << "-" << bed_line[2] << ")"
           << " has unrecognized chromosome name" << endl;
      exit(1);
    }
    if (target.begin < 0 || target.end > ref_reader.chr_size(target.chr)) {
      cerr << "ERROR: Target region " << target.name << " (" << bed_line[0] << ":" << bed_line[1] << "-" << bed_line[2] << ")"
           << " is outside of reference sequence bounds ("
           << bed_line[0] << ":0-" << ref_reader.chr_size(target.chr) << ")" << endl;
      exit(1);
    }
    if (target.end < target.begin) {
      cerr << "ERROR: Target region " << target.name << " (" << bed_line[0] << ":" << bed_line[1] << "-" << bed_line[2] << ")"
           << " has inverted coordinates" << endl;
      exit(1);
    }

    // And now we simply assume that we have a beddetail file with at least 5 columns
    if (num_fields > 4)
      ParseBedInfoField(target, bed_line[num_fields-1]);
  }

  bedfile.close();
  if (raw_targets.empty()) {
    cerr << "ERROR: No targets loaded from " << bed_filename << " after parsing " << line_number << " lines" << endl;
    exit(1);
  }
}

// -------------------------------------------------------------------------------------

bool ParseInfoKey(const string key, string info, long &value) {

  size_t found = info.find(key);
  if (found == string::npos){
    return false;
  }

  info.erase(0, found+key.size());
  value = strtol (info.c_str(), NULL, 0);
  if (value>=0)
    return true;
  else
    return false;
}

// -------------------------------------------------------------------------------------

void TargetsManager::ParseBedInfoField(UnmergedTarget& target, const string info)
{
  // parse extra parameters out of the bed file info fields
  long int temp = 0;

  target.trim_left = 0;
  if (ParseInfoKey("TRIM_LEFT=", info, temp))
    target.trim_left = temp;

  target.trim_right = 0;
  if (ParseInfoKey("TRIM_RIGHT=", info, temp))
    target.trim_right = temp;

  target.hotspots_only = 0;
  if (ParseInfoKey("HS_ONLY=", info, temp))
    target.hotspots_only = temp;

  target.read_mismatch_limit = -1;
  if (ParseInfoKey("READ_MISMATCH_LIMIT=", info, temp)){
      target.read_mismatch_limit = temp;
  }
}

// -------------------------------------------------------------------------------------
// expects a zero-based position to match target indexing.

int TargetsManager::ReportHotspotsOnly(const MergedTarget &merged, int chr, long pos0)
{
  // Verify that (chr,pos) is inside the merged target
  if (merged.chr != chr or pos0 < merged.begin or pos0 >= merged.end){
    cerr << "ReportHotspotsOnly ERROR: (chr=" << chr << ",pos=" << pos0 << ") not in target chr "
         << merged.chr << ":" << merged.begin << "-" << merged.end << endl;
    exit(1);
  }

  // We take the first target index that contains the variant position, excluding primers.
  unsigned int target_idx = merged.first_unmerged;
  while (target_idx < unmerged.size() and unmerged[target_idx].end-unmerged[target_idx].trim_right < pos0)
    ++target_idx;

  if (target_idx < unmerged.size() and pos0 >= unmerged[target_idx].begin)
    return  unmerged[target_idx].hotspots_only;

  return 0;
}

// -------------------------------------------------------------------------------------

void TargetsManager::GetBestTargetIndex(Alignment *rai, int unmerged_target_hint, int& best_target_idx, int& best_fit_penalty, int& best_overlap) const
{

  // set these before any trimming
  rai->align_start = rai->alignment.Position;
  rai->align_end = rai->alignment.GetEndPosition(false, true);
  rai->old_cigar = rai->alignment.CigarData;

  // Step 1: Find the first potential target region

  int target_idx = unmerged_target_hint;
  while (target_idx and (rai->alignment.RefID < unmerged[target_idx].chr or
          (rai->alignment.RefID == unmerged[target_idx].chr and rai->alignment.Position < unmerged[target_idx].end)))
    --target_idx;

  while (target_idx < (int)unmerged.size() and (rai->alignment.RefID > unmerged[target_idx].chr or
          (rai->alignment.RefID == unmerged[target_idx].chr and rai->alignment.Position >= unmerged[target_idx].end)))
    ++target_idx;


  // Step 2: Iterate over potential target regions, evaluate fit, pick the best fit
  best_target_idx = -1;
  best_fit_penalty = 100;
  best_overlap = 0;

  while (target_idx < (int)unmerged.size() and rai->alignment.RefID == unmerged[target_idx].chr and rai->end >= unmerged[target_idx].begin) {

    int read_start = rai->alignment.Position;
    int read_end = rai->end;
    int read_prefix_size = unmerged[target_idx].begin - read_start;
    int read_postfix_size = read_end - unmerged[target_idx].end;
    int overlap = min(unmerged[target_idx].end, read_end) - max(unmerged[target_idx].begin, read_start);
    int fit_penalty = 100;

    float overlap_ratio = (float) overlap / (float) (unmerged[target_idx].end - unmerged[target_idx].begin);
    if (overlap_ratio > min_coverage_fraction)
      rai->target_coverage_indices.push_back(target_idx);

    if (not rai->alignment.IsReverseStrand()) {
      if (read_prefix_size > 0)
        fit_penalty = min(read_prefix_size,50) + max(0,50-overlap);
      else
        fit_penalty = min(-3*read_prefix_size,50) + max(0,50-overlap);
      if (read_postfix_size > 30) fit_penalty += read_postfix_size-10;
    } else {
      if (read_postfix_size > 0)
        fit_penalty = min(read_postfix_size,50) + max(0,50-overlap);
      else
        fit_penalty = min(-3*read_postfix_size,50) + max(0,50-overlap);
      if (read_prefix_size > 30) fit_penalty += read_prefix_size-10;
    }
    if (read_prefix_size > 0 and read_postfix_size > 0)
      fit_penalty -= 10;

    if ((best_fit_penalty > fit_penalty and overlap > 0) or (best_fit_penalty == fit_penalty and overlap > best_overlap)) {
      best_fit_penalty = fit_penalty;
      best_target_idx = target_idx;
      best_overlap = overlap;
    }

    ++target_idx;
  }
  if (rai->target_coverage_indices.size() > 1){
    sort(rai->target_coverage_indices.begin(), rai->target_coverage_indices.end());
  }
}


void TargetsManager::TrimAmpliseqPrimers(Alignment *rai, int unmerged_target_hint) const
{
  int best_target_idx = -1;
  int best_fit_penalty = 100;
  int best_overlap = 0;
  GetBestTargetIndex(rai, unmerged_target_hint, best_target_idx, best_fit_penalty, best_overlap);

  if (best_target_idx < 0 or rai->target_coverage_indices.empty()){
	  rai->filtered = true;
	  return;
  }

  if (not trim_ampliseq_primers){
	  return;
  }

  // Step 1: Find the best target index
  if (best_target_idx < 0){
	  rai->filtered = true;
	  return;
  }
  // Step 2: Do the actual primer trimming.
  //
  // For now, only adjust Position and Cigar.
  // Later, also adjust MD tag.
  // Even later, ensure the reads stay sorted, so no extra sorting is required outside of tvc

  vector<CigarOp>& old_cigar = rai->alignment.CigarData;
  vector<CigarOp> new_cigar;
  new_cigar.reserve(old_cigar.size() + 2);
  vector<CigarOp>::iterator old_op = old_cigar.begin();
  int ref_pos = rai->alignment.Position;

  // 2A: Cigar ops left of the target

  int begin = unmerged[best_target_idx].begin + unmerged[best_target_idx].trim_left;
  if (begin > unmerged[best_target_idx].end)
    begin = unmerged[best_target_idx].end;

  int end = unmerged[best_target_idx].end - unmerged[best_target_idx].trim_right;
  if (end <= begin)
    end = begin;
    

  while (old_op != old_cigar.end() and ref_pos <= begin) {
    if (old_op->Type == 'H') {
      ++old_op;
      continue;
    }

    if (old_op->Type == 'S' or old_op->Type == 'I') {
      if (new_cigar.empty())
        new_cigar.push_back(CigarOp('S'));
      new_cigar.back().Length += old_op->Length;
      ++old_op;
      continue;
    }

    unsigned int gap = begin - ref_pos;
    if (gap == 0)
      break;

    if (old_op->Type == 'M' or old_op->Type == 'N') {
      if (new_cigar.empty())
        new_cigar.push_back(CigarOp('S'));
      if (old_op->Length > gap) {
        new_cigar.back().Length += gap;
        old_op->Length -= gap;
        ref_pos += gap;
        break;
      } else {
        new_cigar.back().Length += old_op->Length;
        ref_pos += old_op->Length;
        ++old_op;
        continue;
      }
    }

    if (old_op->Type == 'D') {
      if (old_op->Length > gap) {
        //old_op->Length -= gap;
        //ref_pos += gap; 
	// avoid #S#D case, extend align position, remove the leading D.
	ref_pos += old_op->Length;
	++old_op;
        break;
      } else {
        ref_pos += old_op->Length;
        ++old_op;
        continue;
      }
    }
  }


  // 2B: Cigar ops in the middle of the target

  rai->alignment.Position = ref_pos;

  while (old_op != old_cigar.end() and ref_pos < end) {
    if (old_op->Type == 'H') {
      ++old_op;
      continue;
    }

    unsigned int gap = end - ref_pos;

    if (old_op->Type == 'S' or old_op->Type == 'I') {
      new_cigar.push_back(*old_op);
      ++old_op;
      continue;
    }

    if (old_op->Type == 'M' or old_op->Type == 'N') {
      new_cigar.push_back(CigarOp(old_op->Type));
      if (old_op->Length > gap) {
        new_cigar.back().Length = gap;
        old_op->Length -= gap;
        ref_pos += gap;
        break;
      } else {
        new_cigar.back().Length = old_op->Length;
        ref_pos += old_op->Length;
        ++old_op;
        continue;
      }
    }

    if (old_op->Type == 'D') {
      if (old_op->Length > gap) {
	// last D op, remove this one
        ref_pos += old_op->Length;
	++old_op; 
        break;
      } else {
        new_cigar.push_back(CigarOp('D'));
        new_cigar.back().Length = old_op->Length;
        ref_pos += old_op->Length;
        ++old_op;
        continue;
      }
    }
  }

  // 2C: Cigar ops to the right of the target

  for (; old_op != old_cigar.end(); ++old_op) {
    if (old_op->Type == 'H' or old_op->Type == 'D')
      continue;

    if (new_cigar.empty() or new_cigar.back().Type != 'S')
      new_cigar.push_back(CigarOp('S'));
    new_cigar.back().Length += old_op->Length;
  }

  rai->alignment.CigarData.swap(new_cigar);


  // Debugging info

  stringstream ZL;
  ZL << unmerged[best_target_idx].name << ":" <<  best_fit_penalty << ":" << best_overlap;

  rai->alignment.AddTag("ZL", "Z", ZL.str());
}


// Filter out a read if any of the following conditions is satisfied
// a) The read does not cover any region.
// b) The coverage ratio at the best region < min_coverage_ratio.
bool TargetsManager::FilterReadByRegion(Alignment* rai, int unmerged_target_hint) const
{
	bool is_filtered_out = false;
	int best_target_idx = -1;
	int best_fit_penalty = 100;
	int best_overlap = 0;
	GetBestTargetIndex(rai, unmerged_target_hint, best_target_idx, best_fit_penalty, best_overlap);

	// Filter out the read if it does not cover any region.
	if (best_target_idx < 0 or rai->target_coverage_indices.empty()){
		is_filtered_out = true;
		rai->filtered = true;
	    return is_filtered_out;
	}

	return is_filtered_out;
}


void TargetsManager::WriteTargetsCoverage(const string& target_cov_file, const ReferenceReader& ref_reader) const {
	ofstream target_cov_out;
	target_cov_out.open(target_cov_file.c_str(), ofstream::out);
	target_cov_out << "chr"           << "\t"
				   << "pos_start"     << "\t"
				   << "pos_end"       << "\t"
				   << "name"          << "\t"
				   << "read_depth"    << "\t"
				   << "family_depth"  << "\t"
				   << "fam_size_hist" << endl;

	for (vector<UnmergedTarget>::const_iterator target_it = unmerged.begin(); target_it != unmerged.end(); ++target_it){
		unsigned int check_read_cov = 0;
		unsigned int check_fam_cov = 0;
		target_cov_out << ref_reader.chr_str(target_it->chr) << "\t"
				       << target_it->begin << "\t"
					   << target_it->end << "\t"
					   << (target_it->name.empty()? "." : target_it->name) << "\t"
					   << target_it->my_stat.read_coverage << "\t"
					   << target_it->my_stat.family_coverage << "\t";
		for (map<int, unsigned int>::const_iterator hist_it = target_it->my_stat.fam_size_hist.begin(); hist_it != target_it->my_stat.fam_size_hist.end(); ++hist_it){
			target_cov_out << "(" << hist_it->first << "," << hist_it->second <<"),";
			check_fam_cov += hist_it->second;
			check_read_cov += (hist_it->second * (unsigned int) hist_it->first);
		}
		target_cov_out << endl;
		// Check fam_size_hist matches read/family coverages.
		assert((check_read_cov == target_it->my_stat.read_coverage) and (check_fam_cov == target_it->my_stat.family_coverage));
	}
	target_cov_out.close();
}

void TargetsManager::AddCoverageToRegions(const map<int, TargetStat>& stat_of_targets){
	pthread_mutex_lock(&coverage_counter_mutex_);
	for (map<int, TargetStat>::const_iterator stat_it = stat_of_targets.begin(); stat_it != stat_of_targets.end(); ++stat_it){
		int target_idx = stat_it->first;
		unmerged[target_idx].my_stat.read_coverage += stat_it->second.read_coverage;
		unmerged[target_idx].my_stat.family_coverage += stat_it->second.family_coverage;
		for (map<int, unsigned int>::const_iterator hist_it = stat_it->second.fam_size_hist.begin(); hist_it != stat_it->second.fam_size_hist.end(); ++hist_it){
			unmerged[target_idx].my_stat.fam_size_hist[hist_it->first] += hist_it->second;
		}
	}
	pthread_mutex_unlock(&coverage_counter_mutex_);
}
