/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     TargetsManager.cpp
//! @ingroup  VariantCaller
//! @brief    BED loader

#include "TargetsManager.h"

#include <stdlib.h>
#include <algorithm>
#include "BAMWalkerEngine.h"
//#include "ExtendParameters.h"


TargetsManager::TargetsManager()
{
  trim_ampliseq_primers = false;
}

TargetsManager::~TargetsManager()
{
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



void TargetsManager::Initialize(const ReferenceReader& ref_reader, const string& _targets, bool _trim_ampliseq_primers /*const ExtendParameters& parameters*/)
{

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




void TargetsManager::LoadRawTargets(const ReferenceReader& ref_reader, const string& bed_filename, list<UnmergedTarget>& raw_targets)
{
  FILE *bed_file = fopen(bed_filename.c_str(), "r");
  if (not bed_file) {
    cerr << "ERROR: Unable to open target file " << bed_filename << " : " << strerror(errno) << endl;
    exit(1);
  }

  char line[4096];
  char chr_name[4096];
  int begin;
  int end;
  char region_name[4096];
  int line_number = 0;

  while (fgets(line, 4096, bed_file)) {
    ++line_number;

    if (strncmp(line,"track",5) == 0) {
      // Parse track line if needed
      continue;
    }


    int num_fields = sscanf(line, "%s\t%d\t%d\t%s", chr_name, &begin, &end, region_name);
    if (num_fields == 0)
      continue;
    if (num_fields < 3) {
      cerr << "ERROR: Failed to parse target file line " << line_number << endl;
      exit(1);
    }

    raw_targets.push_back(UnmergedTarget());
    UnmergedTarget& target = raw_targets.back();
    target.begin = begin;
    target.end = end;
    target.chr = ref_reader.chr_idx(chr_name);
    if (num_fields > 3 and strcmp(region_name,".") != 0)
      target.name = region_name;

    if (target.chr < 0) {
      cerr << "ERROR: Target region " << target.name << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " has unrecognized chromosome name" << endl;
      exit(1);
    }

    if (begin < 0 || end > ref_reader.chr_size(target.chr)) {
      cerr << "ERROR: Target region " << target.name << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " is outside of reference sequence bounds ("
           << chr_name << ":0-" << ref_reader.chr_size(target.chr) << ")" << endl;
      exit(1);
    }
    if (end < begin) {
      cerr << "ERROR: Target region " << target.name << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " has inverted coordinates" << endl;
      exit(1);
    }
    AddExtraTrim(target, line, num_fields);
  }

  fclose(bed_file);

  if (raw_targets.empty()) {
    cerr << "ERROR: No targets loaded from " << bed_filename
         << " after parsing " << line_number << " lines" << endl;
    exit(1);
  }
}

void TargetsManager::AddExtraTrim(UnmergedTarget& target, char *line, int num_fields)
{
  // parse extra trimming out of the bed file
  // would be nice to unify with validate_bed using BedFile.h
  char keybuffer[4096];
  target.trim_left = 0;
  target.trim_right = 0;

  int numfields = sscanf(line, "%*s\t%*d\t%*d\t%*s\t%*s\t%*s\t%*s\t%s", keybuffer);
  if (numfields == 1) {
    string keypairs = keybuffer; 
    string left = "TRIM_LEFT=";
    string right = "TRIM_RIGHT=";
    long int trim_l = 0;
    long int trim_r = 0;

    // look for a trim_left field
    size_t found = keypairs.find(left);
    if (found != string::npos){
      string r1 = keypairs;
      r1.erase(0, found+left.size());
      trim_l = strtol (r1.c_str(), NULL, 0);
    }

    // look for a trim_right field
    found = keypairs.find(right);
    if (found != string::npos){
      string r1 = keypairs;
      r1.erase(0, found+right.size());
      trim_r = strtol (r1.c_str(), NULL, 0);
    }
    assert(trim_l>=0);
    assert(trim_r>=0);
    target.trim_left = trim_l;
    target.trim_right = trim_r;
  }
  // cout << "trim assigned to amplicon starting " << target.begin << " with trimming " << target.trim_left << " and " << target.trim_right << endl;
}


void TargetsManager::TrimAmpliseqPrimers(Alignment *rai, int unmerged_target_hint) const
{
  // set these before any trimming
  rai->align_start = rai->alignment.Position;
  rai->align_end = rai->alignment.GetEndPosition(false, true);

  if (not trim_ampliseq_primers)
    return;

  // Step 1: Find the first potential target region

  int target_idx = unmerged_target_hint;
  while (target_idx and (rai->alignment.RefID < unmerged[target_idx].chr or
          (rai->alignment.RefID == unmerged[target_idx].chr and rai->alignment.Position < unmerged[target_idx].end)))
    --target_idx;

  while (target_idx < (int)unmerged.size() and (rai->alignment.RefID > unmerged[target_idx].chr or
          (rai->alignment.RefID == unmerged[target_idx].chr and rai->alignment.Position >= unmerged[target_idx].end)))
    ++target_idx;


  // Step 2: Iterate over potential target regions, evaluate fit, pick the best fit

  int best_target_idx = -1;
  int best_fit_penalty = 100;
  int best_overlap = 0;

  while (target_idx < (int)unmerged.size() and rai->alignment.RefID == unmerged[target_idx].chr and rai->end >= unmerged[target_idx].begin) {

    int read_start = rai->alignment.Position;
    int read_end = rai->end;
    int read_prefix_size = unmerged[target_idx].begin - read_start;
    int read_postfix_size = read_end - unmerged[target_idx].end;
    int overlap = min(unmerged[target_idx].end, read_end) - max(unmerged[target_idx].begin, read_start);
    int fit_penalty = 100;

    if (not rai->alignment.IsReverseStrand()) {
      if (read_prefix_size > 0)
        fit_penalty = min(read_prefix_size,50) + max(0,50-overlap);
      else
        fit_penalty = min(-3*read_prefix_size,50) + max(0,50-overlap);
    } else {
      if (read_postfix_size > 0)
        fit_penalty = min(read_postfix_size,50) + max(0,50-overlap);
      else
        fit_penalty = min(-3*read_postfix_size,50) + max(0,50-overlap);
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

  if (best_target_idx == -1) {
    rai->filtered = true;
    rai->evaluator_filtered = true;
    return;
  }


  // Step 3: Do the actual primer trimming.
  //
  // For now, only adjust Position and Cigar.
  // Later, also adjust MD tag.
  // Even later, ensure the reads stay sorted, so no extra sorting is required outside of tvc

  vector<CigarOp>& old_cigar = rai->alignment.CigarData;
  vector<CigarOp> new_cigar;
  new_cigar.reserve(old_cigar.size() + 2);
  vector<CigarOp>::iterator old_op = old_cigar.begin();
  int ref_pos = rai->alignment.Position;

  // 3A: Cigar ops left of the target

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
        old_op->Length -= gap;
        ref_pos += gap;
        break;
      } else {
        ref_pos += old_op->Length;
        ++old_op;
        continue;
      }
    }
  }


  // 3B: Cigar ops in the middle of the target

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
      new_cigar.push_back(CigarOp('D'));
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
  }


  // 3C: Cigar ops to the right of the target

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








