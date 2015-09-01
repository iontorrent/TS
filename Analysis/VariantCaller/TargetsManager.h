/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     TargetsManager.h
//! @ingroup  VariantCaller
//! @brief    BED loader


#ifndef TARGETSMANAGER_H
#define TARGETSMANAGER_H

#include <vector>
#include <list>
#include <string>
#include "ReferenceReader.h"

struct Alignment;

struct MergedTarget {
  int         chr;
  int         begin;
  int         end;
  int         first_unmerged;
};

class TargetsManager {
public:
  TargetsManager();
  ~TargetsManager();

  void Initialize(const ReferenceReader& ref_reader, const string& _targets, bool _trim_ampliseq_primers = false);


  struct UnmergedTarget {
    int         chr;
    int         begin;
    int         end;
    string      name;
    int         merged;
    int         trim_left;
    int         trim_right;
  };


  void LoadRawTargets(const ReferenceReader& ref_reader, const string& bed_filename, list<UnmergedTarget>& raw_targets);
  void AddExtraTrim(UnmergedTarget& target, char *region_name, int num_fields);
  void TrimAmpliseqPrimers(Alignment *rai, int unmerged_target_hint) const;

  vector<UnmergedTarget>  unmerged;
  vector<MergedTarget>    merged;
  bool  trim_ampliseq_primers;

};



#endif //TARGETSMANAGER_H
