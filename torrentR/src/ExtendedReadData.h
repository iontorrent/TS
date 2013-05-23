/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef EXTENDEDREADDATA_H
#define EXTENDEDREADDATA_H

#include "api/BamReader.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include "../Analysis/file-io/ion_util.h"
#include "BamHelper.h"

using namespace std;

// WARNING: Not quite the same as the one in VariantCaller
// Reconciliation later
class ExtendedReadInfo{
  public:
  BamTools::BamAlignment alignment;
  string tDNA, qDNA;

  vector<float> measurementValue;
  vector<float> phase_params;
 
  int start_flow;
  int col, row;
  

  bool CheckHappyRead(BamHeaderHelper &my_helper, string &variant_contig, int variant_start_pos, int DEBUG);
  bool UnpackThisRead(BamHeaderHelper &my_helper, string &variant_contig, int variant_start_pos, int DEBUG);
  void UnpackAlignment();
  void GetTags();
};


#endif // EXTENDEDREADDATA_H
