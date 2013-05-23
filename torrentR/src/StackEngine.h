/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef STACKENGINE_H
#define STACKENGINE_H

#include "api/BamReader.h"

#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include "BamHelper.h"
#include "ExtendedReadData.h"

#include "calcHypothesesDistancesEngine.h"

using namespace std;

// stack = pileup
// 

// grab the "stack" of reads associated with a given sequence location in a bam file
class StackPlus{
public:
  vector<ExtendedReadInfo> read_stack;
  string flow_order;
  int GrabStack(char *bamFile, string &variant_contig, unsigned int variant_position);
};

int GrabStack(StackPlus &my_data, char *bamFile, string &variant_contig, unsigned int variant_position);

#endif // STACKENGINE_H
