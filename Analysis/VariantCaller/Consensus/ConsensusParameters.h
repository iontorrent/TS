/* Copyright (C) 2016 Thermo Fisher Scientific, Inc. All Rights Reserved */

#ifndef CONSENSUSPARAMETERS_H
#define CONSENSUSPARAMETERS_H

#include <string>
#include <vector>
#include "OptArgs.h"
#include "OptBase.h"
#include "json/json.h"
#include "ExtendParameters.h"

using namespace std;


class ConsensusParameters : public ExtendParameters {
public:
  string consensus_bam; // Base name of the two consensus bam files.

  bool need_3_end_adapter = false;
  bool filter_qt_reads = false;
  bool filter_single_read_consensus = false;
  bool skip_consensus = false;
//  bool use_flowspace_clustering; // generate one consensus read for each flowspace cluster
  // functions
  ConsensusParameters(int argc, char** argv);
  void SetupFileIO(OptArgs &opts);
};


#endif // CONSENSUSPARAMETERS_H

