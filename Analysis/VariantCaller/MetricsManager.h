/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MetricsManager.h
//! @ingroup  VariantCaller
//! @brief    Collects analysis metrics and saves them to a json file


#ifndef METRICSMANAGER_H
#define METRICSMANAGER_H

#include <string>
#include <list>
#include "ReferenceReader.h"
#include "BAMWalkerEngine.h"

using namespace std;

struct MetricsAccumulator {

  // Counters used for computing the deamination metric
  long int substitution_events[64];


  MetricsAccumulator() {
    for (int i = 0; i < 64; ++i)
      substitution_events[i] = 0;
  }

  void operator+= (const MetricsAccumulator& other) {
    for (int i = 0; i < 64; ++i)
      substitution_events[i] += other.substitution_events[i];
  }


  void CollectMetrics(list<PositionInProgress>::iterator& position_ticket, int haplotyle_length, const ReferenceReader* ref_reader);

};


class MetricsManager {
public:
  MetricsManager() {}
  ~MetricsManager() {}

  MetricsAccumulator& NewAccumulator();
  void FinalizeAndSave(const string& output_json);

private:
  list<MetricsAccumulator>  accumulators_;

};


#endif //METRICSMANAGER_H


