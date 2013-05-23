/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef CALCHYPOTHESESDISTANCESENGINE_H
#define CALCHYPOTHESESDISTANCESENGINE_H

#include "DPTreephaser.h"

void CalculateHypDistances(const vector<float>& NormalizedMeasurements,
				  const float& cf,
				  const float& ie,
				  const float& droop,
				  const ion::FlowOrder& flow_order,
				  const vector<string>& Hypotheses,
				  const int& startFlow,
				  vector<float>& DistanceObserved,
				  vector<float>& DistanceHypotheses,
				  vector<vector<float> >& predictions,
				  vector<vector<float> >& normalizedMeasurements,
				  int applyNormalization,
				  int verbose);

#endif // CALCHYPOTHESESDISTANCESENGINE_H
