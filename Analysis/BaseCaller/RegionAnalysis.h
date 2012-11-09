/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONANALYSIS_H
#define REGIONANALYSIS_H

//! @file     RegionAnalysis.h
//! @ingroup  BaseCaller
//! @brief    RegionAnalysis. Older Nelder-Mead phasing estimator superseded by PhaseEstimator. Delete soon.

#include "Mask.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "RegionWellsReader.h"
#include "BaseCallerUtils.h"

// RegionAnalysis
//
// Provides a parallel processing structure where one thread processes one region at a time.
// Currently treephaser-based phase parameter estimation is implemented here.
// May be easy to extend to other analysis code here.

class RawWells;
class RegionAnalysis {
public:
  RegionAnalysis();

  void analyze(vector<float> *_cf, vector<float> *_ie, vector<float> *_dr, RawWells *_wells, Mask *_mask,
      const vector<KeySequence>& _keys, const ion::FlowOrder& _flowOrder, int numWorkers,
      int cfiedrRegionsX, int cfiedrRegionsY);

private:
  friend void * RegionAnalysisWorker(void *);
  void worker_Treephaser();

  void NelderMeadOptimization(std::vector<BasecallerRead> &dataAll, DPTreephaser& treephaser, float *parameters, int numEvaluations, int numParameters);
  float evaluateParameters(std::vector<BasecallerRead> &dataAll, DPTreephaser& treephaser, float *parameters);

  // Command Line Options
  RegionWellsReader   wellsReader;
  Mask *              mask;
  vector<KeySequence> keys;
  ion::FlowOrder      flowOrder;

  vector<float> *     cf;
  vector<float> *     ie;
  vector<float> *     dr;

  pthread_mutex_t *   common_output_mutex;
};

void *RegionAnalysisWorker(void *);

#endif // REGIONANALYSIS_H
