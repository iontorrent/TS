/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KCLASS_H
#define KCLASS_H

#include <vector>
#include <string>
#include "Traces.h"
#include "ZeromerDiff.h"
#include "Mask.h"
#include "KeyClassifier.h"
#include "KeyReporter.h"
#include "TraceStore.h"
#include "ZeromerModelBulk.h"
/** Wrapper around KeyClassifier algorithms, fills in data before calling them */
class KClass {
 public:

void InitialClassifyWell(Mask &mask,
                                 ZeromerDiff<double> &bg,
                                 std::vector<KeySeq> &keys, 
                                 TraceStore<double> &traceStore,
                                 int nFlows,
                                 Col<double> &time,
                                 vector<KeyReporter<double> *>&report,
                                 double minSnr,
                                 std::vector<double> &dist,
                                 std::vector<std::vector<float> *> &distValues,
                                 KeyFit &fit);
	
  void ClassifyWellKnownTau(Mask &mask,
                                  ZeromerModelBulk<double> &bg,
                                  std::vector<KeySeq> &keys, 
                                 TraceStore<double> &traceStore,
                                 int nFlows,
                                  Col<double> &time,
                                  Mat<double> *darkMatter,
                                  Mat<double> *onemers,
                                  size_t frameEnd,
                                  vector<KeyReporter<double> *>&report,
                                  GridMesh<SampleQuantiles<double> > &emptyEstimates,
                                  double tauEEst,
                                  double minSnr,
                                  std::vector<double> &dist,
                                  std::vector<std::vector<float> *> &distValues,
                            KeyFit &fit);
	
  void FillInData(TraceStore<double> &traceStore,
                  int nFlows,
                  KeyFit &fit);

 private:
  std::vector<double> distances;
  std::vector<SampleQuantiles<double> *> values;
  Mat<double> wellFlows;
  Mat<double> refFlows;
  vector<float> reference;
  Col<double> incRef;
  vector<float> signal;
  Mat<double> predicted;
  KeyClassifier mKClass;
};

#endif // KCLASS_H
