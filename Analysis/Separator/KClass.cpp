/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "KClass.h"

void KClass::FillInData(TraceStore<double> &traceStore,
                        int nFlows,
                        KeyFit &fit) {
  Col<double> trace;
  int minFrame = 0;
  int maxFrame = traceStore.GetNumFrames();
  // int minFrame = max(traceStore.GetT0(fit.wellIdx)-4, 0.0f);
  // int maxFrame = min(minFrame + 25.0f, (float)traceStore.GetNumFrames());
  int nFrames = maxFrame - minFrame;
  if (wellFlows.n_rows != (size_t)nFrames || wellFlows.n_cols != (size_t)nFlows) {
    wellFlows.set_size(nFrames, nFlows);
    refFlows.set_size(nFrames, nFlows);
    predicted.set_size(nFrames, nFlows);
  }

  trace.resize(traceStore.GetNumFrames());
  fill(refFlows.begin(), refFlows.end(), 0);
  fill(wellFlows.begin(), wellFlows.end(), 0);

  for (size_t flowIx = 0; flowIx < (size_t)nFlows; flowIx++) {
    traceStore.GetTrace(fit.wellIdx, flowIx, trace.begin());
    copy(trace.begin() + minFrame, trace.begin() + maxFrame, wellFlows.begin_col(flowIx));
    traceStore.GetReferenceTrace(fit.wellIdx, flowIx, trace.begin());
    copy(trace.begin() + minFrame, trace.begin() + maxFrame, refFlows.begin_col(flowIx));
  }
  signal.resize(nFlows);
}

void KClass::InitialClassifyWell(Mask &mask,
                                 ZeromerDiff<double> &bg,
                                 std::vector<KeySeq> &keys, 
                                 //               std::vector<Traces> &traces,
                                 TraceStore<double> &traceStore,
                                 int nFlows,
                                 Col<double> &time,
                                 vector<KeyReporter<double> *>&report,
                                 double minSnr,
                                 std::vector<double> &dist,
                                 std::vector<std::vector<float> *> &distValues,
                                 KeyFit &fit) {
  if (mask[fit.wellIdx] & MaskExclude || mask[fit.wellIdx] & MaskWashout || mask[fit.wellIdx] & MaskPinned) {
    fit.ok = false;
    return;
  }
  FillInData(traceStore, nFlows, fit);
  mKClass.Classify(keys, bg, wellFlows, refFlows,
                   time, minSnr, fit, predicted);
  for (size_t rIx = 0; rIx < report.size(); rIx++) {
    report[rIx]->Report(fit, wellFlows, refFlows, predicted);
  }
}

void KClass::ClassifyWellKnownTau(Mask &mask,
                                  //                                  ZeromerDiff<double> &bg,
                                  ZeromerModelBulk<double> &bg,
                                  std::vector<KeySeq> &keys, 
                                 TraceStore<double> &traceStore,
                                 int nFlows,
                                  //                                  std::vector<Traces> &traces,
                                  Col<double> &time,
                                  Col<double> &incorp,
                                  vector<KeyReporter<double> *>&report,
                                  GridMesh<SampleQuantiles<double> > &emptyEstimates,
                                  double tauEEst,
                                  double minSnr,
                                  std::vector<double> &dist,
                                  std::vector<std::vector<float> *> &distValues,
                                  KeyFit &fit) {

  if (mask[fit.wellIdx] & MaskExclude  || mask[fit.wellIdx] & MaskWashout || mask[fit.wellIdx] & MaskPinned) {
    fit.ok = false;
    return;
  }
  FillInData(traceStore, nFlows, fit);

  incRef.set_size(wellFlows.n_rows); 
  if (incRef.n_rows > incorp.n_rows) {
    incRef.set_size(incorp.n_rows);
  }
  fill(incRef.begin(), incRef.end(), 0.0);
  copy(incorp.begin(), incorp.begin() + incRef.n_rows, incRef.begin());

  size_t row, col;
  traceStore.WellRowCol(fit.wellIdx, row, col);
  mKClass.ClassifyKnownTauE(keys, bg,
                            wellFlows, refFlows, time,
                            incRef, minSnr, 
                            tauEEst,fit, traceStore, predicted);
  for (size_t rIx = 0; rIx < report.size(); rIx++) {
    report[rIx]->Report(fit, wellFlows, refFlows, predicted);
  }
}
