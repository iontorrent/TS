/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "KClass.h"

void KClass::InitialClassifyWell
(Mask &mask,
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
  int nFrames = traceStore.GetNumFrames();
  if (wellFlows.n_rows != (size_t)nFrames || wellFlows.n_cols != (size_t)nFlows) {
    wellFlows.set_size(nFrames, nFlows);
    refFlows.set_size(nFrames, nFlows);
    predicted.set_size(nFrames, nFlows);
  }
  reference.resize(nFrames);
  signal.resize(nFlows);

  for (size_t flowIx = 0; flowIx < (size_t)nFlows; flowIx++) {
    traceStore.GetTrace(fit.wellIdx, flowIx, wellFlows.begin_col(flowIx));
    //@todo - does this have to be thread safe?
    traceStore.GetReferenceTrace(fit.wellIdx, flowIx, refFlows.begin_col(flowIx));
    //traces[flowIx].CalcMedianReference(fit.wellIdx, traces[flowIx].mGridMedian, dist, distValues, reference);
    //    copy(reference.begin(), reference.end(), refFlows.begin_col(flowIx));
  }
  mKClass.Classify(keys, bg,
                   wellFlows, refFlows,
                   time, minSnr, fit, predicted);
  // if (fit.wellIdx == 0) {
  //   KeyClassifier::PrintVec(fit.param);
  //   KeyClassifier::PrintVec(wellFlows);
  //   KeyClassifier::PrintVec(refFlows);
  //   KeyClassifier::PrintVec(predicted);
  // }
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
  int nFrames = traceStore.GetNumFrames();
  if (wellFlows.n_rows != (size_t)nFrames || wellFlows.n_cols != (size_t)nFlows) {
    wellFlows.set_size(nFrames, nFlows);
    refFlows.set_size(nFrames, nFlows);
    predicted.set_size(nFrames, nFlows);
  }
  reference.resize(nFrames);
  signal.resize(nFlows);

  fill(refFlows.begin(), refFlows.end(), 0);
  fill(wellFlows.begin(), wellFlows.end(), 0);
  for (size_t flowIx = 0; flowIx < (size_t)nFlows; flowIx++) {
    traceStore.GetTrace(fit.wellIdx, flowIx, wellFlows.begin_col(flowIx));
    //@todo - does this have to be thread safe?
    traceStore.GetReferenceTrace(fit.wellIdx, flowIx, refFlows.begin_col(flowIx));
    //traces[flowIx].CalcMedianReference(fit.wellIdx, traces[flowIx].mGridMedian, dist, distValues, reference);
    //    copy(reference.begin(), reference.end(), refFlows.begin_col(flowIx));
  }
  size_t row, col;
  traceStore.WellRowCol(fit.wellIdx, row, col);
  mKClass.ClassifyKnownTauE(keys, bg,
                            wellFlows, refFlows, time,
                            incorp, minSnr, 
                            tauEEst,fit, traceStore, predicted);
  for (size_t rIx = 0; rIx < report.size(); rIx++) {
    report[rIx]->Report(fit, wellFlows, refFlows, predicted);
  }
}
