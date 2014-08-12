/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMERMODEL_H
#define ZEROMERMODEL_H

#include <vector>
#include <algorithm>
#include <armadillo>
#include <string>
#include "KeyClassifier.h"
#include "TraceStore.h"
#include "PJobQueue.h"

using namespace arma;

/**
 * Interface for fitting zeromers.
 */
template <class T>
class ZeromerModel {

 public:

  virtual void SetTime(const Col<T> &time) = 0;

  virtual void FitWellZeromers(PJobQueue &jQueue,
                               TraceStore &traceStore,
                               std::vector<char> &keyAssignments,
                               Col<int> &zeroFlows,
                               std::vector<KeySeq> &keys) = 0;

  virtual int ZeromerPrediction(int wellIdx,
                                int flowIdx,
                                TraceStore &store,
                                const Col<T> &ref,
                                Col<T> &zeromer) = 0;

  virtual int GetNumModels() = 0;

  virtual bool HaveModel(size_t wellIdx) = 0;

};

#endif // ZEROMERMODEL_H
