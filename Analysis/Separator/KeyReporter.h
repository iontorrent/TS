/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYREPORTER_H
#define KEYREPORTER_H

#include <string>
#include "KeyClassifier.h"
#include "EvaluateKey.h"

template <class T>
class KeyReporter {
public: 

  virtual void Prepare() {}

  virtual void Report(const KeyFit &fit, 
		      const Mat<T> &wellFlows,
		      const Mat<T> &refFlows,
		      const Mat<T> &predicted) = 0;

  virtual void Finish() {}
};

#endif // KEYREPORTER_H
