/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCLASSIFIER_H
#define KEYCLASSIFIER_H

#include <armadillo>
#include <vector>
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "ZeromerDiff.h"
#include "ZeromerModelBulk.h"
//#define FRAME_SIGNAL 20
using namespace arma;


/** Try fitting a set of keys and determine which one fits best. Replacing old separator code to  */
class KeyClassifier {

 public:

  /** Utility function to print out a column vector from arma. */
  static void PrintVec(const Col<float> &vec) {
    vec.raw_print();
  }

  static void PrintVec(const ivec &vec) {
    vec.raw_print();
  }

  /** Utility function to print out a matrix from arma. */
  static void PrintVec(const Mat<float> &m) {
    m.raw_print();
  }

  /** Project one column vector onto another. Used to see how a well projects onto 
      consensus 1mer incorporation. Incorporation must be unit vector I = I/norm(I,2) */
  static double GetProjection(const Col<float> &well, const Col<float> &incorporation) {
    Col<float>::fixed<1> val;
    val = (trans(well) * incorporation);
    return val.at(0); 
  }

  KeyClassifier() { 
    onemerSig.Init(100);
    zeromerSig.Init(100);
  }

  /** Classify a well as a particular key */
  void Classify(std::vector<KeySeq> &keys, 
                ZeromerDiff<float> &bg,
                Mat<float> &wellFlows,
                Mat<float> &refFlows,
                Col<float> &time,
                double minSnr,
                KeyFit &fit,
                Mat<float> &predicted);
	
  /** Classify a well using a regional estimate for tauE */
  void ClassifyKnownTauE(std::vector<KeySeq> &keys, 
                         ZeromerModelBulk<float> &bg,
                         Mat<float> &wellFlows,
                         Mat<float> &refFlows,
                         Col<float> &time,
                         Mat<float> *darkMatter,
                         Mat<float> *onemers,
                         size_t frameCandEnd,
                         double minSnr,
                         double tauE,
                         KeyFit &fit,
                         TraceStore &store,
                         Mat<float> &predicted);

 private:
  Col<float> param;
  Col<float> p;
  Col<float> diff;
  Col<float> signal;
  Col<float> projSignal;
  Col<float> weights;
  Mat<float> mPredictions;
  SampleQuantiles<float> onemerSig;
  SampleQuantiles<float> zeromerSig;
  //SampleStats<float> zeroStats;
  SampleQuantiles<float> zeroStats;
  SampleStats<float> mad;
  SampleStats<float> traceSd;  
  //  SampleStats<float> sigVar;  
  SampleQuantiles<float> sigVar;  
  SampleStats<float> onemerProj;  
  SampleQuantiles<float> onemerProjMax;  
  SampleStats<float> onemerIncorpMad;  
  ZeromerDiff<float> mBg;
  Col<float> trace;
  Col<float> traceIntegral;
  Col<float> bulk;
  Col<float> bulkTrace;
  Col<float> bulkIntegral;
  std::vector<double> mDist;
  std::vector<std::vector<double> *> mValues;
  
};

#endif // KEYCLASSIFIER_H

