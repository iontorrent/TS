/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef KEYCLASSIFIER_H
#define KEYCLASSIFIER_H

#include <armadillo>
#include <vector>
#include <string>
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "ZeromerDiff.h"
#include "ZeromerModelBulk.h"
#define FRAME_SIGNAL 20
using namespace arma;


/** Try fitting a set of keys and determine which one fits best. Replacing old separator code to  */
class KeyClassifier {

 public:

  /** Utility function to print out a column vector from arma. */
  static void PrintVec(const Col<double> &vec) {
    vec.raw_print();
  }

  static void PrintVec(const Col<float> &vec) {
    vec.raw_print();
  }

  static void PrintVec(const ivec &vec) {
    vec.raw_print();
  }

  /** Utility function to print out a matrix from arma. */
  static void PrintVec(const Mat<double> &m) {
    m.raw_print();
  }

  /** Project one column vector onto another. Used to see how a well projects onto 
      consensus 1mer incorporation. Incorporation must be unit vector I = I/norm(I,2) */
  static double GetProjection(const Col<double> &well, const Col<double> &incorporation) {
    Col<double>::fixed<1> val = (trans(well) * incorporation);
    return val.at(0); 
  }

  KeyClassifier() { 
    onemerSig.Init(100);
    zeromerSig.Init(100);
  }

  /** Classify a well as a particular key */
  void Classify(std::vector<KeySeq> &keys, 
                ZeromerDiff<double> &bg,
                Mat<double> &wellFlows,
                Mat<double> &refFlows,
                Col<double> &time,
                double minSnr,
                KeyFit &fit,
                Mat<double> &predicted);
	
  /** Classify a well using a regional estimate for tauE */
  void ClassifyKnownTauE(std::vector<KeySeq> &keys, 
                         ZeromerModelBulk<double> &bg,
                         Mat<double> &wellFlows,
                         Mat<double> &refFlows,
                         Col<double> &time,
                         const Col<double> &incorp,
                         double minSnr,
                         double tauE,
                         KeyFit &fit,
                         TraceStore<double> &store,
                         Mat<double> &predicted);


 private:
  Col<double> param;
  Col<double> p;
  Col<double> diff;
  Col<float> signal;
  Col<float> projSignal;
  SampleQuantiles<double> onemerSig;
  SampleQuantiles<double> zeromerSig;
  SampleStats<double> zeroStats;
  SampleStats<double> mad;
  SampleStats<double> traceSd;  
  SampleStats<double> sigVar;  
  SampleStats<double> onemerProj;  
  SampleQuantiles<double> onemerProjMax;  
  SampleStats<double> onemerIncorpMad;  
  ZeromerDiff<double> mBg;
  std::vector<double> mDist;
  std::vector<std::vector<double> *> mValues;
};

#endif // KEYCLASSIFIER_H

