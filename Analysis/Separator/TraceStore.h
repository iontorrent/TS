/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACESTORE_H
#define TRACESTORE_H

#include <vector>
#include <string>
#include <armadillo>
#include "Utils.h"
#include "IonErr.h"
#include "MathOptim.h"
#include "KeyClassifier.h"

/** Definititon of a binary map of key flows to pattern match. */
class KeySeq {
 public:
  KeySeq() {
    usableKeyFlows = 0;
    minSnr = 0;
  }

  std::string name;
  std::vector<int> flows;
  unsigned int usableKeyFlows;
  ivec zeroFlows;
  double minSnr;
};

/** Statistics about how well a particular key fits a well. */
class KeyFit {

 public:

  KeyFit() {
    wellIdx = -1;
    keyIndex = -1;
    snr = 0;
    mad = -1;
    ok = -1;
    bestKey = -1;
    sd = 0;
    onemerAvg = 0;
    onemerProjAvg = 0;
    traceMean = 0;
    traceSd = 0;
    projResid = 0;
    peakSig = 0;
    param.set_size(2);
    param.at(0) = 0;
    param.at(1) = 0;
    flag = -1;
    bfMetric = -1;
    bufferMetric = -1;
    goodLive = false;
    isRef = false;
  }
  
  int wellIdx;
  int8_t keyIndex;
  int8_t bestKey;
  Col<double> param;
  float snr;
  float mad;
  float sd;
  float onemerAvg;
  float onemerProjAvg;
  float traceMean;
  float traceSd;
  float projResid;
  float peakSig;
  int8_t ok;
  float bfMetric;
  float bufferMetric;
  int flag;
  bool goodLive;
  bool isRef;
};


/** 
 * Abstract interface to a repository for getting traces.
 */
template <typename _T>
class TraceStore {

 public:
  const static int TS_OK = 0;
  const static int TS_BAD_DATA = 1;
  const static int TS_BAD_REGION = 5;

  enum Nuc { A_NUC=0,C_NUC=1,G_NUC=2,T_NUC=3 };	
  /** Accessors. */
  virtual size_t GetNumFrames() = 0;
  virtual size_t GetNumWells() = 0;
  virtual size_t GetNumRows() = 0;	
  virtual size_t GetNumCols() = 0;
  virtual void SetKeyAssignments(const std::vector<char> &keyAssign) { mKeyAssign = keyAssign; }
  virtual int GetKeyAssignment(size_t wIx) { return mKeyAssign[wIx]; }
  virtual void SetKeys(const std::vector<KeySeq> &keys) { mKeys = keys; }
  virtual std::vector<KeySeq> GetKeys() { return mKeys; }
  virtual const std::string & GetFlowOrder() = 0;
  // @todo - fix this...
  static float WeightDist(float dist) {
    return ExpApprox(-1*dist/60.0);
  }
  virtual size_t GetNumNucs() { return 4; }
  virtual enum Nuc GetNucForFlow(size_t flowIx) { 
    const std::string &order = GetFlowOrder();
    char c = order.at(flowIx % order.size()); 
    c = toupper(c);
    switch (c) {
    case 'A' :
      return A_NUC;
    case 'C' :
      return C_NUC;
    case 'G' :
      return G_NUC;
    case 'T' :
      return T_NUC;
    default:
      ION_ABORT(ToStr("Don't recognize character: ") + c);
    }
    ION_ABORT("Should never get here..");
    return A_NUC;
  }

  virtual size_t GetFlowBuff() = 0;
  virtual double GetTime(size_t frame) = 0;
  virtual void SetTime(Col<double> &time)  = 0;
  virtual void SetSize(int frames) = 0;
  virtual void SetFlowIndex(size_t flowIx, size_t index) = 0;
  virtual bool HaveWell(size_t wellIx) = 0;
  virtual void SetHaveWellFalse(size_t wellIx) = 0;
  virtual void SetReference(size_t wellIx, bool isReference) = 0;
  virtual bool IsReference(size_t wellIx) = 0;
  virtual bool HaveFlow(size_t flowIx) = 0;
  virtual size_t WellIndex(size_t row, size_t col) {return row * GetNumCols() + col; }
  virtual void Dump(ofstream &out) = 0;
  static void WellRowCol(size_t idx, size_t nCols, size_t &row, size_t &col) { 
    row = idx / nCols;
    col = idx % nCols;
  }

  virtual void WellRowCol(size_t idx, size_t &row, size_t &col) { 
    WellRowCol(idx, GetNumCols(), row, col);
  }

  /* Getting the traces */
  virtual int GetTrace(size_t wellIx, size_t flowIx, typename std::vector<_T>::iterator traceBegin) = 0;
  virtual int GetTrace(size_t wellIx, size_t flowIx, typename arma::Col<_T>::col_iterator traceBegin) = 0;

  virtual int GetReferenceTrace(size_t wellIx, size_t flowIx, 
                                typename arma::Col<_T>::col_iterator traceBegin) = 0;
  
  virtual int PrepareReference(size_t flowIx) = 0;
  /* Setting the traces. */
  virtual int SetTrace(size_t wellIx, size_t flowIx, 
                       typename std::vector<_T>::iterator traceBegin, typename std::vector<_T>::iterator traceEnd ) = 0;
  virtual int SetTrace(size_t wellIx, size_t flowIx, 
                       typename arma::Col<_T>::col_iterator traceBegin, typename arma::Col<_T>::col_iterator traceEnd) = 0;
	
  virtual void SetT0(std::vector<float> &t0) = 0;

  virtual float GetT0(int idx) = 0;
  virtual void SetMeshDist(int size) = 0;
  virtual int GetMeshDist() = 0;
  virtual void SetTemp(Col<double> &col) { mTemp = col; }
  virtual void SetFirst() {
    Col<double> trace(GetNumFrames());
    GetTrace(0, 0, trace.begin());
    SetTemp(trace);
  }
  virtual void CheckFirst() {
    Col<double> trace(GetNumFrames());
    GetTrace(0, 0, trace.begin());
    assert(trace.size() == mTemp.size());
    for (size_t f = 0; f < GetNumFrames(); f++) {
      if (mTemp[f] != trace[f]) {
        assert(0);
      }
    }
             
  }
 private: 
  std::vector<char> mKeyAssign;
  std::vector<KeySeq> mKeys;
  arma::Col<double> mTemp;
};

#endif // TRACESTORE_H
