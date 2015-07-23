/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef TRACECLASSIFIER_H
#define TRACECLASSIFIER_H

#include "PinnedInFlow.h"
#include "Mask.h"
#include "SynchDat.h"
#include <vector>
#include <iostream>


class TraceClassifier
{
 public:
  TraceClassifier(Region *region,PinnedInFlow *pinnedInFlow,Mask *bfmask,Image *img,int flow,int imgFrames,MaskType referenceMask,std::vector<float>& t0_map,std::string flowsOpt="",int model=1);
  ~TraceClassifier() {};
  int fitEmptyTrace();
  int computeClusterMean(int HiLo);
  void copyAvgTrace(float *bPtr);
  void copyAvgTrace(std::vector<float>& bPtr);

private:
  // pointers passed to the constructor
  Region *region;
  PinnedInFlow *pinnedInFlow;
  Mask *bfmask;
  Image *img;
  int flow;
  int imgFrames;
  MaskType referenceMask;
  float *t0_map;
  std::string flowsOpt;
  int mModel;

  // for the --bkg_avgEmpty-flows option
  bool mDoAllFlows;
  bool mForceAllFlows;
  std::vector<int> flows_on;
  std::vector<float> mFstat;
  int parse_flowsOpt(const std::string &flowsOpt, std::vector<int>& flows_on);
  bool flow_is_on(int flow);
  bool needTraceClassifier(int flow);
  bool noEmptyWells() {return (mNumEmptyWells <= 0 ? true:false);}
  bool notEnoughEmptyWells() {return (mNumEmptyWells < mNumEmptyWells_min ? true:false);}
  //bool notEnoughEmptyWells() {return (true);} // off: force emptyTrace->GenerateAverageEmptyTrace() to be called
  bool needTopPicks();
  float compute_avgTrace_snr();
  float compute_avgTrace_cv();
  float computeKurtosis();
  float computeMinFs(const std::vector<int>& peaks,std::vector<int>& valleys);
  float computeVariance(const std::vector<int>& v);
  float anova(const std::vector<int>& borders1, const std::vector<int>& borders2);
  bool delta_too_low() {return (mDelta<deltaMin?true:false);}
  bool cv_too_low() {return (mCV<cvMin?true:false);}
  bool rms_too_low() {return (mRMS<rmsMin?true:false);}
  bool snr_too_low() {return (mSNR<snrMin?true:false);}
  bool kurt_too_high() {return (mKurt>kurtMax?true:false);}
  bool fstat_too_small(float fs) {return ((fs> 0.0 && fs<mMinFs)?true:false);}
  float get_delta() {return mDelta;}
  float get_cv() {return mCV;}
  float get_rms() {return mRMS;}
  float get_snr() {return mSNR;}
  float get_kurtosis() {return mKurt;}

  // for the --bkg_avgEmpty-model option
  int mNumWellsInRegion;
  int mNumEmptyWells;
  int mNumEmptyWells_min;
  int mNumEmptyWells_min2;
  int mNumEmptyWells_min3;
  int mNumTrueEmpties;
  int mNumTrueEmpties_min;


  //std::vector<char> emptyWells;
  std::vector<float> hi;
  std::vector<float> lo;
  std::vector<float> dy;
  std::vector<float> dYY;
  std::vector<float> trace_sum;
  std::vector<float> workTrace;
  std::vector<float> avgTrace;
  std::vector<float> avgTrace_old;
  std::vector<bool> picks;
  float pick_ratio;
  float mTotalLo;
  float mSNR;
  float snrMin;
  float mDelta;
  float deltaMin;
  float mCV;
  float cvMin;
  float mRMS;
  float rmsMin;
  float mKurt;
  float kurtMax;
  bool mKurtAvailable;
  float mMinFs;
  std::vector<bool>mTrueEmpties;
  int setTrueEmpties(float *sig, int nSig, float thresh);

  // finding and averaging empty traces
  std::vector<bool>mNoiseTraces;
  bool isNoiseTrace();

  bool isEmptyWell(int ax, int ay);
  void copy_ShiftTrace_To_WorkTrace(int ax, int ay, int iWell);
  void copyToWorkTrace(float *tmp_shifted);
  void sumWorkTrace();
  void loWorkTrace();
  void hiloWorkTrace();
  void zeroWorkTrace();
  void compute_dYY();
  int generateLowTrace();
  int generateAvgTrace(bool emptyonly=true);
  void calcAvgTrace(int nTotal);
  void setAvgTraceDefault();
  void copyLowTrace();
  void findTopPicks();
  void setPicks(bool flag);
  float mean_picked(std::vector<float>& data);

  int mHiLo;
  int clusterTraces(bool emptyonly=true);
  int cluster_regression(float *sig ,bool emptyonly=true);
  float regress_workTrace();
  float computeKurtosis(float *sig, int nSig);


  // fitTraces
  int mNoiseMax;
  int mSigFrame_start;
  int mSigFrame_end;
  int mSigFrame_win;
  int mSigFrame_start_max;
  int mSigFrame_tail_min;
  float mBaselineFraction;
  std::vector<bool> baseframes;
  int fitBaseline(bool emptyonly=true);
  void findBaseFrames();

  // clustering & thresholding
  std::string mThresholdMethod;
  int mMinFinalPeaks;
  int mNumBins;
  int mMaxNumPeaks;
  int mThreshIdx;
  float mThreshold;
  float get_threshold() {return mThreshold;}
  float mTrueRatio;
  int mTruePercent;
  int mTruePercent_min;
  float mCumHist_min;
  int passMinCDF(int mid);
  int passMinCDF(int mid, std::vector<int>& valleys);

  std::vector<float> bins;
  std::vector<int> hist;
  std::vector<float> cumhist;

  int mValley2CallClassifier;
  std::vector<int> mGaps;
  std::vector<int> mPeaks;
  std::vector<int> mValleys;
  std::vector<int> mFinalPeaks;
  std::vector<int> mFinalValleys;

  void makeHistogram(float *sig, int nSig, int nBins);
  float findThreshold(float *sig, int nSig, const char *method, bool emptyonly=true);
  int threshold_BMT_method(int order=1);
  int threshold_PVT_method(bool emptyonly=true);
  int averageTrueEmpties(float *sig, int nSig, float thresh, bool emptyonly);
  bool hasNoEmpties(int nTrueEmpties, int nEmpties);
  bool hasEnoughEmpties(int nTrueEmpties, int nEmpties);
  bool useFinalValley(int nValleys) ;

  template<class T> void findLocalMin(const std::vector<T>& v, std::vector<int>& valleys);
  template<class T> void findLocalMax(const std::vector<T>& v, std::vector<int>& peaks);
  void findGaps(const std::vector<int>& peaks, std::vector<int>& gaps);
  int getLastGapSize() { return mGaps[mGaps.size()-1];}
  int mergePeaks(const std::vector<int>& peaks, std::vector<int>& newPeaks);
  int reducePeaks(const std::vector<int>& peaks, std::vector<int>& newPeaks, std::vector<int>& valleys, std::vector<int>& newValleys);
  void findPeakBoarders(const std::vector<int>&valleys,int peak,std::vector<int>& boarders);
  void findClosestNeighbors(const std::vector<int>&peaks,int peak,std::vector<int>&neighbors);
  int findValleyBeforePeak(const std::vector<int>&valleys,int peak);
  int findValleyAfterPeak(const std::vector<int>&valleys,int peak);
  int findLowestHist(int n1, int n2, std::vector<int>& grounds);
  int findLowerHist(int n1, int n2, int maxHeight);
  int findFirstGround(const std::vector<int>& finalPeaks);
  int findFinalValleys(const std::vector<int>& finalPeaks, const std::vector<int>& valleys, std::vector<int>&finalValleys);
  int findValley_FWHM(const std::vector<int>& valleys, const std::vector<int>& finalPeaks, bool emptyonly=true);
  int findFirstValley(const std::vector<int>& valleys, const std::vector<int>& finalPeaks, std::vector<int>& finalValleys);
  int findFirstValley(const std::vector<int>& valleys, int peak);
  int findCenterValley(const std::vector<int>& valleys, const std::vector<int>& finalPeaks, std::vector<int>& finalValleys);
  int findLastValley(const std::vector<int>& valleys, const std::vector<int>& finalPeaks, std::vector<int>& finalValleys);
  int findLastValley(const std::vector<int>& valleys, int peak, int peak2);
  void findNeighborsToElimiate(const std::vector<int>& newPeaks, std::vector<bool>& available, int refPeak, std::vector<int>& neigobors);
  void sort_peaks_by_height(const std::vector<int>& peaks, std::vector<int>& sortedPeaks, bool ascending=true);
  void removePeak_lowestFs(const std::vector<int>& peaks, std::vector<int>& newPeaks, float minFs, const std::vector<int>& valleys, std::vector<int>& newValleys);
  void removePeakZero(std::vector<int>& peaks);

  bool avgCluster(int HiLo=0, bool emptyonly=true);
  void copyToAvgTrace(float *bPtr);
  void copyToAvgTrace(std::vector<float>& bPtr);

  // debugging
  std::string val2str(int val);
  std::string val2str(size_t val);
  std::string val2str(float val);
  std::string flowRegionStr();
  std::string get_decision_metrics();
  //std::string formatVector(std::vector<int>& peaks, std::string msg="");
  //std::string formatVector(std::vector<float>& peaks, std::string msg="");
  //std::string formatVector(float *v, int sz, std::string msg);
  std::string formatPeakHist(const std::vector<int>& peaks, std::string msg="");

  // obsolete
  //void copySingleTrace(int offset,float *tmp_shifted);
  //void copyAllEmptyTraces(); // don't use, too much memory required
  //void get_HiLo_fromEmptyWells();

};


#endif // TRACECLASSIFIER_H
