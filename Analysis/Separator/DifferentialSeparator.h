/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFERENTIALSEPARATOR_H
#define DIFFERENTIALSEPARATOR_H

#include <string>
#include <vector>
#include <fstream>
#include "Mask.h"
#include "BFReference.h"
#include "KeyClassifier.h"
#include "Separator.h"
#include "RegionAvgKeyReporter.h"
#include "Traces.h"
#include "TraceStore.h"

#define FRAMEZERO 0
#define FRAMELAST 100
#define FIRSTDCFRAME 3
#define LASTDCFRAME 12
#define BUFFEREXTRA 20

/** Collection of different options for doing beadfind and separation. */
class DifSepOpt {
 public:
  DifSepOpt() {
    maxKeyFlowLength = 7;
    flowOrder = "TCAG";
    reportStepSize = 0; 
    maxMad = 30;
    bfThreshold = .5;
    minSnr = 3;
    minBfGoodWells = 80;
    bfMeshStep = 100;
    t0MeshStep = 100;
    useMeshNeighbors = 1;
    tauEEstimateStep = 100;
    nCores = -1;
    minTauESnr = 12;
    sigSdMult = 6;
    doMeanFilter = true;
    doSigVarFilter = true;
    doMadFilter = true;
    doRecoverSdFilter  = true;
    doRemoveLowSignalFilter = true;
    doEmptyCenterSignal = false;
    regionXSize = 100;
    regionYSize = 100;
    mask = NULL;
    justBeadfind = false;
    clusterTrim = .01;
    bfNeighbors = 3;
    samplingStep = 10;
    signalBased = true;
    help = false;
    ignoreChecksumErrors = false;
    minRatioLiveWell = .0001;
    noduds = false;
    filterLagOneSD = false;
    iqrMult = 3;
    tfFilterQuantile = .5;
    libFilterQuantile = .5;
    useProjectedCurve = true;
  }

  Mask *mask;
  int maxKeyFlowLength;
  string signalBf;
  string analysisDir;
  string resultsDir;
  string flowOrder;
  string wellsReportFile;
  int reportStepSize;
  string outData;
  string maskFile;
  double maxMad;
  double bfThreshold;
  double minSnr;
  size_t minBfGoodWells;
  int bfMeshStep;
  int t0MeshStep;
  int useMeshNeighbors;
  int tauEEstimateStep;
  int nCores;
  int minTauESnr;
  double sigSdMult;
  bool doMeanFilter;
  bool doSigVarFilter;
  bool doMadFilter;
  bool doRecoverSdFilter;
  bool doRemoveLowSignalFilter;
  bool doEmptyCenterSignal;
  string bfType;
  string bfDat;
  string bfBgDat;
  bool justBeadfind;
  int bfNeighbors;
  bool help;
  double clusterTrim;
  int samplingStep;
  int regionXSize;
  int regionYSize;
  bool signalBased;
  int ignoreChecksumErrors;
  double minRatioLiveWell;
  bool noduds;
  bool filterLagOneSD;
  float iqrMult;
  float tfFilterQuantile;
  float libFilterQuantile;
  bool useProjectedCurve;
};

/** 
 * Class that does beadfinding and classification of beads into
 * various keys. Designed to replace original Separator. Called differential
 * as fits a basic version of Todd's differential model to do classification.
 */
class DifferentialSeparator : public AvgKeyIncorporation { 

 public: 

  enum OutlierType {
    SdNoKeyHigh,
    SdKeyLow,
    MadHigh,
    BfNoKeyHigh,
    BfKeyLow,
    LibKey,
    TFKey,
    EmptyWell,
    LowKeySignal,
    KeyLowSignalFilt
  };

  enum WellType {
    WellNone,   // 0
    WellEmpty,  // 1
    WellBead,   // 2
    WellLive,   // 3
    WellDud,    // 4
    WellAmbiguous, // 5
    WellTF,  // 6 
    WellLib,  // 7 
    WellPinned,  // 8
    WellIgnore,  // 9 
    WellWashout, // 10
    WellExclude, // 11
    WellKeypass, // 12
    WellMeanFilter, // 13
    WellSdFilter, // 14 
    WellEmptyVar, // 15
    WellRecoveredEmpty, // 16 
    WellLowSignal,  // 17 
    WellNoTauE,  // 18
    WellMadHigh, // 19
    WellBadFit,  // 20 
    WellBfBad,   // 21
    WellBfBufferFilter, // 22
    WellEmptySignal, // 23
    WellBadTrace // 24
  };

  /** Fit gaussian mixture model to find empties and bead wells for region. */
  void ClusterRegion(int rowStart, int rowEnd,
		     int colStart, int colEnd,
		     float maxMad,
		     float minBeadSnr,
		     size_t minGoodWells,
		     const BFReference &reference,
		     const vector<KeyFit> &wells,
		     double trim,
		     MixModel &model);
	
  /** Make usual TCAG and ATCG keys. */
  void MakeStadardKeys(std::vector<KeySeq> &keys);

  /** Load mask or make a mask with all empties (for cropped sets) */
	void LoadInitialMask(Mask *preMask, const std::string &maskFile, const std::string &imgFile, Mask &mask, int ignoreChecksumErrors = 0);

  /** Set keys to use. */
  void SetKeys(const std::vector<KeySeq> &_keys) {keys = _keys;	}

  /** Set the keys from Analysis binary format. */
  void SetKeys(SequenceItem *seqList, int numSeqListItems, float minLibSnr, float minTfSnr);

  /** Utility function to print keys to stdout. */
  void PrintKey(const KeySeq &k, int kIx);

  /** Don't do separation just clustering from beadfind statistic */
  void DoJustBeadfind(DifSepOpt &opts, BFReference &reference);

  void LoadKeyDats(TraceStore<double> &traceStore, DifSepOpt &opts);

  /** Do a beadfind/bead classification based on options passed in. */
  int Run(DifSepOpt opts);

  void SetReportSet(int rows, int cols,
                    const std::string &wellsReportFile,
                    int reportStepSize) {
    reportSet.SetSize(rows, cols);
    if (wellsReportFile.empty()) {
      reportSet.SetStepSize(reportStepSize);
    }
    else {
      reportSet.ReadSetFromFile(wellsReportFile, 0);
    }
  }

  /** Return mask used */
  Mask *GetMask() {
    return &bfMask;
  }

  /** Get the average key signal for a region (for initializing bkmodels) */
  float *GetAvgKeySig(int region, int rStart, int rEnd, int cStart, int cEnd) {
    return mRegionIncorpReporter.GetAvgKeySig(region, rStart, rEnd, cStart, cEnd);
  }

  /** Get the average key signal length (for initializing bkmodels) */
  double GetAvgKeySigLen() {
    return mRegionIncorpReporter.GetAvgKeySigLen();
  }

  /** Get the nucleotide incorporating start (in frames) */
  int GetStart(int region, int rStart, int rEnd, int cStart, int cEnd) {
    return mRegionIncorpReporter.GetStart(region, rStart, rEnd, cStart, cEnd);
  }
	
  /** Get estimated t0 for a particular well. */
  std::vector<float> GetT0() { return t0; }

  /** Get estimated t0 for a particular well. */
  float GetWellT0(int wellIndex) {
    return t0[wellIndex];
  }
  
  float LowerQuantile(SampleQuantiles<float> &s);

  float IQR(SampleQuantiles<float> &s);

  void DetermineBfFile(const std::string &resultsDir, bool &signalBased,					    const std::string &bfType, const string &bfDat,
                       const std::string &bfBgDat,
                       std::string &bfFile, std::string &bfFile2, std::string &bfBkgFile);

  void PredictFlow(const std::string &datFile, 
		   const std::string &outFile, int ignoreChecksumErrors, DifSepOpt &opts,
                   TraceStore<double> &store,
                   ZeromerModelBulk<double> &zModelBulk);
  
  bool InSpan(size_t rowIx, size_t colIx,
	      const std::vector<int> &rowStarts,
	      const std::vector<int> &colStarts,
	      int span);


  double GetAvg1mer(int row, int col,
		    Mask &mask, enum MaskType type,
		    std::vector<KeyFit> &wells,
		    int distance);

  void CalcDensityStats(const std::string &prefix, Mask &mask, std::vector<KeyFit> &wells);

  int GetWellCount(int row, int col,
		   Mask &mask, enum MaskType type, int distance);

  void DumpDiffStats(Traces &traces, std::ofstream &o);

  void PinHighLagOneSd(Traces &traces, float iqrMult);

  void CheckFirstAcqLagOne(DifSepOpt &opts);

  /** 
   * Return the fit from the ZeromerModelBulk for a particular well. NULL if 
   * data not available (e.g. excluded wells, pinned wells, etc.)
   */
  const KeyBulkFit *GetBulkFit(size_t wellIx) { return zModelBulk.GetKeyBulkFit(wellIx); }

 private:

  void OutputOutliers(DifSepOpt &opts, TraceStore<double> &store,
                      ZeromerModelBulk<double> &bg,
                      const vector<KeyFit> &wells,
                      double sdNoKeyHighT, double sdKeyLowT,
                      double madHighT, double bfNoKeyHighT, double bfKeyLowT,
                      double lowKeySignalT);

  void OutputOutliers(TraceStore<double> &store,
                      ZeromerModelBulk<double> &bg,
                      const vector<KeyFit> &wells,
                      int outlierType,
                      const vector<int> &outputIdx,
                      std::ostream &traceOut,
                      std::ostream &refOut,
                      std::ostream &bgOut
                      );

  void OutputWellInfo(TraceStore<double> &store,
                      ZeromerModelBulk<double> &bg,
                      const vector<KeyFit> &wells,
                      int outlierType,
                      int wellIdx,
                      std::ostream &traceOut,
                      std::ostream &refOut,
                      std::ostream &bgOut);


  RegionAvgKeyReporter<double> mRegionIncorpReporter;
  ReportSet reportSet;
  Mask mask;
  Mask bfMask;
  std::vector<char> keyAssignments;
  std::vector<KeySeq> keys;
  std::vector<float> t0;
  ZeromerModelBulk<double> zModelBulk;
  vector<KeyFit> wells;
  Col<double> mTime;
};

#endif // DIFFERENTIALSEPARATOR_H

