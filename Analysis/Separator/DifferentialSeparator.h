/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DIFFERENTIALSEPARATOR_H
#define DIFFERENTIALSEPARATOR_H

#include <string>
#include <vector>
#include <fstream>
#include <armadillo>
#include "Mask.h"
#include "BFReference.h"
//#include "KeyClassifier.h"
#include "Separator.h"
#include "RegionAvgKeyReporter.h"
#include "TraceStore.h"
#include "TraceStoreCol.h"
#include "H5File.h"
#include "RawWells.h"
#include "AdvCompr.h"
#include "ImageNNAvg.h"
#include "PJobQueue.h"
#include "TauEFitter.h"
#define FRAMEZERO 0
#define FRAMELAST 100
#define FIRSTDCFRAME 3
#define LASTDCFRAME 12
#define BUFFEREXTRA 20
class TraceSaver;

/** Collection of different options for doing beadfind and separation. */
class DifSepOpt
{
  public:
    DifSepOpt()
    {
      predictRow = 0;
      predictHeight = 0;
      predictCol = 0;
      predictWidth = 0;

      maxKeyFlowLength = 7;
      flowOrder = "TACG";
      reportStepSize = 0;
      maxMad = 30;
      bfThreshold = .5;
      minSnr = 8;
      minBfGoodWells = 100;
      bfMeshStep = 50;
      clusterMeshStep = 50;
      clusterFineMeshStep = 10;
      t0MeshStep = 50;
      useMeshNeighbors = 1;
      tauEEstimateStep = 50;
      nCores = -1;

      minTauESnr = 6;
      sigSdMult = 6;
      doMeanFilter = true;
      doSigVarFilter = true;
      doMadFilter = true;
      doRecoverSdFilter  = true;
      doRemoveLowSignalFilter = true;
      doEmptyCenterSignal = false;
      regionXSize = 50;
      regionYSize = 50;
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

      //filterLagOneSD = false;
      smoothTrace = false;
      iqrMult = 3;
      tfFilterQuantile = .5;
      libFilterQuantile = .5;
      minTfPeakMax = 10.0f;
      minLibPeakMax = 10.0f;
      useProjectedCurve = true;
      outputDebug = 0;
      percentReference = .01;
      useSignalReference = 1;
      doSdat = false;
      sdatSuffix = "sdat";
      useSeparatorRef = false;
      isThumbnail = false;
      doComparatorCorrect = false;
      doGainCorrect = true;
      sdAsBf = true;
      bfMult = 1.0;
      aggressive_cnc = false;
      referenceStep = 50;
      referencePickStep = 25;
      blobFilter = true;
      blobFilterStep = 50;
      predictFlowStart = -1;
      predictFlowEnd = -1;
      gainMult = 1;
      skipBuffer = false;
      filterNoisyCols = "none";
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
    int clusterMeshStep;
    int clusterFineMeshStep;
    int t0MeshStep;
    int useMeshNeighbors;
    int tauEEstimateStep;
    int nCores;
    int minTauESnr;
    int referenceStep;
    int referencePickStep;
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
    //    bool filterLagOneSD;
    bool smoothTrace;
    float iqrMult;
    float tfFilterQuantile;
    float libFilterQuantile;
    bool useProjectedCurve;
    int outputDebug;
    float percentReference;
    int useSignalReference;
    bool doSdat;
    float minTfPeakMax,minLibPeakMax;
    int blobFilterStep;
    std::string sdatSuffix;
    bool useSeparatorRef;
    bool isThumbnail;
    bool doGainCorrect;
    bool doComparatorCorrect;
    float bfMult;
    bool sdAsBf;
    bool aggressive_cnc;
    bool blobFilter; 
    string doubleTapFlows;
    int predictFlowStart, predictFlowEnd;
    int predictRow,predictHeight,predictCol,predictWidth;
    string predictRegion; // row, height, col width
    int gainMult;
    bool skipBuffer;
  string  filterNoisyCols;
};

/**
 * Class that does beadfinding and classification of beads into
 * various keys. Designed to replace original Separator. Called differential
 * as fits a basic version of Todd's differential model to do classification.
 */
class DifferentialSeparator : public AvgKeyIncorporation
{

  public:
  enum FilterType {
    GoodWell,          // 0
    PinnedExcluded,    // 1 
    NotCompressable,   // 2
    LowTraceSd,        // 3
    BeadfindFiltered,  // 4
    RegionTraceSd,     // 5
    WellDevZeroNorm,   // 6
    NoisyColumn        // 7
  };

  static std::string NameForFilter(enum FilterType filter) {
    switch (filter) {
    case GoodWell:
      return "GoodWell";
    case PinnedExcluded :
      return "PinExclude";
    case NotCompressable :
      return "NotCompress";
    case LowTraceSd :
      return "LowTraceSd";
    case BeadfindFiltered :
      return "BeadfindFilt";
    case RegionTraceSd :
      return "RegionTraceSd";
    case WellDevZeroNorm :
      return "WellDevZeroNorm";
    case NoisyColumn :
      return "NoisyColumn";
    default:
      return "Unknown";
    }
  }        
    enum OutlierType
    {
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

    enum WellType
    {
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
      WellBadTrace, // 24
      WellNoisyColumn // 25
    };

    /** Fit gaussian mixture model to find empties and bead wells for region. */
    void ClusterRegion (int rowStart, int rowEnd,
                        int colStart, int colEnd,
                        float maxMad,
                        float minBeadSnr,
                        size_t minGoodWells,
                        vector<float> &bfMetric,
                        vector<KeyFit> &wells,
                        double trim,
                        bool doCenter,
                        MixModel &model);


    /** Load mask or make a mask with all empties (for cropped sets) */
    void LoadInitialMask(Mask *preMask, const std::string &maskFile, const std::string &imgFile, Mask &mask, int ignoreChecksumErrors = 0);

    void FilterNoisyColumns(int row_step, int col_step,
                            Mask &mask, DifSepOpt &opts,
                            std::vector<char> &filtered_wells);

    /** Calculate the noise per pixel */
    void CalculatePixelNoise(RawImage *raw, Mask &mask, 
                             int row_step, int col_step, 
                             std::vector<float> &pixel_sd);
    void FilterPixelSd(struct RawImage *raw, float min_val, vector<char> &well_filters);
        
    /** Make usual TCAG and ATCG keys. */
    void MakeStadardKeys (std::vector<KeySeq> &keys);

    /** Set keys to use. */
    void SetKeys (const std::vector<KeySeq> &_keys) {mKeys = _keys; }

    /** Set the keys from Analysis binary format. */
    void SetKeys (SequenceItem *seqList, int numSeqListItems, 
                  float minLibSnr, float minTfSnr,
                  float minLibPeak, float minTfPeak);
    

    /** Utility function to print keys to stdout. */
    void PrintKey (const KeySeq &k, int kIx);

    /** Don't do separation just clustering from beadfind statistic */
    void DoJustBeadfind (DifSepOpt &opts, vector<float> &bfMetric);

    void CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, const std::string &file);
    void CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq, Image &img);
    void CalcAcqT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq, const std::string &file);
    void CalcAcqT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq, Image &img, bool filt);
    void CalcRegionEmptyStat(H5File &h5File, GridMesh<MixModel> &mesh, TraceStore &store, 
                             const string &fileName, 
                             vector<int> &flows, Mask &mask);
    static void PrintVec(arma::Col<float> &vec);
    static void PrintWell(TraceStore &store, int well, int flow);    
    /** Do a beadfind/bead classification based on options passed in. */
    void FitTauE(DifSepOpt &opts, TraceStoreCol &traceStore, GridMesh<struct FitTauEParams> &emptyEstimates,
                 std::vector<char> &filteredWells, std::vector<float> &ftime, std::vector<int> &allZeroFlows, float *taub_est);
    void FitKeys(DifSepOpt &opts, GridMesh<struct FitTauEParams> &emptyEstimates, 
                 TraceStoreCol &traceStore, std::vector<KeySeq> &keys, 
                 std::vector<float> &ftime, TraceSaver &saver,
                 Mask &mask, std::vector<KeyFit> &wells);
    void FitKeys(PJobQueue &jQueue, DifSepOpt &opts, GridMesh<struct FitTauEParams> &emptyEstimates, 
                 TraceStoreCol &traceStore, std::vector<KeySeq> &keys, 
                 std::vector<float> &ftime, TraceSaver &saver,
                 Mask &mask, std::vector<KeyFit> &wells);
    void LoadKeyDats(PJobQueue &jQueue, TraceStoreCol &traceStore, 
                     vector<float> &bfMetric, DifSepOpt &opts, std::vector<float> &traceSd);
    void DoRegionClustering(DifSepOpt &opts, Mask &mask, vector<float> &bfMetric, float madThreshold,
                            std::vector<KeyFit> &wells, GridMesh<MixModel> &modelMesh);
    void ClusterIndividualWells(DifSepOpt &opts, Mask &bfMask, Mask &mask, TraceStoreCol &traceStore,
                                GridMesh<MixModel> &modelMesh, std::vector<KeyFit> &wells, 
                                std::vector<float> &confidence, std::vector<char> &clusters);
    void AssignAndCountWells(DifSepOpt &opts, std::vector<KeyFit> &wells, Mask &bfMask, 
                             std::vector<char> &filteredWells,int minLibPeak, int minTfPeak,
                             float sepRefSdThresh, float madThreshold );
    void ReduceMetric(std::vector<float> &metric, int col_ix, 
                      ChipReduction &reduce, const char *filtered_wells,
                      arma::Mat<float> &M);
    void SpatialSummary(const std::string &h5_file_name, const std::string &h5path, 
                        DifSepOpt &opts, Mask &mask, int x_step, int y_step);
    void HandleDebug(std::vector<KeyFit> &wells, DifSepOpt &opts, const std::string &h5SummaryRoot, 
                     TraceSaver &saver, Mask &mask, TraceStoreCol &traceStore, GridMesh<MixModel> &modelMesh);
    void OutputStats(DifSepOpt &opts, Mask &bfMask);
    void DoBeadfindFlowAndT0(DifSepOpt &opts, Mask &mask, const std::string &bfFile);
    int Run(DifSepOpt opts);
    void CalculateFrames(SynchDat &sdat, int &minFrame, int &maxFrame); 
    void FilterRegionBlobs(Mask &mask, int rowStart, int rowEnd, int colStart, int colEnd, int chipWidth,
                           arma::Col<float> &metric, vector<char> &filteredWells, int smoothWindow,
                           int filtWindow, float filtThreshold);
    /** Return mask used */
    Mask *GetMask() { return &mBfMask; }

    /** Get the average key signal for a region (for initializing bkmodels) */
    float *GetAvgKeySig (int region, int rStart, int rEnd, int cStart, int cEnd) {
      return mRegionIncorpReporter.GetAvgKeySig (region, rStart, rEnd, cStart, cEnd);
    }

    /** Get the average key signal length (for initializing bkmodels) */
    double GetAvgKeySigLen()  {
      return mRegionIncorpReporter.GetAvgKeySigLen();
    }

    /** Get the nucleotide incorporating start (in frames) */
    float GetStart (int region, int rStart, int rEnd, int cStart, int cEnd)  {
      return mRegionIncorpReporter.GetStart (region, rStart, rEnd, cStart, cEnd);
    }

    /** Get estimated t0 for a particular well. */
    std::vector<float> GetT0() { return mT0; }

    /** Get estimated t0 for a particular well. */
    float GetWellT0 (int wellIndex) { return mT0[wellIndex]; }

    void WellDeviation(TraceStoreCol &store,
                       int rowStep, int colStep,
                       vector<char> &filter,
                       vector<float> &mad);

    void WellDeviationRegion(TraceStoreCol &store,
                             int row_start, int row_end,
                             int col_start, int col_end,
                             int frame_start, int frame_end,
                             int flow_start, int flow_end,
                             float *mean, float *m2,
                             float *normalize,
                             float *summary,
                             vector<char> &filters,
                             vector<float> &mad);

    void RankWellsBySignal(int flow0, int flow1, TraceStore &store,
                           float iqrMult,
                           int numBasis,
                           Mask &mask,
                           int minWells,
                           int rowStart, int rowEnd,
                           int colStart, int colEnd,
                           std::vector<char> &filter,
                           std::vector<float> &mad);
    
    void CreateSignalRef(int flow0, int flow1, TraceStore &store,
                         int rowStep, int colStep,
                         float iqrMult,
                         int numBasis,
                         Mask &mask,
                         int minWells,
                         std::vector<char> &filter,
                         std::vector<float> &mad);

    void PickCombinedRank(vector<vector<float> > &mads,
                          vector<char> &filter, vector<char> &refWells,
                          int numWells,
                          int rowStart, int rowEnd, int colStart, int colEnd);

    /** Pick reference wells using a combined ranking of buffering and signal difference.*/ 
    void PickCombinedRank(vector<vector<float> > &mads,
                          int rowStep, int colStep,
                          float minPercent, int numWells,
                          vector<char> &filter, vector<char> &refWells);

    /** Pick flows for this key that have 1mer and 0mer with same nuc. */
    bool Find0merAnd1merFlows(KeySeq &key, TraceStore &store,
                              int &flow0mer, int &flow1mer);

    /** Find flow where all keys are 0 and 1 */
    bool FindCommon0merAnd1merFlows(std::vector<KeySeq> &key_vectors,
                                    TraceStore &store,
                                    int &flow0mer, int &flow1mer);

    bool FindKey0merAnd1merFlows(KeySeq &key,
                                 TraceStore &store,
                                 std::vector<int>  &flow0mer, std::vector<int> &flow1mer);

    /** 
     * Create our initial set of reference wells using the filters set up in the BFReference object
     * and additionally using the signal seen between flows if useKeySignal is true. The overall goal
     * is to pick wells that buffer the least and that are the most similar across a key 0mer and 1mer flow
     * at the same time the wells are close to what the other wells are doing and compress well as they 
     * should be smoothly changing. Each well gets a rank in these different metrics and the wells with the
     * lowest combined rank are chosen.
     *
     * In short to pick good empty wells that aren't outliers in some other way.
     */
    void RankReference(TraceStoreCol &store,
                       vector<float> &bfMetric, 
                       int rowStep, int colStep,
                       int useKeySignal,
                       float iqrMult,
                       int numBasis,
                       float minPercent,
                       Mask &mask,
                       int minWells,
                       vector<char> &filter,
                       vector<char> &refWells);

    static float IQR (SampleQuantiles<float> &s);

    double GetAvg1mer (int row, int col,
                       Mask &mask, enum MaskType type,
                       std::vector<KeyFit> &wells,
                       int distance);

    void CalcDensityStats (const std::string &prefix, Mask &mask, std::vector<KeyFit> &wells);

    int GetWellCount (int row, int col,
                      Mask &mask, enum MaskType type, int distance);

    /**
     * Return the fit from the ZeromerModelBulk for a particular well. NULL if
     * data not available (e.g. excluded wells, pinned wells, etc.)
     */
    float GetTauE (size_t wellIx) { return mWells[wellIx].tauE; }
    float GetTauB (size_t wellIx) { return mWells[wellIx].tauB; }

  private:

    void ClusterWells(DifSepOpt &opts, TraceStoreCol &traceStore, Mask &mask, Mask &bfMask, 
                      float madThreshold, GridMesh<MixModel> &modelMesh);
    void ClusterToSelectReference(DifSepOpt &opts, TraceStoreCol &traceStore,
                                  GridMesh<MixModel> &modelMesh, GridMesh<struct FitTauEParams> &emptyEstimates);
    void SetUp(DifSepOpt &opts, std::string &bfFile, arma::Col<int> &zeroFlows, 
               vector<int> &flowsAllZero, string &h5SummaryRoot);
    void SetupTraceStore(DifSepOpt &opts, TraceStoreCol &traceStore, std::vector<float> &ftime);
    RegionAvgKeyReporter<double> mRegionIncorpReporter;
    Mask mMask;
    Mask mBfMask;
    std::vector<KeySeq> mKeys;
    std::vector<float> mT0;
    vector<KeyFit> mWells;
    arma::Col<float> mTime;
    vector<char> mFilteredWells;
    vector<char> mRefWells;
    vector<int> mBFTimePoints;
    vector<float> mBfSdFrame;
    vector<float> mBfSSQ;
    vector<float> mAcqSSQ;
    vector<float> mBfMetric;
    vector<float> mTraceMad;
    vector<float> mColNoise;
    vector<float> mWellNoise;
    vector<vector<float> > mBfConfidence;
    std::map<std::string, Image *> mImageCache;
    std::vector<std::vector<float> > mEmptyMetrics;
    ClockTimer mTotalTimer;
    PJobQueue mQueue;
    size_t mNumWells;
    //    ImageNNAvg mImageNN;
};

#endif // DIFFERENTIALSEPARATOR_H

