/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stddef.h>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>

#include "DifferentialSeparator.h"
#include "TraceStore.h"
#include "TraceSaver.h"
#include "KeySummaryReporter.h"
#include "AvgKeyReporter.h"
#include "AvgKeyIncorporation.h"
#include "Image.h"
#include "OptArgs.h"
#include "PJobQueue.h"
#include "Utils.h"
#include "HandleExpLog.h"
#include "IonErr.h"
#include "TraceStoreCol.h"
#include "EvaluateKey.h"
#include "TauEFitter.h"
#include "ReservoirSample.h"
#include "BfMetric.h"
#include "IonH5Arma.h"
#include "T0Calc.h"
#include "Stats.h"
#include "ComparatorNoiseCorrector.h"
#include "CorrNoiseCorrector.h"
#include "BkgTrace.h"
#include "RawWells.h"
#include "crop/Acq.h"
#include "PairPixelXtalkCorrector.h"

#define DIFFSEP_ERROR 3
// Amount in frames back from estimated t0 to use for dc offset estimation 
// @hack - Note this must be kept in synch with VFC_T0_OFFSET in BkgMagicDefines.h
#define T0_LEFT_OFFSET 4 
#define T0_RIGHT_OFFSET 20
#define T0_WINDOW_END 30
#define MIN_SD 9 // represents a nuc step with less than 35 counts
#define MIN_BF_SD 100 // represents a nuc step with less than 35 counts
#define T0_OFFSET_LEFT_TIME 0.0f
#define T0_OFFSET_RIGHT_TIME 1.3f // seconds
#define BF_NN_AVG_WINDOW 50
#define BF_THUMBNAIL_SIZE 100
#define MIN_LIB_PEAK 10
#define MIN_SAMPLE_TAUE_STATS 200
#define PCA_COMP_GRID_SIZE 100
#define RESCUE_SNR_THRESH 2
#define RESCUE_PEAK_THRESH 20
#define WELL_DEV_FRAME_START_OFFSET 4
#define WELL_DEV_FRAME_END_OFFSET 4
#define REF_SAMPLE_SIZE 1000
#define RELIABLE_PEAK_HEIGHT 40
using namespace std;

#define MAD_STAT 0
#define KEY_PEAK_STAT 1
#define BUFFER_STAT 2
#define TAUB_STAT 3
#define TAUE_STAT 4
#define KEY_SD_STAT 5
#define SNR_STAT 6
#define SIG_STAT 7
#define NUM_KEY_STATS 8


/** 
 * Utility class for keeping track of some well statistics.
 */
class WellSetKeyStats {

public:

  /** Identifier and size of pool to keep as sample. */
  WellSetKeyStats(const std::string &name, int n) {
    SetName(name);
    Init(n);
  }

  void SetName(const std::string &name) { m_stat_name = name; }

  int NumSeen() { return m_quantiles[0].GetNumSeen(); }

  void Init(int n) {
    m_quantiles.resize(NUM_KEY_STATS);
    for (size_t i = 0; i < NUM_KEY_STATS; i++) {
      m_quantiles[i].Init(n);
    }
  }
  
  void AddWell(KeyFit &fit) {
    if (isfinite(fit.mad)) { m_quantiles[MAD_STAT].AddValue(fit.mad); }
    if (isfinite(fit.peakSig)) { m_quantiles[KEY_PEAK_STAT].AddValue(fit.peakSig); }
    if (isfinite(fit.bufferMetric)) { m_quantiles[BUFFER_STAT].AddValue(fit.bufferMetric); }
    if (isfinite(fit.tauB)) { m_quantiles[TAUB_STAT].AddValue(fit.tauB); }
    if (isfinite(fit.tauE)) { m_quantiles[TAUE_STAT].AddValue(fit.tauE); }
    if (isfinite(fit.sd)) { m_quantiles[KEY_SD_STAT].AddValue(fit.sd); }
    if (isfinite(fit.snr)) { m_quantiles[SNR_STAT].AddValue(fit.snr); }
    if (isfinite(fit.onemerAvg)) { m_quantiles[SIG_STAT].AddValue(fit.onemerAvg); }
  }

  static void PrintHeader(FILE *out) {
    fprintf(stdout, "  name    median        iqr      .05    .25    .50    .75    .95\n");
    fprintf(stdout, "  ----    ------        ---   ------ ------ ------ ------ ------\n");
  }

  void ReportStats(FILE *out) {
    fprintf(out, "%s key stats from %d wells:\n", m_stat_name.c_str(), m_quantiles[0].GetNumSeen());
    PrintHeader(out);
    fprintf(out, "   snr    %6.2f +/- %6.2f   %6.2f %6.2f %6.2f %6.2f %6.2f\n", m_quantiles[SNR_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[SNR_STAT]), 
            m_quantiles[SNR_STAT].GetQuantile(.05),
            m_quantiles[SNR_STAT].GetQuantile(.25),
            m_quantiles[SNR_STAT].GetQuantile(.5),
            m_quantiles[SNR_STAT].GetQuantile(.75),
            m_quantiles[SNR_STAT].GetQuantile(.95));
    fprintf(out, "   peak   %6.2f +/- %6.2f   %6.2f %6.2f %6.2f %6.2f %6.2f\n", m_quantiles[KEY_PEAK_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[KEY_PEAK_STAT]), 
            m_quantiles[KEY_PEAK_STAT].GetQuantile(.05),
            m_quantiles[KEY_PEAK_STAT].GetQuantile(.25),
            m_quantiles[KEY_PEAK_STAT].GetQuantile(.5),
            m_quantiles[KEY_PEAK_STAT].GetQuantile(.75),
            m_quantiles[KEY_PEAK_STAT].GetQuantile(.95));
    fprintf(out, "   mad    %6.2f +/- %6.2f   %6.2f %6.2f %6.2f %6.2f %6.2f\n", m_quantiles[MAD_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[MAD_STAT]), 
            m_quantiles[MAD_STAT].GetQuantile(.05),
            m_quantiles[MAD_STAT].GetQuantile(.25),
            m_quantiles[MAD_STAT].GetQuantile(.5),
            m_quantiles[MAD_STAT].GetQuantile(.75),
            m_quantiles[MAD_STAT].GetQuantile(.95));
    fprintf(out, "   taub   %6.2f +/- %6.2f   %6.2f %6.2f %6.2f %6.2f %6.2f\n", m_quantiles[TAUB_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[TAUB_STAT]), 
            m_quantiles[TAUB_STAT].GetQuantile(.05),
            m_quantiles[TAUB_STAT].GetQuantile(.25),
            m_quantiles[TAUB_STAT].GetQuantile(.5),
            m_quantiles[TAUB_STAT].GetQuantile(.75),
            m_quantiles[TAUB_STAT].GetQuantile(.95));
    fprintf(out, "   taue   %6.2f +/- %6.2f   %6.2f %6.2f %6.2f %6.2f %6.2f\n", m_quantiles[TAUE_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[TAUE_STAT]), 
            m_quantiles[TAUE_STAT].GetQuantile(.05),
            m_quantiles[TAUE_STAT].GetQuantile(.25),
            m_quantiles[TAUE_STAT].GetQuantile(.5),
            m_quantiles[TAUE_STAT].GetQuantile(.75),
            m_quantiles[TAUE_STAT].GetQuantile(.95));
    fprintf(out, "   sig    %6.1f +/- %6.1f   %6.1f %6.1f %6.1f %6.1f %6.1f\n", m_quantiles[SIG_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[SIG_STAT]), 
            m_quantiles[SIG_STAT].GetQuantile(.05),
            m_quantiles[SIG_STAT].GetQuantile(.25),
            m_quantiles[SIG_STAT].GetQuantile(.5),
            m_quantiles[SIG_STAT].GetQuantile(.75),
            m_quantiles[SIG_STAT].GetQuantile(.95));
    fprintf(out, "   sd     %6.1f +/- %6.1f   %6.1f %6.1f %6.1f %6.1f %6.1f\n", m_quantiles[KEY_SD_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[KEY_SD_STAT]), 
            m_quantiles[KEY_SD_STAT].GetQuantile(.05),
            m_quantiles[KEY_SD_STAT].GetQuantile(.25),
            m_quantiles[KEY_SD_STAT].GetQuantile(.5),
            m_quantiles[KEY_SD_STAT].GetQuantile(.75),
            m_quantiles[KEY_SD_STAT].GetQuantile(.95));
    fprintf(out, "   bf     %6.0f +/- %6.0f   %6.0f %6.0f %6.0f %6.0f %6.0f\n", m_quantiles[BUFFER_STAT].GetMedian(), 
            DifferentialSeparator::IQR(m_quantiles[BUFFER_STAT]), 
            m_quantiles[BUFFER_STAT].GetQuantile(.05),
            m_quantiles[BUFFER_STAT].GetQuantile(.25),
            m_quantiles[BUFFER_STAT].GetQuantile(.5),
            m_quantiles[BUFFER_STAT].GetQuantile(.75),
            m_quantiles[BUFFER_STAT].GetQuantile(.95));
    
  }

  std::string m_stat_name;
  std::vector<SampleQuantiles<float> > m_quantiles;
};

void OutputFitSummary(DifSepOpt &opts, std::vector<KeyFit> &wells, GridMesh<FitTauEParams> &emptyEstimates, int &gotKeyCount) {
  SampleQuantiles<float> tauBQ(10000);
  SampleQuantiles<float> tauEQ(10000);
  SampleQuantiles<float> shiftQ(10000);
  for (size_t idx = 0; idx < emptyEstimates.GetNumBin(); idx++) {
    struct FitTauEParams &param = emptyEstimates.GetItem(idx);
    if (param.converged) {
      shiftQ.AddValue(param.ref_shift);
      tauEQ.AddValue(param.taue);
    }
  }
  for (size_t idx = 0; idx < wells.size(); idx++) {
    if (wells[idx].keyIndex == 0) {
      tauBQ.AddValue(wells[idx].tauB);
    }
    if (wells[idx].keyIndex >= 0  && wells[idx].mad < opts.maxMad) { gotKeyCount++; }
  }
  cout << "TauE Dist is: " << tauEQ.GetMedian() << " +/- " <<  DifferentialSeparator::IQR (tauEQ) << endl;
  cout << "TauB Dist is: " << tauBQ.GetMedian() << " +/- " <<  DifferentialSeparator::IQR (tauBQ) << endl;
  cout << "Shift Dist is: " << shiftQ.GetMedian() << " +/- " <<  DifferentialSeparator::IQR (shiftQ) << endl;
}

void CncProcessImage(Image &img, bool is_thumbnail, bool cnc_correct, Mask *cncMask, int region_size_x, int region_size_y, bool aggressive_cnc) {
  if (!(img.raw->imageState & IMAGESTATE_ComparatorCorrected) &&
      cnc_correct) {
    ComparatorNoiseCorrector cnc;
    if (is_thumbnail) {
      cnc.CorrectComparatorNoiseThumbnail(img.raw, cncMask, region_size_x, region_size_y, false);
    }
    else {
      cnc.CorrectComparatorNoise(img.raw, cncMask, false, aggressive_cnc);
    }
  }
}

void CalcImageGain(Image *img, Mask *mask, char *bad_wells, int row_step, int col_step, ImageNNAvg *imageNN) {
  // Use bfimg to calculate gain if necessary
  if (ImageTransformer::gain_correction == NULL) {
    //    ImageTransformer::CalculateGainCorrectionFromBeadfindFlow ( false, img, *mask);
    ImageTransformer::GainCalculationFromBeadfindFasterSave(img->raw, mask, bad_wells, row_step, col_step, imageNN);
  }
}

void GainCorrectImage(bool gain_correct, Image &img) {
  if(gain_correct && !(img.raw->imageState & IMAGESTATE_GainCorrected) &&
     ImageTransformer::gain_correction != NULL) {
    ImageTransformer::GainCorrectImage(img.raw);
  }
}

// Don't forget to clean up with img.Close()
void OpenAndProcessImage(const char *name, char *results_dir, bool ignore_checksum, 
                         bool gain_correct, Mask *mask, bool filt_pinned, Image &img,
                         bool col_pair_pixel_xtalk_correct,
                         float pair_xtalk_fraction,
                         bool corr_noise_correct, int thumbnail) {
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (ignore_checksum);
  bool loaded = img.LoadRaw(name);
  if (!loaded) { ION_ABORT ("Couldn't load file: " + ToStr(name)); }
  const RawImage *raw = img.GetImage(); 
  if (filt_pinned) {
    img.FilterForPinned (mask, MaskEmpty, false);  
  }

  ImageTransformer::XTChannelCorrect (img.raw, results_dir);

  if(ImageTransformer::PCATest[0]) {
    AdvComprTest(name,&img,ImageTransformer::PCATest,false );
  }
  else{
	  if( col_pair_pixel_xtalk_correct ){
	      PairPixelXtalkCorrector xtalkCorrector;
	      xtalkCorrector.Correct(img.raw, pair_xtalk_fraction);
	  }
    if (corr_noise_correct){
      CorrNoiseCorrector rnc;
      rnc.CorrectCorrNoise(img.raw,3,thumbnail);
    }
  }

  img.SetMeanOfFramesToZero (1,3);


  GainCorrectImage(gain_correct, img);
}

void CountReference(const string &s, vector<char> &filter) {
  vector<size_t> counts(10);
  for (size_t i = 0; i < filter.size(); i++) {
    int val = (int)filter[i];
    val = min(counts.size() -1, (size_t)val);
    counts[val]++;
  }
  cout << s << endl;
  cout << "Filtered well counts: " << endl;
  for (size_t i = 0; i < counts.size(); i++) {
    cout << DifferentialSeparator::NameForFilter((enum DifferentialSeparator::FilterType)i) << ":\t" << counts[i] << endl;
  }
}

void ZeroFlows(std::vector<KeySeq> &keys, DifSepOpt &opts, Col<int> &zeroFlows, vector<int> &flowsAllZero) {
  for (int i = 0; i < opts.maxKeyFlowLength; i++) {
    bool allZero = true;
    for (size_t kIx = 0; kIx < keys.size(); kIx++) {
      if ((int)keys[kIx].usableKeyFlows <= i || keys[kIx].flows[i] != 0) {
        allZero = false;
      }
    }
    if (allZero) {
      flowsAllZero.push_back(i);
    }
  }
  zeroFlows.resize(flowsAllZero.size());
  copy(flowsAllZero.begin(), flowsAllZero.end(), zeroFlows.begin());
  ION_ASSERT(zeroFlows.n_rows > 0, "Must have 1 all zeros flow in key");
}

void DumpStep(const char *name, const int flowIx, const int w, const int h, DifSepOpt *opts) {
    Image img;
    bool loaded = img.LoadRaw(name);
    if (!loaded) { ION_ABORT ("Couldn't load file: " + ToStr(name)); }

    string nucStepDir = opts->nucStepDir;
    char nucChar = opts->flowOrder[flowIx];
    int nSample=100000;
    img.DumpDcOffset(nSample, nucStepDir, nucChar, flowIx);

    size_t loadMinFlows = max (9, opts->maxKeyFlowLength+2);
    Mask mask(w, h);
    PinnedInFlow pinnedInFlow( &mask, loadMinFlows );
    pinnedInFlow.Initialize(&mask);

    // Set region width & height
    int rWidth=50;
    int rHeight=50;

    // Lower left corner of the region should equal 0 modulus xModulus or yModulus
    int xModulus=50;
    int yModulus=50;

    vector<string> regionName;
    vector<int> regionStartCol;
    vector<int> regionStartRow;
    vector<int> regionWidth;
    vector<int> regionHeight;

    const RawImage *rawImg = img.GetImage();

    // region choice based on chip type
    if ( ChipIdDecoder::IsProtonChip())
    {
      // Proton chips
      regionName.push_back ( "inlet" );
      regionStartCol.push_back ( xModulus * floor ( 0.1 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );

      regionName.push_back ( "middle" );
      regionStartCol.push_back ( xModulus * floor ( 0.5 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );

      regionName.push_back ( "outlet" );
      regionStartCol.push_back ( xModulus * floor ( 0.9 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );
    }
    else
    {
      // PGM chips
      regionName.push_back ( "inlet" );
      regionStartCol.push_back ( xModulus * floor ( 0.9 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.9 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );

      regionName.push_back ( "middle" );
      regionStartCol.push_back ( xModulus * floor ( 0.5 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );

      regionName.push_back ( "outlet" );
      regionStartCol.push_back ( xModulus * floor ( 0.1 * ( float ) rawImg->cols / ( float ) xModulus ) );
      regionStartRow.push_back ( yModulus * floor ( 0.1 * ( float ) rawImg->rows / ( float ) yModulus ) );
      regionWidth.push_back ( rWidth );
      regionHeight.push_back ( rHeight );
    }

    for ( unsigned int iRegion=0; iRegion<regionStartRow.size(); iRegion++ )
    {
        img.DumpStep(
                    regionStartCol[iRegion],
                    regionStartRow[iRegion],
                    regionWidth[iRegion],
                    regionHeight[iRegion],
                    regionName[iRegion],
                    nucChar,
                    nucStepDir,
                    &mask,
                    &pinnedInFlow,
                    flowIx
                    );
    }
}

/** 
 * Job class for multithreading the evaluation of different
 * keys. Basically just popluate a struct and then call Run() 
*/
class EvalKeyJob : public PJob {

public:
  EvalKeyJob() {
    mRowStart = mRowEnd = mColStart = mColEnd = 0;
    mMask = NULL;
    mOpts = NULL;
    mFilteredWells = NULL;
    mWells = NULL;
    mKeys = NULL;
    mFTime = NULL;
    mSaver = NULL;
    mTraceStore = NULL;
    mEmptyEstimates = NULL;
    mKeySumReport = NULL;
    mAvgReport = NULL;
    mDefaultParam = NULL;
  }

  void Init(int rowStart, int rowEnd, int colStart, int colEnd, Mask *mask,
            DifSepOpt *opts, std::vector<char> *filteredWells, std::vector<KeyFit> *wells,
            std::vector<KeySeq> *keys, std::vector<float> *ftime, TraceSaver *saver,
            TraceStoreCol *traceStore, 
            GridMesh<struct FitTauEParams> *emptyEstimates,
            KeySummaryReporter<double> *keySumReport,
            AvgKeyReporter<double> *avgReport,
            struct FitTauEParams *defaultParam) {
    mRowStart = rowStart;
    mRowEnd = rowEnd;
    mColStart = colStart;
    mColEnd = colEnd;
    mMask = mask;
    mOpts = opts;
    mFilteredWells = filteredWells;
    mWells = wells;
    mKeys = keys;
    mFTime = ftime;
    mSaver = saver;
    mTraceStore = traceStore;
    mEmptyEstimates = emptyEstimates;
    mKeySumReport = keySumReport;
    mAvgReport = avgReport;
    mDefaultParam = defaultParam;
  }

  virtual void Run() {
    ClockTimer keyTimer;
    int numWells = mMask->W() * mMask->H();

    EvaluateKey evaluator;
    evaluator.m_doing_darkmatter = true;
    evaluator.m_peak_signal_frames = true;
    evaluator.m_integration_width = true;
    evaluator.m_normalize_1mers = true;
    
    int usable_flows = 0;
    for (size_t i = 0; i < mKeys->size(); i++) {
      usable_flows = max(usable_flows, (int)mKeys->at(i).usableKeyFlows);
    }

    std::vector<double> taue_dist(7);
    std::vector<struct FitTauEParams *> taue_values(7);
    std::vector<int> flow_order(mTraceStore->GetNumFlows(),-1);
    for (size_t i = 0; i < mTraceStore->GetNumFlows(); i++) {
      flow_order[i] = mTraceStore->GetNucForFlow(i);
    }
    int query_row = (mRowStart + mRowEnd) / 2;
    int query_col = (mColStart + mColEnd) / 2;
    mEmptyEstimates->GetClosestNeighbors(query_row, query_col, mOpts->useMeshNeighbors * 2, taue_dist, taue_values);
    struct FitTauEParams mean_taue;
    mean_taue.taue = 0;
    mean_taue.ref_shift = 0;
    int taue_count = 0;
    for (size_t i = 0; i < taue_values.size(); i++) {
      if (taue_values[i]->converged > 0) {
        mean_taue.taue += taue_values[i]->taue;
        mean_taue.ref_shift += taue_values[i]->ref_shift;
        taue_count++;
      }
    }
    if (taue_count > 0) {
      mean_taue.taue /= taue_count;
      mean_taue.ref_shift /= taue_count;
    }
    else {
      mean_taue.taue = mDefaultParam->taue;
      mean_taue.ref_shift = mDefaultParam->ref_shift;
    }

    double taue_weight = 0;
    evaluator.SetSizes(mRowStart, mRowEnd,
                       mColStart, mColEnd,
                       0, usable_flows,
                       0, mTraceStore->GetNumFrames());
    
    if (mOpts->outputDebug > 3) {
      evaluator.m_debug = true;
    }
    evaluator.m_flow_order = flow_order;
    evaluator.m_doing_darkmatter = true;
    evaluator.m_peak_signal_frames = true;
    evaluator.m_integration_width = true;
    //    evaluator.m_use_projection = true;
    evaluator.SetUpMatrices(*mTraceStore, mMask->W(), 0, 
                            mRowStart, mRowEnd,
                            mColStart, mColEnd,
                            0, usable_flows,
                            0, mTraceStore->GetNumFrames());
    evaluator.FindBestKey(mRowStart, mRowEnd,
                          mColStart, mColEnd,
                          0, mTraceStore->GetNumFrames(),
                          0, mMask->W(), &(*mFTime)[0],
                          *mKeys, mean_taue.taue, mean_taue.ref_shift,
                          &(*mFilteredWells)[0],
                          (*mWells));
    mKeySumReport->Report(evaluator);
    mAvgReport->Report(evaluator);
    if (mOpts->outputDebug > 3) {
      mSaver->StoreResults(mRowStart, mRowEnd, mColStart, mColEnd, 0, usable_flows, evaluator);
    }
  }

  int mRowStart, mRowEnd, mColStart, mColEnd;
  Mask *mMask;
  DifSepOpt *mOpts;
  std::vector<char> *mFilteredWells;
  std::vector<KeyFit> *mWells;
  std::vector<KeySeq> *mKeys;
  std::vector<float> *mFTime;
  TraceSaver *mSaver;
  TraceStoreCol *mTraceStore;
  GridMesh<struct FitTauEParams> *mEmptyEstimates;
  KeySummaryReporter<double> *mKeySumReport;
  AvgKeyReporter<double> *mAvgReport;
  struct FitTauEParams *mDefaultParam;
};

/** 
 * Job class for multithreading reading dat files from disk, processing them and 
 * loading them into data structure.
 */
class LoadDatJob : public PJob {
public:
  LoadDatJob() {
    mCNCMask = NULL;
    mFlowIx = 0;
    mMask = NULL;
    mFilteredWells = NULL;
    mT0 = NULL;
    mOpts = NULL;
    mTraceStore  = NULL;
    mTraceSdMin = NULL;
  }

  void Init(const std::string &_fileName, int _flowIx, TraceStoreCol *_traceStore, float *_traceSdMin,
	    Mask *_mask, Mask *_cmpMask,  std::vector<char> *_filteredWells, std::vector<float> *_t0, DifSepOpt *_opts) {
    mFileName = _fileName;
    mFlowIx = _flowIx;
    mTraceStore = _traceStore;
    mTraceSdMin = _traceSdMin;
    mMask = _mask;
    mCNCMask = _cmpMask;
    mFilteredWells = _filteredWells;
    mT0 = _t0;
    mOpts = _opts;
  }

  static void LoadDat(const std::string &fileName, DifSepOpt *opts, TraceStoreCol *traceStore, 
                      std::vector<float> *t0, Mask *mask, Mask *cncMask, int flowIx, 
                      std::map<std::string, Image *> *imageCache,
                      std::vector<char> *filteredWells, float *traceSdMin) {
    ClockTimer datTimer;
    Image imgLoad;
    Image *img;
    if (imageCache->find(fileName) != imageCache->end()) {
      img = imageCache->at(fileName);
      //ProcessImage(*img, mask, opts->doGainCorrect, (char *)opts->resultsDir.c_str());
    }
    else {
      OpenAndProcessImage(fileName.c_str(), (char *)opts->resultsDir.c_str(), opts->ignoreChecksumErrors, 
                          opts->doGainCorrect, mask, true, imgLoad, opts->col_pair_pixel_xtalk_correct, 
                          opts->pair_xtalk_fraction, opts->corr_noise_correct, opts->isThumbnail);
      img = &imgLoad;
    }
    CncProcessImage(*img, opts->isThumbnail, opts->doComparatorCorrect, cncMask, opts->clusterMeshStepX, opts->clusterMeshStepY, opts->aggressive_cnc);
    img->SetMeanOfFramesToZero (1,3);

#ifdef DEBUG_IMAGE_CORR_ISSUES

        char newName[1024];
        char *nptr,*ptr=(char *)fileName.c_str();
        sprintf(newName,"%s_bfproc",ptr);
        {
        	short int *rawPtr = img->raw->image;
        	int frameStride=img->raw->cols*img->raw->rows;

        	for(int frame=0;frame<img->raw->frames;frame++){
        		for(int idx=0;idx<frameStride;idx++){
        			*rawPtr += 8192;
        			rawPtr++;
        		}
        	}
        }
        Acq saver;
        saver.SetData ( img );
        saver.WriteVFC(newName, 0, 0,img->raw->cols, img->raw->rows);
        printf("wrote %s\n",newName);

        {
        	short int *rawPtr = img->raw->image;
        	int frameStride=img->raw->cols*img->raw->rows;

        	for(int frame=0;frame<img->raw->frames;frame++){
        		for(int idx=0;idx<frameStride;idx++){
        			*rawPtr -= 8192;
        			rawPtr++;
        		}
        	}
        }
#endif


    RawImage *raw = img->raw;
    // datTimer.PrintMicroSecondsUpdate(stdout, "Dat Timer: Image Processing complete");
    
    // 3. Get an average t0 for wells that we couldn't get a good local number
    double t0_global_sum = 0;
    int t0_global_count = 0;
    float *__restrict t0_global_start = &(*t0)[0];
    float *__restrict t0_global_end = t0_global_start + t0->size();
    while(t0_global_start != t0_global_end) {
      if (*t0_global_start > 0) {
        t0_global_sum += *t0_global_start;
        t0_global_count++;
      }
      t0_global_start++;
    }
    int meanT0 = 0;
    if (t0_global_count > 0) {
      meanT0 = (int)(t0_global_sum / t0_global_count + .5);
    }

    // Loop through and load each region with appropriate data
    GridMesh<int> t0Mesh;
    //    t0Mesh.Init (mask->H(), mask->W(), LOAD_MESH_SIZE, LOAD_MESH_SIZE);
    t0Mesh.Init (mask->H(), mask->W(), opts->clusterMeshStepY, opts->clusterMeshStepX);
    for (size_t binIx = 0; binIx < t0Mesh.GetNumBin(); binIx++) {
      // Get the region
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      t0Mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      int width = colEnd - colStart;
      int dc_offset[width];

      // Calculate our region t0 minimum // starting frame
      float t0_est = raw->frames;
      for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
        float *__restrict t0_start = &(*t0)[0] + rowIx * raw->cols + colStart;
        float *__restrict t0_end = t0_start + width;
        while(t0_start != t0_end) {
          if (*t0_start > 0) {
            t0_est = min(t0_est, *t0_start);
          }
          t0_start++;
        }
      }
      if (t0_est == raw->frames) {
        t0_est = meanT0;
      }
      int t0_uncomp_frame = (int) t0_est;
      int t0_frame = 0;
      for (t0_frame = 0; t0_frame < raw->frames; t0_frame++) {
        if (t0_uncomp_frame <= raw->compToUncompFrames[t0_frame]) {
          break;
        }
      }
      t0_frame = min(t0_frame, raw->frames - (T0_RIGHT_OFFSET+2));

      // copy data starting at t0 frame, each column is contiguous so copy in column stripes
      for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
        memset(dc_offset, 0, sizeof(int) * width);
        int dc_count = 0;
        int start_frame = t0_frame;
        int end_frame = t0_frame + T0_LEFT_OFFSET;
        // determine dc offset with first few frames
        for (int frameIx = start_frame; frameIx < end_frame; frameIx++) {
          int store_frame = frameIx - start_frame;
          const int16_t *__restrict img_start = raw->image + rowIx * raw->cols + colStart + raw->frameStride * frameIx;
          const int16_t *__restrict img_end = img_start + width;
          int *__restrict dc_ptr = &dc_offset[0];
          dc_count++;
          while (img_start != img_end) {
            *dc_ptr++ += *img_start++;
          }
        }
        int *__restrict dc_start = &dc_offset[0];
        int *__restrict dc_end = dc_start + width;
        while (dc_start != dc_end) {
          *dc_start = (int)( *dc_start / (float)dc_count+.5f);
          dc_start++;
        }

        // Load the frames subtracting off the dc offset
        end_frame = t0_frame + T0_RIGHT_OFFSET;
        for (int frameIx = start_frame; frameIx < end_frame; frameIx++) {
          int store_frame = frameIx - start_frame;
          float *__restrict t0_start = &(*t0)[0] + rowIx * raw->cols + colStart;
          const int16_t *__restrict img_start = raw->image + rowIx * raw->cols + colStart + raw->frameStride * frameIx;
          const int16_t *__restrict next_img_start = raw->image + rowIx * raw->cols + colStart + raw->frameStride * (frameIx+1);
          const int16_t *__restrict next_next_img_start = raw->image + rowIx * raw->cols + colStart + raw->frameStride * (frameIx+2);
          const int16_t *__restrict img_end = img_start + width;
          int16_t *__restrict out_start = traceStore->GetMemPtr() + store_frame * traceStore->mFlowFrameStride + 
            traceStore->mFrameStride * flowIx + rowIx * raw->cols + colStart;
          int *__restrict dc_ptr = &dc_offset[0];
          while (img_start != img_end) {
            float loc_shift = *t0_start - t0_uncomp_frame;
            if (loc_shift < 1) {
              *out_start = (int16_t) (*img_start - *dc_ptr + loc_shift * (*next_img_start - *img_start) + .5f);
            }
            else {
              *out_start = (int16_t) (*next_img_start - *dc_ptr + (loc_shift - 1.0f) * (*next_next_img_start - *next_img_start) + .5f);
            }
            //            *out_start = (int16_t) (*img_start - *dc_ptr);
            t0_start++;
            dc_ptr++;
            img_start++;
            out_start++;
            next_img_start++;
          }
        }
      }
    }

    img->Close();
  }

  virtual void Run() {
    LoadDat(mFileName, mOpts, mTraceStore, mT0, mMask, mCNCMask, mFlowIx, &mImageCache, mFilteredWells, mTraceSdMin);
    size_t num_wells = mMask->H() * mMask->W();
  }
  std::map<std::string, Image *> mImageCache;
  std::string mFileName;
  int mFlowIx;
  TraceStoreCol *mTraceStore;
  Mask *mMask;
  Mask *mCNCMask;
  std::vector<char> *mFilteredWells;
  float *mTraceSdMin;
  std::vector<float> *mT0;
  DifSepOpt *mOpts;
};

void DifferentialSeparator::CalculatePixelNoise(RawImage *raw, Mask &mask, 
                                                int row_step, int col_step, 
                                                std::vector<float> &pixel_sd) {
  std::vector<char> filtered_avg(mask.H() * mask.W());
  std::fill(filtered_avg.begin(), filtered_avg.end(), 0);
  int fsize = sizeof(float);
  int vsize = VEC8F_SIZE_B;
  size_t n_well = raw->rows * raw->cols;
  size_t n_frame = raw->frames;
  float *__restrict sum_stats = (float *) memalign(vsize, fsize * 2 * n_well);
  assert(sum_stats);
  float *__restrict image_nn_sub = (float *) memalign(vsize, fsize * n_frame * n_well);
  assert(image_nn_sub);
  float *__restrict trace_min =  (float *) memalign(vsize, fsize * n_well);
  assert(trace_min);
  int *__restrict trace_min_frame = (int *) memalign(vsize, sizeof(int) * n_well);
  assert(trace_min_frame);
  memset(trace_min_frame, 0, sizeof(int) * n_well);

  ImageNNAvg avg_nn;
  avg_nn.Init(raw->rows, raw->cols, n_frame);
  avg_nn.CalcCumulativeSum(raw->image, &mask, &mFilteredWells[0]);

  float *__restrict nn_start = image_nn_sub;
  int16_t *__restrict image_start = raw->image;
  int16_t *__restrict image_end = image_start + (raw->rows * raw->cols * n_frame);
  while(image_start != image_end) {
    *nn_start++ = *image_start++;
  }

  avg_nn.CalcNNAvgAndMinFrame(row_step, col_step, 
                              5, 5,
                              trace_min, trace_min_frame,
                              raw->image, false);
  const float *__restrict nn_avg = avg_nn.GetNNAvgImagePtr();
  memset(sum_stats, 0, sizeof(float) * n_well * 2);
  for (size_t fIx = 0; fIx < n_frame; fIx++) {
    const float * __restrict frame_nn_start = nn_avg + fIx * n_well;
    const float * __restrict frame_nn_end = frame_nn_start + n_well;
    float * __restrict image_start = image_nn_sub + fIx * n_well;
    while (frame_nn_start != frame_nn_end) {
      *image_start -= *frame_nn_start;
      image_start++;
      frame_nn_start++;
    }
  }

  float *__restrict m2 = (float *) memalign(vsize, fsize * n_well);
  float *__restrict mean = (float *) memalign(vsize, fsize * n_well);
  memset(m2, 0, sizeof(float) * n_well);
  memset(mean, 0, sizeof(float) * n_well);
  for (size_t f_ix = 0; f_ix < n_frame; f_ix++) {
    float *__restrict mean_start = mean;
    float *__restrict m2_start = m2;
    float *__restrict diff_start = image_nn_sub + f_ix * n_well;
    float *__restrict diff_end = diff_start + n_well;
    float delta;
    int n = f_ix + 1;
    while (diff_start != diff_end) {
      delta = (*diff_start) - *mean_start;
      *mean_start += delta / n;
      *m2_start += delta * (*diff_start - *mean_start);
      diff_start++;
      mean_start++;
      m2_start++;
    }
  }
  pixel_sd.resize(n_well);
  std::fill(pixel_sd.begin(), pixel_sd.end(), 0.0f);
  for (size_t w_ix = 0; w_ix < n_well; w_ix++) {
    pixel_sd[w_ix] = sqrt(m2[w_ix]/n_frame);
  }
  free(sum_stats);
  free(image_nn_sub);
  free(trace_min);
  free(trace_min_frame);
  free(mean);
  free(m2);
}

void DifferentialSeparator::FilterNoisyColumns(int row_step, int col_step,
                                               Mask &mask, DifSepOpt &opts,
                                               std::vector<char> &filtered_wells) {
  char buffer[MAX_PATH_LENGTH];
  const char *prerun_prefix = "prerun_000";
  size_t n_well = mask.H() * mask.W();
  //  vector<float> mWellNoise(n_well, 0.0f);
  mWellNoise.resize(n_well);
  std::fill(mWellNoise.begin(), mWellNoise.end(), 0.0f);
  vector<float> current_error(n_well, 0.0f);
  int n_prerun = 4;
  for (int pre_ix = 0; pre_ix < n_prerun; pre_ix++) {
    snprintf(buffer, sizeof(buffer), "%s/%s%d.%s", opts.resultsDir.c_str(), prerun_prefix, pre_ix, opts.datPostfix.c_str());
    Image img;
    OpenAndProcessImage(buffer, (char *)opts.resultsDir.c_str(), opts.ignoreChecksumErrors,
                        opts.doGainCorrect, &mask, true, img, opts.col_pair_pixel_xtalk_correct, 
                        opts.pair_xtalk_fraction, opts.corr_noise_correct, opts.isThumbnail);
    CalculatePixelNoise(img.raw, mask, row_step, col_step, current_error);
    for (size_t w_ix = 0; w_ix < n_well; w_ix++) {
      mWellNoise[w_ix] += current_error[w_ix];
    }
  }

  size_t n_comp = mask.W();
  // usually just one region per chip but for thumbnail top and bottom are separate
  if (opts.isThumbnail) { n_comp = n_comp * 2; } 
  SampleQuantiles<float> m_quantiles(n_comp);  
  for (size_t w_ix = 0; w_ix < n_well; w_ix++) {
    mWellNoise[w_ix] /= n_prerun;
  }
  mColNoise.resize(n_comp);
  std::fill(mColNoise.begin(), mColNoise.end(), 0.0f);
  std::vector<int> col_count(n_comp, 0);
  int comp_offset = 0;
  int half_rows = mask.H() / 2;
  // Bottom half in case thumbnail
  for (int col_ix = 0; col_ix < mask.W(); col_ix++) {
    for (int row_ix = 0; row_ix < half_rows; row_ix++) {
      int well_ix = row_ix * mask.W() + col_ix;
      if ((mask[well_ix] & MaskPinned) == 0) {
        mColNoise[col_ix + comp_offset] += mWellNoise[well_ix];
        col_count[col_ix + comp_offset]++;
      }
    }
  }
  // Top half in case thumbnail
  if (opts.isThumbnail) { comp_offset = mask.W(); }
  for (int col_ix = 0; col_ix < mask.W(); col_ix++) {
    for (int row_ix = half_rows; row_ix < mask.H(); row_ix++) {
      int well_ix = row_ix * mask.W() + col_ix;
      if ((mask[well_ix] & MaskPinned) == 0) {
        mColNoise[col_ix + comp_offset] += mWellNoise[well_ix];
        col_count[col_ix + comp_offset]++;
      }
    }
  }
  
  for (size_t comp_ix = 0; comp_ix < n_comp; comp_ix++) {
    if (col_count[comp_ix] > 0) {
      mColNoise[comp_ix] = mColNoise[comp_ix] / col_count[comp_ix];
      m_quantiles.AddValue(mColNoise[comp_ix]);
    }
    else {
      mColNoise[comp_ix] = -1.0f;
    }
  }
  
  float threshold = 0;
  if (opts.filterNoisyCols == "strict") {
    threshold = m_quantiles.GetMedian() + (m_quantiles.GetMedian() - m_quantiles.GetQuantile(.01));
  }
  else if (opts.filterNoisyCols == "medium") {
    threshold = m_quantiles.GetQuantile(.75f) + 1.5 * DifferentialSeparator::IQR(m_quantiles);
  }
  else if (opts.filterNoisyCols == "loose") {
    threshold = m_quantiles.GetQuantile(.75f) + 3.0 * DifferentialSeparator::IQR(m_quantiles);
  }
  else {
    ION_ABORT("Don't recognize filterNoisyCols option '" + opts.filterNoisyCols + "'");
  }
  int filt_count = 0;
  comp_offset = 0;
  //  Again doing the top and bottom separately here 
  for (int col_ix = 0; col_ix < mask.W(); col_ix++) {
    if (mColNoise[col_ix + comp_offset] >= threshold) {
      filt_count++;
      for (int row_ix = 0; row_ix < half_rows; row_ix++) {
        int well_ix = row_ix * mask.W() + col_ix;
        filtered_wells[well_ix] = NoisyColumn;
      }
    } 
  }
  if (opts.isThumbnail) { comp_offset = mask.W(); }
  for (int col_ix = 0; col_ix < mask.W(); col_ix++) {
    if (mColNoise[col_ix + comp_offset] >= threshold) {
      if (opts.isThumbnail) { filt_count++; }
      for (int row_ix = half_rows; row_ix < mask.H(); row_ix++) {
        int well_ix = row_ix * mask.W() + col_ix;
        filtered_wells[well_ix] = NoisyColumn;
      }
    } 
  }
  
  fprintf(stdout, "column_threshold '%s' of %.3f filtered %d of %d (%.2f%%) %6.1f %6.1f %6.1f %6.1f %6.1f\n", 
          opts.filterNoisyCols.c_str(),
          threshold,
          filt_count,
          (int) n_comp,
          100.0f * filt_count / n_comp,
          m_quantiles.GetQuantile(.05),
          m_quantiles.GetQuantile(.25),
          m_quantiles.GetQuantile(.5),
          m_quantiles.GetQuantile(.75),
          m_quantiles.GetQuantile(.95));

}

void DifferentialSeparator::FilterPixelSd(struct RawImage *raw, float min_val, vector<char> &well_filters) {
  float *mean = (float *)memalign(32, raw->frameStride * sizeof(float));
  float *m2 =  (float *)memalign(32, raw->frameStride * sizeof(float));
  float *chip_sd =  (float *)memalign(32, raw->frameStride * sizeof(float));
  float *chip_good_sd =  (float *)memalign(32, raw->frameStride * sizeof(float));
  memset(mean, 0, raw->frameStride * sizeof(float));
  memset(m2, 0, raw->frameStride * sizeof(float));
  memset(chip_sd, 0, raw->frameStride * sizeof(float));
  memset(chip_good_sd, 0, raw->frameStride * sizeof(float));
  int count = 0;

  // Loop through frame by frame to calculate the sd, could probably vectorize this
  for (int frame_ix = 0; frame_ix < raw->frames; frame_ix++) {
    count++;
    int16_t *img_start = raw->image + frame_ix * raw->frameStride;
    int16_t *img_end = img_start + raw->frameStride;
    float *mean_start = mean;
    float *m2_start = m2;
    float delta;
    while (img_start != img_end) {
      delta = *img_start - *mean_start;
      *mean_start += delta/count;
      *m2_start += delta * (*img_start - *mean_start);
      img_start++;
      mean_start++;
      m2_start++;
    }
  }
  printf("Finished summaries\n");
  // Grab the sd for all wells and for good wells
  char *filter_start = &well_filters[0];
  char *filter_end = filter_start + well_filters.size();
  float *m2_start = m2;
  float *chip_sd_start = chip_sd;
  float *chip_good_sd_start = chip_good_sd;

  // summarize the variance for each well
  while (filter_start != filter_end) {
    float val = sqrt(*m2_start/count);
    *chip_sd_start  = val;
    if (*filter_start == 0 && val >= min_val) {
      *chip_good_sd_start++ = val;
    }
    chip_sd_start++;
    m2_start++;
    filter_start++;
  }
  
  printf("Finished sd\n");
  // calculate our threshold
  int good_count = chip_good_sd_start - chip_good_sd;
  if (good_count > MIN_SAMPLE_TAUE_STATS) {
    std::sort(chip_good_sd, chip_good_sd_start);
    float q25 = ionStats::quantile_sorted(chip_good_sd, good_count, .25);
    float q75 = ionStats::quantile_sorted(chip_good_sd, good_count, .75);
    float sd_threshold = q25 - 3 * (q75-q25);
    sd_threshold = std::max(min_val, sd_threshold);
    printf("finished thresholds\n");
    // do the filtering and place back the results
    filter_start = &well_filters[0];
    filter_end = filter_start + well_filters.size();
    chip_sd_start = chip_sd;
    int index = 0;
    while (filter_start != filter_end) {
      float val = *chip_sd_start;
      mWells[index++].traceSd = val;
      if (val < sd_threshold) {
        *filter_start = DifferentialSeparator::LowTraceSd;
      }
      filter_start++;
      chip_sd_start++;
    }
  }
  printf("finished filtering.\n");
  // cleanup
  free(mean);
  free(m2);
  free(chip_sd);
  free(chip_good_sd);
}

float DifferentialSeparator::IQR (SampleQuantiles<float> &s) {
  if (s.GetNumSeen() == 0) {
    return 0;
  }
  return (s.GetQuantile (.75) - s.GetQuantile (.25));
}

void DifferentialSeparator::ClusterRegion (int rowStart, int rowEnd,
					   int colStart, int colEnd,
					   float maxMad,
					   float minBeadSnr,
					   size_t minGoodWells,
					   vector<float> &bfMetric,
					   vector<KeyFit> &wells,
					   double trim,
                                           bool doCenter,
					   MixModel &model)
{
  vector<float> metric;
  vector<int8_t> cluster;
  int numWells = (rowEnd - rowStart) * (colEnd - colStart);
  metric.reserve (numWells);
  cluster.reserve (numWells);
  for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
    for (int colIx = colStart; colIx < colEnd; colIx++) {
      size_t idx = rowIx * mMask.W() + colIx;
      if (wells[idx].ok == 1 && mFilteredWells[idx] == GoodWell && isfinite(bfMetric[idx])) {
        metric.push_back(bfMetric[idx]);
        if (wells[idx].goodLive) {
          cluster.push_back (1);
        }
        else if (wells[idx].isRef) {
          cluster.push_back(0);
        }
        else {
          cluster.push_back (-1);
        }
      }
    }
  }
  if (metric.size() < minGoodWells) {
    return;
  }
  DualGaussMixModel dgm (metric.size());
  dgm.SetTrim (trim);
  model = dgm.FitDualGaussMixModelFaster (&metric[0], &cluster[0], metric.size());
  model.refMean = 0;
  DualGaussMixModel::SetThreshold(model);
}

void DifferentialSeparator::MakeStadardKeys (vector<KeySeq> &keys)
{
  //                                   0 1 2 3 4 5 6 7
  vector<int> libKey = char2Vec<int> ("1 0 1 0 0 1 0 1", ' ');
  vector<int> tfKey =  char2Vec<int> ("0 1 0 0 1 0 1 1", ' ');
  KeySeq lKey, tKey;
  lKey.name = "lib";
  lKey.flows = libKey;
  lKey.zeroFlows.resize (4);
  lKey.zeroFlows[0] = 1;
  lKey.zeroFlows[1] = 3;
  lKey.zeroFlows[2] = 4;
  lKey.zeroFlows[3] = 6;
  lKey.minPeak = 15;
  lKey.onemerFlows.resize(3);
  lKey.onemerFlows[0] = 0;
  lKey.onemerFlows[1] = 2;
  lKey.onemerFlows[2] = 5;
  lKey.minSnr = 5.5;
  lKey.good_enough_peak = 15;
  lKey.good_enough_snr = 2.5;
  lKey.usableKeyFlows = 7;
  keys.push_back (lKey);
  tKey.name = "tf";
  tKey.flows = tfKey;
  tKey.minSnr = 7;
  tKey.zeroFlows.resize (4);
  tKey.zeroFlows[0] = 0;
  tKey.zeroFlows[1] = 2;
  tKey.zeroFlows[2] = 3;
  tKey.zeroFlows[3] = 5;
  tKey.minPeak = 20;
  tKey.good_enough_snr = 4.0;
  tKey.good_enough_peak = 40;
  tKey.onemerFlows.resize(3);
  tKey.onemerFlows[0] = 1;
  tKey.onemerFlows[1] = 4;
  tKey.onemerFlows[2] = 6;
  tKey.usableKeyFlows = 7;
  keys.push_back (tKey);
}

void DifferentialSeparator::LoadInitialMask (Mask *preMask, const std::string &maskFile, 
                                             const std::string &imgFile, Mask &mask, int ignoreChecksumErrors) {
  if (preMask != NULL) {
      mask.Init (preMask);
  }
  else if (!maskFile.empty()) {
      mask.SetMask (maskFile.c_str());
  }
  else {
    // @todo cws - If we open beadfind file here, don't open it again...
    Image bfImg;
    bfImg.SetImgLoadImmediate (false);
    bfImg.SetIgnoreChecksumErrors (ignoreChecksumErrors);
    bool loaded = bfImg.LoadRaw (imgFile.c_str());
    if (!loaded) {
      ION_ABORT ("Couldn't load file: " + imgFile);
    }
    const RawImage *raw = bfImg.GetImage();
    int cols = raw->cols;
    int rows = raw->rows;
    mask.Init (cols,rows,MaskEmpty);
  }
}

void DifferentialSeparator::PrintKey (const KeySeq &k, int kIx) {
  cout << "Key: " << kIx << "\t" << k.name << "\t" << k.usableKeyFlows << "\t" << k.minSnr << endl;
  for (size_t i = 0; i < k.flows.size(); i++) {
    cout << k.flows[i] << ' ';
  }
  cout << endl;
  for (size_t i = 0; i < k.zeroFlows.size(); i++) {
    cout << k.zeroFlows.at (i) << ' ';
  }
  cout << endl;
}

void DifferentialSeparator::SetKeys (SequenceItem *seqList, int numSeqListItems, 
                                     float minLibSnr, float minTfSnr,
                                     float minLibPeak, float minTfPeak) {
  mKeys.clear();
  for (int i = numSeqListItems - 1; i >= 0; i--) {
    KeySeq k;
    k.name = seqList[i].seq;
    k.flows.resize (seqList[i].numKeyFlows);
    k.usableKeyFlows = seqList[i].usableKeyFlows;
    int zero_count = 0;
    int onemer_count = 0;
    for (int flowIx = 0; flowIx < seqList[i].usableKeyFlows; flowIx++) {
      if (seqList[i].Ionogram[flowIx] == 0) {
        zero_count++;
      }
      if (seqList[i].Ionogram[flowIx] == 1) {
        onemer_count++;
      }
    }

    k.zeroFlows.resize (zero_count);
    k.onemerFlows.resize (onemer_count);
    zero_count = 0;
    onemer_count = 0;
    for (int flowIx = 0; flowIx < seqList[i].numKeyFlows; flowIx++){
      k.flows[flowIx] = seqList[i].Ionogram[flowIx];
      if (seqList[i].Ionogram[flowIx] == 0 && flowIx < seqList[i].usableKeyFlows) {
        k.zeroFlows.at (zero_count++) = flowIx;
      }
      if (seqList[i].Ionogram[flowIx] == 1 && flowIx < seqList[i].usableKeyFlows) {
        k.onemerFlows.at (onemer_count++) = flowIx;
      }
    }
    // @todo cws - put this all in a centralized place rather than hard coding here..
    if (i == 1) {//  @todo - this is hacky to assume the key order
      k.minSnr = minLibSnr;
      k.minPeak = minLibPeak;
      k.good_enough_snr = minLibSnr * .5;
      k.good_enough_peak = minLibPeak * 2;
    }
    else {
      k.minSnr = minTfSnr;
      k.minPeak = minTfPeak;;
      k.good_enough_snr = minTfSnr * .5;
      k.good_enough_peak = minTfPeak * 2;
    }
    mKeys.push_back (k);
  }
  for (size_t i = 0; i < mKeys.size(); i++) {
    PrintKey (mKeys[i], i);
  }
}

int DifferentialSeparator::GetWellCount (int row, int col,
					 Mask &mask, enum MaskType type, int distance)
{

  int count = 0;
  int rowStart = max (0,row-distance);
  int rowEnd = min (mask.H()-1,row+distance);
  int colStart = max (0,col-distance);
  int colEnd = min (mask.W()-1,col+distance);
  int wellIx = mask.ToIndex (row, col);
  for (int r = rowStart; r <= rowEnd; r++)
    {
      for (int c = colStart; c <= colEnd; c++)
	{
	  int idx = mask.ToIndex (r, c);
	  if (idx != wellIx && mask[idx] & type)
	    {
	      count++;
	    }
	}
    }
  return count;
}

double DifferentialSeparator::GetAvg1mer (int row, int col,
					  Mask &mask, enum MaskType type,
					  std::vector<KeyFit> &wells,
					  int distance)
{

  int count = 0;
  int rowStart = max (0,row-distance);
  int rowEnd = min (mask.H()-1,row+distance);
  int colStart = max (0,col-distance);
  int colEnd = min (mask.W()-1,col+distance);
  int wellIx = mask.ToIndex (row, col);
  double onemerAvg = 0;
  for (int r = rowStart; r <= rowEnd; r++)
    {
      for (int c = colStart; c <= colEnd; c++)
	{
	  int idx = mask.ToIndex (r, c);
	  if (idx != wellIx && mask[idx] & type)
	    {
	      onemerAvg += wells[idx].peakSig;
	      count++;
	    }
	}
    }
  if (count > 0)
    {
      onemerAvg = onemerAvg / count;
    }
  return onemerAvg;
}

void DifferentialSeparator::CalcDensityStats (const std::string &prefix, Mask &mask, std::vector<KeyFit> &wells)
{
  string s = prefix + ".density.txt";
  ofstream out (s.c_str());
  char d = '\t';
  out << "row\tcol\tidx\tsnr\tempty1\tlive1\tdud1\tignore1\tsig1\tempty2\tlive2\tdud2\tignore2\tsig2" << endl;
  for (int rowIx = 0; rowIx < mask.H(); rowIx++)
    {
      for (int colIx = 0; colIx < mask.W(); colIx++)
	{
	  int idx = mask.ToIndex (rowIx, colIx);
	  if (mask[idx] & MaskLib)
	    {
	      out << rowIx << d << colIx << d << idx << d << wells[idx].snr;
	      out << d << GetWellCount (rowIx, colIx, mask, MaskEmpty, 1);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskLive, 1);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskDud, 1);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskIgnore, 1);
	      out << d << GetAvg1mer (rowIx, colIx, mask, MaskLive, wells, 1);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskEmpty, 2);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskLive, 2);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskDud, 2);
	      out << d << GetWellCount (rowIx, colIx, mask, MaskIgnore, 2);
	      out << d << GetAvg1mer (rowIx, colIx, mask, MaskLive, wells, 2);
	      out << endl;
	    }
	}
    }
}



//void DifferentialSeparator::CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, const std::string &file) {
void DifferentialSeparator::CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq, Image &img) {
  // string bfFile = opts.resultsDir + "/" + file;
  // Image img;
  // img.SetImgLoadImmediate (false);
  // img.SetIgnoreChecksumErrors (opts.ignoreChecksumErrors);
  // bool loaded =   img.LoadRaw(bfFile.c_str());
  // if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }

  ClockTimer t0Timer;
  RawImage *raw = img.raw;
  FilterPixelSd(raw, MIN_BF_SD, mFilteredWells);
  ssq.resize(t0vec.size());
  std::fill(ssq.begin(), ssq.end(), std::numeric_limits<float>::quiet_NaN());
  // GridMesh<int> pcaMesh;
  // pcaMesh.Init (mask.H(), mask.W(), PCA_COMP_GRID_SIZE, PCA_COMP_GRID_SIZE);
  // int converged = 0;
  // for (size_t binIx = 0; binIx < pcaMesh.GetNumBin(); binIx++) {
  //   // Get the region
  //   int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  //   pcaMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
  //   bool ok = TraceStoreCol::PcaLossyCompressChunk(rowStart, rowEnd, colStart, colEnd,
  //                                        raw->rows, raw->cols, raw->frames,
  //                                        raw->frameStride, 0, raw->frameStride,
  //                                        raw->image, false,
  //                                        &ssq[0], &mFilteredWells[0],
  //                                        3, 3, 5);
  //   if (ok) { converged++; }
  // }
  // std::vector<float> ssq_sorted;
  // ssq_sorted.reserve(ssq.size());
  // for (size_t i = 0; i < ssq.size(); i++) {
  //   ssq[i] = sqrt(ssq[i]);
  //   if (mFilteredWells[i] == 0 && isfinite(ssq[i])) {
  //     ssq_sorted.push_back(ssq[i]);
  //   }
  // }
  // if (ssq_sorted.size() > MIN_SAMPLE_TAUE_STATS) {
  //   std::sort(ssq_sorted.begin(), ssq_sorted.end());
  //   float q25 = ionStats::quantile_sorted(ssq_sorted, .25);
  //   float q75 = ionStats::quantile_sorted(ssq_sorted, .75);
  //   float threshold = q75 + 3 * (q75-q25);
  //   int bad_count = 0, nan_count = 0;
  //   for (size_t i = 0; i < ssq.size(); i++) {
  //     if (mFilteredWells[i] == 0 && (ssq[i] >= threshold || isnan(ssq[i]))) {
  //     if (isnan(ssq[i])) { nan_count++; }
  //     mFilteredWells[i] = NotCompressable;
  //     bad_count++;
  //     }
  //   }
  //   fprintf(stdout, "%d wells of %d (%.2f%%) marked as not compressable (%d nan) in beadfind flow.\nThreshold %.2f with %d of %d regions not compressing\n", 
  //           bad_count, (int)ssq.size(), 100.0 * bad_count / ssq.size(),
  //           nan_count, threshold, 
  //           (int)(pcaMesh.GetNumBin() - converged), (int)pcaMesh.GetNumBin());
  // }
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(3);
  t0.SetMinFirstHingeSlope(opts.beadfindBfThreshold[0]/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(opts.beadfindBfThreshold[1]/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(opts.beadfindBfThreshold[2]/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(opts.beadfindBfThreshold[3]/raw->baseFrameRate);
  t0.SetBadWells(&mFilteredWells[0]);
  t0.SetDebugLevel(opts.outputDebug);

  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mMask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStepY, opts.t0MeshStepX, opts.nCores);
  int *timestamps = raw->timestamps;
  t0.SetTimeStamps(timestamps, frames);
  T0Prior prior;
  prior.mTimeEnd = 3000;
  t0.SetGlobalT0Prior(prior);
  t0Timer.PrintMicroSecondsUpdate(stdout, "T0 Timer: Before Sum .");
  t0.CalcAllSumTrace(data);
  t0Timer.PrintMicroSecondsUpdate(stdout, "T0 Timer: Before T0 Calc.");
  t0.CalcT0FromSum();
  t0Timer.PrintMicroSecondsUpdate(stdout, "T0 Timer: Before Individual T0 Calc .");
  // @todo - cws calculate for a 10x10 region rather than per well
  t0.CalcIndividualT0(t0vec, opts.useMeshNeighbors);
  t0Timer.PrintMicroSecondsUpdate(stdout, "T0 Timer: After Individual T0 Calc .");
  for (size_t i = 0; i < t0vec.size(); i++) {
    if (t0vec[i] > 0) {
      int tmpT0End = img.GetFrame(t0vec[i]-img.GetFlowOffset());
      int tmpT0Start = tmpT0End - 1;
      //      assert(tmpT0Start >= 0);
      if (tmpT0Start < 0)
        t0vec[i] = 0;
      else {
        float t0Frames = (t0vec[i] - raw->timestamps[tmpT0Start])/(raw->timestamps[tmpT0End]-raw->timestamps[tmpT0Start])*(raw->compToUncompFrames[tmpT0End] - raw->compToUncompFrames[tmpT0Start]) + raw->compToUncompFrames[tmpT0Start];
        t0Frames = max(0.0f,t0Frames);
        t0Frames = min(t0Frames, frames - 1.0f);
        t0vec[i] = t0Frames;
      }
    }
  }
  if (opts.outputDebug > 1) {
    string refFile = opts.outData + ".reference_bf_t0.txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
  t0Timer.PrintMicroSecondsUpdate(stdout, "T0 Timer: Finished.");
}

void DifferentialSeparator::CalcAcqT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq,
                                      const std::string &file) {
  string bfFile = opts.resultsDir + "/" + file;
  Image img;
  img.SetImgLoadImmediate (false);
  bool loaded =   img.LoadRaw(bfFile.c_str());
  ssq.resize(t0vec.size());
  std::fill(ssq.begin(), ssq.end(), std::numeric_limits<float>::quiet_NaN());
  if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }
  RawImage *raw = img.raw;
  // GridMesh<int> pcaMesh;
  // pcaMesh.Init (mask.H(), mask.W(), PCA_COMP_GRID_SIZE, PCA_COMP_GRID_SIZE);
  // for (size_t binIx = 0; binIx < pcaMesh.GetNumBin(); binIx++) {
  //   // Get the region
  //   int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  //   pcaMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
  //   TraceStoreCol::PcaLossyCompressChunk(rowStart, rowEnd, colStart, colEnd,
  //                                        raw->rows, raw->cols, raw->frames,
  //                                        raw->frameStride, 0, raw->frameStride,
  //                                        raw->image, false,
  //                                        &ssq[0], &mFilteredWells[0],
  //                                        2, 2, 6);
  // }
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(4);
  t0.SetMinFirstHingeSlope(opts.beadfindAcqThreshold[0]/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(opts.beadfindAcqThreshold[1]/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(opts.beadfindAcqThreshold[2]/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(opts.beadfindAcqThreshold[3]/raw->baseFrameRate);
  t0.SetBadWells(&mFilteredWells[0]);
  t0.SetDebugLevel(opts.outputDebug);

  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mMask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStepY, opts.t0MeshStepX, opts.nCores);
  int *timestamps = raw->timestamps;
  t0.SetTimeStamps(timestamps, frames);
  T0Prior prior;
  prior.mTimeEnd = 3000;
  t0.SetGlobalT0Prior(prior);
  t0.CalcAllSumTrace(data);
  t0.CalcT0FromSum();
  t0.CalcIndividualT0(t0vec, opts.useMeshNeighbors);
  for (size_t i = 0; i < t0vec.size(); i++) {
    if (t0vec[i] > 0) {
      int tmpT0End = img.GetFrame(t0vec[i]-img.GetFlowOffset());
      int tmpT0Start = tmpT0End - 1;
      if (tmpT0Start < 0)
        t0vec[i] = 0;
      else {
        float t0Frames = (t0vec[i] - raw->timestamps[tmpT0Start])/(raw->timestamps[tmpT0End]-raw->timestamps[tmpT0Start])*(raw->compToUncompFrames[tmpT0End] - raw->compToUncompFrames[tmpT0Start]) + raw->compToUncompFrames[tmpT0Start];
        t0Frames = max(0.0f,t0Frames);
        t0Frames = min(t0Frames, frames - 1.0f);
        t0vec[i] = t0Frames;
      }
    }
  }
  if (opts.outputDebug > 1) {
    string refFile = opts.outData + ".reference_bf_t0." + file + ".txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
  img.Close();

}

void DifferentialSeparator::CalcAcqT0(DifSepOpt &opts, std::vector<float> &t0vec, std::vector<float> &ssq,
                                      Image &img, bool filt) {
  ssq.resize(t0vec.size());
  std::fill(ssq.begin(), ssq.end(), std::numeric_limits<float>::quiet_NaN());
  RawImage *raw = img.raw;
  // GridMesh<int> pcaMesh;
  // pcaMesh.Init (mask.H(), mask.W(), PCA_COMP_GRID_SIZE, PCA_COMP_GRID_SIZE);
  // for (size_t binIx = 0; binIx < pcaMesh.GetNumBin(); binIx++) {
  //   // Get the region
  //   int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  //   pcaMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
  //   TraceStoreCol::PcaLossyCompressChunk(rowStart, rowEnd, colStart, colEnd,
  //                                        raw->rows, raw->cols, raw->frames,
  //                                        raw->frameStride, 0, raw->frameStride,
  //                                        raw->image, false,
  //                                        &ssq[0], &mFilteredWells[0],
  //                                        2, 2, 6);
  // }

  if (filt) {
    FilterPixelSd(raw, MIN_SD, mFilteredWells);
  }
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(4);
  t0.SetMinFirstHingeSlope(opts.beadfindAcqThreshold[0]/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(opts.beadfindAcqThreshold[1]/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(opts.beadfindAcqThreshold[2]/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(opts.beadfindAcqThreshold[3]/raw->baseFrameRate);
  t0.SetBadWells(&mFilteredWells[0]);
  t0.SetDebugLevel(opts.outputDebug);

  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mMask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStepY, opts.t0MeshStepX, opts.nCores);
  int *timestamps = raw->timestamps;
  t0.SetTimeStamps(timestamps, frames);
  T0Prior prior;
  prior.mTimeEnd = 3000;
  t0.SetGlobalT0Prior(prior);
  t0.CalcAllSumTrace(data);
  t0.CalcT0FromSum();
  t0.CalcIndividualT0(t0vec, opts.useMeshNeighbors);
  for (size_t i = 0; i < t0vec.size(); i++) {
    if (t0vec[i] > 0) {
      int tmpT0End = img.GetFrame(t0vec[i]-img.GetFlowOffset());
      int tmpT0Start = tmpT0End - 1;
      if (tmpT0Start < 0)
        t0vec[i] = 0;
      else {
        float t0Frames = (t0vec[i] - raw->timestamps[tmpT0Start])/(raw->timestamps[tmpT0End]-raw->timestamps[tmpT0Start])*(raw->compToUncompFrames[tmpT0End] - raw->compToUncompFrames[tmpT0Start]) + raw->compToUncompFrames[tmpT0Start];
        t0Frames = max(0.0f,t0Frames);
        t0Frames = min(t0Frames, frames - 1.0f);
        t0vec[i] = t0Frames;
      }
    }
  }
}


void DifferentialSeparator::PrintVec(Col<float> &vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    cout << ", " << vec[i];
  }
  cout << endl;
}

void DifferentialSeparator::PrintWell(TraceStore &store, int well, int flow) {
  Col<float> vec(store.GetNumFrames());
  store.GetTrace(well, flow, vec.begin());
  cout << "Well: " << well;
  PrintVec(vec);
}

void DifferentialSeparator::WellDeviation(TraceStoreCol &store,
                                          int rowStep, int colStep,
                                          vector<char> &filter,
                                          vector<float> &mad) {

  GridMesh<float> mesh;
  mesh.Init(mMask.H(), mMask.W(), rowStep, colStep);
  vector<float> mean(filter.size());
  vector<float> m2(filter.size());
  vector<float> summary(filter.size());
  vector<float> normalize(filter.size() * store.GetNumFrames());
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++) {
    mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    WellDeviationRegion(store, rowStart, rowEnd, colStart, colEnd,
                        0, store.GetNumFrames(), 0, store.GetNumFlows(),
                        &mean[0], &m2[0],
                        &normalize[0],
                        &summary[0], filter, mad);
  }
}
                                          

void DifferentialSeparator::WellDeviationRegion(TraceStoreCol &store,
                                                int row_start, int row_end,
                                                int col_start, int col_end,
                                                int frame_start, int frame_end,
                                                int flow_start, int flow_end,
                                                float *mean, float *m2,
                                                float *normalize,
                                                float *summary,
                                                vector<char> &filters,
                                                vector<float> &mad) {
  int loc_num_frames = frame_end - frame_start;
  // calculate the average trace
  float region_sum[loc_num_frames];
  memset(region_sum,0,sizeof(float) *loc_num_frames);
  int count = 0;
  
  int store_num_wells = store.GetNumWells();
  int store_num_cols = store.GetNumCols();
  size_t loc_num_wells = (row_end - row_start) * (col_end - col_start);
  size_t loc_num_cols =  (col_end - col_start);
  size_t loc_num_well_flows = loc_num_wells * (flow_end - flow_start);
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
      for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
        size_t offset = row_ix * store_num_cols + col_start;
        char *__restrict filter_start = &filters[0] + offset;
        int16_t *__restrict store_start = &store.mData[0] + (frame_ix * store.mFlowFrameStride) + (flow_ix * store_num_wells) + offset;
        int16_t *__restrict store_end = store_start + loc_num_cols;
        //        assert(store_end - &store.mData[0] <= store.mData.size());
        //        assert(filter_start - &filters[0] <= filters.size());
        float &frame_sum = region_sum[frame_ix-frame_start];
        while (store_start != store_end) {
          if ((*filter_start) == 0) {
            frame_sum += *store_start;
             count++;
          }
          filter_start++;
          store_start++;
        }
      }
    }
  }
  // find maximum frame
  float max_value = region_sum[0];
  int max_value_frame = 0;
  for (int i = 1; i < loc_num_frames; i++) {
    if (region_sum[i] >= max_value) {
      max_value = region_sum[i];
      max_value_frame = i;
    }
  }
  // calculate the mean average deviation over all the flows
  int norm_start = max(0, max_value_frame -1);
  int norm_end = min(frame_end, max_value_frame+1);

  memset(normalize, 0, sizeof(float) * loc_num_well_flows); 
  int norm_count = norm_end - norm_start;
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
      for (int frame_ix = norm_start; frame_ix < norm_end; frame_ix++) {
        size_t store_offset = row_ix * store_num_cols + col_start;
        int16_t *__restrict store_start = store.GetMemPtr() + frame_ix * store.mFlowFrameStride + flow_ix * store_num_wells + store_offset;
        int16_t *__restrict store_end = store_start + loc_num_cols;
        size_t loc_offset = (row_ix - row_start) * loc_num_cols + (flow_ix-flow_start) * loc_num_wells;
        float *__restrict normalize_start = normalize + loc_offset;
        while(store_start != store_end) {
          *normalize_start++ += *store_start++;
        }
      }
    }
  }
  
  // Normalize per flow per well factor
  float *__restrict normalize_start = normalize;
  float *__restrict normalize_end = normalize_start + loc_num_well_flows;
  while (normalize_start != normalize_end) {
    *normalize_start++ /= norm_count;
  }
  
  // Loop over all the wells doing normalization per flow and calcule the mean and variance
  memset(region_sum,0,sizeof(float) *loc_num_frames);
  memset(summary, 0, sizeof(float) * loc_num_wells);
  int frame_mad_start = min(frame_start + WELL_DEV_FRAME_START_OFFSET, frame_end);
  int frame_mad_end = max(frame_start, max_value_frame - WELL_DEV_FRAME_END_OFFSET);
  for (int frame_ix = frame_mad_start; frame_ix < frame_mad_end; frame_ix++) {
    // rezero for each frame
    memset(mean, 0, sizeof(float) * loc_num_wells);
    memset(m2, 0, sizeof(float) * loc_num_wells);
    int count = 0;
    for (int row_ix = row_start; row_ix < row_end; row_ix++) {
      count = 0;
      for (int flow_ix = flow_start; flow_ix < flow_end; flow_ix++) {
        count++;
        int well_offset = row_ix * store_num_cols + col_start;
        size_t store_offset = frame_ix *store.mFlowFrameStride + flow_ix * store_num_wells + well_offset;
        int16_t *__restrict store_start = store.GetMemPtr() + store_offset;
        int16_t *__restrict store_end = store_start + loc_num_cols;
        int loc_offset = (row_ix - row_start) * loc_num_cols;
        float *__restrict norm_start = normalize + flow_ix * loc_num_wells + loc_offset;
        float *__restrict mean_start = mean + loc_offset;
        float *__restrict m2_start = m2 + loc_offset;
        float delta;
        while (store_start != store_end) {
          if (*norm_start == 0.0f) {
            *norm_start = 1.0f;
            if (filters[well_offset] == GoodWell) {
              filters[well_offset] = WellDevZeroNorm;
            }
          }
          float value = (float)(*store_start) / *norm_start;
          delta = value - *mean_start;
          *mean_start += delta / count;
          *m2_start += delta * (value - *mean_start);
          store_start++;
          norm_start++;
          mean_start++;
          m2_start++;
          well_offset++;
        }
      }
    }
    // Add in the sd of this frame to avg
    float *__restrict summary_start = summary;
    float *__restrict summary_end = summary_start + loc_num_wells;
    float *__restrict m2_start = m2;
    int loc_num_flows = flow_end - flow_start;
    while(summary_start != summary_end) {
      float value = (*m2_start / loc_num_flows);
      region_sum[frame_ix] += value;
      *summary_start += value;
      summary_start++;
      m2_start++;
    }
  }
  // Copy back to our vector of results
  for (int frame_ix = frame_start; frame_ix < frame_end; frame_ix++) {
    region_sum[frame_ix] /= loc_num_wells;
    region_sum[frame_ix] = sqrt(region_sum[frame_ix]);
  }
  for (int row_ix = row_start; row_ix < row_end; row_ix++) {
    float *__restrict summary_start = summary + (row_ix - row_start) * loc_num_cols;
    float *__restrict summary_end = summary_start + loc_num_cols;
    float *__restrict mad_start = &mad[0] + row_ix * store_num_cols + col_start;
    while(summary_start != summary_end) {
      *mad_start++ = sqrt(*summary_start++ / loc_num_frames);
    }
  }
}

void DifferentialSeparator::RankWellsBySignal(int flow0, int flow1, TraceStore &store,
                                              float iqrMult,
                                              int numBasis,
                                              Mask &mask,
                                              int minWells,
                                              int rowStart, int rowEnd,
                                              int colStart, int colEnd,
                                              std::vector<char> &filter,
                                              std::vector<float> &mad) {


  // How many wells are good? (not pinned and not excluded)
  int okCount = 0;
  for (int r = rowStart; r < rowEnd; r++) {
    for (int c = colStart; c < colEnd; c++) {
      int idx = mask.ToIndex(r, c);
      mad[idx] = std::numeric_limits<float>::max();
      if (!mask.Match(c, r, MaskPinned) && !mask.Match(c,r,MaskExclude) && filter[idx] == GoodWell) {
        okCount++;
      }
      else if (filter[idx] == GoodWell) {
        filter[idx] = PinnedExcluded;
      }
    }
  }
  if (okCount < minWells) {
    return;
  }
  // Create onemer (O) and zeromer (Z) matrices
  fmat O(store.GetNumFrames(), okCount);
  fmat Z(store.GetNumFrames(), okCount);
  vector<int> mapping(okCount);
  int count = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int idx = mask.ToIndex(row, col);
      if (!mask.Match(col, row, MaskPinned) && !mask.Match(col,row,MaskExclude) && filter[idx] == GoodWell) {
        mapping[count] = idx;
        store.GetTrace(idx, flow1, O.begin_col(count));
        store.GetTrace(idx, flow0, Z.begin_col(count));
        count++;
      }
    }
  }
  
  // Join together into a single matrix so we have both 0mer and 1mer flows for each well.
  fmat X = join_rows(O,Z); // L x N

  // Form covariance matrix and get first B (numBasis) vectors
  fmat Cov = X * X.t();    // L x L
  arma::fmat EVec;
  arma::Col<float> EVal;
  eig_sym(EVal, EVec, Cov);
  fmat V(EVec.n_rows, numBasis);  // L x numBasis
  count = 0;
  for(size_t v = V.n_rows - 1; v >= V.n_rows - numBasis; v--) {
    copy(EVec.begin_col(v), EVec.end_col(v), V.begin_col(count++));
  }

  // Linear algegra trick
  // Normally we'd have to solve via something like t(A) = (t(V)*V)^-1 * t(V) * X but since V is orthonormal 
  // t(V)*V = I and t(A) = t(V)*X and A = t(X) * V
  fmat A = X.t() * V;  // N x numBasis
  fmat P = A * V.t();  // Prediction N x L 
  fmat D = X - P.t();  // Difference L x N
  fmat d = mean(abs(D)); // Mean average deviation per well

  // Filter out wells that don't compress well.
  SampleQuantiles<float> samp(1000);
  SampleQuantiles<float> Osamp(1000);
  SampleQuantiles<float> Zsamp(1000);
  Row<float> Olast = O.row(O.n_rows - 1);
  Row<float> Zlast = Z.row(Z.n_rows - 1);
  samp.AddValues(d.memptr(), d.n_elem);
  Osamp.AddValues(Olast.memptr(), Olast.n_elem);
  Zsamp.AddValues(Zlast.memptr(), Zlast.n_elem);
  double maxMad = samp.GetQuantile(.75) + iqrMult * samp.GetIQR();
  double minMad = samp.GetQuantile(.25) - iqrMult * samp.GetIQR();
  minMad = max(0.0, minMad);

  // double maxO = Osamp.GetQuantile(.75) + iqrMult * Osamp.GetIQR();
  // double minO = Osamp.GetQuantile(.25) - iqrMult * Osamp.GetIQR();


  // double maxZ = Zsamp.GetQuantile(.75) + iqrMult * Zsamp.GetIQR();
  // double minZ = Zsamp.GetQuantile(.25) - iqrMult * Zsamp.GetIQR();
  for (size_t i = 0; i < O.n_cols; i++) {
    float dx = d[i];
    float dox = d[i+O.n_cols];
    //    float o = O.at(O.n_rows - 1, i);
    //    float z = Z.at(Z.n_rows - 1, i);
    if (dx < maxMad && dx > minMad && dox < maxMad && dox > minMad) {// &&
      //        o > minO && o < maxO &&
      //        z > minZ && z < maxZ) {
      SampleStats<float> sam;
      Row<float> vec = abs(P.row(i) - P.row(i + O.n_cols));
      sam.AddValues(vec.memptr(), vec.n_elem);
      mad[mapping[i]] = fabs(sam.GetMean());
    }
    else {
      filter[mapping[i]] = NotCompressable;
    }
  }
}

void DifferentialSeparator::CreateSignalRef(int flow0, int flow1, TraceStore &store,
					    int rowStep, int colStep,
					    float iqrMult,
					    int numBasis,
					    Mask &mask,
					    int minWells,
					    std::vector<char> &filter,
					    std::vector<float> &mad) {
  GridMesh<float> mesh;
  mesh.Init(mask.H(), mask.W(), rowStep, colStep);
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++) {
    mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    RankWellsBySignal(flow0, flow1,  store, iqrMult, numBasis, mask,
                      minWells, rowStart, rowEnd, colStart,colEnd,
                      filter, mad);
  }
}


void DifferentialSeparator::PickCombinedRank(vector<vector<float> > &mads,
                                             vector<char> &filter, vector<char> &refWells,
                                             int numWells,
                                             int rowStart, int rowEnd, int colStart, int colEnd) {
  int size = (colEnd - colStart) * (rowEnd - rowStart);
  std::vector<std::pair<float,int> > rankValues(size);
  std::vector<std::pair<float, int> > combinedRanks(size);
  std::vector<int> mapping(size);
  int count = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      combinedRanks[count].first = 0.0f;
      combinedRanks[count].second = count;
      mapping[count] = mMask.ToIndex(row, col);
      count++;
    }
  }
  // /// Sort the bf metric
  // for (size_t i = 0; i < combinedRanks.size(); i++) {
  //   rankValues[i].second = i;
  //   if (filter[mapping[i]] == GoodWell) {
  //     rankValues[i].first = bfMetric[mapping[i]];
  //   }
  //   else {
  //     rankValues[i].first = size;
  //   }
  // }
  // sort(rankValues.begin(), rankValues.end());
  // // store the ranked values in the combined ranks.
  // for (size_t i = 0; i < combinedRanks.size(); i++) {
  //   combinedRanks[rankValues[i].second].first += i;
  // }
  // Do individual metrics
  for (size_t keyIx = 0; keyIx < mads.size(); keyIx++) {
    for (size_t i = 0; i < combinedRanks.size(); i++) {
      rankValues[i].second = i;
      if (filter[mapping[i]] == GoodWell) {
        rankValues[i].first = mads[keyIx][mapping[i]];
      }
      else {
        rankValues[i].first = size;
      }
    }
    sort(rankValues.begin(), rankValues.end());
    for (size_t i = 0; i < combinedRanks.size(); i++) {
      combinedRanks[rankValues[i].second].first += i;
    }
  }
  sort(combinedRanks.begin(), combinedRanks.end());
  size_t found = 0;
  for (size_t i = 0; found < (size_t)numWells && i < combinedRanks.size(); i++) {
    if (filter[mapping[combinedRanks[i].second]] == GoodWell) {
      refWells[mapping[combinedRanks[i].second]] = 1;
      found++;
    }
  }

}

void DifferentialSeparator::PickCombinedRank(vector<vector<float> > &mads,
                                             int rowStep, int colStep,
                                             float minPercent, int numWells,
                                             vector<char> &filter, vector<char> &refWells) {
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before picking reference.");
  refWells.resize(filter.size());
  std::fill(refWells.begin(), refWells.end(), 0);
  GridMesh<float> mesh;
  mesh.Init(mMask.H(), mMask.W(), rowStep, colStep);
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++) {
    mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    int goodWells = 0;
    int num_cols = mMask.W();
    float totalWells = (colEnd - colStart) * (rowEnd - rowStart);
    for (int row_ix = rowStart; row_ix < rowEnd; row_ix++) {
      for (int col_ix = colStart; col_ix < colEnd; col_ix++) {
        int index = row_ix * num_cols + col_ix;
        if (filter[index] == GoodWell) {
          goodWells++;
        }
      }
    }
    float percent = goodWells / totalWells;
    // Only pick reference wells for regions that aren't crazy
    if (goodWells >= MIN_REGION_REF_WELLS && percent >= MIN_REGION_REF_PERCENT) {
      PickCombinedRank(mads, filter, refWells,
                       numWells,
                       rowStart, rowEnd, colStart, colEnd);
    }
  }
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After picking reference.");
}

bool DifferentialSeparator::Find0merAnd1merFlows(KeySeq &key, TraceStore &store,
                                                 int &flow0mer, int &flow1mer) {
  bool found = false;
  // Looking at first 1,0 and then 0, 1 to try and use different flows for different keys.
  for (size_t flow1 = 0; flow1 < key.usableKeyFlows; flow1++) {
    for (size_t flow2 = 1; flow2 < key.usableKeyFlows; flow2++) {
      if (key.flows[flow1] == 1 && key.flows[flow2]  == 0 && store.GetNucForFlow(flow1) == store.GetNucForFlow(flow2)) {
        flow0mer = flow2; 
        flow1mer = flow1;
        found = true;
        return found;
      }
    }
  }
  for (size_t flow1 = 0; flow1 < key.usableKeyFlows; flow1++) {
    for (size_t flow2 = 1; flow2 < key.usableKeyFlows; flow2++) {
      if (key.flows[flow1] == 0 && key.flows[flow2]  == 1 && store.GetNucForFlow(flow1) == store.GetNucForFlow(flow2)) {
        flow0mer = flow1; 
        flow1mer = flow2;
        found = true;
        return found;
      }
    }
  }
  if (!found) {
    ION_WARN("Couldn't find good candiate for signal difference in key: " + key.name);
  }
  return found;
}

bool AllMatch(std::vector<KeySeq> & key_vectors, int flow, int hp) {
  for (size_t i = 0; i <key_vectors.size(); i++) {
    if (key_vectors[i].flows[flow] != hp) {
      return false;
    }
  }
  return true;
}

bool DifferentialSeparator::FindCommon0merAnd1merFlows(std::vector<KeySeq> &key_vectors,
                                                       TraceStore &store,
                                                       int &flow0mer, int &flow1mer) {
  int minflows = std::numeric_limits<int>::max();
  for (size_t keyIx = 0; keyIx < key_vectors.size(); keyIx++) {
    minflows = min((int)key_vectors[keyIx].usableKeyFlows, minflows);
  }
  for (int flow1 = minflows; flow1 >= 0; flow1--) {
    for (int flow2 = minflows; flow2 >= 0; flow2--) {
      if (store.GetNucForFlow(flow1) == store.GetNucForFlow(flow2) &&
          AllMatch(key_vectors, flow1, 1) && 
          AllMatch(key_vectors, flow2, 0)) {
        flow0mer = flow2;
        flow1mer = flow1;
        return true;
      }
    }
  }
  return false;
}

bool DifferentialSeparator::FindKey0merAnd1merFlows(KeySeq &key,
                                                    TraceStore &store,
                                                    std::vector<int>  &flow0mer, std::vector<int> &flow1mer) {
  int minflows = key.usableKeyFlows;
  flow0mer.resize(0);
  flow1mer.resize(0);
  for (int flow1 = minflows; flow1 >= 0; flow1--) {
    for (int flow2 = minflows; flow2 >= 0; flow2--) {
      if (store.GetNucForFlow(flow1) == store.GetNucForFlow(flow2) &&
          key.flows[flow1] == 1 && key.flows[flow2] == 0) {
        flow0mer.push_back(flow2);
        flow1mer.push_back(flow1);
        //        flow0mer = flow2;
        //        flow1mer = flow1;
      }
    }
  }
  return true;
}


void DifferentialSeparator::RankReference(TraceStoreCol &store,
                                          vector<float> &bfMetric, 
                                          int rowStep, int colStep,
                                          int useKeySignal,
                                          float iqrMult,
                                          int numBasis,
                                          float minPercent,
                                          Mask &mask,
                                          int minWells,
                                          vector<char> &filter,
                                          vector<char> &refWells) {

  int count = 0; 
  if (useKeySignal == 1) {
    mEmptyMetrics.resize(mKeys.size());
    for (size_t i = 0; i < mKeys.size(); i++) {
      int flow1 = -1, flow0 = -1;
      mEmptyMetrics[i].resize(mask.H() * mask.W(), std::numeric_limits<float>::max());
      bool found = Find0merAnd1merFlows(mKeys[i], store, flow1, flow0);
      cout << "for key: " << mKeys[i].name << " using differential flows: " << flow1 << " 1mer and " << flow0 << " 0mer." << endl;
      std::fill(mEmptyMetrics[i].begin(), mEmptyMetrics[i].end(), 0);
      if (found) {
        CreateSignalRef(flow0, flow1, store, rowStep, colStep, iqrMult, numBasis,
                        mask, minWells, filter, mEmptyMetrics[i]);
        //CountReference("After signal ref", mFilteredWells);
      }
    }
  }
  else if (useKeySignal == 2) {
    mEmptyMetrics.resize(1);
    mEmptyMetrics[0].resize(mask.H() * mask.W());
    std::fill(mEmptyMetrics[0].begin(), mEmptyMetrics[0].end(), 0);
    int flow0 = -1, flow1 = -1;
    bool found = FindCommon0merAnd1merFlows(mKeys, store, flow0, flow1);
    if (found) {
      CreateSignalRef(flow0, flow1, store, rowStep, colStep, iqrMult, numBasis,
                      mask, minWells, filter, mEmptyMetrics[0]);
    }
  }
  else if (useKeySignal == 3) {
    vector<int> flow0vec, flow1vec;
    bool found = FindKey0merAnd1merFlows(mKeys[0], store, flow0vec, flow1vec);
    mEmptyMetrics.resize(flow0vec.size());
    for (size_t i = 0; i < flow0vec.size() && found; i++) {
      cout << "Adding signal ref: " << i << ": " << flow0vec[i] << "," << flow1vec[i] << endl;
      mEmptyMetrics[i].resize(mask.H() * mask.W());
      std::fill(mEmptyMetrics[i].begin(), mEmptyMetrics[i].end(), 0);
      CreateSignalRef(flow0vec[i], flow1vec[i], store, rowStep, colStep, iqrMult, numBasis,
                      mask, minWells, filter, mEmptyMetrics[i]);
    }
  }
  else if (useKeySignal == 4) {
    mEmptyMetrics.resize(2);
    mEmptyMetrics[0].resize(mask.H() * mask.W());
    mEmptyMetrics[1].resize(mask.H() * mask.W());
    std::fill(mEmptyMetrics[0].begin(), mEmptyMetrics[0].end(), 0.0f);
    std::fill(mEmptyMetrics[1].begin(), mEmptyMetrics[1].end(), 0.0f);
    store.WellProj(store, mKeys, filter, mEmptyMetrics[0]);
    WellDeviation(store, rowStep, colStep, filter, mEmptyMetrics[1]);
    for (size_t well_ix = 0; well_ix < mWells.size(); well_ix++) {
      mWells[well_ix].bfMetric2 = mEmptyMetrics[0][well_ix];
      mWells[well_ix].bfMetric3 = mEmptyMetrics[1][well_ix];
    }
  }
  else {
    cout << "Not using key signal reference." << endl;
  }
  cout << "Picking reference with: " << mEmptyMetrics.size() << " vectors useSignalReference " << useKeySignal << endl;
  //  CountReference("After signal combined ranke", mFilteredWells);
}

void DifferentialSeparator::LoadKeyDats(PJobQueue &jQueue, TraceStoreCol &traceStore, 
                                        vector<float> &bfMetric, DifSepOpt &opts, std::vector<float> &traceSd) {
                               

  string resultsRoot = opts.resultsDir + "/" + opts.acqPrefix;
  string resultsSuffix = opts.datPostfix;
  size_t numWells = mMask.H() * mMask.W();
  if (mKeys.empty()) {  DifferentialSeparator::MakeStadardKeys (mKeys); }
  cout.flush();
  vector<int> rowStarts;
  vector<int> colStarts;
  size_t nRow = traceStore.GetNumRows();
  size_t nCol = traceStore.GetNumCols();
  double percents[3] = {.2, .5, .8};
  ClockTimer loadTimer;
  //int span = 7;
  for (size_t i = 0; i < ArraySize (percents); i++) {
    rowStarts.push_back (percents[i] * nRow);
    colStarts.push_back (percents[i] * nCol);
  }
  vector<float> t;
  Col<float> traceBuffer;
  traceStore.SetMeshDist (opts.useMeshNeighbors);
  size_t loadMinFlows = traceStore.GetNumFlows();
  traceStore.SetSize(T0_RIGHT_OFFSET);
  traceStore.SetT0(mT0);
  Mask cncMask(&mMask);
  Mat<float> traceSdMin(numWells, loadMinFlows);
  traceSdMin.zeros();
  // Ready in the sdata data
  std::vector<LoadDatJob> loadJobs(loadMinFlows);
  char buff[resultsSuffix.size() + resultsRoot.size() + 21];
  const char *p = resultsRoot.c_str();
  const char *s = resultsSuffix.c_str();
  // Read following flows multithreaded
  for (size_t i = 0; i < loadMinFlows; i++) {
    //    traceStore.SetFlowIndex (i, i);
    buff[resultsSuffix.size() + resultsRoot.size() + 21];
    p = resultsRoot.c_str();
    s = resultsSuffix.c_str();
    snprintf (buff, sizeof (buff), "%s%.4d.%s", p, (int) i, s);
    printf("Loading %s\n",buff);

    DumpStep(buff, i, mMask.W(), mMask.H(), &opts);

    loadJobs[i].Init(buff, i, &traceStore, traceSdMin.colptr(i), &mMask, &cncMask, &mFilteredWells, &mT0, &opts);
    //  loadJobs[i].Run();
    jQueue.AddJob(loadJobs[i]);
  }
  jQueue.WaitUntilDone();
  traceBuffer.resize(traceStore.GetNumFrames());
  //  traceStore.SplineLossyCompress("explicit:2,4,5,6,8,10,12,14,16,18,20", 4, &traceSd[0]);    
  //  traceStore.SplineLossyCompress("explicit:4,8,12,16,20", 4, &traceSd[0]);    
  loadTimer.PrintMicroSecondsUpdate(stdout, "Load Timer: Dats loaded");
  // ofstream storeOut("trace-store.txt");
  // traceStore.Dump(storeOut);
  // storeOut.close();
  //  cout << "Done loading all traces - took: " << allLoaded.elapsed() <<  " seconds." << endl;
  mRefWells.resize(numWells, 0);

  traceSd.resize(numWells);
  int non_filt_wells = 0;
  for (size_t i = 0; i < traceSdMin.n_rows; i++) {
    float mean = 0;
    if (mFilteredWells[i] == GoodWell) {
      non_filt_wells++;
    }
    for (size_t j = 0; j < traceSdMin.n_cols; j++) {
      mean += traceSdMin(i,j);
    }
    traceSd[i] = sqrt(mean / traceSdMin.n_cols);
  }
  std::vector<float> comp_mad_stats;
  comp_mad_stats.reserve(non_filt_wells);
  for(size_t i = 0; i < traceSd.size(); i++) {
    if (mFilteredWells[i] == GoodWell) {
      comp_mad_stats.push_back(traceSd[i]);
    }
  }
  if (comp_mad_stats.size() > MIN_SAMPLE_TAUE_STATS) {
    std::sort(comp_mad_stats.begin(), comp_mad_stats.end());
    float comp_mad_75 = ionStats::quantile_sorted(comp_mad_stats, .75);
    float comp_mad_25 = ionStats::quantile_sorted(comp_mad_stats, .25);
    float comp_mad_threshold = comp_mad_75 + 3 * (comp_mad_75 - comp_mad_25);
    int not_compressable = 0;
    for (size_t i = 0; i < traceSd.size(); i++) {
      if (mFilteredWells[i] == GoodWell && traceSd[i] > comp_mad_threshold) {
        mFilteredWells[i] = NotCompressable;
        not_compressable++;
      }
    }
    fprintf(stdout, "Got %d uncompressable wells for %.2f +/- %.2f\n", not_compressable, ionStats::median(comp_mad_stats), comp_mad_75 - comp_mad_25);
  }
  else {
    fprintf(stdout, "Not enough wells for compression statisticss.\n");
  }
    //  CountReference("After beadfind reference", mFilteredWells);
  size_t sdCount = 0;
  size_t blobCount = 0;
  for(size_t i = 0; i < mFilteredWells.size(); i++) {
    if (mFilteredWells[i] == LowTraceSd) {
      sdCount++;
    }
    if (mFilteredWells[i] ==  RegionTraceSd) {
      blobCount++;
    }
  }
  cout << "Filtered: " << sdCount << " wells based on sd, " << blobCount << " in blobs" << endl;
  loadTimer.PrintMicroSecondsUpdate(stdout, "Load Timer: Reference Prepared");  
}

void DifferentialSeparator::OutputWellInfo (TraceStore &store,
                                            ZeromerDiff<float> &bg,
					    const vector<KeyFit> &wells,
					    int outlierType,
					    int wellIdx,
					    std::ostream &traceOut,
					    std::ostream &refOut,
					    std::ostream &bgOut)
{
  char d = '\t';
  const KeyFit &w = wells[wellIdx];
  vector<float> f (store.GetNumFrames());
  Col<float> ref (store.GetNumFrames());
  Col<float> p (store.GetNumFrames());
  for (size_t flow = 0; flow < 8 && store.HaveFlow (flow); flow++)
    {
      traceOut << w.wellIdx << d << outlierType << d << flow << d << (int) w.keyIndex << d << w.snr << d << w.bfMetric << d << w.peakSig << d << w.mad;
      refOut   << w.wellIdx << d << outlierType << d << flow << d << (int) w.keyIndex << d << w.snr << d << w.bfMetric << d << w.peakSig << d << w.mad;
      bgOut    << w.wellIdx << d << outlierType << d << flow << d << (int) w.keyIndex << d << w.snr << d << w.bfMetric << d << w.peakSig << d << w.mad;
      store.GetTrace (w.wellIdx, flow, f.begin());
      for (size_t fIx = 0; fIx < f.size(); fIx++)
	{
	  traceOut << d << f[fIx];
	}
      traceOut << endl;

      store.GetReferenceTrace (w.wellIdx, flow, ref.begin());
      for (size_t fIx = 0; fIx < ref.n_rows; fIx++)
	{
	  refOut << d << ref.at (fIx);
	}
      refOut << endl;

      bg.PredictZeromer (ref, mTime, w.tauB, w.tauE, p);
      for (size_t fIx = 0; fIx < p.n_rows; fIx++)
	{
	  bgOut << d << p.at (fIx);
	}
      bgOut << endl;
    }

}

void DifferentialSeparator::OutputOutliers (TraceStore &store,
                                            ZeromerDiff<float> bg,
					    const vector<KeyFit> &wells,
					    int outlierType,
					    const vector<int> &outputIdx,
					    std::ostream &traceOut,
					    std::ostream &refOut,
					    std::ostream &bgOut
					    )
{
  for (size_t i = 0; i < outputIdx.size(); i++)
    {
      if (store.HaveWell (outputIdx[i]))
	{
	  OutputWellInfo (store, bg, wells, outlierType, outputIdx[i], traceOut, refOut, bgOut);
	}
    }
}


void DifferentialSeparator::OutputOutliers (DifSepOpt &opts, TraceStore &store,
                                            ZeromerDiff<float> bg,
					    const vector<KeyFit> &wells,
					    double sdNoKeyHighT, double sdKeyLowT,
					    double madHighT, double bfNoKeyHighT, double bfKeyLowT,
					    double lowKeySignalT)
{
  int nSample = 50;
  ReservoirSample<int> sdNoKeyHigh (nSample);
  ReservoirSample<int> sdKeyLow (nSample);
  ReservoirSample<int> madHigh (nSample);
  ReservoirSample<int> bfNoKeyHigh (nSample);
  ReservoirSample<int> bfKeyLow (nSample);
  ReservoirSample<int> libOk (nSample);
  ReservoirSample<int> tfOk (nSample);
  ReservoirSample<int> emptyOk (nSample);
  ReservoirSample<int> lowKeySignal (nSample);
  ReservoirSample<int> wellLowSignal (nSample);
  string traceOutFile = opts.outData + ".outlier-trace.txt";
  string refOutFile = opts.outData + ".outlier-ref.txt";
  string bgOutFile = opts.outData + ".outlier-bg.txt";
  ofstream traceOut;
  ofstream refOut;
  ofstream bgOut;
  traceOut.open (traceOutFile.c_str());
  refOut.open (refOutFile.c_str());
  bgOut.open (bgOutFile.c_str());

  for (size_t i = 0; i < wells.size(); i++)
    {
      if (wells[i].flag == WellEmpty)
	{
	  emptyOk.Add (wells[i].wellIdx);
	}
      if (wells[i].flag == WellLib)
	{
	  libOk.Add (wells[i].wellIdx);
	}
      if (wells[i].flag == WellTF)
	{
	  tfOk.Add (wells[i].wellIdx);
	}
      if (wells[i].flag == WellEmpty && wells[i].sd >= sdNoKeyHighT)
	{
	  sdNoKeyHigh.Add (wells[i].wellIdx);
	}
      if ( (wells[i].flag == WellLib || wells[i].flag == WellTF) && wells[i].sd <= sdKeyLowT)
	{
	  sdKeyLow.Add (wells[i].wellIdx);
	}
      if (wells[i].mad >= madHighT)
	{
	  madHigh.Add (wells[i].wellIdx);
	}
      if (wells[i].flag == WellEmpty && wells[i].bfMetric >= bfNoKeyHighT)
	{
	  bfNoKeyHigh.Add (wells[i].wellIdx);
	}
      if ( (wells[i].flag == WellLib || wells[i].flag == WellTF) && wells[i].bfMetric <= bfKeyLowT)
	{
	  bfKeyLow.Add (wells[i].wellIdx);
	}
      if ( (wells[i].flag == WellLib || wells[i].flag == WellTF) && wells[i].peakSig <= lowKeySignalT)
	{
	  lowKeySignal.Add (wells[i].wellIdx);
	}
      if (wells[i].flag == WellLowSignal)
	{
	}
    }

  sdNoKeyHigh.Finished();
  sdKeyLow.Finished();
  madHigh.Finished();
  bfNoKeyHigh.Finished();
  bfKeyLow.Finished();
  libOk.Finished();
  tfOk.Finished();
  emptyOk.Finished();
  lowKeySignal.Finished();
  wellLowSignal.Finished();
  OutputOutliers (store, bg, wells, SdNoKeyHigh, sdNoKeyHigh.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, SdKeyLow, sdKeyLow.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, MadHigh, madHigh.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, BfNoKeyHigh, bfNoKeyHigh.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, BfKeyLow, bfKeyLow.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, LibKey, libOk.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, EmptyWell, emptyOk.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, TFKey, tfOk.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, LowKeySignal, lowKeySignal.GetData(), traceOut, refOut, bgOut);
  OutputOutliers (store, bg, wells, KeyLowSignalFilt, wellLowSignal.GetData(), traceOut, refOut, bgOut);
}

/*
 * For each region calculate:
 * 1) the variance of the variance of the individual frames.
 * 2) The 25th,50th and 75th quantiles frame wise
 */
void DifferentialSeparator::CalcRegionEmptyStat(H5File &h5File, GridMesh<MixModel> &mesh, TraceStore &store, const string &fileName, 
                                                vector<int> &flows, Mask &mask) {
  fmat refVarSummary, refFrameIqr, bgVarSummary, bgFrameIqr;
  refVarSummary.set_size(mesh.GetNumBin() * flows.size(), 9); // rowstart, rowend, colstart, colend, count var of var, mean of var, mean iqr
  bgVarSummary.set_size(mesh.GetNumBin() * flows.size(), 9);
  refFrameIqr.set_size(3 * mesh.GetNumBin() * flows.size(), store.GetNumFrames() + 7);
  bgFrameIqr.set_size(3 * mesh.GetNumBin() * flows.size(), store.GetNumFrames() + 7);
  refVarSummary.fill(0);
  refFrameIqr.fill(0);
  bgVarSummary.fill(0);
  bgFrameIqr.fill(0);
  vector<float> wellData(store.GetNumFrames());
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);

    for (size_t flow = 0; flow < flows.size(); flow++) {
      vector<SampleQuantiles<float> > refFrameQuantiles(store.GetNumFrames());
      vector<SampleQuantiles<float> > bgFrameQuantiles(store.GetNumFrames());
      vector<SampleStats<float> > refFrameStats(store.GetNumFrames());
      vector<SampleStats<float> > bgFrameStats(store.GetNumFrames());
      for (size_t i = 0; i < refFrameQuantiles.size(); i++) {
        refFrameQuantiles[i].Init(1000);
        bgFrameQuantiles[i].Init(1000);
      }
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          size_t wellIx = store.WellIndex(row, col);
          bool needToLoad = true;
          if (store.IsReference(wellIx)) {
            store.GetTrace(wellIx, flows[flow], wellData.begin());
            needToLoad = false;
            for (size_t frame = 0; frame < store.GetNumFrames(); frame++) {
              refFrameQuantiles[frame].AddValue(wellData[frame]);
              refFrameStats[frame].AddValue(wellData[frame]);
            }
          }
          if (mask[wellIx] & MaskReference) {
            if (needToLoad) {
              store.GetTrace(wellIx, flows[flow], wellData.begin());
              needToLoad = false;
            }
            for (size_t frame = 0; frame < store.GetNumFrames(); frame++) {
              bgFrameQuantiles[frame].AddValue(wellData[frame]);
              bgFrameStats[frame].AddValue(wellData[frame]);
            }
          }
        }
      }
      size_t wIdx = flow * mesh.GetNumBin() + binIx;
      {
	int col = 0;
	float quantiles[3] = {.25,.5,.75};
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = rowStart;    col++;
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = rowEnd;      col++;
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = colStart;    col++;
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = colEnd;      col++;
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = flows[flow]; col++;
	bgVarSummary(wIdx,col) = refVarSummary(wIdx, col) = refFrameStats[0].GetCount(); col++;
	for (size_t q = 0; q < ArraySize(quantiles); q++) {
	  size_t qIdx = q + binIx * ArraySize(quantiles) + flow * ArraySize(quantiles) * mesh.GetNumBin();
	  size_t lcol = 0;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = rowStart;    lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = rowEnd;      lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = colStart;    lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = colEnd;      lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = flows[flow]; lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = bgFrameStats[0].GetCount(); lcol++;
	  bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = quantiles[q];
	}

	SampleStats<float> refVar, refIqr, bgVar, bgIqr;
	for (size_t frame = 0; frame < store.GetNumFrames(); frame++) {
	  for (size_t q = 0; q < ArraySize(quantiles); q++) {
	    size_t qIdx = q + binIx * ArraySize(quantiles) + flow * ArraySize(quantiles) * mesh.GetNumBin();
	    if (refFrameQuantiles[frame].GetNumSeen() > 2) {
	      refFrameIqr(qIdx, col + 1 + frame) = refFrameQuantiles[frame].GetQuantile(quantiles[q]);
	    }
	    if (bgFrameQuantiles[frame].GetNumSeen() > 2) {
	      bgFrameIqr(qIdx, col + 1 + frame) = bgFrameQuantiles[frame].GetQuantile(quantiles[q]);
	    }
	  }
	  if (refFrameQuantiles[frame].GetNumSeen() > 2) {
	    refVar.AddValue(refFrameStats[frame].GetSD());
	    refIqr.AddValue(refFrameQuantiles[frame].GetIQR());
	  }
	  if (bgFrameQuantiles[frame].GetNumSeen() > 2) {
	    bgVar.AddValue(bgFrameStats[frame].GetSD());
	    bgIqr.AddValue(bgFrameQuantiles[frame].GetIQR());
	  }
	}
	refVarSummary(wIdx, col) = refVar.GetSD();
	bgVarSummary(wIdx, col) = bgVar.GetSD();
	col++;
	refVarSummary(wIdx, col) = refVar.GetMean();
	bgVarSummary(wIdx, col) = bgVar.GetMean();
	col++;
	refVarSummary(wIdx, col) = refIqr.GetMean();
	bgVarSummary(wIdx, col) = bgIqr.GetMean();
	col++;
      }
    }
  }    
  string refVarSummaryH5 = "/separator/refVarSummary";
  H5Arma::WriteMatrix(h5File, refVarSummaryH5, refVarSummary);
  string refFrameStats = "/separator/refFrameStats";
  H5Arma::WriteMatrix(h5File, refFrameStats, refFrameIqr);
  string bgVarSummaryH5 = "/separator/bgVarSummary";
  H5Arma::WriteMatrix(h5File, bgVarSummaryH5, bgVarSummary);
  string bgFrameStats = "/separator/bgFrameStats";
  H5Arma::WriteMatrix(h5File, bgFrameStats, bgFrameIqr);
}

void DifferentialSeparator::FitKeys(DifSepOpt &opts, GridMesh<struct FitTauEParams> &emptyEstimates, 
                                    TraceStoreCol &traceStore, std::vector<KeySeq> &keys, 
                                    std::vector<float> &ftime, TraceSaver &saver,
                                    Mask &mask, std::vector<KeyFit> &wells) {
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Starting Calling Keys.");
  int numWells = mask.W() * mask.H();
  EvaluateKey evaluator;
  evaluator.m_doing_darkmatter = true;
  evaluator.m_peak_signal_frames = true;
  evaluator.m_integration_width = true;
  evaluator.m_normalize_1mers = true;
  //  evaluator.m_use_projection = true;
  int usable_flows = 0;
  for (size_t i = 0; i < keys.size(); i++) {
    usable_flows = max(usable_flows, (int)keys[i].usableKeyFlows);
  }

  if (opts.outputDebug > 3) {
    saver.Alloc(mask.W(), numWells, traceStore.GetNumFrames(),usable_flows, 10);
  }

  SampleQuantiles<float> tauE_stats(1000);
  SampleQuantiles<float> ref_shift_stats(1000);
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    struct FitTauEParams &param = emptyEstimates.GetItem(binIx);
    if (param.converged > 0) {
      tauE_stats.AddValue(param.taue);
      ref_shift_stats.AddValue(param.ref_shift);
    }
  }

  if (tauE_stats.GetNumSeen() == 0) { // nothing converged just use the median...
    for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      struct FitTauEParams &param = emptyEstimates.GetItem(binIx);
      if (isfinite(param.taue)) {
        tauE_stats.AddValue(param.taue);
      }
      if (isfinite(param.ref_shift)) {
        ref_shift_stats.AddValue(param.ref_shift);
      }
    }
  }

  std::vector<double> taue_dist(7);
  std::vector<struct FitTauEParams *> taue_values(7);
  std::vector<int> flow_order(traceStore.GetNumFlows(),-1);
  for (size_t i = 0; i < traceStore.GetNumFlows(); i++) {
    flow_order[i] = traceStore.GetNucForFlow(i);
  }
  KeySummaryReporter<double> keySumReport;
  keySumReport.Init (opts.flowOrder, opts.analysisDir, mask.H(), mask.W(), traceStore.GetNumFrames(),
                     std::min (128,mask.W()), std::min (128,mask.H()), keys);
  AvgKeyReporter<double> avgReport(keys, opts.outData, opts.flowOrder, opts.analysisDir, 
                                   usable_flows, traceStore.GetNumFrames());
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    int query_row = (rowStart + rowEnd) / 2;
    int query_col = (colStart + colEnd) / 2;
    emptyEstimates.GetClosestNeighbors(query_row, query_col, opts.useMeshNeighbors * 2, taue_dist, taue_values);
    struct FitTauEParams mean_taue;
    mean_taue.taue = 0;
    mean_taue.ref_shift = 0;
    int taue_count = 0;
    for (size_t i = 0; i < taue_values.size(); i++) {
      if (taue_values[i]->converged > 0) {
        mean_taue.taue += taue_values[i]->taue;
        mean_taue.ref_shift += taue_values[i]->ref_shift;
        taue_count++;
      }
    }
    if (taue_count > 0) {
      mean_taue.taue /= taue_count;
      mean_taue.ref_shift /= taue_count;
    }
    else if (taue_count == 0 && tauE_stats.GetNumSeen() > 0 && ref_shift_stats.GetNumSeen() > 0) {
      mean_taue.taue = tauE_stats.GetMedian();
      mean_taue.ref_shift = ref_shift_stats.GetMedian();
    }
    else {
      mean_taue.taue = 3.0f;
      mean_taue.ref_shift = 0.0f;
    }

    double taue_weight = 0;
    evaluator.SetSizes(rowStart, rowEnd,
                       colStart, colEnd,
                       0, usable_flows,
                       0, traceStore.GetNumFrames());
    
    if (opts.outputDebug > 3) {
      evaluator.m_debug = true;
    }
    evaluator.m_flow_order = flow_order;
    evaluator.m_doing_darkmatter = true;
    evaluator.m_peak_signal_frames = true;
    evaluator.m_integration_width = true;
    //    evaluator.m_use_projection = true;
    evaluator.SetUpMatrices(traceStore, mask.W(), 0, 
                            rowStart, rowEnd,
                            colStart, colEnd,
                            0, usable_flows,
                            0, traceStore.GetNumFrames());
    evaluator.FindBestKey(rowStart, rowEnd,
                          colStart, colEnd,
                          0, usable_flows,
                          0, mask.W(), &ftime[0],
                          keys, mean_taue.taue, mean_taue.ref_shift,
                          &mFilteredWells[0],
                          wells);
    keySumReport.Report(evaluator);
    avgReport.Report(evaluator);
    if (opts.outputDebug > 3) {
      saver.StoreResults(rowStart, rowEnd, colStart, colEnd, 0, usable_flows, evaluator);
    }
  }
  avgReport.Finish();
  keySumReport.Finish();
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Finished Calling Keys.");
}

void DifferentialSeparator::FitKeys(PJobQueue &jQueue, DifSepOpt &opts, GridMesh<struct FitTauEParams> &emptyEstimates, 
                                    TraceStoreCol &traceStore, std::vector<KeySeq> &keys, 
                                    std::vector<float> &ftime, TraceSaver &saver,
                                    Mask &mask, std::vector<KeyFit> &wells) {
  int usable_flows = 0;
  for (size_t i = 0; i < keys.size(); i++) {
    usable_flows = max(usable_flows, (int)keys[i].usableKeyFlows);
  }

  int numWells = mask.W() * mask.H();
  if (opts.outputDebug > 3) {
    saver.Alloc(mask.W(), numWells, traceStore.GetNumFrames(), usable_flows, 10);
  }

  SampleQuantiles<float> tauE_stats(1000);
  SampleQuantiles<float> ref_shift_stats(1000);
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    struct FitTauEParams &param = emptyEstimates.GetItem(binIx);
    if (param.converged > 0) {
      tauE_stats.AddValue(param.taue);
      ref_shift_stats.AddValue(param.ref_shift);
    }
  }

  if (tauE_stats.GetNumSeen() == 0) { // nothing converged just use the median...
    for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      struct FitTauEParams &param = emptyEstimates.GetItem(binIx);
      if (isfinite(param.taue)) {
        tauE_stats.AddValue(param.taue);
      }
      if (isfinite(param.ref_shift)) {
        ref_shift_stats.AddValue(param.ref_shift);
      }
    }
  }

  struct FitTauEParams defaultParam;
  if (tauE_stats.GetNumSeen() > 0 && ref_shift_stats.GetNumSeen() > 0) {
    defaultParam.taue = tauE_stats.GetMedian();
    defaultParam.ref_shift = ref_shift_stats.GetMedian();
  }
  else {
    defaultParam.taue = 3.0f;
    defaultParam.ref_shift = 0.0f;
  }

  std::vector<double> taue_dist(7);
  std::vector<struct FitTauEParams *> taue_values(7);
  std::vector<int> flow_order(traceStore.GetNumFlows(),-1);
  for (size_t i = 0; i < traceStore.GetNumFlows(); i++) {
    flow_order[i] = traceStore.GetNucForFlow(i);
  }
  KeySummaryReporter<double> keySumReport;
  keySumReport.Init (opts.flowOrder, opts.analysisDir, mask.H(), mask.W(), traceStore.GetNumFrames(),
                     std::min (128,mask.W()), std::min (128,mask.H()), keys);
  AvgKeyReporter<double> avgReport(keys, opts.outData, opts.flowOrder, opts.analysisDir, 
                                   usable_flows, traceStore.GetNumFrames());
  std::vector<EvalKeyJob> evalJobs(emptyEstimates.GetNumBin());
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    evalJobs[binIx].Init(rowStart, rowEnd, colStart, colEnd, &mask, &opts,
                         &mFilteredWells, &wells, &keys, &ftime,
                         &saver, &traceStore, &emptyEstimates, 
                         &keySumReport, &avgReport, &defaultParam);
    jQueue.AddJob(evalJobs[binIx]);
  }
  jQueue.WaitUntilDone();
  avgReport.Finish();
  keySumReport.Finish();

}


void DifferentialSeparator::FitTauE(DifSepOpt &opts, TraceStoreCol &traceStore, GridMesh<struct FitTauEParams> &emptyEstimates,
                                    std::vector<char> &filteredWells, std::vector<float> &ftime, std::vector<int> &allZeroFlows, float *taub_est) {
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before Zeromers.");
  emptyEstimates.Init (mMask.H(), mMask.W(), opts.tauEEstimateStepY, opts.tauEEstimateStepX);
  ZeromerMatDiff z_diff;
  ZeromerMatDiff z_diff_big;
  int converged = 0;
  int no_wells = 0;
  int zero_flows[1] = {0};
  if (taub_est != NULL) {  memset(taub_est, 0, sizeof(float) * filteredWells.size()); }
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    z_diff.SetUpMatricesClean(traceStore, &filteredWells[0], &ftime[0], 2, 3, 
                              mMask.W(), mMask.W() * mMask.H(),
                              rowStart, rowEnd, colStart, colEnd,
                              &allZeroFlows[0], allZeroFlows.size(),
                              0, traceStore.GetNumFrames());
    struct FitTauEParams &param = emptyEstimates.GetItem(binIx);
    if (z_diff.m_num_wells < MIN_SAMPLE_TAUE_STATS) {
      param.taue = std::numeric_limits<float>::quiet_NaN();
      param.ref_shift = std::numeric_limits<float>::quiet_NaN();
      param.converged = false;
      no_wells++;
    }
    else {
      // @todo cws - expose these magic params somewhere
      struct FitTauEParams taue_param;
      taue_param.ref_shift = 0;
      taue_param.taue = 3.0f;
      taue_param.converged = 0.0f;
      struct FitTauEParams taue_param_min;
      taue_param_min.taue = 1;
      taue_param_min.ref_shift = -.05;
      taue_param_min.converged = 0.0f;
      struct FitTauEParams taue_param_max;
      taue_param_max.taue = 8;
      taue_param_max.ref_shift = .05f;
      taue_param_max.converged = 0.0f;
      TauEFitter taue_fitter(z_diff.m_total_size, z_diff.m_trace_data, &z_diff);
      taue_fitter.SetParamMax(taue_param_max);
      taue_fitter.SetParamMin(taue_param_min);
      taue_fitter.SetInitialParam(taue_param);
      taue_fitter.Fit(true, 100, z_diff.m_trace_data);
      taue_param.ref_shift = taue_fitter.m_params.ref_shift;
      taue_param.taue = taue_fitter.m_params.taue;
 
      taue_param.converged = taue_fitter.IsConverged() ? 1.0f : 0.0f;
      param = taue_param;
      // fprintf(stdout, "Fitted params: %d %d %.2f %.2f %f\n", rowStart, colStart,
      //        param.taue, param.ref_shift, param.converged);
      if (param.converged) {
        converged++;
      }
    }
    if (taub_est != NULL) {
      z_diff_big.SetUpMatrices(traceStore, &filteredWells[0], &ftime[0], 1, 1, 
                               mMask.W(), mMask.W() * mMask.H(),
                               rowStart, rowEnd, colStart, colEnd,
                               allZeroFlows[0], allZeroFlows[0] + 1,
                               0, traceStore.GetNumFrames());
      z_diff_big.FitTauB(zero_flows, 1,
                         z_diff_big.m_trace_data, z_diff_big.m_ref_data,
                         z_diff_big.m_num_wells, z_diff_big.m_num_flows, z_diff_big.m_num_well_flows,
                         z_diff_big.m_total_size / z_diff_big.m_num_well_flows,
                         param.taue, z_diff_big.m_taub);
      int chip_col = mMask.W();
      float *__restrict local_taub = z_diff_big.m_taub;
      for (int row_ix = rowStart; row_ix < rowEnd; row_ix++) {
        for (int col_ix = colStart; col_ix < colEnd; col_ix++) {
          int well_ix = row_ix * chip_col + col_ix;
          float value = *local_taub++;
          if (value > 0) {
            taub_est[well_ix] = value;
          }
          else {
            filteredWells[well_ix] = LowTraceSd;
          }
        }
      }
    }
  }
  fprintf(stdout, "FitTauE() - %d %d %d\n", converged, no_wells, (int)emptyEstimates.GetNumBin());
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After well Zeromers.");
}

void DifferentialSeparator::DoRegionClustering(DifSepOpt &opts, Mask &mask, vector<float> &bfMetric, float madThreshold,
                                               std::vector<KeyFit> &wells, GridMesh<MixModel> &modelMesh) {
  int numWells = mask.H() * mask.W();
  ofstream modelOut;
  if (opts.outputDebug > 1) {
    string modelFile = opts.outData + ".mix-model.txt";
    modelOut.open(modelFile.c_str());
    modelOut << "bin\tbinRow\tbinCol\trowStart\trowEnd\tcolStart\tcolEnd\tcount\tmix\tmu1\tvar1\tmu2\tvar2\tthreshold\trefMean" << endl;
  }
  
  modelMesh.Init (mask.H(), mask.W(), opts.clusterMeshStepY, opts.clusterMeshStepX);
  SampleStats<double> bfSnr;

  //  double bfMinThreshold = bfQuantiles.GetQuantile (.02);
  //  cout << "Bf min threshold is: " << bfMinThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " << ( (bfQuantiles.GetQuantile (.75) - bfQuantiles.GetQuantile (.25)) /2) << endl;

  // Should we use standard deviation of signal as the beadfind metric
  // (eg cluster signal vs no signal instead of buffering vs no
  // buffering.


  // Anchor clusters based off of set of good live beads and the empty wells.
  for (int i = 0; i < numWells; i++) {
    if (wells[i].snr >= opts.minTauESnr && wells[i].peakSig > RELIABLE_PEAK_HEIGHT) {
      wells[i].goodLive = true;
    }
  }

  for (size_t binIx = 0; binIx < modelMesh.GetNumBin(); binIx++)
    {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      modelMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      MixModel &model = modelMesh.GetItem (binIx);
      // Center the reference wells in this region to make things comparable.
      // double meanBuff = 0;
      // size_t meanCount = 0;
      int goodCount = 0;
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          size_t idx = row * mask.W() + col;
          if (mFilteredWells[idx] == 0) {
            goodCount++;
          }
          // if (wells[idx].isRef) {
          //   meanCount++;
          //   meanBuff += wells[idx].bfMetric;
          // }
        }
      }
      // if (meanCount > 0) {
      //   meanBuff /= meanCount;
      // }
      // for (int row = rowStart; row < rowEnd; row++) {
      //   for (int col = colStart; col < colEnd; col++) {
      //     size_t idx =  row * mask.W() + col;
      //     wells[idx].bfMetric -= meanBuff;
      //   }
      // }
      int minBfGoodWells = max (200, (int) (goodCount * .5));
      ClusterRegion (rowStart, rowEnd, colStart, colEnd, madThreshold, opts.minTauESnr,
		     minBfGoodWells, bfMetric, wells, opts.clusterTrim, false,  model);
      if ( model.count > minBfGoodWells) {
        double bf = ( (model.mu2 - model.mu1) / ( (sqrt (model.var2) + sqrt (model.var1)) /2));
        if (isfinite (bf) && bf > 0)  {
          bfSnr.AddValue (bf);
        }
        else  {
          cout << "Region: " << binIx << " has snr of: " << bf << " " << model.mu1 << "  " << model.var1 << " " << model.mu2 << " " << model. var2 << endl;
        }
      }
      int binRow, binCol;
      modelMesh.IndexToXY (binIx, binRow, binCol);
      if (opts.outputDebug > 1) {
	modelOut << binIx << "\t" << binRow << "\t" << binCol << "\t"
		 << rowStart << "\t" << rowEnd << "\t" << colStart << "\t" << colEnd << "\t"
		 << model.count << "\t" << model.mix << "\t"
		 << model.mu1 << "\t" << model.var1 << "\t"
		 << model.mu2 << "\t" << model.var2 << "\t"
		 << model.threshold << "\t" << model.refMean << endl;
      }
    }
  cout << "BF SNR: " << bfSnr.GetMean() << " +/- " << (bfSnr.GetSD()) << endl;
  if (modelOut.is_open()) {
    modelOut.close();
  }
}


void DifferentialSeparator::ClusterIndividualWells(DifSepOpt &opts, Mask &bfMask, Mask &mask, TraceStoreCol &traceStore,
                                                   GridMesh<MixModel> &modelMesh, std::vector<KeyFit> &wells, 
                                                   std::vector<float> &confidence, std::vector<char> &clusters) {
  bfMask.Init (&mask);
  int notGood = 0;
  vector<MixModel *> bfModels;
  std::vector<double> dist;
  bool cluster_diagnostics = opts.outputDebug > 0;
  if (cluster_diagnostics) {
    confidence.resize(mask.H() * mask.W());
    std::fill(confidence.begin(), confidence.end(), -1.0f);
  }

  // For speed create a fine grain mesh where we have pre averaged the smoothing
  GridMesh<MixModel> fine_mesh;
  fine_mesh.Init(mask.H(), mask.W(), opts.clusterFineMeshStep, opts.clusterFineMeshStep);
  for (size_t binIx = 0; binIx < fine_mesh.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    fine_mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    int row_center = (rowStart + rowEnd) / 2;
    int col_center = (colStart + colEnd) / 2;
    int good = 0;
    modelMesh.GetClosestNeighbors (row_center, col_center, opts.bfNeighbors, dist, bfModels);
    MixModel &m = fine_mesh.GetItem(binIx);
    m.mu1 = m.mu2 = m.var1 = m.var2 = m.mix = 0.0f;
    m.count = 0;
    m.thresholdSet = false;
    float weight = 0.0f;
    for (size_t i = 0; i < bfModels.size(); i++) {
      if ( (size_t) bfModels[i]->count > opts.minBfGoodWells) {
        good++;
        float w = 1.0/ (log (dist[i] + 2.5));
        weight += w;
        m.mu1 += w * bfModels[i]->mu1;
        m.mu2 += w * bfModels[i]->mu2;
        m.var1 += w * bfModels[i]->var1;
        m.var2 += w * bfModels[i]->var2;
        m.mix += w * bfModels[i]->mix;
        m.count += w * bfModels[i]->count;
      }
    }
    if (weight > 0) {
      m.mu1 = m.mu1 / weight;
      m.mu2 = m.mu2 / weight;
      m.var1 = m.var1 / weight;
      m.var2 = m.var2 / weight;
      m.mix = m.mix / weight;
      m.count = m.count / weight;
      m.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * m.var1);
      m.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * m.var2);
    }
    DualGaussMixModel::SetThreshold(m);
  }

  // Just look up the well's location in pre calculated fine grain mesh and use that for estimation
  size_t row = 0, col = 0;
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    bfMask[bIx] = mask[bIx];
    if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned || mFilteredWells[bIx] != GoodWell) {
      continue;
    }
    if (cluster_diagnostics || wells[bIx].keyIndex < 0) {
      traceStore.WellRowCol (wells[bIx].wellIdx, row, col);
      MixModel &m = fine_mesh.GetItemByRowCol(row, col);
      double ownership = 0;
      int bCluster = 0;
      if (m.var1 > 0.0f) {
        bCluster = DualGaussMixModel::PredictCluster (m, wells[bIx].bfMetric, opts.bfThreshold, ownership);
        if (cluster_diagnostics) {
          float conf = 0;
          DualGaussMixModel::CalculateResponsibility(m, wells[bIx].bfMetric, conf);
          confidence[bIx] = std::max(1.0f - conf, conf);
        }
      }
      else {
        bCluster = 0;
      }
      clusters[bIx] = bCluster;
    }
  }
}

void DifferentialSeparator::AssignAndCountWells(DifSepOpt &opts, std::vector<KeyFit> &wells, Mask &bfMask, 
                                                std::vector<char> &filteredWells, int minLibPeak, int minTfPeak,
                                                float sepRefSdThresh, float madThreshold ) {
  int overMaxMad = 0,  tooVar = 0, badFit = 0, noTauE = 0, varMinCount = 0, traceMeanMinCount = 0, sdRefCalled = 0, badSignal = 0, beadLow = 0;
  int tooBf = 0;
  int tooTauB = 0;
  int poorSignal = 0;
  int poorLibPeakSignal = 0;
  int poorTfPeakSignal = 0;
  int emptyWithSignal = 0;
  int noisyColumn = 0;
  int tooRefVar = 0;
  int tooRefBuffer = 0;
  int filtWells = 0;
   // CountReference("Before counting", filteredWells);
  SampleQuantiles<float> libSnrQuantiles(10000);
  SampleQuantiles<float> tfSnrQuantiles(10000);
  int softFilterCount = 0;
  for (size_t bIx = 0; bIx < wells.size(); bIx++)
    {
      if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned || bfMask[bIx] & MaskIgnore) {
	continue;
      }
      if (mFilteredWells[bIx] == NoisyColumn) {
        bfMask[bIx] = MaskIgnore;
        wells[bIx].keyIndex = -1;
        wells[bIx].flag = WellNoisyColumn;
        noisyColumn++;
        continue;
      }
      // if (wells[bIx].keyIndex < 0 && (filteredWells[bIx] == LowTraceSd || filteredWells[bIx] == PinnedExcluded) ) {
      //   filtWells++;
      //   mBfMask[bIx] = MaskIgnore;
      //   continue;
      // }
      if (wells[bIx].keyIndex < 0 && filteredWells[bIx] != GoodWell) {
        //        filtWells++;
        softFilterCount++;
	bfMask[bIx] = MaskIgnore;
	continue;
      }
      // if ((bfMask[bIx] & MaskEmpty) && (filteredWells[bIx] != GoodWell)) {
      //   bfMask[bIx] &= ~MaskReference;
      //   softFilterCount++;
      //   continue;
      // }
      if (bfMask[bIx] & MaskReference && wells[bIx].sd > sepRefSdThresh) {
        bfMask[bIx] &= ~MaskReference;
        wells[bIx].flag = WellEmptyVar;
        tooRefVar++;
        continue;
      }
      if (bfMask[bIx] & MaskWashout)
	{
	  bfMask[bIx] = MaskIgnore;
	  wells[bIx].flag = WellBadTrace;
	  continue;
	}
      if (bfMask[bIx] & MaskExclude)
	{
	  wells[bIx].flag = WellExclude;
	  continue;
	}
      if (wells[bIx].keyIndex == -2)
	{
	  bfMask[bIx] = MaskIgnore;
	  wells[bIx].flag = WellNoTauE;
	  noTauE++;
	  continue;
	}
      if (wells[bIx].keyIndex == 0) {
        if (isfinite(wells[bIx].snr)) {
          libSnrQuantiles.AddValue(wells[bIx].snr);
        }
        else {
          ION_ABORT("Can't have nan key snr.");
        }
	continue;
      }
      if (wells[bIx].keyIndex == 1) {
	tfSnrQuantiles.AddValue(wells[bIx].snr);
	continue;
      }
      if (opts.doMadFilter && wells[bIx].keyIndex < 0 && (wells[bIx].mad > madThreshold))
        {
          bfMask[bIx] = MaskIgnore;
          wells[bIx].flag = WellMadHigh;
          overMaxMad++;
          continue;
        }
      if (wells[bIx].keyIndex < 0 && wells[bIx].mad < 0)
	{
	  bfMask[bIx] = MaskIgnore;
	  wells[bIx].flag = WellBadFit;
	  badFit++;
	  continue;
	}

    }

  SampleQuantiles<float> bfEmptyQuantiles (10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    if (bfMask[bIx] & MaskEmpty) {
      bfEmptyQuantiles.AddValue (wells[bIx].bfMetric);
    }
  }
  //   CountReference("After filters.", filteredWells);
  // --- Apply filters
  //  double bfThreshold = bfEmptyQuantiles.GetQuantile (.75) + (3 * IQR (bfEmptyQuantiles));
  //  cout << "Bf threshold is: " << bfThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " <<  IQR (bfQuantiles) << endl;
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    if (bfMask[bIx] & MaskExclude) {
      continue;
    }
    if (mFilteredWells[bIx] == NoisyColumn) {
      bfMask[bIx] = MaskIgnore;
      wells[bIx].keyIndex = -1;
      wells[bIx].flag = WellNoisyColumn;
      noisyColumn++;
      continue;
    }
    if (opts.doRemoveLowSignalFilter && (wells[bIx].keyIndex == 0 && (wells[bIx].peakSig < minLibPeak))) {
      wells[bIx].keyIndex = -1;
      wells[bIx].flag = WellLowSignal;
      poorLibPeakSignal++;
      bfMask[bIx] = MaskIgnore;
      continue;
    }
    if (opts.doRemoveLowSignalFilter && ( (wells[bIx].keyIndex == 1 && (wells[bIx].peakSig < minTfPeak)))) {
      wells[bIx].keyIndex = -1;
      wells[bIx].flag = WellLowSignal;
      poorTfPeakSignal++;
      bfMask[bIx] = MaskIgnore;
      continue;
    }

    if (wells[bIx].keyIndex >= 0)
      {
	bfMask[bIx] = MaskBead;
	bfMask[bIx] |= MaskLive;
	if (wells[bIx].keyIndex == 0)
	  {
	    bfMask[bIx] |= MaskLib;
	    wells[bIx].flag = WellLib;
	  }
	else if (wells[bIx].keyIndex == 1)
	  {
	    bfMask[bIx] |= MaskTF;
	    wells[bIx].flag = WellTF;
	  }
      }
    else if (bfMask[bIx] & MaskBead)
      {
	if (opts.noduds)
	  {
	    bfMask[bIx] = MaskBead;
	    bfMask[bIx] |= MaskLive;
	    bfMask[bIx] |= MaskLib;
	    wells[bIx].flag = WellLib;
	  }
	else
	  {
	    bfMask[bIx] |= MaskDud;
	    wells[bIx].flag = WellDud;
	  }
      }
  }
  
  // Mark all the separator's reference wells as reference just for good measure
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    if (mRefWells[bIx] == 1) {
      bfMask[bIx] = MaskReference | MaskEmpty; // reference is a subset of empty
    }
    // If just use the separator reference unset the non- separator references.
    else if (opts.useSeparatorRef) {
      bfMask[bIx] &= ~MaskReference;
    }
  }
  //mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After Post Filtering.");  

  cout << "Lib snr: " << mKeys[0].minSnr << endl;
  if (libSnrQuantiles.GetNumSeen() > 100) {
    cout << "Lib SNR: " << endl;
    for (size_t i = 0; i < 10; i++) {
      cout << i*10 << ":\t" << libSnrQuantiles.GetQuantile(i/10.0) << endl;
    }
  }
  // if (tfSnrQuantiles.GetNumSeen() > 100) {
  //   cout << "Tf SNR: " << endl;
  //   for (size_t i = 0; i < 10; i++) {
  //     cout << i*10 << ":\t" << tfSnrQuantiles.GetQuantile(i/10.0) << endl;
  //   }
  // }
  //  cout << "SD IQR mult: " << opts.sigSdMult << endl;
  cout << "Using: " << opts.bfNeighbors << " bf neighbors (" << opts.useMeshNeighbors << ")" << endl;
  //  cout << badFit << " bad fit "  <<  noTauE << " no tauE. " << varMinCount << " under min sd." << endl;
  //  cout << "Trace sd thresh: " << traceSDThresh << ", mean thresh: " << traceMeanThresh << endl;
  cout << "Ignore: " << overMaxMad << " over max MAD. ( " << madThreshold << " )" << endl;
  //  cout << "Ignore: " << filtWells << " filtered wells. " << endl;
  cout << "Ignore: " << tooRefVar << " too var compared to reference ( " << sepRefSdThresh << " )" << endl;
  cout << "Ignore: " << softFilterCount << " soft filtered empty wells" << endl;
  //  cout << "Ignore: " << tooRefBuffer << " too high bf metric compared to reference ( " << sepRefBfThresh << " )" << endl;
  //  cout << "Ignore: " << tooVar << " too var " << endl;
  //  cout << "Ignore: " << tooBf << " too high bf metric." << endl;
  //  cout << "Ignore: " << tooTauB << " too high tauB metric." << endl;
  //  cout << "Marked: " << poorSignal << " wells as ignore based on poor signal." << endl;
  cout << "Marked: " << poorLibPeakSignal << " lib wells as ignore based on poor peak signal. ( " << minLibPeak << " )" << endl;
  cout << "Marked: " << poorTfPeakSignal << " tf wells as ignore based on poor peak signal. ( " << minTfPeak << " )" << endl;
  cout << "Noisy Column: " << noisyColumn << " wells from noisy columns" << endl;
  //  cout << "Marked: " << emptyWithSignal << " empty wells as ignore based on too much peak signal. (" << peakSigEmptyThreshold << " )" << endl;
  //  cout << "Marked: " << sdRefCalled << " wells as empty based on signal sd." << endl;
  //  cout << "Marked: " << badSignal << " wells ignore based on mean 1mer signal." << endl;
  //  cout << "Marked: " << beadLow << " wells ignore based on low bead mean 1mer signal." << endl;
  //  cout << traceMeanMinCount << " were less than mean threshold. " << notGood << " not good." << endl;


}

void DifferentialSeparator::ReduceMetric(std::vector<float> &metric, int col_ix, 
                                         ChipReduction &reduce, const char *filtered_wells,
                                         arma::Mat<float> &M) {
  reduce.Reset();
  reduce.Reduce(&metric[0], filtered_wells);
  for (size_t i = 0; i < M.n_rows; i++) { M(i,col_ix) = reduce.GetBlockAvg(i, 0); }
  std::fill(metric.begin(), metric.end(), 0.0f);
}


void DifferentialSeparator::SpatialSummary(const std::string &h5_file_name, const std::string &h5path, 
                                           DifSepOpt &opts, Mask &mask, int x_step, int y_step) {
  int x_clip = mask.W();
  int y_clip = mask.H();
  string json_header = "{";
  json_header += " \"row_step\" : " + ToStr(y_step);
  json_header += ", \"col_step\" : " + ToStr(x_step);
  json_header += ", \"headers\" : [\"row_start\",\"row_end\",\"col_start\",\"col_end\"";
  int chip_offset_x = 0, chip_offset_y = 0;
  string first_dat = opts.resultsDir + "/" + opts.acqPrefix + "0000." + opts.datPostfix;
  Image::GetOffsetFromChipPath(first_dat.c_str(), chip_offset_x, chip_offset_y);
  int n_wells = mask.W() * mask.H();
  if (opts.isThumbnail) { x_clip = y_clip = BF_THUMBNAIL_SIZE; }
  ChipReduction reduce(mask.H(), mask.W(), 1,
                       y_step, x_step, y_clip, x_clip, (int)(.2 * y_step * x_step));
  reduce.SetAvgReduce(true);
  vector<float> metric(n_wells, 0.0f);
  vector<char> bad_wells(n_wells, 0);
  arma::Mat<float> M(reduce.GetNumSmoothBlocks(), 24);
  M.zeros();
  // Fill in the row and colum areads
  int row_start, row_end, col_start, col_end;
  for (size_t i = 0; i < M.n_rows; i++) {
    reduce.GetBlockDims(i, row_start, row_end, col_start,col_end);
    M(i,0) = row_start + chip_offset_y;
    M(i,1) = row_end + chip_offset_y;
    M(i,2) = col_start + chip_offset_x;
    M(i,3) = col_end + chip_offset_x;
  }
  int col = 4;
  float cv = 10000.0f; // crazy value limit
  float nan = std::numeric_limits<float>::quiet_NaN();

  json_header += ",\"t0\"";   // t0 4
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mT0[w_ix] > 0 ? mT0[w_ix] : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"snr\"";   // snr 5
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = isfinite(mWells[w_ix].snr) && fabs(mWells[w_ix].snr) < cv ? mWells[w_ix].snr : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"mad\"";   // mad 6
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].mad > 0 && mWells[w_ix].mad < cv ? mWells[w_ix].mad : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"sd\"";    // sd 7
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].sd > 0 && mWells[w_ix].sd < cv ? mWells[w_ix].sd : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"bf_metric\"";   // bf_metric 8
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mask[w_ix] & MaskPinned) == 0 ? mWells[w_ix].bufferMetric : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"tau_e\""; // tauE 9
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].tauE; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"tau_b\""; // tauB 10
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].tauB; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"peak_sig\""; // peakSig 11
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].peakSig < cv ? mWells[w_ix].peakSig : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"trace_sd\""; // traceSd 12
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = mWells[w_ix].traceSd > 0 ? mWells[w_ix].traceSd : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"bead\""; // mask bead 13
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskBead) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"empty\"";   // mask empty 14
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskEmpty) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"ignore\""; // mask ignore 15
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskIgnore) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"pinned\""; // mask pinned 16
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskPinned) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"reference\""; // mask reference 17
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskReference) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"tf\""; // mask tf 18
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskTF) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"lib\""; // mask lib 19
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskLib) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"dud\""; // mask dud 20
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskDud) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"soft_filt\""; // mask pinned 21
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mFilteredWells[w_ix] > 0 && (mBfMask[w_ix] & MaskPinned) == 0) ? 1 : 0; }
  ReduceMetric(metric, col++, reduce, &bad_wells[0], M);

  json_header += ",\"buff_clust_conf\""; // beadfind cluster buffer confidence 22
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskPinned) == 0 ? mBfConfidence[0][w_ix] : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);

  json_header += ",\"sig_clust_conf\""; // beadfind clustering signal confidence 23
  for (int w_ix = 0; w_ix < n_wells; w_ix++) { metric[w_ix] = (mBfMask[w_ix] & MaskPinned) == 0 ? mBfConfidence[1][w_ix] : nan; }
  ReduceMetric(metric, col++, reduce, &mFilteredWells[0], M);
  
  json_header += "] }";
  
  
  // Write out the results
  H5File h5file(h5_file_name);
  h5file.Open(true);
  h5file.WriteString(h5path + "_header", json_header.c_str());
  H5Arma::WriteMatrix (h5file, h5path + "_table", M);
  h5file.Close();
}


void DifferentialSeparator::HandleDebug(std::vector<KeyFit> &wells, 
                                        DifSepOpt &opts, const std::string &h5SummaryRoot, 
                                        TraceSaver &saver, Mask &mask, 
                                        TraceStoreCol &traceStore, GridMesh<MixModel> &modelMesh,
                                        float sdHigh, float sdLow,
                                        float madHigh, float bfHigh, 
                                        float bfLow, float peakLow) {

  if (opts.outputDebug > 1) {
    ZeromerDiff<float> bg;
    size_t numWells = mask.H() *mask.W();
    OutputOutliers (opts, traceStore, bg, wells,
                    sdHigh, sdLow, madHigh, bfHigh,
                    bfLow, peakLow);

    // Write out debugging matrix
    int n_cols = 25;
    bool do_noise = false;
    if (opts.filterNoisyCols != "none") {
      //      n_cols++;
      do_noise = true;
    }
    arma::Mat<float> wellMatrix (numWells, n_cols);
    std::fill (wellMatrix.begin(), wellMatrix.end(), 0.0f);
    for (size_t i = 0; i < numWells; i++) {
      // if (mask[i] & MaskExclude || !zModelBulk.HaveModel (i)) {
      //   continue;
      // }
      int currentCol = 0;
      KeyFit &kf = wells[i];
      //      const KeyBulkFit *kbf = zModelBulk.GetKeyBulkFit (i);
      wellMatrix.at (i, currentCol++) = (int) kf.keyIndex;                               // 0
      wellMatrix.at (i, currentCol++) = mT0[kf.wellIdx];                                  // 1
      wellMatrix.at (i, currentCol++) = kf.snr;                                          // 2
      wellMatrix.at (i, currentCol++) = kf.mad;                                          // 3 
      wellMatrix.at (i, currentCol++) = kf.sd;                                           // 4
      wellMatrix.at (i, currentCol++) = kf.bfMetric;                                     // 5
      wellMatrix.at (i, currentCol++) = kf.tauE;                                         // 6
      wellMatrix.at (i, currentCol++) = kf.tauB;                                         // 7
      wellMatrix.at (i, currentCol++) = kf.onemerAvg;                                    // 8
      wellMatrix.at (i, currentCol++) = kf.bfMetric2;                                    // 9
      wellMatrix.at (i, currentCol++) = kf.peakSig;                                      // 10
      wellMatrix.at (i, currentCol++) = kf.flag;                                         // 11
      wellMatrix.at (i, currentCol++) = kf.goodLive;                                     // 12
      wellMatrix.at (i, currentCol++) = kf.isRef;                                        // 13
      wellMatrix.at (i, currentCol++) = kf.bufferMetric;                                 // 14
      wellMatrix.at (i, currentCol++) = kf.traceSd;                                      // 15
      wellMatrix.at (i, currentCol++) = kf.acqT0;                                        // 16
      wellMatrix.at (i, currentCol++) = mFilteredWells[i];                               // 17
      wellMatrix.at (i, currentCol++) = mBfSdFrame[i];                                   // 18
      wellMatrix.at (i, currentCol++) = mBfSSQ[i];                                       // 19
      wellMatrix.at (i, currentCol++) = mAcqSSQ[i];                                      // 20
      wellMatrix.at (i, currentCol++) = kf.bfMetric3;                             // 21
      wellMatrix.at (i, currentCol++) = mBfConfidence[0][i];                             // 22
      wellMatrix.at (i, currentCol++) = mBfConfidence[1][i];                             // 23
      if (do_noise) {
        wellMatrix.at (i, currentCol++) = mWellNoise[i];                                 // 24
      }
      else {
        wellMatrix.at (i, currentCol++) = mEmptyMetrics.back().at(i);                             // 24
      }
    }

    string h5Summary = "/separator/summary";
    vector<int> flows(2);
    flows[0] = opts.maxKeyFlowLength;
    flows[1] = opts.maxKeyFlowLength+1;
    H5File h5file(h5SummaryRoot);
    h5file.Open(false);
    if (opts.outputDebug > 3) 
      saver.WriteResults(h5file);
    H5Arma::WriteMatrix (h5file, h5Summary, wellMatrix);
    CalcRegionEmptyStat(h5file, modelMesh, traceStore, h5SummaryRoot, flows, mask);
    h5file.Close();
  }
}

void DifferentialSeparator::OutputStats(DifSepOpt &opts, Mask &bfMask) {
  int beadCount = 0, emptyCount = 0, ignoreCount = 0, libCount = 0, tfCount = 0, dudCount = 0, pinnedCount = 0, referenceCount = 0, excludedCount = 0, excludeCount = 0;
  for (size_t bIx = 0; bIx < mWells.size(); bIx++) {
    if (bfMask[bIx] & MaskExclude) { excludeCount++; }
    if (bfMask[bIx] & MaskBead) { beadCount++; }
    if (bfMask[bIx] & MaskPinned) { pinnedCount++; }
    if (bfMask[bIx] & MaskEmpty) { emptyCount++; }
    if (bfMask[bIx] & MaskReference) { referenceCount++; }
    if (bfMask[bIx] & MaskIgnore) { ignoreCount++; }
    if (bfMask[bIx] & MaskLib) { libCount++; }
    if (bfMask[bIx] & MaskTF) { tfCount++; }
    if (bfMask[bIx] & MaskDud) { dudCount++; }
    if (bfMask[bIx] & MaskExclude) { excludedCount++; }
  }
  cout << "Exclude:\t" << excludeCount << endl;
  cout << "Empties:\t" << emptyCount << endl;
  cout << "Reference:\t" << referenceCount << endl;
  cout << "Pinned :\t" << pinnedCount << endl;
  cout << "Excluded :\t" << excludedCount << endl;
  cout << "Ignored:\t" << ignoreCount << endl;
  cout << "Beads  :\t" << beadCount << endl;
  cout << "Duds   :\t" << dudCount << endl;
  cout << "Live   :\t" << libCount + tfCount << endl;
  cout << "TFBead :\t" << tfCount << endl;
  cout << "Library:\t" << libCount << endl;
  if (opts.outputDebug > 1) {
    string outMask = opts.outData + ".mask.bin";
    bfMask.WriteRaw (outMask.c_str());
  }
}

void SetBeadfindType(DifSepOpt &opts, std::string &bfFile) {
  bfFile = opts.resultsDir + "/" + opts.bfDat;
  cout << "bfDat " << opts.bfDat << endl;
  cout << "bfType " << opts.bfType << endl;
  if (opts.bfType == "positive") {
    opts.bfMult = -1.0f;
    opts.gainMult = -1.0f;
  }
  else if (opts.bfType == "nobuffer") {
    opts.skipBuffer = true;
  }
  else if (opts.bfType == "naoh") {
    opts.bfMult = 1.0f;
  }
  else {
    ION_ABORT("don't recogize bfType" + ToStr(opts.bfType));
  }
}

void DifferentialSeparator::DoBeadfindFlowAndT0(DifSepOpt &opts, Mask &mask, const std::string &bfFile) {
  ClockTimer mTotalTimer;
  std::vector<float> t02;
  Image img;
  size_t numWells = mask.H() * mask.W();
  mFilteredWells.resize(numWells, 0);

  OpenAndProcessImage(bfFile.c_str(), (char *)opts.resultsDir.c_str(), opts.ignoreChecksumErrors, 
		      false, &mask, false, img, opts.col_pair_pixel_xtalk_correct, 
		      opts.pair_xtalk_fraction, opts.corr_noise_correct, opts.isThumbnail);

  // apply gain correction if requested
  if (opts.doGainCorrect) {

    ImageTransformer::gain_correction = NULL;

    if (opts.useBeadfindGainCorrect) {
      ImageNNAvg imageNN;
      imageNN.SetGainMinMult(opts.gainMult);
      mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: before bf gain.");
      if (opts.isThumbnail) { CalcImageGain(&img, &mask, &mFilteredWells[0], BF_THUMBNAIL_SIZE, BF_THUMBNAIL_SIZE, &imageNN); }
      else { CalcImageGain(&img, &mask, &mFilteredWells[0], mask.H(), mask.W(), &imageNN); }
      mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: after bf gain.");
    }
    else if (opts.useDataCollectGainCorrect) {
      const string gainFile = "Gain.lsr";
      const string destFile = opts.resultsDir + "/" + gainFile; 
      std::cout << "DataCollect gain: reading per pixel gain from " << destFile << std::endl; 
      ImageTransformer::ReadDataCollectGainCorrection(destFile, img.GetRows(), img.GetCols());
    }

    GainCorrectImage(opts.doGainCorrect, img);
  }
  
  // Caclulate t0 with the same file
  mT0.resize(numWells);
  std::fill(mT0.begin(), mT0.end(), 0.0f);
  t02.resize(numWells);
  std::fill(t02.begin(), t02.end(), 0.0f);
  char incorporationFlowBuff[MAX_PATH_LENGTH];
  int mask_bad = MaskIgnore | MaskPinned | MaskExclude;
  snprintf(incorporationFlowBuff, sizeof(incorporationFlowBuff), "%s%.4d.%s", opts.acqPrefix.c_str(), (int)mKeys[0].flows.size()-1, opts.datPostfix.c_str());

  //  CountReference("Starting", mFilteredWells);
  for (size_t i = 0; i < numWells; i++) {
    if ((mask[i] & mask_bad) != 0) {
      mFilteredWells[i] = DifferentialSeparator::PinnedExcluded;
    }
  }
  const RawImage *raw = img.GetImage();
  mWells.resize (mT0.size());
  //  CountReference("Initial From Mask", mFilteredWells);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: before bf t0.");
  std::vector<float> bf_ssq;
  int bf_t0_range_start = 0;
  int bf_t0_range_end = 15;
  if (!opts.skipBuffer) {
    if(opts.gainMult == 1) {
      CalcBfT0(opts, mT0, bf_ssq, img);
    }
    else {
      CalcAcqT0(opts, mT0, bf_ssq, img, true);
      bf_t0_range_start = 0;
      bf_t0_range_end = 30;
    }
  }
  //  CountReference("After BF t0", mFilteredWells);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: after bf t0.");
  std::vector<float> acq_ssq;
  CalcAcqT0(opts, t02,  acq_ssq, incorporationFlowBuff);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: after acq t0.");
  //  CountReference("After Acq t0", mFilteredWells);

  //  CalcBfT0(opts, t02, "beadfind_pre_0001.dat");
  for (size_t i = 0; i < mT0.size(); i++) {
    mWells[i].bfT0 = mT0[i];
    mWells[i].acqT0 = t02[i];
    float ot = mT0[i];
    if (mT0[i] > 0 && t02[i] > 0) {
      mT0[i] = (.3 * mT0[i] + .7 * t02[i]);
    }
    else {
      mT0[i] = max(mT0[i], t02[i]);
    }
    mT0[i] = max(-1.0f,mT0[i] - T0_LEFT_OFFSET);
    if (ot > 0) {
      t02[i] = raw->interpolatedFrames[max((int)(mT0[i]+.5)-1,0)];
    }
  }
  for (size_t i = 0; i < numWells; i++) {
    if ((mask[i] & mask_bad) != 0) {
      mFilteredWells[i] = DifferentialSeparator::PinnedExcluded;
    }
  }
  mRefWells.resize(mFilteredWells.size());
  // This is for t0 regions
  mRegionIncorpReporter.Init(opts.outData, mask.H(), mask.W(),
			     opts.regionXSize, opts.regionYSize,
			     mKeys, mT0);
  mRegionIncorpReporter.SetMinKeyThreshold(1, 0);
  mRegionIncorpReporter.SetMinKeyThreshold(0, 0);
  mRegionIncorpReporter.Finish(); // this is for t0 regions

  //  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: after t0/gain.");
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: before bf metric.");
  BfMetric bf_metric;
  bf_metric.Init(raw->rows, raw->cols, raw->frames);
  bf_metric.SetGainMinMult(opts.gainMult);
  string bf_metric_file = "";
  int bf_frame_window = 1; // opts.gainMult == 1 ? 1 : 0;
  if (!opts.skipBuffer) {
    if (opts.outputDebug > 2) {
      bf_metric_file = opts.outData + "_beadfind.h5";
    }
    if (opts.isThumbnail) {
      bf_metric.CalcBfSdMetricReduce(raw->image, &mask, &mFilteredWells[0],
                                     &t02[0], bf_metric_file, bf_t0_range_start, bf_t0_range_end,
                                     BF_THUMBNAIL_SIZE, BF_THUMBNAIL_SIZE, 
                                     BF_NN_AVG_WINDOW, BF_NN_AVG_WINDOW, bf_frame_window);
    }
    else {
      bf_metric.CalcBfSdMetricReduce(raw->image, &mask, &mFilteredWells[0],
                                     &t02[0], bf_metric_file, bf_t0_range_start, bf_t0_range_end,
                                     mask.H(), mask.W(),
                                     BF_NN_AVG_WINDOW, BF_NN_AVG_WINDOW, bf_frame_window);    
    }
  }
  // unset pinned from beadfind as looks wrong for positive beadfind...
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: after bf metric.");
  //  size_t numWells = mask.H() * mask.W();


  mBfMetric.resize(numWells,0.0f);
  std::fill(mBfMetric.begin(), mBfMetric.end(), 0.0f);
  mBfSdFrame.resize(numWells, 0);
  mBfSSQ.resize(numWells, 0);
  mAcqSSQ.resize(numWells, 0);
  for (size_t i = 0; i < numWells; i++) {
    float value = 0;
    if (!opts.skipBuffer) {
       value = bf_metric.GetBfMetric(i);
       mBfSdFrame[i] = bf_metric.GetSdFrame(i);
    }
    mBfMetric[i] = opts.bfMult * value;
    mAcqSSQ[i] = acq_ssq[i];
  }
  bf_metric.Cleanup();
  img.Close();
}

void SmoothTraces(DifSepOpt &opts, TraceStoreCol &traceStore, Mask &mask, float *traceMad, char *filtWells) {
  ClockTimer timer;
  if (opts.smoothTrace) {
    printf("Smoothing traces.\n");
    timer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before smoothing traces.");
    for (size_t flow_ix = 0; flow_ix < traceStore.GetNumFlows(); flow_ix++) {
      GridMesh<int> pcaMesh;
      pcaMesh.Init (mask.H(), mask.W(), PCA_COMP_GRID_SIZE, PCA_COMP_GRID_SIZE);
      for (size_t binIx = 0; binIx < pcaMesh.GetNumBin(); binIx++) {
        int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
        pcaMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
        traceStore.PcaLossyCompress(rowStart, rowEnd, colStart, colEnd, flow_ix,
                                    traceMad, filtWells,
                                    2, 2, 6);
      }
    }
    timer.PrintMicroSecondsUpdate(stdout, "Total Timer: After smoothing traces.");
  }
}

void DifferentialSeparator::ClusterWells(DifSepOpt &opts, TraceStoreCol &traceStore, Mask &mask, Mask &bfMask, 
                                         float madThreshold, GridMesh<MixModel> &modelMesh) {
  size_t numWells = mask.H() * mask.W();

  GridMesh<MixModel> sdModelMesh;
  std::vector<char> buffCluster(mWells.size(), -1);
  std::vector<char> sdCluster(mWells.size(), -1);
  // If we didn't run a beadfind flow then use the tauB estimate. 
  if (opts.skipBuffer || opts.gainMult == -1) {
    cout << "Using taub estimate as buffering." << endl;
    mBfMetric.resize(numWells);
    //    std::copy(mEmptyMetrics.back().begin(), mEmptyMetrics.back().end(), mBfMetric.begin());
    for (size_t i = 0; i < numWells; i++) {
      mBfMetric[i] = mWells[i].tauB;
      mWells[i].bfMetric = mWells[i].tauB;
    }
  }
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before Regional Buffering Clustering.");
  DoRegionClustering(opts, mask, mBfMetric, madThreshold, mWells, modelMesh);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After Regional Buffering Clustering.");
  mBfConfidence.resize(2); // for debugging diagnostics
  ClusterIndividualWells(opts, bfMask, mask, traceStore, modelMesh, mWells, mBfConfidence[0], buffCluster);

  cout << "Using signal for clustering" << endl;
  for (size_t i = 0; i < numWells; i++) {
    // wells[i].bfMetric = mWells[i].onemerAvg;
    // reference.SetBfMetricVal(i, mWells[i].bfMetric);
    if (opts.skipBuffer) {
      mBfMetric[i] = mWells[i].peakSig;
      mWells[i].bfMetric = mWells[i].peakSig;
    }
    else {
      mBfMetric[i] = mWells[i].sd;
      mWells[i].bfMetric = mWells[i].sd;
    }
  }
  // Cluster the individual wells
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before Regional Signal Clustering.");
  DoRegionClustering(opts, mask, mBfMetric, madThreshold, mWells, sdModelMesh);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After Regional Signal Clustering.");
  ClusterIndividualWells(opts, bfMask, mask, traceStore, sdModelMesh, mWells, mBfConfidence[1], sdCluster);
  //  copy(sdCluster.begin(), sdCluster.end(), buffCluster.begin());
  if (opts.sdAsBf) {
    buffCluster.swap(sdCluster);
  }
  int num_clipped_live_wells = 0, ref_wells = 0, diff=0, rescued_lib = 0, same=0;
  for (size_t bIx = 0; bIx < numWells; bIx++) {
    if (sdCluster[bIx] != buffCluster[bIx] && buffCluster[bIx] >= 0) 
      diff++;
    else if (buffCluster[bIx] >= 0) 
      same++;
    if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned) { // || mFilteredWells[bIx] != GoodWell) {
      continue;
    }
    if (sdCluster[bIx] == 0 || buffCluster[bIx] == 0) {
        bfMask[bIx] = MaskIgnore;
    }
    if (buffCluster[bIx] == 2) {
      mWells[bIx].flag = WellBead;
      bfMask[bIx] = MaskBead;
      if (sdCluster[bIx] == 2 && mWells[bIx].keyIndex == -1 && mWells[bIx].snr > RESCUE_SNR_THRESH && mWells[bIx].peakSig > RESCUE_PEAK_THRESH) {
        rescued_lib++;
        mWells[bIx].keyIndex = 0;
      }
    }
    if (buffCluster[bIx] == 1) {
      mWells[bIx].flag = WellEmpty;
      bfMask[bIx] = MaskEmpty;
    }
    if (buffCluster[bIx] == 1 && sdCluster[bIx] == 1) {
      ref_wells++;
      if (mWells[bIx].keyIndex < 0) {
        bfMask[bIx] = MaskReference | MaskEmpty;
      }
    }
  }
  fprintf(stdout, "Clipped %d live wells with both low signal and low buffering, rescued %d lib %d reference (%d %.2f diff).\n", num_clipped_live_wells, rescued_lib, ref_wells, diff, 1.0*diff/(same+diff));
}

void DifferentialSeparator::SetUp(DifSepOpt &opts, std::string &bfFile, arma::Col<int> &zeroFlows, 
                                  vector<int> &flowsAllZero,   string &h5SummaryRoot) {
 // Create the mKeys if not specified.
  if (mKeys.empty())  { MakeStadardKeys (mKeys); }
  for (size_t kIx = 0; kIx < mKeys.size(); kIx++) { opts.maxKeyFlowLength = max ( (unsigned int) opts.maxKeyFlowLength, mKeys[kIx].usableKeyFlows); }
  // Setup our job queue
  int qSize = (mMask.W() / opts.t0MeshStepX + 1) * (mMask.H() / opts.t0MeshStepY + 1);
  if (opts.nCores <= 0) {  opts.nCores = numCores(); }
  mQueue.Init (opts.nCores, qSize);

  // Figure out which flows can be used for fitting taue
  ZeroFlows(mKeys, opts, zeroFlows, flowsAllZero);
  SetBeadfindType(opts, bfFile);
  LoadInitialMask(opts.mask, opts.maskFile, bfFile, mMask, opts.ignoreChecksumErrors);
  mNumWells = mMask.H() * mMask.W();  
  // Open our h5 file if necessary

  if (opts.outputDebug > 1) {
    h5SummaryRoot = opts.outData + ".h5";
    H5File h5file(h5SummaryRoot);
    h5file.Open(true);
    h5file.Close();
  }

}

void DifferentialSeparator::ClusterToSelectReference(DifSepOpt &opts, const std::string &h5SummaryRoot,
                                                     TraceSaver &saver, TraceStoreCol &traceStore,
                                                     GridMesh<MixModel> &modelMesh,
                                                       GridMesh<struct FitTauEParams> &emptyEstimates) {
  for (size_t i = 0; i < mNumWells; i++) {
    mWells[i].isRef = traceStore.IsReference(i);
    mWells[i].bufferMetric = mWells[i].bfMetric = mBfMetric[i];
  }

  WellSetKeyStats libStats("Library", REF_SAMPLE_SIZE), refStats("SepRef Wells", REF_SAMPLE_SIZE), 
    tfStats("TF", REF_SAMPLE_SIZE), filtStats("Soft Filters",REF_SAMPLE_SIZE), 
    allStats("All Stats", REF_SAMPLE_SIZE), maskRefStats("MaskRef", REF_SAMPLE_SIZE), 
    maskEmptyStats("MaskEmpty", REF_SAMPLE_SIZE), softFiltStats("SoftFilt", REF_SAMPLE_SIZE);

  for (size_t i = 0; i < mNumWells; i++) {
    if (mWells[i].keyIndex == 0) { libStats.AddWell(mWells[i]); }
    if (mWells[i].keyIndex == 1) { tfStats.AddWell(mWells[i]); }
    if (mWells[i].isRef == 1) { refStats.AddWell(mWells[i]); }
    if (mFilteredWells[i] != 0) {filtStats.AddWell(mWells[i]);}
    if (isfinite(mWells[i].mad) && mWells[i].mad >= 0) { allStats.AddWell(mWells[i]); }
    if (isfinite(mWells[i].mad) && mWells[i].mad >= 0 && mFilteredWells[i] == 0) { softFiltStats.AddWell(mWells[i]); }
  }
  // --- Check to make sure we got some live wells, do we really need this anymore?
  int gotKeyCount = 0;
  OutputFitSummary(opts, mWells, emptyEstimates, gotKeyCount);
  int notExcludePinnedWells = 0;
  for (size_t i = 0; i < mNumWells; i++) {
    if (! (mMask[i] & MaskExclude || mMask[i] & MaskPinned)) { notExcludePinnedWells++; }
  }

  int minOkCount = max (10.0, opts.minRatioLiveWell * notExcludePinnedWells);
  if (gotKeyCount <= minOkCount) { 
    ION_ABORT_CODE ("Only got: " + ToStr (gotKeyCount) + " key passing wells. Couldn't find enough (" + 
                    ToStr (minOkCount) + ") wells with key signal.", DIFFSEP_ERROR); }
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Fitting Filters.");
  double minTfPeak = opts.minTfPeakMax;
  if (tfStats.NumSeen() > 5 * MIN_SAMPLE_TAUE_STATS) {
    minTfPeak = max(minTfPeak, 
                    tfStats.m_quantiles[KEY_PEAK_STAT].GetQuantile(.25) - (3 * IQR (tfStats.m_quantiles[KEY_PEAK_STAT])));
  }
  double minLibPeak = opts.minLibPeakMax;
  if (libStats.NumSeen() > 5 * MIN_SAMPLE_TAUE_STATS) {
    minLibPeak = max(minLibPeak, 
                     libStats.m_quantiles[KEY_PEAK_STAT].GetQuantile(.25) - (3 * IQR (libStats.m_quantiles[KEY_PEAK_STAT])));
  }
  cout << "Min Tf peak is: " << minTfPeak << " lib peak is: " << minLibPeak << endl;
  double madThreshold = softFiltStats.m_quantiles[MAD_STAT].GetQuantile(.75) + (3 * IQR (softFiltStats.m_quantiles[MAD_STAT]));
  //  madThreshold = max(madThreshold, 8.0);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After fitting filters.");

  ClusterWells(opts, traceStore, mMask, mBfMask, madThreshold, modelMesh);
  double sepRefSdThresh = refStats.m_quantiles[KEY_SD_STAT].GetQuantile(.75) + (3.0* IQR (refStats.m_quantiles[KEY_SD_STAT]));
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before Post Filtering.");
  AssignAndCountWells(opts, mWells, mBfMask, mFilteredWells, minLibPeak, minTfPeak, sepRefSdThresh, madThreshold);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After Post Filtering.");

  for (size_t i = 0; i < mNumWells; i++) {
    if (mBfMask[i] & MaskReference) {maskRefStats.AddWell(mWells[i]);}
    if (mBfMask[i] & MaskEmpty) {maskEmptyStats.AddWell(mWells[i]);}
  }
  //  CountReference("Filtered wells", mFilteredWells);
  if (libStats.NumSeen() > 10) { libStats.ReportStats(stdout);}
  if (tfStats.NumSeen() > 50) { tfStats.ReportStats(stdout);}
  maskRefStats.ReportStats(stdout);
  maskEmptyStats.ReportStats(stdout);
  refStats.ReportStats(stdout);
  filtStats.ReportStats(stdout);
  OutputStats(opts, mBfMask);
  HandleDebug(mWells, opts, h5SummaryRoot, saver, mMask, traceStore, modelMesh,
              maskRefStats.m_quantiles[KEY_SD_STAT].GetQuantile(.9),
              maskRefStats.m_quantiles[KEY_SD_STAT].GetQuantile(.9),
              allStats.m_quantiles[MAD_STAT].GetQuantile(.9),
              maskRefStats.m_quantiles[BUFFER_STAT].GetQuantile(.9),
              maskRefStats.m_quantiles[BUFFER_STAT].GetQuantile(.9),
              libStats.m_quantiles[KEY_PEAK_STAT].GetQuantile(.1)
              );
}

void DifferentialSeparator::SetupTraceStore(DifSepOpt &opts, TraceStoreCol &traceStore, std::vector<float> &ftime) {
  // --- Load up the key flows into our trace store
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before Loading Dats.");
  traceStore.Init (mMask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
                            opts.maxKeyFlowLength+2, opts.maxKeyFlowLength+2,
                            opts.referenceStep, opts.referenceStep);
  traceStore.SetMinRefProbes (opts.percentReference * opts.referenceStep * opts.referenceStep);
  vector<float> traceSdMin(mNumWells);
  LoadKeyDats (mQueue, traceStore, mBfMetric, opts, traceSdMin);
  // Currently time is just linear
  mTime.set_size (traceStore.GetNumFrames());
  for (size_t i = 0; i < mTime.n_rows; i++)  { mTime[i] = i; }
  ftime.resize(traceStore.GetNumFrames());
  std::copy(mTime.begin(), mTime.end(), ftime.begin());
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After Loading Dats.");
}

int DifferentialSeparator::Run(DifSepOpt opts) {
  Col<int> zeroFlows;
  vector<int> flowsAllZero;
  string bfFile;
  TraceSaver saver;
  std::vector<float> ftime;
  GridMesh<struct FitTauEParams> emptyEstimates;
  GridMesh<MixModel> modelMesh;
  TraceStoreCol traceStore;
  string h5SummaryRoot;

  // Some intialization based on keys and flow order
  SetUp(opts, bfFile, zeroFlows, flowsAllZero, h5SummaryRoot);

  // Calculate initial t0 and beadfind metric
  DoBeadfindFlowAndT0(opts, mMask, bfFile);

  // Load up traces into our store
  SetupTraceStore(opts, traceStore, ftime);
  int row_step = mMask.H(), col_step = mMask.W();

  if (opts.isThumbnail) { row_step = col_step = BF_THUMBNAIL_SIZE; }

  if (opts.filterNoisyCols != "none") {
    FilterNoisyColumns(row_step, col_step, mMask, opts, mFilteredWells);
  }

  // Pick which wells to use for initial reference for backgound
  int minWells = ceil(opts.referenceStep*opts.referenceStep * opts.percentReference);
  RankReference(traceStore, mBfMetric, opts.referenceStep, opts.referenceStep, opts.useSignalReference,
                opts.iqrMult, 7, opts.percentReference, mMask, minWells,
                mFilteredWells, mRefWells);
  if (!opts.skipBuffer) {
    mEmptyMetrics.push_back(mBfMetric);
  }
  PickCombinedRank(mEmptyMetrics, opts.referenceStep, opts.referenceStep, opts.percentReference, 
                   minWells, mFilteredWells, mRefWells);
  
  // Set the reference well sin the trace store and 
  for (size_t i = 0; i < mRefWells.size(); i++) { traceStore.SetReference(i, mRefWells[i] == 1); }
  size_t loadMinFlows = max (9, opts.maxKeyFlowLength+2);
  traceStore.mRefReduction.resize(loadMinFlows);
  for (size_t i = 0; i < loadMinFlows; i++) { traceStore.PrepareReference (i, mFilteredWells); }

  if (opts.skipBuffer) {
    cout << "Fitting taub for use as buffering estimate." << endl;
    // Do inital fit of tauE
    int metric_size = mEmptyMetrics.size();
    mEmptyMetrics.resize(metric_size + 1);
    mEmptyMetrics[metric_size].resize(mFilteredWells.size());
    float *taub_est = &mEmptyMetrics[metric_size][0];
    FitTauE(opts,traceStore, emptyEstimates, mFilteredWells, 
            ftime, flowsAllZero, taub_est);
    
    PickCombinedRank(mEmptyMetrics, opts.referenceStep, opts.referenceStep, 
                     opts.percentReference, minWells, mFilteredWells, mRefWells);
  }

  // Smooth or not according to the options
  SmoothTraces(opts, traceStore, mMask, &mTraceMad[0], &mFilteredWells.at(0));

  // Set the reference well sin the trace store and 
  for (size_t i = 0; i < mRefWells.size(); i++) { traceStore.SetReference(i, mRefWells[i] == 1); }
  traceStore.mRefReduction.resize(loadMinFlows);
  for (size_t i = 0; i < loadMinFlows; i++) { traceStore.PrepareReference (i, mFilteredWells); }

  // Fit a global tauE for each region
  FitTauE(opts,traceStore, emptyEstimates, mFilteredWells, 
          ftime, flowsAllZero, NULL);

  // See which keys are the best match for wells.
  FitKeys(mQueue, opts, emptyEstimates, traceStore, mKeys, ftime, saver, mMask, mWells);

  // Use our dual gaussian mixture clustering to split wells into empty/live
  ClusterToSelectReference(opts, h5SummaryRoot, saver, traceStore, modelMesh, emptyEstimates);

  // Reporting for the log.  
  if (opts.outputDebug > 0) {
    mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Before spatial summary.");
    SpatialSummary(opts.analysisDir + "/separator.spatial.h5", "/spatial", opts, mMask, 16, 16);
    mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: After spatial summary.");
  }

  //  CountReference("At end", mFilteredWells);
  mTotalTimer.PrintMicroSecondsUpdate(stdout, "Total Timer: Total Time.");
  return 0;
}


