/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "DifferentialSeparator.h"
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <map>
#include "ZeromerDiff.h"
#include "SampleKeyReporter.h"
#include "IncorpReporter.h"
#include "KeySummaryReporter.h"
#include "AvgKeyReporter.h"
#include "RegionAvgKeyReporter.h"
#include "DualGaussMixModel.h"
#include "KeyClassifyJob.h"
#include "KeyClassifyTauEJob.h"
#include "LoadTracesJob.h"
#include "FillCriticalFramesJob.h"
#include "AvgKeyIncorporation.h"
#include "Traces.h"
#include "Image.h"
#include "OptArgs.h"
#include "KClass.h"
#include "PJobQueue.h"
#include "Utils.h"
#include "HandleExpLog.h"
#include "IonErr.h"
#include "TraceStoreMatrix.h"
#include "ReservoirSample.h"
#include "H5File.h"
#include "T0Calc.h"
#include "SynchDatSerialize.h"
#include "Stats.h"
#include "ComparatorNoiseCorrector.h"
#include "BkgTrace.h"
#include "RawWells.h"
#define DIFFSEP_ERROR 3
// Amount in frames back from estimated t0 to use for dc offset estimation 
// @hack - Note this must be kept in synch with VFC_T0_OFFSET in BkgMagicDefines.h
#define T0_LEFT_OFFSET 4 
#define T0_RIGHT_OFFSET 25
#define MIN_SD 10 // represents a nuc step with less than 35 counts
#define T0_OFFSET_LEFT_TIME 0.0f
#define T0_OFFSET_RIGHT_TIME 1.3f // seconds

using namespace std;

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
    if (counts[i] > 0) {
      cout << DifferentialSeparator::NameForFilter((enum DifferentialSeparator::FilterType)i) << ":\t" << counts[i] << endl;
    }
  }
}

class LoadSDatJob : public PJob {
public:
  LoadSDatJob() {
    mFlowIx = 0;
    mMask = NULL;
    mFilteredWells = NULL;
    mT0 = NULL;
    mOpts = NULL;
  }

  void Init(const std::string &_fileName, int _flowIx, TraceStore<double> *_traceStore, 
	    Mask *_mask, std::vector<char> *_filteredWells, std::vector<float> *_t0, DifSepOpt *_opts) {
    mFileName = _fileName;
    mFlowIx = _flowIx;
    mTraceStore = _traceStore;
    mMask = _mask;
    mFilteredWells = _filteredWells;
    mT0 = _t0;
    mOpts = _opts;
  }

  virtual void Run() {
    SynchDat sdat;
    mReadSerializer.Read(mFileName.c_str(), sdat);
    ComparatorNoiseCorrector cnc;
    Col<double> traceBuffer;
    size_t numWells = mMask->H() * mMask->W();
    if (mOpts->doComparatorCorrect) {
      for (size_t rIx = 0; rIx < sdat.GetNumBin(); rIx++) {
	TraceChunk &chunk = sdat.GetChunk(rIx);
	// Copy over temp mask for normalization
	Mask m(chunk.mWidth, chunk.mHeight);
	for (size_t r = 0; r < chunk.mHeight; r++) {
	  for (size_t c = 0; c < chunk.mWidth; c++) {
	    m[r*chunk.mWidth+c] = (*mMask)[(r+chunk.mRowStart) * mMask->W() + (c+chunk.mColStart)];
	  }
	}
	cnc.CorrectComparatorNoise(&chunk.mData[0], chunk.mHeight, chunk.mWidth, chunk.mDepth, &m, false, mOpts->aggressive_cnc);
      } 
    }

    const short pinHigh = GetPinHigh();
    const short pinLow = GetPinLow();
    size_t row, col;
    int numFrames = 0;
    if (0 == mFlowIx) {
      for (float f = T0_OFFSET_LEFT_TIME; f < T0_OFFSET_RIGHT_TIME; f+= sdat.FrameRate() / 1000.0f) {
	numFrames++;
      }
      mTraceStore->SetSize(numFrames);
      mT0->resize(numWells);
      for (size_t wIx = 0; wIx < numWells; wIx++) {
	mTraceStore->WellRowCol(wIx, row, col);
	(*mT0)[wIx] = sdat.GetT0(row, col);
      }
      mTraceStore->SetT0(*mT0);
    }
    
    for (size_t rIx = 0; rIx < sdat.GetNumBin(); rIx++) {
      TraceChunk &chunk = sdat.GetChunk(rIx);
      // Copy over temp mask for normalization
      for (size_t r = 0; r < chunk.mHeight; r++) {
	for (size_t c = 0; c < chunk.mWidth; c++) {
	  size_t chipIdx = (r+chunk.mRowStart) * mMask->W() + (chunk.mColStart + c);
	  int16_t *start = &chunk.mData[0] + r * chunk.mWidth + c;
	  for (size_t f = 0; f < chunk.mDepth; f++) {
	    if (*start <= pinLow || *start >= pinHigh) {
	      (*mMask)[chipIdx] = MaskPinned;
	      break;
	    }
	    start += chunk.mFrameStep;
	  }
	}
      }
    }

    sdat.AdjustForDrift();
    sdat.SubDcOffset();
    SampleStats<float> sd;
    traceBuffer.set_size( mTraceStore->GetNumFrames());
    for (size_t rIx = 0; rIx < sdat.GetNumBin(); rIx++) {
      TraceChunk &chunk = sdat.GetChunk(rIx);
      // Copy over temp mask for normalization
      for (size_t r = 0; r < chunk.mHeight; r++) {
	for (size_t c = 0; c < chunk.mWidth; c++) {
	  size_t chipIdx = (r+chunk.mRowStart) * mMask->W() + (chunk.mColStart + c);

	  if (mTraceStore->HaveWell(chipIdx)) {
	    int index = 0;
	    sd.Clear();
	    for (float f = T0_OFFSET_LEFT_TIME; f < T0_OFFSET_RIGHT_TIME; f+= (sdat.FrameRate() / 1000.0f)) {
	      assert(index < (int)traceBuffer.n_rows);
	      //	      double val = sdat.GetValue(f, row, col);
	      traceBuffer[index] = chunk.GetT0TimeVal(r, c, f);
	      sd.AddValue(traceBuffer[index]);
	      index++;
	    }
	    mTraceStore->SetTrace(chipIdx, mFlowIx, traceBuffer.begin(), traceBuffer.begin() + mTraceStore->GetNumFrames());

	    if (sd.GetSD() < MIN_SD) {
	      (*mFilteredWells)[chipIdx] = DifferentialSeparator::LowTraceSd;
	    }
	  }
	}
      }
    }
  }

  std::string mFileName;
  int mFlowIx;
  TraceStore<double> *mTraceStore;
  Mask *mMask;
  std::vector<char> *mFilteredWells;
  std::vector<float> *mT0;
  DifSepOpt *mOpts;
  TraceChunkSerializer mReadSerializer;
};

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

  void Init(const std::string &_fileName, int _flowIx, TraceStore<double> *_traceStore, float *_traceSdMin,
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

  static void LoadDat(const std::string &fileName, DifSepOpt *opts, TraceStore<double> *traceStore, 
                      std::vector<float> *t0, Mask *mask, Mask *cncMask, int flowIx, float *traceSdMin) {
    Image img;
    img.SetImgLoadImmediate (false);
    img.SetIgnoreChecksumErrors (opts->ignoreChecksumErrors);
    bool loaded = img.LoadRaw (fileName.c_str());
    if (!loaded) { ION_ABORT ("Couldn't load file: " + fileName); }
    RawImage *raw = img.raw;
    img.FilterForPinned(mask, MaskEmpty, false);
    ImageTransformer::XTChannelCorrect (raw, (char *) opts->resultsDir.c_str());    // @todo gain correction
    if (ImageTransformer::gain_correction != NULL) {
      ImageTransformer::GainCorrectImage(raw);
    }
    // 2. Comparator correction, cross talk correction, pinning and other image normalization
    if (opts->doComparatorCorrect) {
      ComparatorNoiseCorrector cnc;
      if (opts->isThumbnail) {
	cnc.CorrectComparatorNoiseThumbnail(raw, cncMask, opts->clusterMeshStep, opts->clusterMeshStep, false);
      }
      else {
	cnc.CorrectComparatorNoise(raw, cncMask, false, opts->aggressive_cnc);
      }
    }

    Col<double> traceBuffer;
    // 3. Get an average t0 for wells that we couldn't get a good local number
    SampleStats<float> meanT0Sample;
    for (size_t i = 0; i < t0->size(); i++) {
      if (t0->at(i) > 0) {
	meanT0Sample.AddValue(t0->at(i));
      }
    }
    int meanT0 = (int)(meanT0Sample.GetMean() + .5);

    /*
      4. Interpolate the points we want, times desired are in seconds and timestamps
      from image are in millisecond integers.
      example
      time desired = 2
      frames = 1000,2000,3000
      fAbove is 3
      fBelow is 2
      y = mx + b;
    */
    traceBuffer.set_size( traceStore->GetNumFrames());
    float rawFrames[traceBuffer.n_rows];
    float t0shifted[traceBuffer.n_rows];
    SampleStats<float> sd;    
    for (int row = 0; row < raw->rows; row++) {
      for (int col = 0; col < raw->cols; col++) {
	size_t idx = row * raw->cols + col;
	if (traceStore->HaveWell(idx)) {
	  sd.Clear();
          // Round to nearest frame.
	  int wT0 = (int)((*t0)[idx] + .5);
          double t0Shift = (*t0)[idx] - wT0;
	  if (wT0 <= 0) {
	    wT0 = meanT0;
            t0Shift = 0;
	  }
	  double dc = 0.0;
	  int dcCount = 0;
          // float version1[raw->uncompFrames];
          // float version2[raw->uncompFrames];
          // img.GetUncompressedTrace(version1, raw->uncompFrames, col, row);
	  // for (int frame = 0; frame < raw->uncompFrames; frame++) {
          //   version2[frame] = img.GetInterpolatedValue(frame, col, row);
          // }
          
	  for (int frame = 0; frame < T0_RIGHT_OFFSET; frame++) {
            //traceBuffer[frame] = img.GetInterpolatedValue(frame+wT0, col, row);
            rawFrames[frame] = img.GetInterpolatedValue(frame+wT0, col, row);
	    if (frame < T0_LEFT_OFFSET) {
              //	      dc += traceBuffer[frame];
	      dc += rawFrames[frame];
	      dcCount++;
	    }
	    sd.AddValue(rawFrames[frame]);
	  }
          TraceHelper::ShiftTrace(rawFrames, t0shifted, traceBuffer.n_rows, t0Shift);
	  if (dcCount > 0) {
	    dc = dc / dcCount;
	  }
	  else {
	    dc = t0shifted[0];
	  }
	  for (int frame = 0; frame < T0_RIGHT_OFFSET; frame++) {
	    traceBuffer[frame] = t0shifted[frame] -  dc;
	  }
	  traceStore->SetTrace(idx, flowIx, traceBuffer.begin(), traceBuffer.begin() + T0_RIGHT_OFFSET);
          traceSdMin[idx] = sd.GetSD();
        }
      }
    }
    // 5. Cleanup
    img.Close();
  }
                      

  virtual void Run() {
    LoadDat(mFileName, mOpts, mTraceStore, mT0, mMask, mCNCMask, mFlowIx, mTraceSdMin);
    size_t num_wells = mMask->H() * mMask->W();
    for (size_t i = 0; i < num_wells; i++) {
      if (mTraceSdMin[i] < MIN_SD) {
        (*mFilteredWells)[i] = DifferentialSeparator::LowTraceSd;
      }
    }
  }
    
  std::string mFileName;
  int mFlowIx;
  TraceStore<double> *mTraceStore;
  Mask *mMask;
  Mask *mCNCMask;
  std::vector<char> *mFilteredWells;
  float *mTraceSdMin;
  std::vector<float> *mT0;
  DifSepOpt *mOpts;
};

float DifferentialSeparator::LowerQuantile (SampleQuantiles<float> &s)
{
  if (s.GetNumSeen() == 0)
    {
      return 0;
    }
  return (s.GetQuantile (.5) - s.GetQuantile (.25));
}

float DifferentialSeparator::IQR (SampleQuantiles<float> &s)
{
  if (s.GetNumSeen() == 0)
    {
      return 0;
    }
  return (s.GetQuantile (.75) - s.GetQuantile (.25));
}

void DifferentialSeparator::ClusterRegion (int rowStart, int rowEnd,
					   int colStart, int colEnd,
					   float maxMad,
					   float minBeadSnr,
					   size_t minGoodWells,
					   BFReference &reference,
					   vector<KeyFit> &wells,
					   double trim,
					   MixModel &model)
{
  vector<float> metric;
  vector<int8_t> cluster;
  int numWells = (rowEnd - rowStart) * (colEnd - colStart);
  metric.reserve (numWells);
  cluster.reserve (numWells);
  double regionRefMean = 0;
  if (!wells.empty()) {
    SampleQuantiles<double> refMean(1000);
    for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
      for (int colIx = colStart; colIx < colEnd; colIx++) {
        size_t idx = reference.RowColToIndex (rowIx, colIx);
        if(wells[idx].isRef) {
          float val = wells[idx].bfMetric;
          if (isfinite(val)) {
            refMean.AddValue(val);
          }
        }
      }
    }
    regionRefMean = refMean.GetMedian();
  }
  int madFilt = 0, bfMetricFilt = 0, filteredWells = 0, pinnedWells = 0, excludedWells = 0, notOk = 0;
  for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
    for (int colIx = colStart; colIx < colEnd; colIx++) {
      size_t idx = reference.RowColToIndex (rowIx, colIx);
      if (mask[idx] & MaskPinned || mask[idx] & MaskExclude || !isfinite (reference.GetBfMetricVal (idx)) || mFilteredWells[idx] != GoodWell || wells[idx].mad > maxMad || wells[idx].ok != 1) {
        if (wells[idx].mad > maxMad) {
          madFilt++;
        }
        if (mFilteredWells[idx] != GoodWell) {
          filteredWells++;
        }
        if (!isfinite (reference.GetBfMetricVal (idx))) {
          bfMetricFilt++;
        }
        if (mask[idx] & MaskPinned) {
          pinnedWells++;
        }
        if (mask[idx] & MaskExclude) {
          excludedWells++;
        }
        if (!wells[idx].ok) {
          notOk++;
        }
        continue;
      }
      if (wells.empty() || (wells[idx].ok == 1 && wells[idx].mad <= maxMad)) {
        if (wells.empty()) {
          reference.SetBfMetricVal(idx, reference.GetBfMetricVal(idx) - regionRefMean);
          metric.push_back (reference.GetBfMetricVal (idx));
        }
        else {
          wells[idx].bfMetric -= regionRefMean;
          metric.push_back (wells[idx].bfMetric);
        }
        if (wells.empty()) {
          cluster.push_back (-1);
        }
        else if (wells[idx].goodLive) {
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
    //    cout << "Bad cluster region: " << rowStart << "\t" << colStart << "\trequired: " << minGoodWells << "\t" << "\tgood: " << metric.size() << "\tnotOk: " << notOk << "\tmad: " << madFilt << "\tfilt: " << filteredWells << "\tbfmet: " << bfMetricFilt << "\texclude: " << excludedWells << "\tpinned: " << pinnedWells << endl;
    return;
  }
  DualGaussMixModel dgm (metric.size());
  dgm.SetTrim (trim);
  model = dgm.FitDualGaussMixModel (&metric[0], &cluster[0], metric.size());
  model.refMean = regionRefMean;
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
  lKey.zeroFlows.set_size (4);
  lKey.zeroFlows << 1 << 3  << 4 << 6;
  lKey.onemerFlows.set_size(3);
  lKey.onemerFlows << 0 << 2 << 5;
  lKey.minSnr = 5.5;
  lKey.usableKeyFlows = 7;
  // lKey.zeroFlows.set_size(1);
  // lKey.zeroFlows << 3;
  keys.push_back (lKey);
  tKey.name = "tf";
  tKey.flows = tfKey;
  tKey.minSnr = 7;
  tKey.zeroFlows.set_size (4);
  tKey.zeroFlows << 0 << 2 << 3 << 5;
  tKey.onemerFlows.set_size(3);
  tKey.onemerFlows << 1 << 4 << 6;
  tKey.usableKeyFlows = 7;
  // tKey.zeroFlows.set_size(1);
  // tKey.zeroFlows << 3;
  keys.push_back (tKey);
}

void DifferentialSeparator::LoadInitialMask (Mask *preMask, const std::string &maskFile, const std::string &imgFile, Mask &mask, int ignoreChecksumErrors)
{
  if (preMask != NULL)
    {
      mask.Init (preMask);
    }
  else if (!maskFile.empty())
    {
      mask.SetMask (maskFile.c_str());
    }
  else
    {
      Image bfImg;
      bfImg.SetImgLoadImmediate (false);
      bfImg.SetIgnoreChecksumErrors (ignoreChecksumErrors);
      bool loaded = bfImg.LoadRaw (imgFile.c_str());
      if (!loaded)
	{
	  ION_ABORT ("Couldn't load file: " + imgFile);
	}
      const RawImage *raw = bfImg.GetImage();
      int cols = raw->cols;
      int rows = raw->rows;
      mask.Init (cols,rows,MaskEmpty);
    }
}

void DifferentialSeparator::PrintKey (const KeySeq &k, int kIx)
{
  cout << "Key: " << kIx << "\t" << k.name << "\t" << k.usableKeyFlows << "\t" << k.minSnr << endl;
  for (size_t i = 0; i < k.flows.size(); i++)
    {
      cout << k.flows[i] << ' ';
    }
  cout << endl;
  for (size_t i = 0; i < k.zeroFlows.n_rows; i++)
    {
      cout << k.zeroFlows.at (i) << ' ';
    }
  cout << endl;
}

void DifferentialSeparator::SetKeys (SequenceItem *seqList, int numSeqListItems, float minLibSnr, float minTfSnr)
{
  keys.clear();
  for (int i = numSeqListItems - 1; i >= 0; i--)
    {
      KeySeq k;
      k.name = seqList[i].seq;
      k.flows.resize (seqList[i].numKeyFlows);
      k.usableKeyFlows = seqList[i].usableKeyFlows;
      int zero_count = 0;
      int onemer_count = 0;
      for (int flowIx = 0; flowIx < seqList[i].usableKeyFlows; flowIx++)
	{
	  if (seqList[i].Ionogram[flowIx] == 0)
	    {
	      zero_count++;
	    }
          if (seqList[i].Ionogram[flowIx] == 1)
            {
              onemer_count++;
            }
	}
      k.zeroFlows.set_size (zero_count);
      k.onemerFlows.set_size (onemer_count);
      zero_count = 0;
      onemer_count = 0;
      for (int flowIx = 0; flowIx < seqList[i].numKeyFlows; flowIx++)
	{
	  k.flows[flowIx] = seqList[i].Ionogram[flowIx];
	  if (seqList[i].Ionogram[flowIx] == 0 && i < seqList[i].usableKeyFlows)	    {
	      k.zeroFlows.at (zero_count++) = flowIx;
	    }
	  if (seqList[i].Ionogram[flowIx] == 1 && i < seqList[i].usableKeyFlows)
	    {
	      k.onemerFlows.at (onemer_count++) = flowIx;
	    }
	}
      if (i == 1) // @todo - this is hacky to assume the
	k.minSnr = minLibSnr;
      else
	k.minSnr = minTfSnr;
      keys.push_back (k);
    }
  for (size_t i = 0; i < keys.size(); i++)
    {
      PrintKey (keys[i], i);
    }
}

void DifferentialSeparator::DoJustBeadfind (DifSepOpt &opts, BFReference &reference)
{
  struct timeval st;
  GridMesh<MixModel> modelMesh;
  opts.bfMeshStep = min (min (mask.H(), mask.W()), opts.bfMeshStep);
  cout << "bfMeshStep is: " << opts.bfMeshStep << endl;
  modelMesh.Init (mask.H(), mask.W(), opts.clusterMeshStep, opts.clusterMeshStep);
  gettimeofday (&st, NULL);
  size_t numWells = mask.H() * mask.W();
  SampleStats<double> bfSnr;
  // For each region do seeded/masked clustering and get mean and s
  // @todo - parallelize
  vector<KeyFit> wells;
  string modelFile = opts.outData + ".mix-model.txt";
  string bfStats = opts.outData + ".bf-stats.txt";
  ofstream modelOut (modelFile.c_str());
  ofstream bfStatsOut (bfStats.c_str());
  opts.minBfGoodWells = min (300, (int) (opts.bfMeshStep * opts.bfMeshStep * .4));
  modelOut << "bin\tbinRow\tbinCol\tcount\tmix\tmu1\tvar1\tmu2\tvar2\tthreshold\trefMean" << endl;
  for (size_t binIx = 0; binIx < modelMesh.GetNumBin(); binIx++)
    {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      modelMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      MixModel &model = modelMesh.GetItem (binIx);
      ClusterRegion (rowStart, rowEnd, colStart, colEnd, 0, opts.minTauESnr, opts.minBfGoodWells, reference, wells, opts.clusterTrim, model);
      if ( (size_t) model.count > opts.minBfGoodWells)
	{
	  double bf = ( (model.mu2 - model.mu1) / ( (sqrt (model.var2) + sqrt (model.var1)) /2));
	  if (isfinite (bf) && bf > 0)
	    {
	      bfSnr.AddValue (bf);
	    }
	  else
	    {
	      cout << "Region: " << binIx << " has snr of: " << bf << " " << model.mu1 << "  " << model.var1 << " " << model.mu2 << " " << model. var2 << endl;
	    }
	}

      int binRow, binCol;
      modelMesh.IndexToXY (binIx, binRow, binCol);
      modelOut << binIx << "\t" << binRow << "\t" << binCol << "\t"
	       << model.count << "\t" << model.mix << "\t"
	       << model.mu1 << "\t" << model.var1 << "\t"
	       << model.mu2 << "\t" << model.var2 << "\t" 
	       << model.threshold << "\t" << model.refMean << endl;
    }

  cout << "BF SNR: " << bfSnr.GetMean() << " +/- " << (bfSnr.GetSD()) << endl;
  modelOut.close();
  std::vector<double> dist;
  std::vector<std::vector<float> *> values;
  vector<MixModel *> bfModels;
  int notGood = 0;
  bfMask.Init (&mask);
  bfStatsOut << "row\tcol\twell\ttype\tbfstatistic\townership\tmu1\tvar1\tmu2\tvar2" << endl;
  for (size_t wIx = 0; wIx < numWells; wIx++)
    {
      bfMask[wIx] = mask[wIx];
      if (bfMask[wIx] & MaskExclude || bfMask[wIx] & MaskPinned)
	{
	  continue;
	}
      size_t row, col;
      double weight = 0;
      MixModel m;
      int good = 0;
      row = wIx / mask.W();
      col = wIx % mask.W();
      modelMesh.GetClosestNeighbors (row, col, opts.bfNeighbors, dist, bfModels);
      for (size_t i = 0; i < bfModels.size(); i++)
	{
	  if ( (size_t) bfModels[i]->count > opts.minBfGoodWells)
	    {
	      good++;
	      float w = 1.0/ (log (dist[i] + 2.00));
	      weight += w;
	      m.mu1 += w * bfModels[i]->mu1;
	      m.mu2 += w * bfModels[i]->mu2;
	      m.var1 += w * bfModels[i]->var1;
	      m.var2 += w * bfModels[i]->var2;
	      m.mix += w * bfModels[i]->mix;
	      m.count += w * bfModels[i]->count;
	    }
	}
      if (good == 0)
	{
	  notGood++;
	  bfMask[wIx] = MaskIgnore;
	}
      else
	{
	  m.mu1 = m.mu1 / weight;
	  m.mu2 = m.mu2 / weight;
	  m.var1 = m.var1 / weight;
	  m.var2 = m.var2 / weight;
	  m.mix = m.mix / weight;
	  m.count = m.count / weight;
	  m.var1sq2p = 1 / sqrt (2 * DualGaussMixModel::GPI * m.var1);
	  m.var2sq2p = 1 / sqrt (2 * DualGaussMixModel::GPI * m.var2);
	  double p2Ownership = 0;
	  int bCluster = DualGaussMixModel::PredictCluster (m, reference.GetBfMetricVal (wIx), opts.bfThreshold, p2Ownership);
	  if (bCluster == 2)
	    {
	      bfMask[wIx] = MaskBead;
	    }
	  else if (bCluster == 1)
	    {
	      bfMask[wIx] = (MaskEmpty | MaskReference);
	    }
	  bfStatsOut << row << "\t" << col << "\t" << wIx << "\t" << bCluster << "\t" << reference.GetBfMetricVal (wIx) << "\t"
		     << p2Ownership << "\t" << m.mu1 << "\t" << m.var1 << "\t" << m.mu2 << "\t" << m.var2 << endl;
	}
    }
  bfStatsOut.close();
  string outMask = opts.outData + ".mask.bin";
  bfMask.WriteRaw (outMask.c_str());
  int beadCount = 0, emptyCount = 0, pinnedCount = 0;
  for (size_t bIx = 0; bIx < numWells; bIx++)
    {
      if (bfMask[bIx] & MaskBead)
	{
	  beadCount++;
	}
      if (bfMask[bIx] & MaskPinned)
	{
	  pinnedCount++;
	}
      if (bfMask[bIx] & MaskEmpty)
	{
	  emptyCount++;
	}
    }
  cout << "Empties:\t" << emptyCount << endl;
  cout << "Pinned :\t" << pinnedCount << endl;
  cout << "Beads  :\t" << beadCount << endl;
}


void DifferentialSeparator::DetermineBfFile (const std::string &resultsDir, bool &signalBased,         const std::string &bfType, const string &bfDat,
					     const std::string &bfBgDat,
					     std::string &bfFile, std::string &bfFile2, std::string &bfBkgFile)
{
  string expLog = resultsDir + "/explog.txt";
  //  string possibleBeadfind = resultsDir + "/acq_0007.dat";
  string possibleBeadfind = resultsDir + "/beadfind_pre_0004.dat";
  string preBeadFind = resultsDir + "/beadfind_pre_0003.dat";
  string preBeadFind2 = resultsDir + "/beadfind_pre_0001.dat";
  string postBeadFind = resultsDir + "/beadfind_post_0003.dat";
  vector<string> values;
  GetExpLogParameters (expLog.c_str(), "AdvScriptFeaturesName", values);
  bfFile2 = "";
  string advScript;
  // Believe user if specified.
  if (bfType != "")
    {
      if (bfType == "signal")
	{
	  signalBased = true;
	}
      else if (bfType == "buffer")
	{
	  signalBased = false;
	}
      else
	{
	  ION_ABORT ("Don't recognize beadfind type: '" + bfType + "', signal or buffer expected.");
	}
    }
  if (!bfDat.empty())
    {
      bfFile = bfDat;
      if(bfFile.at(0) != '/') {
	bfFile = resultsDir + "/" + bfFile;
      }
    }
  if (!bfBgDat.empty())
    {
      bfBkgFile = bfBgDat;
      if(bfBkgFile.at(0) != '/') {
	bfBkgFile = resultsDir + "/" + bfBkgFile;
      }
    }

  if (bfFile.empty())
    {
      bool onlyG = false;
      for (size_t i = 0; i < values.size(); i++)
	{
	  if (values[i].find ("16 G instead of W1 for Beadfind") == 0 || (values[i].find ("0x10000") == 0))
	    {
	      onlyG = true;
	      break;
	    }

	}
      if (onlyG)
	{
	  bfFile = resultsDir + "/beadfind_pre_0001.dat";
	  bfBkgFile = resultsDir + "/beadfind_pre_0003.dat";
	  signalBased = true;
	}
      else if ( (isFile (possibleBeadfind.c_str()) && signalBased))
	{
	  bfFile = possibleBeadfind;
	  // @todo - figure out proper bg file with keys and flow order
	  bfBkgFile = resultsDir + "/acq_0003.dat";
	}
      else if (isFile (preBeadFind.c_str()))
	{
	  bfFile = preBeadFind2;
	  bfFile2 = "";//preBeadFind2;
	  signalBased = false;
	}
      else if (isFile (postBeadFind.c_str()))
	{
	  bfFile = postBeadFind;
	  signalBased = false;
	}
      else
	{
	  ION_ABORT ("Error: Can't find any beadfind files.");
	}
    }
  cout << "Using " << (signalBased ? "signal" : "buffer") << " based beadfind on file: " << bfFile << endl;
  if (!bfBkgFile.empty()) {
    cout << "BkgFile: " << bfBkgFile << endl;
  }
}

bool DifferentialSeparator::InSpan (size_t rowIx, size_t colIx,
                                    const std::vector<int> &rowStarts,
                                    const std::vector<int> &colStarts,
                                    int span)
{
  for (size_t rIx = 0; rIx < rowStarts.size(); rIx++)
    {
      if ( (int) rowIx >= rowStarts[rIx] && (int) rowIx < (rowStarts[rIx] + span) &&
	   (int) colIx >= colStarts[rIx] && (int) colIx < (colStarts[rIx] + span))
	{
	  return true;
	}
    }
  return false;
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

void DifferentialSeparator::DumpDiffStats (Traces &traces, std::ofstream &o)
{
  o << "x\ty\tsd\t\tsmean\tssd\n";
  size_t nRow = traces.GetNumRow();
  size_t nCol = traces.GetNumCol();
  vector<float> t;
  for (size_t rowIx = 0; rowIx < nRow; rowIx++)
    {
      for (size_t colIx = 0; colIx < nCol; colIx++)
	{
	  o << colIx << "\t" << rowIx;
	  traces.GetTraces (traces.RowColToIndex (rowIx, colIx), t);
	  SampleStats<float> summary;
	  SampleStats<float> step;
	  for (size_t i = 0; i < t.size(); i++)
	    {
	      if (i != 0)
		{
		  step.AddValue (fabs (t[i] - t[i-1]));
		}
	      summary.AddValue (t[i]);
	    }
	  o << "\t" << summary.GetSD();
	  o << "\t" << step.GetMean();
	  o << "\t" << step.GetSD();
	  o << endl;
	}
    }
}

void DifferentialSeparator::PinHighLagOneSd (Traces &traces, float iqrMult)
{
  size_t nRow = traces.GetNumRow();
  size_t nCol = traces.GetNumCol();
  vector<float> t;
  SampleQuantiles<float> quants (10000);
  vector<float> stepSd (nRow * nCol);
  for (size_t rowIx = 0; rowIx < nRow; rowIx++)
    {
      for (size_t colIx = 0; colIx < nCol; colIx++)
	{
	  size_t wellIx = traces.RowColToIndex (rowIx, colIx);
	  traces.GetTraces (wellIx, t);
	  SampleStats<float> step;
	  for (size_t i = 0; i < t.size(); i++)
	    {
	      if (i != 0)
		{
		  step.AddValue (fabs (t[i] - t[i-1]));
		}
	    }
	  stepSd[wellIx] = step.GetSD();
	  quants.AddValue (stepSd[wellIx]);
	}
    }
  float threshold = quants.GetQuantile (.75) + iqrMult * (quants.GetQuantile (.75) - quants.GetQuantile (.25));
  size_t pCount = 0;
  for (size_t rowIx = 0; rowIx < nRow; rowIx++)
    {
      for (size_t colIx = 0; colIx < nCol; colIx++)
	{
	  size_t wellIx = traces.RowColToIndex (rowIx, colIx);
	  if (stepSd[wellIx] >= threshold && ! (mask[wellIx] & MaskExclude))
	    {
	      mask[wellIx] = MaskPinned;
	      pCount++;
	    }
	}
    }
  cout << "LagOne Pinned: " << pCount << " wells. step sd threshold is: " << threshold << " (" << quants.GetQuantile(.5) << "+/-" << (quants.GetQuantile(.75) - quants.GetQuantile(.25)) << ")" << endl;
}


void DifferentialSeparator::CheckFirstAcqLagOne (DifSepOpt &opts)
{


  // size_t numWells = mask.H() * mask.W();
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = ".dat";

  cout << "Checking lag one: " << endl;
  vector<float> t;
  vector<Traces> traces;
  traces.resize (1);
  size_t i = 0;
  char buff[resultsSuffix.size() + resultsRoot.size() + 20];
  const char *p = resultsRoot.c_str();
  const char *s = resultsSuffix.c_str();
  snprintf (buff, sizeof (buff), "%s%.4d%s", p, (int) i, s);
  Image img;
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (opts.ignoreChecksumErrors);
  bool loaded = img.LoadRaw (buff);
  if (!loaded)
    {
      ION_ABORT ("Couldn't load file: " + ToStr (buff));
    }
  img.FilterForPinned (&mask, MaskEmpty, false);
  traces[i].Init (&img, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME, LASTDCFRAME); //frames 0-75, dc offset using 3-12
  PinHighLagOneSd (traces[i], opts.iqrMult);
  img.Close();
}


void DifferentialSeparator::CalcBfT0(DifSepOpt &opts, std::vector<float> &t0vec, const std::string &file) {
  string bfFile = opts.resultsDir + "/" + file;
  Image img;
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (opts.ignoreChecksumErrors);
  bool loaded =   img.LoadRaw(bfFile.c_str());
  if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }
  const RawImage *raw = img.GetImage(); 
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(3);
  t0.SetMinFirstHingeSlope(-5.0/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(300.0/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(-20000.0/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(-10.0/raw->baseFrameRate);
  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStep, opts.t0MeshStep, opts.nCores);
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
  if (opts.outputDebug > 0) {
    string refFile = opts.outData + ".reference_bf_t0." + file + ".txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
  img.Close();

}

void DifferentialSeparator::CalcAcqT0(DifSepOpt &opts, std::vector<float> &t0vec, const std::string &file) {
  string bfFile = opts.resultsDir + "/" + file;
  Image img;
  img.SetImgLoadImmediate (false);
  bool loaded =   img.LoadRaw(bfFile.c_str());
  if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }
  const RawImage *raw = img.GetImage(); 
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(3);
  t0.SetMinFirstHingeSlope(-10.0/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(3.0/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(5.0/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(500.0/raw->baseFrameRate);
  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStep, opts.t0MeshStep, opts.nCores);
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
  if (opts.outputDebug > 0) {
    string refFile = opts.outData + ".reference_bf_t0." + file + ".txt";
    ofstream out(refFile.c_str());
    t0.WriteResults(out);
    out.close();
  }
  img.Close();

}

void DifferentialSeparator::PrintVec(Col<double> &vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    cout << ", " << vec[i];
  }
  cout << endl;
}

void DifferentialSeparator::PrintWell(TraceStore<double> &store, int well, int flow) {
  Col<double> vec(store.GetNumFrames());
  store.GetTrace(well, flow, vec.begin());
  cout << "Well: " << well;
  PrintVec(vec);
}

void DifferentialSeparator::RankWellsBySignal(int flow0, int flow1, TraceStore<double> &store,
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
  mat O(store.GetNumFrames(), okCount);
  mat Z(store.GetNumFrames(), okCount);
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
  mat X = join_rows(O,Z); // L x N

  // Form covariance matrix and get first B (numBasis) vectors
  mat Cov = X * X.t();    // L x L
  arma::Mat<double> EVec;
  arma::Col<double> EVal;
  eig_sym(EVal, EVec, Cov);
  mat V(EVec.n_rows, numBasis);  // L x numBasis
  count = 0;
  for(size_t v = V.n_rows - 1; v >= V.n_rows - numBasis; v--) {
    copy(EVec.begin_col(v), EVec.end_col(v), V.begin_col(count++));
  }

  // Linear algegra trick
  // Normally we'd have to solve via something like t(A) = (t(V)*V)^-1 * t(V) * X but since V is orthonormal 
  // t(V)*V = I and t(A) = t(V)*X and A = t(X) * V
  mat A = X.t() * V;  // N x numBasis
  mat P = A * V.t();  // Prediction N x L 
  mat D = X - P.t();  // Difference L x N
  mat d = mean(abs(D)); // Mean average deviation per well

  // Filter out wells that don't compress well.
  SampleQuantiles<double> samp(1000);
  SampleQuantiles<double> Osamp(1000);
  SampleQuantiles<double> Zsamp(1000);
  Row<double> Olast = O.row(O.n_rows - 1);
  Row<double> Zlast = Z.row(Z.n_rows - 1);
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
      SampleStats<double> sam;
      Row<double> vec = abs(P.row(i) - P.row(i + O.n_cols));
      sam.AddValues(vec.memptr(), vec.n_elem);
      mad[mapping[i]] = fabs(sam.GetMean());
    }
    else {
      filter[mapping[i]] = NotCompressable;
    }
  }
}

void DifferentialSeparator::CreateSignalRef(int flow0, int flow1, TraceStore<double> &store,
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


void DifferentialSeparator::PickCombinedRank(BFReference &reference, vector<vector<float> > &mads,
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
      mapping[count] = mask.ToIndex(row, col);
      count++;
    }
  }
  /// Sort the bf metric
  for (size_t i = 0; i < combinedRanks.size(); i++) {
    rankValues[i].second = i;
    if (filter[mapping[i]] == GoodWell) {
      rankValues[i].first = reference.GetBfMetricVal(mapping[i]);
    }
    else {
      rankValues[i].first = numeric_limits<float>::max();
    }
  }
  sort(rankValues.begin(), rankValues.end());
  for (size_t i = 0; i < combinedRanks.size(); i++) {
    combinedRanks[rankValues[i].second].first += i;
  }
  // Do individual flows
  for (size_t keyIx = 0; keyIx < mads.size(); keyIx++) {
    for (size_t i = 0; i < combinedRanks.size(); i++) {
      rankValues[i].second = i;
      if (filter[mapping[i]] == GoodWell) {
        rankValues[i].first = mads[keyIx][mapping[i]];
      }
      else {
        rankValues[i].first = numeric_limits<float>::max();
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

void DifferentialSeparator::PickCombinedRank(BFReference &reference, vector<vector<float> > &mads,
                                             int rowStep, int colStep,
                                             int numWells,
                                             vector<char> &filter, vector<char> &refWells) {
  refWells.resize(filter.size());
  fill(refWells.begin(), refWells.end(), 0);
  GridMesh<float> mesh;
  mesh.Init(mask.H(), mask.W(), rowStep, colStep);
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++) {
    mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    PickCombinedRank(reference, mads, filter, refWells,
                     numWells,
                     rowStart, rowEnd, colStart, colEnd);
  }
}

bool DifferentialSeparator::Find0merAnd1merFlows(KeySeq &key, TraceStore<double> &store,
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
                                                       TraceStore<double> &store,
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

void DifferentialSeparator::PickReference(TraceStore<double> &store,
                                          BFReference &reference, 
                                          int rowStep, int colStep,
                                          int useKeySignal,
                                          float iqrMult,
                                          int numBasis,
                                          Mask &mask,
                                          int minWells,
                                          vector<char> &filter,
                                          vector<char> &refWells) {
  int count = 0; 
  for (int i = 0; i < reference.GetNumWells(); i++) {
    //    filter[i] = reference.GetType(i) != BFReference::Exclude && reference.GetType(i) != BFReference::Filtered;
    if (filter[i] == GoodWell && (reference.GetType(i) == BFReference::Exclude || reference.GetType(i) == BFReference::Filtered)) {
      //      filter[i] = BeadfindFiltered;
    }
    if (reference.GetType(i) == BFReference::Filtered) {
      count++;
    }
  }
  cout << "Beadfind filtered: " << count << " wells of: " << filter.size() << " (" << float(count)/filter.size() << ")" << endl;
  //  CountReference("After beadfind reference", mFilteredWells);
  vector<vector<float> > mads;
  if (useKeySignal == 1) {
    mads.resize(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
      int flow1 = -1, flow0 = -1;
      mads[i].resize(mask.H() * mask.W(), std::numeric_limits<float>::max());
      bool found = Find0merAnd1merFlows(keys[i], store, flow1, flow0);
      cout << "for key: " << keys[i].name << " using differential flows: " << flow1 << " 1mer and " << flow0 << " 0mer." << endl;
      fill(mads[i].begin(), mads[i].end(), 0);
      if (found) {
        CreateSignalRef(flow0, flow1, store, rowStep, colStep, iqrMult, numBasis,
                        mask, minWells, filter, mads[i]);
        //CountReference("After signal ref", mFilteredWells);
      }
    }
  }
  else if (useKeySignal == 2) {
    mads.resize(1);
    mads[0].resize(mask.H() * mask.W());
    fill(mads[0].begin(), mads[0].end(), 0);
    int flow0 = -1, flow1 = -1;
    bool found = FindCommon0merAnd1merFlows(keys, store, flow0, flow1);
    if (found) {
      CreateSignalRef(flow0, flow1, store, rowStep, colStep, iqrMult, numBasis,
                      mask, minWells, filter, mads[0]);
    }
  }
  else {
    cout << "Not using key signal reference." << endl;
  }
  cout << "Picking reference with: " << mads.size() << " vectors useSignalReference " << useKeySignal << endl;
  PickCombinedRank(reference, mads, rowStep, colStep, minWells, filter, refWells);
  //  CountReference("After signal combined ranke", mFilteredWells);
}

void DifferentialSeparator::CalculateFrames(SynchDat &sdat, int &minFrame, int &maxFrame) {
  vector<float> t0Frames;
  minFrame = -1;
  maxFrame = -1;
  for (size_t i = 0; i < sdat.GetNumRegions(); i++) {
    float t = sdat.T0RegionFrame(i);
    if (t > 0) {
      t0Frames.push_back(t);
    }
  }
  if (t0Frames.size() > 0) {
    sort(t0Frames.begin(), t0Frames.end());
    float t = ionStats::quantile_sorted(t0Frames, .5);
    int midT = floor(t);
    
    minFrame = max(midT - T0_LEFT_OFFSET, 0);
    maxFrame = min(midT + T0_RIGHT_OFFSET, sdat.GetMaxFrames());
  }
}

void DifferentialSeparator::LoadKeySDats(PJobQueue &jQueue, TraceStore<double> &traceStore, BFReference &reference, DifSepOpt &opts) {

  // size_t numWells = mask.H() * mask.W();
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = opts.sdatSuffix;
  size_t numWells = mask.H() * mask.W();
  SetReportSet (mask.H(), mask.W(), opts.wellsReportFile, opts.reportStepSize);
  if (keys.empty())
    {
      DifferentialSeparator::MakeStadardKeys (keys);
    }
  if (opts.outputDebug > 0) {
    std::vector<float> t02;
    CalcBfT0(opts, t02,  "beadfind_pre_0003.dat");
    CalcBfT0(opts, t02, "beadfind_pre_0001.dat");
  }
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  vector<int> rowStarts;
  vector<int> colStarts;
  size_t nRow = traceStore.GetNumRows();
  size_t nCol = traceStore.GetNumCols();
  double percents[3] = {.2, .5, .8};
  //int span = 7;
  Timer first4;
  for (size_t i = 0; i < ArraySize (percents); i++)
    {
      rowStarts.push_back (percents[i] * nRow);
      colStarts.push_back (percents[i] * nCol);
    }
  vector<float> t;
  Timer allLoaded;
  Col<double> traceBuffer;
  traceStore.SetMeshDist (opts.useMeshNeighbors);
  mFilteredWells.resize(numWells);
  fill(mFilteredWells.begin(), mFilteredWells.end(), GoodWell);
  size_t loadMinFlows = max (9, opts.maxKeyFlowLength+2);

  // Ready in the sdata data
  std::vector<LoadSDatJob> loadJobs(loadMinFlows);
  size_t first = 0;
  traceStore.SetFlowIndex (first, first);
  char buff[resultsSuffix.size() + resultsRoot.size() + 21];
  const char *p = resultsRoot.c_str();
  const char *s = resultsSuffix.c_str();
  snprintf (buff, sizeof (buff), "%s%.4d.%s", p, (int) first, s);
  // Do the first flow alone to get timings and finish setup of tracestore
  loadJobs[first].Init(buff, first, &traceStore, &mask, &mFilteredWells, &t0, &opts);
  jQueue.AddJob(loadJobs[first]);
  jQueue.WaitUntilDone();
  // Read following flows multithreaded
  for (size_t i = 1; i < loadMinFlows; i++) {
    traceStore.SetFlowIndex (i, i);
    buff[resultsSuffix.size() + resultsRoot.size() + 21];
    p = resultsRoot.c_str();
    s = resultsSuffix.c_str();
    snprintf (buff, sizeof (buff), "%s%.4d.%s", p, (int) i, s);
    loadJobs[i].Init(buff, i, &traceStore, &mask, &mFilteredWells, &t0, &opts);
    jQueue.AddJob(loadJobs[i]);
  }
  jQueue.WaitUntilDone();
  traceStore.SetFirst();
  // Do some post filtering
  Col<double> tmpData;
  for (size_t fIx = 0; fIx < loadMinFlows; fIx++) {
    for (size_t wIx = 0; wIx < traceStore.GetNumWells(); wIx++) {
      if (traceStore.HaveWell(wIx)) {
	tmpData.resize(traceStore.GetNumFrames());
	traceStore.GetTrace(wIx, fIx, tmpData.begin());
	for (size_t frameIx = 0; frameIx < traceStore.GetNumFrames(); frameIx++) {
	  if(!isfinite(tmpData[frameIx]) || fabs(tmpData[frameIx]) > 10000) {
	    cout << "error at: " << fIx << "\t" << wIx << "\t" << frameIx++ << endl;
	    assert(0);
	  }
	}
      }
    }
  }
  // ofstream storeOut("trace-store.txt");
  // traceStore.Dump(storeOut);
  // storeOut.close();
  cout << "Done loading all traces - took: " << allLoaded.elapsed() <<  " seconds." << endl;

  mRefWells.resize(numWells, 0);
  // PickReference(traceStore, reference, opts.bfMeshStep, opts.bfMeshStep, opts.useSignalReference,
  //               opts.iqrMult, 7, mask, (int)opts.bfMeshStep *opts.bfMeshStep * opts.percentReference,
  //               mFilteredWells, mRefWells);
  
  PickReference(traceStore, reference, 25, 25, opts.useSignalReference,
                opts.iqrMult, 7, mask, ceil(25*25 * opts.percentReference),
                mFilteredWells, mRefWells);

  for (size_t i = 0; i < mRefWells.size(); i++) {
    traceStore.SetReference(i, mRefWells[i] == 1);
  }
  for (size_t i = 0; i < loadMinFlows; i++) {
    traceStore.PrepareReference (i);
  }
}

//void DifferentialSeparator::PredictFlow() {

  // Load up the data for the flow
  // Prepare the reference traces
  // For each well
    // Predict the zeromer
    // Integrate the frames
    // Write to buffer
//}

void DifferentialSeparator::FilterRegionBlobs(Mask &mask,
                                              int rowStart, int rowEnd, int colStart, int colEnd, int chipWidth,
                                        Col<float> &my_metric, vector<char> &filteredWells, int smoothWindow,
                                        int filtWindow, float filtThreshold) {
  Col<double> sum_metric((rowEnd - rowStart + 1) * (colEnd - colStart + 1));
  Col<float> smooth_metric((rowEnd-rowStart) * (colEnd-colStart));
  sum_metric.zeros();
  Col<int> sum_filt_wells(sum_metric.n_rows);
  size_t patch_width = colEnd - colStart + 1;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int orow = row-rowStart;
      int ocol = col-colStart;
      sum_metric[(orow+1) * patch_width + (ocol+1)] = my_metric[row*chipWidth+col] 
        + sum_metric[(orow) * patch_width + (ocol+1)] 
        + sum_metric[(orow+1) * patch_width + (ocol)] 
        - sum_metric[(orow) * patch_width + (ocol)];
    }
  }
  // cout << "Orig:" << endl;
  // for (int r = 0; r < 10; r++) {
  //   for (int c = 0; c < 10; c++) {
  //     printf("%.2f\t", my_metric[r*chipWidth+c]);
  //   }
  //   printf("\n");
  // }
  // cout << "Sum:" << endl;
  // for (int r = 0; r < 10; r++) {
  //   for (int c = 0; c < 10; c++) {
  //     printf("%.2f\t", sum_metric[r*patch_width+c]);
  //   }
  //   printf("\n");
  // }

  int smooth_width = patch_width - 1; 
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int orow = row - rowStart;
      int ocol = col - colStart;
      int srow = max(0,orow-smoothWindow);
      int scol = max(0,ocol-smoothWindow);
      int erow = min(rowEnd-rowStart-1,orow+smoothWindow);
      int ecol = min(colEnd-colStart-1,ocol+smoothWindow); 
      int count = 0; 
      for (int r = srow; r < erow; r++) {
        for (int c = scol; c < ecol; c++) {
          if (my_metric[(r + rowStart) * chipWidth + c + colStart] > 0) {
            count++;
          }
        }
      }
      //      int count = (erow - srow) * (ecol - scol);
      float val = 0;
      if (count > 0) {
        val = (1.0f * sum_metric[(erow+1) * patch_width + (ecol+1)] 
                   + sum_metric[(srow+1) * patch_width + (scol+1)]
                   - sum_metric[(erow+1) * patch_width + (scol+1)]
                   - sum_metric[(srow+1) * patch_width + (ecol+1)])/count; 
      }
      smooth_metric[orow*smooth_width+ocol] = val;
    }
  }
  // cout << "Smooth" << endl;
  // for (int r = 0; r < 10; r++) {
  //   for (int c = 0; c < 10; c++) {
  //     printf("%.2f\t", smooth_metric[r*smooth_width+c]);
  //   }
  //   printf("\n");
  // }
  int good = 0;
  for (size_t i = 0; i < smooth_metric.n_rows; i++) {
    if (smooth_metric[i] > 0) {
      good++;
    }
  }
  if (good < 1000) {
    return;
  }
  Col<float> smooth_metric_sorted(good);
  int sm_index = 0; 
  for (size_t i = 0; i < smooth_metric_sorted.n_rows; i++) {
    if (smooth_metric[i] > 0) {
      smooth_metric_sorted[sm_index++] = smooth_metric[i];
    }
  }
  std::sort(smooth_metric_sorted.begin(), smooth_metric_sorted.end());
  
  float q75 = ionStats::quantile_sorted(smooth_metric_sorted.memptr(), smooth_metric_sorted.n_rows, .75);
  float q25 = ionStats::quantile_sorted(smooth_metric_sorted.memptr(), smooth_metric_sorted.n_rows, .25);
  float iqr = q75 - q25;
  float min_thresh = max(q25 - 3 * iqr, (float)MIN_SD); // for bubbles have minimum threshold
  float max_thresh = q75 + 3 * iqr;
  
  sum_metric.zeros();
  int bad_count = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int orow = row-rowStart;
      int ocol = col-colStart;
      float val = smooth_metric[orow * smooth_width + ocol];
      float f = (val > max_thresh || val < min_thresh) ? 1 : 0;
      bool isExclude = (mask[row * chipWidth + col] & MaskExclude);
      if (isExclude) {
        f = 0;
      }
      if (!isExclude && (val > max_thresh || val < min_thresh)) {
        bad_count++;
      }
      sum_metric[(orow+1) * patch_width + (ocol+1)] = f
        + sum_metric[(orow) * patch_width + (ocol+1)] 
        + sum_metric[(orow+1) * patch_width + (ocol)] 
        - sum_metric[(orow) * patch_width + (ocol)];
    }
  }
  int filt_count = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int orow = row - rowStart; // offset row
      int ocol = col - colStart; 
      int srow = max(0,orow-filtWindow); // start row of our square
      int scol = max(0,ocol-filtWindow);
      int erow = min(rowEnd-rowStart-1,orow+filtWindow); // end row of our square
      int ecol = min(colEnd-colStart-1,ocol+filtWindow); 
      // How many valid wells are there?
      int count = 0;
      for (int r = srow; r < erow; r++) {
        for (int c = scol; c < ecol; c++) {
          if (my_metric[(r + rowStart) * chipWidth + c + colStart] > 0) {
            count++;
          }
        }
      }
      //      assert(count > 0);
      //      float ratioBad = (1.0f * sum_metric[(erow+1) * patch_width + (ecol+1)] - sum_metric[(srow+1) * patch_width + (scol+1)])/count; 
      float ratioBad = 0; 
      if (count > 0) {
        ratioBad = (1.0f * sum_metric[(erow+1) * patch_width + (ecol+1)] 
                   + sum_metric[(srow+1) * patch_width + (scol+1)]
                   - sum_metric[(erow+1) * patch_width + (scol+1)]
                   - sum_metric[(srow+1) * patch_width + (ecol+1)])/count; 
      }
      if (ratioBad > filtThreshold) {
        int rs = max(orow-filtWindow,0);
        int re = min(orow+filtWindow, rowEnd-rowStart);
        int cs = max(ocol-filtWindow, 0);
        int ce = min(ocol+filtWindow, colEnd-colStart);
        for(int r = rs; r < re; r++) {
          for (int c = cs; c < ce; c++) {
            if(filteredWells[(r+rowStart)*chipWidth+(c+colStart)] != RegionTraceSd) {
              filt_count++;
            } 
            filteredWells[(r+rowStart)*chipWidth+(c+colStart)] = RegionTraceSd;
          }
        }
      }
    }
  }
  //  cout << "Region: (" << rowStart << "," << colEnd << ") had " << filt_count << " blob filtered wells. " << bad_count << endl;  
}
                                        
void DifferentialSeparator::FilterBlobs(Mask &mask, int step,
                                        Col<float> &metric, vector<char> &filteredWells, int smoothWindow,
                                        int filtWindow, float filtThreshold) {
  GridMesh<double> mesh;
  mesh.Init(mask.H(), mask.W(), step, step);
  Col<float> my_metric = metric;
  for (size_t i = 0; i < my_metric.n_rows; i++) {
    if (filteredWells[i] != GoodWell || (mask[i] & MaskExclude) || (mask[i] &MaskPinned)) {
      my_metric[i] = 0;
    }
  }
  for (size_t binIx = 0; binIx < mesh.GetNumBin(); binIx++)  {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      mesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      // int goodCount = 0;
      // for (int r = rowStart; r < rowEnd; r++) {
      //   for (int c = colStart; c < colEnd; c++) {
      //     if (!(mask[r * mask.W() + c] & MaskExclude)) {
      //       goodCount++;
      //     }
      //   }
      // }
      FilterRegionBlobs(mask, rowStart, rowEnd, colStart, colEnd, mask.W(),
                        my_metric, filteredWells, smoothWindow, filtWindow, filtThreshold);
  }
}


void DifferentialSeparator::LoadKeyDats(PJobQueue &jQueue, TraceStoreMatrix<double> &traceStore, 
                                        BFReference &reference, DifSepOpt &opts, std::vector<float> &traceSd,
                                        Col<int> &zeroFlows) {

  // size_t numWells = mask.H() * mask.W();
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = "dat";
  size_t numWells = mask.H() * mask.W();
  SetReportSet (mask.H(), mask.W(), opts.wellsReportFile, opts.reportStepSize);
  if (keys.empty()) {  DifferentialSeparator::MakeStadardKeys (keys); }
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  vector<int> rowStarts;
  vector<int> colStarts;
  size_t nRow = traceStore.GetNumRows();
  size_t nCol = traceStore.GetNumCols();
  double percents[3] = {.2, .5, .8};
  //int span = 7;
  Timer first4;
  for (size_t i = 0; i < ArraySize (percents); i++) {
    rowStarts.push_back (percents[i] * nRow);
    colStarts.push_back (percents[i] * nCol);
  }
  vector<float> t;
  Timer allLoaded;
  Col<double> traceBuffer;
  traceStore.SetMeshDist (opts.useMeshNeighbors);
  mFilteredWells.resize(numWells);
  fill(mFilteredWells.begin(), mFilteredWells.end(), GoodWell);
  size_t loadMinFlows = max (9, opts.maxKeyFlowLength+2);
  loadMinFlows = max(loadMinFlows, (size_t)(zeroFlows[zeroFlows.n_rows-1] + 1));
  traceStore.SetSize(T0_RIGHT_OFFSET);
  traceStore.SetT0(t0);
  Mask cncMask(&mask);
  Mat<float> traceSdMin(numWells, loadMinFlows);
  traceSdMin.zeros();
  // Ready in the sdata data
  std::vector<LoadDatJob> loadJobs(loadMinFlows);
  char buff[resultsSuffix.size() + resultsRoot.size() + 21];
  const char *p = resultsRoot.c_str();
  const char *s = resultsSuffix.c_str();
  // Read following flows multithreaded
  for (size_t i = 0; i < loadMinFlows; i++) {
    traceStore.SetFlowIndex (i, i);
    buff[resultsSuffix.size() + resultsRoot.size() + 21];
    p = resultsRoot.c_str();
    s = resultsSuffix.c_str();
    snprintf (buff, sizeof (buff), "%s%.4d.%s", p, (int) i, s);
    loadJobs[i].Init(buff, i, &traceStore, traceSdMin.colptr(i), &mask, &cncMask, &mFilteredWells, &t0, &opts);
    //  loadJobs[i].Run();
    jQueue.AddJob(loadJobs[i]);
  }
  jQueue.WaitUntilDone();

  // ofstream storeOut("trace-store.txt");
  // traceStore.Dump(storeOut);
  // storeOut.close();
  cout << "Done loading all traces - took: " << allLoaded.elapsed() <<  " seconds." << endl;
  mRefWells.resize(numWells, 0);

  traceSd.resize(numWells);
  for (size_t i = 0; i < traceSdMin.n_rows; i++) {
    float minVal = std::numeric_limits<float>::max();
    for (size_t j = 0; j < traceSdMin.n_cols; j++) {
      minVal = min(traceSdMin(i,j), minVal);
    }
    traceSd[i] = minVal;
    if (minVal < MIN_SD) {
      mFilteredWells[i] = LowTraceSd;
    }
  }
  Col<float> metric = traceSdMin.col(zeroFlows[0]);
  if (opts.blobFilter) {
    FilterBlobs(mask, opts.blobFilterStep, metric, mFilteredWells, 3, 5, .5);
  }
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
  //  CountReference("Before picking reference", mFilteredWells);

  PickReference(traceStore, reference, opts.referenceStep, opts.referenceStep, opts.useSignalReference,
                opts.iqrMult, 7, mask, ceil(opts.referenceStep*opts.referenceStep * opts.percentReference),
                mFilteredWells, mRefWells);
  
  int filtered = 0, refChosen = 0, possible = 0;
  for (size_t i = 0; i < mFilteredWells.size(); i++) {
    if (mask[i] == MaskIgnore) {
      mFilteredWells[i] = LowTraceSd;
    }
    if (!(mask[i] & MaskPinned || mask[i] & MaskExclude)) {
      possible++;
      if (mFilteredWells[i] != GoodWell) {
        filtered++;
      }
      if (mRefWells[i] == 1) {
        refChosen++;
      }
    }
  }
  cout << filtered << " wells filtered of: " << possible << " (" <<  (float)filtered/possible << ") " << refChosen << " reference wells." << endl;
  // for (size_t i = 0; i < t0.size(); i++) {
  //   t0[i] = max(-1.0f,t0[i] - 4.0f);
  // }
  // traceStore.SetT0(t0);


  
  for (size_t i = 0; i < mRefWells.size(); i++) {
    traceStore.SetReference(i, mRefWells[i] == 1);
  }
  for (size_t i = 0; i < loadMinFlows; i++) {
    traceStore.PrepareReference (i);
  }
}

void DifferentialSeparator::CalcNumFrames(int tx, std::vector<int> &stamps, int &before_last, int &after_last, int maxStep) {
  before_last = 0;
  for (size_t i = (size_t)tx; i > 1; i--) {
    if (stamps[i] - stamps[i-1] <= maxStep) {
      before_last++;
    }
    else {
      break;
    }
  }
  after_last = 0;
  for (size_t i = (size_t)tx; i < stamps.size() - 1; i++) {
    if (stamps[i+1] - stamps[i] <= maxStep) {
      after_last++;
    }
    else {
      break;
    }
  }
}

void DifferentialSeparator::CheckT0VFC(std::vector<float> &t, std::vector<int> &stamps) {
  SampleStats<double> before;
  SampleStats<double> after;
  int t0_last = -1;
  int before_last = -1;
  int after_last = -1;
  size_t count = 0;
  for (size_t i =0; i < t.size(); i++) {
    int tx = floor(t[i]);
    if (tx > 0) {
      if (tx != t0_last) {
        CalcNumFrames(tx, stamps,before_last,after_last, 70);
        t0_last = tx;
      }
      before.AddValue(before_last);
      after.AddValue(after_last);
      count++;
    }
  }
  cout << "VFC Compression Stats: " << count << " wells with t0. Mean before: " << before.GetMean() << " mean after: " << after.GetMean() << endl;
}

void DifferentialSeparator::OutputWellInfo (TraceStore<double> &store,
					    ZeromerModelBulk<double> &bg,
					    const vector<KeyFit> &wells,
					    int outlierType,
					    int wellIdx,
					    std::ostream &traceOut,
					    std::ostream &refOut,
					    std::ostream &bgOut)
{
  char d = '\t';
  const KeyFit &w = wells[wellIdx];
  vector<double> f (store.GetNumFrames());
  Col<double> ref (store.GetNumFrames());
  Col<double> p (store.GetNumFrames());
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

      bg.ZeromerPrediction (w.wellIdx, flow, store, ref ,p);
      for (size_t fIx = 0; fIx < p.n_rows; fIx++)
	{
	  bgOut << d << p.at (fIx);
	}
      bgOut << endl;
    }

}

void DifferentialSeparator::OutputOutliers (TraceStore<double> &store,
					    ZeromerModelBulk<double> &bg,
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


void DifferentialSeparator::OutputOutliers (DifSepOpt &opts, TraceStore<double> &store,
					    ZeromerModelBulk<double> &bg,
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
void DifferentialSeparator::CalcRegionEmptyStat(H5File &h5File, GridMesh<MixModel> &mesh, TraceStore<double> &store, const string &fileName, 
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
  vector<double> wellData(store.GetNumFrames());
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
  h5File.WriteMatrix(refVarSummaryH5, refVarSummary);
  string refFrameStats = "/separator/refFrameStats";
  h5File.WriteMatrix(refFrameStats, refFrameIqr);
  string bgVarSummaryH5 = "/separator/bgVarSummary";
  h5File.WriteMatrix(bgVarSummaryH5, bgVarSummary);
  string bgFrameStats = "/separator/bgFrameStats";
  h5File.WriteMatrix(bgFrameStats, bgFrameIqr);
}

void DifferentialSeparator::PredictFlow(const std::string &datFile, const std::string &debugFile,
                                        DifSepOpt &opts, Mask &mask,
                                        std::vector<KeyFit> &fits, std::vector<float> &metric, 
                                        Col<double> &time) {
  TraceStoreMatrix<double> traceStore (mask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
                                       1, 1,
				       opts.referenceStep, opts.referenceStep);
  traceStore.SetMinRefProbes (opts.percentReference * opts.referenceStep * opts.referenceStep);
  traceStore.SetMeshDist (opts.useMeshNeighbors);
  traceStore.SetSize(T0_RIGHT_OFFSET);
  traceStore.SetT0(t0);
  traceStore.SetFlowIndex (0, 0);
  size_t num_wells = mask.H() * mask.W();
  vector<float> traceSd(num_wells);
  fill(traceSd.begin(), traceSd.end(), 0);
  LoadDatJob::LoadDat(datFile, &opts, &traceStore, &t0, &mask, &mask, 0, &traceSd[0]);
  for (size_t i = 0; i < mRefWells.size(); i++) {
    traceStore.SetReference(i, mRefWells[i] == 1);
  }
  traceStore.PrepareReference (0);
  metric.resize(num_wells);
  fill(metric.begin(), metric.end(), -1);
  Mat<double> wellFlows, refFlows;
  Mat<float> incorps;
  Col<double> diff(traceStore.GetNumFrames()), predicted;
  ZeromerDiff<double> bg;
  if (!debugFile.empty()) {
    incorps.set_size(num_wells, traceStore.GetNumFrames());
  }
  for (size_t i = 0; i < num_wells; i++) {
    fill(diff.begin(), diff.end(), 0);
    if (fits[i].tauB > 0) {
      TauEBulkErr<double>::FillInData(traceStore, wellFlows, refFlows, predicted, 1, i);
      bg.PredictZeromer(refFlows.unsafe_col(0), time, fits[i].tauB, fits[i].tauE, predicted);
      diff = wellFlows.col(0) - predicted;
    }
    if (incorps.n_rows > 0) {
      for (size_t c = 0; c < diff.n_rows; c++) {
        incorps.at(i,c) = diff.at(c);
      }
    }
    metric[i] = sum(diff);
  }
  if (incorps.n_rows > 0) {
    H5File::WriteMatrix(debugFile + ":/predictions/incorp", incorps, false);
  }
}

void DifferentialSeparator::PredictWellsFlow(DifSepOpt &opts, Mask &mask,
                                             Mat<float> &raw_frames,
                                             Mat<float> &predicted_frames,
                                             Mat<float> &reference_frames,
                                             RawWells &sep_wells,
                                             int flow,
                                             ZeromerModelBulk<double> &zBulk,
                                             Col<double> &time) {
  TraceStoreMatrix<double> traceStore (mask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
                                       1, 1,
				       opts.referenceStep, opts.referenceStep);
  traceStore.SetMinRefProbes (opts.percentReference * opts.referenceStep * opts.referenceStep);
  traceStore.SetMeshDist (opts.useMeshNeighbors);
  traceStore.SetSize(T0_RIGHT_OFFSET);
  traceStore.SetT0(t0);
  traceStore.SetFlowIndex (0, 0);
  size_t num_wells = mask.H() * mask.W();
  vector<float> traceSd(num_wells);
  fill(traceSd.begin(), traceSd.end(), 0);
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = ".dat";

  cout << "Checking lag one: " << endl;
  vector<float> t;
  size_t i = 0;
  char buff[resultsSuffix.size() + resultsRoot.size() + 20];
  const char *p = resultsRoot.c_str();
  const char *s = resultsSuffix.c_str();
  snprintf (buff, sizeof (buff), "%s%.4d%s", p, (int) flow, s);
  LoadDatJob::LoadDat(buff, &opts, &traceStore, &t0, &mask, &mask, 0, &traceSd[0]);
  for (i = 0; i < mRefWells.size(); i++) {
    traceStore.SetReference(i, mRefWells[i] == 1);
  }
  traceStore.PrepareReference (0);
  Mat<double> wellFlows, refFlows;
  Mat<float> incorps;
  Col<double> diff, predicted;
  ZeromerDiff<double> bg;
  int region_wells = opts.predictHeight * opts.predictWidth;
  int nucIx = traceStore.GetNucForFlow(flow);
  for (size_t i = 0; i < num_wells; i++) {
    int row = i / mask.W();
    int col = i % mask.W();
    fill(diff.begin(), diff.end(), 0);
    if (zBulk.HaveModel(i)) {
      KeyBulkFit &kbf = zBulk.GetFit(i);
      TauEBulkErr<double>::FillInData(traceStore, wellFlows, refFlows, predicted, 1, i);
      bg.PredictZeromer(refFlows.unsafe_col(0), zBulk.mTime, kbf.param.at(nucIx,0), kbf.param.at(nucIx,1), predicted);
      Col<double> &dm = zBulk.GetDarkMatter(i);
      predicted = predicted + dm;
      diff = wellFlows.col(0) - predicted;
    }
    if (raw_frames.n_rows > 0 &&
        row >= opts.predictRow && row < opts.predictRow + opts.predictHeight &&
        col >= opts.predictCol && col < opts.predictCol + opts.predictHeight) {
      //        size_t row_index = flow * num_wells + i;
      size_t row_index = flow * region_wells + ((row - opts.predictRow) * opts.predictWidth + col - opts.predictCol);
      raw_frames(row_index, 0) = reference_frames(row_index, 0) = predicted_frames(row_index, 0)= row;
      raw_frames(row_index, 1) = reference_frames(row_index, 1) = predicted_frames(row_index, 1)= col;
      raw_frames(row_index, 2) = reference_frames(row_index, 2) = predicted_frames(row_index, 2)= flow;
      for (size_t fIx = 0; fIx < predicted.n_rows; fIx++) {
        raw_frames.at(row_index, fIx+3) = wellFlows.at(fIx, 0);
        reference_frames(row_index, fIx+3) = refFlows.at(fIx, 0);
        predicted_frames(row_index, fIx+3) = predicted[fIx];
      }
    }
    float val = sum(diff);
    sep_wells.Set(row, col, flow, val);
  }
}

int DifferentialSeparator::Run(DifSepOpt opts) {
  ClockTimer totalTimer;
  // --- Fill in the beadfind reference and buffering
  KClass kc;
  BFReference reference;
  std::vector<float> t02;

  CalcBfT0(opts, t0,  "beadfind_pre_0003.dat");
  CalcAcqT0(opts, t02,  "acq_0007.dat");
  wells.resize (t0.size());
  //  CalcBfT0(opts, t02, "beadfind_pre_0001.dat");
  for (size_t i = 0; i < t0.size(); i++) {
    wells[i].bfT0 = t0[i];
    wells[i].acqT0 = t02[i];
    if (t0[i] > 0 && t02[i] > 0) {
      t0[i] = (.3 * t0[i] + .7 * t02[i]);
    }
    else {
      t0[i] = max(t0[i], t02[i]);
    }
    t0[i] = max(-1.0f,t0[i] - T0_LEFT_OFFSET);
  }
  CheckT0VFC(t0, mBFTimePoints);
  reference.SetT0(t0);
  cout << "Separator region size is: " << opts.regionXSize << "," << opts.regionYSize << endl;
  reference.SetRegionSize (opts.regionXSize, opts.regionYSize);
  reference.SetNumEmptiesPerRegion(opts.percentReference * opts.bfMeshStep * opts.bfMeshStep);
  reference.SetIqrOutlierMult(opts.iqrMult);
  reference.SetDoRegional (opts.useMeshNeighbors == 0);
  string bfDat;

  string bfImgFile;
  string bfImgFile2;
  string bfBkgImgFile;
  DetermineBfFile (opts.resultsDir, opts.signalBased, opts.bfType,
                   opts.bfDat, opts.bfBgDat, bfImgFile, bfImgFile2, bfBkgImgFile);

  LoadInitialMask(opts.mask, opts.maskFile, bfImgFile, mask, opts.ignoreChecksumErrors);
  size_t numWells = mask.H() * mask.W();

  // Look for pinned and lag one
  ReportPinned(mask, "Start");
  if (opts.filterLagOneSD) {
    CheckFirstAcqLagOne(opts);
  }
  ReportPinned(mask, "After LagOne");
  ReportSet reportSet(mask.H(), mask.W());
  if (opts.wellsReportFile.empty()) {
    reportSet.SetStepSize(opts.reportStepSize);
  }
  else  {
    reportSet.ReadSetFromFile (opts.wellsReportFile, 0);
  }

  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = ".dat";
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Before Reference.");
  int qSize = (mask.W() / opts.t0MeshStep + 1) * (mask.H() / opts.t0MeshStep + 1);
  if (opts.nCores <= 0)
    {
      opts.nCores = numCores();
    }
  cout << "Num cores: " << opts.nCores << endl;
  PJobQueue jQueue (opts.nCores, qSize);
  reference.Init (mask.H(), mask.W(),
                  opts.regionYSize, opts.regionXSize,
                  .95, .98);
  
  string h5SummaryRoot;
  if (opts.outputDebug > 0) {
    h5SummaryRoot = opts.outData + ".h5";
    H5File h5file(h5SummaryRoot);
    h5file.Open(true);
    h5file.Close();
    reference.SetDebugH5(h5SummaryRoot);
  }
  vector<double> bfMetric;
  if (opts.signalBased) {
    opts.sdAsBf = false;
    reference.SetComparatorCorrect(opts.doComparatorCorrect);
    reference.SetThumbnail(opts.isThumbnail);
    reference.CalcSignalReference(bfImgFile, bfBkgImgFile, mask, 9);
  }
  else {
    string debugSample = opts.outData + ".bftraces.txt";
    reference.SetDebugFile (debugSample);
    if (bfImgFile2.empty())  {
      //      reference.CalcReference (bfImgFile, mask, BFReference::BFIntMaxSd);
      reference.CalcReference (bfImgFile, mask, BFReference::BFIntMaxSd);
    }
    else {
      cout << "Using average beadfind." << endl;
      reference.CalcDualReference (bfImgFile, bfImgFile2, mask);
    }
  }
  for (int i = 0; i < reference.GetNumWells(); i++) {
    reference.SetBfMetricVal(i, opts.bfMult * reference.GetBfMetricVal(i));
  }
  ReportPinned(mask, "After Reference");
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After reference.");
  if (opts.justBeadfind) {
    DoJustBeadfind (opts, reference);
    return 0;
  }

  // Create the keys if not specified.
  if (keys.empty())  { MakeStadardKeys (keys); }
  for (size_t kIx = 0; kIx < keys.size(); kIx++) {
    opts.maxKeyFlowLength = max ( (unsigned int) opts.maxKeyFlowLength, keys[kIx].usableKeyFlows);
    cout << "key: " << kIx << " min snr is: " << keys[kIx].minSnr << endl;
  }
  Col<int> zeroFlows;
  vector<int> flowsAllZero;
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
  assert(zeroFlows.n_rows > 0);
  cout << "Got " << zeroFlows.n_rows << " all zero flows. {";
  for (size_t i = 0; i < zeroFlows.size(); i++) {
    cout << zeroFlows[i] << ",";
  }
  if (!opts.doubleTapFlows.empty()) {
    vector<std::string> dts;
    split(opts.doubleTapFlows, ',', dts);
    zeroFlows.set_size(dts.size());
    for (size_t i = 0; i < zeroFlows.n_rows; i++) {
      zeroFlows[i] = atoi(dts[i].c_str());
    }
  }
  cout << "}" <<  endl;
  // --- Load up the key flows into our data
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  // TraceStoreMatrix<double> traceStore (mask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
  //                                      opts.maxKeyFlowLength+2, opts.maxKeyFlowLength+2,
  //       			       opts.bfMeshStep, opts.bfMeshStep);
  // TraceStoreMatrix<double> traceStore (mask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
  //                                      opts.maxKeyFlowLength+2, opts.maxKeyFlowLength+2,
  //       			       opts.referenceStep, opts.referenceStep);
  int maxFlow = max(zeroFlows.n_rows > 0 ? zeroFlows[zeroFlows.n_rows -1] + 1 : 0, opts.maxKeyFlowLength+2);
  TraceStoreMatrix<double> traceStore (mask, T0_RIGHT_OFFSET, opts.flowOrder.c_str(), 
                                       maxFlow, maxFlow,
         			       opts.referenceStep, opts.referenceStep);

  traceStore.SetMinRefProbes (opts.percentReference * opts.referenceStep * opts.referenceStep);
  std::string firstFile = opts.resultsDir + "/acq_0000.dat";
  if (H5File::IsH5File(firstFile.c_str())) {
    opts.doSdat = true;
    opts.sdatSuffix = "dat";
  }
  vector<float> traceSdMin(numWells);

  totalTimer.PrintMilliSeconds(cout, "Total Timer: Before Loading Dats.");
  if (opts.doSdat) {
    LoadKeySDats (jQueue, traceStore, reference, opts);
  } 
  else {
    LoadKeyDats (jQueue, traceStore, reference, opts, traceSdMin, zeroFlows);
  }
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Loading Dats.");



  // Currently time is just linear
  mTime.set_size (traceStore.GetNumFrames());
  for (size_t i = 0; i < mTime.n_rows; i++)  { mTime[i] = i; }

  // --- Do inital fit of zeromers and tauE
  ZeromerDiff<double> bg;
  vector<KeyReporter<double> *> reporters;
  GridMesh<SampleQuantiles<double> > emptyEstimates;
  emptyEstimates.Init (mask.H(), mask.W(), opts.tauEEstimateStep, opts.tauEEstimateStep);
  
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Before fitting param.");
  std::vector<double> dist;
  std::vector<std::vector<float> *> values;
  zModelBulk.SetFiltered(mFilteredWells);
  zModelBulk.Init (opts.nCores, emptyEstimates.GetNumBin());
  zModelBulk.SetRegionSize (opts.tauEEstimateStep, opts.tauEEstimateStep);
  keyAssignments.resize (wells.size());
  fill(keyAssignments.begin(), keyAssignments.end(), -1);
   // CountReference("Before Zeromers", mFilteredWells);
  ClockTimer zeromerTimer;
  zModelBulk.SetTime (mTime);
  
  string h5tau_file = opts.outData + ".tau.h5";
  H5File h5_tau_file(h5tau_file);
  h5_tau_file.Open(true);
  Mat<float> taub_mat(numWells,zeroFlows.n_rows);
  taub_mat.fill(0);
  Mat<float> taue_mat(numWells,zeroFlows.n_rows);
  taue_mat.fill(0);
  cout << "Starting fitting zeromers." << endl;
  for (size_t i = 0; i < zeroFlows.n_rows; i++) {
    Col<int> z_flow(1);
    z_flow[0] = zeroFlows[i];
    zModelBulk.FitWellZeromers (jQueue,
                                traceStore,
                                keyAssignments,
                                z_flow,
                                keys);
    for (size_t r = 0; r < numWells; r++) {
      if (zModelBulk.HaveModel(r)) {
        KeyBulkFit &kbf = zModelBulk.GetFit(r);
        taub_mat(r,i) = kbf.param.at(0,0);
        taue_mat(r,i) = kbf.param.at(0,1);
      }
    }
  }
  h5_tau_file.WriteMatrix("/tau_e", taue_mat);
  h5_tau_file.WriteMatrix("/tau_b", taub_mat);
  h5_tau_file.Close();

  zModelBulk.FitWellZeromers (jQueue,
                              traceStore,
                              keyAssignments,
                              zeroFlows,
                              keys);
  vector<KeyBulkFit> zeromerTauFit(numWells);
  for (size_t i = 0; i < numWells; i++) {
    if (zModelBulk.HaveModel(i)) {
      zeromerTauFit[i] = zModelBulk.GetFit(i);
    }
  }
  vector<string> words;
  split(opts.predictRegion, ',', words);
  ION_ASSERT(words.size() == 4, "Must specify row, height, col, width");
  opts.predictRow = atoi(words[0].c_str());
  opts.predictHeight = atoi(words[1].c_str());
  opts.predictCol = atoi(words[2].c_str());
  opts.predictWidth = atoi(words[3].c_str());
  if (opts.predictFlowStart >= 0 && opts.predictFlowEnd >= 0 && opts.predictFlowEnd > opts.predictFlowStart) {
    int num_flows = opts.predictFlowEnd - opts.predictFlowStart;
    int wells_save = opts.predictHeight * opts.predictWidth;
    int num_frames = traceStore.GetNumFrames();
    //    size_t total_rows = wells_save * num_flows;
    size_t total_rows = wells_save * num_flows;
    string wells_path = opts.outData + ".wells";
    // sep_wells.SetCols(mask.W());
    // sep_wells.SetRows(mask.H());
    RawWells sep_wells(wells_path.c_str(), mask.H(), mask.W());
    sep_wells.SetCols(mask.W());
    sep_wells.SetRows(mask.H());
    sep_wells.SetFlows(num_flows);
    sep_wells.SetFlowOrder(opts.flowOrder);
    sep_wells.SetChunk(0, sep_wells.NumRows(), 0, sep_wells.NumCols(),  opts.predictFlowStart, opts.predictFlowEnd);
    sep_wells.OpenForWrite();
    Mat<float> raw_frames(total_rows, num_frames), predicted_frames(total_rows, num_frames), reference_frames(total_rows, num_frames);
    raw_frames.fill(-1.0f);
    predicted_frames.fill(-1.0f);
    reference_frames.fill(-1.0f);
    for (int flowIx = opts.predictFlowStart; flowIx < opts.predictFlowEnd; flowIx++) {
      cout << "Doing prediction for flow: " << flowIx << endl;
      PredictWellsFlow(opts, mask, raw_frames, predicted_frames, reference_frames, sep_wells, flowIx, zModelBulk, mTime);
    }
    string traces_file = opts.outData + ".traces.h5";
    H5File h5file_traces(traces_file);
    h5file_traces.Open(true);
    h5file_traces.WriteMatrix("/traces/raw", raw_frames);
    h5file_traces.WriteMatrix("/traces/ref", reference_frames);
    h5file_traces.WriteMatrix("/traces/predicted", predicted_frames);
    h5file_traces.Close();
    sep_wells.WriteWells();
    sep_wells.WriteInfo();
    sep_wells.WriteRanks();
    sep_wells.Close();
  }

  for (size_t i = 0; i < numWells; i++) {
    if (zModelBulk.HaveModel(i)) {
      KeyBulkFit kf = zModelBulk.GetFit(i);
      wells[i].tauB = kf.param.at(0,0);
      wells[i].tauE = kf.param.at(0,1);
    }
    else {
      wells[i].tauB = -1;
      wells[i].tauE = -1;
    }
  }

  // ofstream zout("zeromer-dump.txt");
  // zModelBulk.Dump(zout);
  // zout.close();
   // CountReference("After Zeromers", mFilteredWells);
  zeromerTimer.PrintMilliSeconds(cout, "Fitting zeromers took:");
  SampleQuantiles<float> tauEQuant (10000);
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Fitting param.");

  // Stats for some of the distributions we'd like to keep track of
  SampleStats<float> sdStats;
  SampleQuantiles<float> traceMean (10000);
  SampleQuantiles<float> traceSd (10000);
  SampleQuantiles<float> sdQuantiles (10000);
  SampleQuantiles<float> sdKeyQuantiles (10000);
  SampleQuantiles<float> bfQuantiles (10000);
  SampleQuantiles<float> onemerQuantiles (10000);
  SampleQuantiles<float> madQuantiles (10000);
  SampleQuantiles<float> peakSigKeyQuantiles (10000);
  SampleQuantiles<float> peakSigQuantiles (10000);

  vector<KeyClassifyTauEJob> finalJobs (emptyEstimates.GetNumBin());

  
  for (size_t i = 0; i < numWells; i++) {
    wells[i].traceSd = traceSdMin[i];
    // Trace sd is something different in signal based estimation.
    if (!opts.signalBased) {
      wells[i].traceSd = min(wells[i].traceSd, reference.GetTraceSd(i));
    }
  }

  // Median incorporation shape, we project onto this so if shape changes need to be updated
  //  double incorpShape[19] = { 0.00983675,0.0304656,-0.0236857,-0.0118875,-2.06058e-05,0.0602131,0.111392,0.193223,0.275271,0.33843,0.378151,0.382406,0.361487,0.326388,0.280362,0.255308,0.20398,0.160868, 0.129953 };
  double incorpShape[19] = {0,0.803257,0.567454,0.45336,0.936769,2.92208,7.64964,16.2563,29.1471,45.8955,61.698,73.3103,79.2256,80.6034,79.0002,75.0392,69.8592,64.3944,58.9767};
  Col<double> medianIncorp(19);
  copy(&incorpShape[0], &incorpShape[19], medianIncorp.begin());

  medianIncorp = medianIncorp / norm (medianIncorp,2);
  //  medianIncorp.raw_print(cout, "median incorp");
  double basicTauEEst = 6;

  GridMesh<SampleQuantiles<float> > bfMesh;
  bfMesh.Init (mask.H(), mask.W(), opts.clusterMeshStep, opts.clusterMeshStep);
  for (size_t i = 0; i < bfMesh.GetNumBin(); i++) {
    SampleQuantiles<float> &x = bfMesh.GetItem (i);
    x.Init (10000);
  }
  ClockTimer keyFitTimer;
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Starting Calling Live.");
  // --- Setup for final fit and key calls
  SampleKeyReporter<double> report (opts.outData, numWells);
  report.SetReportSet (reportSet.GetReportIndexes());
  reporters.push_back (&report);
  AvgKeyReporter<double> avgReport (keys, opts.outData, opts.flowOrder, opts.analysisDir);
  KeySummaryReporter<double> keySumReport;
  keySumReport.Init (opts.flowOrder, opts.analysisDir, mask.H(), mask.W(),
                     std::min (128,mask.W()), std::min (128,mask.H()), keys);

  keySumReport.SetMinKeyThreshold (1, 15);
  keySumReport.SetMinKeyThreshold (0, 10);

  if (opts.outputDebug > 0) {
    avgReport.SetMinKeyThreshold(1, 0);
    avgReport.SetMinKeyThreshold(0, 0);
    reporters.push_back(&avgReport);
  }
  mRegionIncorpReporter.Init(opts.outData, mask.H(), mask.W(),
			     opts.regionXSize, opts.regionYSize,
			     keys, t0);
  mRegionIncorpReporter.SetMinKeyThreshold(1, 0);
  mRegionIncorpReporter.SetMinKeyThreshold(0, 0);
  reporters.push_back(&keySumReport);
  reporters.push_back(&mRegionIncorpReporter);
   // CountReference("Before keys", mFilteredWells);
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++) {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    finalJobs[binIx].Init (rowStart,rowEnd,colStart,colEnd,
                           min(keys[0].minSnr,keys[1].minSnr), &mask, &wells, &keys,
                           //                           opts.minSnr, &mask, &wells, &keys,
                           &mTime, basicTauEEst, &medianIncorp,
                           &zModelBulk,
                           &reporters, &traceStore, opts.maxKeyFlowLength,
                           &emptyEstimates, tauEQuant);
    jQueue.AddJob (finalJobs[binIx]);
  }
  jQueue.WaitUntilDone();
  // CountReference("After keys", mFilteredWells);

  for (size_t rIx = 0; rIx < reporters.size(); rIx++) { reporters[rIx]->Finish(); }
  keyFitTimer.PrintMilliSeconds(cout, "Calling keys took.");
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Calling Keys.");
  /// --- Check to make sure we got some live wells, do we really need this anymore?
  int notExcludePinnedWells = 0;
  for (size_t i = 0; i < numWells; i++) {
    if (! (mask[i] & MaskExclude || mask[i] & MaskPinned)) {
      notExcludePinnedWells++;
    }
  }
  int gotKeyCount = 0;
  for (size_t idx = 0; idx < wells.size(); idx++) {
    if (wells[idx].keyIndex >= 0  && wells[idx].mad < opts.maxMad) { gotKeyCount++; }
  }

  int minOkCount = max (10.0, opts.minRatioLiveWell * notExcludePinnedWells);
  if (gotKeyCount <= minOkCount) { ION_ABORT_CODE ("Only got: " + ToStr (gotKeyCount) + " key passing wells. Couldn't find enough (" + ToStr (minOkCount) + ") wells with key signal.", DIFFSEP_ERROR); }
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Fitting Filters.");
  /// --- Calculate filters based on distribution
  peakSigKeyQuantiles.Clear();
  for (size_t i = 0; i < numWells; i++) {
    if (isfinite (wells[i].mad) && wells[i].mad >= 0) {
      madQuantiles.AddValue (wells[i].mad);
      traceMean.AddValue (wells[i].traceMean);
      traceSd.AddValue (wells[i].traceSd);
    }
    //    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad && isfinite(wells[i].peakSig))
    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad) {
      sdStats.AddValue (wells[i].sd);
      sdQuantiles.AddValue (wells[i].sd);
      peakSigQuantiles.AddValue (wells[i].peakSig);

      if (isfinite (wells[i].onemerAvg)) {
        onemerQuantiles.AddValue (wells[i].onemerAvg);
      }
    }
    if (wells[i].keyIndex >= 0) {
      peakSigKeyQuantiles.AddValue (wells[i].peakSig);
      sdKeyQuantiles.AddValue(wells[i].sd);
    }
  }
  // double minTfPeak = min ((double)opts.minTfPeakMax, peakSigQuantiles.GetQuantile (.75) + opts.tfFilterQuantile * IQR (peakSigQuantiles));
  // double minLibPeak = min ((double)opts.minLibPeakMax,peakSigQuantiles.GetQuantile (.75) + opts.libFilterQuantile * IQR (peakSigQuantiles));
  double minTfPeak = opts.minTfPeakMax;
  double minLibPeak = 10; //oopts.minLibPeakMax;
  cout << "Min Tf peak is: " << minTfPeak << " lib peak is: " << minLibPeak << endl;

  double madThreshold = madQuantiles.GetQuantile (.75) + (3 * IQR (madQuantiles));
  madThreshold = max(madThreshold, 10.0);
  double varThreshold = sdQuantiles.GetQuantile(.5) + (sdQuantiles.GetQuantile(.5) - sdQuantiles.GetQuantile(.05));
  //  double varThreshold = sdQuantiles.GetQuantile(.75) + 3 * (sdQuantiles.GetQuantile(.75) - sdQuantiles.GetQuantile(.25));
  //double varBeadThreshold = sdQuantiles.GetMedian() + LowerQuantile(sdQuantiles);
  double meanSigThreshold = std::numeric_limits<double>::max() * -1;
  if (onemerQuantiles.GetNumSeen() > 10) {
    meanSigThreshold = onemerQuantiles.GetMedian() + (2.5 * IQR (onemerQuantiles) /2);
  }

  double peakSigThreshold = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10) {
    peakSigThreshold = peakSigQuantiles.GetMedian() + (IQR (peakSigQuantiles)) /2; //sdStats.GetMean() + (2.5*sdStats.GetSD());
  }
  double peakSigEmptyThreshold = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10) {
    //    peakSigEmptyThreshold = peakSigQuantiles.GetMedian() + (4* (IQR (peakSigQuantiles)) /2); //sdStats.GetMean() + (2.5*sdStats.GetSD());    
    peakSigEmptyThreshold = peakSigQuantiles.GetMedian() + (peakSigQuantiles.GetMedian() - peakSigQuantiles.GetQuantile(.05));
    cout << "Peak Signal Empty Threshold: " << peakSigEmptyThreshold << endl;
    peakSigEmptyThreshold = min(peakSigEmptyThreshold, 10.0);
  }

  if (peakSigKeyQuantiles.GetNumSeen() > 10) {
    cout << "Key Peak distribution is: " << peakSigKeyQuantiles.GetMedian() << " +/- " <<  IQR (peakSigKeyQuantiles) /2 << " "
         << peakSigKeyQuantiles.GetQuantile (.1) << ", " << peakSigKeyQuantiles.GetQuantile (.25) << ", "
         << peakSigKeyQuantiles.GetQuantile (.5) << ", " << peakSigKeyQuantiles.GetQuantile (.75) << endl;

  }

  double keyPeakSigThresh = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10) {
    cout << "Empty Peak distribution is: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR (peakSigQuantiles) /2 << endl;
    keyPeakSigThresh = peakSigQuantiles.GetQuantile (.75);
    cout << "Key peak signal theshold: " << keyPeakSigThresh
         << " (" << peakSigQuantiles.GetQuantile (.75) << ", " << IQR (peakSigQuantiles) << ")" << endl;
    if (peakSigQuantiles.GetNumSeen() > 10) {
      cout << "Empty Key Peak distribution is: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR (peakSigQuantiles) /2 << " "
           << peakSigQuantiles.GetQuantile (.1) << ", " << peakSigQuantiles.GetQuantile (.25) << ", "
           << peakSigQuantiles.GetQuantile (.5) << ", " << peakSigQuantiles.GetQuantile (.75) << endl;
    }
  }
  else {
    cout << "Saw less than 10 peakSigQuantiles." << endl;
  }

  double refMinSD = sdQuantiles.GetMedian() - (3 * (IQR(sdQuantiles))/2);
  double refMaxSD = sdQuantiles.GetQuantile(.5);// + (opts.sigSdMult * (IQR(sdQuantiles))/2);
  double varMin =LowerQuantile (sdQuantiles);
  double traceMeanThresh = traceMean.GetMedian() - 3 * (IQR (traceMean) /2);
  double traceSDThresh = traceSd.GetMedian() - 5 * (LowerQuantile (traceSd));
  // if (opts.signalBased) {
  //   PredictFlow (bfImgFile, opts.outData, opts.ignoreChecksumErrors, opts, traceStore, zModelBulk);
  //   opts.doRecoverSdFilter = false;
  // }
  // else {
  if (opts.signalBased) { 
    std::vector<float> incorpMetric;
    PredictFlow(opts.resultsDir + "/beadfind_pre_0004.dat", h5SummaryRoot, opts, mask, wells, incorpMetric, mTime);
    for (size_t i = 0; i < numWells; i++) {
      wells[i].bfMetric = incorpMetric[i];
    }
  }
  else {
    for (size_t i = 0; i < numWells; i++) {
      wells[i].bfMetric = wells[i].bufferMetric = reference.GetBfMetricVal (i);
    }
  }
  
  if (opts.signalBased) {
    opts.doRecoverSdFilter = false;
  }
  for (size_t i = 0; i < numWells; i++) {
    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad && isfinite (wells[i].bfMetric))  {
      bfQuantiles.AddValue (wells[i].bfMetric);
      SampleQuantiles<float> &bfQ = bfMesh.GetItem (bfMesh.GetBin (wells[i].wellIdx));
      bfQ.AddValue (wells[i].bfMetric);
    }
  }

  cout << "Mad threshold is: " << madThreshold << " for: " << madQuantiles.GetMedian() << " +/- " << (madQuantiles.GetQuantile (.75) - madQuantiles.GetQuantile(.25)) << endl;
  cout << "Var threshold is: " << varThreshold << " for: " << sdQuantiles.GetMedian() << " +/- " <<  IQR (sdQuantiles) /2 << endl;
  cout << "Var min is: " << varMin << " for: " << sdQuantiles.GetMedian() << " +/- " << (sdQuantiles.GetQuantile (.75)) << endl;
  if (onemerQuantiles.GetNumSeen() > 10) {
    cout << "Signal threshold is: " << meanSigThreshold << " for: " << onemerQuantiles.GetMedian() << " +/- " <<  IQR (onemerQuantiles) /2 << endl;
  }
  else {
    cout << "Saw less than 10 wells with onemer signal." << endl;
  }
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After fitting filters.");
  cout << "Mean Trace threshold: " << traceMeanThresh << " median: " << traceMean.GetMedian() << " +/- " << traceMean.GetQuantile (.25) << endl;
  // Set up grid mesh for beadfinding.
  ofstream modelOut;
  if (opts.outputDebug > 0) {
    string modelFile = opts.outData + ".mix-model.txt";
    modelOut.open(modelFile.c_str());
    modelOut << "bin\tbinRow\tbinCol\trowStart\trowEnd\tcolStart\tcolEnd\tcount\tmix\tmu1\tvar1\tmu2\tvar2\tthreshold\trefMean" << endl;
  }
  GridMesh<MixModel> modelMesh;
  opts.bfMeshStep = min (min (mask.H(), mask.W()), opts.bfMeshStep);
  cout << "bfMeshStep is: " << opts.bfMeshStep << endl;
  modelMesh.Init (mask.H(), mask.W(), opts.clusterMeshStep, opts.clusterMeshStep);
  SampleStats<double> bfSnr;

  double bfMinThreshold = bfQuantiles.GetQuantile (.02);
  cout << "Bf min threshold is: " << bfMinThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " << ( (bfQuantiles.GetQuantile (.75) - bfQuantiles.GetQuantile (.25)) /2) << endl;

  // Should we use standard deviation of signal as the beadfind metric
  // (eg cluster signal vs no signal instead of buffering vs no
  // buffering.
  if (opts.sdAsBf) {
    cout << "Using signal for clustering" << endl;
    for (size_t i = 0; i < numWells; i++) {
      reference.SetBfMetricVal(i, wells[i].sd);
      wells[i].bfMetric = wells[i].sd;
    }
  }
  int refCount = 0;
  for (size_t i = 0; i < numWells; i++) {
    if(traceStore.IsReference(i)) {
      wells[i].isRef = true;
      refCount++;
    }
    else { 
      wells[i].isRef = false;
    }
  }
  cout << "Got: " << refCount << " reference wells." << endl;
  // Anchor clusters based off of set of good live beads and the empty wells.
  for (size_t i = 0; i < numWells; i++) {
    if (wells[i].snr >= opts.minTauESnr && wells[i].peakSig > 70) {
      wells[i].goodLive = true;
    }
  }

  // Do clustering for wells into beads and empties for each
  // region. Wells will be classified based on smoothed cluster
  // parameters later.
   // CountReference("Before clustering", mFilteredWells);
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Before Clustering.");
  opts.minBfGoodWells = max (300, (int) (opts.clusterMeshStep * opts.clusterMeshStep * .5));
  for (size_t binIx = 0; binIx < modelMesh.GetNumBin(); binIx++)
    {
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      modelMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      MixModel &model = modelMesh.GetItem (binIx);
      // Center the reference wells in this region to make things comparable.
      double meanBuff = 0;
      size_t meanCount = 0;
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          size_t idx = traceStore.WellIndex(row, col);
          if (wells[idx].isRef) {
            meanCount++;
            meanBuff += wells[idx].bufferMetric;
          }
        }
      }
      if (meanCount > 0) {
        meanBuff /= meanCount;
      }
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          size_t idx =  traceStore.WellIndex(row, col);
          wells[idx].bufferMetric -= meanBuff;
        }
      }

      ClusterRegion (rowStart, rowEnd, colStart, colEnd, madThreshold, opts.minTauESnr,
		     opts.minBfGoodWells, reference, wells, opts.clusterTrim, model);
      if ( (size_t) model.count > opts.minBfGoodWells)
	{
	  double bf = ( (model.mu2 - model.mu1) / ( (sqrt (model.var2) + sqrt (model.var1)) /2));
	  if (isfinite (bf) && bf > 0)
	    {
	      bfSnr.AddValue (bf);
	    }
	  else
	    {
	      cout << "Region: " << binIx << " has snr of: " << bf << " " << model.mu1 << "  " << model.var1 << " " << model.mu2 << " " << model. var2 << endl;
	    }
	}

      int binRow, binCol;
      modelMesh.IndexToXY (binIx, binRow, binCol);
      if (opts.outputDebug > 0) {
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

  bfMask.Init (&mask);
  int notGood = 0;
  vector<MixModel *> bfModels;
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    bfMask[bIx] = mask[bIx];
    if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned || mFilteredWells[bIx] != GoodWell) {
      continue;
    }
    if (wells[bIx].keyIndex < 0) {
      size_t row, col;
      double weight = 0;
      MixModel m;
      int good = 0;
      traceStore.WellRowCol (wells[bIx].wellIdx, row, col);
      modelMesh.GetClosestNeighbors (row, col, opts.bfNeighbors, dist, bfModels);
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
      if (good == 0) {
        notGood++;
        bfMask[bIx] = MaskIgnore;
        wells[bIx].flag = WellBfBad;
      }
      else {
        m.mu1 = m.mu1 / weight;
        m.mu2 = m.mu2 / weight;
        m.var1 = m.var1 / weight;
        m.var2 = m.var2 / weight;
        m.mix = m.mix / weight;
        m.count = m.count / weight;
        m.var1sq2p = 1 / sqrt (2 * DualGaussMixModel::GPI * m.var1);
        m.var2sq2p = 1 / sqrt (2 * DualGaussMixModel::GPI * m.var2);
        double ownership = 0;
        int bCluster = DualGaussMixModel::PredictCluster (m, wells[bIx].bfMetric, opts.bfThreshold, ownership);
        if (bCluster == 2) {
          wells[bIx].flag = WellBead;
          bfMask[bIx] = MaskBead;
        }
        else if (bCluster == 1) {
          wells[bIx].flag = WellEmpty;
          bfMask[bIx] = MaskEmpty | MaskReference;
        }
      }
    }
  }
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Clustering.");

  SampleQuantiles<float> sepRefSdQuantiles (10000);
  SampleQuantiles<float> sepRefBfQuantiles (10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++)  {
    if (wells[bIx].isRef) {
      sepRefBfQuantiles.AddValue(wells[bIx].bufferMetric);
      sepRefSdQuantiles.AddValue(wells[bIx].sd);
    }
  }

  cout << "Sep Quantiles:" << endl;
  for (size_t i = 0; i <= 100; i+=5) {
    cout << i << "\t" << sepRefBfQuantiles.GetQuantile(i/100.0f) << "\t" << sepRefSdQuantiles.GetQuantile(i/100.0f) << endl;
  }
  float sepRefBfThresh = sepRefBfQuantiles.GetQuantile(.75) + 3 * IQR(sepRefBfQuantiles);
  //float sepRefBfThresh = sepRefBfQuantiles.GetQuantile(.5) + sepRefBfQuantiles.GetQuantile(.5) - sepRefBfQuantiles.GetQuantile(.05);
  float sepRefSdThresh = sepRefSdQuantiles.GetQuantile(.75) + 3 * IQR(sepRefSdQuantiles);
  cout << "Sep BF Thresh: " << sepRefBfThresh << endl;
  cout << "Sep SD Thresh: " << sepRefSdThresh << endl;

  totalTimer.PrintMilliSeconds(cout, "Total Timer: Before Post Filtering.");
  SampleQuantiles<float> sdEmptyQuantiles (10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++)  {
    if (wells[bIx].flag == WellEmpty || wells[bIx].isRef == true) {
      sdEmptyQuantiles.AddValue(wells[bIx].sd);
    }
  }
  double varEmptyThreshold = sdEmptyQuantiles.GetQuantile(.5) + (sdEmptyQuantiles.GetQuantile(.5) - sdEmptyQuantiles.GetQuantile(.1));
  //  double varEmptyThreshold = sdEmptyQuantiles.GetQuantile(.75) + 3 * IQR(sdEmptyQuantiles);
  cout << "Empty var threshold is: " << varEmptyThreshold << " for: " << sdEmptyQuantiles.GetMedian() << " +/- " <<  IQR (sdEmptyQuantiles) /2 << " for: " << sdEmptyQuantiles.GetNumSeen() << endl;
  varThreshold = min(varEmptyThreshold, varThreshold);
  // --- For each bead call based on filters and nearest gridmesh neighbors

  int overMaxMad = 0,  tooVar = 0, badFit = 0, noTauE = 0, varMinCount = 0, traceMeanMinCount = 0, sdRefCalled = 0, badSignal = 0, beadLow = 0;
  int tooBf = 0;
  int tooTauB = 0;
  int poorSignal = 0;
  int poorLibPeakSignal = 0;
  int poorTfPeakSignal = 0;
  int emptyWithSignal = 0;
  int tooRefVar = 0;
  int tooRefBuffer = 0;
  int filteredWells = 0;
   // CountReference("Before counting", mFilteredWells);
  SampleQuantiles<float> libSnrQuantiles(10000);
  SampleQuantiles<float> tfSnrQuantiles(10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++)
    {
      if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned) {
	continue;
      }
      if (wells[bIx].traceSd < MIN_SD) {
        filteredWells++;
        bfMask[bIx] = MaskIgnore;
        continue;
      }
      if (wells[bIx].keyIndex < 0 && (mFilteredWells[bIx] == LowTraceSd || mFilteredWells[bIx] == PinnedExcluded) ) {
        filteredWells++;
	bfMask[bIx] = MaskIgnore;
	continue;
      }
      if ((bfMask[bIx] & MaskEmpty) && (mFilteredWells[bIx] != GoodWell)) {
	bfMask[bIx] &= ~MaskReference;
	continue;
      }
      if (bfMask[bIx] & MaskReference && wells[bIx].sd > sepRefSdThresh) {
        bfMask[bIx] &= ~MaskReference;
        wells[bIx].flag = WellEmptyVar;
        tooRefVar++;
        continue;
      }
      // if (bfMask[bIx] & MaskReference && wells[bIx].bufferMetric > sepRefBfThresh) {
      //   bfMask[bIx] &= ~MaskReference;
      //   wells[bIx].flag = WellBfBufferFilter;
      //   tooRefBuffer++;
      //   continue;
      // }
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
      // if (wells[bIx].keyIndex == -1 && wells[bIx].sd <= varMin) {
      //   bfMask[bIx] = MaskIgnore;
      //   varMinCount++;
      //   continue;
      // }
      //   if (opts.doRemoveLowSignalFilter && wells[bIx].keyIndex >=0 && ((wells[bIx].peakSig < peakSigThreshold && wells[bIx].snr < 8) || wells[bIx].onemerAvg < 10)) {
      //    if (opts.doRemoveLowSignalFilter && wells[bIx].keyIndex >=0 && (wells[bIx].sd < varBeadThreshold)) {

      /* if (wells[bIx].keyIndex >=0 && ((wells[bIx].onemerAvg < meanSigThreshold && wells[bIx].snr < 10) || wells[bIx].onemerAvg < 20)) { */
      /*  poorSignal++; */
      /*  bfMask[bIx] = MaskIgnore; */
      /*  continue; */
      /* } */
      // if (opts.doMeanFilter && wells[bIx].keyIndex < 0 && (wells[bIx].traceMean <= traceSDThresh)) {
      //   wells[bIx].flag = WellMeanFilter;
      //   bfMask[bIx] = MaskIgnore;
      //   traceMeanMinCount++;
      //   continue;
      // }
      // if (opts.doEmptyCenterSignal && (wells[bIx].onemerAvg < refMinSig || wells[bIx].onemerAvg > refMaxSig) && wells[bIx].keyIndex < 0) {
      //   bfMask[bIx] = MaskIgnore;
      //   badSignal++;
      //   continue;
      //  }
      if (wells[bIx].keyIndex == 0) {
	libSnrQuantiles.AddValue(wells[bIx].snr);
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
   CountReference("After filters.", mFilteredWells);
  // --- Apply filters
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Cluster.");
  double bfThreshold = bfEmptyQuantiles.GetQuantile (.75) + (3 * IQR (bfEmptyQuantiles));
  cout << "Bf threshold is: " << bfThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " <<  IQR (bfQuantiles) << endl;
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
    // if (bfMask[bIx] & MaskEmpty && wells[bIx].keyIndex < 0 && wells[bIx].sd >= varThreshold) {
    //   bfMask[bIx] &= ~MaskReference;
    //   wells[bIx].flag = WellSdFilter;
    //   tooVar++;
    //   continue;
    // }
    // if (wells[bIx].keyIndex >= 0 && wells[i].peakSig < peakSigThreshold) {
    //   bfMask[bIx] = MaskDud;
    //   wells[bIx].flag = WellLowSignal;
    //   wells[bIx].keyIndex = -1;
    //   tooLowPeakSignal++;
    //   continue;
    // }
    // if (bfMask[bIx] & MaskEmpty && opts.doSigVarFilter && wells[bIx].keyIndex < 0 && wells[bIx].sd >= varThreshold)
    // {
    //   bfMask[bIx] &= ~MaskReference;
    //   wells[bIx].flag = WellSdFilter;
    //   tooVar++;
    //   continue;
    // }
    double bfVal = wells[bIx].bfMetric;
    if (bfMask[bIx] & MaskEmpty && opts.doSigVarFilter && wells[bIx].keyIndex < 0 && bfVal > bfThreshold)
      {
	bfMask[bIx] &= ~MaskReference;
	wells[bIx].flag = WellBfBufferFilter;
	tooBf++;
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
    //  if (wells[bIx].keyIndex < 0 && (bfMask[bIx] & MaskReference) && wells[bIx].peakSig >= peakSigEmptyThreshold) {
    //   bfMask[bIx] &= ~MaskReference;
    //   wells[bIx].flag = WellEmptySignal;
    //   emptyWithSignal++;
    //   continue;
    // }
    //    if (opts.doRemoveLowSignalFilter && (wells[bIx].keyIndex == 0 && (wells[bIx].peakSig < minLibPeak)))
    // if (opts.doRemoveLowSignalFilter && (wells[bIx].keyIndex == 0 && (wells[bIx].sd < varBeadThreshold)))    {
    //   wells[bIx].keyIndex = -1;
    //   wells[bIx].flag = WellLowSignal;
    //   poorLibPeakSignal++;
    //   bfMask[bIx] = MaskIgnore;
    //   continue;
    // }
    // //    if (opts.doRemoveLowSignalFilter && ( (wells[bIx].keyIndex == 1 && (wells[bIx].peakSig < minTfPeak))))
    // if (opts.doRemoveLowSignalFilter && ( (wells[bIx].keyIndex == 1 && (wells[bIx].sd < varBeadThreshold)))) {
    //   wells[bIx].keyIndex = -1;
    //   wells[bIx].flag = WellLowSignal;
    //   poorTfPeakSignal++;
    //   bfMask[bIx] = MaskIgnore;
    //   continue;
    // }
    // Call low variance non-key wells as reference
    //    double bf2IqrThreshold = bfEmptyQuantiles.GetQuantile (.75) + (2 * IQR (bfEmptyQuantiles));
    if (opts.doRecoverSdFilter && wells[bIx].keyIndex < 0 && (bfMask[bIx] & MaskBead) && 
        (wells[bIx].sd >= refMinSD && wells[bIx].sd < refMaxSD) && 
        (wells[bIx].peakSig <= peakSigThreshold) &&
        (bfVal <= bfThreshold && bfVal > bfMinThreshold)) {
      bfMask[bIx] |= MaskReference;
      wells[bIx].flag = WellRecoveredEmpty;
      sdRefCalled++;
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
  totalTimer.PrintMilliSeconds(cout, "Total Timer: After Post Filtering.");  

  cout << "Lib snr: " << keys[0].minSnr << endl;
  if (libSnrQuantiles.GetNumSeen() > 100) {
    cout << "Lib SNR: " << endl;
    for (size_t i = 0; i < 10; i++) {
      cout << i*10 << ":\t" << libSnrQuantiles.GetQuantile(i/10.0) << endl;
    }
  }
  if (tfSnrQuantiles.GetNumSeen() > 100) {
    cout << "Tf SNR: " << endl;
    for (size_t i = 0; i < 10; i++) {
      cout << i*10 << ":\t" << tfSnrQuantiles.GetQuantile(i/10.0) << endl;
    }
  }
  //  cout << "SD IQR mult: " << opts.sigSdMult << endl;
  cout << "Using: " << opts.bfNeighbors << " bf neighbors (" << opts.useMeshNeighbors << ")" << endl;
  cout << badFit << " bad fit "  <<  noTauE << " no tauE. " << varMinCount << " under min sd." << endl;
  cout << "Trace sd thresh: " << traceSDThresh << ", mean thresh: " << traceMeanThresh << endl;
  cout << "Ignore: " << overMaxMad << " over max MAD. ( " << madThreshold << " )" << endl;
  cout << "Ignore: " << filteredWells << " filtered wells. " << endl;
  cout << "Ignore: " << tooRefVar << " too var compared to reference ( " << sepRefSdThresh << " )" << endl;
  cout << "Ignore: " << tooRefBuffer << " too high bf metric compared to reference ( " << sepRefBfThresh << " )" << endl;
  cout << "Ignore: " << tooVar << " too var " << endl;
  cout << "Ignore: " << tooBf << " too high bf metric." << endl;
  cout << "Ignore: " << tooTauB << " too high tauB metric." << endl;
  cout << "Marked: " << poorSignal << " wells as ignore based on poor signal." << endl;
  cout << "Marked: " << poorLibPeakSignal << " lib wells as ignore based on poor peak signal. ( " << minLibPeak << " )" << endl;
  cout << "Marked: " << poorTfPeakSignal << " tf wells as ignore based on poor peak signal. ( " << minTfPeak << " )" << endl;
  cout << "Marked: " << emptyWithSignal << " empty wells as ignore based on too much peak signal." << endl;
  cout << "Marked: " << sdRefCalled << " wells as empty based on signal sd." << endl;
  cout << "Marked: " << badSignal << " wells ignore based on mean 1mer signal." << endl;
  cout << "Marked: " << beadLow << " wells ignore based on low bead mean 1mer signal." << endl;
  cout << traceMeanMinCount << " were less than mean threshold. " << notGood << " not good." << endl;

  if (opts.outputDebug > 0) {
    OutputOutliers (opts, traceStore, zModelBulk, wells,
		    sdQuantiles.GetQuantile (.9), sdQuantiles.GetQuantile (.9), madQuantiles.GetQuantile (.9),
		    bfQuantiles.GetQuantile (.9), bfQuantiles.GetQuantile (.9), peakSigKeyQuantiles.GetQuantile (.1));

    // Write out debugging matrix
    arma::Mat<float> wellMatrix (numWells, 18);
    fill (wellMatrix.begin(), wellMatrix.end(), 0.0f);
    for (size_t i = 0; i < numWells; i++) {
      if (mask[i] & MaskExclude || !zModelBulk.HaveModel (i)) {
	continue;
      }
      int currentCol = 0;
      KeyFit &kf = wells[i];
      const KeyBulkFit *kbf = zModelBulk.GetKeyBulkFit (i);
      wellMatrix.at (i, currentCol++) = (int) kf.keyIndex;                               // 0
      wellMatrix.at (i, currentCol++) = t0[kf.wellIdx]; //traceStore.GetT0 (kf.wellIdx); // 1
      wellMatrix.at (i, currentCol++) = kf.snr;                                          // 2
      wellMatrix.at (i, currentCol++) = kf.mad;                                          // 3 
      wellMatrix.at (i, currentCol++) = kf.sd;                                           // 4
      wellMatrix.at (i, currentCol++) = kf.bfMetric;                                     // 5
      wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::A_NUC,0);     // 6
      wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::C_NUC,0);     // 7
      wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::G_NUC,0);     // 8
      wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::T_NUC,0);     // 9
      wellMatrix.at (i, currentCol++) = kf.peakSig;                                      // 10
      wellMatrix.at (i, currentCol++) = kf.flag;                                         // 11
      wellMatrix.at (i, currentCol++) = kf.goodLive;                                     // 12
      wellMatrix.at (i, currentCol++) = kf.isRef;                                        // 13
      wellMatrix.at (i, currentCol++) = kf.bufferMetric;                                 // 14
      wellMatrix.at (i, currentCol++) = kf.traceSd;                                      // 15
      wellMatrix.at (i, currentCol++) = kf.acqT0;                                      // 16
      wellMatrix.at (i, currentCol++) = mFilteredWells[i];                                      // 17
    }

    string h5Summary = "/separator/summary";
    vector<int> flows(2);
    flows[0] = opts.maxKeyFlowLength;
    flows[1] = opts.maxKeyFlowLength+1;
    H5File h5file(h5SummaryRoot);
    h5file.Open(false);
    h5file.WriteMatrix (h5Summary, wellMatrix);
    CalcRegionEmptyStat(h5file, modelMesh, traceStore, h5SummaryRoot, flows, mask);
    h5file.Close();

    if (opts.outputDebug > 1) {
      // Write out smaller sampled file
      string summaryFile = opts.outData + ".summary.txt";
      ofstream o (summaryFile.c_str());
      o << "well\tkey\tt0\tsnr\tmad\ttraceMean\ttraceSd\tsigSd\tok\tbfMetric\ttauB.A\ttauB.C\ttauB.G\ttauB.T\ttauE.A\ttauE.C\ttauE.G\ttauE.T\tmeanSig\tmeanProj\tprojMad\tprojPeak\tflag\tbuffMetric";
      o << endl;
      for (size_t i = 0; i < numWells; i+=opts.samplingStep) {
        if (mask[i] & MaskExclude) {
          continue;
        }
        KeyFit &kf = wells[i];
        const KeyBulkFit *kbf = zModelBulk.GetKeyBulkFit (i);
        if (kbf != NULL) {
          assert (kf.wellIdx == kbf->wellIdx);
          o << kf.wellIdx << "\t" << (int) kf.keyIndex << "\t" << traceStore.GetT0 (kf.wellIdx) << "\t" << kf.snr << "\t"
            <<  kf.mad << "\t" << kf.traceMean << "\t" << kf.traceSd << "\t" << kf.sd << "\t" << (int) kf.ok << "\t"
            << kf.bfMetric << "\t"
            << kbf->param.at (TraceStore<double>::A_NUC,0) << "\t"
            << kbf->param.at (TraceStore<double>::C_NUC,0) << "\t"
            << kbf->param.at (TraceStore<double>::G_NUC,0) << "\t"
            << kbf->param.at (TraceStore<double>::T_NUC,0) << "\t"
            << kbf->param.at (TraceStore<double>::A_NUC,1) << "\t"
            << kbf->param.at (TraceStore<double>::C_NUC,1) << "\t"
            << kbf->param.at (TraceStore<double>::G_NUC,1) << "\t"
            << kbf->param.at (TraceStore<double>::T_NUC,1) << "\t"
            << kf.onemerAvg << "\t" << kf.onemerProjAvg << "\t" << kf.projResid << "\t"
            << kf.peakSig << "\t" << kf.flag << "\t" << kf.bufferMetric;
          o << endl;
        }
      }
      o.close();
    }
  }

  // --- Some reporting for the log.
  int beadCount = 0, emptyCount = 0, ignoreCount = 0, libCount = 0, tfCount = 0, dudCount = 0, pinnedCount = 0, referenceCount = 0, excludedCount = 0, excludeCount = 0;
  for (size_t bIx = 0; bIx < wells.size(); bIx++) {
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
  if (opts.outputDebug) {
    string outMask = opts.outData + ".mask.bin";
    bfMask.WriteRaw (outMask.c_str());
  }
  totalTimer.PrintMilliSeconds(cout, "Total Timer: Total Time.");
  return 0;
}


void DifferentialSeparator::PredictFlow (const std::string &datFile,
					 const std::string &outPrefix,
					 int ignoreChecksumErrors,
					 DifSepOpt &opts,
					 TraceStore<double> &store,
					 ZeromerModelBulk<double> &zModelBulk)
{
  Image img;
  Traces mTrace;
  img.SetImgLoadImmediate (false);
  img.SetIgnoreChecksumErrors (ignoreChecksumErrors);
  bool loaded = img.LoadRaw (datFile.c_str());
  if (!loaded)
    {
      ION_ABORT ("Couldn't load file: " + datFile);
    }
  mTrace.Init (&img, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  mTrace.SetT0Step (opts.t0MeshStep);
  mTrace.SetMeshDist (opts.useMeshNeighbors * 2);

  img.Close();
  //   mTrace.SetReportSampling(*mReportSet, false);
  //   mTrace.SetT0(t0);
  mTrace.CalcT0 (true);
  size_t numWells = mTrace.GetNumRow() * mTrace.GetNumCol();
  for (size_t i = 0; i < numWells; i++)
    {
      mTrace.SetT0 (max (mTrace.GetT0 (i) - 3, 0.0f), i);
    }
  mTrace.T0DcOffset (0,4);
  mTrace.FillCriticalFrames();
  mTrace.CalcReference (opts.t0MeshStep,opts.t0MeshStep,mTrace.mGridMedian);
  //  ZeromerDiff<double> bg;
  int nFrames = mTrace.mFrames;
  vector<float> reference (nFrames, 0);
  std::vector<double> dist;
  std::vector<std::vector<float> *> distValues;
  Col<double> param (2);
  Col<double> zero (nFrames);
  Col<double> diff (nFrames);
  Col<double> raw (nFrames);
  Col<double> ref (nFrames);
  Col<double> signal (nFrames);
  string outFile = outPrefix + ".predict-summary.txt";
  string traceFile = outPrefix + ".predict-trace.txt";
  string zeroFile = outPrefix + ".predict-zero.txt";
  string signalFile = outPrefix + ".predict-signal.txt";
  string referenceFile = outPrefix + ".predict-reference.txt";
  ofstream out;
  out.open (outFile.c_str());
  ofstream traceOut (traceFile.c_str());
  ofstream zeroOut (zeroFile.c_str());
  ofstream signalOut (signalFile.c_str());
  ofstream refOut (referenceFile.c_str());
  for (size_t wellIdx = 0; wellIdx < numWells; wellIdx++)
    {
      if (mask[wellIdx] & MaskExclude || mask[wellIdx] & MaskPinned)
	{
	  //      out << wellIdx << "\t" << "nan" << "\t" << (int)wells[wellIdx].bestKey << "\t" << wells[wellIdx].snr << "\t" << wells[wellIdx].bfMetric << "\t" << "-1" << endl;
	  continue;
	}
      mTrace.GetTraces (wellIdx, raw.begin());
      mTrace.CalcMedianReference (wellIdx, mTrace.mGridMedian, dist, distValues, reference);
      copy (reference.begin(), reference.end(), ref.begin());
      zModelBulk.ZeromerPrediction (wellIdx, 3, store, ref,zero);
      //    bg.PredictZeromer(ref, mTime, wells[wellIdx].param, zero);
      signal = raw - zero;
      double sig = 0;
      for (size_t frameIx = 4; frameIx < 20; frameIx++)
	{
	  sig += signal.at (frameIx);
	}
      // @todo - turn off this reporting once things look ok.
      if (wellIdx % 100 == 0)
	{
	  traceOut << wellIdx;
	  zeroOut << wellIdx;
	  signalOut << wellIdx;
	  refOut << wellIdx;
	  for (int frameIx = 0; frameIx < nFrames; frameIx++)
	    {
	      traceOut << "\t" << raw.at (frameIx);
	      zeroOut << "\t" << zero.at (frameIx);
	      signalOut << "\t" << signal.at (frameIx);
	      refOut << "\t" << ref.at (frameIx);
	    }
	  traceOut << endl;
	  zeroOut << endl;
	  signalOut << endl;
	  refOut << endl;
	  out << wellIdx << "\t" << sig << "\t" << (int) wells[wellIdx].bestKey << "\t" << wells[wellIdx].snr << "\t" << wells[wellIdx].bfMetric << "\t" <<  mTrace.GetT0 (wellIdx) << endl;
	}
      wells[wellIdx].bfMetric = sig;
    }
}
