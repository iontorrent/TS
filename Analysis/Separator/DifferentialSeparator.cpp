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
#include "TraceStoreDelta.h"
#include "ReservoirSample.h"
#include "H5File.h"
#include "T0Calc.h"
#include "SynchDatSerialize.h"
#include "Stats.h"
#define DIFFSEP_ERROR 3
#define T0_LEFT_OFFSET 4
#define T0_RIGHT_OFFSET 25
#define MIN_SD 5
#define T0_OFFSET_LEFT_TIME -.132f
#define T0_OFFSET_RIGHT_TIME 1.5f
using namespace std;

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
    const BFReference &reference,
    const vector<KeyFit> &wells,
    double trim,
    MixModel &model)
{
  vector<float> metric;
  vector<int8_t> cluster;
  int numWells = (rowEnd - rowStart) * (colEnd - colStart);
  metric.reserve (numWells);
  cluster.reserve (numWells);
  for (int rowIx = rowStart; rowIx < rowEnd; rowIx++)
  {
    for (int colIx = colStart; colIx < colEnd; colIx++)
    {
      size_t idx = reference.RowColToIndex (rowIx, colIx);
      //      if (mask[idx] & MaskPinned || mask[idx] & MaskExclude || !isfinite (reference.GetBfMetricVal (idx))
      //      if (mask[idx] & MaskPinned || mask[idx] & MaskExclude || !isfinite (reference.GetBfMetricVal (idx)) || mFilteredWells[idx] != 1)
      if (mask[idx] & MaskPinned || mask[idx] & MaskExclude || !isfinite (reference.GetBfMetricVal (idx)))
      {
        continue;
      }
      if (wells.empty() || (wells[idx].ok == 1 && wells[idx].mad <= maxMad))
      {
        //    metric.push_back(reference.GetBfMetricVal(idx));
        if (wells.empty())
        {
          metric.push_back (reference.GetBfMetricVal (idx));
        }
        else
        {
          metric.push_back (wells[idx].bfMetric);
        }
        if (wells.empty())
        {
          cluster.push_back (-1);
        }
        else if (wells[idx].keyIndex >= 0 && wells[idx].snr > minBeadSnr)
        {
          cluster.push_back (1);
        }
        // else if (reference.IsReference(idx)) {
        //   cluster.push_back(0);
        // }
        else
        {
          cluster.push_back (-1);
        }
      }
    }
  }
  if (metric.size() < minGoodWells)
  {
    return;
  }
  DualGaussMixModel dgm (2000);
  dgm.SetTrim (trim);
  model = dgm.FitDualGaussMixModel (&metric[0], &cluster[0], metric.size());
}

void DifferentialSeparator::MakeStadardKeys (vector<KeySeq> &keys)
{
  //                                  0 1 2 3 4 5 6 7
  vector<int> libKey = char2Vec<int> ("1 0 1 0 0 1 0 1", ' ');
  vector<int> tfKey = char2Vec<int> ("0 1 0 0 1 0 1 1", ' ');
  KeySeq lKey, tKey;
  lKey.name = "lib";
  lKey.flows = libKey;
  lKey.zeroFlows.set_size (4);
  lKey.zeroFlows << 1 << 3  << 4 << 6;
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
    int count = 0;
    for (int flowIx = 0; flowIx < seqList[i].numKeyFlows; flowIx++)
    {
      if (seqList[i].Ionogram[flowIx] == 0)
      {
        count++;
      }
    }
    k.zeroFlows.set_size (count);
    count = 0;
    for (int flowIx = 0; flowIx < seqList[i].numKeyFlows; flowIx++)
    {
      k.flows[flowIx] = seqList[i].Ionogram[flowIx];
      if (seqList[i].Ionogram[flowIx] == 0)
      {
        k.zeroFlows.at (count++) = flowIx;
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
  modelMesh.Init (mask.H(), mask.W(), opts.bfMeshStep, opts.bfMeshStep);
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
  modelOut << "bin\tbinRow\tbinCol\tcount\tmix\tmu1\tvar1\tmu2\tvar2" << endl;
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
    << model.mu2 << "\t" << model.var2 << endl;
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
    //    modelMesh.GetClosestNeighbors(row, col, 0, dist, bfModels);
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
  }
  if (!bfBgDat.empty())
  {
    bfBkgFile = bfBgDat;
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
      bfFile = preBeadFind;
      bfFile2 = preBeadFind2;
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
  cout << "BkgFile: " << bfBkgFile << endl;
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
  bool loaded =   img.LoadRaw(bfFile.c_str());
  if (!loaded) { ION_ABORT ("Couldn't load file: " + bfFile); }
  const RawImage *raw = img.GetImage(); 
  mBFTimePoints.resize(raw->frames);
  copy(raw->timestamps, raw->timestamps+raw->frames, &mBFTimePoints[0]);
  T0Calc t0;
  t0.SetWindowSize(3);
  t0.SetMinFirstHingeSlope(-1.0/raw->baseFrameRate);
  t0.SetMaxFirstHingeSlope(300.0/raw->baseFrameRate);
  t0.SetMinSecondHingeSlope(-20000.0/raw->baseFrameRate);
  t0.SetMaxSecondHingeSlope(-10.0/raw->baseFrameRate);
  short *data = raw->image;
  int frames = raw->frames;
  t0.SetMask(&mask);
  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStep, opts.t0MeshStep, opts.nCores);
  //  t0.Init(raw->rows, raw->cols, frames, opts.t0MeshStep, opts.t0MeshStep, 1);
  int *timestamps = raw->timestamps;
  t0.SetTimeStamps(timestamps, frames);
  t0.CalcAllSumTrace(data);
  t0.CalcT0FromSum();
  t0.CalcIndividualT0(t0vec, opts.useMeshNeighbors);
  for (size_t i = 0; i < t0vec.size(); i++) {
    if (t0vec[i] > 0) {
      float tmpT0Start = floor(t0vec[i]/raw->baseFrameRate);
      float tmpT0End = ceil(t0vec[i]/raw->baseFrameRate);
      float t0Frames = (t0vec[i]/raw->baseFrameRate - tmpT0Start)/(tmpT0End - tmpT0Start) + tmpT0Start-1; 
      t0Frames = max(0.0f,t0Frames);
      t0Frames = min(t0Frames, frames - 1.0f);
      t0vec[i] = t0Frames;
    }
  }
  if (opts.outputDebug) {
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
      if (!mask.Match(c, r, MaskPinned) && !mask.Match(c,r,MaskExclude) && filter[idx] == 1) {
        okCount++;
      }
      else {
        filter[idx] = 2;
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
      if (!mask.Match(col, row, MaskPinned) && !mask.Match(col,row,MaskExclude) && filter[idx] == 1) {
        mapping[count] = idx;
        store.GetTrace(idx, flow0, O.begin_col(count));
        store.GetTrace(idx, flow1, Z.begin_col(count));
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
  double maxO = Osamp.GetQuantile(.75) + iqrMult * Osamp.GetIQR();
  double minO = Osamp.GetQuantile(.25) - iqrMult * Osamp.GetIQR();


  double maxZ = Zsamp.GetQuantile(.75) + iqrMult * Zsamp.GetIQR();
  double minZ = Zsamp.GetQuantile(.25) - iqrMult * Zsamp.GetIQR();
  //  cout << "MinO is: " << minO << " for: " << Osamp.GetQuantile(.25) << "+/-" << Osamp.GetIQR() << endl;
  // cout << "MinZ is: " << minZ << " for: " << Zsamp.GetQuantile(.25) << "+/-" << Zsamp.GetIQR() << endl;
  // int tooMinMad = 0, tooMaxMad = 0, tooMinO = 0, tooMaxO = 0, tooMinZ = 0, tooMaxZ = 0;
  // Mark wells with good residuals as ok.
  for (size_t i = 0; i < O.n_cols; i++) {
  //   if (!(d[i] > minMad && d[i+O.n_cols] > minMad)) {
  //     tooMinMad++;
  //   }
  //   if (!(d[i] < maxMad && d[i+O.n_cols] < maxMad)) {
  //     tooMaxMad++;
  //   }
  //   if (!(O.at(O.n_rows - 1, i) > minO)) {
  //     tooMinO++;
  //   }
  //   if (!(O.at(O.n_rows - 1, i) < maxO)) {
  //     tooMaxO++;
  //   }
  //   if (!(Z.at(Z.n_rows - 1, i) > minZ)) {
  //     tooMinZ++;
  //   }
  //   if (!(Z.at(Z.n_rows - 1, i) < maxZ)) {
  //     tooMaxZ++;
  //   }
    float dx = d[i];
    float dox = d[i+O.n_cols];
    float o = O.at(O.n_rows - 1, i);
    float z = Z.at(Z.n_rows - 1, i);
    if (dx < maxMad && dx > minMad && dox < maxMad && dox > minMad &&
        o > minO && o < maxO &&
        z > minZ && z < maxZ) {
       SampleStats<double> sam;
       Row<double> vec = abs(P.row(i) - P.row(i + O.n_cols));
       sam.AddValues(vec.memptr(), vec.n_elem);
       mad[mapping[i]] = fabs(sam.GetMean());
    }
    else {
      filter[mapping[i]] = 3;
    }
  }
  //  cout << "Filtered: " << tooMinMad << "," << tooMaxMad << "," << tooMinO << "," << tooMaxO << "," << tooMinZ << "," << tooMaxZ << endl;
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
    if (filter[mapping[i]]) {
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
      if (filter[mapping[i]] == 1) {
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
    if (filter[mapping[combinedRanks[i].second]] == 1) {
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

void DifferentialSeparator::PickReference(TraceStore<double> &store,
                                          BFReference &reference, 
                                          int rowStep, int colStep,
                                          bool useKeySignal,
                                          float iqrMult,
                                          int numBasis,
                                          Mask &mask,
                                          int minWells,
                                          vector<char> &filter,
                                          vector<char> &refWells) {
  int count = 0; 
  for (int i = 0; i < reference.GetNumWells(); i++) {
    filter[i] = reference.GetType(i) != BFReference::Exclude && reference.GetType(i) != BFReference::Filtered;
    if (reference.GetType(i) == BFReference::Filtered) {
      count++;
    }
  }
  cout << "Beadfind filtered: " << count << " wells of: " << filter.size() << " (" << float(count)/filter.size() << ")" << endl;

  vector<vector<float> > mads;
  if (useKeySignal) {
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
      }
      int goodCount = 0, badCount2 = 0, badCount3 = 0;
      for (size_t fIx = 0; fIx < filter.size(); fIx++) {
        switch (filter[fIx]) 
          {
          case (1) : {
            goodCount++;
            break;
          }
          case (2) : {
            badCount2++;
            break;
          }
          case (3) : { 
            badCount3++;
            break;
          }
          default : {
            ION_ABORT("Bad value.");
          }
          }
      }
      cout << "For Key: " << i << " good wells: " << goodCount << " bad orig: " << badCount2 << " new bad: " << badCount3 << endl;
    }
  }
  else {
    cout << "Not using key signal reference." << endl;
  }

  PickCombinedRank(reference, mads, rowStep, colStep, minWells, filter, refWells);
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

void DifferentialSeparator::LoadKeySDats(TraceStore<double> &traceStore, BFReference &reference, DifSepOpt &opts) {

  // size_t numWells = mask.H() * mask.W();
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = opts.sdatSuffix;
  size_t numWells = mask.H() * mask.W();
  SetReportSet (mask.H(), mask.W(), opts.wellsReportFile, opts.reportStepSize);
  if (keys.empty())
  {
    DifferentialSeparator::MakeStadardKeys (keys);
  }
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  string s = opts.outData + ".trace.txt";
  ofstream out (s.c_str());
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
  int sdFilt = 0;
  int numFrames = 0;
  mFilteredWells.resize(numWells, 1);
  size_t loadMinFlows = max (9, opts.maxKeyFlowLength+2);
  {
    for (size_t i = 0; i < loadMinFlows; i++) {
      traceStore.SetFlowIndex (i, i);
      TraceChunkSerializer readSerializer;
      SynchDat sdat;
      char buff[resultsSuffix.size() + resultsRoot.size() + 21];
      const char *p = resultsRoot.c_str();
      const char *s = resultsSuffix.c_str();
      snprintf (buff, sizeof (buff), "%s%.4d.%s", p, (int) i, s);
      readSerializer.Read(buff, sdat);
      sdat.AdjustForDrift();
      /* Check for pinned wells. */
      const short pinHigh = GetPinHigh();
      const short pinLow = GetPinLow();
      size_t row, col;
      for (size_t wIx = 0; wIx < numWells; wIx++) {
        traceStore.WellRowCol(wIx, row, col);
        for (size_t f = 0; f < sdat.NumFrames(wIx); f++) {
          if (sdat.At(row,col,f) <= pinLow || sdat.At(row,col,f) >= pinHigh) {
            mask[wIx] = MaskPinned;
          }
        }
      }
      /* If first sdat, figure out where data is centered. */
      if (0 == i) {
        numFrames = 0;
        for (float f = T0_OFFSET_LEFT_TIME; f < T0_OFFSET_RIGHT_TIME; f+= sdat.FrameRate() / 1000.0f) {
          numFrames++;
        }
        traceStore.SetSize(numFrames);
        t0.resize(numWells);
        for (size_t wIx = 0; wIx < numWells; wIx++) {
          traceStore.WellRowCol(wIx, row, col);
          t0[wIx] = sdat.GetT0(row, col);
        }
        traceStore.SetT0(t0);
      }
      SampleStats<float> sd;
      traceBuffer.resize(traceStore.GetNumFrames());
      // @todo set time for each well...
      for (size_t wIx = 0; wIx < numWells; wIx++) {
        if (traceStore.HaveWell (wIx) && !(mask[wIx] & MaskIgnore) && !(mask[wIx] & MaskPinned)) {
          traceStore.WellRowCol(wIx, row, col);
          sd.Clear();
          int index = 0;
          fill(traceBuffer.begin(), traceBuffer.end(), 0.0);
          float dc = sdat.DcOffset(row, col);
          for (float f = T0_OFFSET_LEFT_TIME; f < T0_OFFSET_RIGHT_TIME; f+= (sdat.FrameRate() / 1000.0f)) {
            assert(index < (int)traceBuffer.n_rows);
            double val = sdat.GetValue(f, row, col) - dc;
            traceBuffer[index] = val;
            sd.AddValue(traceBuffer[index]);
            index++;
          }
          traceStore.SetTrace(wIx, i, traceBuffer.begin(), traceBuffer.begin() + traceStore.GetNumFrames());
          if (sd.GetSD() < MIN_SD) {
            sdFilt++;
            mFilteredWells[wIx] = 4;
          }
        }
      }
    }
  }
  cout << "Filtered: " << sdFilt << " based on sd." << endl;
  traceStore.SetFirst();
  Col<double> tmpData;
  for (size_t fIx = 0; fIx < loadMinFlows; fIx++) {
    for (size_t wIx = 0; wIx < traceStore.GetNumWells(); wIx++) {
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
  // ofstream storeOut("trace-store.txt");
  // traceStore.Dump(storeOut);
  // storeOut.close();
  cout << "Done loading all traces - took: " << allLoaded.elapsed() <<  " seconds." << endl;

  mRefWells.resize(numWells, 0);
  PickReference(traceStore, reference, opts.bfMeshStep, opts.bfMeshStep, opts.useSignalReference,
                opts.iqrMult, 7, mask, (int)opts.bfMeshStep *opts.bfMeshStep * opts.percentReference,
                mFilteredWells, mRefWells);
  int filtered = 0, refChosen = 0, possible = 0;
  for (size_t i = 0; i < mFilteredWells.size(); i++) {
    if (mask[i] == MaskIgnore) {
      mFilteredWells[i] = 4;
    }
    if (!(mask[i] & MaskPinned || mask[i] & MaskExclude)) {
      possible++;
      if (mFilteredWells[i] != 1) {
        filtered++;
      }
      if (mRefWells[i] == 1) {
        refChosen++;
      }
    }
  }
  cout << filtered << " wells filtered of: " << possible << " (" <<  (float)filtered/possible << ") " << refChosen << " reference wells." << endl;
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

void DifferentialSeparator::LoadKeyDats(TraceStore<double> &traceStore, BFReference &reference, DifSepOpt &opts) {

  // size_t numWells = mask.H() * mask.W();
  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = ".dat";
  size_t numWells = mask.H() * mask.W();
  SetReportSet (mask.H(), mask.W(), opts.wellsReportFile, opts.reportStepSize);
  if (keys.empty())
  {
    DifferentialSeparator::MakeStadardKeys (keys);
  }
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  string s = opts.outData + ".trace.txt";
  ofstream out (s.c_str());
  vector<int> rowStarts;
  vector<int> colStarts;
  size_t nRow = traceStore.GetNumRows();
  size_t nCol = traceStore.GetNumCols();
  double percents[3] = {.2, .5, .8};
  int span = 7;
  Timer first4;
  for (size_t i = 0; i < ArraySize (percents); i++)
  {
    rowStarts.push_back (percents[i] * nRow);
    colStarts.push_back (percents[i] * nCol);
  }
  vector<float> t;
  char d = '\t';
  Timer allLoaded;
  //size_t loadMinFlows = min (9, opts.maxKeyFlowLength+2);
  size_t loadMinFlows = max (9, opts.maxKeyFlowLength+2);
  vector<SampleStats<float> > t0Stats (mask.W() * mask.H());
  string refFile = opts.outData + ".reference_t0.txt";
  std::ofstream diffOut;
  std::vector<float> t02;
  CalcBfT0(opts, t0,  "beadfind_pre_0003.dat");
  CalcBfT0(opts, t02, "beadfind_pre_0001.dat");
  for (size_t i = 0; i < t0.size(); i++) {
    if (t0[i] > 0 && t02[i] > 0) {
      t0[i] = (t0[i] + t02[i])/2.0;
    }
    else {
      t0[i] = max(t0[i], t02[i]);
    }
  }
  CheckT0VFC(t0, mBFTimePoints);
  for (size_t i = 0; i < t0.size(); i++) {
    t0[i] = max(-1.0f,t0[i] - 4.0f);
  }

  // Let's keep the traces in asep specific scope
  {
    std::ofstream referenceOut;
    vector<Traces> traces;
    traces.resize (loadMinFlows);
    // Loop through the first N flows and use them to calculate average t0
    for (size_t i = 0; i < loadMinFlows; i++)
    {
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
      img.FilterForPinned(&mask, MaskEmpty, false);
      ReportPinned(mask, ToStr("After Flow: ") + ToStr(i));
      traces[i].Init(&img, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME, LASTDCFRAME);  //frames 0-75, dc offset using 3-12

      if (i == 0 && opts.outputDebug)
      {

        referenceOut.open (refFile.c_str());
        std::ostream &o = referenceOut;
        o << "flow" << "\t" << "rowStart" << "\t" << "rowEnd" << "\t" << "colStart" << "\t" << "colEnd" << "\t"
        << "ok" << "\t" << "t0" << "\t" << "numWells";
        for (int fIx = 0; fIx < traces[i].GetNumFrames(); fIx++)
        {
          o << "\t" << fIx;
        }
        o << endl;
        traces[i].SetReferenceOut (&referenceOut);
        DumpDiffStats (traces[i], diffOut);
      }
      if (opts.filterLagOneSD)
      {
        PinHighLagOneSd (traces[i], 1.5);
      }
      traces[i].SetFlow (i);
      traces[i].SetT0Step (opts.t0MeshStep);
      traces[i].SetMeshDist (opts.useMeshNeighbors);
      for (size_t rowIx = 0; rowIx < nRow; rowIx++)
      {
        for (size_t colIx = 0; colIx < nCol; colIx++)
        {
          if (InSpan (rowIx, colIx, rowStarts, colStarts, span))
          {
            out << rowIx << d << colIx << d << i;
            traces[i].GetTraces (traces[i].RowColToIndex (rowIx, colIx), t);
            for (size_t ii = 0; ii < t.size(); ii++)
            {
              out << d << t[ii];
            }
            out << endl;
          }
        }
      }
      img.Close();
      traces[i].SetReportSampling (reportSet, false);
      /* // @todoo - are we using a common t0? */
      traces[i].CalcT0 (true);
      img.Close();
      for (size_t ii = 0; ii < t0Stats.size(); ii++)
      {
        if (traces[i].IsGood (ii) && (i == 0 || i == 2 || i == 5))
        {
          t0Stats[ii].AddValue (traces[i].GetT0 (ii));
        }
      }
    }

    Col<double> traceBuffer(traces[0].GetNumFrames());

    // Average in nuc t0
    for (size_t i = 0; i < t0.size(); i++) {
      float nT0 = t0Stats[i].GetMean() - 3;
      float nucWeight = nT0 > 5 ? .01 : 0;
      float bfWeight = t0[i] > 5 ? .99 : 0;
      if (bfWeight + nucWeight > 0) {
        t0[i] = (t0[i] * bfWeight + nT0 * nucWeight) / (bfWeight + nucWeight);
      }
      else {
        t0[i] = 0;
      }
    }

    size_t t0Ok = 0, notPinned = 0, allOk = 0;
    for (size_t i = 0; i < t0.size(); i++)
    {
      if (t0[i] >= 1.0)
      {
        t0Ok++;
      }
      if (! (mask[i] & MaskPinned))
      {
        notPinned++;
      }
      if (t0[i] >= 1.0 && ! (mask[i] & MaskPinned))
      {
        allOk++;
      }
    }
    cout << "t0Ok: " << t0Ok << " not pinned: " << notPinned << " ok: " << allOk << endl;
    traceStore.SetT0 (t0);
    traceStore.SetMeshDist (opts.useMeshNeighbors);
    // Now that we have t0 for multiple flows, set it and fill in the critical frames
    for (size_t iFlow = 0; iFlow < loadMinFlows; iFlow++)
    {
      traceStore.SetFlowIndex (iFlow, iFlow);
      traces[iFlow].SetT0 (t0);
      traces[iFlow].T0DcOffset (0,4);
      traces[iFlow].FillCriticalFrames();
    }

    // Load into the data store
    for (size_t wIx = 0; wIx < numWells; wIx++)
    {
      if (traceStore.HaveWell (wIx))
      {
        for (size_t flowIx = 0; flowIx < traces.size(); flowIx++)
        {
          traceBuffer.set_size (traces[flowIx].GetNumFrames());
          traces[flowIx].GetTraces (wIx, traceBuffer.begin());
          traceStore.SetTrace (wIx, flowIx, traceBuffer.begin(), traceBuffer.begin() +traceStore.GetNumFrames());
        }
      }
    }
  }  // Get the traces memory back
  cout << "Done loading first 4 traces - took: " << first4.elapsed() <<  " seconds." << endl;
  cout << "Loading up to flow: " << opts.maxKeyFlowLength + 1 << endl;
  for (size_t i = loadMinFlows; i < (size_t) (opts.maxKeyFlowLength + 1); i++)
  {
    cout << "Loading flow: " << i << endl;
    traceStore.SetFlowIndex (i, i);
    char buff[resultsRoot.size() + resultsSuffix.size() + 20];
    const char *p = resultsRoot.c_str();
    const char *s = resultsSuffix.c_str();
    Traces trace;
    snprintf (buff, sizeof (buff), "%s%.4d%s", p, (int) i, s);
    Image img;
    img.SetImgLoadImmediate (false);
    img.SetIgnoreChecksumErrors (opts.ignoreChecksumErrors);
    bool loaded = img.LoadRaw (buff);
    if (!loaded)
    {
      ION_ABORT ("Couldn't load file: " + ToStr (buff));
    }
    trace.Init (&img, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME, LASTDCFRAME); //frames 0-75, dc offset using 3-12
    for (size_t rowIx = 0; rowIx < nRow; rowIx++)
    {
      for (size_t colIx = 0; colIx < nCol; colIx++)
      {
        if (InSpan (rowIx, colIx, rowStarts, colStarts, span))
        {
          out << rowIx << d << colIx << d << i;
          trace.GetTraces (trace.RowColToIndex (rowIx, colIx), t);
          for (size_t ii = 0; ii < t.size(); ii++)
          {
            out << d << t[ii];
          }
          out << endl;
        }
      }
    }
    img.Close();
    trace.SetReportSampling (reportSet, false);
    trace.SetT0 (t0);
    trace.T0DcOffset (0,4);
    trace.FillCriticalFrames();
    Col<double> traceBuffer (trace.GetNumFrames());
    for (size_t wIx = 0; wIx < numWells; wIx++)
    {
      if (traceStore.HaveWell (wIx))
      {
        trace.GetTraces (wIx, traceBuffer.begin());
        traceStore.SetTrace (wIx, i, traceBuffer.begin(), traceBuffer.begin() +traceStore.GetNumFrames());
      }
    }
  }
  out.close();
  cout << "Done loading all traces - took: " << allLoaded.elapsed() <<  " seconds." << endl;
  mFilteredWells.resize(numWells, 1);
  mRefWells.resize(numWells, 0);
  PickReference(traceStore, reference, opts.bfMeshStep, opts.bfMeshStep, opts.useSignalReference,
                opts.iqrMult, 7, mask, (int)opts.bfMeshStep *opts.bfMeshStep * opts.percentReference,
                mFilteredWells, mRefWells);
  int filtered = 0, refChosen = 0, possible = 0;
  for (size_t i = 0; i < mFilteredWells.size(); i++) {
    if (!(mask[i] & MaskPinned || mask[i] & MaskExclude)) {
      possible++;
      if (mFilteredWells[i] != 1) {
        filtered++;
      }
      if (mRefWells[i] == 1) {
        refChosen++;
      }
    }
  }
  cout << filtered << " wells filtered of: " << possible << " (" <<  (float)filtered/possible << ") " << refChosen << " reference wells." << endl;
  for (size_t i = 0; i < mRefWells.size(); i++) {
    traceStore.SetReference(i, mRefWells[i] == 1);
  }
  for (size_t flowIx = 0; flowIx < (size_t) opts.maxKeyFlowLength+1; flowIx++)
  {
    traceStore.PrepareReference (flowIx);
  }
  // ofstream storeOut("trace-store.txt");
  // traceStore.Dump(storeOut);
  // storeOut.close();
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
        bgFrameIqr(qIdx, lcol) = refFrameIqr(qIdx, lcol) = refFrameStats[0].GetCount(); lcol++;
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


int DifferentialSeparator::Run(DifSepOpt opts) {

  // Fill in references
  //  opts.reportStepSize = 1;
  KClass kc;
  BFReference reference;
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
  int exWells = 0;
  size_t numWells = mask.H() * mask.W();
  for (size_t i = 0; i < numWells; i++) {
    if (mask[i] & MaskExclude) {
      exWells++;
    }
  }
  cout << "Number excluded starting is: " << exWells;
  ReportPinned(mask, "Start");
  if (opts.filterLagOneSD) {
    CheckFirstAcqLagOne(opts);
  }
  ReportPinned(mask, "After LagOne");
  ReportSet reportSet(mask.H(), mask.W());
  if (opts.wellsReportFile.empty()) {
    reportSet.SetStepSize(opts.reportStepSize);
  }
  else
  {
    reportSet.ReadSetFromFile (opts.wellsReportFile, 0);
  }


  string resultsRoot = opts.resultsDir + "/acq_";
  string resultsSuffix = ".dat";

  time_t start = time (NULL);
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
  vector<double> bfMetric;
  if (opts.signalBased)
  {
    reference.CalcSignalReference
    (bfImgFile, bfBkgImgFile, mask, 9);
  }
  else
  {
    string debugSample = opts.outData + ".bftraces.txt";
    reference.SetDebugFile (debugSample);
    if (bfImgFile2.empty())
    {
      reference.CalcReference (bfImgFile, mask);
    }
    else
    {
      cout << "Using average beadfind." << endl;
      reference.CalcDualReference (bfImgFile, bfImgFile2, mask);
    }
  }
  ReportPinned(mask, "After Reference");

  time_t end = time (NULL);
  cout << "Reference Step took: " << (end - start) << " seconds." << endl;
  if (opts.justBeadfind)
  {
    DoJustBeadfind (opts, reference);
    return 0;
  }
  if (keys.empty())
  {
    MakeStadardKeys (keys);
  }
  for (size_t kIx = 0; kIx < keys.size(); kIx++)
  {
    opts.maxKeyFlowLength = max ( (unsigned int) opts.maxKeyFlowLength, keys[kIx].usableKeyFlows);
  }
  // For each key flow load up the dat file
  cout << "Loading: " << opts.maxKeyFlowLength << " traces...";
  cout.flush();
  double mil = 1000000.0;
  struct timeval st, et;
  gettimeofday (&st, NULL);

  TraceStoreDelta<double> traceStore (mask, 31, opts.flowOrder.c_str(), opts.maxKeyFlowLength+2, opts.maxKeyFlowLength+2,
                                      opts.regionYSize, opts.regionXSize);
  traceStore.SetMinRefProbes (20);
  std::string firstFile = opts.resultsDir + "/acq_0000.dat";
  if (H5File::IsH5File(firstFile.c_str())) {
    opts.doSdat = true;
    opts.sdatSuffix = "dat";
  }
  if (opts.doSdat) {
    LoadKeySDats (traceStore, reference, opts);
  } 
  else {
    LoadKeyDats (traceStore, reference, opts);
  }
  zModelBulk.SetFiltered(mFilteredWells);
  gettimeofday (&et, NULL);
  cout << "Done." << endl;
  cout << "Loading traces took: " << ( (et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec)) /mil << " seconds." << endl;
  start = time (NULL);

  mTime.set_size (traceStore.GetNumFrames());
  for (size_t i = 0; i < mTime.n_rows; i++)
  {
    mTime[i] = i;
  }

  ZeromerDiff<double> bg;
  wells.resize (numWells);
  vector<KeyFit> wellsInitial (numWells);
  vector<KeyReporter<double> *> reporters;
  //  reporters.push_back(&reportInitial);
  GridMesh<SampleQuantiles<double> > emptyEstimates;
  emptyEstimates.Init (mask.H(), mask.W(), opts.tauEEstimateStep, opts.tauEEstimateStep);
  for (size_t i = 0; i < keys.size(); i++)
  {
    cout << "key: " << i << " min snr is: " << keys[i].minSnr << endl;
  }
  cout << "Doing initial estimates...";
  cout.flush();
  gettimeofday (&st, NULL);
  std::vector<double> dist;
  std::vector<std::vector<float> *> values;
  IncorpReporter<double> incorp (&keys, opts.minTauESnr);
  reporters.push_back (&incorp);

  vector<KeyClassifyJob> initialJobs (emptyEstimates.GetNumBin());
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++)
  {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    //    CheckFirst();
    initialJobs[binIx].Init (rowStart,rowEnd,colStart,colEnd,
                             opts.minSnr, &mask, &wellsInitial, &keys,
                             &mTime, &reporters, &traceStore, opts.maxKeyFlowLength);
    jQueue.AddJob (initialJobs[binIx]);
  }
  jQueue.WaitUntilDone();

  gettimeofday (&et, NULL);
  cout << "Done." << endl;
  cout << "Initial Classification took: " << ( (et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec)) /mil << " seconds." << endl;

  for (size_t rIx = 0; rIx < reporters.size(); rIx++)
  {
    reporters[rIx]->Finish();
  }
  zModelBulk.Init (opts.nCores, emptyEstimates.GetNumBin());
  zModelBulk.SetRegionSize (opts.regionYSize, opts.regionXSize);
 keyAssignments.resize (wellsInitial.size());
  int keyCounts[3] = {0,0,0};
  SampleQuantiles<float> peakSigQuantiles (10000);
  string mykeyFile=opts.analysisDir+"/keys.assignments.txt";
  ofstream keyOut(mykeyFile.c_str());
  for (size_t wIx = 0; wIx < keyAssignments.size(); wIx++)
  {
    keyAssignments[wIx] = wellsInitial[wIx].keyIndex;
    keyOut << wIx << "\t" << wellsInitial[wIx].keyIndex;
    keyCounts[keyAssignments[wIx]+1]++;
    
  }
  keyOut.close();
  cout << "Key Counts: " << keyCounts[0] << " " << keyCounts[1] << "  " << keyCounts[2] << endl;

  Timer timer;
  zModelBulk.SetTime (mTime);
  Timer zeromerTimer;
  cout << "Starting fitting zeromers." << endl;
  zModelBulk.FitWellZeromers (traceStore,
                              keyAssignments,
                              keys);
  // ofstream zout("zeromer-dump.txt");
  // zModelBulk.Dump(zout);
  // zout.close();
  cout << "Fitting zeromers took: " << timer.elapsed() << " seconds." << endl;
  SampleQuantiles<float> tauEQuant (10000);


  for (size_t i = 0; i < emptyEstimates.GetNumBin(); i++)
  {
    SampleQuantiles<double> &item = emptyEstimates.GetItem (i);
    //double val = -1;
    if (item.GetNumSeen() > 0)
    {
      item.GetMedian();
    }
  }

  SampleStats<float> sdStats;
  SampleQuantiles<float> traceMean (10000);
  SampleQuantiles<float> traceSd (10000);
  SampleQuantiles<float> sdQuantiles (10000);
  SampleQuantiles<float> tauBQuantiles (10000);
  SampleQuantiles<float> bfQuantiles (10000);
  SampleQuantiles<float> onemerQuantiles (10000);
  SampleQuantiles<float> madQuantiles (10000);
  SampleQuantiles<float> peakSigKeyQuantiles (10000);
  reporters.clear();

  vector<KeyClassifyTauEJob> finalJobs (emptyEstimates.GetNumBin());
  for (size_t i = 0; i < numWells; i++)
  {
    wells[i] = wellsInitial[i];
  }
  Col<double> medianIncorp = incorp.GetMeanTrace();

  medianIncorp = medianIncorp / norm (medianIncorp,2);
  medianIncorp.raw_print(cout, "median incorp");
  double basicTauEEst = 6;

  timer.restart();
  cout << "Starting estimates." << endl;
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++)
  {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    finalJobs[binIx].Init (rowStart,rowEnd,colStart,colEnd,
                           opts.minSnr, &mask, &wells, &keys,
                           &mTime, basicTauEEst, &medianIncorp,
                           &zModelBulk,
                           &reporters, &traceStore, opts.maxKeyFlowLength,
                           &emptyEstimates, tauEQuant);
    jQueue.AddJob (finalJobs[binIx]);
  }
  jQueue.WaitUntilDone();
  cout << "Final classification took: " << timer.elapsed() << " seconds" << endl;


  GridMesh<SampleQuantiles<float> > bfMesh;
  bfMesh.Init (mask.H(), mask.W(), 50, 50);
  for (size_t i = 0; i < bfMesh.GetNumBin(); i++)
  {
    SampleQuantiles<float> &x = bfMesh.GetItem (i);
    x.Init (10000);
  }
  int gotKeyCount = 0, gotNoKeyCount = 0;
  for (size_t idx = 0; idx < wells.size(); idx++)
  {
    if (wells[idx].keyIndex >= 0  && wells[idx].mad < opts.maxMad)
    {
      gotKeyCount++;
    }
    if (wellsInitial[idx].mad < opts.maxMad && wellsInitial[idx].keyIndex < 0)
    {
      gotNoKeyCount++;
    }
  }

  int notExcludePinnedWells = 0;
  for (size_t i = 0; i < numWells; i++)
  {
    if (! (mask[i] & MaskExclude || mask[i] & MaskPinned))
    {
      notExcludePinnedWells++;
    }
  }

  int minOkCount = max (10.0, opts.minRatioLiveWell * notExcludePinnedWells);
  if (gotKeyCount <= minOkCount)
  {
    ION_ABORT_CODE ("Only got: " + ToStr (gotKeyCount) + " key passing wells. Couldn't find enough (" + ToStr (minOkCount) + ") wells with key signal.", DIFFSEP_ERROR);
  }
  if (gotNoKeyCount <= minOkCount)
  {
    ION_ABORT_CODE ("Only got: " + ToStr (gotKeyCount) + " no key wells. Couldn't find enough (" + ToStr (minOkCount) + ") wells without key signal.", DIFFSEP_ERROR);
  }

  peakSigKeyQuantiles.Clear();
  for (size_t i = 0; i < numWells; i++)
  {

    if (isfinite (wells[i].mad) && wells[i].mad >= 0)
    {
      madQuantiles.AddValue (wells[i].mad);
      traceMean.AddValue (wells[i].traceMean);
      traceSd.AddValue (wells[i].traceSd);
    }
    //    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad && isfinite(wells[i].peakSig))
    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad)
    {
      sdStats.AddValue (wells[i].sd);
      sdQuantiles.AddValue (wells[i].sd);
      peakSigQuantiles.AddValue (wells[i].peakSig);

      if (isfinite (wells[i].param.at (0)) && isfinite (wells[i].param.at (1)))
      {
        tauBQuantiles.AddValue (wells[i].param.at (0) - wells[i].param.at (1));
      }
      if (isfinite (wells[i].onemerAvg))
      {
        onemerQuantiles.AddValue (wells[i].onemerAvg);
      }
    }
    if (wells[i].keyIndex >= 0)
    {
      peakSigKeyQuantiles.AddValue (wells[i].peakSig);
    }
  }

  SampleKeyReporter<double> report (opts.outData, numWells);
  report.SetReportSet (reportSet.GetReportIndexes());
  reporters.push_back (&report);
  AvgKeyReporter<double> avgReport (keys, opts.outData, opts.flowOrder, opts.analysisDir);
  KeySummaryReporter<double> keySumReport;
  keySumReport.Init (opts.flowOrder, opts.analysisDir, mask.H(), mask.W(),
                     std::min (128,mask.W()), std::min (128,mask.H()), keys);
  double minTfPeak = min ((double)opts.minTfPeakMax, peakSigQuantiles.GetQuantile (.75) + opts.tfFilterQuantile * IQR (peakSigQuantiles));
  double minLibPeak = min ((double)opts.minLibPeakMax,peakSigQuantiles.GetQuantile (.75) + opts.libFilterQuantile * IQR (peakSigQuantiles));
  //  minTfPeak = max(15.0,minTfPeak);
  cout << "Min Tf peak is: " << minTfPeak << " lib peak is: " << minLibPeak << endl;
  keySumReport.SetMinKeyThreshold (1, minTfPeak);
  keySumReport.SetMinKeyThreshold (0, minLibPeak);

  if (opts.outputDebug) {
    avgReport.SetMinKeyThreshold(1, minTfPeak);
    avgReport.SetMinKeyThreshold(0, minLibPeak);
    reporters.push_back(&avgReport);
  }
  mRegionIncorpReporter.Init(opts.outData, mask.H(), mask.W(),
			     opts.regionXSize, opts.regionYSize,
			     keys, t0);
  mRegionIncorpReporter.SetMinKeyThreshold(1, minTfPeak);
  mRegionIncorpReporter.SetMinKeyThreshold(0, minLibPeak);
  reporters.push_back(&keySumReport);
  reporters.push_back(&mRegionIncorpReporter);

  timer.restart();
  cout << "Starting final classification." << endl;
  for (size_t binIx = 0; binIx < emptyEstimates.GetNumBin(); binIx++)
  {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    emptyEstimates.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    finalJobs[binIx].Init (rowStart,rowEnd,colStart,colEnd,
                           opts.minSnr, &mask, &wells, &keys,
                           &mTime, basicTauEEst, &medianIncorp,
                           &zModelBulk,
                           &reporters, &traceStore, opts.maxKeyFlowLength,
                           &emptyEstimates, tauEQuant);
    jQueue.AddJob (finalJobs[binIx]);
  }
  jQueue.WaitUntilDone();
  cout << "Final classification took: " << timer.elapsed() << " seconds" << endl;

  gettimeofday (&et, NULL);
  cout << "Done." << endl;
  cout << "Final Classification took: " << ( (et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec)) /mil << " seconds." << endl;
  for (size_t rIx = 0; rIx < reporters.size(); rIx++)
  {
    reporters[rIx]->Finish();
  }

  double madThreshold = madQuantiles.GetQuantile (.75) + (3 * IQR (madQuantiles));
  //  double varThreshold = sdQuantiles.GetQuantile(.75) + (1.5 * IQR(sdQuantiles));
  double varThreshold = sdQuantiles.GetMedian() + (3 * IQR (sdQuantiles) /1.35);
  //double varBeadThreshold = sdQuantiles.GetMedian() + LowerQuantile(sdQuantiles);
  double meanSigThreshold = std::numeric_limits<double>::max() * -1;
  if (onemerQuantiles.GetNumSeen() > 10)
  {

    meanSigThreshold = onemerQuantiles.GetMedian() + (2.5 * IQR (onemerQuantiles) /2);
  }

  double tauBThreshold = tauBQuantiles.GetMedian() + (opts.sigSdMult * (IQR (tauBQuantiles)) /2);
  double peakSigThreshold = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10)
  {
    peakSigThreshold = peakSigQuantiles.GetMedian() + (IQR (peakSigQuantiles)) /2; //sdStats.GetMean() + (2.5*sdStats.GetSD());
  }
  double peakSigEmptyThreshold = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10)
  {
    peakSigEmptyThreshold = peakSigQuantiles.GetMedian() + (4* (IQR (peakSigQuantiles)) /2); //sdStats.GetMean() + (2.5*sdStats.GetSD());
    cout << "Peak Signal Empty Threshold: " << peakSigEmptyThreshold << endl;
    peakSigEmptyThreshold = min(peakSigEmptyThreshold, 10.0);
  }

  if (peakSigKeyQuantiles.GetNumSeen() > 10)
  {
    cout << "Key Peak distribution is: " << peakSigKeyQuantiles.GetMedian() << " +/- " <<  IQR (peakSigKeyQuantiles) /2 << " "
         << peakSigKeyQuantiles.GetQuantile (.1) << ", " << peakSigKeyQuantiles.GetQuantile (.25) << ", "
         << peakSigKeyQuantiles.GetQuantile (.5) << ", " << peakSigKeyQuantiles.GetQuantile (.75) << endl;

  }
  double keyPeakSigThresh = std::numeric_limits<double>::max() * -1;
  if (peakSigQuantiles.GetNumSeen() > 10)
  {
    cout << "Empty Peak distribution is: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR (peakSigQuantiles) /2 << endl;
    keyPeakSigThresh = peakSigQuantiles.GetQuantile (.75);
    cout << "Key peak signal theshold: " << keyPeakSigThresh
         << " (" << peakSigQuantiles.GetQuantile (.75) << ", " << IQR (peakSigQuantiles) << ")" << endl;
    if (peakSigQuantiles.GetNumSeen() > 10)
    {
      cout << "Empty Key Peak distribution is: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR (peakSigQuantiles) /2 << " "
           << peakSigQuantiles.GetQuantile (.1) << ", " << peakSigQuantiles.GetQuantile (.25) << ", "
           << peakSigQuantiles.GetQuantile (.5) << ", " << peakSigQuantiles.GetQuantile (.75) << endl;

    }
  }
  else
  {
    cout << "Saw less than 10 peakSigQuantiles." << endl;
  }

  double refMinSD = sdQuantiles.GetMedian() - (3 * (IQR(sdQuantiles))/2);
  double refMaxSD = sdQuantiles.GetQuantile(.5);// + (opts.sigSdMult * (IQR(sdQuantiles))/2);
  double varMin =LowerQuantile (sdQuantiles);
  double traceMeanThresh = traceMean.GetMedian() - 3 * (IQR (traceMean) /2);
  double traceSDThresh = traceSd.GetMedian() - 5 * (LowerQuantile (traceSd));
  if (opts.signalBased)
  {
    PredictFlow (bfImgFile, opts.outData, opts.ignoreChecksumErrors, opts, traceStore, zModelBulk);
    opts.doRecoverSdFilter = false;
  }
  else
  {
    for (size_t i = 0; i < numWells; i++)
    {
      wells[i].bfMetric = reference.GetBfMetricVal (i);
    }
  }
  if (opts.signalBased)
  {
    opts.doRecoverSdFilter = false;
  }
  for (size_t i = 0; i < numWells; i++)
  {
    wells[i].bfMetric = reference.GetBfMetricVal (i);
  }
  for (size_t i = 0; i < numWells; i++)
  {
    if (wells[i].keyIndex < 0 && wells[i].sd != 0 && isfinite (wells[i].sd) && wells[i].mad < opts.maxMad && isfinite (wells[i].bfMetric))
    {
      bfQuantiles.AddValue (wells[i].bfMetric);
      SampleQuantiles<float> &bfQ = bfMesh.GetItem (bfMesh.GetBin (wells[i].wellIdx));
      bfQ.AddValue (wells[i].bfMetric);
    }
  }

  cout << "Mad threshold is: " << madThreshold << " for: " << madQuantiles.GetMedian() << " +/- " << (madQuantiles.GetQuantile (.75) - madQuantiles.GetMedian()) << endl;
  cout << "Var threshold is: " << varThreshold << " for: " << sdQuantiles.GetMedian() << " +/- " <<  IQR (sdQuantiles) /2 << endl;

  cout << "TauB threshold is: " << tauBThreshold << " for: " << tauBQuantiles.GetMedian() << " +/- " <<  IQR (tauBQuantiles) /2 << endl;
  cout << "Var min is: " << varMin << " for: " << sdQuantiles.GetMedian() << " +/- " << (sdQuantiles.GetQuantile (.75)) << endl;
  if (onemerQuantiles.GetNumSeen() > 10)
  {
    cout << "Signal threshold is: " << meanSigThreshold << " for: " << onemerQuantiles.GetMedian() << " +/- " <<  IQR (onemerQuantiles) /2 << endl;
  }
  else
  {
    cout << "Saw less than 10 wells with onemer signal." << endl;
  }
  // if (peakSigQuantiles.GetNumSeen() > 10) {
  //   cout << "Peak threshold is: " << peakSigThreshold << " for: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR(peakSigQuantiles)/2 << endl;
  // }
  // else {
  //   cout << "Saw less than 10 wells with peakSigQuantiles." << endl;
  // }


  // cout << "Peak empty threshold is: " << peakSigEmptyThreshold << " for: " << peakSigQuantiles.GetMedian() << " +/- " <<  IQR(peakSigQuantiles)/2 << endl;

  cout << "Mean Trace threshold: " << traceMeanThresh << " median: " << traceMean.GetMedian() << " +/- " << traceMean.GetQuantile (.25) << endl;
  // Set up grid mesh for beadfinding.
  ofstream modelOut;
  if (opts.outputDebug) {
    string modelFile = opts.outData + ".mix-model.txt";
    modelOut.open(modelFile.c_str());
    modelOut << "bin\tbinRow\tbinCol\trowStart\trowEnd\tcolStart\tcolEnd\tcount\tmix\tmu1\tvar1\tmu2\tvar2" << endl;
  }
  GridMesh<MixModel> modelMesh;
  opts.bfMeshStep = min (min (mask.H(), mask.W()), opts.bfMeshStep);
  cout << "bfMeshStep is: " << opts.bfMeshStep << endl;
  modelMesh.Init (mask.H(), mask.W(), opts.bfMeshStep, opts.bfMeshStep);
  gettimeofday (&st, NULL);

  SampleStats<double> bfSnr;

  double bfMinThreshold = bfQuantiles.GetQuantile (.02);

  cout << "Bf min threshold is: " << bfMinThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " << ( (bfQuantiles.GetQuantile (.75) - bfQuantiles.GetQuantile (.25)) /2) << endl;
  opts.minBfGoodWells = min (300, (int) (opts.bfMeshStep * opts.bfMeshStep * .4));
  for (size_t binIx = 0; binIx < modelMesh.GetNumBin(); binIx++)
  {
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    modelMesh.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
    MixModel &model = modelMesh.GetItem (binIx);
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
    if (opts.outputDebug) {
      modelOut << binIx << "\t" << binRow << "\t" << binCol << "\t"
               << rowStart << "\t" << rowEnd << "\t" << colStart << "\t" << colEnd << "\t"
               << model.count << "\t" << model.mix << "\t"
               << model.mu1 << "\t" << model.var1 << "\t"
               << model.mu2 << "\t" << model.var2 << endl;
    }
  }
  cout << "BF SNR: " << bfSnr.GetMean() << " +/- " << (bfSnr.GetSD()) << endl;
  if (modelOut.is_open()) {
    modelOut.close();
  }
  // For each bead call based on nearest gridmesh neighbors
  bfMask.Init (&mask);
  vector<MixModel *> bfModels;
  int overMaxMad = 0, notGood = 0, tooVar = 0, badFit = 0, noTauE = 0, varMinCount = 0, traceMeanMinCount = 0, sdRefCalled = 0, badSignal = 0, beadLow = 0;
  int tooBf = 0;
  int tooTauB = 0;
  int poorSignal = 0;
  int poorLibPeakSignal = 0;
  int poorTfPeakSignal = 0;
  int emptyWithSignal = 0;
  SampleQuantiles<float> libSnrQuantiles(10000);
  SampleQuantiles<float> tfSnrQuantiles(10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++)
  {
    bfMask[bIx] = mask[bIx];
    if (bfMask[bIx] & MaskExclude || bfMask[bIx] & MaskPinned) {
      continue;
    }
    if (bfMask[bIx] & MaskEmpty && mFilteredWells[bIx] != 1) {
      bfMask[bIx] = MaskIgnore;
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
    if (bfMask[bIx] & MaskPinned)
    {
      wells[bIx].flag = WellPinned;
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
      wells[bIx].flag = WellMadHigh,
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

    if (wells[bIx].keyIndex < 0)
    {
      size_t row, col;
      double weight = 0;
      MixModel m;
      int good = 0;
      traceStore.WellRowCol (wells[bIx].wellIdx, row, col);
      modelMesh.GetClosestNeighbors (row, col, opts.bfNeighbors, dist, bfModels);
      //      modelMesh.GetClosestNeighbors(row, col, 0, dist, bfModels);
      for (size_t i = 0; i < bfModels.size(); i++)
      {
        if ( (size_t) bfModels[i]->count > opts.minBfGoodWells)
        {
          good++;
          float w = 1.0/ (log (dist[i] + 2.50));
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
        bfMask[bIx] = MaskIgnore;
        wells[bIx].flag = WellBfBad;
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
        double ownership = 0;
        int bCluster = DualGaussMixModel::PredictCluster (m, wells[bIx].bfMetric, opts.bfThreshold, ownership);
        if (bCluster == 2)
        {
          wells[bIx].flag = WellBead;
          bfMask[bIx] = MaskBead;
        }
        else if (bCluster == 1)
        {
          wells[bIx].flag = WellEmpty;
          bfMask[bIx] = MaskEmpty | MaskReference;
        }
      }
    }
  }
  SampleQuantiles<float> bfEmptyQuantiles (10000);
  for (size_t bIx = 0; bIx < wells.size(); bIx++)
  {
    if (bfMask[bIx] & MaskEmpty)
    {
      bfEmptyQuantiles.AddValue (wells[bIx].bfMetric);
    }
  }
  double bfThreshold = bfEmptyQuantiles.GetQuantile (.75) + (3 * IQR (bfEmptyQuantiles));
  cout << "Bf threshold is: " << bfThreshold << " for: " << bfQuantiles.GetMedian() << " +/- " <<  IQR (bfQuantiles) << endl;
  for (size_t bIx = 0; bIx < wells.size(); bIx++)
  {

    if (bfMask[bIx] & MaskEmpty && opts.doSigVarFilter && wells[bIx].keyIndex < 0 && wells[bIx].sd >= varThreshold)
    {
      bfMask[bIx] &= ~MaskReference;
      wells[bIx].flag = WellSdFilter;
      tooVar++;
      continue;
    }
    // if (wells[bIx].keyIndex >= 0 && wells[i].peakSig < peakSigThreshold) {
    //   bfMask[bIx] = MaskDud;
    //   wells[bIx].flag = WellLowSignal;
    //   wells[bIx].keyIndex = -1;
    //   tooLowPeakSignal++;
    //   continue;
    // }
    if (bfMask[bIx] & MaskEmpty && opts.doSigVarFilter && wells[bIx].keyIndex < 0 && wells[bIx].sd >= varThreshold)
    {
      bfMask[bIx] &= ~MaskReference;
      wells[bIx].flag = WellSdFilter;
      tooVar++;
      continue;
    }
    double bfVal = wells[bIx].bfMetric;
    if (bfMask[bIx] & MaskEmpty && opts.doSigVarFilter && wells[bIx].keyIndex < 0 && bfVal > bfThreshold)
    {
      bfMask[bIx] &= ~MaskReference;
      wells[bIx].flag = WellBfBufferFilter;
      tooBf++;
      continue;
    }
    if (wells[bIx].keyIndex < 0 && (bfMask[bIx] & MaskEmpty) && wells[bIx].peakSig >= peakSigEmptyThreshold) {
      bfMask[bIx] &= ~MaskReference;
      wells[bIx].flag = WellEmptySignal;
      emptyWithSignal++;
      continue;
    }
    if (opts.doRemoveLowSignalFilter && (wells[bIx].keyIndex == 0 && (wells[bIx].peakSig < minLibPeak)))
    {
      wells[bIx].keyIndex = -1;
      wells[bIx].flag = WellLowSignal;
      poorLibPeakSignal++;
      bfMask[bIx] = MaskIgnore;
      continue;
    }
    if (opts.doRemoveLowSignalFilter && ( (wells[bIx].keyIndex == 1 && (wells[bIx].peakSig < minTfPeak))))
    {
      wells[bIx].keyIndex = -1;
      wells[bIx].flag = WellLowSignal;
      poorTfPeakSignal++;
      bfMask[bIx] = MaskIgnore;
      continue;
    }
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
  

  gettimeofday (&et, NULL);
  cout << "Min snr: " << opts.minSnr << endl;
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
  cout << "SD IQR mult: " << opts.sigSdMult << endl;
  cout << badFit << " bad fit "  <<  noTauE << " no tauE. " << varMinCount << " under min sd." << endl;
  cout << "Trace sd thresh: " << traceSDThresh << ", mean thresh: " <<traceMeanThresh << endl;
  cout << "Ignore: " << overMaxMad << " over max MAD. " << endl;
  cout << "Ignore: " << tooVar << " too var " << endl;
  cout << "Ignore: " << tooBf << " too high bf metric." << endl;
  cout << "Ignore: " << tooTauB << " too high tauB metric." << endl;
  cout << "Marked: " << poorSignal << " wells as ignore based on poor signal." << endl;
  cout << "Marked: " << poorLibPeakSignal << " lib wells as ignore based on poor peak signal." << endl;
  cout << "Marked: " << poorTfPeakSignal << " tf wells as ignore based on poor peak signal." << endl;
  cout << "Marked: " << emptyWithSignal << " empty wells as ignore based on too much peak signal." << endl;
  cout << "Marked: " << sdRefCalled << " wells as empty based on signal sd." << endl;
  cout << "Marked: " << badSignal << " wells ignore based on mean 1mer signal." << endl;
  cout << "Marked: " << beadLow << " wells ignore based on low bead mean 1mer signal." << endl;
  cout << traceMeanMinCount << " were less than mean threshold. " << notGood << " not good." << endl;
  cout << "Clustering and prediction took: " << ( (et.tv_sec*mil+et.tv_usec) - (st.tv_sec * mil + st.tv_usec)) /mil << " seconds." << endl;

  if (opts.outputDebug) {
    OutputOutliers (opts, traceStore, zModelBulk, wells,
                  sdQuantiles.GetQuantile (.9), sdQuantiles.GetQuantile (.9), madQuantiles.GetQuantile (.9),
                  bfQuantiles.GetQuantile (.9), bfQuantiles.GetQuantile (.9), peakSigKeyQuantiles.GetQuantile (.1));

    //  CalcDensityStats(opts.outData, bfMask, wells);
    arma::Mat<float> wellMatrix (numWells, 12);
    fill (wellMatrix.begin(), wellMatrix.end(), 0.0f);
    for (size_t i = 0; i < numWells; i++)
      {
        if (mask[i] & MaskExclude || !zModelBulk.HaveModel (i))
          {
            continue;
          }
        int currentCol = 0;
        KeyFit &kf = wells[i];
        const KeyBulkFit *kbf = zModelBulk.GetKeyBulkFit (i);
        wellMatrix.at (i, currentCol++) = (int) kf.keyIndex;
        wellMatrix.at (i, currentCol++) = t0[kf.wellIdx]; //traceStore.GetT0 (kf.wellIdx);
        wellMatrix.at (i, currentCol++) = kf.snr;
        wellMatrix.at (i, currentCol++) = kf.mad;
        wellMatrix.at (i, currentCol++) = kf.sd;
        wellMatrix.at (i, currentCol++) = kf.bfMetric;
        wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::A_NUC,0);
        wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::C_NUC,0);
        wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::G_NUC,0);
        wellMatrix.at (i, currentCol++) = kbf->param.at (TraceStore<double>::T_NUC,0);
        wellMatrix.at (i, currentCol++) = kf.peakSig;
        wellMatrix.at (i, currentCol++) = kf.flag;
      }
    string h5SummaryRoot = opts.outData + ".h5";
    string h5Summary = "/separator/summary";
    vector<int> flows(2);
    flows[0] = opts.maxKeyFlowLength;
    flows[1] = opts.maxKeyFlowLength+1;
    H5File h5file(h5SummaryRoot);
    h5file.Open(true);
    h5file.WriteMatrix (h5Summary, wellMatrix);
    CalcRegionEmptyStat(h5file, modelMesh, traceStore, h5SummaryRoot, flows, mask);
    h5file.Close();

    string summaryFile = opts.outData + ".summary.txt";
    ofstream o (summaryFile.c_str());
    o << "well\tkey\tt0\tsnr\tmad\ttraceMean\ttraceSd\tsigSd\tok\tbfMetric\ttauB.A\ttauB.C\ttauB.G\ttauB.T\ttauE.A\ttauE.C\ttauE.G\ttauE.T\tmeanSig\tmeanProj\tprojMad\tprojPeak\tflag";
    o << endl;
    for (size_t i = 0; i < numWells; i+=opts.samplingStep)
      {

        if (mask[i] & MaskExclude)
          {
            continue;
          }
        KeyFit &kf = wells[i];
        const KeyBulkFit *kbf = zModelBulk.GetKeyBulkFit (i);
        if (kbf != NULL)
          {
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
              << kf.peakSig << "\t" << kf.flag;
            o << endl;
          }
      }
    o.close();
  }

  int beadCount = 0, emptyCount = 0, ignoreCount = 0, libCount = 0, tfCount = 0, dudCount = 0, pinnedCount = 0, referenceCount = 0, excludedCount = 0;

  for (size_t bIx = 0; bIx < wells.size(); bIx++)
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
    if (bfMask[bIx] & MaskReference)
    {
      referenceCount++;
    }
    if (bfMask[bIx] & MaskIgnore)
    {
      ignoreCount++;
    }
    if (bfMask[bIx] & MaskLib)
    {
      libCount++;
    }
    if (bfMask[bIx] & MaskTF)
    {
      tfCount++;
    }
    if (bfMask[bIx] & MaskDud)
    {
      dudCount++;
    }
    if (bfMask[bIx] & MaskExclude) {
      excludedCount++;
    }

  }

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
  mTrace.SetMeshDist (opts.useMeshNeighbors);

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
