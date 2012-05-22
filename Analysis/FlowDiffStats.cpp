/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <sstream>
#include <stdlib.h>
#include <assert.h>
#include <iomanip>
#include <limits>
#include "FlowDiffStats.h"
#include "ReservoirSample.h"
#include "file-io/ion_util.h"
#include "Utils.h"

using namespace std;

const char* Q_LENGTH_COL = "q7Len";

FlowDiffStats::FlowDiffStats(int maxH) {
  Init(maxH, 0, 0, COL_BIN_SIZE, ROW_BIN_SIZE, Q_LENGTH_COL);
}

FlowDiffStats::FlowDiffStats(int maxH, unsigned int nCol, unsigned int nRow) {
  Init(maxH, nCol, nRow, COL_BIN_SIZE, ROW_BIN_SIZE, Q_LENGTH_COL);
}

FlowDiffStats::FlowDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol) {
  Init(maxH, nCol, nRow, COL_BIN_SIZE, ROW_BIN_SIZE, qLengthCol);
}
  
FlowDiffStats::FlowDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, 
			     const unsigned int nColBins, const unsigned int nRowBins) {
  Init(maxH, nCol, nRow, nColBins, nRowBins, qLengthCol);
}
  


void FlowDiffStats::Init(int maxH, unsigned int nCol, unsigned int nRow, unsigned int colBinSize, unsigned int rowBinSize, const std::string &qLengthCol) {
  mWells = NULL;
  mRow = nRow; 
  mCol = nCol;
  mGoodFlow = 0;
  mBadFlow = 0;
  mMaxHomo = maxH;
  int numNucs = NUM_NUCLEOTIDES+1;
  mFlowOrder = "TACG";
  mKeySeq = "TCAG";
  determineFlows(mKeyFlows,mKeySeq,mFlowOrder);
  mSumStats.resize(numNucs);
  for ( int i = 0; i < numNucs; i++) {
    mSumStats[i].resize(mMaxHomo);
    for ( int j = 0; j < mMaxHomo; j++) {
      mSumStats[i][j].resize(mMaxHomo, 0);
    }
  }
  mColBinSize = colBinSize;
  mRowBinSize = rowBinSize;
  mColBins = (int)(ceil((float)mCol/(float)colBinSize));
  mRowBins = (int)(ceil((float)mRow/(float)rowBinSize));
  mFlowBinnedHpSig.resize(mColBins);
  for(unsigned int iCol=0; iCol < mColBins; iCol++) {
    mFlowBinnedHpSig[iCol].resize(mRowBins);
  }

  mSlopFiltered = 0;
  mThresholdIdx = -1;
  mGenomicIdx = -1;
  mReadIdx = -1;
  mSlopIdx = -1;
  mGoodReads = 0;
  mQLengthCol = qLengthCol;
  mGCWindowSize = 20;  // window is +/- mGCWindowSize
}

FlowDiffStats::~FlowDiffStats() {
  if (mFlowErrOut.is_open()) {
    for (unsigned int flowIx = 0; flowIx < mFlowErrNumerator.size(); flowIx++) {
      stringstream errByHp;
      stringstream totalByHp;
      int nErr=0;
      int nTotal=0;
      for (int hpIx = 0; hpIx < mMaxHomo; hpIx++) {
        if(hpIx > 0) {
          errByHp << "\t";
          totalByHp << "\t";
        }
        errByHp << mFlowErrNumerator[flowIx][hpIx];
        totalByHp << mFlowErrDenominator[flowIx][hpIx];
        nErr += mFlowErrNumerator[flowIx][hpIx];
        nTotal += mFlowErrDenominator[flowIx][hpIx];
      }
      mFlowErrOut << (flowIx + mKeyFlows.size()) << "\t" << setprecision(5) << ((double)nErr/(double)nTotal) << "\t" << nErr << "\t" << nTotal << "\t" << errByHp.str() << "\t" << totalByHp.str() << endl;
    }
    mFlowErrOut.close();
  }
  mFlowStatsOut.close();
  if (mWells != NULL) {
    mWells->Close();
    delete mWells;
  }
  cout << "Bad Reads are: " << (mBadFlow * 1.0) / (mBadFlow + mGoodFlow + 1e-10) << " of " << (mBadFlow + mGoodFlow) << " total reads." << endl;
  cout << "Slop filtered reads are: " << (mSlopFiltered* 1.0) / (mSlopFiltered + mGoodReads + 1e-10) << " of total reads." << endl;
}

void FlowDiffStats::SetGCWindowSize(unsigned int sz) {
  mGCWindowSize = sz / 2;  // internally represent window size by half-window
}

void FlowDiffStats::SetStatsOut(const std::string &statsFile) {
  mFlowStatsOut.open(statsFile.c_str());
  assert(mFlowStatsOut.is_open());
  mFlowStatsOut << "name\tflow\tread\tgenomic\tnuc" << endl;
}

void FlowDiffStats::SetSNROut(const std::string &snrFile) {
  mFlowSNROut.open(snrFile.c_str());
  assert(mFlowSNROut.is_open());
}

void FlowDiffStats::SetBinnedHpSigOut(const std::string &binnedHpSigFile) {
  mFlowBinnedHpSigOut.open(binnedHpSigFile.c_str());
  assert(mFlowBinnedHpSigOut.is_open());
}

void FlowDiffStats::SetFlowGCOut(const std::string &gcFile) {
  mFlowGCOut.open(gcFile.c_str());
  assert(mFlowGCOut.is_open());
}

void FlowDiffStats::SetFlowErrOut(const std::string &flowErrFile) {
  mFlowErrOut.open(flowErrFile.c_str());
  assert(mFlowErrOut.is_open());
}

int FlowDiffStats::FillInFlows(int &seqLength, std::vector<int> &flow, const std::string &seq, int nFlows) {
   
  unsigned int seqCur = 0;
  int numFlows = 0;
  flow.resize(nFlows);
  fill(flow.begin(), flow.end(), 0);
  for (unsigned int flowIx = 0; flowIx < flow.size(); flowIx++) {
    numFlows++;
    int nucIx = (mKeyFlows.size()+flowIx) % mFlowOrder.size();
    if (seq[seqCur] == mFlowOrder[nucIx]) {
      // match, get all of them...
      char c = seq[seqCur];
      while(seqCur < seq.size() && seq[seqCur] == c ) {
	flow[flowIx]++;
	seqCur++;
      }
      if (seqCur >= seq.size()) {
	break;
      }
    }
  }
  flow.resize(numFlows);
  seqLength = seqCur;
  return numFlows;
}

int FlowDiffStats::NucToInt(char nuc) {
  switch(nuc) {
    case 'A':
    case 'a':
      return(0);
      break;

    case 'C':
    case 'c':
      return(1);
      break;

    case 'G':
    case 'g':
      return(2);
      break;

    case 'T':
    case 't':
      return(3);
      break;

    default:
      cerr << "Couldn't find index for nuc: " << nuc << endl;
      return -1;
  }
}

std::string FlowDiffStats::IntToNuc(int num) {
  switch(num) {
    case 0:
      return("A");
      break;

    case 1:
      return("C");
      break;

    case 2:
      return("G");
      break;

    case 3:
      return("T");
      break;

    default:
      cerr << "Couldn't find nuc for index: " << num << endl;
      return "UNKNOWN";
  }
}

void FlowDiffStats::RecordDifference(char nuc, int reference, int read) {
  if (reference >= mMaxHomo || read >= mMaxHomo) {
    return;
  }
  int nucIx = NucToInt(nuc);
  int allIx = mSumStats.size()-1;
  mSumStats[nucIx][reference][read]++;
  mSumStats[allIx][reference][read]++;
}

void FlowDiffStats::PrintSumStat(int nucIx, std::ostream &out) {
  string nuc = (nucIx < NUM_NUCLEOTIDES) ? IntToNuc(nucIx) : "All";
  for (unsigned int referenceIx = 0; referenceIx < mSumStats[nucIx].size(); referenceIx++) {
    out << nuc << "\t" << referenceIx;
    for (unsigned int readIx = 0; readIx < mSumStats[nucIx][referenceIx].size(); readIx++) {
      out << "\t" << mSumStats[nucIx][referenceIx][readIx];
    }
    out << endl;
  }
}

void FlowDiffStats::RecordFlowHPStats(int row, int col, std::vector<int> &reference, std::vector<int> &read) {
  size_t numFlows = std::min(reference.size(), read.size());
  unsigned int currentSize = mFlowValues.size();
  if(currentSize < numFlows) {
    mFlowValues.resize(numFlows); 
    for (unsigned int i = currentSize; i <  numFlows; i++) {
      mFlowValues[i].resize(mMaxHomo);
    }
  }

  currentSize = mFlowBinnedHpSig[0][0].size();
  if(currentSize < numFlows) {
    for(unsigned int iCol=0; iCol < mColBins; iCol++) {
      for(unsigned int iRow=0; iRow < mRowBins; iRow++) {
        mFlowBinnedHpSig[iCol][iRow].resize(numFlows);
        for(unsigned int iFlow = currentSize; iFlow <  numFlows; iFlow++) {
          mFlowBinnedHpSig[iCol][iRow][iFlow].resize(mMaxHomo);
        }
      }
    }
  }

  unsigned int rowBin = (unsigned int)(floor(row/(float) mRowBinSize));
  unsigned int colBin = (unsigned int)(floor(col/(float) mColBinSize));
  for (size_t flowIx = 0; flowIx < numFlows; flowIx++) {
    if (reference[flowIx] < mMaxHomo) {
      //      float value = mWellVals[row][col][flowIx];
      float value = mWells->At(row,col,flowIx+ mKeyFlows.size());
      mFlowValues[flowIx][reference[flowIx]].AddValue(value);
      mFlowBinnedHpSig[colBin][rowBin][flowIx][reference[flowIx]].AddValue(value);
    }
  }
}

void FlowDiffStats::CompareFlows(const std::string& name, std::vector<int> &reference, std::vector<int> &read) {
  int row, col;
  GetRowColFromName(name, row, col);
  CompareFlows(row, col, reference, read);
}


void FlowDiffStats::CompareFlows(int row, int col, std::vector<int> &reference, std::vector<int> &read) {
  
  bool badFlow = CheckIllegalFlowSeq(read);

  if (!badFlow) {
    mGoodFlow++;
    size_t numFlows = std::min(reference.size(), read.size());

    // compute GC for flow
    vector<int> gcRunningSum, totalNucRunningSum;
    if (mFlowGCOut.good()) {
      if (mGCCounts.size() == 0) { // time to initialize
	mGCCounts.resize(101);
	for (unsigned int i = 0; i <= 100; i++) {
	  mGCCounts[i].resize(mMaxHomo);
	  for (unsigned int j = 0; j < (unsigned int)mMaxHomo; j++)
	    mGCCounts[i][j].resize(mMaxHomo);
	}
      }

      gcRunningSum.resize(numFlows);
      totalNucRunningSum.resize(numFlows);
      for (size_t flowIx = 0; flowIx < numFlows; flowIx++) {
	char c = mFlowOrder[(mKeyFlows.size() + flowIx) % mFlowOrder.size()];
	if (flowIx == 0)
	  gcRunningSum[flowIx] = totalNucRunningSum[flowIx] = 0;
	else {
	  gcRunningSum[flowIx] = gcRunningSum[flowIx-1];
	  totalNucRunningSum[flowIx] = totalNucRunningSum[flowIx-1];
	}
	
	if (c == 'G' || c == 'C')
	  gcRunningSum[flowIx] += reference[flowIx];
	totalNucRunningSum[flowIx] += reference[flowIx];
      }
    }

    // record stats
    for (size_t flowIx = 0; flowIx < numFlows; flowIx++) {
      char c = mFlowOrder[(mKeyFlows.size() + flowIx) % mFlowOrder.size()];
      if (mFlowStatsOut.good()) {
	mFlowStatsOut << row << ":" << col << "\t" << flowIx << "\t" << read[flowIx] << "\t" << reference[flowIx] << "\t" << c << endl;
      }
      RecordDifference(c, reference[flowIx], read[flowIx]);
      if (mWells != NULL) {
	RecordFlowHPStats(row, col, reference, read);
      }

      if (mFlowGCOut.good() && flowIx >= mGCWindowSize && flowIx + mGCWindowSize < numFlows && reference[flowIx] < mMaxHomo && read[flowIx] < mMaxHomo) {
	float gc = gcRunningSum[flowIx + mGCWindowSize] - gcRunningSum[flowIx - mGCWindowSize];
	float all = totalNucRunningSum[flowIx + mGCWindowSize] - totalNucRunningSum[flowIx - mGCWindowSize];
        all = max(1.0f, all);
	int gcPct = (int) nearbyint(100 * (gc / all));
        if (gcPct >= 0.0f && (size_t)gcPct < mGCCounts.size()) {
          mGCCounts[gcPct][reference[flowIx]][read[flowIx]]++;
        }
	//	cerr << row << "\t" << col << "\t" << c << "\t" << gc << "\t" << all << "\t" << gcPct << "\t" << flowIx << "\t" << reference[flowIx] << "\t" << read[flowIx] << "\t" << (gc / all) << endl;
      }

      if(reference[flowIx] < mMaxHomo) {
        if (flowIx >= mFlowErrNumerator.size()) {
          size_t oldSize = mFlowErrNumerator.size();
          mFlowErrNumerator.resize(flowIx+1);
          mFlowErrDenominator.resize(flowIx+1);
          for (size_t iFlow = oldSize; iFlow <= flowIx; iFlow++) {
            mFlowErrNumerator[iFlow].resize(mMaxHomo,0);
            mFlowErrDenominator[iFlow].resize(mMaxHomo,0);
          }
        }
        mFlowErrDenominator[flowIx][reference[flowIx]]++;
        if (reference[flowIx] != read[flowIx]) {
          mFlowErrNumerator[flowIx][reference[flowIx]]++;
        }
      }
    }
  }
  else {
    mBadFlow++;
  }
}

bool FlowDiffStats::CheckIllegalFlowSeq(std::vector<int> &flowSeq) {
  vector <bool> finishedNuc(NUM_NUCLEOTIDES,false);
  bool illegalFlowSeq = false;
  for (size_t flowIx = 0; flowIx < flowSeq.size(); flowIx++) {
    int testNuc = NucToInt(mFlowOrder[(mKeyFlows.size() + flowIx) % mFlowOrder.size()]);
    if(flowSeq[flowIx] > 0) {
      for(int iNuc=0; iNuc < NUM_NUCLEOTIDES; ++iNuc)
        finishedNuc[iNuc] = (iNuc==testNuc);
    } else {
      // Mark the nuc just tested as finished
      finishedNuc[testNuc] = true;

      // Check if we have now finished all nucs
      bool finished = true;
      for(int iNuc=0; iNuc < NUM_NUCLEOTIDES; ++iNuc) {
        if(!finishedNuc[iNuc]) {
          finished = false;
          break;
        }
      }
      if(finished)
        illegalFlowSeq = true;
    }

    if(illegalFlowSeq)
      break;
  }

  return(illegalFlowSeq);
}

int GetRowColFromBfmask(const std::string &bfmask, int *nRow, int *nCol) {
  FILE *fp = NULL;
  fp = fopen(bfmask.c_str(),"rb");
  if(fp == NULL) {
    return(1);
  }
  if((fread(nRow, sizeof(uint32_t), 1, fp )) != 1) {
    fclose(fp);
    return(1);
  }
  if((fread(nCol, sizeof(uint32_t), 1, fp )) != 1) {
    fclose(fp);
    return(1);
  }
  fclose(fp);
  return(0);
}

void determineFlows(std::vector<unsigned int> &seqFlow, const std::string &seq, const std::string &flowOrder) {
  size_t nFlowPerCycle = flowOrder.length();
  size_t nBases = seq.length();
  seqFlow.resize(0);
  
  unsigned int iFlow=0;
  unsigned int iNuc=0;
  while(iNuc < nBases) {
    char flowBase = flowOrder[iFlow % nFlowPerCycle];
    unsigned int hp=0;
    while((iNuc < nBases) && (seq[iNuc]==flowBase)) {
      iNuc++;
      hp++;
    }
    seqFlow.push_back(hp);
    iFlow++;
  }
}

void FlowDiffStats::FillInSubset(const std::string &samFile, int minVal,
                                 int minRow, int maxRow, 
                                 int minCol, int maxCol,
                                 std::vector<int32_t> &xSubset, 
                                 std::vector<int32_t> &ySubset) {
  SetAlignmentInFile(samFile);
  std::string name, genomic, read;
  int row, col;
  vector<string> words;
  string line;
  bool checkRow = (minRow > -1) && (maxRow > -1);
  bool checkCol = (minCol > -1) && (maxCol > -1);
  ReservoirSample<std::pair<int,int> > sample(100000);
  while(getline(mAlignments, line)) {
    split(line,'\t',words);
    int qLen = atoi(words[mThresholdIdx].c_str());
    int slop = atoi(words[mSlopIdx].c_str());
    if (qLen >= minVal && slop != 0) {
      continue;
    }
    if (qLen >= minVal && slop == 0) {
      name = words[mNameIdx];
      genomic = words[mGenomicIdx];
      read = words[mReadIdx];
      GetRowColFromName(name, row, col);
      if(checkRow && (row < minRow || row >= maxRow))
        continue;
      if(checkCol && (col < minCol || col >= maxCol))
        continue;
      sample.Add(std::pair<int,int>(row,col));
    }
  }
  sample.Finished();
  vector<std::pair<int,int> > vecSamp = sample.GetData();
  for (size_t i = 0; i < vecSamp.size(); i++) {
    xSubset.push_back(vecSamp[i].second);
    ySubset.push_back(vecSamp[i].first);
  }
  cout << "Got: " << xSubset.size() << " wells to process." << endl;
  mAlignments.close();
}

void FlowDiffStats::SetWellsFile(const std::string &wells, int nRow, int nCol, int maxFlow,
                                 const std::vector<int32_t> &xSubset, const std::vector<int32_t> &ySubset) {
  cout << "Reading " << wells << endl;
  std::string file, prefix;
  FillInDirName(wells, prefix, file);
  mWells = new RawWells(prefix.c_str(), file.c_str(), nRow, nCol);
  mWells->SetSubsetToLoad(xSubset, ySubset);

  // Find 1mers in the key flows
  std::vector <unsigned int> keyOneMerFlows;
  for (unsigned int iFlow = 0; iFlow < mKeyFlows.size()-1; iFlow++) {
    if(mKeyFlows[iFlow]==1) {
      keyOneMerFlows.push_back(iFlow);
    }
  }

  mWells->OpenForRead();
  if(keyOneMerFlows.size() > 0) {
    for (size_t wellIx = 0; wellIx < mWells->NumWells(); wellIx++) {
      if (!mWells->HaveWell(wellIx)) {
        continue;
      }
      double scale = 0;
      for(unsigned int iOneMer = 0; iOneMer < keyOneMerFlows.size(); iOneMer++) {
        scale += mWells->At(wellIx, keyOneMerFlows[iOneMer]);
      }
      if(fabs(scale) > numeric_limits<double>::epsilon()) {
        scale /= keyOneMerFlows.size();
        for (size_t i = 0; i < mWells->NumFlows(); i++) {
          mWells->Set(wellIx, i, mWells->At(wellIx, i)/scale);
        }
      }
    }
  }
}

bool FlowDiffStats::GetRowColFromName(const std::string &name, int &row, int &col) {
  int colonIx = name.rfind(':');
  if (colonIx > 0) {
    if(1 != ion_readname_to_rowcol(name.c_str(), &row, &col)) {
      cerr << "Error parsing read name: '" << name << "'" << endl;
      return false;
    } else {
      return true;
    }
  } else {
    int barIx = name.find('|');
    if (barIx < 0) {
      return false;
    }
    string rowS = name.substr(1,barIx-1);
    string colS = name.substr(barIx+2,name.length() - (barIx + 2));
    row = atoi(rowS.c_str());
    col = atoi(colS.c_str());
    return true;
  }
}

bool FlowDiffStats::GetNextAlignment(std::string &name, std::string &genomic, std::string &read, int &row, int &col, int minVal) {
  string line;
  bool found = false;
  vector<string> words;
  while(getline(mAlignments, line)) {
    split(line,'\t',words);
    int qLen = atoi(words[mThresholdIdx].c_str());
    int slop = atoi(words[mSlopIdx].c_str());
    if (qLen >= minVal && slop != 0) {
      mSlopFiltered++;
    }
    if (qLen >= minVal && slop == 0) {
      mGoodReads++;
      name = words[mNameIdx];
      genomic = words[mGenomicIdx];
      read = words[mReadIdx];
      GetRowColFromName(name, row, col);
      found = true;
      break;
    }
  }
  return found;
}

void FlowDiffStats::SetAlignmentInFile(const std::string &samfile) {
  mAlignments.open(samfile.c_str(), ifstream::in);
  assert(mAlignments.is_open());
  string line;
  getline(mAlignments, line);
  stringstream ss(line);
  string s;
  int count =0;
  while (ss >> s) {
    if (s == mQLengthCol) {
      mThresholdIdx = count;
    }
    else if (s == "qDNA.a") {
      mReadIdx = count;
    }
    else if (s == "tDNA.a") {
      mGenomicIdx = count;
    }
    else if (s == "name") {
      mNameIdx = count;
    }
    else if (s == "start.a") {
      mSlopIdx = count;
    }
    count++;
  }
  if (mThresholdIdx < 0 || mReadIdx < 0 || mGenomicIdx < 0) {
    cerr << "Didn't find a column." << endl;
  }
}

void FlowDiffStats::FilterAndCompare(int numFlows,
				       std::ostream &out,
				       int minVals,
				       int minRow,
				       int maxRow,
				       int minCol,
				       int maxCol) {
  string name;
  string genomic;
  string read;
  int row=0;
  int col=0;
  bool checkRow = (minRow > -1) && (maxRow > -1);
  bool checkCol = (minCol > -1) && (maxCol > -1);
  int count = 0;
  while(GetNextAlignment(name, genomic, read, row, col, minVals)) {
    if(checkRow && (row < minRow || row >= maxRow))
      continue;
    if(checkCol && (col < minCol || col >= maxCol))
      continue;
    if (!mWellsToUse.empty() && mWellsToUse[(row * mCol) + col] == false) {
      continue;
    }
    if (mWells != NULL && !mWells->HaveWell(row,col)) {
      continue;
    }
    count++;
    while (read.size() > 0 && genomic.size() > 0 && read[0] == 'G') {
      read.erase(0,1);
      genomic.erase(0,1);
    }
    CompareRead(name, genomic, read, numFlows);
  }
  FinishComparison(numFlows);

  out << "Nuc\tTrueMer";
  for (int i = 0; i < mMaxHomo; i++) {
    out << "\t" << i << "mer";
  }
  out << endl;
  for (int i = 0; i <= NUM_NUCLEOTIDES; i++) {
    PrintSumStat(i, out);
  }
  if (mFlowSNROut.is_open()) {
    cout << "Writing: " << mFlowValues.size() << " Flow snr values." << endl;
    for (unsigned int flowIx = 0; flowIx < mFlowValues.size(); flowIx++) {
      mFlowSNROut << flowIx + mKeyFlows.size();
      for (unsigned int hpIx = 0; hpIx < mFlowValues[flowIx].size(); hpIx++) {
	mFlowSNROut << "\t" << mFlowValues[flowIx][hpIx].GetMean() << "\t" << mFlowValues[flowIx][hpIx].GetSD();
      }
      mFlowSNROut << endl;
    }
    mFlowSNROut.close();
  }
  if (mFlowBinnedHpSigOut.is_open()) {
    cout << "Writing: " << mFlowBinnedHpSig.size() << " flows of binned HP signal values." << endl;
    mFlowBinnedHpSigOut << "flow"
			<< "\t" << "hpLen"
			<< "\t" << "colBin"
			<< "\t" << "rowBin"
			<< "\t" << "sigCount"
			<< "\t" << "sigMean"
			<< "\t" << "sigSD"
			<< endl;
    unsigned int nFlow = mFlowBinnedHpSig[0][0].size();
    for (unsigned int flowIx = 0; flowIx < nFlow; flowIx++) {
      for(int hpLen = 0; hpLen <  mMaxHomo; hpLen++) {
        for(unsigned int iCol=0; iCol < mColBins; iCol++) {
          for(unsigned int iRow=0; iRow < mRowBins; iRow++) {
            if(mFlowBinnedHpSig[iCol][iRow][flowIx][hpLen].GetCount() > 1) {
              mFlowBinnedHpSigOut << (flowIx + mKeyFlows.size())
				  << "\t" << hpLen
				  << "\t" << setprecision(5) << (double)iCol/(double)mColBins
				  << "\t" << setprecision(5) << (double)iRow/(double)mRowBins
				  << "\t" << mFlowBinnedHpSig[iCol][iRow][flowIx][hpLen].GetCount()
				  << "\t" << mFlowBinnedHpSig[iCol][iRow][flowIx][hpLen].GetMean()
				  << "\t" << mFlowBinnedHpSig[iCol][iRow][flowIx][hpLen].GetSD()
				  << endl;
            }
          }
        }
      }
    }
    mFlowBinnedHpSigOut.close();
  }

  if (mFlowGCOut.is_open() && mFlowGCOut.good()) {
    cout << "Writing: Errors by GC" << endl;
    mFlowGCOut << "GC\treference\tread\tcount" << endl;
    for (unsigned int gcPct=0; gcPct < mGCCounts.size(); gcPct++) {
      for (unsigned int i = 0; i < (unsigned int)mMaxHomo; i++) {
	for (unsigned int j = 0; j < (unsigned int)mMaxHomo; j++) {
	  mFlowGCOut << gcPct << "\t" << i << "\t" << j << "\t" << mGCCounts[gcPct][i][j] << endl;
	}
      }
    }
    mFlowGCOut.close();
  }
}


