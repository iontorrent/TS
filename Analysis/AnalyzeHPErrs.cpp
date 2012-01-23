/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

// Generate a bunch of stats regarding HP differences between a reference and
// a run.  A reference may be a genomic alignment (parsed sam file) or two sff files.

#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include "FlowDiffStats.h"
#include "GenomeDiffStats.h"
#include "SffDiffStats.h"
#include "OptArgs.h"
#include "ReservoirSample.h"

using namespace std;

void usage() {
  cout << "AnalyzeHPErrs - Calculate the error rates for different homopolymers and " << endl
       << "per nucleotide.  If a pair of SFF files is specified, then the first is" << endl
       << "the reference and the second is compared to the reference.  Otherwise," << endl
       << "the genomic sequence from the alignment is used as the reference." << endl << endl
       << "Usage: " << endl
       << "  AnalyzeHPErrs --sam-parsed Default.sam.parsed --qscore-col q17Len --min-qlength 50 \\ "<< endl
       << "     [ -sff1 run1.sff -sff2 run2.sff ] \\ "<< endl
       << "     --stats-out stats.out --flows-out flows.out --summary-out summary.out " << endl
       << "options: " << endl
       << "  -h,--help      This message" << endl
       << "  --sam-parsed   Default.sam.parsed alignment file" << endl
       << "  --sff1         SFF file for run 1 -- the reference" << endl
       << "  --sff2         SFF file for run 2 -- the test" << endl
       << "  --stats-out    File to write various statistics" << endl
       << "  --flows-out    File to write flow data to" << endl
       << "  --paired-out   File to write run1 vs run2 alignments" << endl
       << "  --summary-out  File for summary of the various types of errors" << endl
       << "  --max-hp       Maximum length of homopolymer to gather stats for" << endl
       << "  --num-flows    Number of flows to examine (default 40)" << endl
       << "  --qscore-col   Name of column in sam parsed file to use for q length (default q7Len)" << endl
       << "  --min-qlength  Minimum Q length to analyze. (default 25)" << endl
       << "  --min-row      Minimum row to restrict anlaysis (default 0, should be in [0,1])" << endl
       << "  --max-row      Maximum row to restrict anlaysis (default 1, should be in [0,1])" << endl
       << "  --min-col      Minimum col to restrict anlaysis (default 0, should be in [0,1])" << endl
       << "  --max-col      Maximum col to restrict anlaysis (default 1, should be in [0,1])" << endl
       << "  --flow-order   Specify flow order, default \"TACG\"" << endl
       << "  --key-seq      Specify key sequence, default \"TCAG\"" << endl
       << "  --gc-err-file  File to write errors binned by G+C" << endl
       << "  --gc-win       Size of G+C window in flows" << endl
       << "" << endl;
    exit(1);
}

void ReadSetFromFile(const std::string &file, int colIdx, std::vector<int> &wells ) {
	std::string line;
	std::ifstream in(file.c_str());
	assert(in.good());
	std::vector<std::string> words;
	std::vector<int> candidates;
	wells.clear();
	while(getline(in, line)) {
		FlowDiffStats::ChopLine(words, line);
		assert(static_cast<int>(words.size()) > colIdx);
		wells.push_back(atoi(words[colIdx].c_str()));
	}
        // Make a unique sorted list of thingss to read
	std::sort(wells.begin(), wells.end());
	std::vector<int>::iterator it = unique(wells.begin(), wells.end());
	wells.resize(it - wells.begin());
	in.close();
}
  

int main(int argc, const char *argv[]) {
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  int hpLength;
  string statsOut;
  string alignmentOut;
  string pairedOut;
  string flowsOut;
  string summaryOut;
  string samFile;
  string qScoreCol;
  string wellsFile;
  string bfmaskFile;
  string snrFile;
  string binnedHpSigFile;
  string flowErrFile;
  string gcErrFile;
  int gcWin;
  string flowOrder;
  string keySeq;
  int numFlows;
  bool help;
  int qLength;
  double colCenter;
  double rowCenter;
  int colSize;
  int rowSize;
  int sampleSize;
  string wellsToUse;
  string run1, run2;
  opts.GetOption(run1, "", '-', "sff1");
  opts.GetOption(run2, "", '-', "sff2");
  opts.GetOption(wellsToUse, "", '-', "use-wells");
  opts.GetOption(samFile, "", '-', "sam-parsed");
  opts.GetOption(statsOut, "", '-', "stats-out");
  opts.GetOption(flowsOut, "", '-', "flows-out");
  opts.GetOption(alignmentOut, "", '-', "align-out");
  opts.GetOption(summaryOut, "", '-', "summary-out");
  opts.GetOption(pairedOut, "", '-', "paired-out");
  opts.GetOption(numFlows, "40", '-', "num-flows");
  opts.GetOption(hpLength, "6", '-', "max-hp");
  opts.GetOption(qScoreCol, "q7Len", '-', "qscore-col");
  opts.GetOption(qLength, "25", '-', "min-qlength");
  opts.GetOption(help,   "false", 'h', "help");
  opts.GetOption(wellsFile,   "", '-', "wells-file");
  opts.GetOption(bfmaskFile,   "", '-', "bfmask-file");
  opts.GetOption(snrFile,   "", '-', "snr-file");
  opts.GetOption(binnedHpSigFile,   "", '-', "binned-hp-sig-file");
  opts.GetOption(flowErrFile, "", '-', "flow-err-file");
  opts.GetOption(gcErrFile, "", '-', "gc-err-file");
  opts.GetOption(flowOrder, "", '-', "flow-order");
  opts.GetOption(keySeq, "", '-', "key-seq");
  opts.GetOption(colCenter, "0.5", '-', "col-center");
  opts.GetOption(rowCenter, "0.5", '-', "row-center");
  opts.GetOption(colSize, "0", '-', "col-size");
  opts.GetOption(rowSize, "0", '-', "row-size");
  opts.GetOption(gcErrFile, "", '-', "gc-err-file");
  opts.GetOption(gcWin, "40", '-', "gc-win");
  opts.GetOption(sampleSize, "100000", '-', "sample-size");
  if (help || samFile.empty() || statsOut.empty() || summaryOut.empty()) {
    usage();
  }
  opts.CheckNoLeftovers();

  // Some checks to make sure sensible bounds have been set
  if(colCenter < 0 || colCenter > 1) {
    cerr << "AnalyzeHPErrs - col-center must be in the range [0,1]" << endl;
    exit(1);
  }
  if(rowCenter < 0 || rowCenter > 1) {
    cerr << "AnalyzeHPErrs - row-center must be in the range [0,1]" << endl;
    exit(1);
  }
  if(colSize < 0) {
    cerr << "AnalyzeHPErrs - col-size cannot be negative." << endl;
    exit(1);
  }
  if(rowSize < 0) {
    cerr << "AnalyzeHPErrs - row-size cannot be negative." << endl;
    exit(1);
  }

  // Determine rows & cols if a bfmask file was supplied
  int nRow=0;
  int nCol=0;
  if(!bfmaskFile.empty()) {
    if(GetRowColFromBfmask(bfmaskFile, &nRow, &nCol)) {
      cerr << "AnalyzeHPErrs - problem determining rows & columns from bfmask file " << bfmaskFile << endl;
      exit(1);
    }
  }
	
  // Set up fds object
  FlowDiffStats* fds;
  if (!run1.empty()) {
    SffDiffStats* sds = new SffDiffStats(hpLength, nCol, nRow, qScoreCol, run1, run2);
    if (!pairedOut.empty())
      sds->SetPairedOut(pairedOut);
    fds = dynamic_cast<FlowDiffStats*>(sds);
  }
  else {
    GenomeDiffStats* gds = new GenomeDiffStats(hpLength, nCol, nRow, qScoreCol);
    if(alignmentOut != "") {
      gds->SetAlignmentsOut(alignmentOut);
    }
    if (!flowsOut.empty()) {
      gds->SetFlowsOut(flowsOut);
    }
    fds = dynamic_cast<FlowDiffStats*>(gds);
  }

  if (gcErrFile != "") {
    fds->SetFlowGCOut(gcErrFile);
    fds->SetGCWindowSize(gcWin);
  }

  if(keySeq != "") {
    fds->SetKeySeq(keySeq);
  }
  if(flowOrder != "") {
    fds->SetFlowOrder(flowOrder);
  }
  fds->SetStatsOut(statsOut);

  if (!wellsToUse.empty()) {
    std::vector<int> wells;
    std::vector<bool> use;
    ReadSetFromFile(wellsToUse, 0, wells);
    use.resize(nRow * nCol, false);
    int count = 0;
    ReservoirSample<int> wellSample(sampleSize);
    for (size_t i = 0; i < wells.size(); i++) {
      wellSample.Add(wells[i]);
    }
    wells = wellSample.GetData();
    for (size_t i = 0; i < wells.size(); i++) {
      use[wells[i]] = true;
      count++;
    }
    cout << "Read: " << count << " reads." << endl;
    fds->SetWellToAnalyze(use);
  }


  // Set integer-value row & column bounds
  int minRow=-1;
  int maxRow=-1;
  int minCol=-1;
  int maxCol=-1;
  if(colSize > 0 || rowSize > 0) {
    if(bfmaskFile.empty()) {
      cerr << "AnalyzeHPErrs - must specify bfmask file when restricting row or column ranges" << endl;
      exit(1);
    }
    if(rowSize > 0) {
      minRow = floor(nRow * rowCenter - rowSize / 2.0);
      maxRow = minRow + rowSize;
      minRow = std::max(0,minRow);
      maxRow = std::min(nRow,maxRow);
    }
    if(colSize > 0) {
      minCol = floor(nCol * colCenter - colSize / 2.0);
      maxCol = minCol + colSize;
      minCol = std::max(0,minCol);
      maxCol = std::min(nCol,maxCol);
    }
  }

  if (wellsFile != "") {
    std::vector<int32_t> xSubset, ySubset;
    fds->FillInSubset(samFile, qLength, minRow, maxRow, minCol, maxCol, xSubset, ySubset);
    if(bfmaskFile.empty()) {
      cerr << "AnalyzeHPErrs - must specify bfmask file when specifying wells file" << endl;
      exit(1);
    }
    fds->SetWellsFile(wellsFile, nRow, nCol, numFlows, xSubset, ySubset);
  }
  if (snrFile != "") {
    cout << "Opening snr file: " << snrFile << endl;
    fds->SetSNROut(snrFile);
  }
  if (binnedHpSigFile != "") {
    cout << "Opening binned HP signal file: " << binnedHpSigFile << endl;
    fds->SetBinnedHpSigOut(binnedHpSigFile);
  }
  if (flowErrFile != "") {
    cout << "Opening flow err file: " << flowErrFile << endl;
    fds->SetFlowErrOut(flowErrFile);
  }
  ofstream summary;
  summary.open(summaryOut.c_str());
  cout << "Reading and analyzing alignments from: " << samFile << endl;
  if(minCol > -1 || maxCol > -1)
    cout << "  Restricting to " << (maxCol-minCol) << " cols in the range [" << minCol << "," << maxCol << ")" << endl;
  if(minRow > -1 || maxRow > -1)
    cout << "  Restricting to " << (maxRow-minRow) << " rows in the range [" << minRow << "," << maxRow << ")" << endl;

  fds->SetAlignmentInFile(samFile);
  fds->FilterAndCompare(numFlows, summary, qLength, minRow, maxRow, minCol, maxCol);

  summary.close();
  delete fds;
  cout << "Done." << endl;
  return 0;
}
