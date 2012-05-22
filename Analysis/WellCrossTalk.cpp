/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include "OptArgs.h"
#include "Mask.h"
#include "Traces.h"
#include "IonErr.h"
#include "DifferentialSeparator.h"
#include "Utils.h"

int centerSeen = 0;
int haystackNeg = 0;

using namespace std;
void LoadTraces(Mask &mask, vector<string> &files, vector<Traces> &flows) {
  flows.resize(files.size());
  for (size_t i = 0; i < files.size(); i++) {
    Image img;
    bool loaded = img.LoadRaw(files[i].c_str());
    if (!loaded) {
      ION_ABORT("Couldn't load file: " + files[i]);
    }
    flows[i].Init(&img, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
    img.Close();
    flows[i].CalcT0(true);
    flows[i].FillCriticalFrames();
    flows[i].CalcReference(50,50,flows[i].mGridMedian);
  }
}

void LoadRegions(const string &regionFile, vector<struct Region> &regions) {
  std::ifstream in(regionFile.c_str());
  ION_ASSERT(in.good(), "Couldn't open file: " + regionFile);
  regions.clear();
  string line;
  std::vector<std::string> words;
  while(getline(in, line)) {
    split(line,'\t',words);
    struct Region r;
    r.row = atoi(words[0].c_str());
    r.col = atoi(words[1].c_str());
    r.w = atoi(words[2].c_str());
    r.h = atoi(words[3].c_str());
    regions.push_back(r);
  }
  in.close();
}

bool PatternMatch(vector<int> &needle, vector<int> &haystack, Mask &mask, int maskMatch) {
  ION_ASSERT(needle.size() == haystack.size(), "Haystack wrong size");
  for (size_t i = 0; i < needle.size(); i++) {
    if (haystack[i] < 0) {
      return false;
      haystackNeg++;
    }
    if (needle[i] == 0 && (mask[haystack[i]] & maskMatch) > 0) {
      return false;
    }
    else if (needle[i] == 1 && (mask[haystack[i]] & maskMatch) == 0) {
      return false;
    }
  }
  return true;
}

void ParseWell(size_t rowIx, size_t colIx, size_t regionIx,
	       vector<int> &match, Mask &mask, int maskCenter, int maskMatch, 
	       vector<Traces> &flows, ostream &out, int frameStart, int frameEnd) {
  int idx = mask.ToIndex(rowIx, colIx);
  if ((mask[idx] & maskCenter) == 0) {
    return;
  }
  centerSeen++;
  vector<int> wells;
  vector<float> trace;
  mask.GetNeighbors(rowIx, colIx, wells);
  ION_ASSERT(wells.size() == match.size(), "Wells don't match string.");
  if (PatternMatch(match, wells, mask, maskMatch)) {
    out <<rowIx << "\t" << colIx << "\t" << idx << "\t" << regionIx << "\t";
    for (size_t i = 0; i < wells.size(); i++) {
      int full = ((mask[wells[i]] & maskMatch) > 0) ? 1 : 0;
      out << full;
    }
    for (size_t i = 0; i < flows.size(); i++) {
      flows[i].GetTraces(idx, trace);
      float sum = 0;
      for (int fIx = frameStart; fIx < frameEnd; fIx++) {
	sum += trace[fIx];
      }
      sum  =  sum / (frameEnd - frameStart);
      out << "\t" << sum;
    }
    for (size_t i = 0; i < flows.size(); i++) {
      out << "\t" << flows[i].GetT0(idx);
    }
    out << endl;
  }
}

void ParseMetrics(const string &match, size_t matchIx, 
		  Mask &mask,
		  int maskCenter, int maskMatch,
		  vector<struct Region> &regions, 
		  vector<Traces> &flows, string outPrefix, 
		  int frameStart, int frameEnd) {
  
  string outFile = outPrefix + "." + match + ".xtalk.txt";
  vector<int> matchVec(match.length());
  for (size_t i = 0; i < match.length(); i++) {
    if (match.at(i) == '_') {
      matchVec[i] = -1;
    }
    else if (match.at(i) == '0') {
      matchVec[i] = 0;
    }
    else if (match.at(i) == '1') {
      matchVec[i] = 1;
    }
    else {
      ION_ABORT("Don't recognize patter: " + match);
    }
  }
  ofstream out(outFile.c_str());
  out << "row\tcol\twell\tregionIx\tmatch";
  for (size_t i = 0; i < flows.size(); i++) {
    out << "\tflow" << i;
  }
  for (size_t i = 0; i < flows.size(); i++) {
    out << "\tt0." << i;
  }
  out << endl;
  for (size_t regIx = 0; regIx < regions.size(); regIx++) {
    for (int rowIx = regions[regIx].row; rowIx < regions[regIx].row + regions[regIx].h; rowIx++) {
      for (int colIx = regions[regIx].col; colIx < regions[regIx].col + regions[regIx].w; colIx++) {
	ParseWell(rowIx, colIx, regIx, matchVec, mask, maskCenter, maskMatch, flows, out, frameStart, frameEnd);
      }
    }
  }
}

int main(int argc, const char *argv[]) {
  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string regionFile;
  vector<string> matchStrings;
  vector<string> datFiles;
  int maskCenter = MaskEmpty;
  int maskMatch = MaskLive | MaskBead | MaskDud;
  string maskFile;
  string outPrefix;
  bool setHex;
  int frameStart,frameEnd;
  bool help;
  bool useDuds;
  int optCenter, optMatch;
  opts.GetOption(help, "false", 'h', "help");
  opts.GetOption(regionFile, "", '-', "region-file");
  opts.GetOption(datFiles, "", '-', "dat-files");
  opts.GetOption(matchStrings, "", '-', "matches");
  opts.GetOption(outPrefix, "", '-', "out-prefix");
  opts.GetOption(useDuds, "", '-', "use-duds");
  opts.GetOption(maskFile, "", '-', "mask-file");
  opts.GetOption(frameStart, "14", '-', "frame-start");
  opts.GetOption(frameEnd, "20", '-', "frame-end");
  opts.GetOption(optCenter, "0", '-', "center");
  opts.GetOption(optMatch, "0", '-', "match");
  opts.GetOption(setHex, "false", '-', "set-hex");
  
  if (useDuds) {
    maskMatch = MaskDud;
  }
  else if (optMatch != 0) {
    maskMatch = optMatch;
  }
  if (optCenter != 0) {
    maskCenter = optCenter;
  }

  vector<Traces> flows;
  vector<struct Region> regions;
  cout << "Loading mask." << endl;
  Mask mask(maskFile.c_str());
  mask.SetHex(setHex);
  for (size_t i = 0; i < matchStrings.size(); i++) {
    ION_ASSERT(matchStrings[i].length() == matchStrings[0].length(), "Match strings must match in length.");
  }
  cout << "Loading regions." << endl;
  LoadRegions(regionFile, regions);
  cout << "Loading traces." << endl;
  LoadTraces(mask, datFiles, flows);

  for (size_t i = 0; i < matchStrings.size(); i++) {
    cout << "Using frame num: " << frameStart << " to " << frameEnd << " for match string: " << matchStrings[i] << endl;
    ParseMetrics(matchStrings[i], i, mask, maskCenter, maskMatch, regions, flows, outPrefix, frameStart, frameEnd);
  }
  cout << "Saw: " << centerSeen << " wells and: " << haystackNeg << " negatives." << endl;
}
