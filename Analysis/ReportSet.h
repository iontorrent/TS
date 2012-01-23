/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REPORTSET_H
#define REPORTSET_H

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include "FlowDiffStats.h"

class ReportSet {

public:
  
  ReportSet() {
    SetSize(-1,-1);
  }

  ReportSet(int rows, int cols) {
    SetSize(rows, cols);
  }

	void SetSize(int rows, int cols) {
		mRows = rows;
    mCols = cols;
	}

  int GetIndexRange() {
    return mRows * mCols;
  }
  
  void SetStepSize(int step) {
    if (step <= 0) {
      return;
    }
    int count = 0;
    for (int rowIx = 0; rowIx < mRows; rowIx++) {
      for (int colIx = 0; colIx < mCols; colIx++) {
				if (count++ % step == 0) {
					// @todo make rowcol2idx a utility function
					mWellIdx.push_back(rowIx * mCols + colIx);
				}
      }
    }
    std::sort(mWellIdx.begin(), mWellIdx.end());
  }
  
  void ReadSetFromFile(const std::string &file, int colIdx) {
    std::string line;
    std::ifstream in(file.c_str());
    assert(in.good());
    std::vector<std::string> words;
    std::vector<int> candidates;
    while(getline(in, line)) {
      FlowDiffStats::ChopLine(words, line);
      assert(static_cast<int>(words.size()) > colIdx);
      mWellIdx.push_back(atoi(words[colIdx].c_str()));
     }
    // Make a unique sorted list of thingss to read
    std::sort(mWellIdx.begin(), mWellIdx.end());
    std::vector<int>::iterator it = unique(mWellIdx.begin(), mWellIdx.end());
    mWellIdx.resize(it - mWellIdx.begin());
    in.close();
  }
  
  void ReadSetFromFile(const std::string &file, int rowIx, int colIx) {
    std::string line;
    std::ifstream in(file.c_str());
    assert(in.good());
    std::vector<std::string> words;
    while(getline(in, line)) {
      FlowDiffStats::ChopLine(words, line);
      int size = words.size();
      assert(size > colIx && size > rowIx);
      int idx = atoi(words[rowIx].c_str()) * mCols + atoi(words[colIx].c_str());
      mWellIdx.push_back(idx);
    }
    // Make a unique sorted list of thingss to read
    std::sort(mWellIdx.begin(), mWellIdx.end());
    std::vector<int>::iterator it = unique(mWellIdx.begin(), mWellIdx.end());
    mWellIdx.resize(it - mWellIdx.begin());
    in.close();
  }
  
  const std::vector<int> &GetReportIndexes() const {
    return mWellIdx;
  }

	bool ReportWell(int wellIx) {
		return std::binary_search(mWellIdx.begin(), mWellIdx.end(), wellIx);
	}
	
	bool IsEmpty() {
		return mWellIdx.empty();
	}
  
private:
  int mRows, mCols;
  std::vector<int> mWellIdx;
};
  
#endif // REPORTSET_H
  
