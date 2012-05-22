/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include <vector>
#include "FindSlopeChange.h"
#include "Utils.h"
using namespace std;

TEST(FindSlopeChange_Test, HingTest_Test) {
  Mat<float> orig;
  orig.load("separator.reference_bf_t0.txt");
  //  orig.raw_print(cout, "load");
  Mat<float> trimmed;
  trimmed.set_size(orig.n_rows, orig.n_cols-6);
  trimmed = orig.cols(6, orig.n_cols-1);
  vector<float> vals;
  vals.resize(trimmed.n_cols);
  for (size_t i = 0; i < trimmed.n_cols; i++) {
    vals[i] = trimmed.at(0,i);
  }
  //  copy(trimmed.begin_row(0), trimmed.end_row(0), vals.begin());
  FindSlopeChange<float> mFinder;
  mFinder.SetWindowSize(10);
  mFinder.SetMaxFirstHingeSlope(30);
  mFinder.SetMinFirstHingeSlope(-2);
  float  startSumSeq = 0, slope = 0, yIntercept = 0;
  float zero = 0, hinge = 0;
  float t0Float = 0;
  int frameEnd = vals.size();
  int frameStart = 0;
  bool ok = mFinder.findNonZeroRatioChangeIndex(t0Float, startSumSeq, 
                                                slope, yIntercept,
                                                frameStart, frameEnd, 
                                                zero, hinge, vals, -1, 5, -2);
  float t0 = t0Float;
  cout << "t0: " << t0 << "\t" << ok << endl;
}


TEST(FindSlopeChange_Test, findChangeIndex_Test) {
  double tData[] = {-4,2,0,1,-1,2,-9,-13,-6,-4,3,3,7,19,5,8,0,2,3,-8,2,1,6,14,24,27,48,73,81,98,
     123,142,152,168,186,197,211,231,240,256,258,270,289,301,311,324,339,345,345,
     361,380,389,399,422,426,427,437,453,454,460,464,474,482,488,497,509,515,530,
     538,534,553,559,577,572,569,587,587,590,608,621,624,632,633,634,621,598,590,
     592,585,582,565,566,551,543,533,543,537,521,518,508,503,503,500};
  int N = sizeof(tData)/sizeof(double);
  vector<double> trace(N);
  copy(tData, tData+N, trace.begin());
  FindSlopeChange<double> finder;
  float frameIx, sumSqErr, slope, intercept;
  double bestZero = 0, bestHinge = 0;
  int startSearch=5, endSearch= 27;
  finder.findNonZeroRatioChangeIndex(frameIx, sumSqErr, slope, intercept,
				     startSearch, endSearch,
				     bestZero, bestHinge,
				     trace,1,5);
  ASSERT_NEAR(frameIx, 22.7244, .001);
}


