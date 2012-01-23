/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FINDSLOPECHANGE_H
#define FINDSLOPECHANGE_H

#include <vector>
#include <limits>
#include <armadillo>

#define FRAME_ZERO_WINDOW 24
#define MIN_ALLOWED_FRAME 12
using namespace std;
using namespace arma;

template <class T>
class FindSlopeChange {
 public: 

  /** reg is the regression to do after break point is found. */
  bool findNonZeroChangeIndex(float &frameIx, float &sumSqErr,
			      float &slope, float &yIntercept,
			      std::vector<T> &values, int startOffset,
			      int endOffset, int startSearch, int endSearch, 
			      int reg=5) {
    if (startOffset > 2) {}
    frameIx = -1;
    sumSqErr = numeric_limits<float>::max();
    // Convert to doubles for gsl
    endOffset = min((int)values.size(), endOffset);
    endSearch = min((int)values.size(), endSearch);

    // Try all possible values and grab the one that fits best
    double bestFit = numeric_limits<double>::max();
    double bestFitIx = -1;
    double sumSq;
    double bestsc0, bestsc1;
    /* startSearch = startSearch - startOffset; */
    /* endSearch = endSearch - startOffset; */
    mParam.set_size(2,1);

    for (int i = startSearch; i < endSearch; i++) {
      double firstSS = 0, secondSS = 0;

      for (int yIx = 0; yIx < i; yIx++) {
	firstSS += values[yIx] * values[yIx] ;
      }
      // Fit second portion of line
      int rest =  (endOffset- i);
      mY.set_size(rest);
      copy(&values[i], &values[i] + rest, mY.begin());
      mX.set_size(rest, 2);
      for (size_t xIx = 0; xIx < mX.n_rows; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = i+xIx;
      }
      mParam = solve(mX, mY);
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mParam;
      sumSq = 0;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	sumSq += (diff * diff);
      }
      secondSS = sumSq;
      if (firstSS + secondSS < bestFit && mParam[1] > 0) {
	bestFit = firstSS + secondSS;
	bestFitIx = i;
	bestsc0 = mParam[0];
	bestsc1 = mParam[1];
      }
    }
    frameIx = bestFitIx;
    yIntercept = 0;
    slope = 0;

    // Ok now polish off the best fit for intercept
    if (reg > 1) {
      mY.set_size(reg);
      mX.set_size(reg,2);
      for (int i = 0; i < reg; i++) { 
	mY[i] = values[i + frameIx];
	mX.at(i,0) = 1;
	mX.at(i,1) = frameIx + i;
      }
      mParam = solve(mX, mY);
      mPredicted = mX * mParam;

      sumSq = 0;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	sumSq += (diff * diff);
      }
      sumSqErr = bestFit;

      for (int i = 0; i < reg; i++) { 
	mY[i] = values[frameIx - i];
	mX.at(i,0) = 1;
	mX.at(i,1) = frameIx - i;
      }
      mHingeParam = solve(mX, mY);
      mPredicted = mX * mParam;

      double denom = (mParam[1] - mHingeParam[1]);
      double xx = (mHingeParam[0] - mParam[0])/(mParam[1] - mHingeParam[1]);
      yIntercept = mParam[0];
      slope = mParam[1];
      frameIx = startSearch;
      if (denom != 0) {
	// frameIx = -1 * yIntercept / slope;
	frameIx = xx;
      }
    }
    if (frameIx < startSearch) {
      frameIx = -1;
      return false;
    }
    if (frameIx > endSearch) {
      frameIx = -1;
      return false;
    }
    return true;
  }

  /** reg is the regression to do after break point is found. */
  bool findChangeIndex(float &frameIx, float &sumSqErr,
		       float &slope, float &yIntercept,
		       std::vector<T> &values, int startOffset,
		       int endOffset, int startSearch, int endSearch, 
		       int reg=5) {
    if (startOffset > 2) {}
    frameIx = -1;
    double denom = 0, lineMeet = 0;
    double slopeOne = 0, interceptOne = 0;
    double slopeTwo = 0, interceptTwo = 0;
    sumSqErr = numeric_limits<float>::max();
    // Convert to doubles for gsl
    endOffset = min((int)values.size(), endOffset);
    endSearch = min((int)values.size(), endSearch);

    // Try all possible values and grab the one that fits best
    double bestFit = numeric_limits<double>::max();
    double bestFitIx = -1;
    double sumSq;
    double bestsc0, bestsc1;
    /* startSearch = startSearch - startOffset; */
    /* endSearch = endSearch - startOffset; */
    mParam.set_size(2,1);

    for (int i = startSearch; i < endSearch; i++) {
      double firstSS = 0, secondSS = 0;
      double slope1 = 0, slope2 = 0;
      // Fit first portion of line
      mY.set_size(i);
      copy(&values[0], &values[0] + i, mY.begin());
      mX.set_size(i, 2);
      for (size_t xIx = 0; xIx < mX.n_rows; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = xIx;
      }
      mParam = solve(mX, mY);
      slope1 = mParam[1];
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mParam;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	firstSS += (diff * diff);
      }

      // Fit second portion of line
      int rest =  (endOffset- i);
      mY.set_size(rest);
      copy(&values[i], &values[i] + rest, mY.begin());
      mX.set_size(rest, 2);
      for (size_t xIx = 0; xIx < mX.n_rows; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = i+xIx;
      }
      mParam = solve(mX, mY);
      slope2 = mParam[1];
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mParam;
      sumSq = 0;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	sumSq += (diff * diff);
      }
      secondSS = sumSq;
      if (firstSS + secondSS < bestFit && mParam[1] > 1 && slope1 < slope2) {
	bestFit = firstSS + secondSS;
	bestFitIx = i;
	bestsc0 = mParam[0];
	bestsc1 = mParam[1];
      }
    }
    frameIx = bestFitIx;
    yIntercept = 0;
    slope = 0;

    // Ok now polish off the best fit for intercept
    if (reg > 1) {
      mY.set_size(reg);
      mX.set_size(reg,2);
      for (int i = 0; i < reg; i++) { 
	mY[i] = values[i + bestFitIx];
	mX.at(i,0) = 1;
	mX.at(i,1) = frameIx + i;
      }
      mParam = solve(mX, mY);
      mPredicted = mX * mParam;
      sumSq = 0;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	sumSq += (diff * diff);
      }
      sumSqErr = bestFit;

      for (int i = 0; i < reg; i++) { 
	mY[i] = values[frameIx - reg + i];
	mX.at(i,0) = 1;
	mX.at(i,1) = frameIx - reg + i;
      }
      mHingeParam = solve(mX, mY);
      mPredicted = mX * mParam;

      interceptOne = mHingeParam[0];
      slopeOne = mHingeParam[1];
      interceptTwo = mParam[0];
      slopeTwo = mParam[1];

      denom = (slopeTwo - slopeOne);
      lineMeet = (interceptOne - interceptTwo)/denom;

      yIntercept = mParam[0];
      slope = mParam[1];
      frameIx = startSearch;
      if (denom != 0) {
	// frameIx = -1 * yIntercept / slope;
	frameIx = lineMeet;
      }
    }
    if (frameIx < startSearch) {
      frameIx = -1;
      return false;
    }
    if (frameIx > endSearch) {
      frameIx = -1;
      return false;
    }
    return true;
  }

  /** Utility function to print out a column vector from arma. */
  static void PrintVec(const Col<double> &vec) {
    vec.raw_print();
  }

  /** Utility function to print out a matrix from arma. */
  static void PrintVec(const Mat<double> &m) {
    m.raw_print();
  }

  bool findNonZeroRatioChangeIndex(float &frameIx, float &sumSqErr,
				   float &slope, float &yIntercept,
				   int startSearch, int endSearch,
				   double &bestZero, double &bestHinge,
				   std::vector<T> &values, int reg=5) {
    frameIx = -1;
    int windowSize = FRAME_ZERO_WINDOW;
    int segSize = windowSize/2;
    double ssqRatio = -1;
    assert(windowSize % 2 == 0);
    assert((size_t)startSearch < values.size() && startSearch >= 0);
    assert((size_t)endSearch <= values.size() - windowSize && endSearch >= 0);
    sumSqErr = numeric_limits<float>::max();

    double sumSq = 0;
    double bestSlope0 = 0;
    mParam.set_size(2,1);
    for (int i = startSearch; i < endSearch; i++) {
      int seg1Start = i;
      int seg1End = seg1Start+segSize;
      int seg2Start = seg1End;
      int seg2End = seg2Start+segSize;

      // Null model - single line...
      double zeroSsq = 0;
      int size = seg2End - seg1Start;
      mY.set_size(size);
      copy(&values[0]+seg1Start, &values[0]+seg1Start+size, mY.begin()); 
      mX.set_size(size,2);
      for (int xIx = 0; xIx < size; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = seg1Start + xIx;
      }

      mParam = solve(mX, mY);
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mParam;

      for (int s = 0; s < size; s++) {
	double diff = mPredicted[s]  - mY[s];
	zeroSsq += (diff * diff);
      }

      // Hinge model at break point i 
      double hingeSsq = 0;
      size = seg1End - seg1Start;
      mY.set_size(size);
      copy(&values[0]+seg1Start, &values[0]+seg1Start+size, mY.begin()); 
      mX.set_size(size,2);
      mHingeParam.set_size(2,1);
      for (int xIx = 0; xIx < size; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = seg1Start+xIx;
      }
      mHingeParam = solve(mX, mY);
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mHingeParam;			

      for (size_t s1 = 0; s1 < mPredicted.n_rows; s1++) {
	double diff = mPredicted[s1]  - mY[s1];
	hingeSsq += (diff * diff);
      }
      
      mY.set_size(segSize);
      copy(&values[0]+seg2Start, &values[0]+seg2Start +segSize, mY.begin()); 
      mX.set_size(segSize,2);
      for (int xIx = 0; xIx < segSize; xIx++) {
	mX.at(xIx,0) = 1;
	mX.at(xIx,1) = xIx + seg2Start;
      }
      mParam = solve(mX, mY);
      mPredicted.set_size(mX.n_rows);
      mPredicted = mX * mParam;
      sumSq = 0;
      for (size_t rIx = 0; rIx < mPredicted.n_rows; rIx++) {
	double diff = mPredicted[rIx]  - mY[rIx];
	sumSq += (diff * diff);
      }
      hingeSsq += sumSq;
      double currentRatio = zeroSsq / (hingeSsq + .01);
      
      if (currentRatio >= ssqRatio && 
	  mHingeParam[1] < 5 &&    // First part of hinge should be almost flat
	  mHingeParam[1] < mParam[1]) {  // Looking for hinge where first part is flat and second part is steeper than first
	ssqRatio = currentRatio;
	frameIx = seg2Start;
	bestSlope0 = mParam[1];
	bestZero = zeroSsq;
	bestHinge = hingeSsq;
      }
    }
    bool ok = bestSlope0 > .25 && frameIx >= MIN_ALLOWED_FRAME;
    slope = bestSlope0;
    if (ok) {
      ok = findChangeIndex(frameIx, sumSqErr,
                           slope, yIntercept,
                           values,
                           frameIx - MIN_ALLOWED_FRAME, frameIx + MIN_ALLOWED_FRAME,
                           frameIx - MIN_ALLOWED_FRAME, frameIx + MIN_ALLOWED_FRAME,
                           reg);
      if (frameIx < startSearch || frameIx > endSearch) {
	ok = false;
	frameIx = -1;
      }
    }
    else {
      frameIx = -1;
    }
    return ok;
  }
  
 private:
  Col<T> mZeroSlope;
  Col<T> mY;
  Mat<T> mX;
  Col<T> mPredicted;
  Col<T> mParam;
  Col<T> mHingeParam;
};

#endif // FINDSLOPECHANGE_H
