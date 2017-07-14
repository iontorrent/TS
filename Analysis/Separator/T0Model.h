/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef T0MODEL_H
#define T0MODEL_H

#include <assert.h>
#include <vector>
#include <limits>
#include <armadillo>

using namespace std;
using namespace arma;

/** 
 * Class to estimate sigma and t_mid_nuc parameters for background
 * model based off of the t0 of a dat and the slope of the nuc rise
 * after t0.
 */
class SigmaTMidNucEstimation {

public:  
  /** Default constructor. */
  SigmaTMidNucEstimation() {
    mRateSigmaIntercept = 0.0f;
    mRateSigmaSlope = 0.0f;
    mTMidIntercept = 0.0f;
    mTMidSlope = 0.0f;
  }
  
  /** Set our coefficient paramters. */
  void Init(float sigma_intercept, float sigma_slope, float tmid_intercept, float tmid_slope, float frameRate) {
    mRateSigmaSlope = sigma_slope;
    mRateSigmaIntercept = sigma_intercept;
    mTMidIntercept = tmid_intercept;
    mTMidSlope = tmid_slope;
    mFrameRate = frameRate;
  }

  /* t0 in milliseconds, sigma and tmidnuc in frames as that is what background model wants */
  void Predict(int t0, float slope, float &sigma, float &tmidnuc) {
    assert(t0 > 0);
    assert(slope > 0);
    // the .1 addition is to stabilize very low slopes. */
    sigma = (1.1f)/(slope+.1) * mRateSigmaSlope + mRateSigmaIntercept;
    sigma = max(0.1f, sigma);
    sigma = min(8.1f, sigma);
    tmidnuc = (t0 / mFrameRate) * mTMidSlope + mTMidIntercept;
    //    tmidnuc = (t0 * 1/(mFrameRate)) * (1/mTMidSlope) - mTMidIntercept;
  }

private:
  /* Slope an intercept terms for sigma and t_mid_nuc estimation. */
  float mRateSigmaIntercept;
  float mRateSigmaSlope;
  float mTMidIntercept;
  float mTMidSlope;
  float mFrameRate;
};

class SigmaEst {

public:
  SigmaEst() {
    mTMidNuc = 0.0f;
    mSigma = 0.0f;
    mT0 = 0.0f;
    mRate = 0.0f;
  }

  float mRate;
  float mT0;
  float mTMidNuc;
  float mSigma;
};

/** Simple linear model y = m * x + b */
class LineModel {

public:

  /** Compute a fit given the y and x values. */
  void FitModel(const float *yStart, const float *yEnd, float *xStart, float *xEnd) {
    SetSize(yEnd - yStart);
    assert(yEnd - yStart == xEnd - xStart);
    FillY(0, mY.n_rows, yStart);
    FillX(0, mX.n_rows, xStart);
    Fit();
  }

  /** Print to stdout values used. */
  void Dump() {
    mY.raw_print("Y");
    mX.raw_print("X");
    mParam.raw_print("mParam");
    mPrediction.raw_print("mPrediction");
    mDiff.raw_print("mDiff");
  }

  /** How many data points are we looking at for this segment. */
  void SetSize(size_t n) {
    mY.set_size(n);
    mX.set_size(n,2);
    std::fill(mX.begin_col(0), mX.end_col(0), 1.0);
  }

  /** Fill the X values from index start to index end with values. */
  void FillX(size_t start, size_t end, const float *values) {
    for (size_t i = start; i < end; i++) {
      mX.at(i,1) = *(values++);
    }
  }

  /** Fill the Y values from index start to index end with the values. */
  void FillY(size_t start, size_t end, const float *values) {
    for (size_t i = start; i < end; i++) {
      mY.at(i) = *(values++);
    }
  }

  /** Fit the slope and intercept using least squares. */
  void Fit() {
    mParam = solve(mX,mY);
    mPrediction = mX * mParam;
    mDiff = mY - mPrediction;
  }

  /** Get the values predicted by this model. */
  void GetPrediction(Col<float> &predict) const {
    predict = mPrediction;
  }

  /** Return the estimate of the slope */
  float GetSlope() const { return mParam.at(1); }

  /** Return the estimate of the y-intercept */
  float GetY0() const { return mParam.at(0); }

  /** Return the sum of squared difference errors. */
  float SumSqErr() const {
    float sum = accu(mDiff % mDiff);
    return sum;
  }

private:  
  Col<float> mY; ///< values on Y axis
  Mat<float> mX; ///< values on X axis and intercept term
  Col<float> mDiff; ///< Difference between mY and mPrediction
  Col<float> mPrediction; ///< Values predicted by model
  Mat<float> mParam; ///< Slope and intercept
  
};

/** Model that represents to linear models stuck together like a hinge
    of two straight lines. */
class HingeModel {

public:
  /** Fit the second part of the hinge. */
  void FitFirst(const float *yStart, const float *yEnd,
                float *xStart, float *xEnd) {
    mFirst.FitModel(yStart,yEnd,xStart,xEnd);
  }

  /** Fit the second portion of the hinge. */
  void FitSecond(const float *yStart, const float *yEnd,
                float *xStart, float *xEnd) {
    mSecond.FitModel(yStart,yEnd,xStart,xEnd);
  }

  /** Get part of hinge model */
  const LineModel &GetFirst() const { return mFirst; }

  /** Get second part of hinge model. */
  const LineModel &GetSecond() const { return mSecond; }

  /** Total squared error for combined model. */
  float SumSqErr() const { return mFirst.SumSqErr() + mSecond.SumSqErr(); }
  
  /** Calculate t0 which is x at point where lines meet. */
  float GetT0Est() const {
    float linesMeet = (mFirst.GetY0() - mSecond.GetY0())/(mSecond.GetSlope() - mFirst.GetSlope());
    return linesMeet;
  }
  
private:
  LineModel mFirst; ///< First section of the hinge
  LineModel mSecond; ///< Seconde section of the hinge
};

/** Putative t0 site. */
class T0Hypothesis {

public:

  /* Fit a model from the x and y coordinates. */
  void FitModel(float *yStart, float *xStart) {
    /* First setup and fit the simple linear model. */
    mLineModel.SetSize((mFirstRange[1]-mFirstRange[0]) + (mSecondRange[1]-mSecondRange[0]));
    mLineModel.FillY(0,(mFirstRange[1]-mFirstRange[0]),yStart+mFirstRange[0]);
    mLineModel.FillY((mFirstRange[1]-mFirstRange[0]), (mFirstRange[1]-mFirstRange[0])+(mSecondRange[1]-mSecondRange[0]),yStart+mSecondRange[0]);
    mLineModel.FillX(0,(mFirstRange[1]-mFirstRange[0]),xStart+mFirstRange[0]);
    mLineModel.FillX((mFirstRange[1]-mFirstRange[0]), (mFirstRange[1]-mFirstRange[0])+(mSecondRange[1]-mSecondRange[0]),xStart+mSecondRange[0]);
    mLineModel.Fit();
    mHingeModel.FitFirst(yStart + mFirstRange[0], yStart + mFirstRange[1], 
                         xStart + mFirstRange[0], xStart + mFirstRange[1]);
    mHingeModel.FitSecond(yStart + mSecondRange[0], yStart + mSecondRange[1], 
                          xStart + mSecondRange[0], xStart + mSecondRange[1]);
  }

  float GetHingeSlope() const {
    return mHingeModel.GetSecond().GetSlope();
  }

  /** Are the slope constraints satisfied for this fit. */
  bool ConstraintsOk() {
    bool firstOk =  (mHingeModel.GetFirst().GetSlope() >= mFirstSlopeRange[0] && 
                      mHingeModel.GetFirst().GetSlope() <= mFirstSlopeRange[1]);
    bool secondOk =  (mHingeModel.GetSecond().GetSlope() >= mSecondSlopeRange[0] && 
                       mHingeModel.GetSecond().GetSlope() <= mSecondSlopeRange[1]);
    return firstOk && secondOk;
  }

  /** Get the ratio of ssq for a hinge and a regular model */
  float GetSsqRatio() { return mLineModel.SumSqErr() / mHingeModel.SumSqErr(); }

  /** Get the difference of ssq for a hinge and a regular model */  
  float GetSsqDiff() { return mLineModel.SumSqErr() -  mHingeModel.SumSqErr(); }
  
  /** Get the estimate of t0 from the hinge model. */
  float GetT0Est() { return mHingeModel.GetT0Est(); }

  /** Set the range for the first portion of fit. */
  void SetFirstRange(float minRange, float maxRange) {
    assert(minRange < maxRange);
    mFirstRange.resize(2);
    mFirstRange[0] = minRange;
    mFirstRange[1] = maxRange;
  }

  /** Set the range for the second portion of fit. */
  void SetSecondRange(float minRange, float maxRange) {
    assert(minRange < maxRange);
    mSecondRange.resize(2);
    mSecondRange[0] = minRange;
    mSecondRange[1] = maxRange;
  }  

  /** Set the constraints on slope for first portion of line. */
  void SetFirstSlopeRange(float minSlope, float maxSlope) {
    assert(minSlope < maxSlope);
    mFirstSlopeRange.resize(2);
    mFirstSlopeRange[0] = minSlope;
    mFirstSlopeRange[1] = maxSlope;
  }

  /** Set the constraints on slope for second portion of line. */
  void SetSecondSlopeRange(float minSlope, float maxSlope) {
    assert(minSlope < maxSlope);
    mSecondSlopeRange.resize(2);
    mSecondSlopeRange[0] = minSlope;
    mSecondSlopeRange[1] = maxSlope;
  }

  LineModel mLineModel; ///< Model for simple line
  HingeModel mHingeModel; ///< Model for combination of two lines
  std::vector<int> mFirstRange; ///< X range for first portion of line
  std::vector<int> mSecondRange; ///< X range for second portion of line
  std::vector<float> mFirstSlopeRange; ///< Constraints on slope for first portion of line
  std::vector<float> mSecondSlopeRange; ///< Constraints on slope for second portion of line
};

/** Class for looking through average traces and finding the point at which nuc hits the wells. */
class T0Finder {

public:
  /** Set the constraints on slope for first portion of line. */
  void SetFirstSlopeRange(float minSlope, float maxSlope) {
    assert(minSlope < maxSlope);
    mFirstSlopeRange.resize(2);
    mFirstSlopeRange[0] = minSlope;
    mFirstSlopeRange[1] = maxSlope;
  }

  /** Set the constraints on slope for second portion of line. */
  void SetSecondSlopeRange(float minSlope, float maxSlope) {
    assert(minSlope < maxSlope);
    mSecondSlopeRange.resize(2);
    mSecondSlopeRange[0] = minSlope;
    mSecondSlopeRange[1] = maxSlope;
  }

  /** Set the index range of x values to search. */
  void SetSearchRange(int start, int end) {
    assert(start < end);
    mSearchRange.resize(2);
    mSearchRange[0] = start;
    mSearchRange[1] = end;
  }
  
  /** Set the window size to search on each arm of hinge. */
  void SetWindowSize(int size) {
    mWindowSize = size;
  }

  /** Look at trace data provided by Y and find when nuc hits the wells */
  bool FindT0(float *Y, size_t size) {
    mX.resize(size);
    for (size_t i = 0; i < mX.size(); i++) {
      mX[i] = i;
    }
    T0Hypothesis t0;
    t0.SetFirstSlopeRange(mFirstSlopeRange[0], mFirstSlopeRange[1]);
    t0.SetSecondSlopeRange(mSecondSlopeRange[0], mSecondSlopeRange[1]);
    size_t start = max(mSearchRange[0], mWindowSize);
    size_t end = min(mSearchRange[1], size-mWindowSize-1);
    bool first = true;
    bool found = false;
    for (size_t i = start; i < end; i++) {
      // Leave out the middle point as it is always a transition
      t0.SetFirstRange(i - mWindowSize, i);
      t0.SetSecondRange(i+1, i + mWindowSize + 1);
      t0.FitModel(Y, &mX[0]);
      if (first) {
        mBestT0 = t0;
        first =false;
      }
      if (t0.ConstraintsOk() && t0.GetSsqDiff() > mBestT0.GetSsqDiff()) {
        float est = t0.GetT0Est();
        bool inRange = est > mX.front() && est < mX.back();
        if (inRange) {
          found = true;
          mBestT0 = t0;
        }
      }
    }
    
    return found;
  }

  /** Look at trace data provided by Y and find when nuc hits the wells */
  bool FindT0Time(float *Y, float *X, size_t size, FILE* fdbg) {
    mX.resize(size);
    for (size_t i = 0; i < mX.size(); i++) {
      mX[i] = X[i];
    }
    T0Hypothesis t0;
    t0.SetFirstSlopeRange(mFirstSlopeRange[0], mFirstSlopeRange[1]);
    t0.SetSecondSlopeRange(mSecondSlopeRange[0], mSecondSlopeRange[1]);
    size_t start = max(mSearchRange[0], mWindowSize);
    size_t end = min(mSearchRange[1], size-mWindowSize-1);
    bool first = true;
    bool found = false;
    double maxSsqDiff = 0.0;
    for (size_t i = start; i < end; i++) {
      // Leave out the middle point as it is always a transition
      t0.SetFirstRange(i - mWindowSize, i);
      t0.SetSecondRange(i+1, i + mWindowSize + 1);
      t0.FitModel(Y, &mX[0]);
      if( fdbg ){
          bool firstOk =  (t0.mHingeModel.GetFirst().GetSlope() >= mFirstSlopeRange[0] &&
                            t0.mHingeModel.GetFirst().GetSlope() <= mFirstSlopeRange[1]);
          bool secondOk =  (t0.mHingeModel.GetSecond().GetSlope() >= mSecondSlopeRange[0] &&
                             t0.mHingeModel.GetSecond().GetSlope() <= mSecondSlopeRange[1]);
        fprintf(fdbg, "%g %g %g %g %g %d %d\n", t0.GetT0Est()/66., 66.*t0.mHingeModel.GetFirst().GetSlope(), 66.*t0.mHingeModel.GetSecond().GetSlope(), t0.GetSsqDiff(), t0.GetSsqRatio(), firstOk, secondOk );
      }
      if (first) {
        mBestT0 = t0;
        maxSsqDiff = 0.0;
        first =false;
      }
      if (t0.ConstraintsOk() && t0.GetSsqDiff() > maxSsqDiff) {
        float est = t0.GetT0Est();
        bool inRange = est >= mX.front() && est <=  mX.back();
        if (inRange) {
          found = true;
          mBestT0 = t0;
          maxSsqDiff = mBestT0.GetSsqDiff();
        }
      }
    }
    if( fdbg )
        fprintf(fdbg, "%g %g %g %g %g\n", mBestT0.GetT0Est()/66., 66.*mBestT0.mHingeModel.GetFirst().GetSlope(), 66.*mBestT0.mHingeModel.GetSecond().GetSlope(), mBestT0.GetSsqDiff(), mBestT0.GetSsqRatio() );
    return found;
  }

  /** Get the best estimate of when nuc hits the trace. */
  float GetT0Est() {
    return mBestT0.GetT0Est();
  }

  float GetSlope() const {
    return mBestT0.GetHingeSlope();
  }


private:
  size_t mWindowSize; ///< What size window to use for each arm of hinge.
  T0Hypothesis mBestT0; ///< Best t0 hypothesis in terms of sum of square distance
  std::vector<float> mX; ///< X axis values for traces
  std::vector<size_t> mSearchRange; ///< What range of x values to search over
  
  std::vector<float> mFirstSlopeRange; ///< Constraints on slope for first portion of line
  std::vector<float> mSecondSlopeRange; ///< Constraints on slope for second portion of line

};

#endif // T0MODEL_H
