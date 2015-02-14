/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FINDSLOPECHANGE_H
#define FINDSLOPECHANGE_H

#include <vector>
#include <limits>
#include <armadillo>

#define FRAME_ZERO_WINDOW 20
#define MIN_ALLOWED_FRAME 12
#define FSC_RATIO_STABLE 5
using namespace std;
using namespace arma;

template <class T>
class FindSlopeChange
{
public:
  FindSlopeChange()
  {
    mWindowSize = FRAME_ZERO_WINDOW/2;
    mMaxFirstHingeSlope = 5;
    mMinFirstHingeSlope = -10;
  }

  void SetWindowSize ( int size )
  {
    mWindowSize = size;
  }
  void SetMaxFirstHingeSlope ( float maxSlope )
  {
    mMaxFirstHingeSlope = maxSlope;
  }
  void SetMinFirstHingeSlope ( float minSlope )
  {
    mMinFirstHingeSlope = minSlope;
  }

  /** reg is the regression to do after break point is found. */
  bool findNonZeroChangeIndex ( float &frameIx, float &sumSqErr,
                                float &slope, float &yIntercept,
                                std::vector<T> &values, int startOffset,
                                int endOffset, int startSearch, int endSearch,
                                int reg=5 )
  {
    if ( startOffset > 2 ) {}
    frameIx = -1;
    sumSqErr = numeric_limits<float>::max();
    // Convert to doubles for gsl
    endOffset = min ( ( int ) values.size(), endOffset );
    endSearch = min ( ( int ) values.size(), endSearch );

    // Try all possible values and grab the one that fits best
    double bestFit = numeric_limits<double>::max();
    double bestFitIx = -1;
    double sumSq;
    double bestsc0, bestsc1;
    /* startSearch = startSearch - startOffset; */
    /* endSearch = endSearch - startOffset; */
    mParam.set_size ( 2,1 );

    for ( int i = startSearch; i < endSearch; i++ )
    {
      double firstSS = 0, secondSS = 0;

      for ( int yIx = 0; yIx < i; yIx++ )
      {
        firstSS += values[yIx] * values[yIx] ;
      }
      // Fit second portion of line
      int rest = ( endOffset- i );
      mY.set_size ( rest );
      copy ( &values[i], &values[i] + rest, mY.begin() );
      mX.set_size ( rest, 2 );
      for ( size_t xIx = 0; xIx < mX.n_rows; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = i+xIx;
      }
      mParam = solve ( mX, mY );
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mParam;
      sumSq = 0;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        sumSq += ( diff * diff );
      }
      secondSS = sumSq;
      if ( firstSS + secondSS < bestFit && mParam[1] > 0 )
      {
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
    if ( reg > 1 )
    {
      mY.set_size ( reg );
      mX.set_size ( reg,2 );
      for ( int i = 0; i < reg; i++ )
      {
        mY[i] = values[i + frameIx];
        mX.at ( i,0 ) = 1;
        mX.at ( i,1 ) = frameIx + i;
      }
      mParam = solve ( mX, mY );
      mPredicted = mX * mParam;

      sumSq = 0;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        sumSq += ( diff * diff );
      }
      sumSqErr = bestFit;

      for ( int i = 0; i < reg; i++ )
      {
        mY[i] = values[frameIx - i];
        mX.at ( i,0 ) = 1;
        mX.at ( i,1 ) = frameIx - i;
      }
      mHingeParam = solve ( mX, mY );
      mPredicted = mX * mParam;

      double denom = ( mParam[1] - mHingeParam[1] );
      double xx = ( mHingeParam[0] - mParam[0] ) / ( mParam[1] - mHingeParam[1] );
      yIntercept = mParam[0];
      slope = mParam[1];
      frameIx = startSearch;
      if ( denom != 0 )
      {
        // frameIx = -1 * yIntercept / slope;
        frameIx = xx;
      }
    }
    if ( frameIx < startSearch )
    {
      frameIx = -1;
      return false;
    }
    if ( frameIx > endSearch )
    {
      frameIx = -1;
      return false;
    }
    return true;
  }

  /** reg is the regression to do after break point is found. */
  bool findChangeIndex ( float &frameIx, float &sumSqErr,
                         float &slope, float &yIntercept,
                         std::vector<T> &values, int startOffset,
                         int endOffset, int startSearch, int endSearch,
                         int mult = 1, int reg=5 )
  {
    //    printf("start: %d, end: %d\n", startSearch, endSearch);
    double denom = 0, lineMeet = 0;
    double slopeOne = 0, interceptOne = 0;
    double slopeTwo = 0, interceptTwo = 0;
    sumSqErr = numeric_limits<float>::max();
    // Convert to doubles for gsl
    endOffset = min ( ( int ) values.size(), endOffset );
    endSearch = min ( ( int ) values.size(), endSearch );

    // Try all possible values and grab the one that fits best
    double bestFit = numeric_limits<double>::max();
    double bestFitIx = -1;
    double sumSq;
    //double bestsc0, bestsc1;
    /* startSearch = startSearch - startOffset; */
    /* endSearch = endSearch - startOffset; */
    mParam.set_size ( 2,1 );

    for ( int i = startSearch; i < endSearch; i++ )
    {
      double firstSS = 0, secondSS = 0;
      double slope1 = 0, slope2 = 0;
      // Fit first portion of line
      mY.set_size ( i );
      copy ( &values[0], &values[0] + i, mY.begin() );
      mX.set_size ( i, 2 );
      for ( size_t xIx = 0; xIx < mX.n_rows; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = xIx;
      }
      mParam = solve ( mX, mY );
      Col<T> mParamTmp = mParam;
      slope1 = mParam[1];
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mParam;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        firstSS += ( diff * diff );
      }

      // Fit second portion of line
      int rest = ( endOffset- i );
      mY.set_size ( rest );
      copy ( &values[i], &values[i] + rest, mY.begin() );
      mX.set_size ( rest, 2 );
      for ( size_t xIx = 0; xIx < mX.n_rows; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = i+xIx;
      }

      mParam = solve ( mX, mY );
      slope2 = mParam[1];
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mParam;
      sumSq = 0;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        sumSq += ( diff * diff );
      }
      secondSS = sumSq;
      /* printf("FindChangeIndex: %d:%d %d:%d (%.5f,%.5f) (%.5f,%.5f) %.5f %.5f %.5f\n",  */
      /*        startSearch, i, i, endOffset- i, */
      /*        mParamTmp[0], mParamTmp[1], */
      /*        mParam[0], mParam[1], */
      /*        firstSS, secondSS, (firstSS + secondSS)); */
      if ( firstSS + secondSS < bestFit && mult * mParam[1] > 1 && slope1 < mult * slope2 )
      {
        bestFit = firstSS + secondSS;
        bestFitIx = i;
        //bestsc0 = mParam[0];
        //bestsc1 = mParam[1];
      }
    }
    frameIx = bestFitIx;
    yIntercept = 0;
    slope = 0;

    // Ok now polish off the best fit for intercept
    if ( reg > 1 && bestFitIx + reg < values.size() && bestFitIx >= 0 )
    {
      mY.set_size ( reg );
      mX.set_size ( reg,2 );
      for ( int i = 0; i < reg; i++ )
      {
        mY[i] = values[i + bestFitIx];
        mX.at ( i,0 ) = 1;
        mX.at ( i,1 ) = frameIx + i;
      }
      mParam = solve ( mX, mY );
      mPredicted = mX * mParam;
      sumSq = 0;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        sumSq += ( diff * diff );
      }
      sumSqErr = bestFit;

      for ( int i = 0; i < reg; i++ )
      {
        mY[i] = values[frameIx - reg + i];
        mX.at ( i,0 ) = 1;
        mX.at ( i,1 ) = frameIx - reg + i;
      }
      mHingeParam = solve ( mX, mY );
      mPredicted = mX * mParam;

      interceptOne = mHingeParam[0];
      slopeOne = mHingeParam[1];
      interceptTwo = mParam[0];
      slopeTwo = mParam[1];

      denom = ( slopeTwo - slopeOne );
      lineMeet = ( interceptOne - interceptTwo ) /denom;

      yIntercept = mParam[0];
      slope = mParam[1];
      frameIx = startSearch;
      /* printf("Finished: %.5f:%.5f %.5f:%.5f (%.5f,%.5f) (%.5f,%.5f) %.5f\n",  */
      /*        frameIx-reg, frameIx, frameIx, frameIx+reg, */
      /*        mHingeParam[0], mHingeParam[1], */
      /*        mParam[0], mParam[1], */
      /*        lineMeet); */
      if ( denom != 0 )
      {
        // frameIx = -1 * yIntercept / slope;
        frameIx = lineMeet;
      }
    }
    if ( frameIx < startSearch )
    {
      frameIx = -1;
      return false;
    }
    if ( frameIx > endSearch )
    {
      frameIx = -1;
      return false;
    }
    return true;
  }

  /** Utility function to print out a column vector from arma. */
  static void PrintVec ( const Col<double> &vec )
  {
    vec.raw_print();
  }

  /** Utility function to print out a matrix from arma. */
  static void PrintVec ( const Mat<double> &m )
  {
    m.raw_print();
  }

  bool findNonZeroRatioChangeIndex ( float &frameIx, float &sumSqErr,
                                     float &slope, float &yIntercept,
                                     int startSearch, int endSearch,
                                     T &bestZero, T &bestHinge,
                                     std::vector<T> &values, int mult = 1, int reg=5, double minSlope=0.0 )
  {
    frameIx = -1;
    int windowSize = mWindowSize * 2;
    int segSize = windowSize/2;
    double ssqRatio = -1;
    assert ( windowSize % 2 == 0 );
    assert ( ( size_t ) startSearch < values.size() && startSearch >= 0 );
    //    assert((size_t)endSearch <= values.size() - windowSize && endSearch >= 0);
    endSearch = min ( endSearch, ( int ) ( values.size() - windowSize ) );
    sumSqErr = numeric_limits<float>::max();

    double sumSq = 0;
    double bestSlope0 = 0;
    mParam.set_size ( 2,1 );
    for ( int i = startSearch; i < endSearch; i++ )
    {
      int seg1Start = i;
      int seg1End = seg1Start+segSize;
      int seg2Start = seg1End;
      int seg2End = seg2Start+segSize;

      // Null model - single line...
      double zeroSsq = 0;
      int size = seg2End - seg1Start;
      mY.set_size ( size );
      copy ( &values[0]+seg1Start, &values[0]+seg1Start+size, mY.begin() );
      mX.set_size ( size,2 );
      for ( int xIx = 0; xIx < size; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = seg1Start + xIx;
      }

      mParam = solve ( mX, mY );
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mParam;

      for ( int s = 0; s < size; s++ )
      {
        double diff = mPredicted[s]  - mY[s];
        zeroSsq += ( diff * diff );
      }

      // Hinge model at break point i
      double hingeSsq = 0;
      size = seg1End - seg1Start;
      mY.set_size ( size );
      copy ( &values[0]+seg1Start, &values[0]+seg1Start+size, mY.begin() );
      mX.set_size ( size,2 );
      mHingeParam.set_size ( 2,1 );
      for ( int xIx = 0; xIx < size; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = seg1Start+xIx;
      }
      mHingeParam = solve ( mX, mY );
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mHingeParam;

      for ( size_t s1 = 0; s1 < mPredicted.n_rows; s1++ )
      {
        double diff = mPredicted[s1]  - mY[s1];
        hingeSsq += ( diff * diff );
      }

      mY.set_size ( segSize );
      copy ( &values[0]+seg2Start, &values[0]+seg2Start +segSize, mY.begin() );
      mX.set_size ( segSize,2 );
      for ( int xIx = 0; xIx < segSize; xIx++ )
      {
        mX.at ( xIx,0 ) = 1;
        mX.at ( xIx,1 ) = xIx + seg2Start;
      }
      mParam = solve ( mX, mY );
      mPredicted.set_size ( mX.n_rows );
      mPredicted = mX * mParam;
      sumSq = 0;
      for ( size_t rIx = 0; rIx < mPredicted.n_rows; rIx++ )
      {
        double diff = mPredicted[rIx]  - mY[rIx];
        sumSq += ( diff * diff );
      }
      hingeSsq += sumSq;
      double currentRatio = ( FSC_RATIO_STABLE + zeroSsq ) / ( FSC_RATIO_STABLE + ( hingeSsq + .01 ) ); /// @todo - better value than 5 to steady ratio?
      double hingeSlope = mHingeParam[1];
      double paramSlope = mParam[1];
      /* printf("%d:%d %d:%d (%.5f,%.5f) (%.5f,%.5f) %.5f %.5f %.5f\n",  */
      /*        seg1Start,seg1End, seg2Start,seg2End, */
      /*        mHingeParam[0], mHingeParam[1], */
      /*        mParam[0], mParam[1], */
      /*        zeroSsq, hingeSsq, currentRatio); */
      if ( currentRatio >= ssqRatio &&
           hingeSlope < mMaxFirstHingeSlope &&    // First part of hinge should be almost flat
           hingeSlope > mMinFirstHingeSlope &&
           fabs ( paramSlope ) > minSlope &&
           mHingeParam[1] < mult * paramSlope )   // Looking for hinge where first part is flat and second part is steeper than first
      {
        ssqRatio = currentRatio;
        frameIx = seg2Start;
        bestSlope0 = mParam[1];
        bestZero = zeroSsq;
        bestHinge = hingeSsq;
      }
    }
    bool ok = mult * bestSlope0 > .25 && frameIx >= MIN_ALLOWED_FRAME;
    slope = bestSlope0;
    if ( ok )
    {

      ok = findChangeIndex ( frameIx, sumSqErr,
                             slope, yIntercept,
                             values,
                             frameIx - MIN_ALLOWED_FRAME/2, frameIx + MIN_ALLOWED_FRAME/2,
                             frameIx - MIN_ALLOWED_FRAME/2, frameIx + MIN_ALLOWED_FRAME/2,
                             mult,
                             reg );
      if ( frameIx < startSearch || frameIx > endSearch )
      {
        ok = false;
        frameIx = -1;
      }
    }
    else
    {
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
  int mWindowSize;
  float mMaxFirstHingeSlope;
  float mMinFirstHingeSlope;
};

#endif // FINDSLOPECHANGE_H
