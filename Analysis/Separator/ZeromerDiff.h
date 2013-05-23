/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMERDIFF_H
#define ZEROMERDIFF_H

#include <vector>
#include <algorithm>
#include <armadillo>
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "DiffEqModel.h"
#define ZD_START_FRAME 2
#define ZD_END_FRAME 18
using namespace arma;

/**
 * Class to fit Todd's differential equation model to zeromers
 * and then predict zeromer in other flows based on the reference
 * empty flows
 */
template <class T>
class ZeromerDiff
{

  public:

    // flows is a matrix with rows = frames and cols = flows
    template <class U>
    int FitZeromer (const Mat<T> &wellFlows, const Mat<T> &refFlows,
                    const Col<U> &zeroFlows, const Col<T> &time,
                    Col<T> &param) {
      int length = zeroFlows.n_rows * wellFlows.n_rows;
      mIntegral.set_size (length);
      mSignals.set_size (length,2);
      for (size_t flowIx = 0; flowIx < zeroFlows.n_rows; flowIx++)
      {
        int offset = flowIx*wellFlows.n_rows;
        int zIx = zeroFlows.at (flowIx);
        mIntegral.at (offset) = (refFlows.at (0,zIx) - wellFlows.at (0,zIx));

        mSignals.at (offset, 0) = wellFlows.at (0,zIx);
        mSignals.at (offset, 1) = -1 * refFlows.at (0,zIx);

        for (size_t frameIx = 1; frameIx < wellFlows.n_rows; frameIx++)
        {
          T dTime = time.at (frameIx) - time.at (frameIx-1);
          mIntegral.at (frameIx + offset) = mIntegral.at (frameIx-1 + offset) +
                                            (refFlows.at (frameIx,zIx) - wellFlows.at (frameIx,zIx)) * dTime;
          mSignals.at (frameIx + offset, 0) = wellFlows.at (frameIx, zIx);
          mSignals.at (frameIx + offset, 1) = -1 * refFlows.at (frameIx,zIx);
        }

      }

      param.set_size (2,1);
      //  conceptually doing: param = inv(t(mSignals)*mSignals + t(mRidge) * mRidge) * t(mSignals) * mIntegral
      //  e.g.  param = solve(((trans(mSignals) * mSignals) + mRidge),trans(mSignals)*mIntegral);
      // Add ridge regression
      // cout << "Signals: " << endl;
      // mSignals.raw_print();
      // cout << "Integral: " << endl;
      // mIntegral.raw_print();
      mReg = (trans (mSignals) * mSignals);
      //double traceSum = trace(mReg);
      double k = 5; //0 * zeroFlows.n_rows; // std::max(1.0,fabs((RIDGE_TRACE_K * traceSum)));
      mReg.at (0,0) = mReg.at (0,0) + k;
      mReg.at (1,1) = mReg.at (1,1) + k;
      // For speed manually calculate inverse of R as 2x2
      double inv = 1.0 / (mReg.at (0,0) * mReg.at (1,1) - mReg.at (0,1) * mReg.at (1,0));
      T tmp = mReg.at (0,0);
      mReg.at (0,0) = mReg.at (1,1) * inv;
      mReg.at (1,1) = tmp * inv;
      mReg.at (0,1) = mReg.at (0,1) * -1 * inv;
      mReg.at (1,0) = mReg.at (1,0) * -1 * inv;

      param = mReg * trans (mSignals) * mIntegral;
      if (!param.is_finite())
      {
        return 1;
      }
      return 0;
    }


    template <class U>
    int FitZeromerKnownTau (const Mat<T> &wellFlows, const Mat<T> &refFlows,
                            const Col<U> &zeroFlows, const Col<T> &time,
                            T tauE, T &tauB) {
      tauB = 0;
      int length = zeroFlows.n_rows * wellFlows.n_rows;
      mIntegral.set_size (length);
      mSignals.set_size (length,2);
      SampleStats<double> mTauB;
      double sumX2 = 0;
      double sumXY = 0;
      for (size_t flowIx = 0; flowIx < zeroFlows.n_rows; flowIx++) {
	double previous = 0.0;
	int zIx = zeroFlows.at (flowIx); 
	for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
	  double diff = refFlows.at(frameIx, zIx) - wellFlows.at(frameIx, zIx);
	  double tauES = refFlows.at(frameIx, zIx) * tauE;
	  double Y = previous + diff + tauES;
	  previous += diff;
	  double X = wellFlows.at(frameIx, zIx);
	  sumX2 += X * X;
	  sumXY += X * Y;
	}
      }
      tauB = sumXY / (sumX2 + 1);
      if (!isfinite(tauB)) {
	return 1;
      }
      return 0;
    }

    template <class U>
    int FitZeromerKnownTauPerNuc (const Mat<T> &wellFlows, const Mat<T> &refFlows,
                                  const Col<U> &zeroFlows, const Col<T> &time,
                                  vector<char> &nucFlows, int numNucs,
                                  T tauE, std::vector<T> &tauB) {
      int length = zeroFlows.n_rows * wellFlows.n_rows;
      mIntegral.set_size (length);
      mSignals.set_size (length,2);
      SampleStats<double> mTauB;
      
      double sumX2 = 0;
      double sumXY = 0;
      tauB.resize(numNucs);
      fill(tauB.begin(), tauB.end(), -1);
      // for each nuc do the fit
      for (int nucIx = 0; nucIx < numNucs; nucIx++) {
        for (size_t flowIx = 0; flowIx < zeroFlows.n_rows; flowIx++) {
          double previous = 0.0;
          int zIx = zeroFlows.at (flowIx); 
          /* int minFrame = 0; //min(ZD_START_FRAME, (int)wellFlows.n_rows); */
          /* int maxFrame = wellFlows.n_rows; //min(ZD_END_FRAME, (int)wellFlows.n_rows); */
          int minFrame = min(ZD_START_FRAME, (int)wellFlows.n_rows);
          int maxFrame = min(ZD_END_FRAME, (int)wellFlows.n_rows);
          if (nucFlows[zIx] == nucIx) {
            for (int frameIx = minFrame; frameIx < maxFrame; frameIx++) {
              double diff = refFlows.at(frameIx, zIx) - wellFlows.at(frameIx, zIx);
              double tauES = refFlows.at(frameIx, zIx) * tauE;
              double Y = previous + diff + tauES;
              previous += diff;
              double X = wellFlows.at(frameIx, zIx);
              sumX2 += X * X;
              sumXY += X * Y;
            }
          }
        }
        tauB[nucIx] = sumXY / (sumX2 + 1);
      }
      // Average out for nucs that didn't get fit
      vector<T> temp = tauB;
      for (size_t nucIx = 0; nucIx < tauB.size(); nucIx++) {
        double weight = 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < temp.size(); i++) {
          if (temp[i] > 0) {
            if (i == nucIx) {
              weight += .9;
              sum += .9 * temp[i];
            }  
            else {
              weight += .2;
              sum += .2 * temp[i];
            }
          }
        }
        tauB[nucIx] = sum / weight;
      }
      int retval = 0;
      for (size_t i = 0; i < tauB.size(); i++) {
        if (!isfinite(tauB[i])) {
          retval = 1;
        }
      }
      return retval;
    }

    int PredictZeromer (const Col<T> &ref,
                        const Col<T> &time,
                        const Col<T> &param,
                        Col<T> &zero)
    {
      return PredictZeromer (ref, time, param[0], param[1], zero);
    }

    int PredictZeromer (const Col<T> &ref,
                        const Col<T> &time,
                        T tB, T tE,
                        Col<T> &zero)
    {
      zero.set_size(ref.n_rows);
      //      NewBlueSolveBackgroundTrace(zero.memptr(), ref.memptr(), ref.n_rows, time.memptr(), tB, tE/tB);
      //NewBlueSolveBackgroundTrace(zero.memptr(), ref.memptr(), ref.n_rows, time.memptr(), tB, tB/tE);
      /* zero.set_size (ref.n_rows); */
      /* float etbR = tE/tB; */
      /* float one_over_two_taub = 1.0f/(2.0f *tB); */
      /* float one_over_one_plus_aval = 0.0f */
      /* float xt = (time.at(0)/2.0f) * one_over_two_taub; */
      
      zero.at (0) = ref.at (0);
      T cdelta = 0;
      for (size_t frameIx = 1; frameIx < ref.n_rows; frameIx++) {
        T dTime = time.at(frameIx); // time.at (frameIx) - time.at (frameIx-1);
      	//	zero.at(frameIx) = (ref.at(frameIx) * (tE + dTime) + cdelta)/(tB + dTime);
        zero.at (frameIx) = (ref.at (frameIx) * ( (tE+dTime) /tB) + cdelta/tB) / (1+dTime/tB);
        cdelta = cdelta + (ref.at (frameIx) - zero.at (frameIx));
      }
      /* return 0; */
      return (0);
    }

  private:
    // Keeping these as member variables helps to avoid reallocations
    Mat<T> mReg;
    Col<T> mIntegral;
    Mat<T> mSignals;
};

#endif // ZEROMERDIFF_H
