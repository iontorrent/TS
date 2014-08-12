/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMERMODELBULK_H
#define ZEROMERMODELBULK_H

#include <vector>
#include <algorithm>
#include <armadillo>
#include <iostream>
#include <string>
#include <fstream>
#include "Utils.h"
#include "Traces.h"
#include "KeyClassifier.h"
#include "ZeromerDiff.h"
#include "IonErr.h"
#include "SampleQuantiles.h"
#include "ZeromerModel.h"
#include "ReservoirSample.h"
#include "PJobQueue.h"


using namespace arma;
using namespace std;
#define NUM_ZMODELBULK_FRAMES 22
#define NUM_ZMODELBULK_START 0
/**
 * Class holding the result of fitting a Nuc specific bulk stabilized
 * differential equation.
 */
class KeyBulkFit
{

  public:

    KeyBulkFit()
    {
      wellIdx = -1;
      keyIdx = -1;
      ok = -1;
      param.set_size (4,2);
      std::fill (param.begin(), param.end(), 0.0f);
      ssq = -1;
    }
    int wellIdx; ///< Well index on chip.
    int8_t keyIdx; ///< Which key did this well match. 0 for library, 1 for tf, -1 for no fit
    char ok;   ///< fit went ok? 1 for good, 0 for bad
    float ssq; ///< sum of squares difference for the zeromers fit
    /**
     * 4x2 Nuc by buffering matrix with col 0 being tauB (bead buffer) and
     * col 1 being tauE (bulk or reference buffering). The rows are in the order
     * of the TraceStore::Nuc enum (alphanumeric).
     */
    Mat<float> param;

};


template<typename T>
class TauEBulkErr {

public:
    static double PointFcn (const Col<T> &trace,
                            const Col<T> &traceIntegral,
                            const Col<T> &weights,
                            const Col<T> &bulk,
                            int nucIx,
                            double &ssq,
			    double &err,
                            double &tauB)
    {
      Col<T> bulkDiff = bulk - traceIntegral;
      double num = dot (bulkDiff, trace);
      double denom = (dot (trace, trace) +.01);
      //      ION_ASSERT(isfinite(num), "numerator must be finite.");
      //      ION_ASSERT(denom != 0 && isfinite(denom), "denominator must be non-zero and finite.");
      double b = num/denom;
      //      double b = dot (bulkDiff, trace) / (dot (trace, trace) +.01);
      b = (b + fabs (b)) /2.0;
      Col<T> prediction = traceIntegral + b * trace;
      prediction = prediction - bulk;
      err = mean(prediction - trace);
      prediction = prediction % prediction;
      ssq = accu (prediction % weights);
      tauB = b;
      return isfinite(ssq);
    }
  
    static double CalcTotalFit (TraceStore &store,
                                std::vector<size_t> &wells,
                                const Col<T> &weights,
                                size_t flowIx,
                                size_t nucIx,
                                double bulkTau,
				double &wellsTauB,
				double &wellsSsq,
				double &wellsErr)
    {
      double totalSsq = 0;
      int goodWells = 0;
      int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      //int numFrames = store.GetNumFrames();
      Col<T> trace (store.GetNumFrames());
      Col<T> traceIntegral (numFrames);
      Col<T> bulk (numFrames);
      Col<T> bulkTrace (store.GetNumFrames());
      Col<T> bulkIntegral (numFrames);
      SampleStats<double> meanTauB;
      SampleStats<double> meanSsq;
      SampleStats<double> meanErr;
      for (size_t i = 0; i < wells.size(); i++)
      {
        size_t wellIx = wells[i];
        store.GetTrace (wellIx, flowIx, trace.begin());
        store.GetReferenceTrace (wellIx, flowIx,  bulkTrace.begin());
        trace.set_size(numFrames);
        bulkTrace.set_size(numFrames);
        bulkIntegral = cumsum (bulkTrace);
        bulk = bulkIntegral + bulkTau * bulkTrace;
        traceIntegral = cumsum (trace);
        double tauB = 0;
        double ssq = 0;
	double err = 0;
        double sumWeight = 0;
        for (size_t fIx = 0; fIx < bulkTrace.n_rows; fIx++) {
          ION_ASSERT(isfinite(trace[fIx]), "trace not finite.");
          ION_ASSERT(isfinite(traceIntegral[fIx]), "traceIntegral not finite.");
          ION_ASSERT(isfinite(weights[fIx]), "weights not finite.");
          ION_ASSERT(isfinite(bulk[fIx]), "bulk not finite.");
          ION_ASSERT(isfinite(bulkTrace[fIx]), "bulkTrace not finite.");
          ION_ASSERT(isfinite(bulkIntegral[fIx]), "bulkIntegral not finite.");
          sumWeight += weights[fIx];
        }
        ION_ASSERT(sumWeight > 0, "sum of weight must be positive.");
        bool ok = PointFcn (trace, traceIntegral, weights, bulk, nucIx, ssq, err, tauB);
        if (ok) {
	  meanSsq.AddValue(ssq);
	  meanTauB.AddValue(tauB);
	  meanErr.AddValue(err);
          totalSsq += log (ssq + 1.0);
          goodWells++;
        }
      }
      if (goodWells == 0) {
        return std::numeric_limits<double>::max();
      }
      wellsTauB = meanTauB.GetMean();
      wellsSsq  = meanSsq.GetMean();
      wellsErr = meanErr.GetMean();
      return totalSsq / goodWells;
    }


  static void FillInData(TraceStore &traceStore,
			 Mat<float> &_wellFlows,
			 Mat<float> &_refFlows,
			 Mat<float> &_predicted,
			 int nFlows,
			 size_t wellIdx) {
    Col<float> trace;
    int minFrame = min(NUM_ZMODELBULK_START, (int)traceStore.GetNumFrames());
    //    int maxFrame = traceStore.GetNumFrames();
    int maxFrame = min(NUM_ZMODELBULK_FRAMES, (int)traceStore.GetNumFrames());
    // int minFrame = max(traceStore.GetT0(wellIdx)-4, 0.0f);
    // int maxFrame = min(minFrame + 25.0f, (float)traceStore.GetNumFrames());
    int nFrames = maxFrame - minFrame;
    if (_wellFlows.n_rows != (size_t)nFrames || _wellFlows.n_cols != (size_t)nFlows) {
      _wellFlows.set_size(nFrames, nFlows);
      _refFlows.set_size(nFrames, nFlows);
      _predicted.set_size(nFrames, nFlows);
    }
    
    trace.resize(traceStore.GetNumFrames());
    std::fill(_refFlows.begin(), _refFlows.end(), 0);
    std::fill(_wellFlows.begin(), _wellFlows.end(), 0);
    
    for (size_t flowIx = 0; flowIx < (size_t)nFlows; flowIx++) {
      traceStore.GetTrace(wellIdx, flowIx, trace.begin());
      copy(trace.begin() + minFrame, trace.begin() + maxFrame, _wellFlows.begin_col(flowIx));
      std::fill(trace.begin(), trace.end(), 0);
      traceStore.GetReferenceTrace(wellIdx, flowIx, trace.begin());
      copy(trace.begin() + minFrame, trace.begin() + maxFrame, _refFlows.begin_col(flowIx));
    }
  }

    static double CalcTotalFitDiff (TraceStore &store,
				    std::vector<size_t> &wells,
				    const Col<T> &weights,
				    const Col<T> &time,
				    KeySeq &key,
                                    Col<int> &zeroFlows,
				    double bulkTau,
				    double &wellsTauB,
				    double &wellsSsq,
				    double &wellsErr) {
      int goodWells = 0;
      //int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      //      SampleStats<float> meanSsq;
      SampleStats<double> meanError;
      Mat<float> wellFlows;
      Mat<float> refFlows;
      Mat<float> predicted;
      Col<float> zeromer;
      Col<T> diff;
      Col<T> ssq;
      Col<T> resid;
      float tauB;
      ZeromerDiff<float> bg;
      SampleStats<float> meanTauB;
      SampleQuantiles<double> meanSsq(1000);
      for (size_t i = 0; i < wells.size(); i++) {
	size_t wellIx = wells[i];
	FillInData(store, wellFlows, refFlows, predicted, key.usableKeyFlows, wellIx);
	int err = bg.FitZeromerKnownTau(wellFlows, refFlows, zeroFlows, time, bulkTau, tauB);
	if (err) {
	  continue;
	}
	if (isfinite(tauB)) {
	  meanTauB.AddValue(tauB);
	}

	goodWells++;
	for (size_t zIx = 0; zIx < key.zeroFlows.size(); zIx++) {
          bg.PredictZeromer(refFlows.col(key.zeroFlows[zIx]), time, tauB, bulkTau, zeromer);
	  diff = wellFlows.col(key.zeroFlows[zIx]) - zeromer;
	  double wSsq = arma::mean(diff % diff);
	  diff = arma::abs(diff);
	  double wDiff = mean(diff);
	  if (isfinite(wSsq)) {
	    meanSsq.AddValue(wSsq);
	  }
	  if (isfinite(wDiff)) {
	    meanError.AddValue(wDiff);
	  }
	}
      }
      wellsTauB = meanTauB.GetMean();
      //wellsSsq = meanSsq.GetMean();
      wellsSsq = meanSsq.GetMedian();
      wellsErr = meanError.GetMean();
      /* if (goodWells == 0) { */
      /*   return 1; */
      /* } */
      /* return 0; */
      return wellsSsq;
    }

};

template<typename T>
class TauEErr {

 public:

  double operator()(double tauE) const {
    double meanTauB, meanSsq, meanErr;
    double ssq = TauEBulkErr<T>::CalcTotalFit(*mStore, *mSample, *mWeights, mFlowIx, mNucIx, tauE, meanTauB, meanSsq, meanErr);
    return sqrt(ssq) * 2;
  }

  TraceStore *mStore;
  std::vector<size_t> *mSample;
  Col<T> *mWeights;
  int mFlowIx;
  int mNucIx;
    
};

template <typename T> 
static int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static double validDelta(const double delta, const double minVal = 0.01) {
    return abs(delta) > minVal ? delta: sgn(delta) * minVal;
}

/**
 * secantFind - Returns x such that fun(x) = aim. a is a starting value, and fA should be fA = fun(a)
 * b is the next value, these are to get the secant method started. n is the number of iterations used
 */
template<typename F>
double secantFind(const F &fun, int n, double aim, double a, double fA, double b) {
    double fB;

    for (int i = 0; i < n; ++i) {
        const double delta = validDelta(b - a);
        b = a + delta;    

        fB = fun(b);
        const double dErr = (fB - fA) / delta;
        fA = fB;         

        a = b;
        b = max(.1, a + (aim - fB) / dErr); 
    }

    return b;
};


template<class T>
class ZeromerModelBulkAlg
{

  public:
    const static int mLiveSampleN = 500;
    const static int mEmptySampleN = 500;


    static double GridSearchTauE (double start,
                                  double end,
                                  double step,
                                  TraceStore &store,
                                  const Col<T> &weights,
                                  std::vector<size_t> &sample,
                                  size_t flowIx,
                                  size_t nucIx,
                                  double &bulkTau)
    {
      //    cout << "Grid search: " << endl;
      double minSsq = std::numeric_limits<double>::max();
      double bestVal = -1;
      for (double i = start; i < end; i+=step)
      {
	double meanSsq = -1.0;
	double meanErr = -1.0;
	double meanTauB = 0.0;
        int err = TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, i, meanTauB, meanSsq, meanErr);
	cout << "point grid search: " << i << " (" << err << ") mean tauB: " << meanTauB << " : " << meanSsq << "\t" << meanErr << endl;
        if (!err && meanSsq <= minSsq)
        {
          minSsq = meanSsq;
          bestVal = i;
        }
      }
      bulkTau = bestVal;
      cout << "Best of Grid search: " << bestVal <<  endl;
      return bestVal;
    }

  static double GridSearchTauEDiff (double start,
				    double end,
				    double step,
				    TraceStore &store,
				    const Col<T> &time,
				    const Col<T> &weights,
				    std::vector<size_t> &sample,
				    KeySeq &key,
                                    Col<int> &zeroFlows,
                                  double &bulkTau)
    {
      //    cout << "Grid search: " << endl;
      double minSsq = std::numeric_limits<double>::max();
      double bestVal = -1;
      for (double i = start; i < end; i+=step)
      {
	double meanSsq = -1.0;
	double meanErr = -1.0;
	double meanTauB = 0.0;
        TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, time, key, zeroFlows, i, meanTauB, meanSsq, meanErr);
	//	cout << "diff grid search: " << i << " mean tauB: " << meanTauB << " : " << meanSsq << "\t" << sqrt(meanSsq) << "\t" << meanErr << endl;
        if (meanSsq <= minSsq)
        {
          minSsq = meanSsq;
          bestVal = i;
        }
      }
      bulkTau = bestVal;
      cout << "Best of Grid search: " << bestVal <<  endl;
      return bestVal;
    }


    static bool LinearOptTauESecant(TraceStore &store,
				    std::vector<size_t> &sample,
				    Col<T> &weights,
				    size_t flowIx,
				    size_t nucIx,
				    int maxSteps,
				    double &tauE)
    {
      TauEErr<T> mErr;
      mErr.mStore = &store;
      mErr.mSample = &sample;
      mErr.mWeights = &weights;
      mErr.mFlowIx = flowIx;
      mErr.mNucIx = nucIx;
      double fErr = mErr(4.0);
      double errPrime = secantFind(mErr, maxSteps, 0.0, 4.0f, fErr, 6);
      tauE = errPrime;
      return true;
    }

    static bool LinearOptTauE (TraceStore &store,
                               std::vector<size_t> &sample,
                               Col<T> &weights,
                               size_t flowIx,
                               size_t nucIx,
                               double minTauE,
                               double maxTauE,
                               double convergence,
                               int maxSteps,
                               double &tauE,
                               int &steps,
                               double &diff)

    {
      double tauESecant = 0.0;
      // double X = -1;
      //GridSearchTauE (.001, 5.0, .05,
      //		      store, weights, sample, flowIx,
      //		      nucIx, X);
      //      LinearOptTauESecant(store, sample, weights, flowIx, nucIx, maxSteps, tauESecant);
      std::pair<double,double> bestTauE;
      bestTauE.first = std::numeric_limits<double>::max();
      bestTauE.second = -1;
      std::pair<double,double> tmp;
      std::pair<double,double> test;
      std::vector<std::pair<double,double> >tauStats (3);
      double meanTauB, meanSsq, meanErr;
      tauStats[0].second = minTauE;
      tauStats[0].first = TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[0].second,meanTauB, meanSsq, meanErr);
      tauStats[2].second = maxTauE;
      tauStats[2].first = TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[2].second,meanTauB, meanSsq, meanErr);
      tauStats[1].second = (tauStats[2].second + tauStats[1].first) / 2;
      tauStats[1].first = TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[1].second,meanTauB, meanSsq, meanErr);
      double bestSsq = std::numeric_limits<double>::max();
      double currentSsq = tauStats[0].first + tauStats[1].first + tauStats[2].first;
      steps = 0;
      bool converged = true;
  
      while (convergence < fabs (bestSsq-currentSsq)  && steps++ < maxSteps)
      {
        if (currentSsq > bestSsq)
        {
          ION_WARN ("How can a convex curve get worse?");
          converged = false;
        }
        /* cout << "Step: " << steps << " " << currentSsq << " "; */
        /* for (size_t sIx = 0; sIx < tauStats.size(); sIx++) { */
        /*  cout << sIx << ": " << tauStats[sIx].second << "-" << tauStats[sIx].first << " "; */
        /* } */
        /* cout << endl; */

        // First test right hand side
        test.second = (tauStats[1].second + tauStats[2].second) / 2.0;
        test.first =  TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, test.second,meanTauB, meanSsq, meanErr);
        // test is new minimum
        if (test.first < tauStats[1].first)
        {
          swap (tauStats[1],tauStats[0]);
          swap (test, tauStats[1]);
        }
        else if (test.first < tauStats[2].first)
        {
          swap (test, tauStats[2]);
        }

        // then test the left hand side
        test.second = (tauStats[0].second + tauStats[1].second) / 2.0;
        test.first =  TauEBulkErr<T>::CalcTotalFit (store, sample, weights, flowIx, nucIx, test.second,meanTauB, meanSsq, meanErr);
        if (test.first < tauStats[1].first)
        {
          swap (tauStats[1], tauStats[2]);
          swap (tauStats[1], test);
        }
        else if (test.first < tauStats[0].first)
        {
          swap (test, tauStats[0]);
        }
        bestSsq = currentSsq;

        currentSsq = tauStats[0].first + tauStats[1].first + tauStats[2].first;
      }

      tauE = tauStats[1].second;
      double bestX = -1;
      diff = bestSsq - currentSsq;
      // GridSearchTauE (.001, 5.0, .05,
      //		      store, weights, sample, flowIx,
      //		      nucIx, bestX);
      if (converged == false)
      {
        double best = -1;
        double fineBest = -1;
        GridSearchTauE (minTauE, maxTauE, 1.0,
                        store, weights, sample, flowIx,
                        nucIx, best);
        GridSearchTauE (max (0.1, best-1.0), best+1.0, .1,
                        store, weights, sample, flowIx,
                        nucIx, fineBest);
        steps = -1;
        tauE = fineBest;
        diff = best - fineBest;
      }
      /*   for (double i = 0; i < 15.0; i+=.25) { */
      /*  double ssq = TauEBulkErr<T>::CalcTotalFit(store, sample, weights, flowIx, nucIx, i); */
      /*  cout << i << ": " << ssq << endl; */
      /*   } */
      /*   cout << "Got tauE of: " << tauStats[1].second << " with ssq of: " << tauStats[1].first << " in: " << steps << endl; */
      /* } */
      return converged;
    }



    static bool LinearOptTauEDiff (TraceStore &store,
				   std::vector<size_t> &sample,
				   const Col<T> &weights,
				   const Col<T> &times,
				   KeySeq &key,
                                   Col<int> &zeroFlows,
				   double minTauE,
				   double maxTauE,
				   double convergence,
				   int maxSteps,
				   double &tauE,
				   int &steps,
				   double &diff)  {
      std::pair<double,double> bestTauE;
      bestTauE.first = std::numeric_limits<double>::max();
      bestTauE.second = -1;
      std::pair<double,double> tmp;
      std::pair<double,double> test;
      std::vector<std::pair<double,double> >tauStats (3);
      double meanTauB, meanSsq, meanErr;
      tauStats[0].second = minTauE;
      tauStats[0].first = TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, times, key, zeroFlows, tauStats[0].second,meanTauB, meanSsq, meanErr);
      tauStats[2].second = maxTauE;
      tauStats[2].first = TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, times, key, zeroFlows, tauStats[2].second,meanTauB, meanSsq, meanErr);
      tauStats[1].second = (tauStats[2].second + tauStats[1].first) / 2;
      tauStats[1].first = TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, times, key, zeroFlows, tauStats[1].second,meanTauB, meanSsq, meanErr);
      double bestSsq = std::numeric_limits<double>::max();
      double currentSsq = tauStats[0].first + tauStats[1].first + tauStats[2].first;
      steps = 0;
      bool converged = true;
  
      while (convergence < fabs (bestSsq-currentSsq)  && steps++ < maxSteps)
      {
        if (currentSsq > bestSsq)
        {
//          ION_WARN ("How can a convex curve get worse?");
          converged = false;
        }
        /* cout << "Step: " << steps << " " << currentSsq << " "; */
        /* for (size_t sIx = 0; sIx < tauStats.size(); sIx++) { */
        /*  cout << sIx << ": " << tauStats[sIx].second << "-" << tauStats[sIx].first << " "; */
        /* } */
        /* cout << endl; */

        // First test right hand side
        test.second = (tauStats[1].second + tauStats[2].second) / 2.0;
        test.first =  TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, times, key, zeroFlows, test.second,meanTauB, meanSsq, meanErr);
        // test is new minimum
        if (test.first < tauStats[1].first)
        {
          swap (tauStats[1],tauStats[0]);
          swap (test, tauStats[1]);
        }
        else if (test.first < tauStats[2].first)
        {
          swap (test, tauStats[2]);
        }

        // then test the left hand side
        test.second = (tauStats[0].second + tauStats[1].second) / 2.0;
        test.first =  TauEBulkErr<T>::CalcTotalFitDiff (store, sample, weights, times, key, zeroFlows, test.second,meanTauB, meanSsq, meanErr);
        if (test.first < tauStats[1].first)
        {
          swap (tauStats[1], tauStats[2]);
          swap (tauStats[1], test);
        }
        else if (test.first < tauStats[0].first)
        {
          swap (test, tauStats[0]);
        }
        bestSsq = currentSsq;
        currentSsq = tauStats[0].first + tauStats[1].first + tauStats[2].first;
      }
      tauE = tauStats[1].second;
      diff = bestSsq - currentSsq;
      if (converged == false) {
	tauE = -1;
	steps = -1;
	diff = -1;
      }
      /* if (converged == false) */
      /* { */
      /*   double best = -1; */
      /*   double fineBest = -1; */
      /*   GridSearchTauEDiff (minTauE, maxTauE, 1.0, */
      /* 			    store, times, weights, sample, key, best); */
      /*   GridSearchTauEDiff (max (0.1, best-1.0), best+1.0, .1, */
      /* 			    store, times, weights, sample, key, fineBest); */
      /*   steps = -1; */
      /*   tauE = fineBest; */
      /*   diff = best - fineBest; */
      /* } */
      //      cout << "Got tauE of: " << tauStats[1].second << " with ssq of: " << tauStats[1].first << " in steps: " << steps <<  " with diff: " << diff << endl;
      return converged;
    }

    static size_t CalcNumWells (int rowStart, int rowEnd,
                                int colStart, int colEnd,
                                TraceStore &traceStore)
    {
      size_t count = 0;
      for (int rIx = rowStart; rIx < rowEnd; rIx++)
      {
        for (int cIx = colStart; cIx < colEnd; cIx++)
        {
          size_t idx = traceStore.WellIndex (rIx, cIx);
          if (traceStore.HaveWell (idx))
          {
            count++;
          }
        }
      }
      return count;
    }

    static bool CalcRegionTauE (int rowStart, int rowEnd,
                                int colStart, int colEnd,
                                TraceStore &traceStore,
                                std::vector<char> &keyAssignments,
                                std::vector<KeySeq> &keys,
                                Col<int> &zeroFlows,
                                const Col<T> &weights,
				const Col<T> &times,
                                Col<T> &nucWeights,
                                int keyIx,
                                vector<float> &tauE,
                                size_t minLiveSampleSize,
                                size_t minEmptySampleSize,
                                std::vector<char> &filtered) {
      ReservoirSample<size_t> live (mLiveSampleN);
      ReservoirSample<size_t> empties (mEmptySampleN);
      std::fill (tauE.begin(), tauE.end(), -1.0);
      // @todo - Make a big matrix here for zeros & nucs rather than just doing a single key
      for (int rIx = rowStart; rIx < rowEnd; rIx++) {
        for (int cIx = colStart; cIx < colEnd; cIx++) {
          size_t idx = traceStore.WellIndex (rIx, cIx);
          if (filtered[idx] != 1) {
            continue;
          }
	  if (traceStore.HaveWell(idx)) {
	    empties.Add(idx);
	  }
        }
      }
      if (live.GetNumSeen() < minLiveSampleSize && empties.GetNumSeen() < minEmptySampleSize) {
        return false;
      }
      live.Finished();
      empties.Finished();
      // Merge two samples into one
      nucWeights.set_size (tauE.size());
      std::fill (nucWeights.begin(), nucWeights.end(), 0.0);
      std::fill (tauE.begin(), tauE.end(), 0.0);
      std::vector<size_t> &liveSample = live.GetData();
      std::vector<size_t> &emptySample = empties.GetData();
      std::vector<size_t> sample (liveSample.size() + emptySample.size(), -1);
      copy (liveSample.begin(), liveSample.end(), sample.begin());
      copy (emptySample.begin(), emptySample.end(), sample.begin() + liveSample.size());

      double bulkTau = -1.0;
      //GridSearchTauEDiff(2, 10, .2, traceStore, times, weights, sample, keys[0], bulkTau);
      /* //tauE = bulkTau; */
      /* tauE.resize(4); */
      /* for (size_t i = 0; i < tauE.size(); i ++) { */
      /* 	tauE[i] = bulkTau; */
      /* } */
      double diff = 0.0;
      int steps = 0;
      bool converged = LinearOptTauEDiff(traceStore, sample, weights, times, keys[keyIx], zeroFlows, 2, 10, .01, 100, bulkTau, steps, diff);
      tauE.resize(4);
      for (size_t i = 0; i < tauE.size(); i ++) {
	tauE[i] = bulkTau;
      }
      /* std::cout << "Taue for " << rowStart << " " << colStart << " "; */
      /* for (size_t i = 0; i < tauE.size(); i++) { */
      /*   std::cout << tauE[i] << ","; */
      /* } */
      /* std::cout << std::endl; */
      return converged;
    }
};


/** Job for running zeromer optimization on threads. */
template<class T>
class ZeromerModelBulkJob : public PJob
{

  public:
  // @todo - when to use key assignments and when to use zeroflows
    void Init (int rStart, int rEnd,
               int cStart, int cEnd,
               TraceStore &store,
               std::vector<char> &keyAssign,
               std::vector<KeySeq> &keySeq,
               Col<int> &_zeroFlows,
               Col<T> &frameWeights,
	       Col<T> &time,
               int keyNum,
               vector<float> &tauEVec,
               std::vector<int> &indexes,
               std::vector<KeyBulkFit> &fits,
               std::vector<char> &filtered)
    {

      rowStart = rStart;
      rowEnd = rEnd;
      colStart = cStart;
      colEnd = cEnd;
      traceStore = &store;
      keyAssignments = &keyAssign;
      keys = &keySeq;
      zeroFlows = &_zeroFlows;
      weights = &frameWeights;
      deltaTime = &time;
      keyIx = keyNum;
      tauE= &tauEVec;
      mIndexes = &indexes;
      mFits = &fits;
      mFiltered = &filtered;
    }

    void Run()
    {
      Col<T> nucWeights;
      nucWeights.set_size (tauE->size());
      std::fill (nucWeights.begin(), nucWeights.end(), 0);
      std::fill (tauE->begin(), tauE->end(), 0);
      /* try converging. */
      bool converged = ZeromerModelBulkAlg<float>::CalcRegionTauE (rowStart, rowEnd, colStart, colEnd,
                                                                    *traceStore, *keyAssignments,
                                                                    *keys, *zeroFlows, *weights, *deltaTime,
								    nucWeights, 0, *tauE, 50, 50, *mFiltered);
      if (!converged)
      {
        /* are there any ok wells in this region? */
        size_t numOkWells = ZeromerModelBulkAlg<float>::CalcNumWells (rowStart, rowEnd,
                            colStart, colEnd,
                            *traceStore);
        if (numOkWells > .25 * (rowEnd - rowStart) * (colEnd - colStart))
        {
//          ION_WARN("Library didn't converge. trying TFs.");

          /* This should work as there should be TFs...  */
          converged = ZeromerModelBulkAlg<float>::CalcRegionTauE (rowStart, rowEnd, colStart, colEnd,
                                                                   *traceStore, *keyAssignments,
                                                                   *keys, *zeroFlows, *weights, *deltaTime,
								   nucWeights, 1, *tauE, 0, 50, *mFiltered);
          if (!converged)
          {
	    std::fill(tauE->begin(), tauE->end(), -1);
            /* if it doesn't converge then just fill with ones. */
            ION_WARN ("TFs didn't converge either for region: " + ToStr (rowStart) + "," + ToStr (rowEnd) + "," + ToStr (colStart) + "," + ToStr (colEnd));
          }
          // std::cout << "TauE: " << rowStart << "," << rowEnd << " values:" << (*tauE)[0] << ", " << (*tauE)[1] << ", " << (*tauE)[2] << ", " << (*tauE)[3] << endl;
        }
      }
    }

  private:
    int rowStart;
    int rowEnd;
    int colStart;
    int colEnd;
    TraceStore *traceStore;
    std::vector<char> *keyAssignments;
    std::vector<KeySeq> *keys;
    Col<int> *zeroFlows;
    Col<T> *weights;
    Col<T> *deltaTime;
    int keyIx;
    vector<float> *tauE;
    std::vector<int> *mIndexes;
    std::vector<KeyBulkFit> *mFits;
    std::vector<char> *mFiltered;
};

/**
 * Interface for fitting zeromers.
 */
template <typename T>
class ZeromerModelBulk : public ZeromerModel<T>
{

  public:

    ZeromerModelBulk (int cores, int queueSize)
    {
      Init (cores, queueSize);
    }

    ZeromerModelBulk()
    {
      Init (0,0);
    }

    ~ZeromerModelBulk()
    {
      if (mFitOut.is_open())
      {
        mFitOut.close();
      }
    }

    void Init (int cores, int queueSize)
    {
      mUseMeshNeighbors = 1;
      mRowStep = 50;
      mColStep = 50;
      mCores = cores;
      mQSize = queueSize;
    }

    void SetFileOutName (const std::string &filename)
    {
      mFitOutFilename = filename;
      mFitOut.open (mFitOutFilename.c_str());
      mFitOut << "well\tkey\tok\tssq\ttauB0\ttauE0\ttauB1\ttauE1\ttauB2\ttauE2\ttauB3\ttauE3" << endl;
    }

    void SetTime (const Col<T> &time)
    {
      mTime.set_size(time.n_rows);
      mTime[0] = time.at(0);
      for (size_t i = 1; i < time.n_rows; i++) {
	mTime[i] = (time[i]-time[i-1]);
      }
      //      mTime = time;
    }

    const KeyBulkFit *GetKeyBulkFit (size_t wellIdx)
    {
      if (mIndexes[wellIdx] >= 0)
      {
        return &mFits[mIndexes[wellIdx]];
      }
      return NULL;
    }
    
    void SetWellFit(size_t wellIdx, KeyBulkFit &fit) {
      mFits[mIndexes[wellIdx]] = fit;
    }

    KeyBulkFit &GetFit(size_t wellIdx) {
      return mFits[mIndexes[wellIdx]];
    }

    virtual void FitWell (size_t wellIdx,
                          TraceStore &store,
                          KeySeq &key,
                          Col<T> &weights,
			  Col<T> &trace,
			  Col<T> &traceIntegral,
			  Col<T> &bulk,
			  Col<T> &bulkTrace,
			  Col<T> &bulkIntegral,
                          std::vector<double> &dist,
                          std::vector<std::vector<float> *> &values)
    {
      KeyBulkFit &fit = mFits[mIndexes[wellIdx]];
      // @todo - Don't reallocate these arrays over and over
      fit.wellIdx = wellIdx;
      int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      //int numFrames = store.GetNumFrames();
      /* Col<T> trace (store.GetNumFrames()); */
      /* Col<T> traceIntegral (numFrames); */
      /* Col<T> bulk (numFrames); */
      /* Col<T> bulkTrace (store.GetNumFrames()); */
      /* Col<T> bulkIntegral (numFrames); */
      std::fill (fit.param.begin(), fit.param.end(), -1.0);
      Mat<float> wellFlows;
      Mat<float> refFlows;
      Mat<float> predicted;
      Col<float> time;
      Col<float> zeromer;
      Col<T> diff;
      time.set_size(numFrames);
      // @todo - 
      for (size_t i = 0; i < (size_t)numFrames; i++) {
	time[i] = 1;
      }
      //      double tauB;
      vector<char> nucs(key.usableKeyFlows);
      for (size_t i = 0; i < nucs.size(); i++) {
        nucs[i] = store.GetNucForFlow(i);
      }
      vector<float> tauB(store.GetNumNucs(), -1);
      ZeromerDiff<float> bg;
      for (size_t flowIx = 0; flowIx < key.zeroFlows.size(); flowIx++)
      {
        size_t fIx = key.zeroFlows.at (flowIx);
        double bulkTau  = 0;
        int ok = CalcTauEForWell (wellIdx, fIx, store, dist, values, bulkTau);
        if (ok != TraceStore::TS_OK)
        {
          continue;
        }
	TauEBulkErr<float>::FillInData(store, wellFlows, refFlows, predicted, key.usableKeyFlows, wellIdx);
        arma::uvec zeroFlows(key.zeroFlows.size());
        std::copy(key.zeroFlows.begin(), key.zeroFlows.end(), zeroFlows.begin());
	int err = bg.FitZeromerKnownTauPerNuc(wellFlows, refFlows, zeroFlows, time, nucs, store.GetNumNucs(), bulkTau, tauB);
        fit.ok = !err;
	for (size_t i = 0; i < 4; i++) {
	  fit.param.at (i, 0) = tauB[i];
	  fit.param.at (i, 1) = bulkTau;
	}
      }
    }

    virtual void FitWellZeromer (size_t wellIdx,
                                 TraceStore &store,
                                 KeySeq &key,
                                 Col<T> &weights,
                                 Col<T> &trace,
                                 Col<T> &traceIntegral,
                                 Col<T> &bulk,
                                 Col<T> &bulkTrace,
                                 Col<T> &bulkIntegral,
                                 Col<int> &zeroFlows,
                                 Col<T> &residual,
                                 std::vector<double> &dist,
                                 std::vector<std::vector<float> *> &values)
    {

      KeyBulkFit &fit = mFits[mIndexes[wellIdx]];
      // @todo - Don't reallocate these arrays over and over
      uvec zflows(zeroFlows.n_rows);
      for (size_t i =0 ; i < zflows.n_rows; i++) {
        zflows[i] = zeroFlows[i];
      }
      fit.wellIdx = wellIdx;
      //      int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      //int numFrames = store.GetNumFrames();
      /* Col<T> trace (store.GetNumFrames()); */
      /* Col<T> traceIntegral (numFrames); */
      /* Col<T> bulk (numFrames); */
      /* Col<T> bulkTrace (store.GetNumFrames()); */
      /* Col<T> bulkIntegral (numFrames); */
      std::fill (fit.param.begin(), fit.param.end(), -1.0);
      Mat<float> wellFlows;
      Mat<float> refFlows;
      Mat<float> predicted;
      Col<float> zeromer;
      Col<T> diff;

      vector<char> nucs(store.GetFlowBuff());
      for (size_t i = 0; i < nucs.size(); i++) {
        nucs[i] = store.GetNucForFlow(i);
      }
      vector<float> tauB(store.GetNumNucs(), -1);//      double tauB = -1;
      ZeromerDiff<float> bg;
      double bulkTau  = 0;
      int ok = CalcTauEForWell (wellIdx, zeroFlows.at(0), store, dist, values, bulkTau);

      if (ok == TraceStore::TS_OK) {
        TauEBulkErr<float>::FillInData(store, wellFlows, refFlows, predicted, store.GetFlowBuff(), wellIdx);
        int err = bg.FitZeromerKnownTauPerNuc(wellFlows, refFlows, zeroFlows, mTime, nucs,store.GetNumNucs(), bulkTau, tauB);
        if (residual.n_rows == wellFlows.n_rows) { 
          for (size_t flowIx = 0; flowIx < zeroFlows.n_rows; flowIx++) {
            int nucIx =  store.GetNucForFlow (zeroFlows(flowIx));
            Col<float> ref =  refFlows.unsafe_col(zeroFlows(flowIx));
            Col<float> p = predicted.unsafe_col(zeroFlows(flowIx));
            bg.PredictZeromer(ref, mTime, tauB[nucIx], bulkTau, p);
          }
          Mat<float> diff = wellFlows - predicted;
          Mat<float> zdiff = diff.cols(zflows);
          residual = median(zdiff, 1);
        }
        fit.ok = !err;
      }
      for (size_t i = 0; i < 4; i++) {
        fit.param.at (i, 0) = tauB[i];
        fit.param.at (i, 1) = bulkTau;
      }
    }
    
    void SetFiltered(std::vector<char> &filtered) {
      mFiltered = filtered;
    }

    virtual void FitWellZeromers (PJobQueue &jQueue,
                                  TraceStore &traceStore,
                                  std::vector<char> &keyAssignments,
                                  Col<int> &zeroFlows,
                                  std::vector<KeySeq> &keys)
    {
      mIndexes.resize (traceStore.GetNumWells(), -1);
      std::fill (mIndexes.begin(), mIndexes.end(), -1);
      int good = 0;
      for (size_t i = 0; i < mIndexes.size(); i++)
      {
        if (traceStore.HaveWell (i))
        {
          mIndexes[i] = good++;
        }
      }
      mFits.resize (good);
      mGridTauE.Init (traceStore.GetNumRows(), traceStore.GetNumCols(),
                      mRowStep, mColStep);
      int numBin = mGridTauE.GetNumBin();
      int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
      Col<T> weights = ones<fvec> (traceStore.GetNumFrames());
      vector<ZeromerModelBulkJob<float> > jobs (numBin);
      ClockTimer fitTimer;
      for (int binIx = 0; binIx < numBin; binIx++)
      {
        mGridTauE.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
        vector<float> &tauE = mGridTauE.GetItem (binIx);
        tauE.resize (4);
        jobs[binIx].Init (rowStart, rowEnd, colStart, colEnd,
                          traceStore, keyAssignments, 
                          keys, zeroFlows, weights, mTime, 0, tauE, mIndexes, mFits, mFiltered);
        jQueue.AddJob(jobs[binIx]);
        //        jobs[binIx].Run();
      }
      jQueue.WaitUntilDone();
      fitTimer.PrintMicroSecondsUpdate(stdout, "Fit Timer: After TauE.");      
      SampleQuantiles<double> tauE(numBin);
      for (int i = 0; i < numBin; i++) {
	if (isfinite(mGridTauE.GetItem(i).at(0) && mGridTauE.GetItem(i).at(0) > 0)) {
	  tauE.AddValue(mGridTauE.GetItem(i).at(0));
	}
      }
      cout << "Getting median." << endl;
      double chipMeanTauE = tauE.GetMedian();
      int total = 0, converged = 0;
      cout << "TauE distributions: " << tauE.GetMedian() << " +/- " << tauE.GetIqrSd() << endl;
      for (int i = 0; i < numBin; i++) {
	vector<float> &tvec = mGridTauE.GetItem(i);
	for (size_t j = 0; j < tvec.size(); j++) {
	  double val = tvec[j];
	  total++;
	  if (val > 0) {
	    converged++;
            // tvec[j] = (.4 * val + .6 * chipMeanTauE);
            tvec[j] = (.2 * val + .8 * chipMeanTauE);
	  }
	  else {
	    tvec[j] = chipMeanTauE;
	  }
	}
      }
      cout << converged << " regions converged out of: " << total << "( " << 100.0 * converged/(1.0 * total) << "%)" << endl;
      tauE.Clear();
      for (int i = 0; i < numBin; i++) {
	if (isfinite(mGridTauE.GetItem(i).at(0) && mGridTauE.GetItem(i).at(0) > 0)) {
	  tauE.AddValue(mGridTauE.GetItem(i).at(0));
	}
      }
      cout << "TauE distributions: " << tauE.GetMedian() << " +/- " << tauE.GetIqrSd() << endl;
      int numFrames = traceStore.GetNumFrames();
      Col<T> trace (numFrames);
      Col<T> traceIntegral (numFrames);
      Col<T> bulk (numFrames);
      Col<T> bulkTrace (numFrames);
      Col<T> bulkIntegral (numFrames);
      Col<T> residual = zeros< Col<T> >(NUM_ZMODELBULK_FRAMES - NUM_ZMODELBULK_START);
      ClockTimer timer;
      GridMesh<vector<SampleQuantiles<float> > > residualMesh;
      residualMesh.Init (traceStore.GetNumRows(), traceStore.GetNumCols(),
                      mRowStep, mColStep);
      for (size_t i = 0; i < residualMesh.GetNumBin(); i++) {
        vector<SampleQuantiles<float> > &x = residualMesh.GetItem(i);
        x.resize(residual.n_rows);
        for (size_t r = 0; r < x.size(); r++) {
          x[r].Init(1000);
        }
      }
      for (size_t i = 0; i < mIndexes.size(); i++)
      {
        if (mIndexes[i] < 0)
        {
          continue;
        }
        int keyIx = keyAssignments[i];
        if (keyIx < 0)
        {
          keyIx = 0;
        }
        // @todo - make this parallel?
        FitWellZeromer (i, traceStore, keys[keyIx], weights, trace, traceIntegral, bulk, bulkTrace, bulkIntegral, zeroFlows, residual, mDist, mValues );
        float residual_sum = arma::sum(residual);
        if (isfinite(residual_sum)) {
          vector<SampleQuantiles<float> > &r = residualMesh.GetItem(residualMesh.GetBin(i));
          for (size_t rIx = 0; rIx < residual.n_rows; rIx++) {
            r[rIx].AddValue(residual(rIx));
          }
        }
        if (mFitOut.is_open())
        {
          KeyBulkFit &fit = mFits[mIndexes[i]];
          mFitOut << fit.wellIdx << "\t" << (int) fit.keyIdx << "\t" << (int) fit.ok << "\t" << fit.ssq;
          for (size_t rowIx = 0; rowIx < fit.param.n_rows; rowIx++)
          {
            for (size_t colIx = 0; colIx < fit.param.n_cols; colIx++)
            {
              mFitOut << "\t" << fit.param.at (rowIx, colIx);
            }
          }
          mFitOut << endl;
        }
      }
      fitTimer.PrintMicroSecondsUpdate(stdout, "Fit Timer: After wells tauB");
      // @todo - should this be nuc based?
      mDarkMatter.Init (traceStore.GetNumRows(), traceStore.GetNumCols(),
                         mRowStep, mColStep);
      for (size_t i = 0; i < mDarkMatter.GetNumBin(); i++) {
        Col<float> &d = mDarkMatter.GetItem(i);
        vector<SampleQuantiles<float> > &s = residualMesh.GetItem(i);
        d.set_size(s.size());
        d.fill(0);
        if (s.size() > 0 && s[0].GetNumSeen() > 100 ) {
          for (size_t r = 0; r < s.size(); r++) {
            d[r] = s[r].GetMedian();
          }
        }  
      }
      fitTimer.PrintMicroSecondsUpdate(stdout, "Fit Timer: After Dark Matter.");
      timer.PrintMilliSeconds(std::cout, "ZeromerModelBulkJob:: Param fitting took");
    }

    Col<float> & GetDarkMatter(size_t index) {
      return mDarkMatter.GetItem(mDarkMatter.GetBin(index));
    }

    void SetMeshDist (int size) { mUseMeshNeighbors = size; }
    int GetMeshDist() { return mUseMeshNeighbors; }

    int CalcTauEForWell (int wellIdx,
                         int flowIdx,
                         TraceStore &store,
                         std::vector<double> &dist,
                         std::vector<std::vector<float> *> &values,
                         double &tauE)
    {
      size_t row, col;
      tauE = -1;
      double tau = 0;
      double distWeight = 0;
      int nucIx = store.GetNucForFlow (flowIdx);
      int retVal = TraceStore::TS_BAD_DATA;
      store.WellRowCol (wellIdx, row, col);
      mGridTauE.GetClosestNeighbors (row, col, mUseMeshNeighbors, dist, values);
      for (size_t i = 0; i < values.size(); i++)
      {
        if (values[i]->size()  == 0)
        {
          continue;
        }
        double w = store.WeightDist (dist[i]); //, store.GetMaxDist()); //1/sqrt(dist[i]+1);
        distWeight += w;
        tau += w * values[i]->at (nucIx);
      }
      // Divide by our total weight to get weighted mean
      if (distWeight > 0)
      {
        tauE = tau / distWeight;
        retVal = OK;
      }
      else
      {
        retVal = TraceStore::TS_BAD_DATA;
      }
      return retVal;
    }

    virtual int ZeromerPrediction (int wellIdx,
                                   int flowIdx,
                                   TraceStore &store,
                                   const Col<T> &ref,
                                   Col<T> &zeromer)
    {
      //      assert (HaveModel (wellIdx));
      int nucIx =  store.GetNucForFlow (flowIdx);
      return mBg.PredictZeromer (ref, mTime,
                                 mFits[mIndexes[wellIdx]].param.at (nucIx,0),
                                 mFits[mIndexes[wellIdx]].param.at (nucIx,1),
                                 zeromer);
    }

    virtual int GetNumModels() { return mFits.size(); }

    virtual bool HaveModel (size_t wellIdx) { return mIndexes[wellIdx] >= 0; }

    void SetRegionSize (int rowstep, int colstep) { mColStep = colstep, mRowStep = rowstep; }
    void GetRegionSize (int &rowstep, int &colstep) {colstep = mColStep; rowstep = mRowStep; }

    void Dump(std::ofstream &out) {
      for (size_t i = 0; i < mFits.size(); i++) {
        out << mFits[i].wellIdx << "\t" << (int)mFits[i].keyIdx << "\t" << (int)mFits[i].ok << "\t" << mFits[i].ssq;
        for (size_t n = 0; n < mFits[i].param.n_elem; n++) {
          out << "\t" << mFits[i].param.at(n);
        }
        out << endl;
      }
    }


  public:
    int mRowStep;
    int mColStep;
    const static int mLiveSampleN = 1000;
    const static int mEmptySampleN = 1000;
    int mCores;
    int mQSize;
    Col<T>  mTime;
    std::vector<double> mDist;
    ZeromerDiff<float> mBg;
    std::vector<std::vector<float> *> mValues;
    GridMesh<vector<float> > mGridTauE;
    std::vector<int> mIndexes;
    std::vector<KeyBulkFit> mFits;
    std::string mFitOutFilename;
    std::ofstream mFitOut;
    std::vector<char> mFiltered;
    int mUseMeshNeighbors;
    GridMesh<Col<float> > mDarkMatter;
};

#endif // ZEROMERMODELBULK_H
