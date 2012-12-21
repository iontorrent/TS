/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ZEROMERMODELBULK_H
#define ZEROMERMODELBULK_H

#include <vector>
#include <algorithm>
#include <armadillo>
#include <iostream>
#include <string>
#include <fstream>
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
#define NUM_ZMODELBULK_FRAMES 23u
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
      fill (param.begin(), param.end(), 0.0f);
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

template<class T>
class ZeromerModelBulkAlg
{

  public:
    const static int mLiveSampleN = 300;
    const static int mEmptySampleN = 300;
    static double PointFcn (const Col<T> &trace,
                            const Col<T> &traceIntegral,
                            const Col<T> &weights,
                            const Col<T> &bulk,
                            int nucIx,
                            double &ssq,
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
      prediction = prediction % prediction;
      ssq = accu (prediction % weights);
      tauB = b;
      return isfinite(ssq);
    }

    static double CalcTotalFit (TraceStore<T> &store,
                                std::vector<size_t> &wells,
                                const Col<T> &weights,
                                size_t flowIx,
                                size_t nucIx,
                                double bulkTau)
    {
      double totalSsq = 0;
      int goodWells = 0;
      //      int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      int numFrames = store.GetNumFrames();
      Col<T> trace (store.GetNumFrames());
      Col<T> traceIntegral (numFrames);
      Col<T> bulk (numFrames);
      Col<T> bulkTrace (store.GetNumFrames());
      Col<T> bulkIntegral (numFrames);
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
        bool ok = PointFcn (trace, traceIntegral, weights, bulk, nucIx, ssq, tauB);
        if (ok) {
          totalSsq += log (ssq + 1.0);
          goodWells++;
        }
      }
      if (goodWells == 0) {
        return std::numeric_limits<double>::max();
      }
      return totalSsq / goodWells;
    }

    static double GridSearchTauE (double start,
                                  double end,
                                  double step,
                                  TraceStore<T> &store,
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
        double ssq = CalcTotalFit (store, sample, weights, flowIx, nucIx, i);
        if (ssq <= minSsq)
        {
          minSsq = ssq;
          bestVal = i;
        }
      }
      bulkTau = bestVal;
      //cout << "Best of Grid search: " << bestVal <<  endl;
      return bestVal;
    }

    static bool LinearOptTauE (TraceStore<T> &store,
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

      std::pair<double,double> bestTauE;
      bestTauE.first = std::numeric_limits<double>::max();
      bestTauE.second = -1;
      std::pair<double,double> tmp;
      std::pair<double,double> test;
      std::vector<std::pair<double,double> >tauStats (3);
      tauStats[0].second = minTauE;
      tauStats[0].first = CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[0].second);
      tauStats[2].second = maxTauE;
      tauStats[2].first = CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[2].second);
      tauStats[1].second = (tauStats[2].second + tauStats[1].first) / 2;
      tauStats[1].first = CalcTotalFit (store, sample, weights, flowIx, nucIx, tauStats[1].second);
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
        test.first =  CalcTotalFit (store, sample, weights, flowIx, nucIx, test.second);
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
        test.first =  CalcTotalFit (store, sample, weights, flowIx, nucIx, test.second);
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
      /*  double ssq = CalcTotalFit(store, sample, weights, flowIx, nucIx, i); */
      /*  cout << i << ": " << ssq << endl; */
      /*   } */
      /*   cout << "Got tauE of: " << tauStats[1].second << " with ssq of: " << tauStats[1].first << " in: " << steps << endl; */
      /* } */
      return converged;
    }

    static size_t CalcNumWells (int rowStart, int rowEnd,
                                int colStart, int colEnd,
                                TraceStore<T> &traceStore)
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
                                TraceStore<T> &traceStore,
                                std::vector<char> &keyAssignments,
                                std::vector<KeySeq> &keys,
                                Col<T> &weights,
                                Col<T> &nucWeights,
                                int keyIx,
                                vector<double> &tauE,
                                size_t minLiveSampleSize,
                                size_t minEmptySampleSize,
                                std::vector<char> &filtered) {
      ReservoirSample<size_t> live (mLiveSampleN);
      ReservoirSample<size_t> empties (mEmptySampleN);
      fill (tauE.begin(), tauE.end(), -1.0);
      // @todo - Make a big matrix here for zeros & nucs rather than just doing a single key
      for (int rIx = rowStart; rIx < rowEnd; rIx++) {
        for (int cIx = colStart; cIx < colEnd; cIx++) {
          size_t idx = traceStore.WellIndex (rIx, cIx);
          if (filtered[idx] != 1) {
            continue;
          }
          if (traceStore.IsReference(idx)) {
            empties.Add(idx);
          }
          /* if (traceStore.HaveWell (idx) && keyAssignments[idx] == keyIx) */
          /* { */
          /*   live.Add (idx); */
          /* } */
          /* else if (traceStore.HaveWell (idx) && keyAssignments[idx] == -1) */
          /* { */
          /*   empties.Add (idx); */
          /* } */
        }
      }
      //      cout << "Doing tauE fitting with: " << live.GetNumSeen() << " live and " << empties.GetNumSeen() << " empty wells." << endl;
      if (live.GetNumSeen() < minLiveSampleSize && empties.GetNumSeen() < minEmptySampleSize) {
        return false;
      }
      live.Finished();
      empties.Finished();
      // Merge two samples into one
      nucWeights.set_size (tauE.size());
      fill (nucWeights.begin(), nucWeights.end(), 0.0);
      fill (tauE.begin(), tauE.end(), 0.0);
      std::vector<size_t> &liveSample = live.GetData();
      std::vector<size_t> &emptySample = empties.GetData();
      std::vector<size_t> sample (liveSample.size() + emptySample.size(), -1);
      copy (liveSample.begin(), liveSample.end(), sample.begin());
      copy (emptySample.begin(), emptySample.end(), sample.begin() + liveSample.size());
      //      cout << "Doing tauE sample: " << sample.size() << " wells." << endl;
      // @todo - set nuc weigts for averaging
      for (size_t flowIx = 0; flowIx < keys[keyIx].usableKeyFlows; flowIx++) {
        if (keys[keyIx].flows[flowIx] == 0) {
          // @todo - handle missing nucs in key flows
          int nucIx = traceStore.GetNucForFlow (flowIx);
          
          double diff = 0;
          int steps = 0;
          double tauENuc = 0;
          //          int numFrames = min(20, traceStore.GetNumFrames());
          LinearOptTauE (traceStore, sample, weights,
                         flowIx, nucIx,
                         0, 20, .001, 100,
                         tauENuc,steps, diff);
          nucWeights.at (nucIx) = nucWeights.at (nucIx) + 1;
          tauE[nucIx] += tauENuc;
        }
      }
      /* std::cout << "Taue for " << rowStart << " " << colStart << " "; */
      /* for (size_t i = 0; i < tauE.size(); i++) { */
      /*   std::cout << tauE[i] << ","; */
      /* } */
      /* std::cout << std::endl; */
      return true;
    }
};

/** Job for running zeromer optimization on threads. */
template<class T>
class ZeromerModelBulkJob : public PJob
{

  public:

    void Init (int rStart, int rEnd,
               int cStart, int cEnd,
               TraceStore<T> &store,
               std::vector<char> &keyAssign,
               std::vector<KeySeq> &keySeq,
               Col<T> &frameWeights,
               int keyNum,
               vector<double> &tauEVec,
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
      weights = &frameWeights;
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
      fill (nucWeights.begin(), nucWeights.end(), 0);
      fill (tauE->begin(), tauE->end(), 0);
      /* try converging. */
      bool converged = ZeromerModelBulkAlg<double>::CalcRegionTauE (rowStart, rowEnd, colStart, colEnd,
                                                                    *traceStore, *keyAssignments,
                                                                    *keys, *weights, nucWeights, 0, *tauE, 50, 50, *mFiltered);
      if (!converged)
      {
        /* are there any ok wells in this region? */
        size_t numOkWells = ZeromerModelBulkAlg<double>::CalcNumWells (rowStart, rowEnd,
                            colStart, colEnd,
                            *traceStore);
        if (numOkWells > .25 * (rowEnd - rowStart) * (colEnd - colStart))
        {
          ION_WARN("Library didn't converge. trying TFs.");

          /* This should work as there should be TFs...  */
          converged = ZeromerModelBulkAlg<double>::CalcRegionTauE (rowStart, rowEnd, colStart, colEnd,
                                                                   *traceStore, *keyAssignments,
                                                                   *keys, *weights, nucWeights, 1, *tauE, 0, 50, *mFiltered);
          if (!converged)
          {
            /* if it doesn't converge then just fill with ones. */
            ION_WARN ("TFs didn't converge either for region: " + ToStr (rowStart) + "," + ToStr (rowEnd) + "," + ToStr (colStart) + "," + ToStr (colEnd));
          }
          // std::cout << "TauE: " << rowStart << "," << rowEnd << " values:" << (*tauE)[0] << ", " << (*tauE)[1] << ", " << (*tauE)[2] << ", " << (*tauE)[3] << endl;
        }
      }

      /* Fill in tauE based on convergence. */
      if (converged)
      {
        vector<double> tauX = (*tauE);
        for (size_t nucIx = 0; nucIx < (*tauE).size(); nucIx++)
        {
          double w = 0;
          double tau = 0;
          for (size_t i = 0; i < (*tauE).size(); i++)
          {
            if (i != nucIx)
            {
              tau += 0.1 * nucWeights.at (i) * tauX[i];
              w += nucWeights.at (i) * 0.1;
            }
            else
            {
              tau += nucWeights.at (i) * tauX[i];
              w += nucWeights.at (i);
            }
          }
          (*tauE) [nucIx] = tau / (w);
        }
      }
      else
      {
        /* If nothing converged then fill in with 1. */
        for (size_t i = 0; i < tauE->size(); i++)
        {
          if (nucWeights.at (i) == 0)
          {
            (*tauE) [i] = 1;
            nucWeights.at (i) = 1;
          }
        }
      }
    }

  private:
    int rowStart;
    int rowEnd;
    int colStart;
    int colEnd;
    TraceStore<T> *traceStore;
    std::vector<char> *keyAssignments;
    std::vector<KeySeq> *keys;
    Col<T> *weights;
    int keyIx;
    vector<double> *tauE;
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
      mTime = time;
    }

    const KeyBulkFit *GetKeyBulkFit (size_t wellIdx)
    {
      if (mIndexes[wellIdx] >= 0)
      {
        return &mFits[mIndexes[wellIdx]];
      }
      return NULL;
    }

    virtual void FitWell (size_t wellIdx,
                          TraceStore<T> &store,
                          KeySeq &key,
                          Col<T> &weights,
                          std::vector<double> &dist,
                          std::vector<std::vector<double> *> &values)
    {
      assert (HaveModel (wellIdx));
      KeyBulkFit &fit = mFits[mIndexes[wellIdx]];
      // @todo - Don't reallocate these arrays over and over
      fit.wellIdx = wellIdx;
      //      int numFrames = min((int)NUM_ZMODELBULK_FRAMES, (int)store.GetNumFrames());
      int numFrames = store.GetNumFrames();
      Col<T> trace (store.GetNumFrames());
      Col<T> traceIntegral (numFrames);
      Col<T> bulk (numFrames);
      Col<T> bulkTrace (store.GetNumFrames());
      Col<T> bulkIntegral (numFrames);
      fill (fit.param.begin(), fit.param.end(), -1.0);
      for (size_t flowIx = 0; flowIx < key.zeroFlows.n_rows; flowIx++)
      {
        size_t fIx = key.zeroFlows.at (flowIx);
        double bulkTau  = 0;
        int ok = CalcTauEForWell (wellIdx, fIx, store, dist, values, bulkTau);
        if (ok != TraceStore<T>::TS_OK)
        {
          continue;
        }
        int nucIx = store.GetNucForFlow (fIx);
        store.GetTrace (wellIdx, fIx, trace.begin());
        store.GetReferenceTrace (wellIdx, fIx,  bulkTrace.begin());
        trace.set_size(numFrames);
        bulkTrace.set_size(numFrames);
        bulkIntegral = cumsum (bulkTrace);
        bulk = bulkIntegral + bulkTau * bulkTrace;
        traceIntegral = cumsum (trace);
        double tauB=0,ssq=0;
        bool zeromerOk = ZeromerModelBulkAlg<T>::PointFcn (trace, traceIntegral, weights, bulk, nucIx, ssq, tauB);
        fit.ok = ok && zeromerOk;
        fit.ssq += ssq;
        fit.param.at (nucIx, 0) = tauB;
        fit.param.at (nucIx, 1) = bulkTau;
      }
      // If we have a nuc that didn't get fit use avg of others
      for (size_t nucIx = 0; nucIx < store.GetNumNucs(); nucIx++)
      {
        if (fit.param.at (nucIx, 0) == -1)
        {
          double tauBAvg = 0;
          double tauEAvg = 0;
          int count = 0;
          for (size_t i = 0; i < store.GetNumNucs(); i++)
          {
            if (fit.param.at (i, 0) >= 0)
            {
              tauBAvg += fit.param.at (i, 0);
              tauEAvg += fit.param.at (i, 1);
              count++;
            }
          }
          fit.param.at (nucIx, 0) = tauBAvg / count;
          fit.param.at (nucIx, 1) = tauEAvg / count;
        }
      }
      //      @todo - Do we need Nuc specific params?
      Mat<float> param = fit.param;
      for (size_t nucIx = 0; nucIx < store.GetNumNucs(); nucIx++)
      {
        double sum = 0;
        double w = 0;
        for (size_t i = 0; i  < store.GetNumNucs(); i++)
        {
          if (i == nucIx)
          {
            sum += param (i,0);
            w += 1;
          }
          else
          {
            sum += param (i,0) * .1;
            w += .1;
          }
        }
        fit.param (nucIx, 0) = sum/w;
      }
    }

    void SetFiltered(std::vector<char> &filtered) {
      mFiltered = filtered;
    }

    virtual void FitWellZeromers (TraceStore<T> &traceStore,
                                  std::vector<char> &keyAssignments,
                                  std::vector<KeySeq> &keys)
    {
      mIndexes.resize (traceStore.GetNumWells(), -1);
      fill (mIndexes.begin(), mIndexes.end(), -1);
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
      Col<T> weights = ones<vec> (traceStore.GetNumFrames());
      vector<ZeromerModelBulkJob<double> > jobs (numBin);
      for (int binIx = 0; binIx < numBin; binIx++)
      {
        mGridTauE.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
        vector<double> &tauE = mGridTauE.GetItem (binIx);
        tauE.resize (4);
        jobs[binIx].Init (rowStart, rowEnd, colStart, colEnd,
                          traceStore, keyAssignments,
                          keys, weights, 0, tauE, mIndexes, mFits, mFiltered);
        jobs[binIx].Run();
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
        FitWell (i, traceStore, keys[keyIx], weights, mDist, mValues);
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
    }


    void SetMeshDist (int size) { mUseMeshNeighbors = size; }
    int GetMeshDist() { return mUseMeshNeighbors; }

    int CalcTauEForWell (int wellIdx,
                         int flowIdx,
                         TraceStore<T> &store,
                         std::vector<double> &dist,
                         std::vector<std::vector<double> *> &values,
                         double &tauE)
    {
      size_t row, col;
      tauE = -1;
      double tau = 0;
      double distWeight = 0;
      int nucIx = store.GetNucForFlow (flowIdx);
      int retVal = TraceStore<T>::TS_BAD_DATA;
      store.WellRowCol (wellIdx, row, col);
      mGridTauE.GetClosestNeighbors (row, col, mUseMeshNeighbors, dist, values);
      for (size_t i = 0; i < values.size(); i++)
      {
        if (values[i]->size()  == 0)
        {
          continue;
        }
        double w = store.WeightDist (dist[i]); //1/sqrt(dist[i]+1);
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
        retVal = TraceStore<T>::TS_BAD_DATA;
      }
      return retVal;
    }

    virtual int ZeromerPrediction (int wellIdx,
                                   int flowIdx,
                                   TraceStore<T> &store,
                                   const Col<T> &ref,
                                   Col<T> &zeromer)
    {
      assert (HaveModel (wellIdx));
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
    const static int mLiveSampleN = 300;
    const static int mEmptySampleN = 300;
    int mCores;
    int mQSize;
    Col<T>  mTime;
    std::vector<double> mDist;
    ZeromerDiff<double> mBg;
    std::vector<std::vector<double> *> mValues;
    GridMesh<vector<double> > mGridTauE;
    std::vector<int> mIndexes;
    std::vector<KeyBulkFit> mFits;
    std::string mFitOutFilename;
    std::ofstream mFitOut;
    std::vector<char> mFiltered;
    int mUseMeshNeighbors;
};

#endif // ZEROMERMODELBULK_H
