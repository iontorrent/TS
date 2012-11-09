/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "KeyClassifier.h"

#define SDFUDGE .01
#define FRAME_START 3u
#define FRAME_END 20u
//#define FRAME_END 25u

void KeyClassifier::Classify(std::vector<KeySeq> &keys, 
                             ZeromerDiff<double> &bg,
                             Mat<double> &wellFlows,
                             Mat<double> &refFlows,
                             Col<double> &time,
                             double minSnr,
                             KeyFit &fit,
                             Mat<double> &predicted) {
  param.set_size(2);
  param << 0 << 0;
  assert(wellFlows.n_rows > 5 && wellFlows.n_rows == refFlows.n_rows);
  size_t frameStart = min(FRAME_START,wellFlows.n_rows);
  size_t frameEnd = min(FRAME_END,wellFlows.n_rows);
  for (size_t keyIx = 0; keyIx < keys.size(); keyIx++) {
    int bad = bg.FitZeromer(wellFlows, refFlows,
                            keys[keyIx].zeroFlows, time,
                            param);
    double keyMinSnr = std::max(minSnr, keys[keyIx].minSnr);
    onemerSig.Clear();
    zeromerSig.Clear();
    zeroStats.Clear();
    traceSd.Clear();
    
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
        traceSd.AddValue(wellFlows.at(frameIx,flowIx));
      }
      bg.PredictZeromer(refFlows.unsafe_col(flowIx), time, param, p);
      double sig = 0;
      mad.Clear();
      // double zeroSum = 0;
      // for (size_t frameIx = 0; frameIx < 2; frameIx++) {
      //   zeroSum += wellFlows.at(frameIx,flowIx) - p.at(frameIx);
      // }
      //zeroSum = zeroSum / 2;
      diff = wellFlows.unsafe_col(flowIx) - p;
      for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
        sig += diff.at(frameIx);
      }
      if (keys[keyIx].flows[flowIx] == 0) {
        for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
          //          for (size_t frameIx = frameEnd; frameIx < wellFlows.n_rows; frameIx++) {
          mad.AddValue(fabs(diff.at(frameIx)));
        }
        zeroStats.AddValue(mad.GetMean());
        zeromerSig.AddValue(sig);
      }
      else if (keys[keyIx].flows[flowIx] == 1 && flowIx < keys[keyIx].usableKeyFlows) {
        onemerSig.AddValue(sig);
      }
    }
    double snr = (onemerSig.GetMedian() - zeromerSig.GetMedian()) / ((onemerSig.GetIqrSd() + zeromerSig.GetIqrSd() + SDFUDGE)/2);
    //    double snr = (onemerSig.GetTrimmedMean(0,1) - zeromerSig.GetTrimmedMean(0,1)) / ((onemerSig.GetIqrSd() + zeromerSig.GetIqrSd() + SDFUDGE)/2);
    //    double snr = (onemerSig.GetMedian() - zeromerSig.GetMedian()) / ((onemerSig.GetIqrSd() + zeromerSig.GetIqrSd() + SDFUDGE)/2);
    if ((snr >= fit.snr || (isfinite(snr) && !isfinite(fit.snr))) && snr >= keyMinSnr ) {
      fit.keyIndex = keyIx;
      fit.mad = zeroStats.GetMean();
      fit.traceMean = traceSd.GetMean();
      fit.traceSd = traceSd.GetSD();
      fit.snr = snr;
      fit.param = param;
      fit.ok = (bad == 0);
      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
        bg.PredictZeromer(refFlows.unsafe_col(flowIx), time, fit.param, p);
        copy(p.begin(), p.end(), predicted.begin_col(flowIx));
      }
    }
    else if (keyIx == 0) { // just set default...
      fit.mad = zeroStats.GetMean();
      fit.snr = snr;
      fit.traceMean = traceSd.GetMean();
      fit.traceSd = traceSd.GetSD();
      fit.param = param;
      fit.ok = (bad == 0);
      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
        bg.PredictZeromer(refFlows.unsafe_col(flowIx), time, fit.param, p);
        copy(p.begin(), p.end(), predicted.begin_col(flowIx));
      }
    }
  }
  if (!isfinite(fit.mad)) {
    fit.ok = 0;
    fit.mad = std::numeric_limits<float>::max();
  }
}

void KeyClassifier::ClassifyKnownTauE(std::vector<KeySeq> &keys, 
                                      ZeromerModelBulk<double> &bg,
                                      Mat<double> &wellFlows,
                                      Mat<double> &refFlows,
                                      Col<double> &time,
                                      const Col<double> &incorp,
                                      double minSnr,
                                      double tauE,
                                      KeyFit &fit,
                                      TraceStore<double> &store,
                                      Mat<double> &predicted) {

  param.set_size(2);// << 0 << 0;
  param[0] = 0;
  param[1] = 0;
  fit.keyIndex = -1;
  fit.snr = 0;
  fit.sd = 0;
  fit.mad = -1;
  signal.set_size(wellFlows.n_cols);
  projSignal.set_size(wellFlows.n_cols); 
  Col<double> weights = ones<vec>(store.GetNumFrames());
  if (fit.bestKey >= 0) {
    fit.bestKey = -1;
  }
  fit.ok = -1;
  size_t frameStart = min(FRAME_START,wellFlows.n_rows);
  size_t frameEnd = min(FRAME_END,wellFlows.n_rows);
  for (size_t keyIx = 0; keyIx < keys.size(); keyIx++) {
    double keyMinSnr = std::max(minSnr, keys[keyIx].minSnr);
    double tauB = 0;
    bg.FitWell(fit.wellIdx, store, keys[keyIx], weights, mDist, mValues);
    param.at(0) = tauB;
    param.at(1) = tauE;
    onemerIncorpMad.Clear();
    onemerProjMax.Init(10);
    onemerProjMax.Clear();
    onemerSig.Clear();
    zeromerSig.Clear();
    zeroStats.Clear();
    traceSd.Clear();
    sigVar.Clear();
    onemerProj.Clear();
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      bg.ZeromerPrediction(fit.wellIdx, flowIx, store, refFlows.unsafe_col(flowIx),p);
      double sig = 0;
      SampleStats<double> mad;
      //      double zeroSum = 0;
      diff = wellFlows.unsafe_col(flowIx) - p;
      for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
        sig += diff.at(frameIx);
      }

      signal.at(flowIx) = sig;
      /* uvec indices;  */
      double pSig = std::numeric_limits<double>::quiet_NaN();
      if (incorp.n_rows == diff.n_rows) {
        pSig =  GetProjection(diff, incorp);
      }
      // else {
      //   cout << "why not?" << endl;
      // }
      projSignal.at(flowIx) = pSig;
      sigVar.AddValue(sig);

      if (keys[keyIx].flows[flowIx] == 0) {
        for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
          mad.AddValue(fabs(diff.at(frameIx)));
        }
        zeroStats.AddValue(mad.GetMean());
        zeromerSig.AddValue(sig);
      }
      else if (keys[keyIx].flows[flowIx] == 1 && flowIx < keys[keyIx].usableKeyFlows) {
        onemerSig.AddValue(sig);
        if (isfinite(pSig) && incorp.n_rows == p.n_rows) {
          onemerProj.AddValue(pSig);
          double maxSig = 0;
          for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
            double projVal =  pSig * incorp.at(frameIx);
            maxSig = max(maxSig, projVal);
            onemerIncorpMad.AddValue(fabs(projVal - (wellFlows.at(frameIx,flowIx) - p.at(frameIx))));
          }
          onemerProjMax.AddValue(maxSig);
        }
      }
    }
    double snr = (onemerSig.GetMedian() - zeromerSig.GetMedian()) / ((onemerSig.GetIqrSd() + zeromerSig.GetIqrSd() + SDFUDGE)/2);
    float sd = sigVar.GetSD();
    if (!isfinite(sd) || isnan(sd)) {
      sd = 0;
    }
    if ((snr >= fit.snr || (isfinite(snr) && !isfinite(fit.snr))) && snr >= keyMinSnr ) {
      fit.keyIndex = keyIx;
      fit.bestKey = keyIx;
      fit.mad = zeroStats.GetMean();
      fit.snr = snr;
      fit.param = param;
      fit.sd = sd;
      fit.onemerAvg = onemerSig.GetCount() > 0 ? onemerSig.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.peakSig = onemerProjMax.GetCount() > 0 ? onemerProjMax.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.onemerProjAvg = onemerProj.GetCount() > 0 ? onemerProj.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.projResid = onemerIncorpMad.GetCount() > 0 ? onemerIncorpMad.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.ok = true;

      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
        bg.ZeromerPrediction(fit.wellIdx, flowIx, store, refFlows.unsafe_col(flowIx),p);
        copy(p.begin(), p.end(), predicted.begin_col(flowIx));
      }
    }
    else if (keyIx == 0) { // || snr > fit.snr) { // just set default...
      fit.bestKey = keyIx;
      fit.mad = zeroStats.GetMean();
      fit.snr = snr;
      fit.param = param;
      fit.sd = sd;
      fit.onemerAvg = onemerSig.GetCount() > 0 ? onemerSig.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.peakSig = onemerProjMax.GetCount() > 0 ? onemerProjMax.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.onemerProjAvg = onemerProj.GetCount() > 0 ? onemerProj.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.projResid = onemerIncorpMad.GetCount() > 0 ? onemerIncorpMad.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.ok = true;

      for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
        bg.ZeromerPrediction(fit.wellIdx, flowIx, store, refFlows.unsafe_col(flowIx),p);
        copy(p.begin(), p.end(), predicted.begin_col(flowIx));
      }
    }

  }
  // Reset the params to the right key
  if (fit.keyIndex < 0) {
    bg.FitWell(fit.wellIdx, store, keys[0], weights, mDist, mValues);
  }
  else {
    bg.FitWell(fit.wellIdx, store, keys[fit.keyIndex], weights, mDist, mValues);
  }
  if (!isfinite(fit.mad)) {
    fit.ok = 0;
    fit.mad = std::numeric_limits<float>::max();
  }
}
