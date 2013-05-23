/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "KeyClassifier.h"

#define SDFUDGE .01
#define FRAME_START 4u
#define FRAME_END 19u
#define PEAK_SEARCH 15u
#define MIN_PEAK 30
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
      diff = wellFlows.unsafe_col(flowIx) - p;
      for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
        sig += diff.at(frameIx);
      }
      if (keys[keyIx].flows[flowIx] == 0) {
        for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {

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
    if ((snr >= fit.snr || (isfinite(snr) && !isfinite(fit.snr))) && snr >= keyMinSnr ) {
      fit.keyIndex = keyIx;
      fit.mad = zeroStats.GetMedian();
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
      //      fit.mad = zeroStats.GetMean();
      fit.mad = zeroStats.GetMedian();
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
                                      Mat<double> *darkMatter,
                                      Mat<double> *onemers,
                                      size_t frameCandEnd,
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
  KeyBulkFit paramFit;
  paramFit = bg.GetFit(fit.wellIdx);
  if (weights.n_rows != store.GetNumFrames()) {
    weights = ones<vec>(store.GetNumFrames());
  }
  if (fit.bestKey >= 0) {
    fit.bestKey = -1;
  }
  fit.ok = -1;
  size_t frameStart = min(FRAME_START,wellFlows.n_rows);
  size_t frameEnd = min(frameCandEnd,(size_t)wellFlows.n_rows);
  if (trace.n_rows != wellFlows.n_rows) {
    trace.set_size(wellFlows.n_rows);
    traceIntegral.set_size(wellFlows.n_rows);
    bulk.set_size(wellFlows.n_rows);
    bulkTrace.set_size(wellFlows.n_rows);
    bulkIntegral.set_size(wellFlows.n_rows);
  }
  mPredictions.set_size(wellFlows.n_rows, wellFlows.n_cols);
  for (int keyIx = -1; keyIx < (int)keys.size(); keyIx++) {
    double keyMinSnr = std::max(minSnr, keys[keyIx >= 0 ? keyIx : 0].minSnr);
    double tauB = 0;
    if (keyIx >= 0) {
      bg.FitWell(fit.wellIdx, store, keys[keyIx], weights, trace, traceIntegral, bulk, bulkTrace, bulkIntegral,mDist, mValues);
    }
    param.at(0) = tauB;
    param.at(1) = tauE;
    onemerIncorpMad.Clear();
    onemerProjMax.Init(20);
    zeroStats.Init(20);
    sigVar.Init(20);
    onemerProjMax.Clear();
    onemerSig.Clear();
    zeromerSig.Clear();
    zeroStats.Clear();
    traceSd.Clear();
    sigVar.Clear();
    onemerProj.Clear();
    mad.Clear();
    Mat<double> incorporation = wellFlows;
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      bg.ZeromerPrediction(fit.wellIdx, flowIx, store, refFlows.unsafe_col(flowIx),p);
      mPredictions.col(flowIx) = p;
      incorporation.col(flowIx) = wellFlows.unsafe_col(flowIx) - p;
      // remove the dark matter if it is provided
      if (darkMatter != NULL) {
        incorporation.col(flowIx) = incorporation.col(flowIx) - darkMatter->col(store.GetNucForFlow(flowIx));
        mPredictions.col(flowIx) = mPredictions.col(flowIx) + darkMatter->col(store.GetNucForFlow(flowIx));
      }
    }

    // Calculate the maximum average key value
    double keyMax = 0;
    if (keyIx >= 0) {
      // Do some smoothing...
      // Get the mean 1mer as our signal to project onto
      Mat<double> onemer_incorp = incorporation.cols(keys[keyIx].onemerFlows);
      Col<double> onemer_mean = mean(onemer_incorp, 1);
      keyMax = arma::max(onemer_mean.rows(0,min(PEAK_SEARCH,onemer_incorp.n_rows)));
    }
    else {
      Col<double> median_all = mean(incorporation, 1);
      keyMax = max(median_all.rows(0,min(PEAK_SEARCH,median_all.n_rows)));
    }

    // Calculate the mean absolute variation
    for (size_t i = 0; i < incorporation.n_cols; i++) {
      if (keyIx == -1 || keys[keyIx].flows[i] == 0) {
        for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
          mad.AddValue(fabs(incorporation.at(frameIx, i)));
        }
        zeroStats.AddValue(mad.GetMean());
      }
    }

    // Project each flow onto that 1mer as our incorporation signal if we have 
    if (onemers != NULL) {
      for (size_t i = 0; i < incorporation.n_cols; i++) {
        const Col<double> &inc = onemers->col(store.GetNucForFlow(i));
        if (sum(inc) > 0) {
          double o_proj =  GetProjection(incorporation.col(i), inc);
          incorporation.col(i) = o_proj * inc;
        }
      }
    }

    // Calculate integrated signal and variance of signal
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      double sig = 0;
      for (size_t frameIx = frameStart; frameIx < frameEnd; frameIx++) {
        sig += incorporation.at(frameIx, flowIx);
      }
      signal.at(flowIx) = sig;
      sigVar.AddValue(sig);
      if (keyIx == -1 || keys[keyIx].flows[flowIx] == 0) {
        zeromerSig.AddValue(sig);
      }
      else if (keyIx >= 0 && keys[keyIx].flows[flowIx] == 1 && flowIx < keys[keyIx].usableKeyFlows) {
        onemerSig.AddValue(sig);
      }
    }

    // Classify based on SNR of onemer to zeromer flows.
    double snr = 0;
    if (keyIx >= 0) {
      snr = (onemerSig.GetMedian() - zeromerSig.GetMedian()) / ((onemerSig.GetIqrSd() + zeromerSig.GetIqrSd() + SDFUDGE)/2);
    }
    float sd = sigVar.GetIQR();
    if (!isfinite(sd) || isnan(sd)) {
      sd = 0;
    }
    if ((snr >= fit.snr && snr >= keyMinSnr) || 
        (fit.keyIndex == -1 && isfinite(snr) && snr >= keyMinSnr)) {
        //        (keyIx == 0 && fit.keyIndex == -1 && isfinite(snr) && (snr - 2 >= keyMinSnr) && keyMax >= MIN_PEAK)) {
      paramFit = bg.GetFit(fit.wellIdx);
      fit.keyIndex = keyIx;
      fit.bestKey = keyIx;
      //      fit.mad = zeroStats.GetMean();
      fit.mad = zeroStats.GetMedian();
      fit.snr = snr;
      fit.param = param;
      fit.sd = sd;
      fit.onemerAvg = onemerSig.GetCount() > 0 ? onemerSig.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      //      fit.peakSig = onemerProjMax.GetCount() > 0 ? onemerProjMax.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.peakSig = keyMax;
      fit.onemerProjAvg = onemerProj.GetCount() > 0 ? onemerProj.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.projResid = onemerIncorpMad.GetCount() > 0 ? onemerIncorpMad.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.ok = true;
      predicted = mPredictions;
    }
    else if (keyIx == -1) { // || snr > fit.snr) { // just set default...
      // get original
      paramFit = bg.GetFit(fit.wellIdx);
      fit.bestKey = keyIx;
      fit.mad = zeroStats.GetMedian();
      fit.snr = snr;
      fit.param = param;
      fit.sd = sd;
      fit.onemerAvg = onemerSig.GetCount() > 0 ? onemerSig.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.peakSig = keyMax; //onemerProjMax.GetCount() > 0 ? onemerProjMax.GetMedian() : std::numeric_limits<double>::quiet_NaN();
      fit.onemerProjAvg = onemerProj.GetCount() > 0 ? onemerProj.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.projResid = onemerIncorpMad.GetCount() > 0 ? onemerIncorpMad.GetMean() : std::numeric_limits<double>::quiet_NaN();
      fit.ok = isfinite(fit.mad);
      predicted = mPredictions;
    }
  }
  bg.SetWellFit(fit.wellIdx, paramFit);
  if (!isfinite(fit.mad)) {
    fit.ok = 0;
    fit.mad = std::numeric_limits<float>::max();
  }
}
