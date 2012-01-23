/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include "DualGaussMixModel.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "MathOptim.h"

using namespace std;

DualGaussMixModel::DualGaussMixModel(int n) {
  mMaxPoints = n;
  mMaxIter = 50;
  mConvergeDelta = .001;
  mTrim = .1;
}

MixModel DualGaussMixModel::FitDualGaussMixModel(const float *data, int length) {
  std::vector<float> mData;
  mData.reserve(std::min(mMaxPoints,length));
  // Copy over the data into our vector
  int step = (int) floor(1.0 * length/ mMaxPoints);
  step = std::max(step,1); // minimum step of 1;
  for (int i = 0; i < length; i+=step) {
    mData.push_back(data[i]);
  }
  
  // Get rid of any outliers
  TrimData(mData, mTrim);

  MixModel current;
  InitializeModel(mData, current);
  if (current.var1 == 0 || current.var2 == 0) {
    current.count = 0;
    return current;
  }
  ConvergeModel(mData, current);
  if (current.mu1 > current.mu2) {
    std::swap(current.mu1,current.mu2);
    std::swap(current.var1,current.var2);
    current.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var1);
    current.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var2);
    current.mix = 1.0 - current.mix;
  }
  return current;

}

MixModel DualGaussMixModel::FitDualGaussMixModel(const float *data, const int8_t *assignments, int length) {
  std::vector<std::pair<float,int8_t> > mData;
  mData.reserve(std::min(mMaxPoints,length));
  // Copy over the data into our vector
  int step = (int) floor(1.0 * length/ mMaxPoints);
  step = std::max(step,1); // minimum step of 1;
  for (int i = 0; i < length; i++) {
    if ( i % step == 0 || assignments[i] >= 0) {
      mData.push_back(std::pair<float,int8_t>(data[i], assignments[i]));
    }
  }
  
  // Get rid of any outliers
  TrimData(mData, mTrim);

  MixModel current;
  InitializeModel(mData, current);
  if (current.var1 <= 0 || current.var2 <= 0) {
    current.count = 0;
    return current;
  }  
  ConvergeModel(mData, current);
  if (current.mu1 > current.mu2) {
    std::swap(current.mu1,current.mu2);
    std::swap(current.var1,current.var2);
    current.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var1);
    current.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var2);
    current.mix = 1.0 - current.mix;
  }
  return current;

}

int DualGaussMixModel::PredictCluster(MixModel &m, float data, double threshold, double &p2ownership) {
    double p1 = DNorm(data, m.mu1, m.var1, m.var1sq2p);
    p1 = p1 * (1.0 - m.mix);
      
    double p2 = DNorm(data, m.mu2, m.var2, m.var2sq2p);
    p2 = p2 * (m.mix);

    p2ownership = p2 / (p1 + p2);
      
    if(p2ownership >= threshold) {
      return 2;
    }
    return 1;
}

void DualGaussMixModel::AssignToCluster(int *cluster, MixModel &model, const float *data, 
					int length, double threshold) {
  for (int i = 0; i < length; i++) {
      
    double p1 = DNorm(data[i], model.mu1, model.var1, model.var1sq2p);
    p1 = p1 * (1.0 - model.mix);
      
    double p2 = DNorm(data[i], model.mu2, model.var2, model.var2sq2p);
    p2 = p2 * (model.mix);

    double p2ownership = p2 / (p1 + p2);
      
    if(p2ownership >= threshold) {
      cluster[i] = 2;
    }
    else {
      cluster[i] = 1;
    }
  }
}



double DualGaussMixModel::DNorm(double x, double mu, double var, double varsq2p) {
    
  double e = ExpApprox( -1.0 * (x - mu) * (x - mu)/ (2 * var));
  double p = varsq2p * e;

  return p;
}

void DualGaussMixModel::UpdateResponsibility(std::vector<float> &ownership,
					     const std::vector<float> &data, 
					     const MixModel &model) {
//  ownership.resize(data.size());
  for (unsigned int i = 0; i < data.size(); i++) {
      
    double p1 = DNorm(data[i], model.mu1, model.var1, model.var1sq2p);
    p1 = p1 * (1.0 - model.mix);
      
    double p2 = DNorm(data[i], model.mu2, model.var2, model.var2sq2p);
    p2 = p2 * (model.mix);

    assert(p1 >= 0 && p1 <= 1.0 && p2 >= 0 && p2 <= 1.0);

    // as of 11/16/2010, ExpApprox can return exactly zero and this behavior is desired
    // in other areas of the code where it is used.
    // This causes a degenerate case here since p1 and p2 can both be exactly zero, 
    // which in turn causes p2/(p1+p2) to be NaN.
    // The original behavior (before the modification to ExpApprox) would have
    // resulted in equally paritioning between p1 and p2 in this case, so this special
    // case retains that behavior
    if (p1 == 0 && p2 == 0)
      ownership[i] = 0.5;
    else
      ownership[i] = p2 / (p1 + p2);
  }
}

bool DualGaussMixModel::UpdateResponsibility(std::vector<float> &ownership,
					     const std::vector<std::pair<float,int8_t> > &data, 
					     const MixModel &model) {
//  ownership.resize(data.size());
  for (unsigned int i = 0; i < data.size(); i++) {
      
    double p1 = DNorm(data[i].first, model.mu1, model.var1, model.var1sq2p);
    p1 = p1 * (1.0 - model.mix);
      
    double p2 = DNorm(data[i].first, model.mu2, model.var2, model.var2sq2p);
    p2 = p2 * (model.mix);

    if(!(p1 >= 0 && p1 <= 1.0 && p2 >= 0 && p2 <= 1.0)) {
      return false;
    }

    // as of 11/16/2010, ExpApprox can return exactly zero and this behavior is desired
    // in other areas of the code where it is used.
    // This causes a degenerate case here since p1 and p2 can both be exactly zero, 
    // which in turn causes p2/(p1+p2) to be NaN.
    // The original behavior (before the modification to ExpApprox) would have
    // resulted in equally paritioning between p1 and p2 in this case, so this special
    // case retains that behavior
    if (p1 == 0 && p2 == 0)
      ownership[i] = 0.5;
    else
      ownership[i] = p2 / (p1 + p2);

    if (data[i].second >= 0) {
      ownership[i] = .5 * data[i].second + .5 * ownership[i];
    }
  }
  return true;
}

double DualGaussMixModel::CalcWeightedMean(const std::vector<float> &weights,
					   const std::vector<float> &values,
					   bool inverse) {
  double mean = 0;
  double w = 0;
  for (unsigned int i = 0; i < weights.size(); i++) {
    float wi = weights[i];
    if ( inverse ) {
      wi = 1.0f - wi;
    }
    if (wi > 0) {
      w += wi;
      mean += (values[i] - mean) * (wi/w);
    }
  }
  return mean;
}

double DualGaussMixModel::CalcWeightedMean(const std::vector<float> &weights,
					   const std::vector<std::pair<float,int8_t> > &values,
					   bool inverse) {
  double mean = 0;
  double w = 0;
  for (unsigned int i = 0; i < weights.size(); i++) {
    float wi = weights[i];
    if ( inverse ) {
      wi = 1.0f - wi;
    }
    if (wi > 0) {
      w += wi;
      mean += (values[i].first - mean) * (wi/w);
    }
  }
  return mean;
}

double DualGaussMixModel::CalcWeightedVar(double mu,
					  const std::vector<float> &weights,
					  const std::vector<float> &values,
					  bool inverse) {

  double var = 0;
  double w = 0;
  double b = 0;
  for (unsigned int i = 0; i < weights.size(); i++) {
    float wi = weights[i];
    if ( inverse ) {
      wi = 1.0f - wi;
    }
    if (wi > 0) {
      double delta = values[i] - mu;
      w += wi;
      b += wi * wi;
      var += (delta * delta - var) * wi/w;
    }
  }
  double ret = var * (w * w / ((w*w) - b));
  if (ret == 0) {
    ret = 1;
  }
  return ret;
}

double DualGaussMixModel::CalcWeightedVar(double mu,
					  const std::vector<float> &weights,
					  const std::vector<std::pair<float, int8_t> > &values,
					  bool inverse) {

  double var = 0;
  double w = 0;
  double b = 0;
  for (unsigned int i = 0; i < weights.size(); i++) {
    float wi = weights[i];
    if ( inverse ) {
      wi = 1.0f - wi;
    }
    if (wi > 0) {
      double delta = values[i].first - mu;
      w += wi;
      b += wi * wi;
      var += (delta * delta - var) * wi/w;
    }
  }
  double ret = var * (w * w / ((w*w) - b));
  return ret;
}


void DualGaussMixModel::UpdateModel(MixModel &update, 
				    const std::vector<float> &data,
				    const std::vector<float> &cluster) {
  update.mu1 = CalcWeightedMean(cluster, data, true);
  update.var1 = CalcWeightedVar(update.mu1, cluster, data, true);
  update.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var1);

  update.mu2 = CalcWeightedMean(cluster, data, false);
  update.var2 = CalcWeightedVar(update.mu2, cluster, data, false);
  update.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var2);
  update.mix = 0;
  for (unsigned int i = 0; i < cluster.size(); i++) {
    update.mix += cluster[i];
  }
  update.mix /= cluster.size();

  assert(update.mix >= 0.0 && update.mix <= 1.0);
}

void DualGaussMixModel::UpdateModel(MixModel &update, 
				    const std::vector<std::pair<float,int8_t> >&data,
				    const std::vector<float> &cluster) {
  update.mu1 = CalcWeightedMean(cluster, data, true);
  update.var1 = CalcWeightedVar(update.mu1, cluster, data, true);
  update.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var1);

  update.mu2 = CalcWeightedMean(cluster, data, false);
  update.var2 = CalcWeightedVar(update.mu2, cluster, data, false);
  update.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var2);
  update.mix = 0;
  for (unsigned int i = 0; i < cluster.size(); i++) {
    update.mix += cluster[i];
  }
  update.mix /= cluster.size();

  assert(update.mix >= 0.0 && update.mix <= 1.0);
}


void DualGaussMixModel::ConvergeModel(const std::vector<float> &data, MixModel &model) {
  bool converged = false;
  int iterationCount = 0;
  std::vector<float> cluster(data.size(), .5);
  while (!converged && iterationCount++ < mMaxIter) {
    MixModel newModel = model;
        UpdateResponsibility(cluster, data, model);
    UpdateModel(newModel, data, cluster);
    double mu1Diff = fabs(newModel.mu1 - model.mu1);
    double mu2Diff = fabs(newModel.mu2 - model.mu2);
    double var1Diff = fabs(newModel.var1 - model.var1);
    double var2Diff = fabs(newModel.var2 - model.var2);
    double muDiff = std::max(mu1Diff, mu2Diff);
    double varDiff = std::max(var1Diff, var2Diff);
    if (muDiff < mConvergeDelta && varDiff < mConvergeDelta ) {
      converged = true;
    }
    model = newModel;
  }
  UpdateResponsibility(cluster, data, model);
}

 void DualGaussMixModel::ConvergeModel(const std::vector<std::pair<float,int8_t> > &data, MixModel &model) {
  bool converged = false;
  int iterationCount = 0;
  std::vector<float> cluster(data.size(), .5);
  while (!converged && iterationCount++ < mMaxIter) {
    MixModel newModel = model;
    bool ok = UpdateResponsibility(cluster, data, model);
    if (!ok) {
      model.count = 0;
      return;
    }
    UpdateModel(newModel, data, cluster);
    double mu1Diff = fabs(newModel.mu1 - model.mu1);
    double mu2Diff = fabs(newModel.mu2 - model.mu2);
    double var1Diff = fabs(newModel.var1 - model.var1);
    double var2Diff = fabs(newModel.var2 - model.var2);
    double muDiff = std::max(mu1Diff, mu2Diff);
    double varDiff = std::max(var1Diff, var2Diff);
    if (muDiff < mConvergeDelta && varDiff < mConvergeDelta ) {
      converged = true;
    }
    model = newModel;
  }
  UpdateResponsibility(cluster, data, model);
  bool ok = UpdateResponsibility(cluster, data, model);
  if (!ok) {
    model.count = 0;
    return;
  }
}

void DualGaussMixModel::InitializeModel(const std::vector<float> &data, MixModel &model) {
  // Get initial variance estimate;
  SampleStats<float> ss;
  ss.AddValues(data);

  // Find clusters
  int mu1Est = (int)(.25 * data.size());
  int mu2Est = (int)(.75 * data.size());
  model.count = data.size();
  model.mix = .5;
  model.mu1 = data[mu1Est];
  model.mu2 = data[mu2Est];
  model.var1 = ss.GetVar();
  model.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * model.var1);
  model.var2 = ss.GetVar();
  model.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * model.var2);
}

 void DualGaussMixModel::InitializeModel(const std::vector<std::pair<float, int8_t> > &data, MixModel &model) {
  // Get initial variance estimate;
  SampleStats<float> beads;
  SampleStats<float> empties;
  SampleQuantiles<float> allData(1000);

  for (size_t i = 0; i < data.size(); i++) {
    if (data[i].second == 0) {
      empties.AddValue(data[i].first);
    }
    else if (data[i].second == 1) {
      beads.AddValue(data[i].first);
    }
    allData.AddValue(data[i].first);
  }

  // Find clusters
  if (beads.GetCount() > 30) {
    model.mu2 = beads.GetMean();
    model.var2 = beads.GetVar();
  }
  else {
    model.mu2 = allData.GetQuantile(.75);
    double iqr = model.mu2 - allData.GetQuantile(.25);
    model.var2 = iqr * iqr;
  }

  if (empties.GetCount() > 30) {
    model.mu1 = empties.GetMean();
    model.var1 = empties.GetVar();
  }
  else {
    model.mu1 = allData.GetQuantile(.25);
    double iqr = allData.GetQuantile(.75) - model.mu1;
    model.var1 = iqr * iqr;
  }
  model.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * model.var1);
  model.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * model.var2);

  model.mix = .5;
  model.count = data.size();
}


void DualGaussMixModel::TrimData(std::vector<float> &data, double trim) {
  // trim of the min and max quantiles corresponding to our mTrim value
  std::sort(data.begin(), data.end());
  int index1 = (int) floor(data.size() * (trim/2.0)+.5);
  int index2 = (int) floor(data.size() * (1.0 - trim/2.0)+.5);
  int count = 0;
  for (int i = index1; i < index2; i++) {
    data[count++] = data[i];
  }
  data.resize(count);
}  

void DualGaussMixModel::TrimData(std::vector<std::pair<float,int8_t> > &data, double trim) {
  // trim of the min and max quantiles corresponding to our mTrim value
  std::sort(data.begin(), data.end());
  int index1 = (int) floor(data.size() * (trim/2.0)+.5);
  int index2 = (int) floor(data.size() * (1.0 - trim/2.0)+.5);
  int count = 0;
  for (int i = index1; i < index2; i++) {
    data[count++] = data[i];
  }
  data.resize(count);
}  
