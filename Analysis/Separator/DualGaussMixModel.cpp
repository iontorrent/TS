/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <malloc.h>
#include "DualGaussMixModel.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "MathOptim.h"
#include "Utils.h"
#include "IonErr.h"
#include "Stats.h"

#include <Eigen/Dense>
#include <Eigen/LU>
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

int DualGaussMixModel::PredictCluster(MixModel &m, float data, double threshold, double &p2ownership) {
  if (!m.thresholdSet) {
    SetThreshold(m);
  }
  if (data < m.threshold) { return 1;}
  else {
    return 2;
  }
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
  const float *__restrict data_start = &data[0];
  const float *__restrict data_end = data_start + data.size();
  float *__restrict owner_start = &ownership[0];
  while(data_start != data_end) {
    CalculateResponsibility(model, *data_start, *owner_start);
    data_start++;
    owner_start++;
  }
}

void DualGaussMixModel::SetThreshold(MixModel &m) {
  // Set the threshold at which the responsibility is the same. 
  // we want to find the two decision boundaries and only use the one that is between the two means.
  // Usually the bead cluster has much higher variance and will pick up outliers on the
  // other side of the empty distribution otherwise.

  // a,b,c are for usual quadradic equation x = (-b +/- sqrt(b^2 -
  // 4ac))/2a after solving case for two responsibilities being equal
  float a = (m.var1 - m.var2);
  float b = (2*m.mu1 *m.var2) - (2 * m.var1 * m.mu2);
  float v1v2 = m.var2 * m.var1 * 2;
  float c = m.var1 * m.mu2 * m.mu2  - m.var2 * m.mu1 * m.mu1  + v1v2 * (-1 * log(1/sqrt(2 *DualGaussMixModel::GPI*m.var2)) + log(1.0f/sqrt(2 * DualGaussMixModel::GPI * m.var1)) - log(m.mix) +  log(1.0f-m.mix));
  double t1 = (-1*b + sqrt(b*b - 4 * a * c))/(2 * a);
  double t2 = (-1*b - sqrt(b*b - 4 * a * c))/(2 * a);
  double d1 = fabs(m.mu1 - t1) + fabs(m.mu2 - t1);
  double d2 = fabs(m.mu1 - t2) + fabs(m.mu2 - t2);
  if (!isfinite(d1) || !isfinite(d2)) {
    // brute force if not solvable                                                     
    double step = (m.mu2 - m.mu1) / 100;
    double val = m.mu1;
    double minDiff = numeric_limits<double>::max();
    for (size_t i = 0; i < 100; i++) {
      float d = val + i * step;
      float owner = -1;
      bool g = CalculateResponsibility(m, d, owner);
      if (g && fabs(owner - .5) < minDiff) {
        m.threshold = val + i * step;
        minDiff = fabs(owner - .5);
      }
    }
  }
  else if (d1 > d2) {
    m.threshold = t2;
  }
  else {
    m.threshold = t1;
  }
  m.thresholdSet = true;
}


bool DualGaussMixModel::CalculateResponsibility(const MixModel &m, float data, float &ownership) {
  ownership = -1.0f;
  // Ok we're in a not obvious area calculate a true weight.
  double p1 = DNorm(data, m.mu1, m.var1, m.var1sq2p);
  p1 = p1 * (1.0 - m.mix);
  
  double p2 = DNorm(data, m.mu2, m.var2, m.var2sq2p);
  p2 = p2 * (m.mix);
  
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
    ownership = 0.5;
  else
    ownership = p2 / (p1 + p2);
  return true;
}

bool DualGaussMixModel::UpdateResponsibility(std::vector<float> &ownership,
					     const std::vector<std::pair<float,int8_t> > &data, 
					     const MixModel &model) {
//  ownership.resize(data.size());
  bool flag = true;
  float *__restrict owner_start = &ownership[0];
  float *__restrict owner_end = owner_start + ownership.size();
  const std::pair<float,int8_t> *__restrict data_start = &data[0];
  while (owner_start != owner_end) {
    flag &= CalculateResponsibility(model, data_start->first, *owner_start);
    owner_start++;
    data_start++;
  }
  return flag;
}

double DualGaussMixModel::CalcWeightedMean(const std::vector<float> &weights,
					   const std::vector<float> &values,
					   bool inverse) {
  double mean = 0;
  double w = 0;
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + weights.size();
  const float *__restrict values_start = &values[0];
  while (weight_start != weight_end) {
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0) {
      w += wi;
      mean += (*values_start - mean) * wi/w;
    }
    weight_start++;
    values_start++;
  }
  return mean;
}

double DualGaussMixModel::CalcWeightedMean(const std::vector<float> &weights,
					   const std::vector<std::pair<float,int8_t> > &values,
					   bool inverse) {
  double mean = 0;
  double w = 0;
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + weights.size();
  const std::pair<float, int8_t> *__restrict values_start = &values[0];
  while (weight_start != weight_end) {
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0) {
      w += wi;
      mean += (values_start->first - mean) * wi/w;
    }
    weight_start++;
    values_start++;
  }
  return mean;
}


//   double mean = 0;
//   double w = 0;
//   for (unsigned int i = 0; i < weights.size(); i++) {
//     float wi = weights[i];
//     if ( inverse ) {
//       wi = 1.0f - wi;
//     }
//     if (wi > 0) {
//       w += wi;
//       mean += (values[i].first - mean) * (wi/w);
//     }
//   }
//   return mean;
// }

double DualGaussMixModel::CalcWeightedVar(double mu,
					  const std::vector<float> &weights,
					  const std::vector<float> &values,
					  bool inverse) {

  double var = 0;
  double w = 0;
  double b = 0;
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + weights.size();
  const float *__restrict values_start = &values[0];
  while (weight_start != weight_end) {
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0) {
      double delta = *values_start - mu;
      w += wi;
      b += wi * wi;
      var += (delta * delta - var) * wi/w;
    }
    weight_start++;
    values_start++;
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
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + weights.size();
  const std::pair<float, int8_t> *__restrict values_start = &values[0];
  while (weight_start != weight_end) {
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0) {
      double delta = values_start->first - mu;
      w += wi;
      b += wi * wi;
      var += (delta * delta - var) * wi/w;
    }
    weight_start++;
    values_start++;
  }
  double ret = var * (w * w / ((w*w) - b));
  if (ret == 0) {
    ret = 1;
  }
  return ret;
}

double DualGaussMixModel::CalcWeightedVar(double mu,
                                          float *weights,
                                          float *values,
                                          int count,
					  bool inverse) {
  
  double var = 0;
  double w = 0;
  double b = 0;
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + count;
  const float *__restrict values_start = &values[0];
  while (weight_start != weight_end) {
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0) {
      double delta = *values_start - mu;
      w += wi;
      b += wi * wi;
      var += (delta * delta - var) * wi/w;
    }
    weight_start++;
    values_start++;
  }
  double ret = var * (w * w / ((w*w) - b));
  if (ret == 0) {
    ret = 1;
  }
  return ret;
}

void DualGaussMixModel::CalcWeightedMeanVar( const std::vector<float> &weights,
					  const std::vector<std::pair<float, int8_t> > &values,
					  bool inverse,
                                          double &mean, double &var) {
  double sum_weight = 0;
  double m2 = 0, temp = 0, delta = 0, R = 0;
  mean = 0;
  var = 0;
  const float *__restrict weight_start = &weights[0];
  const float *__restrict weight_end = weight_start + weights.size();
  const std::pair<float, int8_t> *__restrict values_start = &values[0];
  int count = 0;
  while (weight_start != weight_end) {
    count ++;
    float wi = inverse ? 1.0f - *weight_start : *weight_start;
    if (wi > 0.0f) {
      temp = wi + sum_weight;
      delta = values_start->first - mean;
      R = delta * wi / temp;
      mean += R;
      m2 += sum_weight * delta * R;
      sum_weight = temp;
    }
    weight_start++;
    values_start++;
  }
  if (sum_weight > 0)
    var = m2/sum_weight;
  assert(isfinite(var));
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
  CalcWeightedMeanVar(cluster, data, true, update.mu1, update.var1);
  //  update.mu1 = CalcWeightedMean(cluster, data, true);
  //  update.var1 = CalcWeightedVar(update.mu1, cluster, data, true);
  update.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var1);

  CalcWeightedMeanVar(cluster, data, false, update.mu2, update.var2);
  //  update.mu2 = CalcWeightedMean(cluster, data, false);
  //  update.var2 = CalcWeightedVar(update.mu2, cluster, data, false);
  update.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var2);
  update.mix = 0;
  const float *__restrict cluster_start = &cluster[0];
  const float *__restrict cluster_end = cluster_start + cluster.size();
  while (cluster_start != cluster_end) {
    update.mix += *cluster_start++;
  }
  update.mix /= cluster.size();
  assert(update.mix >= 0.0 && update.mix <= 1.0);
}

void DualGaussMixModel::UpdateResponsibilityVec(float * _ownership,
                                                float * _data,
                                                float * _p1,
                                                float * _p2,
                                                int size,
                                                const MixModel &model) {
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> data(_data, size);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> p1(_p1, size);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> p2(_p2, size);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> own(_ownership, size);

  p1 = data - model.mu1;
  p1 = p1.square();
  // p1 = -1 * p1;
  // p1 = p1 / (2 * model.var1);
  float mult = -1.0f / (2 * model.var1);
  p1 = p1 * mult;
  p1 = p1.exp();
  mult = model.var1sq2p * (1.0f - model.mix);
  p1 = p1 * mult;
  // p1 = p1 * model.var1sq2p;
  // p1 = p1 * (1.0f - model.mix);

  mult = -1.0f / (2 * model.var2);
  p2 = data - model.mu2;
  p2 = p2.square();
  p2 = p2 * mult;
  // p2 = -1 * p2;
  // p2 = p2 / (2 * model.var2);
  p2 = p2.exp();
  mult = model.var2sq2p * model.mix;
  p2 = p2 * mult;
  // p2 = p2 * model.var2sq2p;
  // p2 = p2 * model.mix;

  own = p2 / (p1 + p2);
  // touch up to fix any crazy values
  float *__restrict own_start = _ownership;
  float *__restrict own_end = own_start + size;
  while(own_start != own_end) {
    if (!isfinite(*own_start)) {
      *own_start = .5f;
    }
    own_start++;
  }
}

// void DualGaussMixModel::ConvergeModelFaster(const std::vector<float> &vec_data, MixModel &model) {
//   bool converged = false;
//   int iterationCount = 0;
//   int total_size = sizeof(float) * vec_data.size();
//   float *cluster = (float *) memalign(32, total_size);
//   float *data = (float *) memalign(32, total_size);
//   float *p1 = (float *) memalign(32, total_size);
//   float *p2 = (float *) memalign(32, total_size);
//   memcpy(data, &vec_data[0], total_size);
//   while (!converged && iterationCount++ < mMaxIter) {
//     MixModel newModel = model;
//     //    UpdateResponsibility(cluster, data, model);
//     UpdateResponsibilityVec(cluster, data, p1, p2, vec_data.size(), model);
//     UpdateModel(newModel, data, cluster, p1, vec_data.size());
//     double mu1Diff = fabs(newModel.mu1 - model.mu1);
//     double mu2Diff = fabs(newModel.mu2 - model.mu2);
//     double var1Diff = fabs(newModel.var1 - model.var1);
//     double var2Diff = fabs(newModel.var2 - model.var2);
//     double muDiff = std::max(mu1Diff, mu2Diff);
//     double varDiff = std::max(var1Diff, var2Diff);
//     if (muDiff < mConvergeDelta && varDiff < mConvergeDelta ) {
//       converged = true;
//     }
//     model = newModel;
//   }
//   UpdateResponsibilityVec(cluster, data, model);
//   free(cluster);
//   free(data);
//   free(p1);
//   free(p2);
// }

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

// void DualGaussMixModel::InitializeModel(const std::vector<std::pair<float, int8_t> > &data, MixModel &model) {
void DualGaussMixModel::InitializeModel(const float *data, const int8_t *assign, int size, MixModel &model) {
  // Get initial variance estimate;
  SampleStats<float> beads;
  SampleStats<float> empties;
  SampleQuantiles<float> allData(1000);

  for (int i = 0; i < size; i++) {
    if (assign[i] == 0) {
      empties.AddValue(data[i]);
    }
    else if (assign[i] == 1) {
      beads.AddValue(data[i]);
    }
    allData.AddValue(data[i]);
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
  model.count = size;
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

void DualGaussMixModel::TrimData(const float *data,  const int8_t *assignments, int size, double trim,
                                 float *trim_data, int8_t * trim_assignments, int &trim_count) {
  vector<float> sdata(size);
  std::copy(data, data+size, sdata.begin());
  // trim of the min and max quantiles corresponding to our mTrim value
  std::sort(sdata.begin(), sdata.end());
  float lower_threshold = ionStats::quantile_sorted(sdata, trim/2.0);
  float higher_threshold = ionStats::quantile_sorted(sdata, (1.0 - (trim/2.0)));
  trim_count = 0;
  for (int i = 0; i < size; i++) {
    if (data[i] >= lower_threshold && data[i] <= higher_threshold) {
      trim_data[trim_count] = data[i];
      trim_assignments[trim_count] = assignments[i];
      trim_count++;
    }
  }
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


MixModel DualGaussMixModel::FitDualGaussMixModelFaster(const float *data, const int8_t *assignments, int length) {
  // Copy over the data into our vector
  int step = (int) floor(1.0 * length/ mMaxPoints);
  float *data_sample = (float *) memalign(32, length * sizeof(float));
  int8_t *assign_sample = (int8_t*) memalign(32, length * sizeof(int8_t));
  step = std::max(step,1); // minimum step of 1;
  int count = 0;
  for (int i = 0; i < length; i++) {
    if ( i % step == 0 || assignments[i] >= 0) {
      data_sample[count] = data[i];
      assign_sample[count] = assignments[i];
      count++;
    }
  }
  
  float *data_trim = (float *) memalign(32, count * sizeof(float));
  int8_t*assign_trim = (int8_t *) memalign(32, count * sizeof(int8_t));
  // Get rid of any outliers
  int trim_count = 0;
  TrimData(data_sample, assign_sample, count, mTrim,
           data_trim, assign_trim, trim_count);

  MixModel current;
  //  InitializeModel(mData, current);
  InitializeModel(data_trim, assign_trim, trim_count, current);
  if (current.var1 <= 0 || current.var2 <= 0) {
    current.count = 0;
    return current;
  }  
  ConvergeModelFaster(data_trim, trim_count, current);
  if (current.mu1 > current.mu2) {
    std::swap(current.mu1,current.mu2);
    std::swap(current.var1,current.var2);
    current.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var1);
    current.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * current.var2);
    current.mix = 1.0 - current.mix;
  }
  free(data_sample);
  free(assign_sample);
  free(data_trim);
  free(assign_trim);
  return current;

}

void DualGaussMixModel::ConvergeModelFaster(float *data, int count, MixModel &model) {
  bool converged = false;
  int iterationCount = 0;
  int total_size = sizeof(float) * count;
  float *cluster = (float *) memalign(32, total_size);
  std::fill(cluster, cluster + count, .5f);
  float *p1 = (float *) memalign(32, total_size);
  float *p2 = (float *) memalign(32, total_size);
  while (!converged && iterationCount++ < mMaxIter) {
    MixModel newModel = model;
    //    UpdateResponsibility(cluster, data, model);
    UpdateResponsibilityVec(cluster, data, p1, p2, count, model);
    UpdateModelVec(newModel, data, cluster, p1, count);
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
  //  UpdateResponsibility(cluster, data, model);
  UpdateResponsibilityVec(cluster, data, p1, p2, count, model);
  FREEZ(&p1);
  FREEZ(&p2);
  FREEZ(&cluster);
}


double DualGaussMixModel::CalcWeightedMeanVec(float *_weights,
                                              float *_values,
                                              float *_scratch,
                                              int size,
                                              bool inverse) {
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> values(_values, size);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> weights(_weights, size);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> scratch(_scratch, size);
  float mean = 0;
  if (inverse) {
    scratch = 1.0f - weights;
    float weight_sum = scratch.sum();
    scratch = scratch * values;
    float scratch_sum = scratch.sum();
    mean = scratch_sum / weight_sum;
  }
  else {
    scratch = weights * values;
    float scratch_sum = scratch.sum();
    float weight_sum = weights.sum();
    mean = scratch_sum / weight_sum;
  }
  return mean;
}

void DualGaussMixModel::UpdateModelVec(MixModel &update, 
                                       float *data,
                                       float *cluster,
                                       float *scratch,
                                       int count) {
  update.mu1 = CalcWeightedMeanVec(cluster, data, scratch, count, true);
  update.var1 = CalcWeightedVar(update.mu1, cluster, data, count, true);
  update.var1sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var1);

  update.mu2 = CalcWeightedMeanVec(cluster, data, scratch, count, false);
  update.var2 = CalcWeightedVar(update.mu2, cluster, data, count, false);
  update.var2sq2p = 1 / sqrt(2 * DualGaussMixModel::GPI * update.var2);
  Eigen::Map<Eigen::ArrayXf, Eigen::Aligned> cluster_vec(cluster, count);
  update.mix = cluster_vec.sum();
  update.mix /= count;

  assert(update.mix >= 0.0 && update.mix <= 1.0);
}
