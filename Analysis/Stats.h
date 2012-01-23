/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef STATS_H
#define STATS_H

#include <algorithm>
#include <vector>
#include <assert.h>
#include <math.h>

using namespace std;

namespace ionStats
{

float  median(vector<float> &v);
double median(vector<double> &v);
double median(double *x, unsigned int n);
float  median(float  *x, unsigned int n);
double sd(double *x, unsigned int n);

float  geman_mcclure(float  x);

double truncated_mean(double *x, unsigned int n, double keep_percent);

template <class T>
double quantile_sorted(T *x, size_t size, float quantile) {
  assert(size > 0 && quantile >= 0 && quantile <= 1.0);
  double idx = quantile * (size - 1);
  size_t start = (size_t) floor(idx);
  size_t end = min(start + 1, size-1);
  double q = x[start] + ((idx - start) * (x[end] - x[start]));
  return q;
}

template <class T>
  double quantile_sorted(std::vector<T> &x, float quantile) {
  return quantile_sorted(&x[0], x.size(), quantile);
 }

template <class T>
double quantile_in_place(T *x, size_t size, float quantile) {
  assert(size > 0);
  assert(quantile >= 0);
  assert(quantile <= 1.0);
  if (size == 1) {
    return x[0];
  }
  std::sort(&x[0], &x[0] + size);
  return quantile_sorted(x, size, quantile);
}

}
#endif // STATS_H
