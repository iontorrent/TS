/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Stats.h"

namespace ionStats
{

double sd(double *x, unsigned int n) {
  assert(n>1);

  double mean = 0;
  for(unsigned int i=0; i<n; i++)
    mean += x[i];
  mean /= n;

  double sd=0;
  for(unsigned int i=0; i<n; i++)
    sd += pow(x[i]-mean,2);
  sd /= (n-1);

  return(sqrt(sd));
}


float median(vector<float> &v) {

  unsigned int n = v.size();
  if(n==0)
    throw("Unable to calculate median of zero elements");

  unsigned int mid = n/2;
  if(n%2==0) {
    std::nth_element(v.begin(),v.begin()+mid,v.end());
    std::nth_element(v.begin(),v.begin()+mid-1,v.end());
    return((v[mid-1]+v[mid])/2);
  } else {
    std::nth_element(v.begin(),v.begin()+mid,v.end());
    return(v[mid]);
  }
}

double median(vector<double> &v) {

  unsigned int n = v.size();
  if(n==0)
    throw("Unable to calculate median of zero elements");

  unsigned int mid = n/2;
  if(n%2==0) {
    std::nth_element(v.begin(),v.begin()+mid,v.end());
    std::nth_element(v.begin(),v.begin()+mid-1,v.end());
    return((v[mid-1]+v[mid])/2);
  } else {
    std::nth_element(v.begin(),v.begin()+mid,v.end());
    return(v[mid]);
  }
}


double median(double *x, unsigned int n) {
  std::vector<double> v(x,x+n);
  return(median(v));
}

float median(float *x, unsigned int n) {
  std::vector<float> v(x,x+n);
  return(median(v));
}

float  geman_mcclure(float  x) {
  float x2 = x*x;
  return(1/(2*(1+1/x2)));
}

int double_sort(const void *_a, const void *_b)
{
	double *a = (double *)_a;
	double *b = (double *)_b;
	return ((a-b) < 0 ? -1 : 1);
}

double truncated_mean(double *x, unsigned int n, double keep_percent)
{
	// keep_percent = 0.50 --> innerquartile mean, at least for reasonably large sample sizes
	// keep_percent = 0.0 --> median (roughly, assumes odd input size)
	// keep_percent = 1.0 --> mean

	// sort the inputs - note this is destructive to the callers original array, done to be efficient
	qsort(x, n, sizeof(double), double_sort);

	// calculate the mean of the inner 'keep' values
	int keep = (int)(n*keep_percent+0.5);
	if (keep <= 0)
		keep = 1;
	int start = (n-keep)/2;
	int i;
	double sum = 0;
	for(i=start;i<(start+keep);i++)
		sum += x[i];
	double trunc_mean = sum/keep;
	return trunc_mean;
}

}
