/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Stats.h"
#include <iostream>
#include <map>
#include "IonErr.h"

namespace ionStats
{
float min(vector<float> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");
  float vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv>v[i]) vv=v[i];
  return (vv) ;
}


double min(vector<double> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");
  double vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv>v[i]) vv=v[i];
  return (vv) ;
}


float min(float *v, unsigned int n) {
  assert(n>1);
  float vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv>v[i]) vv=v[i];
  return (vv) ;
}


double min(double *v, unsigned int n) {
  assert(n>1);
  double vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv>v[i]) vv=v[i];
  return (vv) ;
}


float max(vector<float> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");
  float vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv<v[i]) vv=v[i];
  return (vv) ;
}


double max(vector<double> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");
  double vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv<v[i]) vv=v[i];
  return (vv) ;
}


float max(float *v, unsigned int n) {
  assert(n>1);
  float vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv<v[i]) vv=v[i];
  return (vv) ;
}


double max(double *v, unsigned int n) {
  assert(n>1);
  double vv = v[0];
  for (size_t i=1; i<n; i++)
      if (vv<v[i]) vv=v[i];
  return (vv) ;
}


float average(vector<float> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");

  float total = 0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total/n) ;
}


float mean(vector<float> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");

  float total = 0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total/n) ;
}


double mean(vector<double> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");

  double total = 0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total/n) ;
}


float mean(float *v, unsigned int n) {
  assert(n>1);
  float total = 0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total/n) ;
}


double mean(double *v, unsigned int n) {
  assert(n>1);
  double total = 0;
  for (size_t i=0; i<n; i++)
      total += v[i];
  return (total/n) ;
}


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


float sd(vector<float> &v) {
  size_t n = v.size();
  if(n==0)
    throw("Unable to calculate mean of zero elements");

  if (n==1)
      return (0);
  else
  {
      float mu = mean(v);
      float sd = 0;
      for (size_t i=0; i<n; i++)
      {
          float e = v[i]-mu;
          sd += e*e;
      }
      sd /= (n-1);
      return(sqrt(sd));
  }
}


float rmsd(vector<float> &v, vector<float> &est)
{
  assert(v.size()==est.size());
  size_t n = v.size();
  if(n==0)
      return (0);

  float sd = 0;
  for (size_t i=0; i<n; i++)
  {
      float e = v[i]-est[i];
      sd += e*e;
  }
  sd /= n;
  return(sqrt(sd));
}


float rmsd(float *v, float *est, int n)
{
  if(n==0)
      return (0);

  float sd = 0;
  for (int i=0; i<n; i++)
  {
      float e = v[i]-est[i];
      sd += e*e;
  }
  sd /= n;
  return(sqrt(sd));
}


float rmsd_weighted(float *v, float *est, float *wt, int n,float traceMax=1)
{
  if(n==0)
      return (0);

  float sd = 0;
  float totalWt = 0;
  for (int i=0; i<n; i++)
  {
      float e = v[i]-est[i];
      sd += e*e*wt[i];
      totalWt += wt[i];
  }
  sd /= n;
  if (totalWt>0)
    sd /= totalWt;
  sd = sqrt(sd);
  sd /= traceMax;
  return(sd);
}


float rmsd_positive(float *v, float *est, float *wt, int n)
{
  if(n==0)
      return (0);

  float sd = 0;
  //float totalWt = 0;
  for (int i=0; i<n; i++)
  {
      float e = v[i]-est[i];
      if (wt[i]>0)
        sd += e*e;
  }
  sd /= n;
  return(sqrt(sd));
}


float sumofsquares(float *v,int n)
{
    float ss = 0;
    for (int i=0; i<n; i++)
        ss += v[i]*v[i];
    return (ss);
}


float percentile(vector<float> &v, float percent) {
  unsigned int n = v.size();
  if(n==0)
    throw("Unable to calculate median of zero elements");

  unsigned int ith = int(n*percent);
  std::vector<size_t> order;
  ionStats::sort_order(v.begin(), v.end(), order);
  return(v[order[ith]]);
}


float median(vector<float> &v) {
  unsigned int n = v.size();
  if(n==0)
    throw("Unable to calculate median of zero elements");

  unsigned int mid = n/2;
  std::vector<size_t> order;
  ionStats::sort_order(v.begin(), v.end(), order);
  if(n%2==0) {
    return((v[order[mid-1]]+v[order[mid]])/2);
  } else {
    return(v[order[mid]]);
  }
}


double median(vector<double> &v) {

  unsigned int n = v.size();
  if(n==0)
    throw("Unable to calculate median of zero elements");

  unsigned int mid = n/2;
  std::vector<size_t> order;
  ionStats::sort_order(v.begin(), v.end(), order);
  if(n%2==0) {
    return((v[order[mid-1]]+v[order[mid]])/2);
  } else {
    return(v[order[mid]]);
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
  
  /* accessory function for SmirnovK() */
  void mMultiply(double *A,double *B,double *C,int m) { 
    int i,j,k; double s;
    for(i=0;i<m;i++) 
      for(j=0; j<m; j++) {
        s=0.0; 
        for(k=0;k<m;k++) 
          s+=A[i*m+k]*B[k*m+j]; 
        C[i*m+j]=s;
      }
  }

  /* accessory function for SmirnovK() */
  void mPower(double *A,int eA,double *V,int *eV,int m,int n) { 
    double *B;int eB,i;
    if(n==1) {
      for(i=0;i<m*m;i++) 
        V[i]=A[i];
      *eV=eA; 
      return;
    }
    mPower(A,eA,V,eV,m,n/2);
    B=(double*)malloc((m*m)*sizeof(double));
    mMultiply(V,V,B,m); 
    eB=2*(*eV);
    if(n%2==0) {
      for(i=0;i<m*m;i++) 
        V[i]=B[i]; 
      *eV=eB;
    }
    else {
      mMultiply(A,B,V,m); 
      *eV=eA+eB;
    }
    if(V[(m/2)*m+(m/2)]>1e140) {
      for(i=0;i<m*m;i++) 
        V[i]=V[i]*1e-140;*eV+=140;
    }
    free(B);
  }

  double SmirnovK(int n,double d) { 
    int k,m,i,j,g,eH,eQ;
    double h,s,*H,*Q;
    //OMIT NEXT LINE IF YOU REQUIRE >7 DIGIT ACCURACY IN THE RIGHT TAIL
    s=d*d*n; if(s>7.24||(s>3.76&&n>99)) return 2*exp(-(2.000071+.331/sqrt(n)+1.409/n)*s);
    k=(int)(n*d)+1; 
    m=2*k-1; 
    h=k-n*d;
    H=(double*)malloc((m*m)*sizeof(double));
    Q=(double*)malloc((m*m)*sizeof(double));
    for(i=0;i<m;i++) 
      for(j=0;j<m;j++)
        if(i-j+1<0) 
          H[i*m+j]=0; 
        else 
          H[i*m+j]=1;
    for(i=0;i<m;i++) {
      H[i*m]-=pow(h,i+1); 
      H[(m-1)*m+i]-=pow(h,(m-i));
    }
    H[(m-1)*m]+=(2*h-1>0?pow(2*h-1,m):0);
    for(i=0;i<m;i++) 
      for(j=0;j<m;j++)
        if(i-j+1>0) 
          for(g=1;g<=i-j+1;g++) 
            H[i*m+j]/=g;
    eH=0; 
    mPower(H,eH,Q,&eQ,m,n);
    s=Q[(k-1)*m+k-1];
    for(i=1;i<=n;i++) {
      s=s*i/n; 
      if(s<1e-140) {
        s*=1e140; 
        eQ-=140;
      }
    }
    s*=pow(10.,eQ); 
    free(H); 
    free(Q); 
    return (1.0 - s);
  }

  double KolmogorovTest(int numQuery, const double *query, int numTarget, const double *target, int type) {
    // set some impossible values
    double D = std::numeric_limits<double>::quiet_NaN();
    //      Require at least two points in each graph
    if ((target == NULL) || (query == NULL) || numQuery <= 2 || numTarget <= 2) {
      ION_ABORT("KolmogorovQuery - Sets must have more than 2 points");
    }
    bool valid = false;
    double queryIncrement  = 1.0/numQuery;
    double normIncrement  = 1.0/numTarget;
    double diff = 0;
    double dmax = 0.0;
    double amax = 0.0;
    double dmin = std::numeric_limits<double>::max();
    int tIx = 0;
    int nIx = 0;
    int N = numQuery+numTarget;
    for (int i=0; i < N; i++) {
      if (query[tIx] < target[nIx]) {
        diff -= queryIncrement;
        tIx++;
        if (tIx >= numQuery) {
          valid = true; 
          break;
        }
      } 
      else if (query[tIx] > target[nIx]) {
        diff += normIncrement;
        nIx++;
        if (nIx >= numTarget) {
          valid = true; 
          break;
        }
      } 
      else { // we have a tie, not usually allowed but let's try something
        double x = query[tIx];
       while(tIx < numQuery && query[tIx] == x) {
          diff -= queryIncrement;
          tIx++;
        }
       while(nIx < numTarget && target[nIx] == x) {
          diff += normIncrement;
          nIx++;
        }
        if (tIx >= numQuery || nIx >= numTarget) {
          valid = true; 
          break;
        }
      }
      dmax = std::max(dmax,diff);
      dmin = std::min(dmin,diff);
      amax = std::max(amax,fabs(diff));
    }
    if (valid) {
      dmax = std::max(dmax,diff);
      dmin = -1 * std::min(dmin,diff);
      amax = std::max(amax,fabs(diff));
      if (type == 0) {
        return amax;
      }
      else if (type == 1) {
        return dmax;
      }
      else if (type == 2) {
        return dmin;
      }
      else {
        ION_ABORT("Don't recognize value.");
      }
    }
    return D;
  }


void linear_regression(float *Y, int npts, float *beta)
{
    float sumX = 0;
    float sumY = 0;
    float sumXX = 0;
    float sumXY = 0;
    for (int i=0; i<npts; i++) {
        float x = i;
        float y = Y[i];
        sumX += x;
        sumY += y;
        sumXX += x*x;
        sumXY += x*y;
    }
    float denom = npts*sumXX - sumX*sumX;
    beta[1] = denom != 0 ? (npts*sumXY - sumX*sumY) / denom : 0;
    beta[0] = sumY/npts - sumX/npts * beta[1];
}


void linear_regression(std::vector<float>&Y, std::vector<float>&beta)
{
    size_t npts = Y.size();
    float sumX = 0;
    float sumY = 0;
    float sumXX = 0;
    float sumXY = 0;
    for (size_t i=0; i<npts; i++) {
        float x = i;
        float y = Y[i];
        sumX += x;
        sumY += y;
        sumXX += x*x;
        sumXY += x*y;
    }
    float denom = npts*sumXX - sumX*sumX;
    beta.resize(2);
    beta[1] = denom != 0 ? (npts*sumXY - sumX*sumY) / denom : 0;
    beta[0] = sumY/npts - sumX/npts * beta[1];
}


void linear_regression(std::vector<float>&X, std::vector<float>&Y, std::vector<float>&beta)
{
    size_t npts = Y.size();
    float sumX = 0;
    float sumY = 0;
    float sumXX = 0;
    float sumXY = 0;
    for (size_t i=0; i<npts; i++) {
        float x = X[i];
        float y = Y[i];
        sumX += x;
        sumY += y;
        sumXX += x*x;
        sumXY += x*y;
    }
    float denom = npts*sumXX - sumX*sumX;
    beta.resize(2);
    beta[1] = denom != 0 ? (npts*sumXY - sumX*sumY) / denom : 0;
    beta[0] = sumY/npts - sumX/npts * beta[1];
}


float logistic(float z)
{
    z = exp(z);
    z = z/(1+z);
    return (z);
}


void logistic_regression(std::vector<float>&X, std::vector<float>&Y, std::vector<float>&beta)
{
    assert(X.size()==Y.size());
    std::vector<float>newY(Y.size());
    for (size_t i=0; i<Y.size(); i++)
        newY[i] = logistic(Y[i]);
    linear_regression(newY,beta);
}


void cumsum(vector<float> &v, vector<float> &cum)
{
    size_t nV = v.size();
    if (nV==0) return;
    cum.resize(nV);
    cum[0] = v[0];
    for (size_t i=1; i<nV; i++)
        cum[i] = cum[i-1] + v[i];

}


void cumnorm(vector<float> &v, vector<float> &cum)
{
    assert(v.size()>0);
    if (v.size()==0) return;
    // normalize cumhist to 1
    cumsum(v,cum);
    float scale = 1 / cum[cum.size()-1];
    for (size_t i=0; i<cum.size(); i++)
        cum[i] *= scale;
}


} // namespace IonStats

