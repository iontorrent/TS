/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef STATS_H
#define STATS_H

#include <algorithm>
#include <vector>
#include <limits>
#include <assert.h>
#include <math.h>
#include "IonErr.h"

using namespace std;

namespace ionStats
{
float  min(std::vector<float> &v);
double min(std::vector<double> &v);
double min(double *x, unsigned int n);
float  min(float  *x, unsigned int n);

float  max(std::vector<float> &v);
double max(std::vector<double> &v);
double max(double *x, unsigned int n);
float  max(float  *x, unsigned int n);

float  average(std::vector<float> &v);
float  mean(std::vector<float> &v);
double mean(std::vector<double> &v);
double mean(double *x, unsigned int n);
float  mean(float  *x, unsigned int n);

float  median(std::vector<float> &v);
double median(std::vector<double> &v);
double median(double *x, unsigned int n);
float  median(float  *x, unsigned int n);

double sd(double *x, unsigned int n);
float sd(vector<float> &v);
float rmsd(vector<float> &data,vector<float> &est);
float rmsd(float *data,float *est, int n);
float rmsd_weighted(float *data,float *est, float *wt,int n, float traceMax);
float rmsd_positive(float *data,float *est, float *wt,int n);
float sumofsquares(float *data,int n);

float percentile(vector<float> &v,float percent);
void linear_regression(float *trace, int npts, float *beta);
void linear_regression(std::vector<float>&Y, std::vector<float>&beta);
void linear_regression(std::vector<float>&X, std::vector<float>&Y, std::vector<float>&beta);
void logistic_regression(std::vector<float>&X,std::vector<float>&Y,std::vector<float>&beta);
float logistic(float z);
void cumsum(vector<float> &v, vector<float> &cum);
void cumnorm(vector<float> &v, vector<float> &cum);

float  geman_mcclure(float  x);

double truncated_mean(double *x, unsigned int n, double keep_percent);


template <class T>
  double mean(T first, T last)
  {
    size_t cnt = std::distance(first,last);
    if (cnt > 0)
    {
        double avg = 0;
        T nth = first;
        for (size_t n=0; n<cnt; n++)
        {
            avg += double(*nth);
            std::advance(nth,1);
        }
        avg /= cnt;
      return (avg);
    }
    else
      return(numeric_limits<double>::quiet_NaN());
  }


template <class T>
double quantile_sorted(T *x, size_t size, float quantile) {
  assert(size > 0 && quantile >= 0 && quantile <= 1.0);
  double idx = quantile * (size - 1);
  size_t start = (size_t) floor(idx);
  size_t end = std::min(start + 1, size-1);
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

/** median of part of a numeric sequence, returns a double.
 * Example:
 * vector<int> myvector;
 *  // set some values:
 * for (int i=1; i<10; i++) myvector.push_back(i);   // 1 2 3 4 5 6 7 8 9 10
 * cout << "==> " << median (myvector.begin(), myvector.end()) << endl;
 *
 * ==> 5.5
 *
 * Side effect: data sequence myvector will be changed!
 */
template <class T>
  double median(T first, T last)
  {
    size_t cnt = std::distance(first,last);
    if (cnt > 0)
    {
      T mid = first;
      std::advance(mid, (cnt/2));
      std::nth_element(first, mid, last);

      double median = double(*mid);

      if (cnt % 2 == 0){  // get the average of the 2 middle elements
	std::advance(mid,1);
	T nextVal = std::min_element(mid, last);
	median = (median + double(*nextVal))/2;
      }
      return (median);
    }
    else
      return(numeric_limits<double>::quiet_NaN());
  }

 template <class T>
       bool comparex_ascending (std::pair<int, T> const& a, std::pair<int, T> const& b)
       {
         return ( *a.second < *b.second );
       }

 template <class T>
     bool comparex_descending (std::pair<int, T> const& a, std::pair<int, T> const& b)
     {
       return ( *a.second > *b.second );
     }

/** return sort order of a vector x
 * Example:
 * vector<double> x(5);
 * vector<size_t> order;
 * for (size_t i=0; i<5; i++)
 *   x[i] = 10-i;
 * sort_order(x.begin(), x.end(), order)
 * cout << "sort order =>"
 * for (size_t i=0; i<order.size(); i++)
 *   cout << " " << order[i];
 * cout << endl;
 * cout << "sort data  =>"
 * for (size_t i=0; i<order.size(); i++)
 *   cout << " " << x[order[i]];
 *
 * sort order => 4 3 2 1 0 
 * sort data  => 6 7 8 9 10
 */
 template <class T>
   void sort_order(T iterBegin, T iterEnd, std::vector<size_t>& indexes, bool ascending=true)
   {
     std::vector< std::pair<int, T> > pv ;
     pv.reserve(iterEnd - iterBegin) ;

     T iter ;
     size_t k ;
     for (iter = iterBegin, k = 0 ; iter != iterEnd ; iter++, k++) {
       pv.push_back( std::pair<int,T>(k,iter) ) ;
     }

     if (ascending)
        std::sort( pv.begin(), pv.end(), comparex_ascending<T> );
     else
        std::sort( pv.begin(), pv.end(), comparex_descending<T> );

     indexes.resize(pv.size()) ;
     for (size_t i=0; i<pv.size(); i++)
       indexes[i] = (size_t) pv[i].first;
   }

 /**
  * It turns out that calculating the significance values for Kolmogorov-Smirnov distribution
  * is a somewhat contentious and complicated issue. For a decent review see
  * R. Simard and P. L'Ecuyer, ``Computing the Two-Sided Kolmogorov-Smirnov Distribution'', 
  * Journal of Statistical Software, Vol. 39, Issue 11, Mar 2011.
  * http://www.iro.umontreal.ca/~lecuyer/myftp/papers/ksdist.pdf
  *
  * Here we are calculating the significance according to Marsaglia et al. which is slow but for 
  * our uses we can usually cache the results for quicker calculation.
  * Marsaglia, G., W. Tsang, and J. Wang. "Evaluating Kolmogorov's Distribution." 
  * Journal of Statistical Software. Vol. 8, Issue 18, 2003.
  * http://www.jstatsoft.org/v08/i18/paper
  *
  * @param n - sample size
  * @param d - the D statistic from KolmogorovTest() function.
  */
 double SmirnovK(int n,double d);

 /**
  * Calculate the D statistic for Kolmogorov test between data points
  * for query and target. Both query and target must be sorted into
  * ascending order and be larger that 1 data point. Type specifies the
  * 0 - two sided, 1 - greater, 2 - less than test.
  * @param numQuery - number of items in query array.
  * @param query - sorted array of values from query distribution
  * @param numTarget - number of items in target array.
  * @param target - sorted array of values from target distribution
  * @param type - version of D statistic to calculate (0 - two sided, 1 - greater, 2 - less than test.)
  * @return value D or NaN if error.
  */
 double KolmogorovTest(int numQuery, const double *query, int numTarget, const double *target, int type);

}
#endif // STATS_H
