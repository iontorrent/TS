/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

/*
Copyright (C) 2009 Affymetrix Inc.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation; either version 2.1 of the License,
or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,

59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
*/

#include "FitDensity.h"
#include <stdio.h>

using namespace std;

/**
 * XLimits finds the range of the data to accommodate density bandwidth
 * by trimming. Range = median +- (mad * range_limit + bandwidth * bw_increment)
 * range (from, to) is returned in 2-element double vector out
 * default range_limit = 4, bw_increment = 3
 */
void FitDensity::XLimits(vector<double> const& data, double const bandwidth, vector<double> &out, double const range_limit, double const bw_increment)
{  
  double from, to, m, mad, minimum, maximum;
  vector<double> tmpdata(data);
  m = ionStats::median(tmpdata);
  for (size_t i=0; i<tmpdata.size(); i++)
    tmpdata[i] = fabs(tmpdata[i] - m);

  mad =  ionStats::median(tmpdata);
 
 /* Set the limits. */
  minimum = m - range_limit * mad;
  maximum = m + range_limit * mad;
  from = minimum - bw_increment * bandwidth;
  to = maximum + bw_increment * bandwidth;
  // nbuffer += sprintf(&buffer[nbuffer], "XLimits, m = %f, mad = %f, min = %f, max = %f, from = %f, to = %f\n ", m,mad, minimum, maximum, from, to);
  out[0] = from;
  out[1] = to;
}

/** 
 * kernel density() with an normal kernel
 * @param dat - data of interest.
 * @param weights (matches dat)
 * @param bandWidth
 * 
 * @return - values in density evaluated at values xOut
 *
 * Usage:
 * double bw = findBandWidth(data);
 *
 * vector<double> xlimits(2,0);
 * XLimits(data, bw, xlimits, [optional range_limit, bw_increment]);
 *
 * vector<double> density(npoints);
 * vector<double> xOut(npoints);
 * vector<double> weights(npoints,1);
 * kdensity(data, density, xOut, weights, bw, xlimits[0], xlimits[1]);
 */
void FitDensity::kdensity(vector<double> const& dat, vector<double> &density, vector<double> &xOut, vector<double> const& weight, double const bandWidth, double const from, double const to) {
  // vector<float> xOut(numBins, 0.0);

  assert(dat.size() > 0);
  int n = dat.size();

  assert(density.size() >0);
  int m = density.size();

  assert(bandWidth>0);
  assert(from <= to);

  double xDelta = (to - from) / (m-1);

  for(int i = 0; i < m ; i++) {
    xOut[i] = from + (i*xDelta);
  }

  // get lazy and allocate a big fat vector, no error checking
  vector<double> z(n*m, 0); // n x m matrix
  int k = 0;
  for (int i=0; i<n; ++i){      // i-th row
    for (int j=0; j<m; ++j){  // j-th row
      // apply the kernel
      z[k] = phi((xOut[j] - dat[i])/bandWidth);
      k++;
    }
  }
  vector<int> rowIndex(n,0);
  for (int i=0; i<n; ++i){      // i-th column
    rowIndex[i] = i*m;
  }
  for (int i=0; i<m; ++i){      // i-th column
    for (int j=0; j<n; ++j){  // j-th row
      density[i] += weight[j]*z[i+rowIndex[j]];
    }
    density[i] = density[i]/bandWidth;
  }
}

/** findBandWidth implements a rule-of-thumb for choosing the bandwidth
 * of a Gaussian kernel density estimator given by
 * Silverman's "rule of thumb", Silverman (1986, page 48, eqn (3.31))
 * unless the iqr = 0 when a positive result is guaranteed
 * Tolerates inf but returns inf if data vector x has too many inf values
 * This is a similar function as implemented in R bw.nrd0
 * Does not handle nan's
 * Oversmooths with multimodal data
 */
double FitDensity::findBandWidth(vector<double> const &x) {
  int n = x.size();

  assert( x.size() > 2);

  // Calculate interquartile range into h
  vector<double> tmpdata(n,0);

  int prctile25 = (int)(.25*n -.5);
  if (prctile25 < 0){
    prctile25 = 0;
  }

  for (int i=0; i<n; ++i){
    assert ( !isnan( x[i] ));
    tmpdata[i] = x[i];
  }

  nth_element(tmpdata.begin(), tmpdata.begin()+prctile25, tmpdata.end());
  double val25 = *(tmpdata.begin()+prctile25);
  
  int prctile75 = (int)(.75*n -.5);
  if (prctile75 < 0){
    prctile75 = 0;
  }

  for (int i=0; i<n; ++i){
    tmpdata[i] = x[i];
  }

  nth_element(tmpdata.begin(), tmpdata.begin()+prctile75, tmpdata.end());
  double val75 = *(tmpdata.begin()+prctile75);
  double h = (val75 - val25);

  // calculate variance of x into var
  double sd = 0;
  double var = 0;
  double mean = 0;
  for (int i=0; i<n; ++i){
    if (isinf(x[i])) {
      sd = numeric_limits<double>::infinity();
      break;
    }
    mean += x[i];
  }
  if ( ! isinf(sd) ) {
    mean = mean/n;
    for (int i=0; i<n; ++i){
      var += (x[i]-mean)*(x[i]-mean);
    }
  }

  if ((n > 1) && (sd == 0))
    sd = sqrt(var/(n-1));
  else
    sd = numeric_limits<double>::infinity();

  double bw = min(sd, h/1.34);
  if (bw == 0)
    bw = sd;
  if (bw == 0)
    bw = fabs(x[0]);
  if (bw == 0)
    bw = 1;

  return (0.9 * bw * pow((double)n,-.2));
}

/** findpeaks: Detect peaks and valleys in y = f(x), x is ordered low to high
 *  findpeaks(valleys, peaks, y, delta, x) the indices of x corresponding
 *  to valleys is returned in valleys, the indices of x corresponding to
 *  to peaks is returned in peaks.
 *  A point is considered a maximum peak if it has the maximal
 *  value, and was preceded (to the left) by less than delta
 */
int FitDensity::findpeaks(vector<size_t> &valleyIndex, vector<size_t> &peakIndex, vector<double> const& y, double const delta, vector<double> const& x)
{
  assert( y.size() > 3);
  assert( y.size() == x.size() );
  assert( delta > 0 );

  double mn = numeric_limits<double>::infinity();
  double mx = -numeric_limits<double>::infinity(); 
  bool lookformax = true;
  int imax = 0;
  int imin = 0;

  int npeak = 0;

  for (unsigned int i=0; i<y.size(); ++i){
    double v = y[i];
    if (v>mx) {
      mx = v;
      imax = i;
    }
    if (v<mn) {
      mn = v;
      imin = i;
    }
    if (lookformax) {
      if (v < (mx - delta)){
	peakIndex.push_back(imax);
	npeak++;
	mn = v;
	imin = i;
	lookformax = false;
      }
    } else {
      if (v > (mn + delta)){
	valleyIndex.push_back(imin);
	mx = v;
	imax = i;
	lookformax = true;
      }
    }
  }
  return(npeak);
}

inline bool comparex(std::vector<double> a, std::vector<double> b){
  return (a[0] < b[0]);
}

/** x: x values
 *  y: y = f(x) values
 *  returns the integral of f*f
 */
double FitDensity::SquaredIntegral(vector<double> const& x, vector<double> const& y)
{
  vector<double> y_squared(y.size());
  for (unsigned int k=0; k<y.size(); ++k){
    y_squared[k] = y[k] * y[k];
  }
  double sq_integral = trapzoid(x, y_squared);
  return(sq_integral);
}


/** x: x values
 *  y: y = f(x) values
 *  returns the integral of f
 */
double FitDensity::trapzoid(vector<double> const& x, vector<double> const& y)
{
  int n = x.size();
  assert( y.size() == x.size() );
  assert( n>1 );

  vector< vector<double> > tmpdata(n);

  for (int i=0; i<n; ++i){
    vector<double> z(2);
    z[0] = x[i];
    z[1] = y[i];

    tmpdata[i] = z;
  }
  sort(tmpdata.begin(), tmpdata.end(), comparex);

  double integral = 0;
  for (int i=0; i<(n-1); ++i){
    vector<double> z = tmpdata[i];
    vector<double> z1 = tmpdata[i+1];
    integral += (z1[1] + z[1])*(z1[0] - z[0])/2;
  }
  return(integral);
}

