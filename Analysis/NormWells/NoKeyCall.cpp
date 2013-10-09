/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FitDensity.h"
#include "NoKeyCall.h"
#include <stdio.h>

using namespace std;

NoKeyCall::NoKeyCall()
{
  clear();

  // fitting parameters to density
  npoints = 128;
  cutoff = .05;
  width = .1; // .5;

  // outlier trimming
  range_limit = 4.0f;
  bw_increment = 3.0f;

  // adhoc rule 1
  // min_mass = 0.05;
  min_mass = 0.03;
  // adhoc rule 4
  too_close = log(0.55);
  // adhoc rule 3
  min_separation_0_1 = 1.3; // 1.6;
  // adhoc rule 5
  min_separation_1_2 = 1.3;

  rule_called.resize(11, 0);
  npeak_orig = 0;

}

void NoKeyCall::clear()
{
  nbuffer = 0;
  for (int i=0; i<20000; i++)
    buffer[i] = '\0';
}

NoKeyCall::~NoKeyCall()
{
  if (nbuffer > 0)
    fprintf(stdout, "%s\n", buffer);
}

double NoKeyCall::set_too_close(double val)
{
  too_close = fabs(log(val));
  return (val);
} 

// a very crude measure of incorporation, not zeroed for zeromer estimate
float NoKeyCall::Incorporation(size_t const t0_ix, const size_t t_end_ix, vector<float> const& trace)
{
  float incorporation = 0;
  for (size_t i = t0_ix; i < t_end_ix; i++)
    incorporation += trace[i];

  return incorporation;
}

void NoKeyCall::NormalizeBeadSignal(std::vector<float> const& signal, std::vector<double>& peaks, std::vector<float>& normedSignal, float failure_val)
{
  if (peaks.size() < 2){
    // give up and do nothing
    normedSignal.assign(normedSignal.size(), failure_val);
    return;
  }
  for (size_t i=0; i< peaks.size()-1; i++)
    assert(peaks[i] < peaks[i+1]);
  
  // extrapolate out further peaks using the last pair
  double maxdata = *std::max_element(signal.begin(), signal.end());
  double last = peaks[peaks.size()-1];
  double nextToLast = peaks[peaks.size()-2];
  double diff = last - nextToLast;
  assert(diff>0);
  size_t peaks_length = peaks.size();
  if ( last < maxdata ) {
    // add extra peaks needed for extrapolation
    size_t dd = (size_t)( (maxdata - peaks[peaks.size()-1])/diff + 1.5) + peaks.size();
    assert(dd >= peaks.size());
    peaks.resize(dd+1);
    
    int step = 1;
    for (size_t np = peaks_length; np <= dd; np++, step++)
      peaks.at(np) = last + step*diff;
  }
  
  // find midpoints between peaks
  std::vector<double> mid(peaks.size()+1, 0);
  for (size_t nv = 1; nv < peaks.size(); nv++) 
    mid.at(nv) = ( peaks.at(nv-1) + peaks.at(nv) )*.5;
  // now bracket the min and max peaks
  mid[0] = 2*peaks[0] - mid[1];
  mid[mid.size()-1] = 2*peaks[mid.size()-1]-mid[mid.size()-2];

  // normalize signal
  size_t set = 0;
  for (size_t fnum=0; fnum<signal.size(); fnum++) {
    size_t nv = 0;
    if ( signal[fnum] > mid[0])
      for (; signal[fnum] <= mid.at(nv); nv++)
	continue;
    // signal[fnum] is closest to peaks[nv]
    double delta = signal[fnum] - peaks.at(nv);
    if ((delta > 0) || (nv==0)){ // half of the proportion of the right shoulder
      normedSignal[fnum] = nv + .5*delta/(mid[nv+1] - peaks[nv]);
      set++;
    }
    else { // half of the proportion of the left shoulder
      normedSignal[fnum] = nv + .5*delta/(mid[nv] - peaks[nv]);
      set++;
    }
  }
  assert (set==signal.size());
}


/** usage: GetPeaks(dat, peaks);
 * output in peaks correspond to the sorted positions of peaks of a density
 * that has been fit to input vector dat.  nans present in dat and are ignored
 * values in peaks are ordered low to high
 */
void NoKeyCall::GetPeaks(vector<float> const& dat, vector<double>& peaks){
  vector<double> weights(dat.size(), 1.0f);
  vector<double> data(dat.size(), 0);
  size_t cnt = 0;
  for (size_t i=0; i<dat.size(); i++)
    if ( ! isnan (dat[i]) )
      data[cnt++] = dat[i];
  data.resize(cnt);

  // none of the rest of the code is nan-aware
  getpeaksInternal(data, cutoff, width, npoints, weights, peaks);
}

void NoKeyCall::getpeaksInternal(vector<double>& data, double const _cutoff, double const _width, int const _npoints, vector<double>& weights, vector<double>& peaks)
{
  int m=data.size(); // number of rows

  assert ( m > 2 );
  assert ( (0 <= _cutoff) && (_cutoff < 1) );
  assert (_npoints > 0);
  assert (_width > 0);

  // normalize weight vector to sum to 1
  double wsum = 0;
  for (unsigned int k=0; k<weights.size(); ++k){
    wsum += weights[k];
  }
  assert (wsum > 0);
  for (unsigned int k=0; k<weights.size(); ++k){
    weights[k] = weights[k]/wsum;
  }

  // find density functions
  // vector<double> density(_npoints);
  density.resize(_npoints);
  // vector<double> xi(_npoints);
  xi.resize(_npoints);

  // bandwidth calculation, decrease from default as multimodal
  // double bw = _width*FitDensity::findBandWidth(data);
  std::vector<size_t> order;
  ionStats::sort_order(data.begin(), data.end(), order);
  // P[0-mer] = 2/3, P[1-mer] =1/3*2/3, sum=8/9 = 90%
  size_t o05 = (int)(data.size()*.05 + .5);  // should be 0-mer
  size_t o90 = (int)(data.size()*.9 +.5) -1; // should be 1-mer
  double bw = _width * (data.at(order.at(o90)) - data.at(order.at(o05)));
  if (bw == 0) {
    peaks.resize(0);
    // nbuffer += sprintf(&buffer[nbuffer], "bandwidth 0 with %d points", (int)data.size());  
    return;
  }

  // find limits for range of X
  vector<double> xlimits(2,0);
  // FitDensity::XLimits(data, bw, xlimits);
  // trim the upper and lower 5%
  size_t o95 = (int)(data.size()*.95 +.5) -1;
  xlimits[0] = data.at(order.at(o05))-3*bw;
  xlimits[1] = data.at(order.at(o95))+3*bw;

  // call density and return density in xi, density
  FitDensity::kdensity(data, density, xi, weights, bw, xlimits[0], xlimits[1]);

  // compute overall signal
  double overallSignal = FitDensity::SquaredIntegral(xi, density);

  // find the peaks in (xi, density) and return in peaks
  // vector<size_t> peakIndex;
  vector<size_t> valleyIndex;
  //bool debug=false;
  //if ((beadx == 57) && (beady == 140))
    //debug=true;
  int npeak = FitDensity::findpeaks(valleyIndex, peakIndex, density, 0.046*overallSignal, xi);
  if ( npeak == 0 ) {
    // density fit probably at extreme ends of range, give up
    adhoc = -1;
    npeak_orig = npeak;
    char buff[1000];
    sprintf(buff, "Warning: well(x=%d, y=%d) failed density fit\n", beadx, beady);
    dumpdata(buff);
    peaks.resize(0);
    return;
  }

  // trim off any peaks in extremes of data,
  // these are meaningless wiggles in the density estimation
  // int npeak = TrimExtremalPeaks(data, xi, _cutoff, valleyIndex, peakIndex);
  // triming should get rid of wiggles
  npeak_orig = npeak;
  // nbuffer += sprintf(&buffer[nbuffer], "npeak = %d; ", npeak);  
  if (npeak > 0) {
    // apply ad hoc rules here
    npeak = ApplyAdHocPeakRemoval( xi, density, valleyIndex, peakIndex );
  }

  peaks.resize(npeak);
  for(size_t i = 0; i < peakIndex.size(); ++i)
    peaks[i] = *(xi.begin() + peakIndex[i]);
  
}

int NoKeyCall::ApplyAdHocPeakRemoval(std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex )
{
  // nothing to do
  if (peakIndex.size() == 0)
    return (0);

  //  *** applying these rules can modify valleyIndex and peakIndex ***
  // passing each rule returns true, failing returns false

  bool any_rule_fired = false;

  bool rule_fired;
  // adhoc rule 0, find peak0 have plenty of mass
  adhoc = 0;
  bool peak0_stays_peak0 = AdHoc_0 (xi, density, valleyIndex, peakIndex );
  any_rule_fired = any_rule_fired || peak0_stays_peak0;

  // adhoc rule 1, peak0 not found
  adhoc = 1;
  rule_fired = ( ! AdHoc_1 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = 2;
  rule_fired = AdHoc_2 (xi, density, valleyIndex, peakIndex, peak0_stays_peak0 );

  adhoc = 3;
  rule_fired = ( ! AdHoc_3 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = 4;
  rule_fired = ( ! AdHoc_4 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = 5;
  rule_fired = AdHoc_5 (xi, density, valleyIndex, peakIndex );
  any_rule_fired = any_rule_fired || rule_fired;

  adhoc = 6;
  rule_fired =  ( ! AdHoc_6 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = 7;
  rule_fired = AdHoc_7 (xi, density, valleyIndex, peakIndex );
  any_rule_fired = any_rule_fired || rule_fired;

  adhoc = 8;
  rule_fired = ( ! AdHoc_8 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = 9;
  rule_fired = ( ! AdHoc_9 (xi, density, valleyIndex, peakIndex ) );
  any_rule_fired = any_rule_fired || rule_fired;
  if (rule_fired)
    return ( (int)peakIndex.size() );

  adhoc = -1;

  // default rule
  if ( any_rule_fired ) {
    char buff[1000];
    sprintf(buff, "Adhoc rules all ok: no peak adjustment\n");
    rule_called.at(10) += 1;
    dumpdata(buff);
  }

  return ( (int)peakIndex.size() );
}

bool NoKeyCall::AdHoc_0 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  double max = -1;
  // adhoc rule 0, demand peak0 be biggest
  for (size_t i=0; i<peakIndex.size()-1; i++){
    double w = 0;
    double h = 0;
    if ( ( 0 < peakIndex[i] ) && ( peakIndex[i]+1 < xi.size()) ) {
      w = xi[peakIndex[i]]-xi[peakIndex[i]-1]; // assumes xi are equally spaced
      h = (density[peakIndex[i]-1]+density[peakIndex[i]]+density[peakIndex[i]+1]);
    }
    // else { leave h & w zero, something is weird }
    if (max < h*w)
      max = h*w;
  }

  for (size_t i=0; i<peakIndex.size()-1; i++){
    double w = 0;
    double h = 0;
    if ( ( 0 < peakIndex[i] ) && ( peakIndex[i]+1 < xi.size()) ) {
      w = xi[peakIndex[i]]-xi[peakIndex[i]-1]; // assumes xi are equally spaced
      h = (density[peakIndex[i]-1]+density[peakIndex[i]]+density[peakIndex[i]+1]);
    }
    if (( h*w == max) && (i>0)) {  // get rid of the lower peaks
      int npeak = peakIndex.size();
      for (int j = i; j<npeak; j++) {
	peakIndex[j-i] = peakIndex[j];
      }
      for (size_t j = i; j<valleyIndex.size(); j++) {
	valleyIndex[j-i] = valleyIndex[j];
      }
      peakIndex.resize(peakIndex.size()-i);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 0: high mass peak[%d]\n",(int) i);     
      dumpdata(buff);
      rule_called.at(0) += 1;

      return( false );
    }
  }
  return (true);
}

bool NoKeyCall::AdHoc_1 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 1, demand peak0 have plenty of mass
  // hardcoded for n=128, look for 3/128 = 2.3% of the range
  // and accept only if at least min_mass
  if (peakIndex.size() > 0){
    double w = 0;
    double h = 0;
    float xi0 = xi[peakIndex[0]];
    if ( ( 0 < peakIndex[0] ) && ( peakIndex[0]+1 < xi.size()) ) {
      w = xi[peakIndex[0]]-xi[peakIndex[0]-1]; // assumes xi are equally spaced
      h = (density[peakIndex[0]-1]+density[peakIndex[0]]+density[peakIndex[0]+1]);
    }
    // else { leave h & w zero, something is weird }
    if ( h*w < min_mass){  // get rid of all the peaks
      peakIndex.clear();
      valleyIndex.clear();

      char buff[1000];
      sprintf(buff,  "Adhoc 1: 0 peaks because low mass peak[%f] density h=%f * w=%f = %f < %f\n", xi0, h/3, w, h*w, min_mass);
      rule_called.at(1) += 1;
      dumpdata(buff);

      return( false);
    }
  }
  return (true);
}

 bool NoKeyCall::AdHoc_2 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex, bool no_ave)
{
  // adhoc rule 2, peak0 & peak1 needs a valley
  if ( peakIndex.size() > 1) {
    double xi0 = xi[peakIndex[0]];
    double xi1 = xi[peakIndex[1]];
    //double vx =  xi[valleyIndex[0]];
    double density0 = density[peakIndex[0]];
    double density1 = density[peakIndex[1]];
    double vd = density[valleyIndex[0]];
    float ff= 0.9;
    if ( ( ff*density0 < vd ) || ( ff*density1 < vd )) // reduce to 1 peak
    {
      if (!no_ave){ // average them
	peakIndex[0] = (density0*peakIndex[0] + density1*peakIndex[1])/(density0+density1);
      }
      else {// get rid of the second peak
	int npeak = peakIndex.size();
	for (int j = 2; j<npeak; j++) {
	  peakIndex[j-1] = peakIndex[j];
	}
	for (size_t j = 1; j<valleyIndex.size(); j++) {
	  valleyIndex[j-1] = valleyIndex[j];
	}
      }
      peakIndex.resize(peakIndex.size()-1);
      TrimValleys(peakIndex, valleyIndex);
    
      char buff[1000];
      sprintf(buff, "Adhoc 2: Merged peaks because %f*peak[0]=%f at %f or %f*peak[1]=%f at %f < density[valley[0]] = %f\n", ff, density0, xi0, ff, density1, xi1, vd);
      rule_called.at(2) += 1;
      dumpdata(buff);

      return (false);
    }
  }
  return (true);
}

bool NoKeyCall::AdHoc_3 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 3, 2+ peaks, peak1 must be substantially higher than peak2
  if ( peakIndex.size() > 2 ) {
    double xi1 = xi[peakIndex[1]];
    //double xi2 = xi[peakIndex[2]];
    double density1 = density[peakIndex[1]];
    double density2 = density[peakIndex[2]];
    double sep = .7;
    if ( (sep * density1) < density2 ) {
      peakIndex.resize(2);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 3: 1 peak after removing peak[1] f(%f)=%f, because peak2/peak1 = %f >  ratio=%f\n", xi1, density1, density2/density1, sep);
      rule_called.at(3) += 1;
      dumpdata(buff);

      return ( false );
    }
  }
  return (true);
}

bool NoKeyCall::AdHoc_4 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 4, 3+ peaks, merge peak1 and peak2 if they are close
  if ( peakIndex.size() > 2 ) {
    double xi0 = xi[peakIndex[0]];
    // double density0 = density[peakIndex[0]];
    double xi1 = xi[peakIndex[1]];
    double density1 = density[peakIndex[1]];
    double xi2 = xi[peakIndex[2]];
    double density2 = density[peakIndex[2]];
    double lr = log ( (xi1 - xi0)/( xi2 - xi1) );
    if (lr < too_close ) {
      peakIndex[1] = (density1*peakIndex[1] + density2*peakIndex[2])/(density1+density2);
      peakIndex.resize(2);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 4: x0=%f, x1=%f x2=%f |log((x1-x0)/(x2-x1)) = %f| > %f\n", xi0, xi1, xi2, lr, too_close);
      rule_called.at(4) += 1;
      dumpdata(buff);

      return ( false );
    }
  }
  return (true);
}

bool NoKeyCall::AdHoc_5 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // prefer 3rd peak matching its distance to distance between first & second
  if (peakIndex.size() > 2) {
    double min=fabs(log ( (xi[peakIndex[2]] - xi[peakIndex[1]])/( xi[peakIndex[1]] - xi[peakIndex[0]]) ));
    int ix = 2;
    for (size_t i = 3; i<peakIndex.size(); i++) {
      double lr = fabs(log ( (xi[peakIndex[i]] - xi[peakIndex[1]])/( xi[peakIndex[1]] - xi[peakIndex[0]]) ));
      if (lr<min){
	min = lr;
	ix = i;
      }
    }
    if (ix > 2) { // found a peak that matches
      size_t npeak = peakIndex.size();
      for (size_t j = ix; j<npeak; j++) {
	peakIndex[j-ix+2] = peakIndex[j];
      }
      for (size_t j = ix; j<valleyIndex.size(); j++) {
	valleyIndex[j-ix+2] = valleyIndex[j];
      }
      peakIndex.resize(peakIndex.size()-ix+2);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 5: replaced peak2 by peak%d \n", ix);
      rule_called.at(5) += 1;
      dumpdata(buff);

      return ( false );
    }
    // nbuffer += printf(&buffer[nbuffer], "Adhoc 5 ok\n");
  }
  return (true);
}

bool NoKeyCall::AdHoc_6 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 6 remove third peak if is not equally spaced enough
  if (peakIndex.size() > 2) {
    double v=fabs(log ( (xi[peakIndex[2]] - xi[peakIndex[1]])/( xi[peakIndex[1]] - xi[peakIndex[0]]) ));
    if (v > .29) { // get rid of it
      peakIndex.resize(2);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 6:\n");
      rule_called.at(6) += 1;
      dumpdata(buff);

      return ( false );
    }
  }
  return (true);
}


bool NoKeyCall::AdHoc_7 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // prefer 4th peak matching its distance to distance between second & third
  if (peakIndex.size() > 4) {
    double min=fabs(log ( (xi[peakIndex[3]] - xi[peakIndex[2]])/( xi[peakIndex[2]] - xi[peakIndex[1]]) ));
    size_t ix = 3;
    for (size_t i = 4; i<peakIndex.size(); i++) {
      double lr = fabs(log ( (xi[peakIndex[i]] - xi[peakIndex[2]])/( xi[peakIndex[2]] - xi[peakIndex[1]]) ));
      if (lr<min){
	min = lr;
	ix = i;
      }
    }
    if (ix > 3) { // found a peak that matches
      size_t npeak = peakIndex.size();
      for (size_t j = ix; j<npeak; j++) {
	peakIndex[j-ix+3] = peakIndex[j];
      }
      for (size_t j = ix; j<valleyIndex.size(); j++) {
	valleyIndex[j-ix+3] = valleyIndex[j];
      }
      peakIndex.resize(peakIndex.size()-ix+2);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 7: replaced peak3 by peak%d \n", (int)ix);
      rule_called.at(7) += 1;
      dumpdata(buff);

      return ( false );
    }
    // nbuffer += printf(&buffer[nbuffer], "Adhoc 7 ok\n");
  }
  return (true);
}

bool NoKeyCall::AdHoc_8 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 8 remove fourth peak if is not equally spaced enough
  if (peakIndex.size() > 3) {
    double v=fabs(log ( (xi[peakIndex[3]] - xi[peakIndex[2]])/( xi[peakIndex[2]] - xi[peakIndex[1]]) ));
    if (v > .51) { // get rid of it
      peakIndex.resize(3);
      TrimValleys(peakIndex, valleyIndex);

      char buff[1000];
      sprintf(buff, "Adhoc 9: 4 peaks after trimming peaks from %d\n", (int)peakIndex.size());
      rule_called.at(8) += 1;
      dumpdata(buff);

      return ( false );
    }
  }
  return (true);
}

bool NoKeyCall::AdHoc_9 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 9, 5+ peaks, only allow 4 peaks
  if (peakIndex.size() > 4) {
    peakIndex.resize(4);
    TrimValleys(peakIndex, valleyIndex);

    char buff[1000];
    sprintf(buff, "Adhoc 9: 4 peaks after trimming peaks from %d\n", (int)peakIndex.size());
    rule_called.at(9) += 1;
    dumpdata(buff);

    return ( false );
  }
  return (true);
}

void NoKeyCall::dumpdata(char *buff)
{
  //return;

  // if (rule_called.at(adhoc) ==1)
  if ((beadx == 57) && (beady == 140))
  {
    nbuffer += sprintf(&buffer[nbuffer], "%s", buff);
    nbuffer += sprintf(&buffer[nbuffer], "At x=%d, y=%d, original npeak = %d, now npeak = %d; ", beadx, beady, npeak_orig, (int)peakIndex.size() );  
    if (npeak_orig > 0) {
      for (size_t i=0; i<peakIndex.size(); i++)
	nbuffer += sprintf(&buffer[nbuffer], "f(%f)=%f ", xi[peakIndex[i]],density[peakIndex[i]]);
      nbuffer += sprintf(&buffer[nbuffer], "\n");    
    }
  
    nbuffer += sprintf(&buffer[nbuffer], " xi = ");
    for (size_t i=0; i<density.size(); i++)
      nbuffer += sprintf(&buffer[nbuffer], "%f ", xi[i]);
    nbuffer += sprintf(&buffer[nbuffer], "\nyi = ");
    for (size_t i=0; i<density.size(); i++)
      nbuffer += sprintf(&buffer[nbuffer], "%f ", density[i]);
    nbuffer += sprintf(&buffer[nbuffer], "\n");
    nbuffer += sprintf(&buffer[nbuffer], "\n");
  }
}

/**
 * trim peakIndex and valleyIndex to index values of xi and density
 * only between quantiles [cutoff, 1-cutoff] of data
 * xi, peakIndex and valleyIndex assumed sorted
 */
int NoKeyCall::TrimExtremalPeaks(std::vector<double>& data, std::vector<double> const& xi, double const _cutoff, std::vector<size_t> &valleyIndex, std::vector<size_t> &peakIndex)
{
  
  // calculate lower %ile
  size_t nData = data.size();
  int nLow = (int)(_cutoff*nData -.5);
  if (nLow < 0){
    nLow = 0;
  }

  std::nth_element(data.begin(), data.begin()+nLow, data.end());
  double percentileLower = *(data.begin()+nLow);

  // calculate upper %ile
  int nHigh = (int)((1-_cutoff)*nData -.5);
  if (nHigh < 0){
    nHigh = 0;
  }

  std::nth_element(data.begin(), data.begin()+nHigh, data.end());
  double percentileUpper = *(data.begin()+nHigh);

  // ignore any peaks outside this range
  vector<size_t>::iterator rit = peakIndex.end()-1;
  while ( rit>=peakIndex.begin() && xi[ *rit ]>percentileUpper )
    rit--;

  peakIndex.erase( rit+1, peakIndex.end());

  if (peakIndex.size() == 0) {
    valleyIndex.clear();
    return (peakIndex.size());
  }

  vector<size_t>::iterator it = peakIndex.begin();
  while ( it<peakIndex.end() && xi[ *it ]<percentileLower )
    it++;

  peakIndex.erase(peakIndex.begin(), it);

  // trim valleys to be between peaks
  if ( peakIndex.size() == 0) {
    valleyIndex.clear();
    return (peakIndex.size());
  }

  if (valleyIndex.size() == 0)
    return (peakIndex.size());
  
  TrimValleys(peakIndex, valleyIndex);
  return (peakIndex.size());
}

/**
 * trim valleyIndex to have values strictly contained within peakIndex values
 * both arguments assumed already sorted
 */
void NoKeyCall::TrimValleys(vector<size_t> const& peakIndex, vector<size_t> valleyIndex)
{
  if (peakIndex.size() == 0) {
    valleyIndex.clear();
    return;
  }
  // delete any valleys after the last peak
  vector<size_t>::iterator v_rit = valleyIndex.end()-1;
  while ( v_rit >= valleyIndex.begin() && *v_rit > *(peakIndex.end()-1) )
    v_rit--;

  valleyIndex.erase( v_rit+1, valleyIndex.end() );
  if (valleyIndex.size() == 0)
    return;

  // delete any valleys before the first peak
  vector<size_t>::iterator v_it = valleyIndex.begin();
  while ( v_it < valleyIndex.end() && *v_it < *(peakIndex.begin()) )
    v_it++;

  valleyIndex.erase( valleyIndex.begin(), v_it );
}

/*
// In Analysis.cpp
#include "H5Replay.h"
...
  // Write processParameters.parse file now that processing is about to begin
  my_progress.WriteProcessParameters(inception_state);

  //std::string h5file = std::string(inception_state.sys_context.results_folder) + "/nokeynorm.h5";
  //H5ReplayRecorder recorder = H5ReplayRecorder(h5file, NULL);
  //recorder.CreateFile();
...

void RegionalizedData::ComputeSimpleFlowValues(float nuc_flow_frame_width, int flow, H5ReplayRecorder& recorder)
{
  // create the dataset
  vector<hsize_t> chunk_dims(3);
  chunk_dims[0] = region->w;
  chunk_dims[1] = region->h;
  chunk_dims[2] = NUMFB;

  recorder.CreateDataset(chunk_dims);

  // start of the nuc rise in units of frames in time_c.t0;
  // ballpark end of the nuc rise in units of frames
  float t_end = time_c.t0 + nuc_flow_frame_width * 2/3;

  // index into part of trace that is important
  size_t t0_ix = time_c.SecondsToIndex(time_c.t0/time_c.frames_per_second);
  size_t t_end_ix = time_c.SecondsToIndex(t_end/time_c.frames_per_second);
  NoKeyCall nkc = NoKeyCall();
  std::vector<float> signalInFlow(my_beads.numLBeads, 0);
  for (int fnum=0; fnum < NUMFB; fnum++)
  {
    for (int ibd=0; ibd < my_beads.numLBeads; ibd++)
    {
      // approximate measure related to signal computed from trace
      // trace is assumed zero'd
      vector<float> trace(time_c.npts(), 0);
      my_trace.AccumulateSignal(&trace[0],ibd,fnum,time_c.npts());
      signalInFlow[ibd] = nkc.Incorporation(t0_ix, t_end_ix, trace);
    }

    // write out the data, order matches bfmask.bin
    vector<hsize_t> offset(3);
    offset[0] = region->row;
    offset[1] = region->col;
    offset[2] = fnum+flow+1-NUMFB;

    vector<hsize_t> count(3);
    count[0] = region->h;
    count[1] = region->w;
    count[2] = 1;

    vector<hsize_t> offset_in(1);
    offset_in[0] = 0;

    vector<hsize_t> count_in(1);
    count_in[0] = region->w*region->h;

    vector<hsize_t> extension(3);
    extension[0] = region->row+region->h;
    extension[1] = region->col+region->w;
    extension[2] = flow+1;

    vector<float> out(region->w*region->h, -1);
    for (int ibd=0; ibd < my_beads.numLBeads; ibd++) {
      int i = my_beads.params_nn[ibd].y*region->w + my_beads.params_nn[ibd].x;
      out[i] = signalInFlow[ibd];
    }

    recorder.ExtendDataSet(extension); // extend if necessary
    fprintf(stdout, "H5 buffer, region %d: %d, %d, %d\n", region->index, (int)extension[0], (int)extension[1], (int)extension[2]);
    recorder.Write(offset, count, offset_in, count_in, &out[0]);
  }
}
*/
