/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FitDensity.h"
#include "SampleClonality.h"
#include "Stats.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>


Clonality::Clonality()
{
  clear();

  // fitting parameters to density
  // npoints = 128;
  npoints = 512;
  cutoff = .05;
  datasize = 4000; // how much data to use for density fit, 0 => all

  // bandwidth
  bw_fac = .035; //.05, tuned for 10000 points
  tuning_count = 10000.0;
  convergence_rate = -.33;

  // outlier trimming
  range_limit_low = -2.0f;
  range_limit_high = 4.0f;
  bw_increment = 3.0f;

  // adhoc rule 1
  // min_mass = 0.05;
  min_mass = 0.03*128/npoints;
  // adhoc rule 4
  too_close = fabs(log(0.75));
  // adhoc rule 3
  min_separation_0_1 = 1.3; // 1.6;
  // adhoc rule 5
  min_separation_1_2 = 1.3;

}

void Clonality::clear()
{
  nbuffer = 0;
  for (int i=0; i<NBUFF; i++)
    buffer[i] = '\0';
}

void Clonality::flush()
{
  if (nbuffer > 0)
    fprintf(stdout, "%s\n", buffer);

  clear();
}

Clonality::~Clonality()
{
  // assert (nbuffer<60000); // otherwise all sorts of weird stuff could happen
}

double Clonality::set_too_close(double val)
{
  too_close = fabs(log(val));
  return (val);
} 

void Clonality::SetShiftedBkg(std::vector<float> const& shifted_bkg_in)
{
  assert(shifted_bkg.size() == 0);
  shifted_bkg.assign(shifted_bkg_in.begin(), shifted_bkg_in.end());
}

// a very crude measure of incorporation, not zeroed for zeromer estimate
float Clonality::Incorporation(size_t const t0_ix, const size_t t_end_ix, vector<float> const& trace) const
{
  float incorporation = 0;
  for (size_t i = t0_ix; i < t_end_ix; i++)
    incorporation += trace[i];

  return incorporation;
}

// a very crude measure of incorporation, zero-ed via the empty trace
float Clonality::Incorporation(size_t const t0_ix, const size_t t_end_ix, vector<float> const& trace, int const fnum) const
{
  float incorporation = 0;
  for (size_t i = t0_ix; i < t_end_ix; i++)
    incorporation += trace[i] - shifted_bkg[i + fnum*trace.size()];

  return incorporation;
}

void Clonality::NormalizeSignal(std::vector<float>& signalInFlow, std::vector<float> const& key_zeromer, std::vector<float> const& key_onemer)
{
  // try key normalization
  // calculate per bead mean across zeromer incorporations
  // calculate per bead mean across 1-mer incorporations
  // per bead, incorporation minus mean zeromer and dividing by mean 1-mer

  for (int ibd = 0; ibd < (int)signalInFlow.size(); ibd++)
  {
    float den = key_onemer[ibd] - key_zeromer[ibd];
    //float oldval = 0;
    if (den > 0) {
      //oldval = signalInFlow[ibd];
      signalInFlow[ibd] = (signalInFlow[ibd] - key_zeromer[ibd])/den;
    }
    else { // do not use this value
      signalInFlow[ibd] = numeric_limits<float>::quiet_NaN();
    }
  }
}

void Clonality::AddClonalPenalty(std::vector<float> const& signalInFlow, std::vector<int> const& keyLen, int const fnum, std::vector<int>& flowCount, std::vector<float>& penalty)
{
  // adds the penalty for every bead in this flow
  
  // characterize the flow using signal across beads
  std::vector<double> peaks;
  GetPeaks(signalInFlow, peaks);
  scprint( this, "adjusted_peaks=%d;\n", (int)peaks.size());

  // debug_peaks.assign(peaks.begin(), peaks.end());

  size_t npeaks = peaks.size();
  if (npeaks < 2) {  // this flow needs to have at least 2 peaks to be used
    return;
  }

  // if signal is too low a bead will rejected
  float too_low = peaks[0] -(peaks[1] - peaks[0])*.8;
  scprint( this,"too_low = peaks[0] = %f - (peaks[1]=%f -peaks[0])*.8 = %f\n", peaks[0], peaks[1], too_low);

  // beads with signals near the 5th peak are 4-mers, ignore higher
  float missing = (npeaks < 5) ? 5 - npeaks : 0;
  float too_high = peaks[npeaks-1] + (missing + .5)*(peaks[1]-peaks[0]);
  scprint( this,"too_high = peaks[%d] = %f + (missing +.5 = %f) * (peaks[1] = %f - peaks[0] = %f) = %f\n", (int)npeaks-1, peaks[npeaks-1], (missing+.5), peaks[1], peaks[0], too_high);

  // given the flow characteristics, add the flow penalty to the overall penalty
  // for each bead.  If a penalty is applied, also increment flowCount for the bead
  for (int ibd=0; ibd < (int)signalInFlow.size(); ibd++){
    // key flows don't get penalized
    if (fnum < keyLen[ibd])
      continue;

    // maximum penalty already applied
#if __cplusplus >= 201103L
    if ( std::isinf(penalty[ibd]) )
#else
    if ( isinf(penalty[ibd]) )
#endif
      continue;

    // reject any bead where signal is not present by using maximum penalty
    if ( isnan( signalInFlow[ibd] )) {
      penalty[ibd] = numeric_limits<float>::infinity();
      continue;
    }

    // reject any beads with signals that are too low using maximum penalty
    if (signalInFlow[ibd] < too_low) {
      // fprintf(stdout, "signalInFlow[%d]=%f < %f\n", (int)ibd, signalInFlow[ibd], too_low);
      penalty[ibd] = numeric_limits<float>::infinity();
      continue;
    }

    // find the nearest peak and calculate the penalty
    bool found = false;
    for (size_t i=0; i < npeaks-1; i++) {
      if ( (signalInFlow[ibd] >= peaks[i]*1.5 -peaks[i+1]*.5) &&
	   (signalInFlow[ibd] < (peaks[i]+peaks[i+1])*.5) ) {
	float w = 1.0f/(i+1);  // weight some peaks more heavily?
	penalty[ibd] += (signalInFlow[ibd]-peaks[i])*(signalInFlow[ibd]-peaks[i])*w;
	flowCount[ibd]++;
	found = true;
	break;
      }
    }
    if (found)
      continue;

    if ( (signalInFlow[ibd] >= (peaks[npeaks-2]+peaks[npeaks-1])*.5 ) &&
	 (signalInFlow[ibd] < (peaks[npeaks-1]*1.5 - peaks[npeaks-1]*.5) ) )
    {
      float w = 1.0f/npeaks;
      penalty[ibd] +=  (signalInFlow[ibd]-peaks[npeaks-1])*(signalInFlow[ibd]-peaks[npeaks-1])*w;
      flowCount[ibd]++;
      continue;
    }

    // do not increase the penalty if signal is higher than known peaks
    // but not too high.  Do not penalize the bead in this flow
    if ( (signalInFlow[ibd] >= (peaks[npeaks-1]*1.5 - peaks[npeaks-2]*.5) ) &&
	 (signalInFlow[ibd] < too_high))
    {
      continue;
    }
	
    // reject any beads with signals that are too high
    if (signalInFlow[ibd] >= too_high){
      // fprintf(stdout, "signalInFlow[%d]=%f > %f\n", (int)ibd, signalInFlow[ibd], too_high);
      penalty[ibd] = numeric_limits<float>::infinity();
      continue;
    }
  }
}

/** Input is vector of data, returns bandwidth for density estimation
 * uses the rationale: P[0-mer] = 2/3, P[1-mer] =1/3*2/3, sum=8/9 = 90%
 * returns 0.1 * (Expected 1-mer - Expected 0-mer)
 * can be 0 if data length is small or has repeated values
 */
double Clonality::GetBandWidth(std::vector<double> const& data)
{
  std::vector<size_t> order;
  ionStats::sort_order(data.begin(), data.end(), order);
  
  size_t o33 = (int)(data.size()*.33 + .5);  // should be 0-mer
  size_t o90 = (int)(data.size()*.9 +.5) -1; // should be 1-mer
  //double bw = .2 * (data.at(order.at(o90)) - data.at(order.at(o05)));
  double bw = bw_fac * (data.at(order.at(o90)) - data.at(order.at(o33)));

  // TODO: for the case of a 50/50 mix of 2 barcodes we
  // should be using 25th percentile & 75th percentile, does this matter?

  // tuning done for 10000 points (regions of a Proton thumbnail)
  // optimal bandwidth probably order n**(-1/3) using histogram bin sizes
  // eg Freedman-Diaconis rule
  if (data.size() < tuning_count){
    bw = bw * pow(data.size()/tuning_count, convergence_rate);
  }

  return (bw);
}

/**
 * XLimits finds the range of the data to accommodate density bandwidth
 * by trimming off values that seem to be below the likely range or
 * are above the likely range
 * range (from, to) is returned in 2-element double vector out
 */
void Clonality::XLimits(vector<double> const& data, double const bandwidth, vector<double> &out, double const range_limit_low, double const range_limit_high,  double const bw_increment)
{
  double from, to, minimum, maximum;
  std::vector<size_t> order;
  ionStats::sort_order(data.begin(), data.end(), order);
  
  size_t o05 = (int)(data.size()*.33 + .5);  // should be 0-mer
  size_t o90 = (int)(data.size()*.9 +.5) -1;   // should be 1-mer
  double zero_mer = data.at(order.at(o05));
  double one_mer = data.at(order.at(o90));
  double span = one_mer - zero_mer;
  minimum = zero_mer + range_limit_low * span;
  maximum = one_mer + range_limit_high * span;
  from = minimum - bw_increment * bandwidth;
  to = maximum + bw_increment * bandwidth;
  scprint( this,"Clonality::XLimits; zero_mer=%f; one_mer=%f; bw = %f; range_limit_low = %f; range_limit_high=%f; min = %f; max = %f; from = %f; to = %f;\n ", zero_mer, one_mer, bandwidth, range_limit_low, range_limit_high, minimum, maximum, from, to);
  out[0] = from;
  out[1] = to;
}

bool RelativelyPrime (size_t a, size_t b) { // Assumes a, b > 0
  assert( (a > 0) && (b > 0) );
  for ( ; ; ) {
    if (!(a %= b)) return b == 1 ;
    if (!(b %= a)) return a == 1 ;
  }
}

/**
 * Given target_count number to sample from data_count members
 * find a sampling rate that when used to increment an index 
 * more or less evenly picks modulo data_count.
 * Making the rate relatively prime wrapping means that if an increment
 * wraps back to the beginning of the vector it will find different
 * indices.
 */
size_t SamplingRate(size_t target_count, size_t data_count)
{
  assert( (target_count > 0) && (data_count > 0) );
  size_t rate = 1;

  if (target_count >= data_count)
    return(rate);

  if (target_count < data_count){ 
    rate = (data_count/target_count) + 1;
    while ( (rate<data_count) && !RelativelyPrime (data_count, rate)){
      rate++;
    }
  }
  return(rate);
}

/** usage: GetPeaks(in_data, peaks);
 * output in peaks correspond to the sorted positions of peaks of a density
 * will be fit using bandwith bw to input vector in_data.
 * nans removed from in_data
 * values in peaks are ordered low to high
 */
void Clonality::GetPeaks(vector<float> const& in_data, vector<double>& peaks){
  if (datasize == 0)
    datasize = in_data.size();

  size_t nn = (datasize < in_data.size()) ? datasize : in_data.size();
  size_t rate=SamplingRate(nn, in_data.size());
  
  vector<double> data(nn, 0);
  size_t cnt = 0;
  size_t ix = 0;;
  for (size_t i=0; i < nn; i++) {
    ix += rate;
    int iix = ix % in_data.size();
    if ( ! isnan (in_data[iix]) )
      data[cnt++] = in_data[iix];
  }
  if (cnt > 2) { // this flow has to have beads with usable values to be used
    data.resize(cnt);
    double bw = GetBandWidth(data);
    scprint( this,"bw=%f;\n", bw);
    if (bw > 0) {
      vector<double> weights(data.size(), 1.0f);
      getpeaksInternal(data, cutoff, bw, npoints, weights, peaks);
      return;
    }
  }
  
  // no usable bandwidth or count too low, return a single peak
  scprint( this,"bw=0; cnt=%d;\n", (int)cnt);
  peaks.resize(1);
  peaks[0] = 0;
  size_t n = 0;
  for (size_t i=0; i<cnt; i++) {
#if __cplusplus >= 201103L
    if ( ! std::isinf(data[i]) ) {
#else
    if ( ! isinf(data[i]) ) {
#endif
      n++;
      peaks[0] += data[i];
    }
  }
  assert ( n == cnt ); // no infinities supposed to happen
  if (n>0)
    peaks[0] = peaks[0]/n;
  scprint( this,"npeak=1; ");
  scprint( this,"f(%f)=Inf\n", peaks[0]);
}

void Clonality::getpeaksInternal(vector<double>& data, double const _cutoff, double const bw, int const _npoints, vector<double>& weights, vector<double>& peaks)
{
  int m=data.size(); // number of rows

  assert ( m > 2 );
  assert ( (0 <= _cutoff) && (_cutoff < 1) );
  assert (_npoints > 0);

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
  vector<double> density(_npoints);
  vector<double> xi(_npoints);

  // find limits for range of X
  vector<double> xlimits(2,0);
  // FitDensity::XLimits(data, bw, xlimits, range_limit, bw_increment);
  XLimits(data, bw, xlimits, range_limit_low, range_limit_high, bw_increment);
  scprint( this,"ksdensity, from = %f, to = %f\n", xlimits[0], xlimits[1]);

  // call density and return density in xi, density
  FitDensity::kdensity(data, density, xi, weights, bw, xlimits[0], xlimits[1]);

  // compute overall signal
  double overallSignal = FitDensity::SquaredIntegral(xi, density);

  // find the peaks in (xi, density) and return in peaks
  vector<size_t> peakIndex;
  vector<size_t> valleyIndex;
  int npeak1 = FitDensity::findpeaks(valleyIndex, peakIndex, density, 0.046*overallSignal, xi);
  assert( npeak1 > 0 );

  scprint( this,"before trim, npeak=%d; ", npeak1);  
  for (size_t i=0; i<peakIndex.size(); i++) {
    scprint( this,"f(%f)=%f ", xi[peakIndex[i]],density[peakIndex[i]]);
  }
  scprint( this,"\n");    

  // trim off any peaks in extremes of data,
  // these are meaningless wiggles in the density estimation
  int npeak = TrimExtremalPeaks(data, xi, _cutoff, valleyIndex, peakIndex);
  scprint( this,"after trim, npeak=%d; ", npeak);  
  if (npeak > 0) {
    for (size_t i=0; i<peakIndex.size(); i++) {
      scprint( this,"f(%f)=%f ", xi[peakIndex[i]],density[peakIndex[i]]);
    }
    scprint( this,"\n");    
    // apply ad hoc rules here
    npeak = ApplyAdHocPeakRemoval( xi, density, valleyIndex, peakIndex );
  }
  // debug_density.assign(density.begin(), density.end());
  // debug_xi.assign(xi.begin(), xi.end());

  {
    scprint( this,"xi=[ ");
    for (size_t i=0; i<density.size(); i++)
      scprint( this,"%.3f ", xi[i]);
    scprint( this,"];\n");
    scprint( this,"yi=[ ");
    for (size_t i=0; i<density.size(); i++)
      scprint( this,"%.4f ", density[i]*(xi[1]-xi[0]));
    scprint( this,"];\n");
  }

  peaks.resize(npeak);
  for(size_t i = 0; i < peakIndex.size(); ++i)
    peaks[i] = *(xi.begin() + peakIndex[i]);
  
}

int Clonality::ApplyAdHocPeakRemoval(std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex )
{
  // nothing to do
  if (peakIndex.size() == 0)
    return (0);

  //  *** applying these rules can modify valleyIndex and peakIndex ***
  // passing each rule returns true, failing returns false

  // adhoc rule 1, demand peak0 have plenty of mass
  if ( ! AdHoc_1 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  // adhoc rule 2, 3+ peaks, peak2 must be lower than peak1
  if ( ! AdHoc_2 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  // adhoc rule 3, 2+ peaks, peak1 must be substantially higher than its valley
  if ( ! AdHoc_3 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  // adhoc rule 4, 3+ peaks, merge peak1 and peak2 if they are close
  if ( ! AdHoc_4 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  // adhoc rule 5, 3+ peaks, peak2 must be substantially higher than its valley
  if ( ! AdHoc_5 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  // adhoc rule 6, 4+ peaks, only allow 3 peaks
  if ( ! AdHoc_6 (xi, density, valleyIndex, peakIndex ) )
    return ( (int)peakIndex.size() );

  scprint( this,"Adhoc rules all ok: no peak adjustment\n");
  return ( (int)peakIndex.size() );
}

bool Clonality::AdHoc_1 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 1, demand peak0 have plenty of mass
  // hardcoded for n=128, look for 3/128 = 2.3% of the range
  // and accept only if at least min_mass
  if (peakIndex.size() > 0){
    double w = 0;
    double h = 0;
    if ( ( 0 < peakIndex[0] ) && ( peakIndex[0]+1 < xi.size()) ) {
      w = xi[peakIndex[0]]-xi[peakIndex[0]-1]; // assumes xi are equally spaced
      h = (density[peakIndex[0]-1]+density[peakIndex[0]]+density[peakIndex[0]+1]);
    }
    // else { leave h & w zero, something is weird }
    if ( h*w < min_mass){
      peakIndex.clear();
      valleyIndex.clear();
      scprint( this,"Adhoc 1: 0 peaks because low mass peak[0] density h=%f * w=%f = %f < %f\n", w/2, h, h*w, min_mass);
      return( false);
    }
    scprint( this,"Adhoc 1 ok: peak[0] density h=%f * w=%f = %f >= %f\n", w/2, h, h*w, min_mass);
  }
  return (true);
}

bool Clonality::AdHoc_2 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 2, 3+ peaks, peak2 must be lower than peak1
  if ( peakIndex.size() > 2) {
    if ( density[peakIndex[2]] > density[peakIndex[1]] ) {
      peakIndex.clear();
      valleyIndex.clear();
      scprint( this,"Adhoc 2: 0 peaks because peak[1]=%f at %f < peak[2]=%f at %f\n", density[peakIndex[1]], xi[peakIndex[1]], density[peakIndex[2]], xi[peakIndex[2]]);
      return (false);
    }
    scprint( this,"Adhoc 2 ok: peak[1]=%f at %f >= peak[2]=%f at %f\n",  density[peakIndex[1]], xi[peakIndex[1]], density[peakIndex[2]], xi[peakIndex[2]]);
  }
  return (true);
}

bool Clonality::AdHoc_3 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 3, 2+ peaks, peak1 must be substantially higher than its valley
  if ( peakIndex.size() > 1 ) {
    double peak_valley_ratio = density[peakIndex[1]]/density[valleyIndex[0]];
    if ( peak_valley_ratio < min_separation_0_1 ) {
      scprint( this,"Adhoc 3: 1 peak after removing peak[1] f(%f)=%f, because peak-valley ratio=%f < %f\n", xi[peakIndex[1]], density[peakIndex[1]], peak_valley_ratio, min_separation_0_1);
      peakIndex.resize(1);
      TrimValleys(peakIndex, valleyIndex);
      return ( false );
    }
    scprint( this,"Adhoc 3 ok: peak[1] f(%f)=%f, because peak-valley ratio=%f >= %f\n", xi[peakIndex[1]], density[peakIndex[1]], peak_valley_ratio, min_separation_0_1);
  }
  return (true);
}

bool Clonality::AdHoc_4 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 4, 3+ peaks, merge peak1 and peak2 if they are close
  if ( peakIndex.size() > 2 ) {
    double lr = log ( (xi[peakIndex[1]] - xi[peakIndex[0]])/( xi[peakIndex[2]] - xi[peakIndex[1]]) );
    if (fabs(lr) > too_close ) {
      scprint( this,"Adhoc 4: x0=%f, x1=%f x2=%f |log((x1-x0)/(x2-x1)) = %f| > %f\n", xi[peakIndex[0]], xi[peakIndex[1]], xi[peakIndex[2]], lr, too_close);
      peakIndex[1] = (peakIndex[1] + peakIndex[2])/2;
      peakIndex.resize(2);
      TrimValleys(peakIndex, valleyIndex);
      return ( false );
    }
    scprint( this,"Adhoc 4 ok: x0=%f, x1=%f x2=%f |log((x1-x0)/(x2-x1)) = %f| > %f\n", xi[peakIndex[0]], xi[peakIndex[1]], xi[peakIndex[2]], lr, too_close);
  }
  return (true);
}

bool Clonality::AdHoc_5 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 5, 3+ peaks, peak2 must be substantially higher than its valley
  if ( peakIndex.size() > 2 ) {
    double peak_valley_ratio = density[peakIndex[2]]/density[valleyIndex[1]];
    if ( peak_valley_ratio < min_separation_1_2 ) {
      scprint( this,"Adhoc 5: 2 peaks after removing peak[2] f(%f)=%f because peak-valley ratio=%f < %f\n", xi[peakIndex[2]], density[peakIndex[2]], peak_valley_ratio, min_separation_1_2);
      peakIndex.resize(2);
      TrimValleys(peakIndex, valleyIndex);
      return ( false );
    }
      scprint( this,"Adhoc 5 ok: 2 peaks after removing peak[2] f(%f)=%f because peak-valley ratio=%f < %f\n", xi[peakIndex[2]], density[peakIndex[2]], peak_valley_ratio, min_separation_1_2);
  }
  return (true);
}

bool Clonality::AdHoc_6 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex)
{
  // adhoc rule 6, 4+ peaks, only allow 3 peaks
  if (peakIndex.size() > 3) {
    nbuffer += printf(&buffer[nbuffer], "Adhoc 6: 3 peaks after trimming peaks from %d\n", (int)peakIndex.size());
    peakIndex.resize(3);
    TrimValleys(peakIndex, valleyIndex);
    return ( false );
  }
  return (true);
}

/**
 * trim peakIndex and valleyIndex to index values of xi and density
 * only between quantiles [cutoff, 1-cutoff] of data
 * xi, peakIndex and valleyIndex assumed sorted
 * can re-order data
 */
int Clonality::TrimExtremalPeaks(std::vector<double>& data, std::vector<double> const& xi, double const _cutoff, std::vector<size_t> &valleyIndex, std::vector<size_t> &peakIndex)
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
void Clonality::TrimValleys(vector<size_t> const& peakIndex, vector<size_t> valleyIndex)
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


