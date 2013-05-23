/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SAMPLECLONALITY_H
#define SAMPLECLONALITY_H

#include <assert.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <limits>

#include "Stats.h"

#if 0  // incrementally write debug output to buffer, set to 1 for debugging
#define NBUFF 60000
#define scprint(obj, format, args...)				\
  { (obj)->nbuffer += sprintf(&((obj)->buffer)[(obj)->nbuffer], format, ##args); \
    assert ((obj)->nbuffer < NBUFF); }
#else
#define NBUFF 0
#define scprint( args... ) ((void)0)
#endif

class Clonality {

 public:
  Clonality();
  ~Clonality();

  void clear();
  void flush();
  char buffer[NBUFF];
  int nbuffer;

  float Incorporation(size_t const t0_ix, size_t const tend_ix, std::vector<float> const& trace) const;
  float Incorporation(size_t const t0_ix, size_t const tend_ix, std::vector<float> const& trace, int const fnum) const;
  void NormalizeSignal(std::vector<float>& signalInFlow, std::vector<float> const& key_zeromer, std::vector<float> const& key_onemer);
  void AddClonalPenalty(std::vector<float> const& signalInFlow, std::vector<int> const& keyLen, int const fnum, std::vector<int>& flowCount, std::vector<float>& penalty);

  void SetShiftedBkg(std::vector<float> const& shifted_bkg_in);

  int npoints;
  double cutoff;

  double bw_fac;

  double range_limit_low;
  double range_limit_high;
  double bw_increment;
  
  double min_mass;
  double set_too_close(double val);
  double min_separation_0_1;
  double min_separation_1_2;

  size_t max_allowable_peaks;

 private:
  std::vector<float> shifted_bkg;

  double GetBandWidth(std::vector<double> const& data);
  void XLimits(vector<double> const& data, double const bandwidth, vector<double> &out, double const range_limit_low, double const range_limit_high,  double const bw_increment);
  void GetPeaks(std::vector<float> const& dat, std::vector<double>& out);

  void getpeaksInternal(std::vector<double>& data, double const _cutoff, double const _width, int const _npoints, std::vector<double>& weights, std::vector<double>& out);
  void XLimits(std::vector<double> const& data, double const bandwidth, std::vector<double>& out);
  int TrimExtremalPeaks(std::vector<double>& data, std::vector<double> const& xi, double const _cutoff, std::vector<size_t> &valleyIndex, std::vector<size_t> &peakIndex);
  void TrimValleys(std::vector<size_t> const& peakIndex, std::vector<size_t> valleyIndex);
  int ApplyAdHocPeakRemoval(std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_1 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_2 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_3 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_4 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_5 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_6 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  double too_close;
};

#endif // SAMPLECLONALITY_H
