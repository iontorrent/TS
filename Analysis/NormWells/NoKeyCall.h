/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef NOKEYCALL_H
#define NOKEYCALL_H

#include <assert.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <limits>

#include "Stats.h"

class NoKeyCall {

 public:
  NoKeyCall();
  ~NoKeyCall();

  void clear();
  char buffer[20000];
  int nbuffer;

  float Incorporation(size_t const t0_ix, size_t const tend_ix, std::vector<float> const& trace);
  void NormalizeBeadSignal(std::vector<float> const& signal, std::vector<double>& peaks, std::vector<float>& normedSignal);
  void AddClonalPenalty(std::vector<float> const& signalInFlow, std::vector<float>& penalty, std::vector<int>& flowCount, int fnum, std::vector<int>& keyLen);

  void GetPeaks(std::vector<float> const& dat, std::vector<double>& out);

  int npoints;
  double cutoff;
  double width;

  double range_limit;
  double bw_increment;
  
  double min_mass;
  double set_too_close(double val);
  double min_separation_0_1;
  double min_separation_1_2;

  size_t max_allowable_peaks;

  std::vector<int> rule_called;
  int adhoc;
  std::vector<double> density;
  std::vector<double> xi;
  std::vector<size_t> peakIndex;
  int npeak_orig;

  void SetBeadLocation(int x, int y) { beadx=x; beady=y;}

 private:
  void getpeaksInternal(std::vector<double>& data, double const _cutoff, double const _width, int const _npoints, std::vector<double>& weights, std::vector<double>& out);
  int TrimExtremalPeaks(std::vector<double>& data, std::vector<double> const& xi, double const _cutoff, std::vector<size_t> &valleyIndex, std::vector<size_t> &peakIndex);
  void TrimValleys(std::vector<size_t> const& peakIndex, std::vector<size_t> valleyIndex);
  int ApplyAdHocPeakRemoval(std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_0 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_1 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_2 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex, bool no_ave );
  bool AdHoc_3 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_4 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_5 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_6 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_7 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_8 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  bool AdHoc_9 (std::vector<double> const& xi, std::vector<double> const& density, std::vector<size_t>& valleyIndex, std::vector<size_t>& peakIndex );
  double too_close;

  int beadx;
  int beady;
  void dumpdata(char *buff);

};

#endif // NOKEYCALL_H
