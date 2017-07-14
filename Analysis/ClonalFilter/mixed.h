/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MIXED_H
#define MIXED_H

#include <cmath>
#include <algorithm>
#include <deque>
#include <string>
#include <vector>
#include <armadillo>
#include "bivariate_gaussian.h"
#include "polyclonal_filter.h"

class Mask;
class RawWells;

inline float mixed_ppf_cutoff()
{
  return 0.84;
}
inline float mixed_pos_threshold()
{
  return 0.25;
}

template <class T>
bool all_finite (T first, T last)
{
  bool finite = true;
  for (; first<last; ++first)
  {
    if (*first > 100)
    {
      finite = false;
      break;
    }
  }
  return finite;
}

template <class Ran0, class Ran1>
bool key_is_good (Ran0 observed, Ran1 ideal_begin, Ran1 ideal_end)
{
  bool good = true;
  for (Ran1 ideal=ideal_begin; ideal<ideal_end; ++ideal, ++observed)
    good = good and round (*observed) == *ideal;
  return good;
}

template <class Ran>
inline double percent_positive (Ran first, Ran last, double cutoff=mixed_pos_threshold())
{
  return std::count_if (first, last, std::binder2nd<std::greater<double> > (std::greater<double>(),cutoff)) / static_cast<double> (last - first);
}

template <class Ran>
inline double sum_fractional_part (Ran first, Ran last)
{
  double ret = 0.0;
  for (; first<last; ++first)
  {
    double x = *first - round (*first);
    ret += x * x;
  }
  return ret;
}

bool fit_normals (
  arma::vec mean[2],
  arma::mat sgma[2],
  arma::vec& alpha,
  const std::deque<float>& ppf,
  const std::deque<float>& ssq,
  const PolyclonalFilterOpts & opts
);

class clonal_filter
{
  public:
    clonal_filter() : _ppf_cutoff (0.0), _valid (false) {}

    clonal_filter (bivariate_gaussian clonal, bivariate_gaussian mixed, float ppf_cutoff, bool valid)
        : _clonal (clonal), _mixed (mixed), _ppf_cutoff (ppf_cutoff), _valid (valid) {}

    inline bool filter_is_valid() const
    {
      return _valid;
    }
    inline bool is_clonal (float ppf, float ssq) const
    {
      arma::vec x;
      x.set_size (2);
      x << ppf << ssq;

      return ppf<_ppf_cutoff and _clonal.pdf (x) > _mixed.pdf (x);
    }
    inline bool is_clonal (float ppf, float ssq, double stringency) const
    {
      arma::vec x;
      x.set_size (2);
      x << ppf << ssq;
      if(ppf <_ppf_cutoff){
        double clonal_pdf = _clonal.pdf (x);
        double mixed_pdf = _mixed.pdf (x);
        // Avoid divide by zero: If mixed probablity is too small, we'll just call it clonal.
        if (mixed_pdf < 1e-256)
          return true;
        else
          return clonal_pdf / (mixed_pdf + clonal_pdf) > stringency;
      }
      else
        return false;
    }

  private:
    bivariate_gaussian _clonal;
    bivariate_gaussian _mixed;
    float              _ppf_cutoff;
    bool               _valid;
};

bool clonal_dist (
  clonal_filter& filter,
  RawWells&      wells,
  Mask&          mask
);

// struct for tracking number of reads caught by each filter:
struct filter_counts
{
  filter_counts() : _ninf (0), _nbad_key (0), _nsuper (0), _nmixed (0), _nclonal (0), _nsamp (0) {}

  int _ninf;
  int _nbad_key;
  int _nsuper;
  int _nmixed;
  int _nclonal;
  int _nsamp;
};

std::ostream& operator<< (std::ostream& out, const filter_counts& counts);

// Make a clonality filter from a sample of reads from a RawWells file.
// Record number of reads in sample that are caught by each filter.
void make_filter (clonal_filter& filter, filter_counts& counts, Mask& mask, RawWells& wells, const std::vector<int>& key_ionogram, const PolyclonalFilterOpts & opts);

// Make a clonality filter given ppf and ssq for a random sample of wells:
void make_filter (clonal_filter& filter, filter_counts& counts, const std::deque<float>& ppf, const std::deque<float>& ssq, const PolyclonalFilterOpts & opts);

#endif // MIXED_H

