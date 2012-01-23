/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MIXED_H
#define MIXED_H

#include <cmath>
#include <algorithm>
#include <deque>
#include <string>
#include <vector>
#include <armadillo>
#include "RawWells.h"
#include "Mask.h"
#include "bivariate_gaussian.h"

bool fit_normals(
	arma::vec2 mean[2],
	arma::mat22 sgma[2],
	arma::vec2& alpha,
	const std::deque<float>& ppf,
	const std::deque<float>& ssq
);

template <class Ran>
inline double percent_positive(Ran first, Ran last, double cutoff=0.25)
{
	return std::count_if(first, last, std::binder2nd<std::greater<double> >(std::greater<double>(),cutoff)) / static_cast<double>(last - first);
}

template <class Ran>
inline double sum_fractional_part(Ran first, Ran last)
{
	double ret = 0.0;
	for(; first<last; ++first){
		double x = *first - round(*first);
		ret += x * x;
	}
	return ret;
}

class clonal_filter {
public:
	clonal_filter() : _ppf_cutoff(0.0), _valid(false) {}

	clonal_filter(bivariate_gaussian clonal, bivariate_gaussian mixed, float ppf_cutoff, bool valid)
	: _clonal(clonal), _mixed(mixed), _ppf_cutoff(ppf_cutoff), _valid(valid) {}

	inline bool filter_is_valid() const {return _valid;}
	inline bool is_clonal(float ppf, float ssq) const
	{
		arma::vec2 x;
		x << ppf << ssq;
		return ppf<_ppf_cutoff and _clonal.pdf(x) > _mixed.pdf(x);
	}

private:
	bivariate_gaussian _clonal;
	bivariate_gaussian _mixed;
	float              _ppf_cutoff;
	bool               _valid;
};

bool clonal_dist(
	clonal_filter& filter,
	RawWells&      wells,
	Mask&          mask
);

// struct for tracking number of reads caught by each filter:
struct filter_counts {
	filter_counts() : _ninf(0), _nbad_key(0), _nsuper(0), _nmixed(0), _nclonal(0), _nsamp(0) {}

	int _ninf;
	int _nbad_key;
	int _nsuper;
	int _nmixed;
	int _nclonal;
	int _nsamp;
};

std::ostream& operator<<(std::ostream& out, const filter_counts& counts);

// Make a clonality filter from a sample of reads from a RawWells file.
// Record number of reads in sample that are caught by each filter.
void make_filter(clonal_filter& filter, filter_counts& counts, int nlib, Mask& mask, RawWells& wells, const std::vector<int>& key_ionogram);

template <class Ran>
void NormalizeLib(Ran beg, Ran end)
{
    // Hard wired for TCAG and XDB.
    double keyMean = (beg[0] + beg[2] + beg[5]) / 3.0;
	if(keyMean > 0.0){
	    while(beg < end)
    	    *beg++ /= keyMean;
	}
}

template <class T>
bool all_finite(T first, T last)
{
	bool finite = true;
	for(; first<last; ++first){
		if(*first > 100){
			finite = false;
			break;
		}
	}
	return finite;
}

#endif // MIXED_H

