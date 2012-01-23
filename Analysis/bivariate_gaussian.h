/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BIVARIATE_GAUSSIAN_H
#define BIVARIATE_GAUSSIAN_H

#include <cmath>
#include <ostream>
#include <armadillo>

class bivariate_gaussian {
public:
	bivariate_gaussian();
	bivariate_gaussian(arma::vec2 mean, arma::mat22 sigma);

	// probablility density at x:
	inline double pdf(arma::vec2 x) const
	{
		return std::exp(-0.5 * arma::as_scalar(arma::trans(x-_mean) * _inv_sigma * (x-_mean)) / _two_pi_sqrt_det_sigma);
	}

	// distance from mean in standard deviations:
	inline double sd(arma::vec2 x) const
	{
		return arma::norm(_inv_trans_chol_sigma * (x-_mean), 2);
	}

private:
	friend std::ostream& operator<<(std::ostream& out, const bivariate_gaussian& g);

	arma::vec2  _mean;
	arma::mat22 _sigma;
	arma::mat22 _inv_sigma;
	arma::mat22 _inv_trans_chol_sigma;
	double      _two_pi_sqrt_det_sigma;
};

std::ostream& operator<<(std::ostream& out, const bivariate_gaussian& g);

#endif // BIVARIATE_GAUSSIAN_H

