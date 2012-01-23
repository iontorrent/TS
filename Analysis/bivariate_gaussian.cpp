/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iomanip>
#include "bivariate_gaussian.h"

using namespace std;
using namespace arma;

bivariate_gaussian::bivariate_gaussian()
: _two_pi_sqrt_det_sigma(0.0)
{
}

bivariate_gaussian::bivariate_gaussian(vec2 mean, mat22 sigma)
: _mean(mean)
, _sigma(sigma)
, _inv_sigma(inv(_sigma))
, _inv_trans_chol_sigma(inv(trans(chol(_sigma))))
, _two_pi_sqrt_det_sigma(2 * 3.141593 * sqrt(det(_sigma)))
{
}

ostream& operator<<(ostream& out, const bivariate_gaussian& g)
{
	out << g._mean  << endl;
	out << g._sigma << endl;
	return out;
}

