/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iomanip>
#include "bivariate_gaussian.h"

using namespace std;
using namespace arma;

bivariate_gaussian::bivariate_gaussian()
{
  set_size();
  _two_pi_sqrt_det_sigma = 0.0;
}

bivariate_gaussian::bivariate_gaussian(vec mean, mat sigma)
{
  set_size();
  _mean = mean;
  _sigma = sigma;
  _inv_sigma= inv(_sigma);
  _inv_trans_chol_sigma = inv(trans(chol(_sigma)));
  _two_pi_sqrt_det_sigma = (2 * 3.141593 * sqrt(det(_sigma)));
}

void bivariate_gaussian::set_size() {
  _mean.set_size(2);
  _sigma.set_size(2,2);
  _inv_sigma.set_size(2,2);
  _inv_trans_chol_sigma.set_size(2,2);
}

ostream& operator<<(ostream& out, const bivariate_gaussian& g)
{
	out << g._mean  << endl;
	out << g._sigma << endl;
	return out;
}

