/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <deque>
#include <vector>
#include <armadillo>
#include <Rcpp.h>
#include "mixed.h"
#include "bivariate_gaussian.h"

using namespace std;
using namespace arma;

RcppExport SEXP percentPositive(SEXP RIonoGram, SEXP RCutoff)
{
	vector<double> ionoGram = Rcpp::as<vector<double> >(RIonoGram);
	double cutoff = Rcpp::as<double>(RCutoff);
	double ppf    = percent_positive(ionoGram.begin(), ionoGram.end(), cutoff);

	return Rcpp::List::create(Rcpp::Named("ppf") = ppf);
}

RcppExport SEXP sumFractionalPart(SEXP RIonoGram)
{
	vector<double> ionoGram = Rcpp::as<vector<double> >(RIonoGram);
	double ssq = sum_fractional_part(ionoGram.begin(), ionoGram.end());

	return Rcpp::List::create(Rcpp::Named("ssq") = ssq);
}

RcppExport SEXP fitNormals(SEXP RPPF, SEXP RSSQ)
{
	deque<float> ppf = Rcpp::as<deque<float> >(RPPF);
	deque<float> ssq = Rcpp::as<deque<float> >(RSSQ);

	//vec2  mean[2];
	//mat22 sigma[2];
	//vec2  prior;
	vec mean[2];
	mat sigma[2];
	vec prior;
    for(int i=0; i<2; ++i){
        mean[i].set_size(2);
        sigma[i].set_size(2,2);
    }

// I assume here (possibly falsely) that the default options are good
  PolyclonalFilterOpts local_opts;

	bool converged = fit_normals(mean, sigma, prior, ppf, ssq, false,local_opts); // avoid verbose

	// (Wrapping the results would be much simpler with RcppArmadillo.)
	Rcpp::NumericVector RCloneMean  = Rcpp::wrap(mean[0]);
	Rcpp::NumericVector RMixedMean  = Rcpp::wrap(mean[1]);
	Rcpp::NumericVector RPrior      = Rcpp::wrap(prior);

	Rcpp::NumericMatrix RCloneSigma(2,2);
	Rcpp::NumericMatrix RMixedSigma(2,2);
	for(int r=0; r<2; ++r){
		for(int c=0; c<2; ++c){
			RCloneSigma(r,c) = sigma[0](r,c);
			RMixedSigma(r,c) = sigma[1](r,c);
		}
	}

	return Rcpp::List::create(Rcpp::Named("converged")  = converged,
                              Rcpp::Named("cloneMean")  = RCloneMean,
                              Rcpp::Named("mixedMean")  = RMixedMean,
                              Rcpp::Named("cloneSigma") = RCloneSigma,
                              Rcpp::Named("mixedSigma") = RMixedSigma,
                              Rcpp::Named("prior")      = RPrior);

}

RcppExport SEXP distanceFromMean(SEXP RMean, SEXP RSigma, SEXP RX)
{
	Rcpp::NumericVector tmpMean(RMean);
	Rcpp::NumericMatrix tmpSigma(RSigma);
	Rcpp::NumericVector tmpX(RX);

	//vec2  mean;
	//mat22 sigma;
	//vec2  x;
	vec mean(2);
	mat sigma(2,2);
	vec x(2);

	mean  << tmpMean(0)    << tmpMean(1);
	sigma << tmpSigma(0,0) << tmpSigma(0,1) << endr
	      << tmpSigma(1,0) << tmpSigma(1,1) << endr;
	x     << tmpX(0)       << tmpX(1);

	bivariate_gaussian g(mean, sigma);
	
	return Rcpp::wrap(g.sd(x));
}


