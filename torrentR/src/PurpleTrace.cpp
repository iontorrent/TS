/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "DiffEqModel.h"

using namespace std;

// This set of three functions do hydorgen ion accounting
// red_hydrogen is >cumulative< hydrogen newly generated
// blue_hydrogen is instantaneous background measured in an "empty" well
// delta_frame is time between data points - may be variable
// tau_bead = buffering*conductance in well
// etb_ratio is ratio tau_empty/tau_bead, [less than 1]

RcppExport SEXP PurpleSolveTotalTraceR(SEXP R_blue_hydrogen, SEXP R_red_hydrogen, SEXP R_delta_frame, SEXP R_tau_bead, SEXP R_etb_ratio) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> blue_hydrogen = Rcpp::as< vector<float> >(R_blue_hydrogen);
    vector<float> red_hydrogen = Rcpp::as< vector<float> >(R_red_hydrogen);
    vector<float> delta_frame = Rcpp::as< vector<float> >(R_delta_frame);
    float tau_bead = Rcpp::as<float>(R_tau_bead);
    float etb_ratio = Rcpp::as<float>(R_etb_ratio);

    int my_frame_len = delta_frame.size();

    float *old_blue_hydrogen, *old_red_hydrogen, *old_delta_frame;
    old_blue_hydrogen = new float [my_frame_len];
    old_red_hydrogen = new float [my_frame_len];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_blue_hydrogen[i]=blue_hydrogen[i];
      old_red_hydrogen[i] =red_hydrogen[i];
      old_delta_frame[i] = delta_frame[i];
    }

    float *old_vb_out;

    old_vb_out = new float [my_frame_len];
    
    MathModel::PurpleSolveTotalTrace(old_vb_out,old_blue_hydrogen,old_red_hydrogen, my_frame_len, old_delta_frame,tau_bead,etb_ratio); // generate the trace as seen by C++

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

    ret = Rcpp::List::create(Rcpp::Named("PurpleTrace") = my_vb_out);

    delete[] old_vb_out;

    delete[] old_blue_hydrogen;
    delete[] old_red_hydrogen;
    delete[] old_delta_frame;
    

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP IntegrateRedFromTotalTraceR(SEXP R_purple_obs, SEXP R_blue_hydrogen, SEXP R_delta_frame, SEXP R_tau_bead, SEXP R_etb_ratio) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> blue_hydrogen = Rcpp::as< vector<float> >(R_blue_hydrogen);
    vector<float> purple_obs = Rcpp::as< vector<float> >(R_purple_obs);
    vector<float> delta_frame = Rcpp::as< vector<float> >(R_delta_frame);
    float tau_bead = Rcpp::as<float>(R_tau_bead);
    float etb_ratio = Rcpp::as<float>(R_etb_ratio);

    int my_frame_len = delta_frame.size();

    float *old_blue_hydrogen, *old_purple_obs, *old_delta_frame;
    old_blue_hydrogen = new float [my_frame_len];
    old_purple_obs = new float [my_frame_len];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_blue_hydrogen[i]=blue_hydrogen[i];
      old_purple_obs[i] = purple_obs[i];
      old_delta_frame[i] = delta_frame[i];
    }

    float *old_vb_out;

    old_vb_out = new float [my_frame_len];
    
    MathModel::IntegrateRedFromObservedTotalTrace(old_vb_out,old_purple_obs, old_blue_hydrogen, my_frame_len, old_delta_frame,tau_bead,etb_ratio); 

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

    ret = Rcpp::List::create(Rcpp::Named("IntegratedRed") = my_vb_out);

    delete[] old_vb_out;

    delete[] old_blue_hydrogen;
    delete[] old_purple_obs;
    delete[] old_delta_frame;
    

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP RedSolveHydrogenFlowInWellR( SEXP R_red_hydrogen, SEXP R_i_start, SEXP R_delta_frame, SEXP R_tau_bead) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> red_hydrogen = Rcpp::as< vector<float> >(R_red_hydrogen);
    vector<float> delta_frame = Rcpp::as< vector<float> >(R_delta_frame);
    float tau_bead = Rcpp::as<float>(R_tau_bead);
    int i_start = Rcpp::as<int>(R_i_start);

    int my_frame_len = delta_frame.size();

    float *old_red_hydrogen, *old_delta_frame;
    old_red_hydrogen = new float [my_frame_len];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_red_hydrogen[i] =red_hydrogen[i];
      old_delta_frame[i] = delta_frame[i];
    }

    float *old_vb_out;

    old_vb_out = new float [my_frame_len];
    
    MathModel::RedSolveHydrogenFlowInWell(old_vb_out,old_red_hydrogen, my_frame_len, i_start, old_delta_frame,tau_bead); // generate the trace as seen by C++

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

    ret = Rcpp::List::create(Rcpp::Named("RedTrace") = my_vb_out);

    delete[] old_vb_out;

    delete[] old_red_hydrogen;
    delete[] old_delta_frame;
    

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP BlueSolveBackgroundTraceR( SEXP R_blue_hydrogen, SEXP R_delta_frame, SEXP R_tau_bead, SEXP R_etb_ratio) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> blue_hydrogen = Rcpp::as< vector<float> >(R_blue_hydrogen);
    vector<float> delta_frame = Rcpp::as< vector<float> >(R_delta_frame);
    float tau_bead = Rcpp::as<float>(R_tau_bead);
    float etb_ratio = Rcpp::as<float>(R_etb_ratio);

    int my_frame_len = delta_frame.size();

    float *old_blue_hydrogen, *old_delta_frame;
    old_blue_hydrogen = new float [my_frame_len];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_blue_hydrogen[i] =blue_hydrogen[i];
      old_delta_frame[i] = delta_frame[i];
    }

    float *old_vb_out;

    old_vb_out = new float [my_frame_len];

    MathModel::BlueSolveBackgroundTrace(old_vb_out,old_blue_hydrogen, my_frame_len, old_delta_frame,tau_bead,etb_ratio); // generate the trace as seen by C++

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

    ret = Rcpp::List::create(Rcpp::Named("BlueTrace") = my_vb_out);

    delete[] old_vb_out;

    delete[] old_blue_hydrogen;
    delete[] old_delta_frame;
    

  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
