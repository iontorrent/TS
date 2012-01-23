/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "DiffEqModel.h"
#include "DNTPRiseModel.h"
#include "EmphasisVector.h"

using namespace std;

RcppExport SEXP CalculateEmphasisVectorR(
    SEXP R_emphasis, SEXP R_hp_length,
    SEXP R_timeFrame, SEXP R_frames_per_point, 
    SEXP R_time_center,
    SEXP R_amplitude_multiplier, SEXP R_emphasis_width, SEXP R_emphasis_amplitude
) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    int homopolymer_length = Rcpp::as<int> (R_hp_length);
    vector<float> emphasis_params = Rcpp::as<vector <float> > (R_emphasis);
    vector<int> frames_per_point = Rcpp::as<vector <int> > (R_frames_per_point);
    vector<float> time_frame = Rcpp::as< vector<float> > (R_timeFrame);

    float emphasis_width = Rcpp::as<float> (R_emphasis_width);
    float amplitude_multiplier = Rcpp::as<float> (R_amplitude_multiplier);
    float emphasis_amplitude = Rcpp::as<float> (R_emphasis_amplitude);
    float time_center = Rcpp::as<float> (R_time_center);

    int my_frame_len = time_frame.size();

    float  *old_time_frame;

    old_time_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_time_frame[i] = time_frame[i];
    }

    int *old_frames_per_point;
    old_frames_per_point = new int [my_frame_len];
    for (int i=0; i<my_frame_len; i++){
      old_frames_per_point[i] = frames_per_point[i];
    }
  
    float *old_emphasis_params;
    int emphasis_len = emphasis_params.size();
    old_emphasis_params = new float [emphasis_len];
    for (int i=0; i<emphasis_len; i++){
        old_emphasis_params[i] = emphasis_params[i];
    }
    
    float *vect_out;
    vect_out = new float [my_frame_len];

    // output in frames synchronized
    // use the same nucleotide rise function as the bkgmodel setup uses
    int my_pts = GenerateIndividualEmphasis(vect_out, homopolymer_length, old_emphasis_params, my_frame_len ,time_center,old_frames_per_point, old_time_frame, amplitude_multiplier,emphasis_width, emphasis_amplitude);

    vector<double> my_vect_out;
    for (int i=0; i<my_frame_len; i++)
      my_vect_out.push_back(vect_out[i]);

      RcppResultSet rs;
      rs.add("Emphasis",      my_vect_out);
      rs.add("EmphasisScale", my_pts);
      ret = rs.getReturnList();

    delete[] vect_out;


    delete[] old_time_frame;
    delete[] old_frames_per_point;
    delete[] old_emphasis_params;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}
