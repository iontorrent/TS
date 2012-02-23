/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <vector>
#include <string>
#include <iostream>
#include <Rcpp.h>
#include "DiffEqModel.h"
#include "DNTPRiseModel.h"

using namespace std;


RcppExport SEXP CalculateCumulativeIncorporationHydrogensR(
    SEXP R_nuc_rise, SEXP R_sub_steps, 
    SEXP R_deltaFrame, SEXP R_my_start, 
    SEXP R_C, SEXP R_Amplitude, SEXP R_copies, SEXP R_krate, SEXP R_kmax, SEXP R_diffusion) 
{
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
   
    vector<float> nuc_rise = Rcpp::as< vector<float> > (R_nuc_rise);
    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> delta_frame = Rcpp::as< vector<float> > (R_deltaFrame);
    int my_start_index = Rcpp::as<int> (R_my_start);
    float max_concentration = Rcpp::as<float> (R_C);
    float amplitude = Rcpp::as<float> (R_Amplitude);
    float copies = Rcpp::as<float> (R_copies);
    float krate = Rcpp::as<float> (R_krate);
    float kmax = Rcpp::as<float> (R_kmax);
    float diffusion = Rcpp::as<float> (R_diffusion);

    int my_frame_len = delta_frame.size();

    float *old_nuc_rise, *old_delta_frame;
    old_nuc_rise = new float [nuc_rise.size()];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_delta_frame[i] = delta_frame[i]/FRAMESPERSEC; // keep interface transparent for R as this is now in seconds
    }
    // may have more sub-steps than the output
    int nuc_len = nuc_rise.size();
    for (int i=0; i<nuc_len; i++){
      old_nuc_rise[i] = nuc_rise[i];
    }

    float *old_vb_out;

    
    PoissonCDFApproxMemo my_math;
    my_math.Allocate(MAX_HPLEN+1,512,0.05);
    my_math.GenerateValues();
    
    // output in frames synchronized
    old_vb_out = new float [my_frame_len];
    // calculate cumulative hydrogens from amplitude (hp mixture), copies on bead, krate, kmax, diffusion
    // and of course the rate at which nuc is available above the well
    ComputeCumulativeIncorporationHydrogens(old_vb_out,my_frame_len, old_delta_frame, old_nuc_rise, sub_steps,my_start_index,
                                              max_concentration, amplitude, copies, krate,kmax, diffusion,&my_math,true);

    

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

      RcppResultSet rs;
      rs.add("CumulativeRedHydrogens",      my_vb_out);
      ret = rs.getReturnList();

    delete[] old_vb_out;

    delete[] old_nuc_rise;
    delete[] old_delta_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP SimplifyCalculateCumulativeIncorporationHydrogensR(
    SEXP R_nuc_rise, SEXP R_sub_steps, 
    SEXP R_deltaFrame, SEXP R_my_start, 
    SEXP R_C, SEXP R_Amplitude, SEXP R_copies, SEXP R_krate, SEXP R_kmax, SEXP R_diffusion) 
{
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> nuc_rise = Rcpp::as< vector<float> > (R_nuc_rise);
    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> delta_frame = Rcpp::as< vector<float> > (R_deltaFrame);
    int my_start_index = Rcpp::as<int> (R_my_start);
    float max_concentration = Rcpp::as<float> (R_C);
    float amplitude = Rcpp::as<float> (R_Amplitude);
    float copies = Rcpp::as<float> (R_copies);
    float krate = Rcpp::as<float> (R_krate);
    float kmax = Rcpp::as<float> (R_kmax);
    float diffusion = Rcpp::as<float> (R_diffusion);

    int my_frame_len = delta_frame.size();

    float *old_nuc_rise, *old_delta_frame;
    old_nuc_rise = new float [nuc_rise.size()];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_delta_frame[i] = delta_frame[i]/FRAMESPERSEC; // keep interface transparent for R as this is now in seconds
    }
    // may have more sub-steps than the output
    int nuc_len = nuc_rise.size();
    for (int i=0; i<nuc_len; i++){
      old_nuc_rise[i] = nuc_rise[i];
    }

    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len];
    // calculate cumulative hydrogens from amplitude (hp mixture), copies on bead, krate, kmax, diffusion
    // and of course the rate at which nuc is available above the well
    ComputeCumulativeIncorporationHydrogens(old_vb_out,my_frame_len, old_delta_frame, old_nuc_rise, sub_steps,my_start_index,
                                              max_concentration, amplitude, copies, krate,kmax, diffusion,NULL,true);



    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

      RcppResultSet rs;
      rs.add("CumulativeRedHydrogens",      my_vb_out);
      ret = rs.getReturnList();

    delete[] old_vb_out;

    delete[] old_nuc_rise;
    delete[] old_delta_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP DerivativeCalculateCumulativeIncorporationHydrogensR(
    SEXP R_nuc_rise, SEXP R_sub_steps, 
    SEXP R_deltaFrame, SEXP R_my_start, 
    SEXP R_C, SEXP R_Amplitude, SEXP R_copies, SEXP R_krate, SEXP R_kmax, SEXP R_diffusion) 
{
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> nuc_rise = Rcpp::as< vector<float> > (R_nuc_rise);
    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> delta_frame = Rcpp::as< vector<float> > (R_deltaFrame);
    int my_start_index = Rcpp::as<int> (R_my_start);
    float max_concentration = Rcpp::as<float> (R_C);
    float amplitude = Rcpp::as<float> (R_Amplitude);
    float copies = Rcpp::as<float> (R_copies);
    float krate = Rcpp::as<float> (R_krate);
    float kmax = Rcpp::as<float> (R_kmax);
    float diffusion = Rcpp::as<float> (R_diffusion);

    int my_frame_len = delta_frame.size();

    float *old_nuc_rise, *old_delta_frame;
    old_nuc_rise = new float [nuc_rise.size()];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_delta_frame[i] = delta_frame[i]/FRAMESPERSEC; // keep interface transparent for R as this is now in seconds
    }
    // may have more sub-steps than the output
    int nuc_len = nuc_rise.size();
    for (int i=0; i<nuc_len; i++){
      old_nuc_rise[i] = nuc_rise[i];
    }

    float *old_vb_out;
    float *dA_out;
    float *dk_out;

    Dual dA(amplitude,1.0,0.0);
    Dual dk(krate,0.0,1.0);

    // output in frames synchronized
    old_vb_out = new float [my_frame_len];
    dA_out = new float [my_frame_len];
    dk_out = new float [my_frame_len];

    PoissonCDFApproxMemo    *math_poiss = new PoissonCDFApproxMemo;
    math_poiss->Allocate(MAX_HPLEN+1,512,0.05);
    math_poiss->GenerateValues();
    

    if (max_concentration<0) printf("useless");
    // calculate cumulative hydrogens from amplitude (hp mixture), copies on bead, krate, kmax, diffusion
    // and of course the rate at which nuc is available above the well
    DerivativeComputeCumulativeIncorporationHydrogens(old_vb_out,dA_out,dk_out, my_frame_len, old_delta_frame, old_nuc_rise, sub_steps,my_start_index,
                                               dA, copies, dk,kmax, diffusion, math_poiss);


    delete math_poiss;

    vector<double> my_vb_out;
    vector<double> my_dA_out;
    vector<double> my_dk_out;
    for (int i=0; i<my_frame_len; i++)
    {
      my_vb_out.push_back(old_vb_out[i]);
      my_dA_out.push_back(dA_out[i]);
      my_dk_out.push_back(dk_out[i]);
    }

      RcppResultSet rs;
      rs.add("CumulativeRedHydrogens",      my_vb_out);
      rs.add("dA",my_dA_out);
      rs.add("dk",my_dk_out);
      ret = rs.getReturnList();

    delete[] old_vb_out;
    delete[] dA_out;
    delete[] dk_out;

    delete[] old_nuc_rise;
    delete[] old_delta_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}




RcppExport SEXP ComplexCalculateCumulativeIncorporationHydrogensR(
    SEXP R_nuc_rise, SEXP R_sub_steps, 
    SEXP R_deltaFrame, SEXP R_my_start, 
    SEXP R_C, SEXP R_Amplitude, SEXP R_copies, SEXP R_krate, SEXP R_kmax, SEXP R_diffusion) 
{
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {
    vector<float> nuc_rise = Rcpp::as< vector<float> > (R_nuc_rise);
    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> delta_frame = Rcpp::as< vector<float> > (R_deltaFrame);
    int my_start_index = Rcpp::as<int> (R_my_start);
    float max_concentration = Rcpp::as<float> (R_C);
    float amplitude = Rcpp::as<float> (R_Amplitude);
    float copies = Rcpp::as<float> (R_copies);
    float krate = Rcpp::as<float> (R_krate);
    float kmax = Rcpp::as<float> (R_kmax);
    float diffusion = Rcpp::as<float> (R_diffusion);
    

    int my_frame_len = delta_frame.size();

    float *old_nuc_rise, *old_delta_frame;
    old_nuc_rise = new float [nuc_rise.size()];
    old_delta_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_delta_frame[i] = delta_frame[i]/FRAMESPERSEC; // keep interface transparent for R as this is now in seconds
    }
    // may have more sub-steps than the output
    int nuc_len = nuc_rise.size();
    for (int i=0; i<nuc_len; i++){
      old_nuc_rise[i] = nuc_rise[i];
    }

    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len];
    // calculate cumulative hydrogens from amplitude (hp mixture), copies on bead, krate, kmax, diffusion
    // and of course the rate at which nuc is available above the well
    ComputeCumulativeIncorporationHydrogens(old_vb_out,my_frame_len, old_delta_frame, old_nuc_rise, sub_steps,my_start_index,
                                              max_concentration, amplitude, copies, krate,kmax, diffusion,NULL,false);



    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len; i++)
      my_vb_out.push_back(old_vb_out[i]);

      RcppResultSet rs;
      rs.add("CumulativeRedHydrogens",      my_vb_out);
      ret = rs.getReturnList();

    delete[] old_vb_out;

    delete[] old_nuc_rise;
    delete[] old_delta_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


// now export the nucleotide rise formula
// and I will be nearly happy
// modulo the hidden, annoying formulas

RcppExport SEXP CalculateNucRiseR(
    SEXP R_timeFrame, SEXP R_sub_steps, 
    SEXP R_C, SEXP R_t_mid_nuc, SEXP R_sigma
) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> time_frame = Rcpp::as< vector<float> > (R_timeFrame);

    float max_concentration = Rcpp::as<float> (R_C);
    float t_mid_nuc = Rcpp::as<float> (R_t_mid_nuc);
    float sigma = Rcpp::as<float>(R_sigma);

    int my_frame_len = time_frame.size();

    float  *old_time_frame;

    old_time_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_time_frame[i] = time_frame[i];
    }


    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len*sub_steps];
    // use the same nucleotide rise function as the bkgmodel setup uses
    
    int i_start;
    i_start=SigmaRiseFunction(old_vb_out,my_frame_len,old_time_frame,sub_steps,max_concentration,t_mid_nuc,sigma, true);

    vector<double> my_vb_out;
    vector<double> my_t_out;
    int k=0;
    float tlast = 0; // match code

    for (int i=0; i<my_frame_len; i++)
    { 
      float t = old_time_frame[i];
      for (int st=1; st<=sub_steps; st++)
      {
        float tnew = tlast+(t-tlast)*st/sub_steps;
        my_vb_out.push_back(old_vb_out[k]);
        my_t_out.push_back(tnew);
        k++; 
      }
      tlast = t;
    }

      RcppResultSet rs;
      rs.add("NucConc",      my_vb_out);
      rs.add("Time", my_t_out);
      rs.add("IndexStart",i_start);
      ret = rs.getReturnList();

    delete[] old_vb_out;


    delete[] old_time_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}

// now export the nucleotide rise formula
// and I will be nearly happy
// modulo the hidden, annoying formulas

RcppExport SEXP CalculateNucRiseSplineR(
    SEXP R_timeFrame, SEXP R_sub_steps, 
    SEXP R_C, SEXP R_t_mid_nuc, SEXP R_sigma, SEXP R_zero, SEXP R_one
) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> time_frame = Rcpp::as< vector<float> > (R_timeFrame);

    float max_concentration = Rcpp::as<float> (R_C);
    float t_mid_nuc = Rcpp::as<float> (R_t_mid_nuc);
    float sigma = Rcpp::as<float>(R_sigma);
    float tangent_zero = Rcpp::as<float>(R_zero);
    float tangent_one = Rcpp::as<float>(R_one);

    int my_frame_len = time_frame.size();

    float  *old_time_frame;

    old_time_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_time_frame[i] = time_frame[i];
    }


    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len*sub_steps];
    // use the same nucleotide rise function as the bkgmodel setup uses
    
    int i_start;
    i_start=SplineRiseFunction(old_vb_out,my_frame_len,old_time_frame,sub_steps,max_concentration,t_mid_nuc,sigma,tangent_zero,tangent_one);

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len*sub_steps; i++)
      my_vb_out.push_back(old_vb_out[i]);

      RcppResultSet rs;
      rs.add("NucConc",      my_vb_out);
      rs.add("IndexStart",i_start);
      ret = rs.getReturnList();

    delete[] old_vb_out;


    delete[] old_time_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}

RcppExport SEXP CalculateNucRiseSigmaR(
    SEXP R_timeFrame, SEXP R_sub_steps, 
    SEXP R_C, SEXP R_t_mid_nuc, SEXP R_sigma
) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> time_frame = Rcpp::as< vector<float> > (R_timeFrame);

    float max_concentration = Rcpp::as<float> (R_C);
    float t_mid_nuc = Rcpp::as<float> (R_t_mid_nuc);
    float sigma = Rcpp::as<float>(R_sigma);

    int my_frame_len = time_frame.size();

    float  *old_time_frame;

    old_time_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_time_frame[i] = time_frame[i];
    }


    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len*sub_steps];
    // use the same nucleotide rise function as the bkgmodel setup uses
    
    int i_start;
    i_start=SigmaXRiseFunction(old_vb_out,my_frame_len,old_time_frame,sub_steps,max_concentration,t_mid_nuc,sigma);

    vector<double> my_vb_out;
    for (int i=0; i<my_frame_len*sub_steps; i++)
      my_vb_out.push_back(old_vb_out[i]);

      RcppResultSet rs;
      rs.add("NucConc",      my_vb_out);
      rs.add("IndexStart",i_start);
      ret = rs.getReturnList();

    delete[] old_vb_out;


    delete[] old_time_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


RcppExport SEXP CalculateNucRiseMeasuredR(
    SEXP R_timeFrame, SEXP R_sub_steps, 
    SEXP R_C, SEXP R_t_mid_nuc, SEXP R_sigma
) {
  SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
  char *exceptionMesg = NULL;

  try {

    int sub_steps = Rcpp::as<int> (R_sub_steps);
    vector<float> time_frame = Rcpp::as< vector<float> > (R_timeFrame);

    float max_concentration = Rcpp::as<float> (R_C);
    float t_mid_nuc = Rcpp::as<float> (R_t_mid_nuc);
    float sigma = Rcpp::as<float>(R_sigma);

    int my_frame_len = time_frame.size();

    float  *old_time_frame;

    old_time_frame = new float [my_frame_len];

    for (int i=0; i<my_frame_len; i++){
      old_time_frame[i] = time_frame[i];
    }


    float *old_vb_out;

    // output in frames synchronized
    old_vb_out = new float [my_frame_len*sub_steps];
    // use the same nucleotide rise function as the bkgmodel setup uses
    
    int i_start;
    i_start=MeasuredNucRiseFunction(old_vb_out,my_frame_len,old_time_frame,sub_steps,max_concentration,t_mid_nuc,sigma);

    vector<double> my_vb_out;
    vector<double> my_t_out;
    int k=0;
    float tlast = 0; // match code

    for (int i=0; i<my_frame_len; i++)
    { 
      float t = old_time_frame[i];
      for (int st=1; st<=sub_steps; st++)
      {
        float tnew = tlast+(t-tlast)*st/sub_steps;
        my_vb_out.push_back(old_vb_out[k]);
        my_t_out.push_back(tnew);
        k++; 
      }
      tlast = t;
    }

      RcppResultSet rs;
      rs.add("NucConc",      my_vb_out);
      rs.add("Time", my_t_out);
      rs.add("IndexStart",i_start);
      ret = rs.getReturnList();

    delete[] old_vb_out;


    delete[] old_time_frame;
    

  } catch(...) {
    exceptionMesg = copyMessageToR("unknown reason");
  }
    
  if(exceptionMesg != NULL)
    Rf_error(exceptionMesg);

  return ret;
}


