/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SkewGenerator.h"

// compute skew by method of moments
// skew should be P(x>0|skew) = sk^2/(1+sk^2)


void BasicSkewGenerator::GenerateSkew(CrossHypotheses &my_cross){
  
  my_cross.skew_estimate = latent_skew[my_cross.strand_key];
}

void BasicSkewGenerator::AddOneUpdateForHypothesis(int strand_key, float responsibility, vector<int> &test_flow, vector<float> &residuals){
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     // skew by moments = p(x>0|skew), so compute count of x>0 by responsibility
    if (residuals[t_flow]>0.0f)
       skew_up[strand_key] += responsibility;
     else
       skew_down[strand_key] += responsibility;
  }
}

void BasicSkewGenerator::AddCrossUpdate(CrossHypotheses &my_cross){
   for (unsigned int i_hyp=1; i_hyp<my_cross.residuals.size(); i_hyp++){  // no outlier values count here
      AddOneUpdateForHypothesis(my_cross.strand_key, my_cross.weighted_responsibility[i_hyp], my_cross.test_flow, my_cross.residuals[i_hyp]);
   }
}

void BasicSkewGenerator::ResetUpdate(){
    for (unsigned int i_latent=0; i_latent<latent_skew.size(); i_latent++){
      latent_skew[i_latent] = 1.0f; // fix correctly when I know what I'm doing
      // basic prior: always make sure I have non-zero weighting
      skew_up[i_latent] = 0.5f*dampened_skew;
      skew_down[i_latent] = 0.5f*dampened_skew;
      // not quite right: should dampen around mean
   }
}


void BasicSkewGenerator::DoLatentUpdate(){
  //ResetUpdate();
  for (unsigned int i_latent =0; i_latent<latent_skew.size(); i_latent++){
    latent_skew[i_latent] = sqrt(skew_up[i_latent]/skew_down[i_latent]);
  }
}



void BasicSkewGenerator::UpdateSkewGenerator(ShortStack &total_theory) {
  // put everything to null
  ResetUpdate();

  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    AddCrossUpdate(total_theory.my_hypotheses[i_read]);
  }
  DoLatentUpdate();  // new latent predictors for sigma
}

void BasicSkewGenerator::UpdateSkewEstimates(ShortStack &total_theory) {
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    GenerateSkew(total_theory.my_hypotheses[i_read]);
  }
}

void BasicSkewGenerator::DoStepForSkew(ShortStack &total_theory) {
  UpdateSkewGenerator(total_theory);
  UpdateSkewEstimates(total_theory);
  total_theory.UpdateRelevantLikelihoods();
}

