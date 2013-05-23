/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BiasGenerator.h"



// bias generator handles latent variables representing sources of bias in measurement
// the trivial example is by strand
void BasicBiasGenerator::GenerateBiasByStrand(vector<float> &delta, vector<int> &test_flow, int strand_key, vector<float> &new_residuals){
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     int j_flow = test_flow[t_flow];
     new_residuals[j_flow] -= delta[j_flow]*latent_bias[strand_key];
  }
}

void BasicBiasGenerator::UpdateResiduals(CrossHypotheses &my_cross){
  // move all residuals in direction of bias
   // in theory might have a hypothesis/bias interaction
   for (unsigned int i_hyp=0; i_hyp<my_cross.residuals.size(); i_hyp++){
      GenerateBiasByStrand(my_cross.delta, my_cross.test_flow, my_cross.strand_key, my_cross.residuals[i_hyp]);
   }
}

void BasicBiasGenerator::ResetUpdate(){
  update_latent_bias.assign(update_latent_bias.size(),0.0f);
  weight_update.assign(weight_update.size(), 0.0f);
  update_latent_bias_v.assign(update_latent_bias.size(),0.0f);
  weight_update_v.assign(weight_update.size(), 0.0f);
}

// update by the information from this one item
void BasicBiasGenerator::AddOneUpdate(vector<float> &delta, vector<vector<float> > &residuals, vector<int> &test_flow, int strand_key, vector<float> &responsibility){
  // note bias may vary by more complicated functions
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     int j_flow = test_flow[t_flow];
     for (unsigned int i_hyp=1; i_hyp<responsibility.size(); i_hyp++){  // only non-outliers count!!!
       update_latent_bias[strand_key] += responsibility[i_hyp] * delta[j_flow] * residuals[i_hyp][j_flow]; // estimate projection on delta
       weight_update[strand_key] += responsibility[i_hyp] * delta[j_flow] * delta[j_flow]; // denominator
       
       update_latent_bias_v[i_hyp-1] += responsibility[i_hyp] * delta[j_flow] * residuals[i_hyp][j_flow]; // estimate projection on delta
       weight_update_v[i_hyp-1] += responsibility[i_hyp] * delta[j_flow] * delta[j_flow]; // denominator

     }
  }
}

void BasicBiasGenerator::AddCrossUpdate(CrossHypotheses &my_cross){
   AddOneUpdate(my_cross.delta, my_cross.residuals, my_cross.test_flow, my_cross.strand_key, my_cross.responsibility);
}


// important: residuals need to be reset before this operation
void BasicBiasGenerator::UpdateBiasGenerator(ShortStack &my_theory) {
  ResetUpdate();

  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < my_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = my_theory.valid_indexes[i_ndx];
    AddCrossUpdate(my_theory.my_hypotheses[i_read]);
  }
  DoUpdate();  // new bias estimated
  //cout << "Bias " << bias_generator.latent_bias[0] << "\t" << bias_generator.latent_bias[1] << endl;
}


void BasicBiasGenerator::UpdateResidualsFromBias(ShortStack &total_theory) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    UpdateResiduals(total_theory.my_hypotheses[i_read]);
  }
}


void BasicBiasGenerator::DoStepForBias(ShortStack &total_theory) {
  total_theory.ResetRelevantResiduals();
  UpdateBiasGenerator(total_theory);
  UpdateResidualsFromBias(total_theory);
  total_theory.UpdateRelevantLikelihoods();
}

void BasicBiasGenerator::ResetActiveBias(ShortStack &total_theory) {
  total_theory.ResetRelevantResiduals();
  UpdateResidualsFromBias(total_theory);
  total_theory.UpdateRelevantLikelihoods();
}



void BasicBiasGenerator::DoUpdate(){
   for (unsigned int i_latent=0; i_latent<latent_bias.size(); i_latent++){
       latent_bias[i_latent] = update_latent_bias[i_latent]/(weight_update[i_latent]+damper_bias);
       latent_bias_v[i_latent] = update_latent_bias_v[i_latent]/(weight_update_v[i_latent]+damper_bias);
   }
}

void BasicBiasGenerator::InitForStrand(){

  latent_bias.assign(2,0.0f); // no bias to start with
  latent_bias_v.assign(2,0.0f);
  update_latent_bias.assign(latent_bias.size(), 0.0f);
  update_latent_bias_v.assign(latent_bias_v.size(), 0.0f);
  weight_update.assign(latent_bias.size(), 0.0f);
  weight_update_v.assign(latent_bias.size(), 0.0f);
  
  damper_bias = 30.0f;  // keep things implicitly near zero bias - phrase as precision?
  pseudo_sigma_base = 0.1f; // approximate a penalty
}

float LogOfNormalDensity(float residual, float standard_deviation){
  float log_density = 0.0f;
  log_density += 0.0f-residual*residual/(2.0f*standard_deviation*standard_deviation);
  log_density += 0.0f-0.5*log(2.0f*3.14159f*standard_deviation*standard_deviation);
  return(log_density);
}

// make this relative log-likelihood instead
// will affect nothing
float BasicBiasGenerator::BiasLL(){
  // return estimated likelihood of the bias variables taking on their current forms
  // implicit scaling parameter for "true variance" missing
  float pseudo_sigma = pseudo_sigma_base/sqrt(damper_bias);
  float log_sum= LogOfNormalDensity(latent_bias[0], pseudo_sigma);
  log_sum += LogOfNormalDensity(latent_bias[1], pseudo_sigma);
  // make relative likelihood by subtracting off maximum density
  log_sum -= 2.0f*LogOfNormalDensity(0.0f, pseudo_sigma);
  return(log_sum);
};

float BasicBiasGenerator::BiasHypothesisLL(){
  // return estimated likelihood of the bias variables taking on their current forms
  // implicit scaling parameter for "true variance" missing
  float pseudo_sigma = pseudo_sigma_base/sqrt(damper_bias);
  float log_sum= LogOfNormalDensity(latent_bias_v[0], pseudo_sigma);
  log_sum += LogOfNormalDensity(latent_bias_v[1], pseudo_sigma);
  // make relative likelihood by subtracting off maximum density
  log_sum -= 2.0f*LogOfNormalDensity(0.0f, pseudo_sigma);
  return(log_sum);
};

float BasicBiasGenerator::LikelihoodOfRadius(float radius){
  float pseudo_sigma = pseudo_sigma_base/sqrt(damper_bias);
  float log_sum = LogOfNormalDensity(radius, pseudo_sigma);
  log_sum += LogOfNormalDensity(0.0f, pseudo_sigma);
  log_sum -= 2.0f*LogOfNormalDensity(0.0f, pseudo_sigma);
  return(log_sum);
};

float BasicBiasGenerator::RadiusOfBias(){
  return(sqrt(latent_bias[0]*latent_bias[0]+latent_bias[1]*latent_bias[1]));
};

float BasicBiasGenerator::RadiusOfHypothesisBias(){
  return(sqrt(latent_bias_v[0]*latent_bias_v[0]+latent_bias_v[1]*latent_bias_v[1]));
};
