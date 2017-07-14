/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BiasGenerator.h"

// bias generator handles latent variables representing sources of bias in measurement
// the trivial example is by strand
void BasicBiasGenerator::GenerateBiasByStrand(int i_hyp, HiddenBasis &delta_state,  vector<int> &test_flow, int strand_key, vector<float> &new_residuals, vector<float> &new_predictions){

  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     //float b_val = PredictBias(delta_state, strand_key, i_hyp, j_flow);
     float b_val = delta_state.ServeCommonDirection(t_flow);
     new_residuals[t_flow] -= b_val;
     new_predictions[t_flow] -= b_val;
  }
}

void BasicBiasGenerator::UpdateResiduals(CrossHypotheses &my_cross){
  // move all residuals in direction of bias
  my_cross.delta_state.SetDeltaReturn(latent_bias[my_cross.strand_key]);
    // in theory might have a hypothesis/bias interaction
   for (unsigned int i_hyp=0; i_hyp<my_cross.residuals.size(); i_hyp++){
      GenerateBiasByStrand(i_hyp, my_cross.delta_state, my_cross.test_flow, my_cross.strand_key, my_cross.residuals[i_hyp], my_cross.mod_predictions[i_hyp]);
   }
}

void BasicBiasGenerator::ResetUpdate(){
  for (unsigned int i_strand =0; i_strand<2; i_strand++){
    update_latent_bias[i_strand].assign(update_latent_bias[i_strand].size(),0.0f);
    weight_update[i_strand].assign(weight_update[i_strand].size(), 0.0f);
  }
}

// update by the information from this one item
void BasicBiasGenerator::AddOneUpdate(HiddenBasis &delta_state, const vector<vector<float> > &residuals,
                                      const vector<int> &test_flow, const int strand_key, const vector<float> &responsibility){
  // note bias may vary by more complicated functions
  //cout << "SIZE: " <<  responsibility.size() << "\t" << update_latent_bias.at(0).size() << "\t" << weight_update.at(0).size() << endl;
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     for (unsigned int i_hyp=1; i_hyp<responsibility.size(); i_hyp++){  // only non-outliers count!!!
       float r_val = residuals[i_hyp][t_flow];
       // normally this will be just a single o_alt value
       for (unsigned int o_alt = 0; o_alt<update_latent_bias[strand_key].size(); o_alt++){
          float d_val = delta_state.ServeAltDelta(o_alt, t_flow);
          update_latent_bias[strand_key][o_alt] += responsibility[i_hyp] * d_val * r_val ; // estimate projection on delta
          weight_update[strand_key][o_alt] += responsibility[i_hyp] * d_val * d_val; // denominator
       }
     }
  }
}


void BasicBiasGenerator::AddCrossUpdate(CrossHypotheses &my_cross){
   AddOneUpdate(my_cross.delta_state, my_cross.residuals, my_cross.test_flow, my_cross.strand_key, my_cross.weighted_responsibility);
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
  //cout << "Bias " << bias_generator.latent_bias.at(0) << "\t" << bias_generator.latent_bias.at(1) << endl;
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
  PrintDebug();
}

void BasicBiasGenerator::PrintDebug(bool print_updated){
  if(DEBUG > 1){
	  cout << "    + Latent bias"<< (print_updated? " updated:" : ":") << endl
		   << "      - FWD: latent_bias = " << PrintIteratorToString(latent_bias[0].begin(), latent_bias[0].end()) <<endl
	       << "      - REV: latent_bias = " << PrintIteratorToString(latent_bias[1].begin(), latent_bias[1].end()) <<endl;
  }
}

void BasicBiasGenerator::ResetActiveBias(ShortStack &total_theory) {
  total_theory.ResetRelevantResiduals();
  UpdateResidualsFromBias(total_theory);
  total_theory.UpdateRelevantLikelihoods();
}

void BasicBiasGenerator::DoUpdate(){
  for (unsigned int i_strand=0; i_strand<2; i_strand++){
   for (unsigned int i_latent=0; i_latent<latent_bias[i_strand].size(); i_latent++){
       latent_bias[i_strand][i_latent] = update_latent_bias[i_strand][i_latent]/(weight_update[i_strand][i_latent]+damper_bias);
   }
  }
}

void BasicBiasGenerator::InitForStrand(int num_alt){
  latent_bias.resize(2); // 2 strands
  update_latent_bias.resize(2); // 2 strands
  weight_update.resize(2); // 2 strands
  //int num_alt = 1;
  for (unsigned int i_strand=0; i_strand<2; i_strand++){
    latent_bias[i_strand].assign(num_alt,0.0f);
    update_latent_bias[i_strand].assign(num_alt,0.0f);
    weight_update[i_strand].assign(num_alt,0.0f);
  }
  
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
  int fwd_strand = 0;
  int rev_strand = 1;
  float log_sum = 0.0f;
  // LL taken over all basis vectors
  for  (unsigned int o_alt=0; o_alt<latent_bias[fwd_strand].size(); o_alt++){
      log_sum= LogOfNormalDensity(latent_bias[fwd_strand][o_alt], pseudo_sigma);
       log_sum += LogOfNormalDensity(latent_bias[rev_strand][o_alt], pseudo_sigma);
       // make relative likelihood by subtracting off maximum density
      log_sum -= 2.0f*LogOfNormalDensity(0.0f, pseudo_sigma);
  }
   return log_sum;
};

float BasicBiasGenerator::RadiusOfBias(int o_alt){
  float retval = 0.0f;
  for (unsigned int i_latent=0; i_latent<latent_bias.size(); i_latent++){
    retval += latent_bias[i_latent][o_alt]*latent_bias[i_latent][o_alt];
  }
  return sqrt(retval);
}

// Note: this object does not have the same purpose as bias generator
// this is to check clustering >with modified residuals< not >raw residuals<
// so >after< bias generation has been done and completed
// it is never used within the loop
// also, it iterates over hypotheses, not over strands+hypotheses
// although possibly it should

void BiasChecker::ResetUpdate(){
  update_variant_bias_v.assign(update_variant_bias_v.size(),0.0f);
  weight_variant_v.assign(weight_variant_v.size(), 0.0f);
  update_ref_bias_v.assign(update_ref_bias_v.size(),0.0f);
  weight_ref_v.assign(weight_ref_v.size(), 0.0f);
}

void BiasChecker::Init(int num_hyp_no_null){
  variant_bias_v.assign(num_hyp_no_null,0.0f);
  ref_bias_v.assign(num_hyp_no_null,0.0f);
  update_variant_bias_v.assign(num_hyp_no_null, 0.0f);
  update_ref_bias_v.assign(num_hyp_no_null, 0.0f);
  weight_variant_v.assign(num_hyp_no_null, 0.0f);
  weight_ref_v.assign(num_hyp_no_null, 0.0f);
  damper_bias = 30.0f;  // keep things implicitly near zero bias - phrase as precision?
  soft_clip = 0.1f; // shut down data points that are extremely marginal
}

void BiasChecker::DoUpdate(){
  for (unsigned int i_latent=0; i_latent<variant_bias_v.size(); i_latent++){
     variant_bias_v[i_latent] = update_variant_bias_v[i_latent]/(weight_variant_v[i_latent]+damper_bias);
     ref_bias_v[i_latent] = update_ref_bias_v[i_latent]/(weight_ref_v[i_latent]+damper_bias);
  }
}


void BiasChecker::UpdateBiasChecker(ShortStack &my_theory){
  ResetUpdate();
  // ZZ output
  /*
  for (unsigned int i_read=0; i_read<my_theory.my_hypotheses.size(); i_read++){
HiddenBasis &delta_state = my_theory.my_hypotheses[i_read].delta_state;
  for (unsigned i =0; i < delta_state.delta.size(); i++) {
        cout << "iread hyp " <<  i_read << i << "\t";
        for (unsigned int j = 0; j < delta_state.delta[i].size(); j++)
            cout << delta_state.delta[i][j] << "\t";
        cout << endl;
  }
}
*/
  for (unsigned int i_ndx = 0; i_ndx < my_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = my_theory.valid_indexes[i_ndx];
    AddCrossUpdate(my_theory.my_hypotheses[i_read]);
  }
  DoUpdate();  // new bias estimated
}


void BiasChecker::AddCrossUpdate(CrossHypotheses &my_cross){
  AddOneUpdate(my_cross.delta_state, my_cross.residuals, my_cross.test_flow,  my_cross.weighted_responsibility);
}

// note that this will have to be updated for the new multi-allele world
// to make sure we're checking the correct direction
// in this case, we're checking reference vs single alternate
// which is the direction
void BiasChecker::AddOneUpdate(HiddenBasis &delta_state, const vector<vector<float> > &residuals, const vector<int> &test_flow, const vector<float> &responsibility){
  // note bias may vary by more complicated functions
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     // no null hypothesis
     // shut down crazy data points aggregating by a soft-clip value

     // ref hypothesis special - project on each alternate vector
     for (unsigned int i_hyp=2; i_hyp<responsibility.size(); i_hyp++){
       int o_alt = i_hyp-2;
      float d_val = delta_state.ServeAltDelta(o_alt,t_flow);
      float r_val = responsibility[1];
      if (r_val<soft_clip)
        r_val = 0.0f;
      update_ref_bias_v[i_hyp-1] += r_val * d_val * residuals[1][t_flow]; // estimate projection on delta
      weight_ref_v[i_hyp-1] +=r_val * d_val * d_val; // denominator
     }
// variant hypotheses
     for (unsigned int i_hyp=2; i_hyp<responsibility.size(); i_hyp++){  // only non-outliers count!!!
       // for each alternate hypothesis (or reference), we are checking the shift along the axis joining  reference to variant
       int o_alt = i_hyp-2;
       float d_val = delta_state.ServeAltDelta(o_alt, t_flow);
       float r_val = responsibility[i_hyp];
       if (r_val<soft_clip)
         r_val = 0.0f;
       update_variant_bias_v[i_hyp-1] += r_val * d_val * residuals[i_hyp][t_flow]; // estimate projection on delta
       weight_variant_v[i_hyp-1] += r_val * d_val * d_val; // denominator

     }
  }
}
