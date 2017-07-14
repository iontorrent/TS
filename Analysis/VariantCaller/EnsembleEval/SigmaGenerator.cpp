/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SigmaGenerator.h"




void BasicSigmaGenerator::GenerateSigmaByRegression(vector<float> &prediction, vector<int> &test_flow, vector<float> &sigma_estimate){
     // use latent variable to predict sigma by predicted signal
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     sigma_estimate[t_flow] = InterpolateSigma(prediction[t_flow]);  // it's a prediction! always positive
     //cout << "sigma " << prediction.at(j_flow) << "\t" << sigma_estimate.at(j_flow) << endl;
  }
}

void BasicSigmaGenerator::GenerateSigma(CrossHypotheses &my_cross){
   for (unsigned int i_hyp=0; i_hyp<my_cross.residuals.size(); i_hyp++){
      GenerateSigmaByRegression(my_cross.mod_predictions[i_hyp], my_cross.test_flow, my_cross.sigma_estimate[i_hyp]);
   }
}

void BasicSigmaGenerator::ResetUpdate(){

   SimplePrior(); // make sure we have some stability
}

void BasicSigmaGenerator::ZeroAccumulator(){
    for (unsigned int i_level = 0; i_level<accumulated_sigma.size(); i_level++){
      accumulated_sigma[i_level] = 0.0f;
      accumulated_weight[i_level] = 0.0f;
    }
}

void BasicSigmaGenerator::SimplePrior(){
   // sum r^2
   ZeroAccumulator();
   float basic_weight = prior_weight; // prior strength
   for (unsigned int i_level = 0; i_level<accumulated_sigma.size(); i_level++)
   {
      // expected variance per level
      // this is fairly arbitrary as we expect the data to overcome our weak prior here
      float square_level = i_level*i_level+1.0f;  // avoid zero division
      // approximate quadratic increase in sigma- should be linear, but empirically we see more than expected
      float sigma_square =  prior_sigma_regression[0]+prior_sigma_regression[1]*square_level;
      sigma_square *=sigma_square; // push squared value
      PushLatent(basic_weight, (float) i_level, sigma_square, true);
   }
   DoLatentUpdate();  // the trivial model for sigma by intensity done
}

// retrieve latent
float BasicSigmaGenerator::InterpolateSigma(float x_val){
    float t_val = max(x_val, 0.0f);
    int low_level = (int) t_val; // floor cast because x_val positive
    int hi_level = low_level+1;
    if (low_level>max_level){
       low_level = hi_level = max_level;
       t_val = (float) low_level;
    }    
    if (hi_level>max_level) hi_level = max_level;
    float delta_low = t_val - low_level;
    float delta_hi = 1.0f-delta_low;
    // very sensitive to log-likelihood at margins
    // weight by available data as well
    // with a little safety factor in case something unusual has happened
    delta_low *= (accumulated_weight[hi_level]+0.001f);
    delta_hi  *= (accumulated_weight[low_level]+0.001f);
    float total_weight = delta_low+delta_hi;
    delta_low /= total_weight;
    delta_hi /= total_weight;
    
    return(latent_sigma[low_level]*delta_hi + latent_sigma[hi_level]*delta_low);
}

void BasicSigmaGenerator::PushLatent(float responsibility, float x_val, float y_val, bool do_weight){
    // interpolation and add
    float t_val = max(x_val, 0.0f);
    int low_level = (int) t_val; // floor cast because x_val non-negative
    int hi_level = low_level+1;
    if (low_level>max_level){
       low_level = hi_level = max_level;
       t_val = (float) low_level;
    }
    if (hi_level>max_level) hi_level = max_level;
    float delta_low = t_val - low_level;
    float delta_hi = 1.0f-delta_low;
    accumulated_sigma[low_level] += responsibility*delta_hi*y_val;
    if (do_weight)
      accumulated_weight[low_level] += responsibility*delta_hi;
    accumulated_sigma[hi_level] += responsibility*delta_low*y_val;
    if (do_weight)
      accumulated_weight[hi_level] += responsibility*delta_low;
}

// local weight effective at this location
float BasicSigmaGenerator::RetrieveApproximateWeight(float x_val){
  float t_val = max(x_val, 0.0f);
  int low_level = (int) t_val; // floor cast because x_val non-negative
  int hi_level = low_level+1;
  if (low_level>max_level){
     low_level = hi_level = max_level;
     t_val = (float) low_level;
  }
  if (hi_level>max_level) hi_level = max_level;
  float delta_low = t_val - low_level;
  float delta_hi = 1.0f-delta_low;
  float x_weight = accumulated_weight[low_level]*delta_hi + accumulated_weight[hi_level]*delta_low;
  return(x_weight);
}

// Notice that now prediction and residuals contain only test flows!
void BasicSigmaGenerator::AddOneUpdateForHypothesis(vector<float> &prediction, float responsibility, float skew_estimate, vector<int> &test_flow, vector<float> &residuals, vector<float> &measurements_var){
  bool is_non_empty_measurements_var = not measurements_var.empty();
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     float y_val =residuals[t_flow]*residuals[t_flow];
     if (is_non_empty_measurements_var){
    	 y_val += measurements_var[t_flow];  // add the variance of the measurements if it is a consensus read.
     }
     // handle skew
     // note that this is >opposite< t-dist formula
     if (residuals[t_flow]>0)
       y_val = y_val/(skew_estimate*skew_estimate);
     else
       y_val = y_val * skew_estimate*skew_estimate;
     
     float x_val = prediction[t_flow];
     PushLatent(responsibility,x_val,y_val, true);
  }
}

// additional variation from shifting clusters
void BasicSigmaGenerator::AddShiftUpdateForHypothesis(vector<float> &prediction, vector<float> &mod_prediction, 
                                                      float discount, float responsibility, float skew_estimate, vector<int> &test_flow){
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++){
     float y_val =prediction[t_flow]-mod_prediction[t_flow]; // how much did I shift my prediction?
     y_val = y_val * y_val;

     float x_val = mod_prediction[t_flow];
     float local_weight = RetrieveApproximateWeight(x_val);

     // k_zero * n/(k_zero+n) * (y_mean-u_mean)*(y_mean-u_mean)
     // as we're looking at y_mean(read)-u_mean (read) estimated by difference in predictions, n comes from all the reads
     // and therefore we are left with k_zero/(k_zero+n)

     y_val *= discount/(discount+local_weight); // proportional allocation of weight for the shift being added
     
     PushLatent(responsibility,x_val,y_val, false);
  }
}

void BasicSigmaGenerator::AddShiftCrossUpdate(CrossHypotheses &my_cross, float discount){
   for (unsigned int i_hyp=1; i_hyp<my_cross.residuals.size(); i_hyp++){  // no outlier values count here
      AddShiftUpdateForHypothesis(my_cross.predictions[i_hyp], my_cross.mod_predictions[i_hyp], discount, my_cross.weighted_responsibility[i_hyp], my_cross.skew_estimate, my_cross.test_flow);
   }
}


void BasicSigmaGenerator::AddCrossUpdate(CrossHypotheses &my_cross){
   for (unsigned int i_hyp=0; i_hyp<my_cross.residuals.size(); i_hyp++){  // no outlier values count here
      AddOneUpdateForHypothesis(my_cross.mod_predictions[i_hyp], my_cross.weighted_responsibility[i_hyp], my_cross.skew_estimate, my_cross.test_flow, my_cross.residuals[i_hyp], my_cross.measurement_var);
   }
}

void BasicSigmaGenerator::AddNullUpdate(CrossHypotheses &my_cross){
  unsigned int i_hyp =0;
  /*
  vector<int> all_flows;
  all_flows.assign(my_cross.mod_predictions[0].size(), 0.0f);
  for (unsigned int i_flow=0; i_flow<all_flows.size(); i_flow++)
    all_flows[i_flow] = i_flow;
  */
  AddOneUpdateForHypothesis(my_cross.mod_predictions[i_hyp], 1.0f, 1.0f, my_cross.test_flow, my_cross.residuals[i_hyp], my_cross.measurement_var);

}

void BasicSigmaGenerator::DoLatentUpdate(){

   for (unsigned int i_level=0; i_level<latent_sigma.size(); i_level++){
      latent_sigma[i_level] = sqrt(accumulated_sigma[i_level]/accumulated_weight[i_level]);
      //cout << latent_sigma.at(i_level) << "\t" << i_level << endl;
   }
}

void BasicSigmaGenerator::PushToPrior(){
  prior_latent_sigma = latent_sigma;
};

void BasicSigmaGenerator::PopFromLatentPrior(){
  ZeroAccumulator();
  accumulated_sigma = prior_latent_sigma;
  accumulated_weight.assign(accumulated_sigma.size(), 1.0f);
};


// generate regression using all positions
// good for initial estimate for low intensity reads
void BasicSigmaGenerator::NullUpdateSigmaGenerator(ShortStack &total_theory) {
// put everything to null
  ResetUpdate();

  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    AddNullUpdate(total_theory.my_hypotheses[i_read]);
  }
  DoLatentUpdate();  // new latent predictors for sigma
  PushToPrior();
}

// important: residuals do not need to be reset before this operation (predictions have been corrected for bias already)
void BasicSigmaGenerator::UpdateSigmaGenerator(ShortStack &total_theory) {
// put everything to null
  ResetUpdate();

//  float k_zero = 1.0f;

  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    AddCrossUpdate(total_theory.my_hypotheses[i_read]);
   }
  // now that I've established basic weight, I can update
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    // additional variability from cluster shifting
      // bayesian multidimensional normal
      AddShiftCrossUpdate(total_theory.my_hypotheses[i_read], k_zero);
  }

  DoLatentUpdate();  // new latent predictors for sigma
}

void BasicSigmaGenerator::UpdateSigmaEstimates(ShortStack &total_theory) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    GenerateSigma(total_theory.my_hypotheses[i_read]);
  }
}

void BasicSigmaGenerator::DoStepForSigma(ShortStack &total_theory) {

  UpdateSigmaGenerator(total_theory);
  UpdateSigmaEstimates(total_theory);
  total_theory.UpdateRelevantLikelihoods();
}

// run one sigma generator for each strand
void StrandedSigmaGenerator::DoStepForSigma(ShortStack &total_theory){
  UpdateSigmaGenerator(total_theory);
  UpdateSigmaEstimates(total_theory);
  total_theory.UpdateRelevantLikelihoods();
  PrintDebug();
}

void StrandedSigmaGenerator::PrintDebug(bool print_updated){
  if(DEBUG > 1){
	cout << "    + Latent sigma" << (print_updated? " updated ": ":") << endl
		 << "      - FWD: (amplitude, sigma) = ";
	int max_level_debug = min(10 , fwd.max_level);
	for (int i = 0; i < max_level_debug + 1; ++i)
		cout << "("<< i << ", " << fwd.latent_sigma[i] <<"), ";
	cout<< endl << "      - REV: (amplitude, sigma) = ";
	max_level_debug = min(10 , rev.max_level);
	for (int i = 0; i < max_level_debug + 1; ++i)
		cout << "("<< i << ", " << rev.latent_sigma[i] <<"), ";
  cout << endl;
  }
}

void StrandedSigmaGenerator::UpdateSigmaGenerator(ShortStack &total_theory){
  // put everything to null
    fwd.ResetUpdate();
    rev.ResetUpdate();


    for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
      unsigned int i_read = total_theory.valid_indexes[i_ndx];
      if (total_theory.my_hypotheses[i_read].strand_key==0 || combine_strands)
        fwd.AddCrossUpdate(total_theory.my_hypotheses[i_read]);
      else
        rev.AddCrossUpdate(total_theory.my_hypotheses[i_read]);
     }
    // now that I've established basic weight, I can update
    for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
      unsigned int i_read = total_theory.valid_indexes[i_ndx];
      // additional variability from cluster shifting
        // bayesian multidimensional normal
      if (total_theory.my_hypotheses[i_read].strand_key==0 || combine_strands)
        fwd.AddShiftCrossUpdate(total_theory.my_hypotheses[i_read], fwd.k_zero);
      else
        rev.AddShiftCrossUpdate(total_theory.my_hypotheses[i_read], rev.k_zero);
    }

    fwd.DoLatentUpdate();  // new latent predictors for sigma
    rev.DoLatentUpdate();

}

void StrandedSigmaGenerator::UpdateSigmaEstimates(ShortStack &total_theory){
  for (unsigned int i_ndx = 0; i_ndx < total_theory.valid_indexes.size(); i_ndx++) {
    unsigned int i_read = total_theory.valid_indexes[i_ndx];
    if (total_theory.my_hypotheses[i_read].strand_key==0 || combine_strands)
      fwd.GenerateSigma(total_theory.my_hypotheses[i_read]);
    else
      rev.GenerateSigma(total_theory.my_hypotheses[i_read]);
  }
  //fwd.UpdateSigmaEstimates(total_theory);
}

// return me to scratch for restarts
void StrandedSigmaGenerator::ResetSigmaGenerator(){
  fwd.ResetUpdate();
  rev.ResetUpdate();
}
