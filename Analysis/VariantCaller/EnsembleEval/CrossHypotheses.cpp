/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CrossHypotheses.h"


// model as a t-distribution to slightly resist outliers
/*float TdistThree(float res, float sigma){
  float x=res/sigma;
  float xx = x*x;
  return( 6.0f*sqrt(3.0f)/(sigma*3.14159f*(3.0f+xx)*(3.0f+xx)) );
}*/

//  control degrees of freedom for tradeoff in outlier resistance/sensitivity
float xTDistOddN(float res, float sigma, float skew, int half_n) {
  // skew t-dist one direction or the other
  float l_sigma;
  if (res>0.0f) {
    l_sigma = sigma*skew;
  } else {
    l_sigma = sigma/skew;
  }

  float x = res/l_sigma;
  float xx = x*x;
  float v = 2*half_n-1; // 1,3,5,7,...
  float my_likelihood = 1.0f/(3.14159f*sqrt(v));
  float my_factor = 1.0f/(1.0f+xx/v);

  for (int i_prod=0; i_prod<half_n; i_prod++) {
    my_likelihood *= my_factor;
  }
  for (int i_prod=1; i_prod<half_n; i_prod++) {
    my_likelihood *= (v+1.0f-2.0f*i_prod)/(v-2.0f*i_prod);
  }
  my_likelihood /= l_sigma;
  // account for skew
  float skew_factor = 2.0f*skew/(skew*skew+1.0f);
  my_likelihood *= skew_factor;
  return(my_likelihood);
}

void PrecomputeTDistOddN::SetV(int _half_n){
  half_n = _half_n;
  v = 2*half_n-1;
  pi_factor = 1.0f/(3.14159f*sqrt(v));
  v_factor = 1.0f;
  for (int i_prod=1; i_prod<half_n; i_prod++) {
    v_factor *= (v+1.0f-2.0f*i_prod)/(v-2.0f*i_prod);
  }
};

float PrecomputeTDistOddN::TDistOddN(float res, float sigma, float skew){
  // skew t-dist one direction or the other
  float l_sigma;
  if (res>0.0f) {
    l_sigma = sigma*skew;
  } else {
    l_sigma = sigma/skew;
  }

  float x = res/l_sigma;
  float xx = x*x;

  float my_likelihood = pi_factor;
  float my_factor = v/(v+xx);

  for (int i_prod=0; i_prod<half_n; i_prod++) {
    my_likelihood *= my_factor;
  }
  my_likelihood *= v_factor;
  //  for (int i_prod=1; i_prod<half_n; i_prod++) {
  //    my_likelihood *= (v+1.0f-2.0f*i_prod)/(v-2.0f*i_prod);
  //  }
  my_likelihood /= l_sigma;
  // account for skew
  float skew_factor = 2.0f*skew/(skew*skew+1.0f);
  my_likelihood *= skew_factor;
  return(my_likelihood);
}

HiddenBasis::HiddenBasis(){
  delta_correlation = 0.0f ;
}

float HiddenBasis::ServeDelta(int i_hyp, int j_flow){
  return delta[0][j_flow];
}

float HiddenBasis::ServeAltDelta(int i_alt, int j_flow){
  return delta[i_alt][j_flow];
}


void HiddenBasis::Allocate(int num_hyp, int num_flow){
  // ref-alt are my basis vectors
  // guaranteed to be all different
  int num_alt = num_hyp-2;
  delta.resize(num_alt); // num_alt
  for (unsigned int i_alt=0; i_alt<delta.size(); i_alt++)
   delta[i_alt].assign(num_flow, 0.0f);
}



void CrossHypotheses::CleanAllocate(int num_hyp, int num_flow) {
  // allocate my vectors here
  responsibility.assign(num_hyp, 0.0f);
  log_likelihood.assign(num_hyp, 0.0f);
  scaled_likelihood.assign(num_hyp, 0.0f);

  tmp_prob_f.assign(num_hyp, 0.0f);
  tmp_prob_d.assign(num_hyp, 0.0);

  predictions.resize(num_hyp);
  mod_predictions.resize(num_hyp);
  normalized.resize(num_hyp);
  residuals.resize(num_hyp);
  sigma_estimate.resize(num_hyp);
  basic_likelihoods.resize(num_hyp);

  for (int i_hyp=0; i_hyp<num_hyp; i_hyp++) {
    predictions[i_hyp].assign(num_flow, 0.0f);
    mod_predictions[i_hyp].assign(num_flow, 0.0f);
    normalized[i_hyp].assign(num_flow, 0.0f);
    residuals[i_hyp].assign(num_flow, 0.0f);
    sigma_estimate[i_hyp].assign(num_flow, 0.0f);
    basic_likelihoods[i_hyp].assign(num_flow, 0.0f);
  }
  delta_state.Allocate(num_hyp, num_flow);
}

void CrossHypotheses::SetModPredictions() {
  // modified predictions reset from predictions
  for (unsigned int i_hyp=0; i_hyp<predictions.size(); i_hyp++) {
    mod_predictions[i_hyp] = predictions[i_hyp];
  }
}


void CrossHypotheses::FillInPrediction(PersistingThreadObjects &thread_objects, const Alignment& my_read, const InputStructures &global_context) {


  // allocate everything here
  CleanAllocate(instance_of_read_by_state.size(), global_context.flow_order_vector.at(my_read.flow_order_index).num_flows());
  // We search for test flows in the flow interval [(splice_start_flow-3*max_flows_to_test), (splice_end_flow+4*max_flows_to_test)]
  // We need to simulate further than the end of the search interval to get good predicted values within
  int flow_upper_bound = splice_end_flow + 4*max_flows_to_test + 20;
  max_last_flow = CalculateHypPredictions(thread_objects, my_read, global_context, instance_of_read_by_state,
                                          same_as_null_hypothesis, predictions, normalized, flow_upper_bound);
  SetModPredictions();
  if (my_read.is_reverse_strand)
    strand_key = 1;
  else
    strand_key = 0;
}

void CrossHypotheses::InitializeTestFlows() {

  delta_state.ComputeDelta(predictions); // depends on predicted
  // Compute test flows for all hypotheses: flows changing by more than 0.1, 10 flows allowed
  success = ComputeAllComparisonsTestFlow(min_delta_for_flow,max_flows_to_test);
  // compute cross-data across the deltas for multialleles
  delta_state.ComputeCross(test_flow);
  // now compute possible  correlation amongst test flow data
  delta_state.ComputeDeltaCorrelation(predictions, test_flow);
}

void CrossHypotheses::InitializeDerivedQualities() {

  InitializeResponsibility(); // depends on hypotheses
  // in theory don't need to compute any but test flows
  SetModPredictions();  // make sure that mod-predictions=predictions
  ComputeBasicResiduals(); // predicted and measured

  InitializeSigma(); // depends on predicted

  my_t.SetV(heavy_tailed);

  ComputeBasicLikelihoods(); // depends on residuals and sigma
  // compute log-likelihoods
  ComputeLogLikelihoods();  // depends on test flow(s)
}

void CrossHypotheses::InitializeResponsibility() {
  responsibility[0] = 1.0f;  // everyone is an outlier until we trust you
  for (unsigned int i_hyp=1; i_hyp<responsibility.size(); i_hyp++)
    responsibility[i_hyp] = 0.0f;
}



// responsibility depends on the relative global probability of the hypotheses and the likelihoods of the observations under each hypothesis
// divide the global probabilities into "typical" data points and outliers
// divide the variant probabilities into each hypothesis (summing to 1)
// treat the 2 hypothesis case to start with
void CrossHypotheses::UpdateResponsibility(const vector<float > &hyp_prob, float typical_prob) {

  if (!success){
    //cout << "alert: fail to splice still called" << endl;
    InitializeResponsibility();
  } else {
  //  vector<double> tmp_prob(3);
  tmp_prob_d[0] = (1.0f-typical_prob)*scaled_likelihood[0];   // i'm an outlier
  for (unsigned int i_hyp=1; i_hyp<scaled_likelihood.size(); i_hyp++)
    tmp_prob_d[i_hyp] = typical_prob * hyp_prob[i_hyp-1] * scaled_likelihood[i_hyp];

  double ll_denom = 0.0;
  for (unsigned int i_hyp=0; i_hyp<scaled_likelihood.size(); i_hyp++){
    ll_denom += tmp_prob_d[i_hyp];
  }

  for (unsigned int i_hyp=0; i_hyp<responsibility.size(); i_hyp++)
    responsibility[i_hyp] = tmp_prob_d[i_hyp]/ll_denom;
  }
}

float CrossHypotheses::ComputePosteriorLikelihood(const vector<float > &hyp_prob, float typical_prob) {
  //  vector<float> tmp_prob(3);
  tmp_prob_f[0] = (1.0f-typical_prob)*scaled_likelihood[0];   // i'm an outlier
  for (unsigned int i_hyp=1; i_hyp<scaled_likelihood.size(); i_hyp++){
    tmp_prob_f[i_hyp] = typical_prob * hyp_prob[i_hyp-1] * scaled_likelihood[i_hyp];
  }
  float ll_denom = 0.0f;
  for (unsigned int i_hyp=0; i_hyp<scaled_likelihood.size(); i_hyp++) {
    ll_denom += tmp_prob_f[i_hyp];
  }
  return(log(ll_denom)+ll_scale);  // log-likelihood under current distribution, including common value of log-likelihood-scale
}


void HiddenBasis::ComputeDelta(const vector<vector <float> > &predictions){
  unsigned int num_alt = predictions.size()-2;
  for (unsigned int i_alt=0; i_alt<num_alt; i_alt++){
  for (unsigned int j_flow=0; j_flow<delta[i_alt].size(); j_flow++) {
    int alt_hyp = i_alt+2;
    int ref_hyp = 1;
    delta[i_alt][j_flow] = predictions[alt_hyp][j_flow]-predictions[ref_hyp][j_flow];
  }
  }
}

void HiddenBasis::ComputeCross(const vector<int> &test_flow){
  // d_i = approximate basis vector
  // compute d_j*d_i/(d_i*d_i)
  unsigned int num_alt = delta.size();
  cross_cor.set_size (num_alt,num_alt);
  for (unsigned int i_alt =0 ; i_alt<num_alt; i_alt++){

    for (unsigned int j_alt=0; j_alt<num_alt; j_alt++){
      float  my_top = 0.0f;
      float my_bottom = 0.0001f;  // delta might be all zeros if I am unlucky

      for (unsigned int i_ndx=0; i_ndx<test_flow.size(); i_ndx++){
        int j_flow = test_flow[i_ndx];
        my_top += delta[i_alt][j_flow]*delta[j_alt][j_flow];
        my_bottom += delta[i_alt][j_flow]*delta[i_alt][j_flow];
      }

      cross_cor.at(i_alt,j_alt) = my_top/my_bottom;  // row,column
    }
  }

  //I'll call this repeatedly, so it is worth setting up the inverse at this time
  arma::mat I(num_alt,num_alt);
  I.eye();
  float lambda = 0.00001; // tikhonov in case we have colinearity, very possible for deletions/insertions

  arma::mat A(num_alt,num_alt);
      A = cross_cor.t()*cross_cor + lambda*I;
      cross_inv.set_size(num_alt,num_alt);
      cross_inv = A.i() * cross_cor.t();

      // placeholders for the values that "mix" the basis vector deltas
      // tmp_beta is the projections  onto each vectcor - which are not orthogonal
      // tmp_synthesis is the final direction after transforming the beta values
      // this is so the beta values remain naturally scaled to "size of variant" and can hence be compared across reads/objects
      tmp_beta.set_size(num_alt);
      tmp_synthesis.set_size(num_alt);
/*
      //test this crazy thing
      for (unsigned int j_alt = 0; j_alt<tmp_beta.size(); j_alt++){// 0th basis vector
      for (unsigned int i_alt=0; i_alt<tmp_beta.size(); i_alt++){
        float ij_proj = 0.0f;
        float ij_bot = 0.0f;
        float ij_top = 0.0f;
        for (unsigned int i_ndx=0; i_ndx<test_flow.size(); i_ndx++){
          int j_flow = test_flow[i_ndx];
          ij_top += delta.at(i_alt).at(j_flow)*delta.at(j_alt).at(j_flow);
          ij_bot += delta.at(i_alt).at(j_flow)*delta.at(i_alt).at(j_flow);
        }
        ij_proj = ij_top/ij_bot;
        tmp_beta.at(i_alt)=ij_proj;
      }

      tmp_synthesis = cross_inv * tmp_beta; // solve for projection
      // should recover 1,0,0...
      cout <<"Beta:"<< j_alt << endl;
      cout << tmp_beta << endl;
      cout << "Recovery:" << j_alt << endl;
      cout << tmp_synthesis << endl;
      }
      // should recover the vector we started with
*/
      //cout << "Cross" << endl;
     // cout << cross_cor   << endl;

      //cout << cross_inv << endl;



}

void HiddenBasis::SetDeltaReturn(const vector<float> &beta){
  for  (unsigned int i_alt=0; i_alt<beta.size(); i_alt++)
    tmp_beta[i_alt] = beta[i_alt];
  tmp_synthesis = cross_inv * tmp_beta;
  // tmp_synthesis now set for returning synthetic deviations
 // cout << "Synthetic: "<< tmp_synthesis << endl; // synthetic direction
}

float HiddenBasis::ServeCommonDirection(int j_flow){
  // tmp_synthesis must be set
  // otherwise we're getting nonsense
  float retval=0.0f;
  for (unsigned int i_alt=0; i_alt<delta.size(); i_alt++){
    retval += tmp_synthesis[i_alt]*delta[i_alt][j_flow];
  }
  // check my theory of the world
  //retval = 0.0f;
  return(retval);
}


void HiddenBasis::ComputeDeltaCorrelation(const vector<vector <float> > &predictions, const vector<int> &test_flow) {
  // just do this for the first alternate for now
  int ref_hyp = 1;
  int alt_hyp = 2;
  int i_alt = 0;

  vector<float> average_prediction;
  average_prediction = predictions[ref_hyp];
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
    int j_flow = test_flow[t_flow];
    average_prediction[j_flow] += predictions[alt_hyp][j_flow];
    average_prediction[j_flow] /=2.0f; // just ref/var to correspond to "delta"
  }
  // compute correlation = cos(theta)
  float xy,xx,yy;
  xy = xx = yy = 0.0f;
  for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
    int j_flow = test_flow[t_flow];
    float j_delta = delta[i_alt][j_flow];
    xy += average_prediction[j_flow]*j_delta;
    xx += average_prediction[j_flow]*average_prediction[j_flow];
    yy += j_delta*j_delta;
  }
  float safety_zero = 0.001f;
  delta_correlation = sqrt((xy*xy+safety_zero)/(xx*yy+safety_zero));

}

// really everything here should be across a window of "test flows" only
// making it over the full window to start "make correct, then make fast"
void CrossHypotheses::ComputeBasicResiduals() {

  for (unsigned int i_hyp=0; i_hyp<mod_predictions.size(); i_hyp++) {

    for (unsigned int j_flow = 0; j_flow<mod_predictions[i_hyp].size(); j_flow++) {
      residuals[i_hyp][j_flow] = mod_predictions[i_hyp][j_flow]-normalized[i_hyp][j_flow];
    }
  }
}

void CrossHypotheses::ResetModPredictions() {
  // basic residuals are obviously predicted - normalized under each hypothesis
  for (unsigned int i_hyp=0; i_hyp<mod_predictions.size(); i_hyp++) {
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      mod_predictions[i_hyp][j_flow] = predictions[i_hyp][j_flow];
    }
  }
}

void CrossHypotheses::ResetRelevantResiduals() {
  ResetModPredictions();
  // basic residuals are obviously predicted - normalized under each hypothesis
  for (unsigned int i_hyp=0; i_hyp<mod_predictions.size(); i_hyp++) {
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      residuals[i_hyp][j_flow] = mod_predictions[i_hyp][j_flow]-normalized[i_hyp][j_flow];
    }
  }
}

void CrossHypotheses::ComputeBasicLikelihoods() {

  //  basic_likelihoods.resize(residuals.size());
  for (unsigned int i_hyp=0; i_hyp<basic_likelihoods.size(); i_hyp++) {
    //    basic_likelihoods.at(i_hyp).resize(residuals.at(i_hyp).size());
    //    for (unsigned int j_flow = 0; j_flow<basic_likelihoods.at(i_hyp).size(); j_flow++) {
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      basic_likelihoods[i_hyp][j_flow] = my_t.TDistOddN(residuals[i_hyp][j_flow],sigma_estimate[i_hyp][j_flow],skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
    }
  }
}

void CrossHypotheses::UpdateRelevantLikelihoods() {
  for (unsigned int i_hyp=0; i_hyp<basic_likelihoods.size(); i_hyp++) {
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      basic_likelihoods[i_hyp][j_flow] = my_t.TDistOddN(residuals[i_hyp][j_flow],sigma_estimate[i_hyp][j_flow],skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
    }
  }
  ComputeLogLikelihoods(); // automatically over relevant likelihoods
}

void CrossHypotheses::ComputeLogLikelihoodsSum() {
  for (unsigned int i_hyp=0; i_hyp<log_likelihood.size(); i_hyp++) {
    log_likelihood[i_hyp] = 0.0f;
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      log_likelihood[i_hyp] += log(basic_likelihoods[i_hyp][j_flow]);  // keep from underflowing from multiplying
    }
  }
}


// and again:  project onto the first component when correlation high

void CrossHypotheses::JointLogLikelihood() {
  // if we cannot splice, note that delta is "0" for all entries
  // and this formula can blow up accidentally.
  // normally delta is at least minimum difference for test flows
  // delta is our official vector

  for (unsigned int i_hyp =0; i_hyp<log_likelihood.size(); i_hyp++) {

    float delta_scale = 0.001f; // safety level in case zeros happen
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      float d_val = delta_state.ServeDelta(i_hyp, j_flow);
      delta_scale += d_val *d_val;
    }
    delta_scale = sqrt(delta_scale);

    // now compute projection of residuals on this vector
    float res_projection=0.0f;
    float sigma_projection = 0.001f; // always some minimal variance in case we divide
    for (unsigned int t_flow = 0;  t_flow<test_flow.size(); t_flow++) {
      int j_flow = test_flow[t_flow];
      float d_val = delta_state.ServeDelta(i_hyp, j_flow);
      float res_component = residuals[i_hyp][j_flow] * d_val/ delta_scale;
      res_projection += res_component;

      /*for (unsigned int s_flow=0; s_flow<test_flow.size(); s_flow++) {
        int k_flow = test_flow.at(s_flow);
        // compute variance projection as well
        float sigma_component = delta.at(j_flow) *sigma_estimate.at(i_hyp).at(j_flow) *sigma_estimate.at(i_hyp).at(k_flow) * delta.at(k_flow) / (delta_scale * delta_scale);
        sigma_projection += sigma_component;
      }*/
      
      // only diagonal term to account for this estimate
      float sigma_component = d_val * sigma_estimate[i_hyp][j_flow] * sigma_estimate[i_hyp][j_flow] * d_val/(delta_scale * delta_scale);
      sigma_projection += sigma_component;
    }
    //    cout << i_hyp <<  "\t" << res_projection << "\t" << sqrt(sigma_projection) << endl;
    // now that we have r*u and u*sigma*u
    sigma_projection = sqrt(sigma_projection);
    float b_likelihood = my_t.TDistOddN(res_projection,sigma_projection,skew_estimate);
    log_likelihood[i_hyp] = log(b_likelihood);
  }
}

void CrossHypotheses::ComputeLogLikelihoods() {
  //cout << "ComputeLogLikelihoods" << endl;
  //  log_likelihood.resize(basic_likelihoods.size()); // number of hypotheses
  //@TODO: magic numbers bad for thresholds
  if ((fabs(delta_state.delta_correlation)<0.8f) || !use_correlated_likelihood) // suppress this feature for now
    ComputeLogLikelihoodsSum();
  else
    JointLogLikelihood();
  ComputeScaledLikelihood();
}

void CrossHypotheses::ComputeScaledLikelihood() {
  //cout << "ComputeScaledLikelihood" << endl;
  //  scaled_likelihood.resize(log_likelihood.size());
  // doesn't matter who I scale to, as long as we scale together
  ll_scale = log_likelihood[0];
  for (unsigned int i_hyp =0; i_hyp<scaled_likelihood.size(); i_hyp++)
    if (log_likelihood[i_hyp]>ll_scale)
      ll_scale = log_likelihood[i_hyp];

  for (unsigned int i_hyp =0; i_hyp<scaled_likelihood.size(); i_hyp++)
    scaled_likelihood[i_hyp] = exp(log_likelihood[i_hyp]-ll_scale);
  // really shouldn't happen, but sometimes does if splicing has gone awry
  // prevent log(0) events from happening in case we evaluate under weird circumstances
  scaled_likelihood[0] = max(scaled_likelihood[0], MINIMUM_RELATIVE_OUTLIER_PROBABILITY);
}

// this needs to run past sigma_generator
void CrossHypotheses::InitializeSigma() {
  // guess the standard deviation given the prediction
  // as a reasonable starting point for iteration
  // magic numbers from some typical experiments
  // size out to match predictions
  //  sigma_estimate.resize(predictions.size());
  for (unsigned int i_hyp=0; i_hyp<mod_predictions.size(); i_hyp++) {
    //    sigma_estimate.at(i_hyp).resize(predictions.at(i_hyp).size());
    for (unsigned int j_flow = 0; j_flow<mod_predictions[i_hyp].size(); j_flow++) {
      float square_level = mod_predictions[i_hyp][j_flow]*mod_predictions[i_hyp][j_flow]+1.0f;
      sigma_estimate[i_hyp][j_flow] = magic_sigma_slope*square_level+magic_sigma_base;
    }
  }
}


// The target number of test flows is max_choice, however
// - we add all valid test flows flows in between splice_start_flow and splice_end_flow
// - we limit potential test flows to a vicinity around splice_start_flow and splice_end_flow
bool CrossHypotheses::IsValidTestFlowIndexNew(unsigned int flow,unsigned int max_choice) {

  // Restriction to +40 flows around the splicing vicinity
  /*
  bool is_valid = (flow<(unsigned int)splice_end_flow+4*max_choice) and (flow<(unsigned int)max_last_flow) and (flow<predictions[0].size());
  is_valid = is_valid and ((test_flow.size()<max_choice) or (flow < (unsigned int)splice_end_flow));
  return is_valid;
  */
  return test_flow.size() < max_choice or flow < (unsigned int)splice_end_flow;
}

bool CrossHypotheses::IsValidTestFlowIndexOld(unsigned int flow,unsigned int max_choice) {

  // No restriction to flows around the splicing vicinity and strict enforcement of max_choice
  bool is_valid = (flow<(unsigned int)max_last_flow) and (flow<predictions[0].size()) and (test_flow.size()<max_choice);
  return is_valid;
}


bool CrossHypotheses::ComputeAllComparisonsTestFlow(float threshold, int max_choice)
{
  // test flows from all predictions

  bool tmp_success=true;
  test_flow.reserve(2*max_choice);
  test_flow.clear();
  float best = -1.0f;
  int bestlocus = 0;
  // Make sure our test flows are within a window of the splicing interval
  unsigned int start_flow = max(0, splice_start_flow-3*max_choice);
  unsigned int end_flow = min(splice_end_flow+4*max_choice, max_last_flow);

  // over all flows
  for (unsigned int j_flow=start_flow; j_flow < end_flow and IsValidTestFlowIndexNew(j_flow, max_choice); j_flow++) {
    // all comparisons(!)
    float m_delta = 0.0f;
    for (unsigned int i_hyp=0; i_hyp<predictions.size(); i_hyp++){
      for (unsigned int j_hyp=i_hyp+1; j_hyp<predictions.size(); j_hyp++){
        float t_delta = fabs(predictions[i_hyp][j_flow]-predictions[j_hyp][j_flow]);
        if (t_delta>m_delta)
          m_delta=t_delta;
      }
    }
    // maximum of any comparison
    if (m_delta>best) {
      best = m_delta;
      bestlocus = j_flow;
    }
    // I'm a test flow if I differ between anyone by more than threshold
    // need to make sure outliers are fully compelled
    if (m_delta>threshold) {
      test_flow.push_back(j_flow);
    }
  }
  // always some test flow
  if (test_flow.size()<1) {
    test_flow.push_back(bestlocus); // always at least one difference if nothing made threshold
    tmp_success=false; // but don't want it?
  }

  return(tmp_success);
}


float CrossHypotheses::ComputeLLDifference(int a_hyp, int b_hyp) {
  // difference in likelihoods between hypotheses
  return(fabs(log_likelihood[a_hyp]-log_likelihood[b_hyp]));
}


int CrossHypotheses::MostResponsible(){
  int most_r = 0;
  float best_r = responsibility[0];
  for (unsigned int i_hyp=1; i_hyp<responsibility.size(); i_hyp++){
    if (responsibility[i_hyp]>best_r){
      best_r= responsibility[i_hyp];
      most_r = i_hyp;
    }
  }
  return(most_r);
}

