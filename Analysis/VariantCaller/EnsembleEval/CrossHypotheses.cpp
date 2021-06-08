/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CrossHypotheses.h"
#include "RandSchrange.h"

PrecomputeTDistOddN::PrecomputeTDistOddN(){
    SetV(3);
}

// model as a t-distribution to slightly resist outliers
void PrecomputeTDistOddN::SetV(int _half_n){
	half_n_ = _half_n;
	v_ = (float) (2 * half_n_ - 1);
	pi_factor_ = 1.0f / (M_PI * sqrt(v_));
	v_factor_ = 1.0f;
    for (int i_prod = 1; i_prod < half_n_; ++i_prod){
      v_factor_ *= (v_ + 1.0f - 2.0f * i_prod) / (v_ - 2.0f * i_prod);
    }
    log_v_ = log(v_);
    log_factor_ = log(v_factor_) - (0.5f * log_v_ + log(M_PI)); //log_factor = log(pi_factor * pi_factor), 1.1447299 = log(pi)
};

float PrecomputeTDistOddN::TDistOddN(float res, float sigma, float skew) const{
  // skew t-dist one direction or the other
  float l_sigma = skew == 1.0f? sigma : (res > 0.0f? (sigma * skew) : (sigma / skew));
  float x = res / l_sigma;
  float my_likelihood = pi_factor_;
  float my_factor = v_/(v_+x*x);

  for (int i_prod=0; i_prod<half_n_; i_prod++) {
    my_likelihood *= my_factor;
  }
  my_likelihood *= v_factor_;
  //  for (int i_prod=1; i_prod<half_n; i_prod++) {
  //    my_likelihood *= (v+1.0f-2.0f*i_prod)/(v-2.0f*i_prod);
  //  }
  my_likelihood /= l_sigma;
  if (skew == 1.0f){
    return my_likelihood;
  }
  // account for skew
  float skew_factor = 2.0f*skew/(skew*skew+1.0f);
  my_likelihood *= skew_factor;
  return my_likelihood ;
}

// Operate in the log domain to slightly speed up the calculation
float PrecomputeTDistOddN::LogTDistOddN(float res, float sigma, float skew) const{
  // skew t-dist one direction or the other
  float l_sigma = sigma;
  if (skew != 1.0f){
	  l_sigma = (res > 0.0f)? sigma * skew : sigma / skew;
  }
  float x = res / l_sigma;
  float my_log_likelihood = log_factor_;

  my_log_likelihood += half_n_ * (log_v_ - log(v_ + x * x));
  my_log_likelihood -= log(l_sigma);
  if (skew == 1.0f){
    return my_log_likelihood;
  }
  // account for skew
  float skew_factor = log(2.0f*skew/(skew*skew+1.0f));
  my_log_likelihood += skew_factor;
  return my_log_likelihood;
}

HiddenBasis::HiddenBasis(){
  delta_correlation = 0.0f ;
}

float HiddenBasis::ServeDelta(int i_hyp, int t_flow){
  return delta[0][t_flow];
}

float HiddenBasis::ServeAltDelta(int i_alt, int t_flow){
  return delta[i_alt][t_flow];
}


void HiddenBasis::Allocate(unsigned int num_hyp, unsigned int num_test_flow){
  // ref-alt are my basis vectors
  // guaranteed to be all different
  int num_alt = num_hyp-2;
  delta.resize(num_alt); // num_alt
  for (unsigned int i_alt=0; i_alt<delta.size(); i_alt++)
    delta[i_alt].assign(num_test_flow, 0.0f);
}

CrossHypotheses::CrossHypotheses(){
    strand_key = -1;
    max_last_flow = -1;
    splice_start_flow = -1;
    splice_end_flow = -1;
    start_flow = -1;
    success = false;
    ll_scale = 0.0f;
    use_correlated_likelihood = false;
    read_counter = 1;
    at_least_one_same_as_null = false;
    min_last_flow = -1;
    skew_estimate = 1.0f;
    ptr_query_name = NULL;
}

// Static variables
PrecomputeTDistOddN CrossHypotheses::s_my_t_;
int CrossHypotheses::s_heavy_tailed_ = 3;  // t_5 degrees of freedom
bool CrossHypotheses::s_adjust_sigma_ = false;
float CrossHypotheses::s_sigma_factor_ = 1.0f;
int CrossHypotheses::s_max_flows_to_test = 10;
float CrossHypotheses::s_min_delta_for_flow = 0.1f;
float CrossHypotheses::s_magic_sigma_base = 0.085f;
float CrossHypotheses::s_magic_sigma_slope = 0.0084f;

void CrossHypotheses::SetHeavyTailed(int heavy_tailed, bool adjust_sigma){
	s_heavy_tailed_ = heavy_tailed;
	s_adjust_sigma_ = adjust_sigma;
	ApplyHeavyTailed();
}

void CrossHypotheses::ApplyHeavyTailed(){
	s_my_t_.SetV(s_heavy_tailed_); // 2*heavy_tailed - 1 = DoF of t-dist
	s_sigma_factor_ = s_adjust_sigma_? sqrt((2.0f * s_heavy_tailed_ - 3.0f) / (2.0f * s_heavy_tailed_ - 1.0f)) : 1.0f;
}

void CrossHypotheses::CleanAllocate(int num_hyp, int num_flow) {
  // allocate my vectors here
  responsibility.assign(num_hyp, 0.0f);
  weighted_responsibility.assign(num_hyp, 0.0f);
  log_likelihood.assign(num_hyp, 0.0f);
  scaled_likelihood.assign(num_hyp, 0.0f);

  tmp_prob_f.assign(num_hyp, 0.0f);
  tmp_prob_d.assign(num_hyp, 0.0);

  predictions.resize(num_hyp);
  predictions_all_flows.resize(num_hyp);
  mod_predictions.resize(num_hyp);
  normalized_all_flows.assign(num_flow, 0.0f);
  measurement_sd_all_flows.assign(num_flow, 0.0f);
  residuals.resize(num_hyp);
  sigma_estimate.resize(num_hyp);
  basic_log_likelihoods.resize(num_hyp);

  for (int i_hyp=0; i_hyp<num_hyp; i_hyp++) {
	  predictions_all_flows[i_hyp].assign(num_flow, 0.0f);
	  normalized_all_flows.assign(num_flow, 0.0f);
  }
}

void CrossHypotheses::ClearAllFlowsData(){
	normalized_all_flows.clear();
	predictions_all_flows.clear();
	measurement_sd_all_flows.clear();
}

void CrossHypotheses::FillInPrediction(PersistingThreadObjects &thread_objects, const Alignment& my_read, const InputStructures &global_context) {
  // allocate everything here
  CleanAllocate(instance_of_read_by_state.size(), global_context.flow_order_vector.at(my_read.flow_order_index).num_flows());
  // We search for test flows in the flow interval [(splice_start_flow-3*max_flows_to_test), (splice_end_flow+4*max_flows_to_test)]
  // We need to simulate further than the end of the search interval to get good predicted values within
  int flow_upper_bound = splice_end_flow + 4*s_max_flows_to_test + 20;
  CalculateHypPredictions(thread_objects, my_read, global_context, instance_of_read_by_state,
                                          same_as_null_hypothesis, predictions_all_flows, normalized_all_flows, min_last_flow, max_last_flow, flow_upper_bound);
  ResetModPredictions();
  strand_key = my_read.is_reverse_strand? 1 : 0;
  ptr_query_name = &(my_read.alignment.Name);

  // read_counter and measurements_sd for consensus reads
  read_counter = my_read.read_count;
  if (read_counter > 1){
    measurement_sd_all_flows = my_read.measurements_sd;
  }
  for (unsigned int i_hyp = 1; i_hyp < same_as_null_hypothesis.size(); ++i_hyp){
    at_least_one_same_as_null += same_as_null_hypothesis[i_hyp];
  }
}

void CrossHypotheses::InitializeTestFlows() {
  // Compute test flows for all hypotheses: flows changing by more than 0.1, 10 flows allowed
  ComputeAllComparisonsTestFlow(s_min_delta_for_flow, s_max_flows_to_test);
  InitializeRelevantToTestFlows();
  delta_state.ComputeDelta(predictions); // depends on predicted
  // compute cross-data across the deltas for multialleles
  delta_state.ComputeCross();
  // now compute possible  correlation amongst test flow data
  delta_state.ComputeDeltaCorrelation(predictions, test_flow);
}

// keep the information at test flows only
void CrossHypotheses::InitializeRelevantToTestFlows(){
	unsigned int num_hyp = predictions_all_flows.size();
	unsigned int test_flow_num = test_flow.size();

    for(unsigned int i_hyp = 0; i_hyp < num_hyp; ++i_hyp){
        predictions[i_hyp].assign(test_flow_num, 0.0f);
        mod_predictions[i_hyp].assign(test_flow_num, 0.0f);
        residuals[i_hyp].assign(test_flow_num, 0.0f);
        sigma_estimate[i_hyp].assign(test_flow_num, 0.0f);
        basic_log_likelihoods[i_hyp].assign(test_flow_num, 0.0f);
    }
	normalized.assign(test_flow_num, 0.0f);
	for(unsigned int t_flow = 0; t_flow < test_flow_num; ++t_flow){
		int j_flow = test_flow[t_flow];
		normalized[t_flow] = normalized_all_flows[j_flow];
		for(unsigned int i_hyp = 0; i_hyp < predictions.size(); ++i_hyp){
			predictions[i_hyp][t_flow] = predictions_all_flows[i_hyp][j_flow];
		}
	}
    delta_state.Allocate(num_hyp, test_flow_num);

    // If it is a consensus read
    if (read_counter > 1){
      // Note that if the ZS tag is not presented, I set measurement_sd to be all zeros.
    	measurement_var.assign(test_flow_num, 0.0f);
      	for(unsigned int t_flow = 0; t_flow < test_flow_num; ++t_flow){
          int j_flow = test_flow[t_flow];
          if (j_flow < (int) measurement_sd_all_flows.size()){
            measurement_var[t_flow] = measurement_sd_all_flows[j_flow] * measurement_sd_all_flows[j_flow];
          }
      	}
    }
}

void CrossHypotheses::InitializeDerivedQualities(const vector<vector<float> >& stranded_bias_adj) {
  InitializeResponsibility(); // depends on hypotheses
  // in theory don't need to compute any but test flows
  ResetModPredictions();  // make sure that mod-predictions=predictions
  ComputeResiduals(stranded_bias_adj); // predicted and measured

  InitializeSigma(); // depends on predicted

  ComputeBasicLogLikelihoods(); // depends on residuals and sigma
  // compute log-likelihoods
  ComputeLogLikelihoods();  // depends on test flow(s)
}

void CrossHypotheses::InitializeResponsibility() {
  responsibility[0] = 1.0f;  // everyone is an outlier until we trust you
  weighted_responsibility[0] = (float) read_counter;
  for (unsigned int i_hyp=1; i_hyp<responsibility.size(); i_hyp++){
    responsibility[i_hyp] = 0.0f;
    weighted_responsibility[i_hyp] = 0.0f;
  }
}



// responsibility depends on the relative global probability of the hypotheses and the likelihoods of the observations under each hypothesis
// divide the global probabilities into "typical" data points and outliers
// divide the variant probabilities into each hypothesis (summing to 1)
// treat the 2 hypothesis case to start with
void CrossHypotheses::UpdateResponsibility(const vector<float > &hyp_prob, float outlier_prob) {
  if (at_least_one_same_as_null){
	// In principle, an outlier read means it supports neither alleles.
	// In this case, the read can't be an outlier because the sequence as called is the same as an allele.
	// Hence I override outlier_prob to be extremely low.
	// Otherwise, the reads that supports the allele with AF < outlier prob will be treated as outliers.
	// This step can mitigate the bad parameter setting where min-allele-freq < outlier prob or the true AF ~ or < outlier prob
	// while it does not affect the outlier handling of the read that is a indeed an outlier.
    outlier_prob = min(outlier_prob, MINIMUM_RELATIVE_OUTLIER_PROBABILITY);
  }
  float typical_prob = 1.0f - outlier_prob;
  if (!success){
    //cout << "alert: fail to splice still called" << endl;
    InitializeResponsibility();
  } else {
	//  vector<double> tmp_prob(3);
	tmp_prob_d[0] = outlier_prob * scaled_likelihood[0];   // i'm an outlier
	for (unsigned int i_hyp=1; i_hyp<scaled_likelihood.size(); i_hyp++)
	  tmp_prob_d[i_hyp] = typical_prob * hyp_prob[i_hyp-1] * scaled_likelihood[i_hyp];

	double ll_denom = 0.0;
	for (unsigned int i_hyp=0; i_hyp<scaled_likelihood.size(); i_hyp++){
	  ll_denom += tmp_prob_d[i_hyp];
	}

	for (unsigned int i_hyp=0; i_hyp<responsibility.size(); i_hyp++){
	  responsibility[i_hyp] = tmp_prob_d[i_hyp]/ll_denom;
	  weighted_responsibility[i_hyp] = responsibility[i_hyp] * (float) read_counter;
	}
  }
}

float CrossHypotheses::ComputePosteriorLikelihood(const vector<float > &hyp_prob, float outlier_prob) {
  if (at_least_one_same_as_null){
	// In principle, an outlier read means it supports neither alleles.
	// In this case, the read can't be an outlier because the sequence as called is the same as an allele.
	// Hence I override outlier_prob to be extremely low.
	// Otherwise, the reads that supports the allele with AF < outlier prob will be treated as outliers.
	// This step can mitigate the bad parameter setting where min-allele-freq < outlier prob or the true AF ~ or < outlier prob
	// while it does not affect the outlier handling of the read that is a indeed an outlier.
    outlier_prob = min(outlier_prob, MINIMUM_RELATIVE_OUTLIER_PROBABILITY);
  }
  float typical_prob = 1.0f - outlier_prob;
  tmp_prob_f[0] = outlier_prob * scaled_likelihood[0];   // i'm an outlier
  for (unsigned int i_hyp=1; i_hyp<scaled_likelihood.size(); i_hyp++){
    tmp_prob_f[i_hyp] = typical_prob * hyp_prob[i_hyp-1] * scaled_likelihood[i_hyp];
  }
  float ll_denom = 0.0f;
  for (unsigned int i_hyp=0; i_hyp<scaled_likelihood.size(); i_hyp++) {
    ll_denom += tmp_prob_f[i_hyp];
  }

  // Notes for consensus reads (read_counter > 1):
  // scaled_likelihood is obtained from the "basic" log-likelihood weighted by read_counter, which is based on the assumption that all reads in the consensus read support the same truth of nature.
  // By doing this, error correction has been applied (in basic log-likelihood).
  // In the following line, the log-likelihood of AF contributed from the cosnsensus read is weighted by read counter,
  // which implicitly says that the reads in the consensus reads are "independent". This contradicts the assumption of the consensus read because the reads are highly correlated.
  // In practice, it works fine if the true AF is preserved in both family and read counts, though it is not mathematically strict unless I claim that I intentionally ignore the reason of forming consensus, but I really can't.
  return (read_counter == 1? (log(ll_denom) + ll_scale) : (log(ll_denom) + ll_scale) * (float) read_counter);  // log-likelihood under current distribution, including common value of log-likelihood-scale
}


void HiddenBasis::ComputeDelta(const vector<vector <float> > &predictions){
    unsigned int num_alt = predictions.size() - 2;
    int ref_hyp = 1;
	for (unsigned int i_alt=0; i_alt<num_alt; i_alt++){
		int alt_hyp = i_alt+2;
		for (unsigned int t_flow = 0; t_flow < delta[i_alt].size(); t_flow++) {
			delta[i_alt][t_flow] = predictions[alt_hyp][t_flow] - predictions[ref_hyp][t_flow];
		}
	}
}

// Notice that now delta is the difference at test_flows!
void HiddenBasis::ComputeCross(){
  // d_i = approximate basis vector
  // compute d_j*d_i/(d_i*d_i)
  unsigned int num_alt = delta.size();
  cross_cor.set_size(num_alt,num_alt);
  for (unsigned int i_alt =0; i_alt < num_alt; i_alt++){
    for (unsigned int j_alt=0; j_alt < num_alt; j_alt++){
      float my_top = 0.0f;
      float my_bottom = 0.0001f;  // delta might be all zeros if I am unlucky

      for (unsigned int t_flow = 0; t_flow < delta[j_alt].size(); t_flow++){
        my_top += delta[i_alt][t_flow] * delta[j_alt][t_flow];
        my_bottom += delta[i_alt][t_flow] * delta[i_alt][t_flow];
      }
      cross_cor.at(i_alt,j_alt) = my_top / my_bottom;  // row,column
    }
  }

  //I'll call this repeatedly, so it is worth setting up the inverse at this time
  arma::mat I(num_alt, num_alt);
  I.eye();
  float lambda = 0.00001f; // tikhonov in case we have colinearity, very possible for deletions/insertions

  arma::mat A(num_alt, num_alt);
      A = cross_cor.t() * cross_cor + lambda * I;
      cross_inv.set_size(num_alt, num_alt);
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
  for(unsigned int i_alt=0; i_alt < beta.size(); i_alt++){
    tmp_beta[i_alt] = beta[i_alt];
  }
  tmp_synthesis = cross_inv * tmp_beta;
  // tmp_synthesis now set for returning synthetic deviations
 // cout << "Synthetic: "<< tmp_synthesis << endl; // synthetic direction
}

float HiddenBasis::ServeCommonDirection(int t_flow){
  // tmp_synthesis must be set
  // otherwise we're getting nonsense
  float retval = 0.0f;
  for (unsigned int i_alt = 0; i_alt < delta.size(); i_alt++){
    retval += tmp_synthesis[i_alt] * delta[i_alt][t_flow];
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
    average_prediction[t_flow] += predictions[alt_hyp][t_flow];
    average_prediction[t_flow] /= 2.0f; // just ref/var to correspond to "delta"
  }
  // compute correlation = cos(theta)
  float xy, xx, yy;
  xy = xx = yy = 0.0f;
  for (unsigned int t_flow = 0; t_flow < test_flow.size(); t_flow++) {
    float j_delta = delta[i_alt][t_flow];
    xy += average_prediction[t_flow] * j_delta;
    xx += average_prediction[t_flow] * average_prediction[t_flow];
    yy += j_delta * j_delta;
  }
  float safety_zero = 0.001f;
  delta_correlation = sqrt((xy * xy + safety_zero) / (xx * yy + safety_zero));
}

// Site-specific signal adjustment is applied to the "residuals" if available.
// Important Notes for site-specific signal adjustment:
// 1) I do NOT eliminate the co-linear components of delta if there are adjustments on multiple alleles being applied!
// 2) So please specify FWDB/REVB in just "ONE" Hotspot allele if there are co-linear hotspot alleles.
// 3) The reason I don't use cross_inv to eliminate the co-linear components is that it requires to specify FWDB/REVB for all co-linear alleles, while I'm not able to do it if there is a de novo.
void CrossHypotheses::ComputeResiduals(const vector<vector<float> >& bias_adjustment){
  for (unsigned int t_flow = 0; t_flow < test_flow.size(); t_flow++) {
    // adjusted_normalized is the normalized measurement (i.e., ZM) with site-specific signal adjustment
	float adjusted_normalized = normalized[t_flow];
    // Apply site-specific signal adjustment here!
	if (not bias_adjustment.empty()){
      for (unsigned int i_alt = 0; i_alt < bias_adjustment[strand_key].size(); ++i_alt){
        // I don't adjust a delta of Strong FD.
    	if (local_flow_disruptiveness_matrix[1][i_alt + 2] > 1){
        	continue;
        }
    	// Important: Note that I do NOT eliminate the co-linear components of delta if there are adjustments on multiple alleles being applied!
    	adjusted_normalized += delta_state.delta[i_alt][t_flow] * bias_adjustment[strand_key][i_alt];
      }
    }
  	for (unsigned int i_hyp = 0; i_hyp < mod_predictions.size(); i_hyp++) {
      residuals[i_hyp][t_flow] = mod_predictions[i_hyp][t_flow] - adjusted_normalized;
    }
  }
}

void CrossHypotheses::ResetModPredictions() {
  mod_predictions.resize(predictions.size());
  for (unsigned int i_hyp = 0; i_hyp < mod_predictions.size(); ++i_hyp) {
	  copy(predictions[i_hyp].begin(), predictions[i_hyp].end(), mod_predictions[i_hyp].begin());
  }
}

void CrossHypotheses::ResetRelevantResiduals(const vector<vector<float> >& stranded_bias_adj) {
  ResetModPredictions();
  ComputeResiduals(stranded_bias_adj);
}

void CrossHypotheses::ComputeBasicLogLikelihoods() {
	// Non consensus case
	if (read_counter == 1){
		for (unsigned int i_hyp=0; i_hyp < basic_log_likelihoods.size(); i_hyp++) {
			for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
				float my_sigma = s_adjust_sigma_? sigma_estimate[i_hyp][t_flow] * s_sigma_factor_ : sigma_estimate[i_hyp][t_flow];
				basic_log_likelihoods[i_hyp][t_flow] = s_my_t_.LogTDistOddN(residuals[i_hyp][t_flow], my_sigma, skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
			}
		}
	}
	else{
		for (unsigned int i_hyp=0; i_hyp < basic_log_likelihoods.size(); i_hyp++) {
			for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
				// Super simple adjustment to capture the effect of the variation of the measurement for consensus reads just for now.
				//@TODO: Improve the adjustment or estimation for likelihood calculation.
				float adj_res = residuals[i_hyp][t_flow];
				if (measurement_var[t_flow] != 0.0f){
					adj_res = (adj_res > 0.0f) ? sqrt(adj_res * adj_res + measurement_var[t_flow]) : -sqrt(adj_res * adj_res + measurement_var[t_flow]);
				}
				float my_sigma = s_adjust_sigma_? sigma_estimate[i_hyp][t_flow] * s_sigma_factor_ : sigma_estimate[i_hyp][t_flow];
				basic_log_likelihoods[i_hyp][t_flow] = s_my_t_.LogTDistOddN(adj_res, my_sigma, skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
			}
		}
	}
}

void CrossHypotheses::UpdateRelevantLikelihoods() {
  ComputeBasicLogLikelihoods();
  ComputeLogLikelihoods(); // automatically over relevant likelihoods
}

void CrossHypotheses::ComputeLogLikelihoodsSum() {
  for (unsigned int i_hyp=0; i_hyp<log_likelihood.size(); i_hyp++) {
    log_likelihood[i_hyp] = 0.0f;
    for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
      log_likelihood[i_hyp] += basic_log_likelihoods[i_hyp][t_flow];  // keep from underflowing from multiplying
    }
    if (read_counter > 1){
      log_likelihood[i_hyp] *= (float) read_counter;
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
      float d_val = delta_state.ServeDelta(i_hyp, t_flow);
      delta_scale += d_val *d_val;
    }
    delta_scale = sqrt(delta_scale);

    // now compute projection of residuals on this vector
    float res_projection=0.0f;
    float sigma_projection = 0.001f; // always some minimal variance in case we divide
    for (unsigned int t_flow = 0;  t_flow<test_flow.size(); t_flow++) {
      float d_val = delta_state.ServeDelta(i_hyp, t_flow);
      float res_component = 0.0f;
      if (read_counter == 1){
    	  res_component = residuals[i_hyp][t_flow] * d_val/ delta_scale;
      }
      else{
		  float adj_res = 0.0f;
		  if (measurement_var[t_flow] == 0.0f){
			  adj_res = residuals[i_hyp][t_flow];
		  }
		  else{
			  adj_res = (residuals[i_hyp][t_flow] > 0)?
			     sqrt(residuals[i_hyp][t_flow] * residuals[i_hyp][t_flow] + measurement_var[t_flow]) : -sqrt(residuals[i_hyp][t_flow] * residuals[i_hyp][t_flow] + measurement_var[t_flow]);
		  }
    	  res_component = adj_res * d_val/ delta_scale;
      }



      res_projection += res_component;

      /*for (unsigned int s_flow=0; s_flow<test_flow.size(); s_flow++) {
        int k_flow = test_flow.at(s_flow);
        // compute variance projection as well
        float sigma_component = delta.at(j_flow) *sigma_estimate.at(i_hyp).at(j_flow) *sigma_estimate.at(i_hyp).at(k_flow) * delta.at(k_flow) / (delta_scale * delta_scale);
        sigma_projection += sigma_component;
      }*/
      
      // only diagonal term to account for this estimate
      float sigma_component = d_val * sigma_estimate[i_hyp][t_flow] * sigma_estimate[i_hyp][t_flow] * d_val/(delta_scale * delta_scale);
      sigma_projection += sigma_component;
    }
    //    cout << i_hyp <<  "\t" << res_projection << "\t" << sqrt(sigma_projection) << endl;
    // now that we have r*u and u*sigma*u
    sigma_projection = sqrt(sigma_projection);
    if (s_adjust_sigma_){
    	sigma_projection *= s_sigma_factor_;
    }
    float b_likelihood = s_my_t_.LogTDistOddN(res_projection,sigma_projection,skew_estimate);

    log_likelihood[i_hyp] = (read_counter == 1) ? b_likelihood : (float) read_counter * b_likelihood;
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
  for (unsigned int i_hyp =0; i_hyp<scaled_likelihood.size(); i_hyp++){
    if (log_likelihood[i_hyp] > ll_scale){
      ll_scale = log_likelihood[i_hyp];
    }
  }

  for (unsigned int i_hyp =0; i_hyp<scaled_likelihood.size(); i_hyp++){
    scaled_likelihood[i_hyp] = exp(log_likelihood[i_hyp]-ll_scale);
  }
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
    for (unsigned int t_flow = 0; t_flow<test_flow.size(); t_flow++) {
      float square_level = mod_predictions[i_hyp][t_flow] * mod_predictions[i_hyp][t_flow] + 1.0f;
      sigma_estimate[i_hyp][t_flow] = s_magic_sigma_slope * square_level + s_magic_sigma_base;
    }
  }
}

bool CompareDelta(const pair<float, unsigned int>& delta_1, const pair<float, unsigned int>& delta_2){
	return delta_1.first > delta_2.first;
}

bool CompareDeltaNum(const vector<pair<float, unsigned int> >& delta_vec_1,  const vector<pair<float, unsigned int> >& delta_vec_2){
	return delta_vec_1.size() > delta_vec_2.size();
}

// Select the flows to test.
// The rules are as follows:
// 1) Assuming the measurements are somehow close to the null hyp. A flow is informative for accept/reject i_hyp if fabs(prediction[0][i_flow] - prediction[i_hyp][i_flow]) >= threshold.
// 2) Must make sure the read has at least one flow to accept/reject every hypothesis. Otherwise, not success.
// 3) I force to pick up the most informative flow for each pair (i_hyp vs. null). This may cause the number of test flows > max_choice, but this is what we want.
// 4) Fill the test flows until the number of test flows reaches max_choice.
void CrossHypotheses::ComputeAllComparisonsTestFlow(float threshold, int max_choice){
	unsigned int window_start = (unsigned int) max(0, splice_start_flow - 2 * max_choice);
	// window_end needs to be consistent with the flow_upper_bound in this->FillInPrediction.
	// Which one works better? window_end <= window_end min_last_flow or window_end <= max_last_flow ?
	// My test shows  window_end <= max_last_flow gets better results.
	//@TODO: Can test_flows >= min_last_flow screw up delta correlation if I get a lot of zero padding prediction of the hyp?
	//unsigned int window_end = (unsigned int) min(splice_end_flow + 4 * max_choice, min_last_flow);
	unsigned int window_end = (unsigned int) min(splice_end_flow + 4 * max_choice, max_last_flow);
	// I pick up the flow in the window [start_flow, end_flow)
	unsigned int num_flows_in_window = window_end - window_start;
	unsigned int num_hyp = predictions_all_flows.size();
	// is_flow_selected[win_idx] tells flow (win_idx + window_start) is selected or not.
	vector<bool> is_flow_selected(num_flows_in_window, false);
	// pairwise_delta[pair_idx] records the candidate test flows for the hypotheses pair (null, i_hyp).
	vector<vector<pair<float, unsigned int> > > pairwise_delta;

	// Always make sure test_flow is empty.
	test_flow.resize(0);

	// Invalid read.
	if ((not success) or num_flows_in_window == 0){
		test_flow.push_back(splice_start_flow);
		success = false;
		return;
	}

	// Reserve some memory for the vectors.
	test_flow.reserve(max((int) num_hyp - 1, max_choice));
	pairwise_delta.reserve(num_hyp - 1);

	// Get the informative flows
	for (unsigned int i_hyp = 1; i_hyp < num_hyp; ++i_hyp){
		// Skip the i_hyp if it is the same as null hyp.
		if (same_as_null_hypothesis[i_hyp]){
			continue;
		}
		pairwise_delta.push_back(vector<pair<float, unsigned int> >(0));
		vector<pair<float, unsigned int> >& my_delta_vec = pairwise_delta.back();
		my_delta_vec.reserve(32);  // num_flows_in_window is too large. I usually don't get so many informative flows, particularly for HP-INDEL.
		unsigned int flow_idx = window_start;
		//@TODO: Since my_delta is calculated for each hyp vs. null hyp, I can speed up the multi-allele case by using a hyp-specific window.
		for (unsigned int win_idx = 0; win_idx < num_flows_in_window; ++win_idx, ++flow_idx){
			// Note that the measurements and predictions are all resize to the length of flow order, so accessing flow_idx should be safe.
			float my_delta = fabs(predictions_all_flows[0][flow_idx] - predictions_all_flows[i_hyp][flow_idx]);
			// Is it an informative flow to accept/reject i_hyp vs. null?
			if (my_delta >= threshold){
				// Handle outlier flows (TS-17983)
				if ((int) flow_idx < splice_end_flow){
					// I take the flows of large diff within splicing window in flow space.
					// Large residual (vs. the OL hyp) is fine (e.g., HP-INDEL on a high HP)
					my_delta_vec.push_back(pair<float, unsigned int>(my_delta, win_idx));
				}else if (fabs(normalized_all_flows[flow_idx] - predictions_all_flows[0][flow_idx]) < 0.5f){
					// For the flows outside of the splicing window, I don't want to take the flow if the residual (vs. the OL hyp) is too large.
					my_delta_vec.push_back(pair<float, unsigned int>(my_delta, win_idx));
				}
			}
		}

		// Not success if I didn't get any informative flow for i_hyp vs. null.

		if (my_delta_vec.empty()){
			if (test_flow.empty()){
				test_flow.push_back(splice_start_flow);
			}
			success = false;
			return;
		}

		// Sort the informative flows for i_hyp vs null (First one is the best).
		sort(my_delta_vec.begin(), my_delta_vec.end(), CompareDelta);

		// Always pick up the most informative flow for (i_hyp vs null).
		unsigned int best_flow_for_i_hyp = my_delta_vec[0].second;
		if (not is_flow_selected[best_flow_for_i_hyp]){
			is_flow_selected[best_flow_for_i_hyp] = true;
			test_flow.push_back(best_flow_for_i_hyp + window_start);
		}
	}

	// Not success if so far I can't get any test flow. Shouldn't happen, just for safety.
	if (test_flow.empty()){
		test_flow.push_back(splice_start_flow);
		success = false;
		return;
	}

	// Sort pairwise_delta by the number of informative flows of the pairs (first pair has the most flows).
	sort(pairwise_delta.begin(), pairwise_delta.end(), CompareDeltaNum);

	// At the round = j, add the j-th best informative flow for each hyp pair. Stop as soon as the number of test_flows >= max_choice.
	// Note that round = 0 has been carried out.
	for (unsigned int round = 1; round < pairwise_delta[0].size() and (int) test_flow.size() <= max_choice; ++round){
		// Continue to the next round if round < pairwise_delta[pair_idx].size() because pairwise_delta is sorted by the size of each entry in the descending order.
		for (unsigned int pair_idx = 0; pair_idx < pairwise_delta.size() and round < pairwise_delta[pair_idx].size() and (int) test_flow.size() <= max_choice; ++pair_idx){
			// Pick up this flow.
			unsigned int add_this_flow = pairwise_delta[pair_idx][round].second;
			if (not is_flow_selected[add_this_flow]){
				is_flow_selected[add_this_flow] = true;
				test_flow.push_back(add_this_flow + window_start);
			}
		}
	}

	// Finally sort test_flow.
	sort(test_flow.begin(), test_flow.end());
}

float CrossHypotheses::ComputeLLDifference(int a_hyp, int b_hyp) const{
  // difference in likelihoods between hypotheses
  return(fabs(log_likelihood[a_hyp]-log_likelihood[b_hyp]));
}


int CrossHypotheses::MostResponsible() const{
  int most_r = 0;
  float best_r = responsibility[0];
  for (unsigned int i_hyp = 1; i_hyp < responsibility.size(); i_hyp++){
    if (responsibility[i_hyp] > best_r){
      best_r = responsibility[i_hyp];
      most_r = i_hyp;
    }
  }
  return(most_r);
}

void EvalFamily::InitializeEvalFamily(unsigned int num_hyp){
	my_family_cross_.success = true;
	CleanAllocate(num_hyp);
	InitializeFamilyResponsibility();
}

void EvalFamily::CleanAllocate(unsigned int num_hyp){
	// I only clean and allocate the ones that I need.
	my_family_cross_.responsibility.assign(num_hyp, 0.0f);
	my_family_cross_.weighted_responsibility.assign(num_hyp, 0.0f);
	family_responsibility.assign(num_hyp, 0.0f);
	my_family_cross_.log_likelihood.assign(num_hyp, 0.0f);
	my_family_cross_.scaled_likelihood.assign(num_hyp, 0.0f);
	my_family_cross_.tmp_prob_f.assign(num_hyp, 0.0f);
	my_family_cross_.tmp_prob_d.assign(num_hyp, 0.0f);
}

void EvalFamily::InitializeFamilyResponsibility(){
	my_family_cross_.responsibility[0] = 1.0f;
	family_responsibility[0] = 1.0f;
	for(unsigned int i_hyp = 1; i_hyp < family_responsibility.size(); ++i_hyp){
		my_family_cross_.responsibility[i_hyp] = 0.0f;
		family_responsibility[i_hyp] = 0.0f;
	}
}

void EvalFamily::ComputeFamilyLogLikelihoods(const vector<CrossHypotheses> &my_hypotheses){
	// Let the log-likelihood of the OL hypothesis approach log(0).
	my_family_cross_.log_likelihood[0] = -1.0E16;
	// accumulate the log-likelihood from the reads of the family for not null hypotheses
	for (unsigned int i_hyp = 1 ; i_hyp < my_family_cross_.log_likelihood.size(); i_hyp++) {
		my_family_cross_.log_likelihood[i_hyp] = 0.0f;
		for (unsigned int i_member = 0; i_member < valid_family_members.size(); ++i_member){
			unsigned int i_read = valid_family_members[i_member];
			my_family_cross_.log_likelihood[i_hyp] += ((1.0f - my_hypotheses[i_read].responsibility[0]) * my_hypotheses[i_read].log_likelihood[i_hyp]);
		}
		// Make sure log-likelihood of the OL hypothesis << the log-likelihood of every hypothesis.
		// Do not compare with (my_family_cross_.log_likelihood[i_hyp] subtract something) in case floating point accuracy problem.
		my_family_cross_.log_likelihood[0] = min(my_family_cross_.log_likelihood[0], my_family_cross_.log_likelihood[i_hyp] * 2.0f);
	}
	my_family_cross_.ComputeScaledLikelihood();
}

// ComputeFamilyLogLikelihoods(...) must be done first!
void EvalFamily::UpdateFamilyResponsibility(const vector<float > &hyp_prob, float outlier_prob){
	my_family_cross_.UpdateResponsibility(hyp_prob, outlier_prob);
	family_responsibility = my_family_cross_.responsibility;
}

// (Note 1): my_family_cross_.log_likelihood[0] is set to be super low
// (Note 2): family_responsibility[0] is obtained using an ad hoc fashion (not derived from family log-likelihood)
// So I can't calculate the family posterior likelihood from scaled_likelihood[0].
// I kind of reversely engineering from family_responsibility[0] to family posterior likelihood
float EvalFamily::ComputeFamilyPosteriorLikelihood(const vector<float> &hyp_prob){
	float ll_denom = 0.0f;
	float safety_zero = 1.0E-12;

	for (unsigned int i_hyp = 1; i_hyp < my_family_cross_.scaled_likelihood.size(); ++i_hyp){
		ll_denom += hyp_prob[i_hyp - 1] * my_family_cross_.scaled_likelihood[i_hyp];
	}
	ll_denom *= (1.0f - family_responsibility[0]);
	// If family_responsibility[0] is high, FamilyPosteriorLikelihood is dominated by family_responsibility[0], i.e., not a function of hyp_prob.
	ll_denom += family_responsibility[0];
	return log(ll_denom + safety_zero) + my_family_cross_.ll_scale;  // log-likelihood under current distribution, including common value of log-likelihood-scale
}

int EvalFamily::CountFamSizeFromAll()
{
	fam_size_ = 0;
	fam_cov_fwd_ = 0;
	fam_cov_rev_ = 0;
	for (vector<unsigned int>::iterator read_it = all_family_members.begin(); read_it != all_family_members.end(); ++read_it){
		if (read_stack_->at(*read_it)->is_reverse_strand){
			fam_cov_rev_ += (read_stack_->at(*read_it)->read_count);
		}else{
			fam_cov_fwd_ += (read_stack_->at(*read_it)->read_count);
		}
	}
	fam_size_ = fam_cov_fwd_ + fam_cov_rev_;
	return fam_size_;
}

int EvalFamily::CountFamSizeFromValid()
{
	valid_fam_size_ = 0;
	valid_fam_cov_fwd_ = 0;
	valid_fam_cov_rev_ = 0;
	for (vector<unsigned int>::iterator read_it = valid_family_members.begin(); read_it != valid_family_members.end(); ++read_it){
		if (read_stack_->at(*read_it)->is_reverse_strand){
			valid_fam_cov_rev_ += (read_stack_->at(*read_it)->read_count);
		}else{
			valid_fam_cov_fwd_ += (read_stack_->at(*read_it)->read_count);
		}
	}
	valid_fam_size_ = valid_fam_cov_fwd_ + valid_fam_cov_rev_;
	return valid_fam_size_;
}

// I define Family outlier responsibility = P(# non-outlier read members < min_fam_size)
// In theory, Resp(OL family) is obtained from the cdf of a multinomial distribution.
// Since Resp(OL family) >= min Resp(OL read), I use the lower bound to approximate Resp(OL family) for some extreme cases.
// Otherwise, I calculate Resp(OL family) via Monte-Carlo simulation.
// (Note 1): my_cross_.responsibility is not derived from my_cross_.log_likelihood in this function.
// @TODO: Ugly if conditions that need improvement.
void EvalFamily::ComputeFamilyOutlierResponsibility(const vector<CrossHypotheses> &my_hypotheses, unsigned int min_fam_size)
{
	float family_ol_resp = 1.0f;
	float safety_zero = 0.000001f;
	float min_read_ol_resp = 1.0f;
	float max_read_ol_resp = 0.0f;
	float weighted_avg_read_ol_resp = 0.0f;

	int semi_hard_fam_size = 0;

	vector<unsigned int> potential_outlier_reads;

	for (vector<unsigned int>::iterator read_idx_it = valid_family_members.begin(); read_idx_it != valid_family_members.end(); ++read_idx_it){
		min_read_ol_resp = min(min_read_ol_resp, my_hypotheses[*read_idx_it].responsibility[0]);
		max_read_ol_resp = max(max_read_ol_resp, my_hypotheses[*read_idx_it].responsibility[0]);
		weighted_avg_read_ol_resp += (my_hypotheses[*read_idx_it].responsibility[0] * (float) my_hypotheses[*read_idx_it].read_counter);
		// The following criterion basically claims that the read is not an outlier.
		if (my_hypotheses[*read_idx_it].at_least_one_same_as_null){
			// This condition basically implies that a hypothesis is the same as null. So it can't be an outlier.
			semi_hard_fam_size += my_hypotheses[*read_idx_it].read_counter;
		}
		else{
			potential_outlier_reads.push_back(*read_idx_it);
		}
	}
	int num_fam_needed = (int) min_fam_size - semi_hard_fam_size;

	if (semi_hard_fam_size >= (int) min_fam_size){
		// I am very sure that the the family consists of sufficient number of non-outlier reads (which is the most ubiquitous case).
		// Don't waste time doing Monte Carlo simulation. Set family_ol_resp to close to zero.
		family_ol_resp = safety_zero;
	}
	else if (min_read_ol_resp >= 0.8f){
		//@TODO: Come up with something better than the hard-coded threshold.
		// It seems that all reads in the family are very likely to be outliers.
		// Don't waste time doing Monte Carlo simulation. Use the averaged read_ol_reso as family_ol_resp.
		family_ol_resp = weighted_avg_read_ol_resp / (float) GetValidFamSize();
	}
	else if (potential_outlier_reads.size() == 1){
		// Another trivial case where I don't need to go through Monte Carlo.
		family_ol_resp = my_hypotheses[potential_outlier_reads[0]].responsibility[0];
	}
	else{
		// I calculate family outlier responsibility using Monte Carlo simulation.
		RandSchrange RandGen(1729); // I choose 1729 as the seed because it is the Hardy–Ramanujan number.
		int num_trails = 200;
		int num_ol = 0;
		// Only deal with those potential outlier reads.
		vector<int> min_rand_for_non_ol(potential_outlier_reads.size());
		for (unsigned int read_idx = 0; read_idx < min_rand_for_non_ol.size(); ++read_idx){
			min_rand_for_non_ol[read_idx] = (int) ((double)(RandGen.RandMax) * (double) my_hypotheses[potential_outlier_reads[read_idx]].responsibility[0]);
		}

		for (int trail_round = 0; trail_round < num_trails; ++trail_round){
			int trailed_fam_size = 0;
			for (unsigned int read_idx = 0; read_idx < min_rand_for_non_ol.size(); ++read_idx){
				// Toss a coin: head = outlier, tail = not outlier
				bool is_not_ol = RandGen.Rand() > min_rand_for_non_ol[read_idx];
				if (is_not_ol){
					trailed_fam_size += my_hypotheses[potential_outlier_reads[read_idx]].read_counter;
				}
			}
			num_ol += (trailed_fam_size < num_fam_needed);
		}
		family_ol_resp = (float) num_ol / (float) num_trails;
	}

	// Guard by safety zero in case something crazy happened
	family_ol_resp = max(family_ol_resp, safety_zero);
	family_ol_resp = min(family_ol_resp, 1.0f - safety_zero);

	// Normalize the family responsibility
	float sum_of_resp = family_ol_resp;
	float normalization_factor = (1.0f - family_ol_resp) / (1.0f - my_family_cross_.responsibility[0]);
	my_family_cross_.responsibility[0] = family_ol_resp;
	for (unsigned int i_hyp = 1; i_hyp < my_family_cross_.responsibility.size(); ++i_hyp){
		my_family_cross_.responsibility[i_hyp] *= normalization_factor;
		sum_of_resp += my_family_cross_.responsibility[i_hyp];
	}
	family_responsibility = my_family_cross_.responsibility;
	assert(abs(sum_of_resp - 1.0f) < 0.0001f);
}


void RemoveHp(const string& base_seq, string& hp_removed_base_seq){
    hp_removed_base_seq.resize(0);
    hp_removed_base_seq.reserve(base_seq.size());
    if (base_seq.empty()){
        return;
    }
    for (string::const_iterator nuc_it = base_seq.begin(); nuc_it != base_seq.end(); ++nuc_it){
        if (*nuc_it != hp_removed_base_seq.back()){
            hp_removed_base_seq.push_back(*nuc_it);
        }
    }
}

bool IsHpIndel(const string& seq_1, const string& seq_2)
{
    if (seq_1.empty() or seq_2.empty()){
        return false;
    }

    bool hp_indel_found = false;
    int hp_len_1 = 0;
    int hp_len_2 = 0;
    string::const_iterator nuc_it_1 = seq_1.begin();
    string::const_iterator nuc_it_2 = seq_2.begin();
    while (nuc_it_1 != seq_1.end() or nuc_it_2 != seq_2.end()){
        if (*nuc_it_1 != *nuc_it_2){
            return false;
        }
        ++nuc_it_1;
        ++nuc_it_2;
        hp_len_1 = 1;
        hp_len_2 = 1;
        while (*nuc_it_1 == *(nuc_it_1 -1)){
            ++hp_len_1;
            ++nuc_it_1;
        }
        while (*nuc_it_2 == *(nuc_it_2 -1)){
            ++hp_len_2;
            ++nuc_it_2;
        }
        if (hp_len_1 != hp_len_2){
            if (hp_indel_found){
                return false;
            }
            hp_indel_found = true;
        }
    }
    return true;
}

// The function is implemented based on the assumption that instance_of_read_by_state is obtained from "splicing",
// I claim instance_of_read_by_state[i_hyp] and instance_of_read_by_state[j_hyp] are non-flow-disruptive if the first common suffix bases of the two hypotheses are mainly incorporated at the same flow.
void CrossHypotheses::FillInFlowDisruptivenessMatrix(const ion::FlowOrder &flow_order, const Alignment &my_read)
{
	// Every hypotheses pair starts with indefinite
	local_flow_disruptiveness_matrix.assign(instance_of_read_by_state.size(), vector<int>(instance_of_read_by_state.size(), -1));

	if (not success){
		return;
	}

	int common_prefix_len = -1; // length of common starting bases of all hypotheses. What I mean prefix here is NOT my_read.prefix_bases. It is the common starting bases used in splicing.
	int min_instance_of_read_by_state_len = (int) instance_of_read_by_state[0].size();
	for (unsigned int i_hyp = 1; i_hyp < instance_of_read_by_state.size(); ++i_hyp){
		min_instance_of_read_by_state_len = min(min_instance_of_read_by_state_len, (int) instance_of_read_by_state[i_hyp].size());
	}

	// Find the length of common starting bases for all hypotheses
	int base_idx = 0;
	while (common_prefix_len < 0 and base_idx < min_instance_of_read_by_state_len){
		for (unsigned int i_hyp = 1; i_hyp < instance_of_read_by_state.size(); ++i_hyp){
			if (instance_of_read_by_state[0][base_idx] != instance_of_read_by_state[i_hyp][base_idx]){
				common_prefix_len = base_idx;
				break;
			}
		}
		++base_idx;
	}

	// Check if I didn't see any delta, e.g., variant at the end of the read.
	if (common_prefix_len <= 0){
		common_prefix_len = min_instance_of_read_by_state_len;
	}

	char anchor_base = 0; // anchor_base is the last common prefix base of all hypotheses
	int flow_index_of_anchor_base = my_read.start_flow;
	if (common_prefix_len == 0){
		anchor_base = my_read.prefix_bases.back();
		// Find the flow index of anchor_base
	    while (flow_index_of_anchor_base >= 0 and flow_order.nuc_at(flow_index_of_anchor_base) != anchor_base){
	    	--flow_index_of_anchor_base;
	    }
	}
	else{
		anchor_base = instance_of_read_by_state[0][common_prefix_len - 1];
		flow_index_of_anchor_base = my_read.flow_index[common_prefix_len - 1];
	}
	vector<vector<int> > flow_order_index_start_from_anchor; // flow_order index for each hypothesis, starting from the anchor base
	// i.e., flow_order_index_start_from_anchor[i_hyp][idx] = the index of the main incorporating flow of instance_of_read_by_state[i_hyp][idx + common_prefix_len - 1]
	flow_order_index_start_from_anchor.assign(instance_of_read_by_state.size(), {flow_index_of_anchor_base});

	for (unsigned int i_hyp = 0; i_hyp < instance_of_read_by_state.size(); ++i_hyp){
		local_flow_disruptiveness_matrix[i_hyp][i_hyp] = 0; // identical means INDEL length 0.
		for (unsigned int j_hyp = i_hyp + 1; j_hyp < instance_of_read_by_state.size(); ++j_hyp){
			if (i_hyp == 0 and same_as_null_hypothesis[j_hyp]){
				// same_as_null_hypothesis means identical
				local_flow_disruptiveness_matrix[0][j_hyp] = 0;
				local_flow_disruptiveness_matrix[j_hyp][0] = 0;
				continue;
			}

			// Skip the non-null hyp whcih is the same as the null hyp, since it has been carried out.
			if (i_hyp > 0 and same_as_null_hypothesis[i_hyp]){
				local_flow_disruptiveness_matrix[i_hyp][j_hyp] = local_flow_disruptiveness_matrix[0][j_hyp];
				local_flow_disruptiveness_matrix[j_hyp][i_hyp] = local_flow_disruptiveness_matrix[0][j_hyp];
				continue;
			}

			// determine the common_prefix_len for i_hyp, j_hyp
			int common_prefix_len_i_j_pair = common_prefix_len;
			while (instance_of_read_by_state[i_hyp][common_prefix_len_i_j_pair] == instance_of_read_by_state[j_hyp][common_prefix_len_i_j_pair]){
				++common_prefix_len_i_j_pair;
				if (common_prefix_len_i_j_pair >= (int) min(instance_of_read_by_state[i_hyp].size(), instance_of_read_by_state[j_hyp].size()))
					break;
			}

			// determine the common_suffix_len for i_hyp, j_hyp
			// i_idx + 1, j_idx + 1 are the indices of the first common suffix base of i_hyp, j_hyp, respectively.
			int i_idx = (int) instance_of_read_by_state[i_hyp].size() - 1;
			int j_idx = (int) instance_of_read_by_state[j_hyp].size() - 1;
			int common_suffix_len = 0; // The number of common ending bases of instance_of_read_by_state[i_hyp] and instance_of_read_by_state[j_hyp]
			while ((min(i_idx, j_idx) > common_prefix_len_i_j_pair) and instance_of_read_by_state[i_hyp][i_idx] == instance_of_read_by_state[j_hyp][j_idx]){
				--i_idx;
				--j_idx;
				++common_suffix_len;
			}

			if (common_suffix_len == 0){
				// The flow-disruptiveness is indefinite because there is no common suffix bases (or may be hard-clipped).
				// For example, the variant position is at the end of the read.
				//@TODO: Append or use the suffix bases (if any, e.g., suffix molecular tag) if there is no other hard-clipped base (Usually, it is safe if --trim-ampliseq-primers=on).
				continue;
			}

			// Check HP-INDEL first because it is the easiest (and probably the most ubiquitous) one.
			if (IsHpIndel(instance_of_read_by_state[i_hyp].substr(common_prefix_len_i_j_pair - 1, i_idx + 3 - common_prefix_len_i_j_pair),
				      	  instance_of_read_by_state[j_hyp].substr(common_prefix_len_i_j_pair - 1, j_idx + 3 - common_prefix_len_i_j_pair))){
				local_flow_disruptiveness_matrix[i_hyp][j_hyp] = 0;
				local_flow_disruptiveness_matrix[j_hyp][i_hyp] = 0;
				continue;
			}

			// Now fill flow_order_index_start_from_anchor since I am here (not HP-INDEL).
			int flow_i_hyp = flow_order_index_start_from_anchor[i_hyp].back();
			for (int base_idx = (int) flow_order_index_start_from_anchor[i_hyp].size() + common_prefix_len - 1; base_idx <= i_idx + 1; ++base_idx){
				IncrementFlow(flow_order, instance_of_read_by_state[i_hyp][base_idx], flow_i_hyp);
				flow_order_index_start_from_anchor[i_hyp].push_back(flow_i_hyp);
			}
			int flow_j_hyp = flow_order_index_start_from_anchor[j_hyp].back();
			for (int base_idx = (int) flow_order_index_start_from_anchor[j_hyp].size() + common_prefix_len - 1; base_idx <= j_idx + 1; ++base_idx){
				IncrementFlow(flow_order, instance_of_read_by_state[j_hyp][base_idx], flow_j_hyp);
				flow_order_index_start_from_anchor[j_hyp].push_back(flow_j_hyp);
			}

			bool is_i_j_fd = flow_order_index_start_from_anchor[i_hyp][i_idx + 2 - common_prefix_len] != flow_order_index_start_from_anchor[j_hyp][j_idx + 2 - common_prefix_len];

			if ((not is_i_j_fd) and flow_order_index_start_from_anchor[i_hyp][i_idx + 2 - common_prefix_len] == flow_order.num_flows()){
				// The flow order is not long enough to represent the first common suffix base for both i_hyp and j_hyp. So the flow-disruptiveness is indefinite.
				continue;
			}

			// Compare the index of the main incorporating flows of the first common suffix bases of the two hypotheses
			if (is_i_j_fd){
				local_flow_disruptiveness_matrix[i_hyp][j_hyp] = 2;
				local_flow_disruptiveness_matrix[j_hyp][i_hyp] = 2;
			}
			else{
				// Not FD and not HP-INDEL
				local_flow_disruptiveness_matrix[i_hyp][j_hyp] = 1;
				local_flow_disruptiveness_matrix[j_hyp][i_hyp] = 1;
			}
		}
	}
}

// A simple criterion to determine outlier reads
// If I can not find any non-null hypothesis which is not flow-disruptive with the null hypothesis, I claim the read is an outluer and return true.
// return true if I am an outlier else false.
// stringency_level = 0: disable the filter
// stringency_level = 1: not OL if at least one Hyp vs. Hyp(OL) is FD-5
// stringency_level = 2: not OL if at least one Hyp vs. Hyp(OL) is FD-0 (HP-INDEL)
// stringency_level = 3: not OL if at least one Hyp is the same as Hyp(OL)
bool CrossHypotheses::OutlierByFlowDisruptiveness(unsigned int stringency_level) const {
	if ((not success) or local_flow_disruptiveness_matrix.empty()){
		return true;
	}
	// stringency_level = 0 means disable the filter
	// at_least_one_same_as_null means not an OL. Should be the most common case.
	if (stringency_level == 0 or at_least_one_same_as_null){
		return false;
	}
	else if (stringency_level >= 3){
		// Claime OL because none of the Hyp is the same as Hyp(OL).
		return true;
	}

	int min_fd_code = -1;
	for (unsigned int i_hyp = 1; i_hyp < local_flow_disruptiveness_matrix[0].size(); ++i_hyp){
		if (local_flow_disruptiveness_matrix[0][i_hyp] >= 0){
			if (min_fd_code < 0){
				min_fd_code = local_flow_disruptiveness_matrix[0][i_hyp];
			}else{
				min_fd_code = min(min_fd_code, local_flow_disruptiveness_matrix[0][i_hyp]);
			}
		}
	}

	if (min_fd_code < 0){
		// Filter out indefinite read.
		return true;
	}

	/*
	switch (stringency_level){
	case 1:
		return min_fd_code >= 2;
	case 2:
		return min_fd_code >= 1;
	}
	*/
	// The above switch is equivalent to the following return value.
	return min_fd_code >= (3 - (int) stringency_level);
}

//CZB: Majority rule makes no sense for BIR-DIR UMT.
void EvalFamily::FillInFlowDisruptivenessMatrix(const vector<CrossHypotheses> &my_hypotheses)
{
	unsigned int num_hyp = 0;
	for(unsigned int i_member = 0; i_member < valid_family_members.size(); ++i_member){
		if (my_hypotheses[valid_family_members[i_member]].success){
			num_hyp = my_hypotheses[valid_family_members[i_member]].instance_of_read_by_state.size();
			break;
		}
	}
	if (num_hyp == 0){
		my_family_cross_.local_flow_disruptiveness_matrix.clear();
		return;
	}

	my_family_cross_.local_flow_disruptiveness_matrix.assign(num_hyp, vector<int>(num_hyp, -1));
	// flow_disruptiveness_matrix[0][i_hyp] and flow_disruptiveness_matrix[i_hyp][0] are indefinite.
	for (unsigned int i_hyp = 1; i_hyp < num_hyp; ++i_hyp){
		my_family_cross_.local_flow_disruptiveness_matrix[i_hyp][i_hyp] = 0;
		for (unsigned int j_hyp = i_hyp + 1; j_hyp < num_hyp; ++j_hyp){
			// Note that loacal_flow_disruptiveness_matrix[i_hyp][j_hyp] = -1, 0, 1, 2 means indefinite, HP-INDEL, not-FD and not HP-INDEL, FD, respectively.
			// Use majority rule to determine flow_disruptiveness_matrix of the family.
			vector<int> fd_type_counts(3);
			for (unsigned int i_member = 0; i_member < valid_family_members.size(); ++i_member){
				// Don't count outliers.
				const CrossHypotheses* my_member = &(my_hypotheses[valid_family_members[i_member]]);
				int fd_type = my_member->local_flow_disruptiveness_matrix[i_hyp][j_hyp];
				if (fd_type >= 0){
					fd_type_counts[fd_type] += my_member->read_counter;
				}
			}
			int max_type = -1;
			int max_count = 1;
			for (unsigned int fd_type = 0; fd_type < 3; ++fd_type){
				// Claim higher FD level in the tie cases.
				if (fd_type_counts[fd_type] >= max_count){
					max_type = fd_type;
					max_count = fd_type_counts[fd_type];
				}
			}
			my_family_cross_.local_flow_disruptiveness_matrix[i_hyp][j_hyp] = max_type;
			my_family_cross_.local_flow_disruptiveness_matrix[j_hyp][i_hyp] = max_type;
		}
	}
}


