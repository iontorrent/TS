/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CrossHypotheses.h"
#include "RandSchrange.h"

// model as a t-distribution to slightly resist outliers

void PrecomputeTDistOddN::SetV(int _half_n){
  half_n = _half_n;
  v = (float) (2*half_n-1);
  pi_factor = 1.0f/(3.14159f*sqrt(v));
  v_factor = 1.0f;
  for (int i_prod=1; i_prod<half_n; i_prod++) {
    v_factor *= (v+1.0f-2.0f*i_prod)/(v-2.0f*i_prod);
  }

  log_v = log(v);
  log_factor = log(v_factor) - (0.5f * log_v + 1.1447299f); //log_factor = log(pi_factor * pi_factor), 1.1447299 = log(pi)
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
  if (skew == 1.0f){
    return my_likelihood;
  }
  // account for skew
  float skew_factor = 2.0f*skew/(skew*skew+1.0f);
  my_likelihood *= skew_factor;
  return my_likelihood ;
}

// Operate in the log domain to slightly speed up the calculation
float PrecomputeTDistOddN::LogTDistOddN(float res, float sigma, float skew){
  // skew t-dist one direction or the other
  float l_sigma = sigma;
  if (skew != 1.0f){
	  l_sigma = (res > 0.0f)? sigma * skew : sigma / skew;
  }
  float x = res / l_sigma;
  float my_log_likelihood = log_factor;

  my_log_likelihood += half_n * (log_v - log(v + x * x));
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

void  CrossHypotheses::ClearAllFlowsData(){
	normalized_all_flows.clear();
	predictions_all_flows.clear();
	measurement_sd_all_flows.clear();
}

void CrossHypotheses::SetModPredictions() {
  // modified predictions reset from predictions
  for (unsigned int i_hyp=0; i_hyp<predictions.size(); i_hyp++) {
	mod_predictions[i_hyp].assign(test_flow.size(), 0.0f);
	for(unsigned int t_flow = 0; t_flow < test_flow.size(); ++t_flow){
      mod_predictions[i_hyp][t_flow] = predictions[i_hyp][t_flow];
	}
  }
}


void CrossHypotheses::FillInPrediction(PersistingThreadObjects &thread_objects, const Alignment& my_read, const InputStructures &global_context) {
  // allocate everything here
  CleanAllocate(instance_of_read_by_state.size(), global_context.flow_order_vector.at(my_read.flow_order_index).num_flows());
  // We search for test flows in the flow interval [(splice_start_flow-3*max_flows_to_test), (splice_end_flow+4*max_flows_to_test)]
  // We need to simulate further than the end of the search interval to get good predicted values within
  int flow_upper_bound = splice_end_flow + 4*max_flows_to_test + 20;
  max_last_flow = CalculateHypPredictions(thread_objects, my_read, global_context, instance_of_read_by_state,
                                          same_as_null_hypothesis, predictions_all_flows, normalized_all_flows, flow_upper_bound);
  SetModPredictions();
  if (my_read.is_reverse_strand)
    strand_key = 1;
  else
    strand_key = 0;

  // read_counter and measurements_sd for consensus reads
  read_counter = my_read.read_count;
  read_counter_f = (float) read_counter;
  if (read_counter > 1){
    measurement_sd_all_flows = my_read.measurements_sd;
  }
  for (unsigned int i_hyp = 1; i_hyp < same_as_null_hypothesis.size(); ++i_hyp){
    at_least_one_same_as_null += same_as_null_hypothesis[i_hyp];
  }
}

void CrossHypotheses::InitializeTestFlows() {
  // Compute test flows for all hypotheses: flows changing by more than 0.1, 10 flows allowed
  success = ComputeAllComparisonsTestFlow(min_delta_for_flow, max_flows_to_test);
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

void CrossHypotheses::InitializeDerivedQualities() {

  InitializeResponsibility(); // depends on hypotheses
  // in theory don't need to compute any but test flows
  SetModPredictions();  // make sure that mod-predictions=predictions
  ComputeResiduals(); // predicted and measured

  InitializeSigma(); // depends on predicted

  my_t.SetV(heavy_tailed);
  // 2*heavy_tailed - 1 = Dof of t-dist
  sigma_factor = adjust_sigma? sqrt((2.0f * heavy_tailed - 3.0f) / (2.0f * heavy_tailed - 1.0f)) : 1.0f;

  ComputeBasicLogLikelihoods(); // depends on residuals and sigma
  // compute log-likelihoods
  ComputeLogLikelihoods();  // depends on test flow(s)
}

void CrossHypotheses::InitializeResponsibility() {
  responsibility[0] = 1.0f;  // everyone is an outlier until we trust you
  weighted_responsibility[0] = read_counter_f;
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
    weighted_responsibility[i_hyp] = responsibility[i_hyp] * read_counter_f;
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
  return(log(ll_denom)+ll_scale);  // log-likelihood under current distribution, including common value of log-likelihood-scale
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

void CrossHypotheses::ComputeResiduals() {
  for (unsigned int i_hyp = 0; i_hyp < mod_predictions.size(); i_hyp++) {
    for (unsigned int t_flow = 0; t_flow < test_flow.size(); t_flow++) {
      residuals[i_hyp][t_flow] = mod_predictions[i_hyp][t_flow] - normalized[t_flow];
    }
  }
}

void CrossHypotheses::ResetModPredictions() {
  // basic residuals are obviously predicted - normalized under each hypothesis
  for (unsigned int i_hyp = 0; i_hyp < mod_predictions.size(); i_hyp++) {
    for (unsigned int t_flow=0; t_flow < test_flow.size(); t_flow++) {
      mod_predictions[i_hyp][t_flow] = predictions[i_hyp][t_flow];
    }
  }
}

void CrossHypotheses::ResetRelevantResiduals() {
  ResetModPredictions();
  ComputeResiduals();
}

void CrossHypotheses::ComputeBasicLogLikelihoods() {
	// Non consensus case
	if (read_counter == 1){
		for (unsigned int i_hyp=0; i_hyp < basic_log_likelihoods.size(); i_hyp++) {
			for (unsigned int t_flow=0; t_flow<test_flow.size(); t_flow++) {
				float my_sigma = adjust_sigma? sigma_estimate[i_hyp][t_flow] * sigma_factor : sigma_estimate[i_hyp][t_flow];
				basic_log_likelihoods[i_hyp][t_flow] = my_t.LogTDistOddN(residuals[i_hyp][t_flow], my_sigma, skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
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
				float my_sigma = adjust_sigma? sigma_estimate[i_hyp][t_flow] * sigma_factor : sigma_estimate[i_hyp][t_flow];
				basic_log_likelihoods[i_hyp][t_flow] = my_t.LogTDistOddN(adj_res, my_sigma, skew_estimate);  // pure observational likelihood depends on residual + current estimated sigma under each hypothesis
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
      log_likelihood[i_hyp] *= read_counter_f;
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
    if (adjust_sigma){
    	sigma_projection *= sigma_factor;
    }
    float b_likelihood = my_t.LogTDistOddN(res_projection,sigma_projection,skew_estimate);

    log_likelihood[i_hyp] = (read_counter == 1) ? b_likelihood : read_counter_f * b_likelihood;
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
      sigma_estimate[i_hyp][t_flow] = magic_sigma_slope * square_level + magic_sigma_base;
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
  bool is_valid = (flow<(unsigned int)max_last_flow) and (flow<predictions_all_flows[0].size()) and (test_flow.size()<max_choice);
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
    for (unsigned int i_hyp=0; i_hyp<predictions_all_flows.size(); i_hyp++){
      for (unsigned int j_hyp=i_hyp+1; j_hyp<predictions_all_flows.size(); j_hyp++){
        float t_delta = fabs(predictions_all_flows[i_hyp][j_flow]-predictions_all_flows[j_hyp][j_flow]);
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
  if (test_flow.size() < 1) {
    test_flow.push_back(bestlocus); // always at least one difference if nothing made threshold
    tmp_success = false; // but don't want it?
  }

  return(tmp_success);
}


float CrossHypotheses::ComputeLLDifference(int a_hyp, int b_hyp) {
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
	my_family_cross_.log_likelihood[0] = -999999.9f;
	// accumulate the log-likelihood from the reads of the family for not null hypotheses
	for (unsigned int i_hyp = 1 ; i_hyp < my_family_cross_.log_likelihood.size(); i_hyp++) {
		my_family_cross_.log_likelihood[i_hyp] = 0.0f;
		for (unsigned int i_member = 0; i_member < valid_family_members.size(); ++i_member){
			unsigned int i_read = valid_family_members[i_member];
			my_family_cross_.log_likelihood[i_hyp] += ((1.0f - my_hypotheses[i_read].responsibility[0]) * my_hypotheses[i_read].log_likelihood[i_hyp]);
		}
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

int EvalFamily::MostResponsible(){
	return my_family_cross_.MostResponsible();
}

int EvalFamily::CountFamSizeFromAll()
{
	fam_size_ = 0;
	for (vector<unsigned int>::iterator read_it = all_family_members.begin(); read_it != all_family_members.end(); ++read_it){
		fam_size_ += (read_stack_->at(*read_it)->read_count);
	}
	return fam_size_;
}

int EvalFamily::CountFamSizeFromValid()
{
	valid_fam_size_ = 0;
	for (vector<unsigned int>::iterator read_it = valid_family_members.begin(); read_it != valid_family_members.end(); ++read_it){
		valid_fam_size_ += (read_stack_->at(*read_it)->read_count);
	}
	return valid_fam_size_;
}

// I define Family outlier responsibility = P(# non-outlier read members < min_fam_size)
// In theory, Resp(OL family) is obtained from the cdf of a multinomial distribution.
// Since Resp(OL family) >= min Resp(OL read), I use the lower bound to approximate Resp(OL family) for some extreme cases.
// Otherwise, I calculate Resp(OL family) via Monte-Carlo simulation.
// (Note 1): my_cross_.responsibility is not derived from my_cross_.log_likelihood in this function.
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
		weighted_avg_read_ol_resp += (my_hypotheses[*read_idx_it].responsibility[0] * my_hypotheses[*read_idx_it].read_counter_f);
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
	assert(sum_of_resp > 0.9999f and sum_of_resp < 1.0001f);
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
		if ((int) instance_of_read_by_state[i_hyp].size() < min_instance_of_read_by_state_len){
			min_instance_of_read_by_state_len = (int) instance_of_read_by_state[i_hyp].size();
		}
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
bool CrossHypotheses::OutlierByFlowDisruptiveness() const {
	if (not success or local_flow_disruptiveness_matrix.empty()){
		return true;
	}

	for (unsigned int i_hyp = 1; i_hyp < local_flow_disruptiveness_matrix[0].size(); ++i_hyp){
		if (local_flow_disruptiveness_matrix[0][i_hyp] == 0 or local_flow_disruptiveness_matrix[0][i_hyp] == 1){
			return false;
		}
	}
	// indefinite or flow-disruptive means outlier
	return true;
}

void EvalFamily::FillInFlowDisruptivenessMatrix(const vector<CrossHypotheses> &my_hypotheses)
{
	unsigned int num_hyp = 0;
	for(unsigned int i_member = 0; i_member < all_family_members.size(); ++i_member){
		if (my_hypotheses[all_family_members[i_member]].success){
			num_hyp = my_hypotheses[all_family_members[i_member]].instance_of_read_by_state.size();
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
			for (unsigned int i_member = 0; i_member < all_family_members.size(); ++i_member){
				// Don't count outliers.
				const CrossHypotheses* my_member = &(my_hypotheses[all_family_members[i_member]]);
				if (my_member->OutlierByFlowDisruptiveness()){
					continue;
				}
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


