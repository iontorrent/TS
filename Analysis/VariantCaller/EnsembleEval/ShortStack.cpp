/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ShortStack.h"



void ShortStack::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params){

	for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
      // basic likelihoods
    my_hypotheses[i_read].heavy_tailed = my_params.heavy_tailed;
    my_hypotheses[i_read].adjust_sigma =  my_params.adjust_sigma;
    
    // test flow construction
    my_hypotheses[i_read].max_flows_to_test = my_params.max_flows_to_test;
    my_hypotheses[i_read].min_delta_for_flow = my_params.min_delta_for_flow;
    
    // used to initialize sigma-estimates
    my_hypotheses[i_read].magic_sigma_base = my_params.magic_sigma_base;
    my_hypotheses[i_read].magic_sigma_slope = my_params.magic_sigma_slope;
  }
}


// fill in predictions for each hypothesis and initialze test flows
void ShortStack::FillInPredictionsAndTestFlows(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack,
    const InputStructures &global_context)
{
  //ion::FlowOrder flow_order(my_data.flow_order, my_data.flow_order.length());
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    my_hypotheses[i_read].FillInPrediction(thread_objects, *read_stack[i_read], global_context);
    my_hypotheses[i_read].start_flow = read_stack[i_read]->start_flow;
    my_hypotheses[i_read].InitializeTestFlows();
    if (not preserve_full_data){
      my_hypotheses[i_read].ClearAllFlowsData();
    }
  }
}

void ShortStack::ResetQualities(float outlier_probability) {
  // ! does not reset test flows or delta (correctly)
    for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
        my_hypotheses[i_read].InitializeDerivedQualities();
    }

    // reset the derived qualities of my_families
    if(is_molecular_tag_){
    	float uniform_f = (num_hyp_not_null == 0)? 1.0f: 1.0f / (float) num_hyp_not_null;
    	vector<float> init_hyp_freq(num_hyp_not_null, uniform_f);
    	// I calculate read responsibility is just for obtaining
    	UpdateReadResponsibility_(init_hyp_freq, outlier_probability);
    	ResetQualitiesForFamilies();
    }
}

void ShortStack::ResetQualitiesForFamilies(){
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		// don't count the not functional families
		if(my_eval_families[i_fam].GetFuncFromValid()){
			my_eval_families[i_fam].InitializeFamilyResponsibility();
			// Note that I must calculate read responsibility first because now the read log-likeligood propagated to the family is weighted by Resp(read is not an outlier)
			my_eval_families[i_fam].ComputeFamilyLogLikelihoods(my_hypotheses);
		}
	}
}

// Call PosteriorFrequencyLogLikelihoodFromFamilies_ if is_molecular_tag_
// Call PosteriorFrequencyLogLikelihoodFromReads_ if not is_molecular_tag_
float ShortStack::PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key) {
	return (this->*ptrPosteriorFrequencyLogLikelihood_)(hyp_freq, prior_frequency_weight, prior_log_normalization, outlier_prob, strand_key);
}

float ShortStack::PosteriorFrequencyLogLikelihoodFromReads_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key) {
  //cout << "eval at freq " << my_freq << endl;

  double my_LL = 0.0; // use double to calculate cumsum to enjance numerical stability
  float my_reliability = 1.0f - outlier_prob;
  //for (unsigned int i_read=0; i_read<my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
      my_LL += (double) my_hypotheses[i_read].ComputePosteriorLikelihood(hyp_freq, outlier_prob); // XXX
  }
  // add contribution from prior
  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    // contribution is
    float local_LL = log(hyp_freq[i_hyp]*my_reliability + outlier_prob); // hyp_freq might be exactly zero, whereupon my prior is an outlier
    my_LL += (double) (prior_frequency_weight[i_hyp]*local_LL); // my weight is my number of pseudo-observations with this LL
  }
  my_LL += (double) prior_log_normalization;
  return (float) my_LL;
}


void ShortStack::UpdateRelevantLikelihoods() {
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateRelevantLikelihoods();
  }
}
void ShortStack::ResetRelevantResiduals() {
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].ResetRelevantResiduals();
  }
}


void ShortStack::ResetNullBias() {
  ResetRelevantResiduals();
  UpdateRelevantLikelihoods();
}

void ShortStack::FindValidIndexes() {
  valid_indexes.resize(0);
  // only loop over reads where variant construction worked correctly
  for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
    if (my_hypotheses[i_read].success) {
      valid_indexes.push_back(i_read);
    }
  }
}


void ShortStack::UpdateReadResponsibility_(const vector<float> &hyp_freq, float outlier_prob) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateResponsibility(hyp_freq, outlier_prob);
  }
}

// Call UpdateFamilyAndReadResponsibility_ if is_molecular_tag_
// Call UpdateReadResponsibility_ if not is_molecular_tag_
void ShortStack::UpdateResponsibility(const vector<float> &hyp_freq, float outlier_prob) {
	(this->*ptrUpdateResponsibility_)(hyp_freq, outlier_prob);
}


void ShortStack::MultiFrequencyFromReadResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
  hyp_freq.assign(hyp_freq.size(), 0.0f);

  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); i_hyp++) {
      if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
        hyp_freq[i_hyp] += my_hypotheses[i_read].weighted_responsibility[i_hyp+1];
    }
  }
  // add prior weight to count of cluster
  // weight = effective number of additional pseudo-observations
  // @TODO: Does the prior weight work? Should it be added in log-domain?
  // (Currently we set germline_prior_strength = 0 by default so it doesn't matter.)
  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    hyp_freq[i_hyp] += prior_frequency_weight[i_hyp];
  }
  // safety factor to prevent zero-frequency from preventing progress
  float safety_offset = 0.5f;
  float denom = 0.0f;
  for (unsigned int i_hyp=0; i_hyp< hyp_freq.size(); i_hyp++){
    denom += hyp_freq[i_hyp]+safety_offset;
  }

  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    hyp_freq[i_hyp] = (hyp_freq[i_hyp]+safety_offset)/denom;
  }
}

// Call MultiFrequencyFromFamilyResponsibility_ if is_molecular_tag_
// Call MultiFrequencyFromReadResponsibility_ if not is_molecular_tag_
void ShortStack::MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
	 (this->*ptrMultiFrequencyFromResponsibility_)(hyp_freq, prior_frequency_weight, strand_key);
}

// Update Family Responsibility if the Log Likelihoods of reads are updated.
void ShortStack::UpdateFamilyResponsibility_(const vector<float> &hyp_freq, float outlier_prob) {
	for (vector<EvalFamily>::iterator fam_it = my_eval_families.begin(); fam_it != my_eval_families.end(); ++fam_it){
		if(fam_it->GetFuncFromValid()){
			fam_it->ComputeFamilyLogLikelihoods(my_hypotheses);
			fam_it->UpdateFamilyResponsibility(hyp_freq, outlier_prob);
			// Note that the outlier responsibility was set to (close to) zero in ComputeFamilyLogLikelihoods and UpdateFamilyResponsibility.
			// I am calculating the family outlier responsibility here.
			fam_it->ComputeFamilyOutlierResponsibility(my_hypotheses, effective_min_family_size);
		}
	}
}


// Let the family_responsibility be loc_freq
// UpdateFamilyResponsibility_() must be done first
void ShortStack::UpdateReadResponsibilityFromFamily_(unsigned int num_hyp_no_null, float outlier_prob){
	float safety_zero = 0.00001f; // The close-to-zero outlier prob for the read which is clearly not an outlier.
	vector<float> loc_freq(num_hyp_no_null, 0.0f);
	// Basically, I let the family responsibility be the prior for the read.
	// However, I add a freq offset to the family responsibility to prevent the family bullies a read that has divergent opinion.
	// Otherwise the read will be treated as an outlier and won't came back, which may also let the family become an outlier.
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		// don't count the non-functional families
		if(not my_eval_families[i_fam].GetFuncFromValid()){
			continue;
		}
		float freq_offset = 0.0f;
		float freq_offset_gap = 0.01f; // I shall set the parameter outlier_probability << freq_offset_gap.
		// Let loc_freq be {family_responsibility[1], family_responsibility[2], ...} with freq_offset
		float d_normalize = 0.0f;
		float max_loc_freq = 0.0f;
		for (unsigned int i_hyp = 0; i_hyp < num_hyp_no_null; ++i_hyp){
			loc_freq[i_hyp] = my_eval_families[i_fam].family_responsibility[i_hyp+1];
			d_normalize += loc_freq[i_hyp];
		}
		for (unsigned int i_hyp=0; i_hyp<loc_freq.size(); i_hyp++){
			loc_freq[i_hyp] /= d_normalize;
			max_loc_freq = max(max_loc_freq, loc_freq[i_hyp]);
		}

		// Here is the offset. Basically, I am letting (1 - outlier_probability) loc_freq[i_hyp] > outlier_probability.
		freq_offset = (max_loc_freq > outlier_prob) ? outlier_prob + freq_offset_gap * (max_loc_freq - outlier_prob) : outlier_prob;
		d_normalize = 1.0f + (freq_offset * (float) num_hyp_no_null);
		for (unsigned int i_hyp = 0; i_hyp < num_hyp_no_null; ++i_hyp){
			loc_freq[i_hyp] += freq_offset;
			loc_freq[i_hyp] /= d_normalize;
		}
		for (unsigned int i_ndx = 0; i_ndx < my_eval_families[i_fam].valid_family_members.size(); ++i_ndx){
			int i_read = my_eval_families[i_fam].valid_family_members[i_ndx];
			float my_outlier_prob = my_hypotheses[i_read].at_least_one_same_as_null ? safety_zero: outlier_prob;
			my_hypotheses[i_read].UpdateResponsibility(loc_freq, my_outlier_prob);
		}
	}
}

// pretty much the same as void ShortStack::MultiFrequencyFromResponsibility(...) except we use the family responsibility
void ShortStack::MultiFrequencyFromFamilyResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
	hyp_freq.assign(hyp_freq.size(), 0.0f);

	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++ i_fam){
		if(strand_key >= 0 and my_eval_families[i_fam].strand_key != strand_key){
			continue;
		}

		// don't count the not functional families
		if(not my_eval_families[i_fam].GetFuncFromValid()){
			continue;
		}
		for (unsigned int i_hyp=0; i_hyp < hyp_freq.size(); ++i_hyp){
			hyp_freq[i_hyp] += my_eval_families[i_fam].family_responsibility[i_hyp + 1];
		}
	}

	float denom = 0.0f;
	for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); ++i_hyp){
	    hyp_freq[i_hyp] += prior_frequency_weight[i_hyp];
	    denom += hyp_freq[i_hyp];
	  }

	// In case hyp_freq[1], hyp_freq[2],... are all zeros, although it shouldn't happen.
	if(denom <= 0.0f){
		hyp_freq.assign(hyp_freq.size(), 1.0f / (float) hyp_freq.size());
		return;
	}

	// I add safety zero which is slightly different from MultiFrequencyFromReadResponsibility_ to avoid the perturbation error of the very low frequency allele
	float safety_zero = 0.001f / (float) num_func_families_;
	float normalization_factor = 0.0f;

	for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); ++i_hyp){
		hyp_freq[i_hyp] = hyp_freq[i_hyp] / denom;
		if(hyp_freq[i_hyp] < safety_zero)
			hyp_freq[i_hyp] = safety_zero;
		normalization_factor += hyp_freq[i_hyp];
	 }

	for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); ++i_hyp){
		hyp_freq[i_hyp] = hyp_freq[i_hyp] / normalization_factor;
	 }
}

// Compute the log-posterior of frequency from families
float ShortStack::PosteriorFrequencyLogLikelihoodFromFamilies_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float outlier_prob, int strand_key) {
	double my_LL = 0.0; // use double to calculate cumsum to enjance numerical stability
	float my_reliability = 1.0f - outlier_prob;
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		if(strand_key >= 0 and my_eval_families[i_fam].strand_key != strand_key){
			continue;
		}

		// don't count the non-functional families
		if(!(my_eval_families[i_fam].GetFuncFromValid())){
			continue;
		}
		my_LL += (double) my_eval_families[i_fam].ComputeFamilyPosteriorLikelihood(hyp_freq);
	}

    // add contribution from prior
    for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
        // contribution is
        float local_LL = log(hyp_freq[i_hyp]*my_reliability + outlier_prob); // hyp_freq might be exactly zero, whereupon my prior is an outlier
        my_LL += (double) (prior_frequency_weight[i_hyp]*local_LL); // my weight is my number of pseudo-observations with this LL
    }
    my_LL += (double) prior_log_normalization;
    return (float) my_LL;
}

// A belief propagation approach
//@TODO: The outlier handling can be further improved.
void ShortStack::UpdateFamilyAndReadResponsibility_(const vector<float> &hyp_freq, float outlier_prob){
	// 1. update family responsibility
    UpdateFamilyResponsibility_(hyp_freq, outlier_prob);
    // 2. update read responsibility from family
    UpdateReadResponsibilityFromFamily_(hyp_freq.size(), outlier_prob);
}

void ShortStack::SwitchMolTagsPtr_(void){
	if(is_molecular_tag_){
		// I do inference for the allele frequency from families
		ptrPosteriorFrequencyLogLikelihood_ = &ShortStack::PosteriorFrequencyLogLikelihoodFromFamilies_;
		ptrUpdateResponsibility_ = &ShortStack::UpdateFamilyAndReadResponsibility_;
		ptrMultiFrequencyFromResponsibility_ = &ShortStack::MultiFrequencyFromFamilyResponsibility_;
	}
	else{
		// I do inference for the allele frequency from reads
		ptrPosteriorFrequencyLogLikelihood_ = &ShortStack::PosteriorFrequencyLogLikelihoodFromReads_;
		ptrUpdateResponsibility_ = &ShortStack::UpdateReadResponsibility_;
		ptrMultiFrequencyFromResponsibility_ = &ShortStack::MultiFrequencyFromReadResponsibility_;
	}
}

void ShortStack::SetIsMolecularTag(bool is_mol_tag){
	is_molecular_tag_ = is_mol_tag;
	// Switch the function pointers when we are making change on is_molecular_tag_.
	SwitchMolTagsPtr_();
}

unsigned int ShortStack::DetailLevel(void){
	if (is_molecular_tag_)
		// With mol tagging, we often call variants using just a few (<5) variant families.
		// If I serve detail_level = num_func_families_, then the singularity of log-posterior at f = 0 may cause big interpolation error.
		// Thus I let detail_level finer than num_func_families_to obtain better numerical accuracy.
		// Since the number of scanned points doesn't grow a lot in fast scan, the complexity is manageable.
		// (Note): Setting detail_level to be too large (e.g. num_valid_read_counts_) may cause floating point error on frequency and cause problem in Fibonacci search dueing fast scan.
		return num_func_families_ * 2; // This should be enough.

	return my_hypotheses.size();
}

// Step 1): Initialize all families
// Step 2): Set the functionality of the families in my_families.
// Step 3): Update valid_index
void ShortStack::InitializeMyEvalFamilies(unsigned int num_hyp){
	num_func_families_ = 0;
	for (vector<EvalFamily>::iterator fam_it = my_eval_families.begin(); fam_it != my_eval_families.end(); ++fam_it){
		fam_it->InitializeEvalFamily(num_hyp);
		fam_it->ResetValidFamilyMembers();
		for (vector<unsigned int>::iterator read_it = fam_it->all_family_members.begin(); read_it != fam_it->all_family_members.end(); ++read_it){
			if (my_hypotheses[*read_it].success){
				// valid_family_members contains only the reads that are successfully initialized
				fam_it->valid_family_members.push_back(*read_it);
			}
		}
		fam_it->CountFamSizeFromValid();
		if (fam_it->SetFuncFromValid(effective_min_family_size)){
			++num_func_families_;
			num_valid_read_counts_ += fam_it->GetValidFamSize();
		}
		else{
			// disable the reads in a non-functional family
			for(vector<unsigned int>::iterator read_it = fam_it->valid_family_members.begin(); read_it !=fam_it->valid_family_members.end(); ++read_it){
				my_hypotheses[*read_it].success = false;
			}
		}
	}

	// Some of the reads may be set to not success. Need to update valid_index.
	FindValidIndexes();
}

int ShortStack::OutlierCountsByFlowDisruptiveness(){
	int ol_counts = 0;
	for (unsigned int i_read = 0;  i_read < my_hypotheses.size(); ++i_read){
		ol_counts += my_hypotheses[i_read].OutlierByFlowDisruptiveness();
	}
	if (DEBUG > 0){
		cout << endl << "+ Counting outlier reads using flow-disruptiveness:" <<endl
			 << "  - Read stack size = " << my_hypotheses.size() << endl
			 << "  - Number of outlier reads = " << ol_counts << endl;
	}
	return ol_counts;
}

void ShortStack::FlowDisruptiveOutlierFiltering(bool update_valid_index){
	for (vector<CrossHypotheses>::iterator read_it = my_hypotheses.begin(); read_it != my_hypotheses.end(); ++read_it){
		if (read_it->success){
			read_it->success = not (read_it->OutlierByFlowDisruptiveness());
		}
	}
	if (update_valid_index){
		FindValidIndexes();
	}
}
