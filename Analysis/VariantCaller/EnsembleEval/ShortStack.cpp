/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ShortStack.h"



void ShortStack::PropagateTuningParameters(EnsembleEvalTuningParameters &my_params){
	// Set is_molecular_tag_

    for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
      // basic likelihoods
    my_hypotheses[i_read].heavy_tailed = my_params.heavy_tailed;
    
    // test flow construction
    my_hypotheses[i_read].max_flows_to_test = my_params.max_flows_to_test;
    my_hypotheses[i_read].min_delta_for_flow = my_params.min_delta_for_flow;
    
    // used to initialize sigma-estimates
    my_hypotheses[i_read].magic_sigma_base = my_params.magic_sigma_base;
    my_hypotheses[i_read].magic_sigma_slope = my_params.magic_sigma_slope;

    // preserve the data for all flows?
    my_hypotheses[i_read].preserve_full_data = my_params.preserve_full_data;
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
  }
}

void ShortStack::ResetQualities() {
  // ! does not reset test flows or delta (correctly)
    for (unsigned int i_read = 0; i_read < my_hypotheses.size(); i_read++) {
        my_hypotheses[i_read].InitializeDerivedQualities();
    }

    // reset the derived qualities of my_families
    if(is_molecular_tag_){
    	ResetQualitiesForFamilies();
    }
}


void ShortStack::ResetQualitiesForFamilies(){
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		// don't count the not functional families
		if(my_eval_families[i_fam].GetFunctionality()){
			my_eval_families[i_fam].InitializeFamilyResponsibility();
			my_eval_families[i_fam].ComputeFamilyLogLikelihoods(my_hypotheses);
		}
	}
}

// Call PosteriorFrequencyLogLikelihoodMolTag if is_molecular_tag_ == true
// Call PosteriorFrequencyLogLikelihoodNonMolTag if is_molecular_tag_ == false
float ShortStack::PosteriorFrequencyLogLikelihood(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key) {
	return (this->*ptrPosteriorFrequencyLogLikelihood_)(hyp_freq, prior_frequency_weight, prior_log_normalization, my_reliability, strand_key);
}

float ShortStack::PosteriorFrequencyLogLikelihoodNoMolTags(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key) {
  //cout << "eval at freq " << my_freq << endl;

  float my_LL = 0.0f;

  //for (unsigned int i_read=0; i_read<my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
      my_LL += my_hypotheses[i_read].ComputePosteriorLikelihood(hyp_freq, my_reliability); // XXX
  }
  // add contribution from prior
  for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
    // contribution is
    float local_LL = log(hyp_freq[i_hyp]*my_reliability + (1.0f-my_reliability)); // hyp_freq might be exactly zero, whereupon my prior is an outlier
    my_LL += prior_frequency_weight[i_hyp]*local_LL; // my weight is my number of pseudo-observations with this LL
  }
  my_LL += prior_log_normalization;
  return(my_LL);
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


void ShortStack::UpdateResponsibilityNoMolTags(const vector<float> &hyp_freq, float data_reliability) {
  //for (unsigned int i_read=0; i_read<total_theory.my_hypotheses.size(); i_read++){
  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    my_hypotheses[i_read].UpdateResponsibility(hyp_freq, data_reliability);
  }
}

// Call UpdateResponsibilityMolTag if is_molecular_tag_ == true
// Call UpdateResponsibilityNonMolTag if is_molecular_tag_ == false
void ShortStack::UpdateResponsibility(const vector<float> &hyp_freq, float data_reliability) {
	(this->*ptrUpdateResponsibility_)(hyp_freq, data_reliability);
}


void ShortStack::MultiFrequencyFromResponsibilityNoMolTags(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
  hyp_freq.assign(hyp_freq.size(), 0.0f);

  for (unsigned int i_ndx = 0; i_ndx < valid_indexes.size(); i_ndx++) {
    unsigned int i_read = valid_indexes[i_ndx];
    for (unsigned int i_hyp = 0; i_hyp < hyp_freq.size(); i_hyp++) {
      if ((strand_key < 0) || (my_hypotheses[i_read].strand_key == strand_key))
        hyp_freq[i_hyp] += my_hypotheses[i_read].responsibility[i_hyp+1];
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

// Call MultiFrequencyFromResponsibilityMolTag if is_molecular_tag_ == true
// Call MultiFrequencyFromResponsibilityNonMolTag if is_molecular_tag_ == false
void ShortStack::MultiFrequencyFromResponsibility(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
	 (this->*ptrMultiFrequencyFromResponsibility_)(hyp_freq, prior_frequency_weight, strand_key);
}

void ShortStack::OutlierFiltering(float data_reliability, bool is_update_valid_index){
	vector<float> zeros_vector;
    for(unsigned int i_read = 0; i_read < my_hypotheses.size(); ++i_read){
	  if(my_hypotheses[i_read].success){
		  zeros_vector.assign(my_hypotheses[i_read].responsibility.size(), 0.0f);
		  // Not yet initialize my_hypotheses.
		  if(my_hypotheses[i_read].responsibility == zeros_vector){
			  my_hypotheses[i_read].InitializeDerivedQualities();
		  }
		  my_hypotheses[i_read].success = (!my_hypotheses[i_read].LocalOutlierClassifier(data_reliability));
	  }
    }
    if(is_update_valid_index){
        FindValidIndexes();
    }
}

// family stuff
// Update Family Responsibility if the Log Likelihoods of reads are updated.
void ShortStack::UpdateFamilyResponsibility_(const vector<float> &hyp_freq, float data_reliability) {
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		if(my_eval_families[i_fam].GetFunctionality()){
			my_eval_families[i_fam].ComputeFamilyLogLikelihoods(my_hypotheses);
			my_eval_families[i_fam].UpdateFamilyResponsibility(hyp_freq, data_reliability);
		}
	}
}


// family stuff
// Let the family_responsibility be loc_freq
// UpdateFamilyResponsibility_() must be done first
void ShortStack::UpdateReadResponsibilityFromFamily_(unsigned int num_hyp_no_null, float data_reliability){
	vector<float> loc_freq;
	float safety_zero = 0.00001f;

	loc_freq.assign(num_hyp_no_null, 0.0f);
	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		// don't count the non-functional families
		if(not my_eval_families[i_fam].GetFunctionality()){
			continue;
		}
		// Let loc_freq be {family_responsibility[1], family_responsibility[2], ...} with a safety zero
		float d_normalize = 0.0f;
		for (unsigned int i_hyp=0; i_hyp<loc_freq.size(); i_hyp++){
			loc_freq[i_hyp] = my_eval_families[i_fam].family_responsibility[i_hyp+1] + safety_zero;
			d_normalize += loc_freq[i_hyp];
		}
		for (unsigned int i_hyp=0; i_hyp<loc_freq.size(); i_hyp++){
			loc_freq[i_hyp] /= d_normalize;
		}
		for(unsigned int i_ndx = 0; i_ndx <  my_eval_families[i_fam].family_members.size(); ++i_ndx){
			int i_read = my_eval_families[i_fam].family_members[i_ndx];
			my_hypotheses[i_read].UpdateResponsibility(loc_freq, data_reliability);
		}
	}
}

// family stuff
// pretty much the same as void ShortStack::MultiFrequencyFromResponsibility(...) except we use the family responsibility
void ShortStack::MultiFrequencyFromFamilyResponsibility_(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
	hyp_freq.assign(hyp_freq.size(), 0.0f);


	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++ i_fam){
		if(strand_key >= 0 and my_eval_families[i_fam].strand_key != strand_key){
			continue;
		}

		// don't count the not functional families
		if(!(my_eval_families[i_fam].GetFunctionality())){
			continue;
		}
		for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); ++i_hyp){
			hyp_freq[i_hyp] += my_eval_families[i_fam].family_responsibility[i_hyp+1];
		}
	}

	// safety factor to prevent zero-frequency from preventing progress
	float safety_offset = 0.5f;
	float denom = 0.0f;
	for (unsigned int i_hyp=0; i_hyp< hyp_freq.size(); ++i_hyp){
	    hyp_freq[i_hyp] += prior_frequency_weight[i_hyp];
	    denom += hyp_freq[i_hyp]+safety_offset;
	  }

	for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); ++i_hyp){
		hyp_freq[i_hyp] = (hyp_freq[i_hyp] + safety_offset) / denom;
	 }
}

// family stuff
// Compute the log-posterior of frequency from families
float ShortStack::PosteriorFrequencyFamilyLogLikelihood_(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key) {
	float my_LL = 0.0f;

	for(unsigned int i_fam = 0; i_fam < my_eval_families.size(); ++i_fam){
		if(strand_key >= 0 and my_eval_families[i_fam].strand_key != strand_key){
			continue;
		}

		// don't count the non-functional families
		if(!(my_eval_families[i_fam].GetFunctionality())){
			continue;
		}

		my_LL += my_eval_families[i_fam].ComputeFamilyPosteriorLikelihood(hyp_freq, my_reliability);
	}

    // add contribution from prior
    for (unsigned int i_hyp=0; i_hyp<hyp_freq.size(); i_hyp++){
        // contribution is
        float local_LL = log(hyp_freq[i_hyp]*my_reliability + (1.0f-my_reliability)); // hyp_freq might be exactly zero, whereupon my prior is an outlier
        my_LL += prior_frequency_weight[i_hyp]*local_LL; // my weight is my number of pseudo-observations with this LL
    }
    my_LL += prior_log_normalization;
    return(my_LL);
}

float ShortStack::PosteriorFrequencyLogLikelihoodMolTags(const vector<float> &hyp_freq, const vector<float> &prior_frequency_weight, float prior_log_normalization, float my_reliability, int strand_key){
	 // the posterior likelihood of hyp_freq is the posterior likelihood of hyp_freq from families
	return PosteriorFrequencyFamilyLogLikelihood_(hyp_freq, prior_frequency_weight, prior_log_normalization, my_reliability, strand_key);
}

void ShortStack::UpdateResponsibilityMolTags(const vector<float> &hyp_freq, float data_reliability){
	// first update family responsibility
    UpdateFamilyResponsibility_(hyp_freq, data_reliability);
    // then update read responsibility from family
    UpdateReadResponsibilityFromFamily_(hyp_freq.size(), data_reliability);
}

void ShortStack::MultiFrequencyFromResponsibilityMolTags(vector<float> &hyp_freq, vector<float> &prior_frequency_weight, int strand_key){
	// Update hyp_freq using family responsibility
	MultiFrequencyFromFamilyResponsibility_(hyp_freq, prior_frequency_weight, strand_key);
}

void ShortStack::SwitchMolTagsPtr_(void){
	if(is_molecular_tag_){
		ptrPosteriorFrequencyLogLikelihood_ = &ShortStack::PosteriorFrequencyLogLikelihoodMolTags;
		ptrUpdateResponsibility_ = &ShortStack::UpdateResponsibilityMolTags;
		ptrMultiFrequencyFromResponsibility_ = &ShortStack::MultiFrequencyFromResponsibilityMolTags;
	}
	else{
		ptrPosteriorFrequencyLogLikelihood_ = &ShortStack::PosteriorFrequencyLogLikelihoodNoMolTags;
		ptrUpdateResponsibility_ = &ShortStack::UpdateResponsibilityNoMolTags;
		ptrMultiFrequencyFromResponsibility_ = &ShortStack::MultiFrequencyFromResponsibilityNoMolTags;
	}
}

void ShortStack::SetIsMolecularTag(bool is_mol_tag){
	is_molecular_tag_ = is_mol_tag;
	// Switch the function pointers when we are making change on is_molecular_tag_.
	SwitchMolTagsPtr_();
}


unsigned int ShortStack::DetailLevel(void){
	if(is_molecular_tag_)
		return num_func_families_;

	return my_hypotheses.size();
}

// Set the functionality for all families in my_families.
void ShortStack::SetFuncionalityForFamilies(unsigned int min_fam_size){
	num_func_families_ = 0;
	for(vector<EvalFamily>::iterator it_fam = my_eval_families.begin();
			it_fam != my_eval_families.end(); ++it_fam){
		vector <unsigned int> fam_members_temp(0);
		fam_members_temp.reserve(it_fam->family_members.size());
		for(vector<unsigned int>::iterator it_read = it_fam->family_members.begin();
				it_read !=it_fam->family_members.end(); ++it_read){
			if(my_hypotheses[*it_read].success){
				fam_members_temp.push_back(*it_read);
			}
		}
		it_fam->family_members.swap(fam_members_temp);
		if(not it_fam->SetFunctionality(min_fam_size)){
			// disable the reads in a non-functional family
			for(vector<unsigned int>::iterator it_read = it_fam->family_members.begin();
					it_read !=it_fam->family_members.end(); ++it_read){
				my_hypotheses[*it_read].success = false;
			}
		}else{
			++num_func_families_;
		}
	}
	FindValidIndexes();
}
