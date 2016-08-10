/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HandleVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "HandleVariant.h"
#include "DecisionTreeData.h"


void EnsembleEval::SpliceAllelesIntoReads(PersistingThreadObjects &thread_objects, const InputStructures &global_context,
                                          const ExtendParameters &parameters, const ReferenceReader &ref_reader, int chr_idx)
{
  bool changed_alignment;
  unsigned int  num_valid_reads = 0;
  unsigned int  num_realigned = 0;
  int  num_hyp_no_null = allele_identity_vector.size()+1; // num alleles +1 for ref

  // generate null+ref+nr.alt hypotheses per read in the case of do_multiallele_eval
  allele_eval.total_theory.my_hypotheses.resize(read_stack.size());

  for (unsigned int i_read = 0; i_read < allele_eval.total_theory.my_hypotheses.size(); i_read++) {
    // --- New splicing function ---
    allele_eval.total_theory.my_hypotheses[i_read].success =
        SpliceVariantHypotheses(*read_stack[i_read],
                                *this,
                                seq_context,
                                thread_objects,
                                allele_eval.total_theory.my_hypotheses[i_read].splice_start_flow,
                                allele_eval.total_theory.my_hypotheses[i_read].splice_end_flow,
                                allele_eval.total_theory.my_hypotheses[i_read].instance_of_read_by_state,
                                allele_eval.total_theory.my_hypotheses[i_read].same_as_null_hypothesis,
                                changed_alignment,
                                global_context,
                                ref_reader, chr_idx);

    if (allele_eval.total_theory.my_hypotheses[i_read].success){
      num_valid_reads++;
      if (changed_alignment)
        num_realigned++;
    }

    // if we need to compare likelihoods across multiple possibilities
    if (num_hyp_no_null > 2)
      allele_eval.total_theory.my_hypotheses[i_read].use_correlated_likelihood = false;
  }

  // Check how many reads had their alignment modified
  std::ostringstream my_info;
  my_info.precision(4);
  if (doRealignment and num_valid_reads>0){
	float frac_realigned = (float)num_realigned / (float)num_valid_reads;
	// And re-do splicing without realignment if we exceed the threshold
	if (frac_realigned > parameters.my_controls.filter_variant.realignment_threshold){
      my_info << "SKIPREALIGNx" << frac_realigned;
      doRealignment = false;
      for (unsigned int i_read = 0; i_read < allele_eval.total_theory.my_hypotheses.size(); i_read++) {
          allele_eval.total_theory.my_hypotheses[i_read].success =
              SpliceVariantHypotheses(*read_stack[i_read],
                                      *this,
                                      seq_context,
                                      thread_objects,
                                      allele_eval.total_theory.my_hypotheses[i_read].splice_start_flow,
                                      allele_eval.total_theory.my_hypotheses[i_read].splice_end_flow,
                                      allele_eval.total_theory.my_hypotheses[i_read].instance_of_read_by_state,
                                      allele_eval.total_theory.my_hypotheses[i_read].same_as_null_hypothesis,
                                      changed_alignment,
                                      global_context,
                                      ref_reader, chr_idx);
      }
	}
	else {
      my_info << "REALIGNEDx" << frac_realigned;
	}
    info_fields.push_back(my_info.str());
  }

}




void SummarizeInfoFieldsFromEnsemble(EnsembleEval &my_ensemble, vcf::Variant &candidate_variant, int _cur_allele_index, const string &sample_name) {

  float mean_ll_delta;

  my_ensemble.ScanSupportingEvidence(mean_ll_delta, _cur_allele_index);

  candidate_variant.info["MLLD"].push_back(convertToString(mean_ll_delta));

  float radius_bias, fwd_bias, rev_bias, ref_bias,var_bias;
  int fwd_strand = 0;
  int rev_strand = 1;

  int var_hyp = 1;

  // get bias terms from cur_allele within a single multi-allele structure
  var_hyp = _cur_allele_index+1;
  radius_bias = my_ensemble.allele_eval.cur_state.bias_generator.RadiusOfBias(_cur_allele_index);
  fwd_bias = my_ensemble.allele_eval.cur_state.bias_generator.latent_bias[fwd_strand][_cur_allele_index];
  rev_bias = my_ensemble.allele_eval.cur_state.bias_generator.latent_bias[rev_strand][_cur_allele_index];
  //@TODO: note the disconnect in indexing; inconsistent betwen objects
  ref_bias = my_ensemble.allele_eval.cur_state.bias_checker.ref_bias_v[var_hyp];
  var_bias = my_ensemble.allele_eval.cur_state.bias_checker.variant_bias_v[var_hyp];

  candidate_variant.info["RBI"].push_back(convertToString(radius_bias));
  // this is by strand
  candidate_variant.info["FWDB"].push_back(convertToString(fwd_bias));
  candidate_variant.info["REVB"].push_back(convertToString(rev_bias));
  // this is by hypothesis
  candidate_variant.info["REFB"].push_back(convertToString(ref_bias));
  candidate_variant.info["VARB"].push_back(convertToString(var_bias));

}
// The structure InferenceResult is used for void MultiMinAlleleFreq(...) only.
// qual, gt, gq are the QUAL, GT, GQ computed for min_allele_freq, respectively
struct InferenceResult{
    float min_allele_freq;
    float qual;
    string gt;
    int gq;
    bool operator<(const InferenceResult &rhs) const { return min_allele_freq < rhs.min_allele_freq; } // use the operator "<" for "std::sort"
    bool operator==(const float &rhs) const { return min_allele_freq == rhs; } // use the operator "==" for "std::find"
};

void MultiMinAlleleFreq(EnsembleEval &my_ensemble, VariantCandidate &candidate_variant, int sample_index, ProgramControlSettings &program_flow, int max_detail_level){
    string sample_name = "";
    if(sample_index >= 0) {
        sample_name = candidate_variant.variant.sampleNames[sample_index];
    }

    // info tags needed for multi-min-allele-freq
    vector<string> tags_for_multi_min_allele_freq = {"MUAF", "MUQUAL", "MUGQ", "MUGT",
    		               "SMAF", "SMQUAL", "SMGQ", "SMGT",
						   "MMAF", "MMQUAL", "MMGQ", "MMGT",
						   "IMAF", "IMQUAL", "IMGQ", "IMGT",
						   "HMAF", "HMQUAL", "HMGQ", "HMGT"};
    // Add the info tags for multi-min-allele-freq if we have not added yet.
    for(unsigned int i_tag = 0; i_tag < tags_for_multi_min_allele_freq.size(); ++i_tag){
    	vector<string>::iterator it_format = find(candidate_variant.variant.format.begin(),
    			                                  candidate_variant.variant.format.end(),
    			                                  tags_for_multi_min_allele_freq[i_tag]);
        if(it_format == candidate_variant.variant.format.end()){
        	candidate_variant.variant.format.push_back(tags_for_multi_min_allele_freq[i_tag]);
        }
    }

    // inference_results_union stores the inference results for the union of the min-allele-freq of all available variant types
    vector<InferenceResult> inference_results_union;
    bool is_snp_done = false;
    bool is_mnp_done = false;
    bool is_hotspot_done = false;
    bool is_indel_done = false;

    for(unsigned int alt_allele_index = 0; alt_allele_index < my_ensemble.allele_identity_vector.size(); ++alt_allele_index){
        // ptr_maf_vec = the pointer to the multi_min_allele_freq vector of which type of variant for this allele
    	vector<float> *ptr_maf_vec = &(program_flow.snp_multi_min_allele_freq);
        string type_prefix = "S";

        // Note that no override here!
        if(my_ensemble.allele_identity_vector[alt_allele_index].status.isHotSpot){
        	if(is_hotspot_done){
        		continue;
        	}
        	ptr_maf_vec = &(program_flow.hotspot_multi_min_allele_freq);
        	type_prefix = "H";
        }
        else if(my_ensemble.allele_identity_vector[alt_allele_index].ActAsSNP()){
        	if(is_snp_done){
        		continue;
        	}
        	ptr_maf_vec = &(program_flow.snp_multi_min_allele_freq);
        	type_prefix = "S";
        }
        else if(my_ensemble.allele_identity_vector[alt_allele_index].ActAsMNP()){
        	if(is_mnp_done){
        		continue;
        	}
        	ptr_maf_vec = &(program_flow.mnp_multi_min_allele_freq);
        	type_prefix = "M";
        }
        else if(my_ensemble.allele_identity_vector[alt_allele_index].ActAsHPIndel()){
        	if(is_indel_done){
        		continue;
        	}
        	ptr_maf_vec = &(program_flow.indel_multi_min_allele_freq);
        	type_prefix = "I";
        }else{
        	if(is_snp_done){
        		continue;
        	}
        	ptr_maf_vec = &(program_flow.snp_multi_min_allele_freq);
        	string type_prefix = "S";
        }

        for(unsigned int i_freq = 0; i_freq < ptr_maf_vec->size(); ++i_freq){
            float loc_min_allele_freq = ptr_maf_vec->at(i_freq);
        	vector<InferenceResult>::iterator it = find(inference_results_union.begin(), inference_results_union.end(), loc_min_allele_freq);
            float loc_qual;
        	int loc_gq;
            string loc_gt;

            if(it == inference_results_union.end()){ // This the first time we get loc_min_allele_freq
                int genotype_call;
                float evaluated_genotype_quality;
                // Let's do the inference for the given loc_min_allele_freq
                my_ensemble.allele_eval.CallGermline(loc_min_allele_freq, genotype_call, evaluated_genotype_quality, loc_qual);

                vector<int> genotype_component = {my_ensemble.diploid_choice[0], my_ensemble.diploid_choice[1]}; // starts with het var

                if(genotype_call == 2){ //hom var
                    genotype_component[0] = my_ensemble.diploid_choice[1];
                }
                else if(genotype_call == 0){ //hom ref
                    genotype_component[1] = my_ensemble.diploid_choice[0];
                }

                loc_gt = convertToString(genotype_component[0]) + "/" + convertToString(genotype_component[1]);
                loc_gq = int(round(evaluated_genotype_quality)); // genotype quality is rounded as an integer.
                // append inference_results_union
                inference_results_union.push_back({loc_min_allele_freq, loc_qual, loc_gt, loc_gq});
            }
            else{ // We've seen loc_min_allele_freq before. Don't need to call CallGermline(...) again.
            	loc_qual = it->qual;
            	loc_gq = it->gq;
            	loc_gt = it->gt;
            }

            // write the info tag for the corresponding var type
            candidate_variant.variant.samples[sample_name][type_prefix + "MUAF"].push_back(convertToString(loc_min_allele_freq));
            candidate_variant.variant.samples[sample_name][type_prefix + "MUQUAL"].push_back(convertToString(loc_qual));
            candidate_variant.variant.samples[sample_name][type_prefix + "MUGT"].push_back(loc_gt);
            candidate_variant.variant.samples[sample_name][type_prefix + "MUGQ"].push_back(convertToString(loc_gq));

            switch(type_prefix[0]){
                case 'S':
            	    is_snp_done = true;
            	    break;
                case 'M':
            	    is_mnp_done = true;
            	    break;
                case 'I':
            	    is_indel_done = true;
            	    break;
                case 'H':
            	    is_hotspot_done = true;
            	    break;
            }
        }
    }

    // sort inference_results_union according to min_allele_freq in the ascending order
    sort(inference_results_union.begin(), inference_results_union.end());
    // write the info tag for the union of min_allele_freq of the var types
    for(unsigned int i_freq = 0; i_freq < inference_results_union.size(); ++i_freq){
        candidate_variant.variant.samples[sample_name]["MUAF"].push_back(convertToString(inference_results_union[i_freq].min_allele_freq));
        candidate_variant.variant.samples[sample_name]["MUQUAL"].push_back(convertToString(inference_results_union[i_freq].qual));
        candidate_variant.variant.samples[sample_name]["MUGT"].push_back(inference_results_union[i_freq].gt);
        candidate_variant.variant.samples[sample_name]["MUGQ"].push_back(convertToString(inference_results_union[i_freq].gq));
    }
}


void GlueOutputVariant(EnsembleEval &my_ensemble, VariantCandidate &candidate_variant, const ExtendParameters &parameters, int _best_allele_index, int sample_index){
    string sample_name = "";
    if(sample_index >= 0) {
        sample_name = candidate_variant.variant.sampleNames[sample_index];
    }

    DecisionTreeData my_decision(*(my_ensemble.variant));
    my_decision.tune_sbias = parameters.my_controls.sbias_tune;
    my_decision.SetupFromMultiAllele(my_ensemble);

    // pretend we can classify reads across multiple alleles
    if(not my_ensemble.is_hard_classification_for_reads_done_){
        my_ensemble.ApproximateHardClassifierForReads();
    }

    if(my_ensemble.allele_eval.total_theory.GetIsMolecularTag()){
    	if(not my_ensemble.is_hard_classification_for_families_done_){
            my_ensemble.ApproximateHardClassifierForFamilies();
    	}
  	    // I count the strand coverage by families, not reads.
    	// Note that family_id_ is the hard classification results for functional and non-functional families.
    	// And we classify non-functional families as outliers.
    	// This gives FXX = 1.0 - (# of functional families) / (# of families).
  	    my_decision.all_summary_stats.AssignStrandToHardClassifiedReads(my_ensemble.family_strand_id_, my_ensemble.family_id_);
    }
    else{
    	// non-mol-tag
        my_decision.all_summary_stats.AssignStrandToHardClassifiedReads(my_ensemble.strand_id_, my_ensemble.read_id_);
    }

    my_decision.all_summary_stats.AssignPositionFromEndToHardClassifiedReads(my_ensemble.read_id_, my_ensemble.dist_to_left_, my_ensemble.dist_to_right_);

    float smallest_allele_freq = 1.0f;
    for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.allele_identity_vector.size(); _alt_allele_index++) {
        // for each alt allele, do my best
        // thresholds here can vary by >type< of allele
        float local_min_allele_freq = FreqThresholdByType(my_ensemble.allele_identity_vector[_alt_allele_index], parameters.my_controls,
                                                          candidate_variant.variant_specific_params[_alt_allele_index]);

        if (local_min_allele_freq < smallest_allele_freq){
            smallest_allele_freq = local_min_allele_freq;  // choose least-restrictive amongst multialleles
        }

        /* The following piece of code seems redundant. Perhaps due to historical reasons?
        my_ensemble.ComputePosteriorGenotype(_alt_allele_index, local_min_allele_freq,
            my_decision.summary_info_vector[_alt_allele_index].genotype_call,
            my_decision.summary_info_vector[_alt_allele_index].gt_quality_score,
            my_decision.summary_info_vector[_alt_allele_index].variant_qual_score);
        */
        SummarizeInfoFieldsFromEnsemble(my_ensemble, *(my_ensemble.variant), _alt_allele_index, sample_name);
    }

    my_decision.best_allele_index = _best_allele_index;
    my_decision.best_allele_set = true;

    // tell the evaluator to do a genotype
    // choose a diploid genotype
    // return it and set it so that decision tree cannot override

    //@TODO: fix this frequency to be sensible
    float local_min_allele_freq = smallest_allele_freq; // must choose a qual score relative to some frequency

    my_ensemble.MultiAlleleGenotype(local_min_allele_freq,
                                    my_decision.eval_genotype.genotype_component,
                                    my_decision.eval_genotype.evaluated_genotype_quality,
                                    my_decision.eval_genotype.evaluated_variant_quality,
                                    parameters.my_eval_control.max_detail_level);

    my_decision.eval_genotype.genotype_already_set = true; // because we computed it here

    // Tell me what QUAL means (Is QUAL for a ref call or for a var call?) in case we won't show GT in the info tag.
    candidate_variant.variant.samples[sample_name]["QT"].push_back(convertToString(not my_decision.eval_genotype.IsReference()));

    // and I must also set for each allele so that the per-allele filter works
    for(unsigned int i_allele = 0; i_allele < my_decision.allele_identity_vector.size(); ++i_allele){
        my_decision.summary_info_vector[i_allele].variant_qual_score = my_decision.eval_genotype.evaluated_variant_quality;
        my_decision.summary_info_vector[i_allele].gt_quality_score = my_decision.eval_genotype.evaluated_genotype_quality;
    }

    // now that all the data has been gathered describing the variant, combine to produce the output
    my_decision.DecisionTreeOutputToVariant(candidate_variant, parameters, sample_index);
}

// Read and process records appropriate for this variant; positions are zero based
void EnsembleEval::StackUpOneVariant(const ExtendParameters &parameters, const PositionInProgress& bam_position, int sample_index)
{

  // Initialize random number generator for each stack -> ensure reproducibility
  RandSchrange RandGen(parameters.my_controls.RandSeed);

  read_stack.clear();  // reset the stack
  read_stack.reserve(parameters.my_controls.downSampleCoverage);
  int read_counter = 0;

  for (Alignment* rai = bam_position.begin; rai != bam_position.end; rai = rai->next) {

    // Check global conditions to stop reading in more alignments
    if (rai->original_position > multiallele_window_start)
      break;

    if (rai->alignment.Position > multiallele_window_start)
      continue;

    if (rai->filtered)
      continue;

    if (rai->alignment.GetEndPosition() < multiallele_window_end)
      continue;

    if (parameters.multisample) {
	  if (rai->sample_index != sample_index) {
		  continue;
      }
    }

    // Reservoir Sampling
    if (read_stack.size() < (unsigned int)parameters.my_controls.downSampleCoverage) {
      read_counter++;
      read_stack.push_back(rai);
    } else {
      read_counter++;
      // produces a uniformly distributed test_position between [0, read_counter-1]
      unsigned int test_position = ((double)RandGen.Rand() / ((double)RandGen.RandMax + 1.0)) * (double)read_counter;
      if (test_position < (unsigned int)parameters.my_controls.downSampleCoverage)
        read_stack[test_position] = rai;
    }
  }
}

// The class is simply used for random_shuffle
class MyRandSchrange : private RandSchrange{
public:
    MyRandSchrange(int seed = 1){SetSeed(seed);} ;
    int operator()(int upper_lim){return Rand() % upper_lim;}; // return a random number between 0 and upper_lim-1
};

// We do down sampling based on the following rules when we have molecular tag.
// Rule 1: Try to get as many functional families as possible after down sampling
// Rule 2: Try to equalize the sizes of functional families after down sampling.
// Rule 3: If we need to drop a functional family, drop the functional family with smaller family size.
void EnsembleEval::DoDownSamplingMolTag(const ExtendParameters &parameters, vector< vector< MolecularFamily<Alignment*> > > &my_molecular_families,
			                            unsigned int num_reads_available, unsigned int num_func_fam, int strand_key)
{
	MyRandSchrange my_rand_schrange(parameters.my_controls.RandSeed); 	// The random number generator that we use to guarantee reproducibility.
    unsigned int read_counter = 0;
    unsigned int min_fam_size = (unsigned int) parameters.tag_trimmer_parameters.min_family_size;
    unsigned int downSampleCoverage = (unsigned int) parameters.my_controls.downSampleCoverage;
    unsigned int num_hyp = allele_identity_vector.size() + 2;
	read_stack.clear();  // reset the stack
	read_stack.reserve(downSampleCoverage);
	allele_eval.total_theory.my_eval_families.clear();

	// We can keep all the reads in all functional families :D
	if(num_reads_available <= downSampleCoverage){
		allele_eval.total_theory.my_eval_families.reserve(num_reads_available);
		for(vector< MolecularFamily<Alignment*> >::iterator family_it = my_molecular_families[strand_key].begin();
				family_it != my_molecular_families[strand_key].end(); ++family_it){
			if(family_it->is_func_family_temp){
				allele_eval.total_theory.my_eval_families.push_back(EvalFamily(family_it->family_barcode, family_it->strand_key));
				allele_eval.total_theory.my_eval_families.back().InitializeEvalFamily(num_hyp);
				for(unsigned int i_read = 0; i_read < family_it->family_members_temp.size(); ++i_read){
					read_stack.push_back(family_it->family_members_temp[i_read]);
					allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
					++read_counter;
				}
			}
		}
		return;
	}

    unsigned int num_of_func_fam_after_down_sampling = 0;
    unsigned int min_fam_size_after_down_sampling = 0;
    // The size of this vector is < 3, i.e., we only need to do random shuffle for the at most two different sizes of families.
    // There are two cases that we need to randomly pick families.
    // (random_shuffle case 1): We can not preserve the functional families of size = min_fam_size_after_down_sampling.
    // (random_shuffle case 2): Some families of the same size have one more read than others.
    vector<unsigned int> size_of_fam_need_random_shuffle(0);

	// sort the families by the family size in the ascending order
	sort(my_molecular_families[strand_key].begin(), my_molecular_families[strand_key].end());

	// We need to give up some reads but we can keep all functional families :)
	if(downSampleCoverage >= min_fam_size * num_func_fam){
		num_of_func_fam_after_down_sampling = num_func_fam;
		unsigned int i_fam = my_molecular_families[strand_key].size() - num_of_func_fam_after_down_sampling;
		min_fam_size_after_down_sampling = my_molecular_families[strand_key][i_fam].family_members_temp.size();
	}
	// We need to give up some functional families... :(
	else{
		num_of_func_fam_after_down_sampling = downSampleCoverage / min_fam_size;
		unsigned int i_fam = my_molecular_families[strand_key].size() - num_of_func_fam_after_down_sampling;
		min_fam_size_after_down_sampling = my_molecular_families[strand_key][i_fam].family_members_temp.size();
		if(i_fam < my_molecular_families[strand_key].size() - 1){
			if(min_fam_size_after_down_sampling == my_molecular_families[strand_key][i_fam + 1].family_members_temp.size()){
				// (random_shuffle case 1):
				// We can't preserve all the families of size my_molecular_families[strand_key][i_fam].family_members_temp.size().
				// For fairness, we will to randomly pick some of families with this size.
				size_of_fam_need_random_shuffle.push_back(my_molecular_families[strand_key][i_fam].family_members_temp.size());
			}
		}
	}
	allele_eval.total_theory.my_eval_families.reserve(num_of_func_fam_after_down_sampling);

	// Count how many reads we want to pick in each family.
	unsigned int reads_remaining = downSampleCoverage;
	vector<unsigned int> num_reads_picked_in_fam;
	// num_reads_picked_in_fam[i] is the number of reads we picked for the family my_molecular_families[strand_key][i_fam]
	// where i_fam = my_molecular_families[strand_key].size() - i - 1
	num_reads_picked_in_fam.assign(num_of_func_fam_after_down_sampling, 0);

	unsigned int break_at_i_fam = 0;
	while(reads_remaining > 0){
		for(unsigned int i = 0; i < num_of_func_fam_after_down_sampling; ++i){
			unsigned int i_fam = my_molecular_families[strand_key].size() - i - 1;
			// Note that my_molecular_families[strand_key][i_fam] should be functional since we sort my_molecular_families[strand_key]
			// All the reads in the family are picked up, and of course for the next family since we sort my_molecular_families[strand_key]
			// So let's break the for loop and starts with the family with the largest size.
			if(num_reads_picked_in_fam[i] >= my_molecular_families[strand_key][i_fam].family_members_temp.size()){
				break;
			}

			// We take one more read from the family
		    ++num_reads_picked_in_fam[i];
    		--reads_remaining;

			// Sorry, we can't pick up more reads. We have reached the down sampling limit.
			if(reads_remaining <= 0){
				break_at_i_fam = i_fam;
				break;
			}
		}
	}

	if(break_at_i_fam < my_molecular_families[strand_key].size() - 1){
		if(my_molecular_families[strand_key][break_at_i_fam].family_members_temp.size() == my_molecular_families[strand_key][break_at_i_fam + 1].family_members_temp.size()){
			// (random_shuffle case 2):
			// Some of the families of size my_molecular_families[strand_key][break_at_i_fam].family_members_temp.size() get one more read after down sampling.
			// For fairness, we will do random shuffle for the families of this size.
			size_of_fam_need_random_shuffle.push_back(my_molecular_families[strand_key][break_at_i_fam].family_members_temp.size());
		}
	}

	// sort size_of_fam_need_random_shuffle in increasing order and remove repeats
	if(size_of_fam_need_random_shuffle.size() == 2){
		if(size_of_fam_need_random_shuffle[0] > size_of_fam_need_random_shuffle[1]){
			swap(size_of_fam_need_random_shuffle[0], size_of_fam_need_random_shuffle[1]);
		}
		else if(size_of_fam_need_random_shuffle[0] == size_of_fam_need_random_shuffle[1]){
			size_of_fam_need_random_shuffle.resize(1);
		}
	}

	if(size_of_fam_need_random_shuffle.size() > 0){
		// Now we random shuffle the orders of the families of the same size to give more randomness if we can't pick all among them.
		// Note that my_molecular_families[strand_key] will remain sorted after random_shuffle.
		vector< MolecularFamily<Alignment*> >::iterator current_fam_size_begin_it = my_molecular_families[strand_key].begin();
		for(vector< MolecularFamily<Alignment*> >::iterator family_it = my_molecular_families[strand_key].begin();
				family_it != my_molecular_families[strand_key].end(); ++family_it){
			if(family_it->family_members_temp.size() != current_fam_size_begin_it->family_members_temp.size()){
				// we've got all families of size family_it->family_members_temp.size()
				if(current_fam_size_begin_it->family_members_temp.size() == size_of_fam_need_random_shuffle[0]){
					// random shuffle the families of the size size_of_fam_need_random_shuffle[0]
			    	random_shuffle(current_fam_size_begin_it, family_it, my_rand_schrange);
			    	if(size_of_fam_need_random_shuffle.size() == 1){
			    		// We've done random shuffle for all families of the sizes needed
			    		size_of_fam_need_random_shuffle.clear();
			    		break;
			    	}
			    	else{
			    		// size_of_fam_need_random_shuffle[0] is done. Will do random_shuffle for size_of_fam_need_random_shuffle[1]
			    		// Let's do it like a sliding window. Note that size_of_fam_need_random_shuffle must be sorted as well!!!
			    		size_of_fam_need_random_shuffle.assign(1, size_of_fam_need_random_shuffle[1]);
			    	}
			    }
			    current_fam_size_begin_it = family_it;
			}
		}
		// Don't forget to random shuffle the families if size_of_fam_need_random_shuffle[0] equals the maximum family size
		if(size_of_fam_need_random_shuffle.size() > 0){
		    random_shuffle(current_fam_size_begin_it, my_molecular_families[strand_key].end(), my_rand_schrange);
		}
	}

	for(unsigned int i = 0; i < num_of_func_fam_after_down_sampling; ++i){
		unsigned int i_fam = my_molecular_families[strand_key].size() - i - 1;
		MolecularFamily<Alignment*> *ptr_fam = &(my_molecular_families[strand_key][i_fam]);
		// Add one more family
		allele_eval.total_theory.my_eval_families.push_back(EvalFamily(ptr_fam->family_barcode, ptr_fam->strand_key));
		allele_eval.total_theory.my_eval_families.back().InitializeEvalFamily(num_hyp);
		allele_eval.total_theory.my_eval_families.back().family_members.reserve(num_reads_picked_in_fam[i]);
		random_shuffle(ptr_fam->family_members_temp.begin(), ptr_fam->family_members_temp.end(), my_rand_schrange);
		// Pick up num_reads_picked_in_fam[i] reads from family i_fam.
		for(unsigned int i_read = 0; i_read < num_reads_picked_in_fam[i]; ++i_read){
			// Add the read into read_stack
			read_stack.push_back(ptr_fam->family_members_temp[i_read]);
			// Add the read in to the family
			allele_eval.total_theory.my_eval_families.back().AddNewMember(read_counter);
			++read_counter;
		}
	}
}

// Currently only take the reads on one strand
void EnsembleEval::StackUpOneVariantMolTag(const ExtendParameters &parameters, vector< vector< MolecularFamily<Alignment*> > > &my_molecular_families, int sample_index)
{
	int strand_key = -1;
	unsigned int min_fam_size = (unsigned int) parameters.tag_trimmer_parameters.min_family_size;
	vector<unsigned int> num_func_fam_by_strand = {0, 0};
	vector<unsigned int> num_reads_available_by_strand = {0, 0};

	for(unsigned int i_strand = 0; i_strand < my_molecular_families.size(); ++i_strand){
		for(vector< MolecularFamily<Alignment*> >::iterator family_it = my_molecular_families[i_strand].begin();
				family_it != my_molecular_families[i_strand].end(); ++family_it){
			if(not family_it->SetFunctionality(min_fam_size)){
				continue;
			}
			family_it->family_members_temp.reserve(family_it->family_members.size());
			for(vector<Alignment*>::iterator member_it = family_it->family_members.begin(); member_it != family_it->family_members.end(); ++member_it){

				// Although we have done this in the function GenerateMyMolecularFamilies, do it again to make sure everything is right.
				if ((*member_it)->filtered)
					continue;
				// Although we have done this in the function GenerateMyMolecularFamilies, do it again to make sure everything is right.
				if(parameters.multisample) {
				    if ((*member_it)->sample_index != sample_index) {
					    continue;
				    }
			    }

			    // Check global conditions to stop reading in more alignments
				if ((*member_it)->original_position > multiallele_window_start
						or (*member_it)->alignment.Position > multiallele_window_start
						or (*member_it)->alignment.GetEndPosition() < multiallele_window_end)
					continue;


				// family_members_temp stores the reads which are not filtered out here
				family_it->family_members_temp.push_back((*member_it));
			}

			// We may change the family size since some reads may be filtered out.
			// Need to determine the functionality again!
			if(family_it->family_members_temp.size() >= min_fam_size){
				family_it->is_func_family_temp = true;
				// Count how many reads and functional families available for down sampling
				num_reads_available_by_strand[i_strand] += family_it->family_members_temp.size();
				++num_func_fam_by_strand[i_strand];
			}
			else{
				family_it->is_func_family_temp = false;
			}
		}
	}

	// For the current molecular barcoding scheme (bcprimer), the reads in each amplicom should be on on strand only.
	// However, we sometimes get families on both strands, primarily due to false priming.
	// Here I pick the strand that has more functional families
	strand_key = num_func_fam_by_strand[0] > num_func_fam_by_strand[1] ? 0 : 1;

	// Do down-sampling
	DoDownSamplingMolTag(parameters, my_molecular_families, num_reads_available_by_strand[strand_key], num_func_fam_by_strand[strand_key], strand_key);
}

bool EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, VariantCallerContext& vc,
    VariantCandidate &candidate_variant, const PositionInProgress& bam_position,
	vector< vector< MolecularFamily<Alignment*> > > &molecular_families, int sample_index)
{
  string sample_name = "";
  bool use_molecular_tag = vc.tag_trimmer->HaveTags();
  if (sample_index >= 0) {sample_name = candidate_variant.variant.sampleNames[sample_index];}

  int chr_idx = vc.ref_reader->chr_idx(candidate_variant.variant.sequenceName.c_str());

  EnsembleEval my_ensemble(candidate_variant.variant);
  my_ensemble.allele_eval.total_theory.SetIsMolecularTag(use_molecular_tag);
  my_ensemble.SetupAllAlleles(*vc.parameters, *vc.global_context, *vc.ref_reader, chr_idx);
  my_ensemble.FilterAllAlleles(vc.parameters->my_controls.filter_variant, candidate_variant.variant_specific_params); // put filtering here in case we want to skip below entries

  // We read in one stack per multi-allele variant
  if(use_molecular_tag){
	my_ensemble.StackUpOneVariantMolTag(*vc.parameters, molecular_families, sample_index);
  }
  else{
    my_ensemble.StackUpOneVariant(*vc.parameters, bam_position, sample_index);
  }
  if (my_ensemble.read_stack.empty()) {
    //cerr << "Nonfatal: No reads found for " << candidate_variant.variant.sequenceName << "\t" << my_ensemble.multiallele_window_start << endl;
    NullFilterReason(candidate_variant.variant, sample_name);
    if (not use_molecular_tag) {
  	  RemoveVcfInfo(candidate_variant.variant, vector<string>({"MDP", "MRO", "MAO", "MAF"}), sample_name, sample_index);
    }
    string my_reason = "NODATA";
    AddFilterReason(candidate_variant.variant, my_reason, sample_name);
    SetFilteredStatus(candidate_variant.variant, true);
    candidate_variant.variant.samples[sample_name]["GT"].clear();
    candidate_variant.variant.samples[sample_name]["GT"].push_back("./.");
    return false;
  }


  // handle the unfortunate case in which we must try multiple alleles to be happy
  // try only ref vs alt allele here
  // leave ensemble in ref vs alt state

  // glue in variants
  my_ensemble.SpliceAllelesIntoReads(thread_objects, *vc.global_context, *vc.parameters, *vc.ref_reader, chr_idx);

  my_ensemble.allele_eval.my_params = vc.parameters->my_eval_control;

  // fill in quantities derived from predictions
  int num_hyp_no_null = my_ensemble.allele_identity_vector.size()+1; // num alleles +1 for ref
  my_ensemble.allele_eval.InitForInference(thread_objects, my_ensemble.read_stack, *vc.global_context, num_hyp_no_null, my_ensemble.allele_identity_vector);

  if(use_molecular_tag){
	 // Filter out outlier reads to make sure there is no outlier family!
	 my_ensemble.allele_eval.total_theory.OutlierFiltering(vc.parameters->my_eval_control.DataReliability(), false);
	 my_ensemble.allele_eval.total_theory.SetFuncionalityForFamilies(vc.parameters->tag_trimmer_parameters.min_family_size);
  }

  // do inference
  my_ensemble.allele_eval.ExecuteInference();
  // now we're in the guaranteed state of best index
  int best_allele = my_ensemble.DetectBestMultiAllelePair();

  // output to variant
  GlueOutputVariant(my_ensemble, candidate_variant, *vc.parameters, best_allele, sample_index);

  if (not use_molecular_tag) {
	  RemoveVcfInfo(candidate_variant.variant, vector<string>({"MDP", "MRO", "MAO", "MAF"}), sample_name, sample_index);
  }

  // output the inference results (MUQUAL, MUGT, MUGQ, etc.) if I turn on multi_min_allele_freq
  if(vc.parameters->program_flow.is_multi_min_allele_freq){
	  MultiMinAlleleFreq(my_ensemble, candidate_variant, sample_index, vc.parameters->program_flow, vc.parameters->my_eval_control.max_detail_level);
  }

  // test diagnostic output for this ensemble
  if (vc.parameters->program_flow.rich_json_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)) // look at everything that came through
    JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->program_flow.json_plot_dir, true);
  if (vc.parameters->program_flow.minimal_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)) // look at everything that came through
    JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->program_flow.json_plot_dir, false);

  return true;
}





