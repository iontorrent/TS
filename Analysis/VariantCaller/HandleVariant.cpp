/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HandleVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "HandleVariant.h"
#include "DecisionTreeData.h"
#include "StackEngine.h"
#include "ShortStack.h"
#include "MiscUtil.h"
#include "ExtendedReadInfo.h"
#include "ClassifyVariant.h"
#include "DiagnosticJson.h"
#include "ExtendParameters.h"

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

void EnsembleEval::MultiMinAlleleFreq(const vector<float>& multi_min_allele_freq){
	variant->info["MUQUAL"].clear();
	variant->info["MUGT"].clear();
	variant->info["MUGQ"].clear();
    for (vector<float>::const_iterator maf_it = multi_min_allele_freq.begin(); maf_it != multi_min_allele_freq.end(); ++maf_it){
    	float loc_qual = 0.0f;
       	int loc_gq = 0;
        float evaluated_genotype_quality = 0.0f;
        int quality_type = 0;
        string loc_gt;
        // Let's do the inference for min-allele-freq = *maf_it
        vector<int> genotype_component;
        allele_eval.CallByIntegral(*maf_it, *maf_it, genotype_component, evaluated_genotype_quality, loc_qual, quality_type);
        loc_gt = convertToString(genotype_component[0]) + "/" + convertToString(genotype_component[1]);
        loc_gq = int(round(evaluated_genotype_quality)); // genotype quality is rounded as an integer.
        variant->info["MUQUAL"].push_back(convertToString(loc_qual));
        variant->info["MUGT"].push_back(loc_gt);
        variant->info["MUGQ"].push_back(convertToString(loc_gq));
    }
}

// Override fd-nonsnp-min-var-coverage
void OverrideMinVarCov(int fd_nonsnp_min_var_cov,
                       const vector<AlleleIdentity>& allele_identity_vector,
		               const vector<vector<int> >& global_flow_disruptive_matrix,
		               vector<VariantSpecificParams>& variant_specific_params)
{
	assert(allele_identity_vector.size() + 1 == global_flow_disruptive_matrix.size());
    if (fd_nonsnp_min_var_cov < 1)
    	return;

	for (unsigned int i_alt = 0; i_alt < allele_identity_vector.size(); ++i_alt){
		// Override min_var_coverage of the allele if all the following are satisfied
		// a) Not override by hotspot
		// b) The allele is flow-disrupted
		// c) It is not a SNP and it is not a padding SNP
		if ((not variant_specific_params[i_alt].min_var_coverage_override)
				and (global_flow_disruptive_matrix[0][i_alt + 1] == 2)
				and (not (allele_identity_vector[i_alt].status.isSNP or allele_identity_vector[i_alt].status.isPaddedSNP))){
			variant_specific_params[i_alt].min_var_coverage_override = true;
			variant_specific_params[i_alt].min_var_coverage = fd_nonsnp_min_var_cov;
		}
	}
}

void GlueOutputVariant(EnsembleEval &my_ensemble, VariantCandidate &candidate_variant, const ExtendParameters &parameters, int _best_allele_index, int sample_index){
    string sample_name = (sample_index >= 0) ? candidate_variant.variant.sampleNames[sample_index] : "";
    DecisionTreeData my_decision(*(my_ensemble.variant));
    my_decision.use_molecular_tag = my_ensemble.allele_eval.total_theory.GetIsMolecularTag();
    my_decision.tune_sbias = parameters.my_controls.sbias_tune;
    my_decision.SetupFromMultiAllele(&(my_ensemble.allele_identity_vector), &(my_ensemble.info_fields));

    if (my_ensemble.read_id_.empty()){
        cerr << "ERROR: Can't GlueOutputVariant with empty read id." << endl;
        exit(-1);
    }
    my_decision.all_summary_stats.AssignStrandToHardClassifiedReads(my_ensemble.strand_id_, my_ensemble.read_id_);
    my_decision.all_summary_stats.AssignPositionFromEndToHardClassifiedReads(my_ensemble.read_id_, my_ensemble.dist_to_left_, my_ensemble.dist_to_right_);

	for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.allele_identity_vector->size(); _alt_allele_index++) {
		SummarizeInfoFieldsFromEnsemble(my_ensemble, *(my_ensemble.variant), _alt_allele_index, sample_name);
    }

	if (my_ensemble.allele_eval.total_theory.GetIsMolecularTag()){
		my_ensemble.variant->info["TGSM"].clear();
		for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.allele_identity_vector->size(); _alt_allele_index++) {
			if (my_ensemble.tag_similar_counts_.empty()){
				my_ensemble.variant->info["TGSM"].push_back(".");
			}else{
				my_ensemble.variant->info["TGSM"].push_back(convertToString(my_ensemble.tag_similar_counts_[_alt_allele_index]));
			}
		}
		my_decision.all_summary_stats.tag_similar_counts = my_ensemble.tag_similar_counts_;
	}

    my_decision.best_allele_index = _best_allele_index;
    my_decision.best_allele_set = true;
    my_decision.is_possible_polyploidy_allele = my_ensemble.is_possible_polyploidy_allele;

    float af_cutoff_rej = 1.0f;
    float af_cutoff_gt = 1.0f;
    my_ensemble.ServeAfCutoff(parameters.my_controls, candidate_variant.variant_specific_params, af_cutoff_rej, af_cutoff_gt);

    if (my_ensemble.DEBUG > 0){
    	cout << "+ Evaluating the variant (" << PrintVariant(*my_ensemble.variant) << "):" <<endl
    		 << "  - allele-freq-cutoff for Rej = "<< af_cutoff_rej << endl
    		 << "  - allele-freq-cutoff for GT = "<< af_cutoff_gt << endl;
    }

    int quality_type = 0;
    my_ensemble.MultiAlleleGenotype(af_cutoff_rej, af_cutoff_gt,
                                    my_decision.eval_genotype.genotype_component,
                                    my_decision.eval_genotype.evaluated_genotype_quality,
                                    my_decision.eval_genotype.evaluated_variant_quality,
									quality_type);

    my_decision.eval_genotype.genotype_already_set = true; // because we computed it here

    // and I must also set for each allele so that the per-allele filter works
    for(unsigned int i_allele = 0; i_allele < my_decision.allele_identity_vector->size(); ++i_allele){
        my_decision.summary_info_vector[i_allele].variant_qual_score = my_decision.eval_genotype.evaluated_variant_quality;
        my_decision.summary_info_vector[i_allele].gt_quality_score = my_decision.eval_genotype.evaluated_genotype_quality;
    }

    // Override fd-nonsnp-min-var-coverage here
    OverrideMinVarCov(parameters.my_controls.fd_nonsnp_min_var_cov,
    		my_ensemble.allele_identity_vector,
			my_ensemble.global_flow_disruptive_matrix,
			candidate_variant.variant_specific_params);

    if (parameters.my_controls.disable_filters){
    	my_ensemble.GatherInfoForOfflineFiltering(parameters.my_controls, _best_allele_index);
    }

    // now that all the data has been gathered describing the variant, combine to produce the output
    my_decision.DecisionTreeOutputToVariant(candidate_variant, parameters, sample_index);
}

void RemoveVcfFormat(vcf::Variant &variant, const vector<string> &keys){
	for(vector<string>::const_iterator key_it = keys.begin(); key_it != keys.end(); ++key_it){
		variant.format.erase(std::remove(variant.format.begin(), variant.format.end(), *key_it), variant.format.end());
	}
}


void DoStepsForNoData(VariantCandidate& candidate_variant, const string& sample_name, int sample_index, bool use_molecular_tag, string my_reason){
    //cerr << "Nonfatal: No reads found for " << candidate_variant.variant.sequenceName << "\t" << my_ensemble.multiallele_window_start << endl;
    NullFilterReason(candidate_variant.variant, sample_name);
    if (my_reason.empty()){
    	my_reason = "NODATA";
    }
    AddFilterReason(candidate_variant.variant, my_reason, sample_name);
    SetFilteredStatus(candidate_variant.variant, true);
    candidate_variant.variant.samples[sample_name]["GT"] = {"./."};
}


// return 0: normal termination
// return 1: no data (empty read stack)
// return 2: no data (no valid functional families on read stack)
int EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, VariantCallerContext& vc,
    VariantCandidate &candidate_variant, const PositionInProgress& bam_position,
	vector< vector<MolecularFamily> > &molecular_families, int sample_index)
{
  unsigned long t0 = clock();
  string sample_name = (sample_index >= 0)? candidate_variant.variant.sampleNames[sample_index] : "";
  const bool use_molecular_tag = vc.mol_tag_manager->tag_trimmer->HaveTags();

  if(vc.parameters->program_flow.DEBUG > 0 ){
	  cout<< endl << "[tvc] Start EnsembleProcessOneVariant for (" << PrintVariant(candidate_variant.variant) << ")"<< endl << endl;
  }

  if (not use_molecular_tag){
	  RemoveVcfFormat(candidate_variant.variant, {"MDP", "MAO", "MRO", "MAF"});
  }

  EnsembleEval my_ensemble(candidate_variant.variant);

  // Allele preparation
  my_ensemble.SetupAllAlleles(*vc.parameters, *vc.global_context, *vc.ref_reader);
  my_ensemble.FilterAllAlleles(vc.parameters->my_controls, candidate_variant.variant_specific_params); // put filtering here in case we want to skip below entries

  // set parameters for the evaluator
  my_ensemble.SetAndPropagateParameters(vc.parameters, use_molecular_tag, candidate_variant.variant_specific_params);

  if (vc.parameters->program_flow.DEBUG > 0){
	  list<list<int> >allele_groups;
	  CandidateExaminer my_examiner(&thread_objects, &vc);
	  my_examiner.SetupVariantCandidate(candidate_variant);
	  my_examiner.FindLookAheadEnd0();
	  my_examiner.SplitCandidateVariant(allele_groups);
  }

  // We read in one stack per multi-allele variant
  if (use_molecular_tag){
	my_ensemble.StackUpOneVariantMolTag(*vc.parameters, molecular_families, sample_index);
  }
  else{
    my_ensemble.StackUpOneVariant(*vc.parameters, bam_position, sample_index);
  }

  // No data
  if (my_ensemble.read_stack.empty()) {
    DoStepsForNoData(candidate_variant, sample_name, sample_index, use_molecular_tag, "NODATA");
    if(vc.parameters->program_flow.DEBUG > 0 ){
	  cout<< "+ No data: empty read stack!" << endl << endl
          << "[tvc] Complete EnsembleProcessOneVariant for ("<< PrintVariant(candidate_variant.variant) << "). Processing time = " << (double) (clock() - t0) / 1E6 << " sec." << endl << endl;
	}
    return 1;
  }

  // glue in variants
  my_ensemble.SpliceAllelesIntoReads(thread_objects, *vc.global_context, *vc.parameters, *vc.ref_reader);

  // Calculate flow-disruptiveness in the read level
  my_ensemble.FlowDisruptivenessInReadLevel(*vc.global_context);

  // fill in quantities derived from predictions
  my_ensemble.allele_eval.InitForInference(thread_objects, my_ensemble.read_stack, *vc.global_context, my_ensemble.allele_identity_vector);

  // No valid function family
  if (use_molecular_tag){
    if (my_ensemble.allele_eval.total_theory.GetNumFuncFamilies() == 0) {
	  DoStepsForNoData(candidate_variant, sample_name, sample_index, use_molecular_tag, "NOVALIDFUNCFAM");
	  if (vc.parameters->program_flow.DEBUG > 0){
	    cout << "+ No valid functional families on read stack!" << endl << endl
			 << "[tvc] Complete EnsembleProcessOneVariant for ("<< PrintVariant(candidate_variant.variant) << "). Processing time = " << (double) (clock() - t0) / 1E6 << " sec." << endl << endl;
	  }
	  return 2;
	}
  }


  // do inference
  my_ensemble.allele_eval.ExecuteInference();

  // set fd in the read_stack level.
  my_ensemble.FlowDisruptivenessInReadStackLevel(vc.parameters->my_controls.min_ratio_for_fd);

  // now we're in the guaranteed state of best index
  vector<float> semi_soft_allele_freq_est;
  int best_allele = my_ensemble.DetectBestMultiAllelePair(semi_soft_allele_freq_est);
  if (vc.parameters->my_controls.report_ppa){
    my_ensemble.DetectPossiblePolyploidyAlleles(semi_soft_allele_freq_est, vc.parameters->my_controls, candidate_variant.variant_specific_params);
  }

  if (use_molecular_tag){
    my_ensemble.CalculateTagSimilarity(*vc.mol_tag_manager, vc.parameters->my_controls.tag_sim_max_cov, sample_index);
    my_ensemble.VariantFamilySizeHistogram();
  }

  // output to variant
  GlueOutputVariant(my_ensemble, candidate_variant, *vc.parameters, best_allele, sample_index);

  // output the inference results (MUQUAL, MUGT, MUGQ) if I turn on multi_min_allele_freq
  if (vc.parameters->program_flow.is_multi_min_allele_freq){
     my_ensemble.MultiMinAlleleFreq(vc.parameters->program_flow.multi_min_allele_freq);
  }


  // test diagnostic output for this ensemble
  if (vc.parameters->program_flow.rich_json_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)){ // look at everything that came through
	  cout << "+ Dumping rich json diagnostic for (" << PrintVariant(candidate_variant.variant) << ")" << endl;
	  JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->json_plot_dir, true);
  }
  if (vc.parameters->program_flow.minimal_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)){ // look at everything that came through
	  cout << "+ Dumping minimal json diagnostic for (" << PrintVariant(candidate_variant.variant) << ")" << endl;
	  JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->json_plot_dir, false);
  }

  if(vc.parameters->program_flow.DEBUG > 0){
      cout << endl << "[tvc] Complete EnsembleProcessOneVariant for (" << PrintVariant(candidate_variant.variant) << "). Processing time = " << (double) (clock() - t0) / 1E6 << " sec." << endl << endl;
  }
  return 0;
}

CandidateExaminer::CandidateExaminer(){
	thread_objects_ = NULL;
	vc_ = NULL;
	my_ensemble_ = NULL;
	max_group_size_allowed_ = 32;
}

CandidateExaminer::CandidateExaminer(PersistingThreadObjects* thread_objects, VariantCallerContext* vc){
	thread_objects_ = NULL;
	vc_ = NULL;
	my_ensemble_ = NULL;
	max_group_size_allowed_ = 32;
	Initialize(thread_objects, vc);
}

CandidateExaminer::~CandidateExaminer(){
	ClearVariantCandidate();
}

void CandidateExaminer::Initialize(PersistingThreadObjects* thread_objects, VariantCallerContext* vc){
	thread_objects_ = thread_objects;
	vc_ = vc;
	max_group_size_allowed_ = vc->parameters->max_alt_num;
}

// flow_disruptive_code[i][j] indicates the flow-disruptiveness between the j-th hyp (j=0: ref, j=1: allele 1, etc) and the query sequence of the read in test_read_stack[i]
// flow_disruptive_code[i][j] = -1: indefinite (fail to splicing, not cover the position, etc.)
// flow_disruptive_code[i][j] = 0: the j-th hyp is exactly the same as the query sequence
// flow_disruptive_code[i][j] = 1: the j-th hyp is an HP-INDEL of the query sequence
// flow_disruptive_code[i][j] = 2: the j-th hyp is neither HP-INDEL nor flow-disruptive of the query sequence
// flow_disruptive_code[i][j] = 3: the j-th hyp disrupts the flow of query sequence (i.e., the read is very unlikely to support the j-th hyp).
void CandidateExaminer::QuickExamFD(vector<const Alignment *>& test_read_stack, vector<vector<int> >& flow_disruptive_code)
{
	PersistingThreadObjects& thread_objects = *thread_objects_;
	VariantCallerContext& vc = *vc_;
	my_ensemble_->read_stack.swap(test_read_stack);
	my_ensemble_->SpliceAllelesIntoReads(thread_objects, *vc.global_context, *vc.parameters, *vc.ref_reader);
	my_ensemble_->allele_eval.total_theory.SetIsMolecularTag(false);
	// Calculate flow-disruptiveness in the read level
	my_ensemble_->FlowDisruptivenessInReadLevel(*vc.global_context);
	flow_disruptive_code.resize(my_ensemble_->read_stack.size());
	vector<vector<int> >::iterator flow_disruptive_code_it = flow_disruptive_code.begin();
	int num_hyp_not_null = my_ensemble_->allele_identity_vector.size() + 1;
	for (vector<CrossHypotheses>::iterator read_it = my_ensemble_->allele_eval.total_theory.my_hypotheses.begin(); read_it != my_ensemble_->allele_eval.total_theory.my_hypotheses.end(); ++read_it, ++flow_disruptive_code_it){
		flow_disruptive_code_it->assign(num_hyp_not_null, -1);
		if (not read_it->success){
			continue;
		}
		for (int i_hyp = 0; i_hyp < num_hyp_not_null; ++i_hyp){
			if (read_it->same_as_null_hypothesis[i_hyp + 1]){
				flow_disruptive_code_it->at(i_hyp) = 0;
			}
			else if (read_it->local_flow_disruptiveness_matrix[0][i_hyp + 1] >= 0){
				flow_disruptive_code_it->at(i_hyp) = read_it->local_flow_disruptiveness_matrix[0][i_hyp + 1] + 1;
			}
		}
	}
	my_ensemble_->read_stack.swap(test_read_stack);
}

// Setup the candidate variants for examination
// Minimal requirements of candidate_variant:
// a) candidate_variant.variant.sequenceName
// b) candidate_variant.variant.position
// c) candidate_variant.variant.ref
// d) candidate_variant.variant.alt
// e) candidate_variant.variant.isAltHotspot.size() == candidate_variant.variant.alt.size()
// f) candidate_variant.variant_specific_params.size() == candidate_variant.variant.alt.size()
void CandidateExaminer::SetupVariantCandidate(VariantCandidate& candidate_variant){
	if (my_ensemble_ == NULL){
		my_ensemble_ = new EnsembleEval(candidate_variant.variant);
	}
	else{
		*my_ensemble_ = EnsembleEval(candidate_variant.variant);
	}
	my_ensemble_->DEBUG = vc_->parameters->program_flow.DEBUG;
	PrepareAlleles_(candidate_variant);
}

void CandidateExaminer::ClearVariantCandidate(){
	if (my_ensemble_ != NULL){
		delete my_ensemble_;
	}
	my_ensemble_ = NULL;
}

// Allele preparing/filtering steps
void CandidateExaminer::PrepareAlleles_(VariantCandidate& candidate_variant){
	my_ensemble_->SetupAllAlleles(*(vc_->parameters), *(vc_->global_context), *(vc_->ref_reader));
	my_ensemble_->FilterAllAlleles(vc_->parameters->my_controls, candidate_variant.variant_specific_params); // put filtering here in case we want to skip below entries
}

// Given the candidate alleles, determine the maximally possible split of the variant (or group of the alternative alleles) that can be correctly (i.e., w/o high FXX) evaluated by the evaluator.
// e.g. output: allele_groups = {{0, 1, 2}, {3, 4}, {5}}. Then alt[0], alt[1], alt[2] must be evaluated jointly; alt[3], alt[4] must be evaluated jointly; alt[5] can be evaluated individually.
void CandidateExaminer::SplitCandidateVariant(list<list<int> >& allele_groups){
	my_ensemble_->SplitMyAlleleIdentityVector(allele_groups, *(vc_->ref_reader), max_group_size_allowed_);
}

// Given the variant candidates, calculate the end of the look ahead window for candidate generator,
// where (0-based) look ahead window = [last seen position + 1, look_ahead_end_0)
// I.e., the candidate generator should make sure that there is NO other de novo variant till the (0-based) position @ (look_ahead_end_0 - 1), while a variant @ look_ahead_end_0 is fine.
int CandidateExaminer::FindLookAheadEnd0(){
	int look_ahead_end_0 = my_ensemble_->CalculateLookAheadEnd0(*(vc_->ref_reader));
	return look_ahead_end_0;
}

// return 1-based look ahead end
int CandidateExaminer::FindLookAheadEnd1(){
	return FindLookAheadEnd0() + 1;
}
