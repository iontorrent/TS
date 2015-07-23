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




void SummarizeInfoFieldsFromEnsemble(EnsembleEval &my_ensemble, vcf::Variant &candidate_variant, int _cur_allele_index) {

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




void GlueOutputVariant(EnsembleEval &my_ensemble, VariantCandidate &candidate_variant, const ExtendParameters &parameters, int _best_allele_index)
{

  DecisionTreeData my_decision(*(my_ensemble.variant));
  my_decision.tune_sbias = parameters.my_controls.sbias_tune;

  my_decision.SetupFromMultiAllele(my_ensemble);

  // EnsembleSummaryStats(my_ensemble, my_decision);
  // demonstrate this item
  vector<int> read_id;
  vector<bool> strand_id;
  vector<int> dist_to_left;
  vector<int> dist_to_right;
  // pretend we can classify reads across multiple alleles
  my_ensemble.ApproximateHardClassifierForReads(read_id, strand_id, dist_to_left, dist_to_right);
  my_decision.all_summary_stats.AssignStrandToHardClassifiedReads(strand_id, read_id);
  my_decision.all_summary_stats.AssignPositionFromEndToHardClassifiedReads(read_id, dist_to_left, dist_to_right);


  float smallest_allele_freq = 1.0f;
  for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.allele_identity_vector.size(); _alt_allele_index++) {
    // for each alt allele, do my best
    // thresholds here can vary by >type< of allele
    float local_min_allele_freq = FreqThresholdByType(my_ensemble.allele_identity_vector[_alt_allele_index], parameters.my_controls,
        candidate_variant.variant_specific_params[_alt_allele_index]);
    if (local_min_allele_freq<smallest_allele_freq)
      smallest_allele_freq = local_min_allele_freq;  // choose least-restrictive amongst multialleles

      my_ensemble.ComputePosteriorGenotype(_alt_allele_index, local_min_allele_freq,
            my_decision.summary_info_vector[_alt_allele_index].genotype_call,
            my_decision.summary_info_vector[_alt_allele_index].gt_quality_score,
            my_decision.summary_info_vector[_alt_allele_index].variant_qual_score);

    SummarizeInfoFieldsFromEnsemble(my_ensemble, *(my_ensemble.variant), _alt_allele_index);
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
                                  my_decision.eval_genotype.evaluated_variant_quality, parameters.my_eval_control.max_detail_level);
  my_decision.eval_genotype.genotype_already_set = true; // because we computed it here
  // and I must also set for each allele
  // so that the per-allele  filter works
  for (unsigned int i_allele =0; i_allele<my_decision.allele_identity_vector.size(); i_allele++){
    my_decision.summary_info_vector[i_allele].variant_qual_score = my_decision.eval_genotype.evaluated_variant_quality;
  }


  // now that all the data has been gathered describing the variant, combine to produce the output
  my_decision.DecisionTreeOutputToVariant(candidate_variant, parameters);

}



// Read and process records appropriate for this variant; positions are zero based
void EnsembleEval::StackUpOneVariant(const ExtendParameters &parameters, const PositionInProgress& bam_position)
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

    if (rai->filtered or rai->evaluator_filtered)
      continue;

    if (rai->alignment.GetEndPosition() < multiallele_window_end)
      continue;

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




void EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, VariantCallerContext& vc,
    VariantCandidate &candidate_variant, const PositionInProgress& bam_position)
{
  int chr_idx = vc.ref_reader->chr_idx(candidate_variant.variant.sequenceName.c_str());

  EnsembleEval my_ensemble(candidate_variant.variant);
  my_ensemble.SetupAllAlleles(*vc.parameters, *vc.global_context, *vc.ref_reader, chr_idx);
  my_ensemble.FilterAllAlleles(vc.parameters->my_controls.filter_variant, candidate_variant.variant_specific_params); // put filtering here in case we want to skip below entries

  // We read in one stack per multi-allele variant
  my_ensemble.StackUpOneVariant(*vc.parameters, bam_position);

  if (my_ensemble.read_stack.empty()) {
    cerr << "Nonfatal: No reads found for " << candidate_variant.variant.sequenceName << "\t" << my_ensemble.multiallele_window_start << endl;
    AutoFailTheCandidate(candidate_variant.variant, vc.parameters->my_controls.use_position_bias);
    return;
  }


  // handle the unfortunate case in which we must try multiple alleles to be happy
  // try only ref vs alt allele here
  // leave ensemble in ref vs alt state

  // glue in variants
  my_ensemble.SpliceAllelesIntoReads(thread_objects, *vc.global_context, *vc.parameters, *vc.ref_reader, chr_idx);

  my_ensemble.allele_eval.my_params = vc.parameters->my_eval_control;

  // fill in quantities derived from predictions
  int num_hyp_no_null = my_ensemble.allele_identity_vector.size()+1; // num alleles +1 for ref
  my_ensemble.allele_eval.InitForInference(thread_objects, my_ensemble.read_stack, *vc.global_context, num_hyp_no_null);

  // do inference
  my_ensemble.allele_eval.ExecuteInference(vc.parameters->my_eval_control.max_detail_level);

  // now we're in the guaranteed state of best index
  int best_allele = my_ensemble.DetectBestMultiAllelePair();

  // output to variant
  GlueOutputVariant(my_ensemble, candidate_variant, *vc.parameters, best_allele);

  // test diagnostic output for this ensemble
  if (vc.parameters->program_flow.rich_json_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)) // look at everything that came through
    JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->program_flow.json_plot_dir, true);
  if (vc.parameters->program_flow.minimal_diagnostic & (!(my_ensemble.variant->isFiltered) | my_ensemble.variant->isHotSpot)) // look at everything that came through
    JustOneDiagnosis(my_ensemble, *vc.global_context, vc.parameters->program_flow.json_plot_dir, false);


}





