/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "EnsembleGlue.h"


void BootUpAllAlleles(EnsembleEval &my_ensemble, PersistingThreadObjects &thread_objects, InputStructures &global_context) {

    int num_hyp_no_null=0; // guarantee problems if the following does not happen
    num_hyp_no_null = my_ensemble.multi_allele_var.allele_identity_vector.size()+1; // num alleles +1 for ref

    // build a unique identifier to write out diagnostics
    my_ensemble.allele_eval.variant_position = (*(my_ensemble.multi_allele_var.variant))->position;
    my_ensemble.allele_eval.ref_allele = (*(my_ensemble.multi_allele_var.variant))->ref;
    my_ensemble.allele_eval.var_allele = (*(my_ensemble.multi_allele_var.variant))->alt.at(0);
    for (unsigned int i_allele=1; i_allele<(*(my_ensemble.multi_allele_var.variant))->alt.size(); i_allele++) {
    	my_ensemble.allele_eval.var_allele += ',';
    	my_ensemble.allele_eval.var_allele += (*(my_ensemble.multi_allele_var.variant))->alt.at(i_allele);
    }
    // unique identifier built

    my_ensemble.allele_eval.total_theory.my_hypotheses.resize(my_ensemble.my_data.read_stack.size());
    // generate null+ref+nr.alt hypotheses per read in the case of do_multiallele_eval
    //int my_allele_index = -1;


    for (unsigned int i_read = 0; i_read < my_ensemble.allele_eval.total_theory.my_hypotheses.size(); i_read++) {
      // --- New splicing function ---
      my_ensemble.allele_eval.total_theory.my_hypotheses.at(i_read).success =
        SpliceVariantHypotheses(my_ensemble.my_data.read_stack.at(i_read),
                                  my_ensemble.multi_allele_var,
                                  my_ensemble.multi_allele_var.seq_context,
                                  thread_objects,
                                  my_ensemble.allele_eval.total_theory.my_hypotheses.at(i_read).splice_start_flow,
                                  my_ensemble.allele_eval.total_theory.my_hypotheses.at(i_read).splice_end_flow,
                                  my_ensemble.allele_eval.total_theory.my_hypotheses.at(i_read).instance_of_read_by_state,
                                  global_context);

      // if we need to compare likelihoods across multiple possibilities
      if (num_hyp_no_null>2)
        my_ensemble.allele_eval.total_theory.my_hypotheses.at(i_read).use_correlated_likelihood = false;

    }

    if (global_context.DEBUG>1) cout << "Finished splicing all alleles!" << endl; // XXX
    // fill in quantities derived from predictions

    my_ensemble.allele_eval.InitForInference(thread_objects, my_ensemble.my_data, global_context, num_hyp_no_null);


  if (global_context.DEBUG>1) cout << "All alleles booted up." << endl; // XXX
}


// handle the unfortunate case in which we must try multiple alleles to be happy
// try only ref vs alt allele here
// leave ensemble in ref vs alt state
int TrySolveAllAllelesVsRef(EnsembleEval &my_ensemble, PersistingThreadObjects &thread_objects, InputStructures &global_context) {

  BootUpAllAlleles(my_ensemble, thread_objects, global_context);

  // do inference
  my_ensemble.ExecuteInferenceAllAlleles();
  if (global_context.DEBUG>1) cout << "Inference finished." << endl; // XXX
  // now we're in the guaranteed state of best index
  return(my_ensemble.DetectBestAllele());
}

void EnsembleSummaryStats(EnsembleEval &my_ensemble, DecisionTreeData &my_decision) {
  // demonstrate this item
  vector<int> read_id;
  vector<bool> strand_id;
  // pretend we can classify reads across multiple alleles
  my_ensemble.ApproximateHardClassifierForReads(read_id, strand_id);
  my_decision.all_summary_stats.DigestHardClassifiedReads(strand_id, read_id);
}

void SummarizeInfoFieldsFromEnsemble(EnsembleEval &my_ensemble, vcf::Variant ** candidate_variant, int _cur_allele_index) {

  float mean_ll_delta;

  my_ensemble.ScanSupportingEvidence(mean_ll_delta, _cur_allele_index);

  (*candidate_variant)->info["MLLD"].push_back(convertToString(mean_ll_delta));

  float radius_bias, fwd_bias, rev_bias, ref_bias,var_bias;
  int fwd_strand = 0;
   int rev_strand = 1;

   int var_hyp = 1;

    // get bias terms from cur_allele within a single multi-allele structure

    var_hyp = _cur_allele_index+1;
    radius_bias = my_ensemble.allele_eval.cur_state.bias_generator.RadiusOfBias(_cur_allele_index);
    fwd_bias = my_ensemble.allele_eval.cur_state.bias_generator.latent_bias.at(fwd_strand).at(_cur_allele_index);
    rev_bias = my_ensemble.allele_eval.cur_state.bias_generator.latent_bias.at(rev_strand).at(_cur_allele_index);
    //@TODO: note the disconnect in indexing; inconsistent betwen objects
    ref_bias = my_ensemble.allele_eval.cur_state.bias_checker.ref_bias_v.at(var_hyp);
     var_bias = my_ensemble.allele_eval.cur_state.bias_checker.variant_bias_v.at(var_hyp);

  (*candidate_variant)->info["RBI"].push_back(convertToString(radius_bias));
// this is by strand
  (*candidate_variant)->info["FWDB"].push_back(convertToString(fwd_bias));
  (*candidate_variant)->info["REVB"].push_back(convertToString(rev_bias));
  // this is by hypothesis
  (*candidate_variant)->info["REFB"].push_back(convertToString(ref_bias));
  (*candidate_variant)->info["VARB"].push_back(convertToString(var_bias));
  
}




void GlueOutputVariant(EnsembleEval &my_ensemble, ExtendParameters *parameters, int _best_allele_index) {

  DecisionTreeData my_decision;
  my_decision.tune_sbias = parameters->my_controls.sbias_tune;

  my_decision.SetupFromMultiAllele(my_ensemble.multi_allele_var);

  EnsembleSummaryStats(my_ensemble, my_decision);

  float smallest_allele_freq = 1.0f;
  for (unsigned int _alt_allele_index = 0; _alt_allele_index < my_decision.multi_allele.allele_identity_vector.size(); _alt_allele_index++) {
    // for each alt allele, do my best
    // thresholds here can vary by >type< of allele
    float local_min_allele_freq = FreqThresholdByType(my_ensemble.multi_allele_var.allele_identity_vector.at(_alt_allele_index), parameters->my_controls);
    if (local_min_allele_freq<smallest_allele_freq)
      smallest_allele_freq = local_min_allele_freq;  // choose least-restrictive amongst multialleles

      my_ensemble.ComputePosteriorGenotype(_alt_allele_index, local_min_allele_freq,
            my_decision.summary_info_vector.at(_alt_allele_index).genotype_call,
            my_decision.summary_info_vector.at(_alt_allele_index).gt_quality_score,
            my_decision.summary_info_vector.at(_alt_allele_index).variant_qual_score);
    
    SummarizeInfoFieldsFromEnsemble(my_ensemble, my_ensemble.multi_allele_var.variant, _alt_allele_index);
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
                                    my_decision.eval_genotype.evaluated_variant_quality);
    my_decision.eval_genotype.genotype_already_set = true; // because we computed it here
    // and I must also set for each allele
    // so that the per-allele  filter works
    for (unsigned int i_allele =0; i_allele<my_decision.multi_allele.allele_identity_vector.size(); i_allele++){
      my_decision.summary_info_vector.at(i_allele).variant_qual_score = my_decision.eval_genotype.evaluated_variant_quality;
    }


// now that all the data has been gathered describing the variant, combine to produce the output
  my_decision.DecisionTreeOutputToVariant(my_ensemble.multi_allele_var.variant, parameters);

}

