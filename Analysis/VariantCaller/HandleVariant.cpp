/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     HandleVariant.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "HandleVariant.h"



//void EnsembleProcessOneVariant(BamMultiReader * bamReader, const string & local_contig_sequence,
void EnsembleProcessOneVariant(PersistingThreadObjects &thread_objects, vcf::Variant ** candidate_variant,
                               ExtendParameters * parameters, InputStructures &global_context) {;

  EnsembleEval my_ensemble;
  my_ensemble.multi_allele_var.SetupAllAlleles(candidate_variant, thread_objects.local_contig_sequence, parameters, global_context);
  my_ensemble.multi_allele_var.FilterAllAlleles( parameters->my_controls.filter_variant); // put filtering here in case we want to skip below entries
  //my_ensemble.allele_eval.resize(my_ensemble.multi_allele_var.allele_identity_vector.size());
  my_ensemble.SetupHypothesisChecks(parameters);

  // We read in one stack per multi-allele variant
  my_ensemble.my_data.StackUpOneVariant(thread_objects, my_ensemble.multi_allele_var.window_start,
                             my_ensemble.multi_allele_var.window_end, candidate_variant, parameters, global_context);

  // glue in variants
  if (!my_ensemble.my_data.no_coverage) {
    int best_allele = TrySolveAllAllelesVsRef(my_ensemble, thread_objects, global_context);

    // output to variant
    GlueOutputVariant(my_ensemble, parameters, best_allele);

    // test diagnostic output for this ensemble
    if (parameters->program_flow.rich_json_diagnostic & (!((*(my_ensemble.multi_allele_var.variant))->isFiltered) | parameters->program_flow.skipCandidateGeneration | (*my_ensemble.multi_allele_var.variant)->isHotSpot)) // look at everything that came through
      JustOneDiagnosis(my_ensemble, parameters->program_flow.json_plot_dir, true);
    if (parameters->program_flow.minimal_diagnostic & (!((*(my_ensemble.multi_allele_var.variant))->isFiltered) | parameters->program_flow.skipCandidateGeneration | (*my_ensemble.multi_allele_var.variant)->isHotSpot)) // look at everything that came through
      JustOneDiagnosis(my_ensemble, parameters->program_flow.json_plot_dir, false);

  } else {
    AutoFailTheCandidate(my_ensemble.multi_allele_var.variant, parameters->my_controls.suppress_no_calls);
  }
  // Pointer to candidate variant should still point to the same element as my_ensemble.multi_allele_var.variant
}


//void DoWorkForOneVariant(BamTools::BamMultiReader &bamReader, vcf::Variant **current_variant,  string &local_contig_sequence , ExtendParameters *parameters, InputStructures *global_context_ptr) {
void DoWorkForOneVariant(PersistingThreadObjects &thread_objects, vcf::Variant **current_variant,  ExtendParameters *parameters, InputStructures *global_context_ptr) {

  global_context_ptr->ShiftLocalBamReaderToCorrectBamPosition(thread_objects.bamMultiReader, current_variant);

    EnsembleProcessOneVariant(thread_objects, current_variant, parameters,  *global_context_ptr);
}

