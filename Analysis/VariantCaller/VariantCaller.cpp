/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantCaller.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include <string>
#include <vector>
#include <stdio.h>
#include <pthread.h>
#include <armadillo>

#include "HypothesisEvaluator.h"
#include "ExtendParameters.h"

#include "InputStructures.h"
#include "HandleVariant.h"
#include "ReferenceReader.h"
#include "OrderedVCFWriter.h"
#include "BAMWalkerEngine.h"
#include "SampleManager.h"
#include "ExtendedReadInfo.h"
#include "TargetsManager.h"
#include "HotspotReader.h"
#include "MetricsManager.h"
#include "DecisionTreeData.h"

#include "IonVersion.h"

#include <boost/math/distributions/poisson.hpp>
#include "tvcutils/viterbi.h"
#include "tvcutils/unify_vcf.h"

#include "IndelAssembly/IndelAssembly.h"
#include "MolecularTag.h"
#include "Consensus.h"

using namespace std;


void TheSilenceOfTheArmadillos(ofstream &null_ostream)
{
  // Disable armadillo warning messages.
  arma::set_stream_err1(null_ostream);
  arma::set_stream_err2(null_ostream);
}

void * VariantCallerWorker(void *input);

// --------------------------------------------------------------------------------------------------------------
// The tvc exectuable is currently overloaded and harbors "tvc consensus" within
// Below function is the classic tvc main function


static int main_tvc(int argc, char* argv[])
{
  printf("tvc %s-%s (%s) - Torrent Variant Caller\n\n",
         IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());

  // stolen from "Analysis" to silence error messages from Armadillo library
  ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
  TheSilenceOfTheArmadillos(null_ostream);

  time_t start_time = time(NULL);

  // Read parameters and create output directories
  ExtendParameters parameters(argc, argv);

  ReferenceReader ref_reader;
  ref_reader.Initialize(parameters.fasta);

  TargetsManager targets_manager;
  targets_manager.Initialize(ref_reader, parameters.targets, parameters.min_cov_fraction, parameters.trim_ampliseq_primers);

  BAMWalkerEngine bam_walker;
  bam_walker.Initialize(ref_reader, targets_manager, parameters.bams, parameters.postprocessed_bam, parameters.prefixExclusion);
  bam_walker.GetProgramVersions(parameters.basecaller_version, parameters.tmap_version);

  SampleManager sample_manager;
  sample_manager.Initialize(bam_walker.GetBamHeader(), parameters.sampleName, parameters.force_sample_name, parameters.multisample);

  InputStructures global_context;
  global_context.Initialize(parameters, ref_reader, bam_walker.GetBamHeader());

  MolecularTagTrimmer tag_trimmer;
  tag_trimmer.InitializeFromSamHeader(parameters.tag_trimmer_parameters, bam_walker.GetBamHeader());

  MolecularTagManager mol_tag_manager;
  mol_tag_manager.Initialize(&tag_trimmer, &sample_manager);

  if (tag_trimmer.HaveTags()){
	  cout << "TVC: Call small variants with molecular tags. Indel Assembly will be turned off automatically." << endl;
	  parameters.program_flow.do_indel_assembly = false;
  }

  OrderedVCFWriter vcf_writer;
  vcf_writer.Initialize(parameters.outputDir + "/" + parameters.small_variants_vcf, parameters, ref_reader, sample_manager, tag_trimmer.HaveTags());

  OrderedBAMWriter bam_writer;

  HotspotReader hotspot_reader;
  hotspot_reader.Initialize(ref_reader, parameters.variantPriorsFile);
  if (!parameters.blacklistFile.empty()) {
    if (parameters.variantPriorsFile.empty()) {hotspot_reader.Initialize(ref_reader);}
    hotspot_reader.MakeHintQueue(parameters.blacklistFile);
  }
  string parameters_file = parameters.opts.GetFirstString('-', "parameters-file", "");

  IndelAssemblyArgs parsed_opts;
  parsed_opts.setReference(parameters.fasta);
  parsed_opts.setBams(parameters.bams);
  parsed_opts.setTargetFile(parameters.targets);
  parsed_opts.setOutputVcf(parameters.outputDir + "/" + parameters.indel_assembly_vcf);
  parsed_opts.setParametersFile(parameters_file);
  // Weird behavior of assembly where an empty sample name enables multi-sample analysis
  parsed_opts.setSampleName(parameters.multisample ? "" : sample_manager.primary_sample_name_);
  // Print the indel_assembly parameters if do_indel_assembly = true
  if(parameters.program_flow.do_indel_assembly){
	  cout << "TVC: Parsing Indel Assembly parameters." << endl;
	  parsed_opts.processParameters(parameters.opts);
  }
  else{
	  cout<<"TVC: Indel Assembly off."<< endl;
  }

  IndelAssembly indel_assembly(&parsed_opts, &ref_reader, &sample_manager, &targets_manager);
  
  // set up producer of variants
  AlleleParser candidate_generator(parameters, ref_reader, sample_manager, vcf_writer, hotspot_reader);

  MetricsManager metrics_manager;

  VariantCallerContext vc;
  vc.ref_reader = &ref_reader;
  vc.targets_manager = &targets_manager;
  vc.bam_walker = &bam_walker;
  vc.parameters = &parameters;
  vc.global_context = &global_context;
  vc.candidate_generator = &candidate_generator;
  vc.vcf_writer = &vcf_writer;
  vc.bam_writer = &bam_writer;
  vc.metrics_manager = &metrics_manager;
  vc.sample_manager  = &sample_manager;
  vc.indel_assembly = &indel_assembly;
  vc.mol_tag_manager = &mol_tag_manager;

  pthread_mutex_init(&vc.candidate_generation_mutex, NULL);
  pthread_mutex_init(&vc.read_loading_mutex, NULL);
  pthread_mutex_init(&vc.bam_walker_mutex, NULL);
  pthread_mutex_init(&vc.read_removal_mutex, NULL);
  pthread_cond_init(&vc.memory_contention_cond, NULL);
  pthread_cond_init(&vc.alignment_tail_cond, NULL);

  vc.bam_walker->openDepth(parameters.outputDir + "/depth.txt");

  vc.candidate_counter = 0;
  vc.dot_time = time(NULL) + 30;
  //vc.candidate_dot = 0;

  pthread_t worker_id[parameters.program_flow.nThreads];
  for (int worker = 0; worker < parameters.program_flow.nThreads; worker++)
    if (pthread_create(&worker_id[worker], NULL, VariantCallerWorker, &vc)) {
      printf("*Error* - problem starting thread\n");
      exit(-1);
    }

  for (int worker = 0; worker < parameters.program_flow.nThreads; worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_mutex_destroy(&vc.candidate_generation_mutex);
  pthread_mutex_destroy(&vc.read_loading_mutex);
  pthread_mutex_destroy(&vc.bam_walker_mutex);
  pthread_mutex_destroy(&vc.read_removal_mutex);
  pthread_cond_destroy(&vc.memory_contention_cond);
  pthread_cond_destroy(&vc.alignment_tail_cond);

  vector<MergedTarget>::iterator depth_target = targets_manager.merged.begin();
  Alignment* save_list = vc.bam_writer->process_new_entries(vc.bam_walker->alignments_first_);
  vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
  vc.bam_walker->FinishReadRemovalTask(save_list, -1);
  save_list = vc.bam_writer->flush();
  //vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
  vc.bam_walker->FinishReadRemovalTask(save_list, -1);

  vc.bam_walker->closeDepth(vc.ref_reader);
  
  vcf_writer.Close();
  bam_walker.Close();
  metrics_manager.FinalizeAndSave(parameters.outputDir + "/tvc_metrics.json");

  cerr << endl;
  cout << endl;
  time_t indel_start_time = time(NULL);
  cout << "[tvc] Finished small variants. Processing time: " << (indel_start_time-start_time) << " seconds." << endl;

  // --- Indel Assembly
  
  indel_assembly.onTraversalDone(vc.parameters->program_flow.do_indel_assembly);
  cout << "[tvc] Finished indel assembly. Processing time: " << (time(NULL)-indel_start_time) << " seconds." << endl;

  // --- VCF post processing & merging steps
  /*/ XXX Not being done in TVC 5.4 but in the variant caller pipeline
  
  time_t post_proc_start = time(NULL);

  string small_variants_vcf = parameters.outputDir + "/" + parameters.small_variants_vcf;
  string indel_assembly_vcf = parameters.outputDir + "/" + parameters.indel_assembly_vcf;
  string hotspots_file      = parameters.variantPriorsFile;
  string merged_vcf         = parameters.outputDir + "/" + parameters.merged_vcf;
  string tvc_metrics        = parameters.outputDir + "/tvc_metrics.json"; // Name hard coded in metrics manager FinalizeAndSave call
  string input_depth        = parameters.outputDir + "/depth.txt"; // Name hard coded in BAM walker initialization
  string output_genome_vcf  = parameters.outputDir + "/" + parameters.merged_genome_vcf;

  // VCF merging & post processing filters / subset annotation
  VcfOrderedMerger merger(small_variants_vcf, indel_assembly_vcf, hotspots_file, merged_vcf, tvc_metrics, input_depth,
        output_genome_vcf, ref_reader, targets_manager, 10, max(0, parameters.minCoverage), true);
  merger.SetVCFrecordFilters(parameters.my_controls.filter_by_target, parameters.my_controls.hotspot_positions_only, parameters.my_controls.hotspot_variants_only);
  merger.perform();

  build_index(merged_vcf);
  build_index(output_genome_vcf);

  cout << "[tvc] Finished vcf post processing. Processing time: " << (time(NULL)-post_proc_start) << " seconds." << endl;
  // */

  cerr << endl;
  cout << endl;
  cout << "[tvc] Total Processing time: " << (time(NULL)-start_time) << " seconds." << endl;
  return EXIT_SUCCESS;
}

// --------------------------------------------------------------------------------------------------------------
// The tvc exectuable is currently overloaded and harbors "tvc consensus"

int main(int argc,  char* argv[])
{
   if (argc > 1){
       if (string(argv[1]) == "consensus"){
           return ConsensusMain(argc - 1, argv + 1);
       }
   }
   return main_tvc(argc, argv);
}

// --------------------------------------------------------------------------------------------------------------

void * VariantCallerWorker(void *input)
{
  BamAlignment alignment;
  VariantCallerContext& vc = *static_cast<VariantCallerContext*>(input);

  vector<MergedTarget>::iterator indel_target = vc.targets_manager->merged.begin();
  vector<MergedTarget>::iterator depth_target = vc.targets_manager->merged.begin();

  deque<VariantCandidate> variant_candidates;
  const static int kReadBatchSize = 40;
  Alignment * new_read[kReadBatchSize];
  bool success[kReadBatchSize];
  list<PositionInProgress>::iterator position_ticket;
  int prev_positioin_ticket_begin = -1;  // position_ticket.begin at the previous while loop
  int prev_positioin_ticket_end = -1;    // position_ticket.end at the previous while loop
  PersistingThreadObjects  thread_objects(*vc.global_context);

  Consensus consensus;
  consensus.SetFlowConsensus(false);
  list<PositionInProgress> consensus_position_temp(1);
  list<PositionInProgress>::iterator consensus_position_ticket = consensus_position_temp.begin();
  consensus_position_ticket->begin = NULL;
  consensus_position_ticket->end = NULL;

  CandidateExaminer my_examiner(&thread_objects, &vc);

  bool more_positions = true;
  // Molecular tagging related stuffs
  const bool use_molecular_tag = vc.mol_tag_manager->tag_trimmer->HaveTags();
  vector< vector< vector<MolecularFamily> > > my_molecular_families_multisample;
  MolecularFamilyGenerator my_family_generator;

  pthread_mutex_lock(&vc.bam_walker_mutex);
  MetricsAccumulator& metrics_accumulator = vc.metrics_manager->NewAccumulator();
  pthread_mutex_unlock(&vc.bam_walker_mutex);

  while (true /*more_positions*/) {
	  // Opportunistic read removal

    if (vc.bam_walker->EligibleForReadRemoval()) {
      if (pthread_mutex_trylock(&vc.read_removal_mutex) == 0) {

        Alignment *removal_list = NULL;
        pthread_mutex_lock(&vc.bam_walker_mutex);
        vc.bam_walker->RequestReadRemovalTask(removal_list);
        pthread_mutex_unlock(&vc.bam_walker_mutex);
	//In rare case, the Eligible check pass, but another thread got to remove reads, then when this thread get the lock, it find there
	//is no reads to remove. The unexpected behavior of SaveAlignment() is that when NULL is passed in, it save all the reads and remove 
	// ZM tags. To prevent that, we need to check for empty.
        if (removal_list) {
          Alignment* save_list = vc.bam_writer->process_new_entries(removal_list);
          vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
          pthread_mutex_lock(&vc.bam_walker_mutex);
          vc.bam_walker->FinishReadRemovalTask(save_list);
          pthread_mutex_unlock(&vc.bam_walker_mutex);
        }
        pthread_mutex_unlock(&vc.read_removal_mutex);

        pthread_cond_broadcast(&vc.memory_contention_cond);
      }
    }

    // If too many reads in memory and at least one candidate evaluator in progress, pause this thread
    // Wake up when the oldest candidate evaluator task is completed.

    pthread_mutex_lock(&vc.bam_walker_mutex);
    if (vc.bam_walker->MemoryContention()) {
      pthread_cond_wait (&vc.memory_contention_cond, &vc.bam_walker_mutex);
      pthread_mutex_unlock(&vc.bam_walker_mutex);
      continue;
    }
    pthread_mutex_unlock(&vc.bam_walker_mutex);


    //
    // Task dispatch: Decide whether to load more reads or to generate more variants
    //

    bool ready_for_next_position = false;
    if (vc.bam_walker->EligibleForGreedyRead()) {
      // Greedy reading allowed: if candidate generation in progress, just grab a new read
      if (pthread_mutex_trylock(&vc.candidate_generation_mutex) == 0) {
        pthread_mutex_lock(&vc.bam_walker_mutex);
        ready_for_next_position = vc.bam_walker->ReadyForNextPosition();
        if (not ready_for_next_position) {
          pthread_mutex_unlock(&vc.bam_walker_mutex);
          pthread_mutex_unlock(&vc.candidate_generation_mutex);
        }
      }

    } else {
      // Greedy reading disallowed: if candidate generation in progress,
      // wait for it to finish before deciding what to do.
      pthread_mutex_lock(&vc.candidate_generation_mutex);
      pthread_mutex_lock(&vc.bam_walker_mutex);
      ready_for_next_position = vc.bam_walker->ReadyForNextPosition();
      if (not ready_for_next_position) {
        pthread_mutex_unlock(&vc.bam_walker_mutex);
        pthread_mutex_unlock(&vc.candidate_generation_mutex);
      }
    }

    //
    // Dispatch outcome: Load and process more reads
    //

    if (not ready_for_next_position) {


      pthread_mutex_lock(&vc.read_loading_mutex);

      if (not vc.bam_walker->HasMoreAlignments()) {
        pthread_mutex_unlock(&vc.read_loading_mutex);
        pthread_cond_broadcast(&vc.memory_contention_cond);
        break;
      }

      pthread_mutex_lock(&vc.bam_walker_mutex);
      for (int i = 0; i < kReadBatchSize; ++i) {
        vc.bam_walker->RequestReadProcessingTask(new_read[i]);
        success[i] = false;
      }
      pthread_mutex_unlock(&vc.bam_walker_mutex);

      for (int i = 0; i < kReadBatchSize; ++i) {
        success[i] = vc.bam_walker->GetNextAlignmentCore(new_read[i], vc, indel_target);
        if (not success[i])
          break;
      }
      pthread_mutex_unlock(&vc.read_loading_mutex);

      bool more_read_come = false;
      for (int i = 0; i < kReadBatchSize and success[i]; ++i) {
        // 1) Filling in read body information and do initial set of read filters
        if (not vc.candidate_generator->BasicFilters(*new_read[i]))
          continue;

        // 2) Alignment information altering methods (can also filter a read)
        if (not vc.mol_tag_manager->tag_trimmer->GetTagsFromBamAlignment(new_read[i]->alignment, new_read[i]->tag_info)){
          new_read[i]->filtered = true;
          continue;
        }

        // 3) Filter by target and Trim ampliseq primer here
        vc.targets_manager->TrimAmpliseqPrimers(new_read[i], vc.bam_walker->GetRecentUnmergedTarget());
        if (new_read[i]->filtered)
          continue;


        // 4) Filter by read mismatch limit (note: NOT the filter for read-max-mismatch-fraction) here.
        FilterByModifiedMismatches(new_read[i], vc.parameters->read_mismatch_limit, vc.targets_manager);
        if (new_read[i]->filtered)
          continue;

        // 5) Parsing alignment: Read filtering & populating allele specific data types in Alignment object
        vc.candidate_generator->UnpackReadAlleles(*new_read[i]);

        // 6) Unpacking read meta data for evaluator
        UnpackOnLoad(new_read[i], *vc.global_context);
      }

      pthread_mutex_lock(&vc.bam_walker_mutex);
      for (int i = 0; i < kReadBatchSize; ++i) {
        vc.bam_walker->FinishReadProcessingTask(new_read[i], success[i]);
      }
      pthread_mutex_unlock(&vc.bam_walker_mutex);

      continue;
    }

    //
    // Dispatch outcome: Generate candidates at next position
    //


    if (not vc.bam_walker->HasMoreAlignments() and not more_positions) {
      pthread_mutex_unlock(&vc.bam_walker_mutex);
      pthread_mutex_unlock(&vc.candidate_generation_mutex);
      pthread_cond_broadcast(&vc.memory_contention_cond);
      break;
    }

    vc.bam_walker->BeginPositionProcessingTask(position_ticket);
    pthread_mutex_unlock(&vc.bam_walker_mutex);

    if(use_molecular_tag){
        int sample_num = 1;
        if (vc.parameters->multisample){
            sample_num = vc.sample_manager->num_samples_;
        }
        // No need to generate families if position_ticket is not changed.
        bool is_position_ticket_changed = true;
        if (position_ticket->begin != NULL and position_ticket->end != NULL){
        	is_position_ticket_changed = (prev_positioin_ticket_begin != position_ticket->begin->read_number) or (prev_positioin_ticket_end != position_ticket->end->read_number);
        }

        if (is_position_ticket_changed){
    		for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
    			int overloaded_sample_index = vc.parameters->multisample ? sample_index : -1;
    	        my_molecular_families_multisample.resize(sample_num);
    			my_family_generator.GenerateMyMolecularFamilies(vc.mol_tag_manager, *position_ticket, overloaded_sample_index, my_molecular_families_multisample[sample_index]);
    		}
    		GenerateConsensusPositionTicket(my_molecular_families_multisample, vc, consensus, consensus_position_ticket);
        }
        if (position_ticket->begin != NULL and position_ticket->end != NULL){
        	prev_positioin_ticket_begin = position_ticket->begin->read_number;
        	prev_positioin_ticket_end = position_ticket->end->read_number;
        }else{
        	prev_positioin_ticket_begin = -1;
        	prev_positioin_ticket_end = -1;
        }
    }

    // Candidate Generation
    int haplotype_length = 1;
    if(consensus_position_ticket->begin != NULL){
  		for (Alignment *p = consensus_position_ticket->begin; p; p = p->next) {
            vc.candidate_generator->UnpackReadAlleles(*p);
        }
        vc.bam_walker->SetupPositionTicket(consensus_position_ticket);
        vc.candidate_generator->GenerateCandidates(variant_candidates, consensus_position_ticket, haplotype_length);
        while ((consensus_position_ticket->begin != NULL) and (consensus_position_ticket->begin != consensus_position_ticket->end)) {
            Alignment* p = consensus_position_ticket->begin;
            consensus_position_ticket->begin = consensus_position_ticket->begin->next;
            delete p;
        }
		if (consensus_position_ticket->begin != NULL) {
			delete consensus_position_ticket->begin;
			consensus_position_ticket->begin = NULL;
			consensus_position_ticket->end = NULL;
		}
    }
    else{
    	vc.candidate_generator->GenerateCandidates(variant_candidates, position_ticket, haplotype_length, &my_examiner);
    }

    pthread_mutex_lock(&vc.bam_walker_mutex);
    int next_hotspot_chr = -1;
    long next_hotspot_position = -1;
    if (vc.candidate_generator->GetNextHotspotLocation(next_hotspot_chr, next_hotspot_position))
      more_positions = vc.bam_walker->AdvancePosition(haplotype_length, next_hotspot_chr, next_hotspot_position);
    else
      more_positions = vc.bam_walker->AdvancePosition(haplotype_length);
    pthread_mutex_unlock(&vc.bam_walker_mutex);

    if (not variant_candidates.empty()) {

      int vcf_writer_slot = vc.vcf_writer->ReserveSlot();
      vc.candidate_counter += variant_candidates.size();
      //while (vc.candidate_counter > vc.candidate_dot) {
      //  cerr << ".";
      //  /*
      //  pthread_mutex_lock(&vc.bam_walker_mutex);
      //  vc.bam_walker->PrintStatus();
      //  pthread_mutex_unlock(&vc.bam_walker_mutex);
      //  */
      //  vc.candidate_dot += 50;
      //}
      if (time(NULL) > vc.dot_time) {
        cerr << '.';
        vc.dot_time += 30;
      }

      pthread_mutex_unlock(&vc.candidate_generation_mutex);
      // separate queuing of variants from >actual work< of calling variants
      for (deque<VariantCandidate>::iterator v = variant_candidates.begin(); v != variant_candidates.end(); ++v) {
        if (vc.parameters->multisample) {
          bool pass = false; // if pass == false the no reads for the candidate
          bool filter = true;
          int evaluator_error_code = 0;
          for (int sample_index = 0; (sample_index < vc.sample_manager->num_samples_); ++sample_index) {
        	  evaluator_error_code = EnsembleProcessOneVariant(thread_objects, vc, *v, *position_ticket, my_molecular_families_multisample[sample_index], sample_index);
        	  pass = evaluator_error_code == 0;
        	//TODO: czb: the logic here is a mess. Need clean-up!
            if (!v->variant.isFiltered) {filter = false;}
            v->variant.isFiltered = false;
          }
          if (filter) {
            v->variant.filter = "NOCALL";
            v->variant.isFiltered = true;
          }
          else {
            v->variant.filter = "PASS";
            v->variant.isFiltered = false;
          }
          if (!pass) {
            for (int sample_index = 0; (sample_index < vc.sample_manager->num_samples_); ++sample_index) {
              string my_reason = evaluator_error_code == 2? "NOVALIDFUNCFAM" : "NODATA";
              AutoFailTheCandidate(v->variant, vc.parameters->my_controls.use_position_bias, v->variant.sampleNames[sample_index], use_molecular_tag, my_reason);
            }
          }
        }
        else {
          // We only call based on the reads of the primary sample if  a sample name was provided
          int evaluator_error_code = EnsembleProcessOneVariant(thread_objects, vc, *v, *position_ticket, my_molecular_families_multisample[0], vc.sample_manager->primary_sample_);
          if (evaluator_error_code != 0) {
            string my_reason = evaluator_error_code == 2? "NOVALIDFUNCFAM" : "NODATA";
            AutoFailTheCandidate(v->variant, vc.parameters->my_controls.use_position_bias, v->variant.sampleNames[0], use_molecular_tag, my_reason);
          }
        }
        //v->isFiltered = true;
      }

      vc.vcf_writer->WriteSlot(vcf_writer_slot, variant_candidates);

      variant_candidates.clear();
    } else {
      pthread_mutex_unlock(&vc.candidate_generation_mutex);
    }

    metrics_accumulator.CollectMetrics(position_ticket, haplotype_length, vc.ref_reader);

    pthread_mutex_lock(&vc.bam_walker_mutex);
    bool signal_contention_removal = vc.bam_walker->IsEarlierstPositionProcessingTask(position_ticket);
    vc.bam_walker->FinishPositionProcessingTask(position_ticket);
    pthread_mutex_unlock(&vc.bam_walker_mutex);

    if (signal_contention_removal)
      pthread_cond_broadcast(&vc.memory_contention_cond);

  }
  return NULL;
}





