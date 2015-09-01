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

#include "IonVersion.h"

using namespace std;


void TheSilenceOfTheArmadillos(ofstream &null_ostream)
{
  // Disable armadillo warning messages.
  arma::set_stream_err1(null_ostream);
  arma::set_stream_err2(null_ostream);
}

void * VariantCallerWorker(void *input);


int main(int argc, char* argv[])
{

  printf("tvc %s-%s (%s) - Torrent Variant Caller\n\n",
         IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());

  // stolen from "Analysis" to silence error messages from Armadillo library
  ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
  TheSilenceOfTheArmadillos(null_ostream);

  time_t start_time = time(NULL);


  ExtendParameters parameters(argc, argv);


  mkdir(parameters.outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  if (parameters.program_flow.rich_json_diagnostic || parameters.program_flow.minimal_diagnostic) {
    // make output directory "side effect bad"
    parameters.program_flow.json_plot_dir = parameters.outputDir + "/json_diagnostic/";
    mkdir(parameters.program_flow.json_plot_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }



  ReferenceReader ref_reader;
  ref_reader.Initialize(parameters.fasta);

  TargetsManager targets_manager;
  targets_manager.Initialize(ref_reader, parameters.targets, parameters.trim_ampliseq_primers);

  BAMWalkerEngine bam_walker;
  bam_walker.Initialize(ref_reader, targets_manager, parameters.bams, parameters.postprocessed_bam, parameters.prefixExclusion);
  bam_walker.GetProgramVersions(parameters.basecaller_version, parameters.tmap_version);

  SampleManager sample_manager;
  sample_manager.Initialize(bam_walker.GetBamHeader(), parameters.sampleName, parameters.force_sample_name);

  InputStructures global_context;
  global_context.Initialize(parameters, ref_reader, bam_walker.GetBamHeader());

  OrderedVCFWriter vcf_writer;
  vcf_writer.Initialize(parameters.outputDir + "/" + parameters.outputFile, parameters, sample_manager);

  OrderedBAMWriter bam_writer;

  HotspotReader hotspot_reader;
  hotspot_reader.Initialize(ref_reader, parameters.variantPriorsFile);

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
  pthread_mutex_init(&vc.candidate_generation_mutex, NULL);
  pthread_mutex_init(&vc.read_loading_mutex, NULL);
  pthread_mutex_init(&vc.bam_walker_mutex, NULL);
  pthread_mutex_init(&vc.read_removal_mutex, NULL);
  pthread_cond_init(&vc.memory_contention_cond, NULL);
  pthread_cond_init(&vc.alignment_tail_cond, NULL);

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


  Alignment* save_list = vc.bam_writer->process_new_etries(vc.bam_walker->alignments_first_);
  vc.bam_walker->SaveAlignments(save_list);
  save_list = vc.bam_writer->flush();
  vc.bam_walker->SaveAlignments(save_list);

  vcf_writer.Close();
  bam_walker.Close();
  metrics_manager.FinalizeAndSave(parameters.outputDir + "/tvc_metrics.json");

  cerr << endl;
  cout << endl;
  cout << "[tvc] Normal termination. Processing time: " << (time(NULL)-start_time) << " seconds." << endl;

  return 0;
}






void * VariantCallerWorker(void *input)
{
  VariantCallerContext& vc = *static_cast<VariantCallerContext*>(input);

  deque<VariantCandidate> variant_candidates;
  const static int kReadBatchSize = 40;
  Alignment * new_read[kReadBatchSize];
  bool success[kReadBatchSize];
  list<PositionInProgress>::iterator position_ticket;
  PersistingThreadObjects  thread_objects(*vc.global_context);
  bool more_positions = true;

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
          Alignment* save_list = vc.bam_writer->process_new_etries(removal_list);
          vc.bam_walker->SaveAlignments(save_list);
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
        break;
      }

      pthread_mutex_lock(&vc.bam_walker_mutex);
      for (int i = 0; i < kReadBatchSize; ++i) {
        vc.bam_walker->RequestReadProcessingTask(new_read[i]);
        success[i] = false;
      }
      pthread_mutex_unlock(&vc.bam_walker_mutex);

      for (int i = 0; i < kReadBatchSize; ++i) {
        success[i] = vc.bam_walker->GetNextAlignmentCore(new_read[i]);
        if (not success[i])
          break;
      }
      pthread_mutex_unlock(&vc.read_loading_mutex);


      for (int i = 0; i < kReadBatchSize and success[i]; ++i) {
        vc.candidate_generator->BasicFilters(*new_read[i]);
        if (new_read[i]->filtered)
          continue;
        vc.targets_manager->TrimAmpliseqPrimers(new_read[i], vc.bam_walker->GetRecentUnmergedTarget());
        if (new_read[i]->filtered)
          continue;

        vc.candidate_generator->RegisterAlignment(*new_read[i]);
        UnpackOnLoad(new_read[i], *vc.global_context, *vc.parameters);
      }

      pthread_mutex_lock(&vc.bam_walker_mutex);
      for (int i = 0; i < kReadBatchSize; ++i)
        vc.bam_walker->FinishReadProcessingTask(new_read[i], success[i]);
      pthread_mutex_unlock(&vc.bam_walker_mutex);

      continue;
    }

    //
    // Dispatch outcome: Generate candidates at next position
    //


    if (not vc.bam_walker->HasMoreAlignments() and not more_positions) {
      pthread_mutex_unlock(&vc.bam_walker_mutex);
      pthread_mutex_unlock(&vc.candidate_generation_mutex);
      break;
    }

    vc.bam_walker->BeginPositionProcessingTask(position_ticket);
    pthread_mutex_unlock(&vc.bam_walker_mutex);

    int haplotype_length = 1;
    vc.candidate_generator->GenerateCandidates(variant_candidates, position_ticket, haplotype_length);

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
        EnsembleProcessOneVariant(thread_objects, vc, *v, *position_ticket);
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
      //pthread_cond_broadcast(&vc.memory_contention_cond);
      pthread_cond_signal(&vc.memory_contention_cond);


  }
  return NULL;
}





