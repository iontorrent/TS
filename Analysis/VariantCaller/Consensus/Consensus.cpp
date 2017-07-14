/* Copyright (C) 2016 Thermo Fisher Scientific, All Rights Reserved */

//! @file     Consensus.cpp
//! @ingroup  Consensus
//! @brief    Generate flowspace consensus bam file

#include <string>
#include <vector>
#include <stdio.h>
#include <pthread.h>
#include <armadillo>

#include "ConsensusParameters.h"
#include "FlowSpaceConsensus.h"
#include "InputStructures.h"
#include "ReferenceReader.h"
#include "OrderedVCFWriter.h"
#include "BAMWalkerEngine.h"
#include "SampleManager.h"
#include "ExtendedReadInfo.h"
#include "TargetsManager.h"
#include "HotspotReader.h"
#include "MetricsManager.h"
#include "IonVersion.h"
#include "MolecularTag.h"
#include "IndelAssembly/IndelAssembly.h"

#include "json/json.h"
#include "VcfFormat.h"


using namespace std;

// TODO: get rid of global variables!
pthread_mutex_t mutexbam;

// ----------------------------------------------------------------
namespace consensus{
void TheSilenceOfTheArmadillos(ofstream &null_ostream) {
	// Disable armadillo warning messages.
	arma::set_stream_err1(null_ostream);
	arma::set_stream_err2(null_ostream);
}
};
// ----------------------------------------------------------------

struct ConsensusContext : VariantCallerContext {
	ConsensusBAMWalkerEngine* consensus_bam_walker;
	ConsensusParameters* consensus_parameters;
	pthread_mutex_t read_filter_mutex;
};

// ----------------------------------------------------------------
//! @brief    Write startup info to json structure.
void DumpStartingStateOfConsensus (int argc, char *argv[], time_t analysis_start_time, Json::Value &json)
{
  char my_host_name[128] = { 0 };
  gethostname (my_host_name, 128);
  string command_line = argv[0];
  for (int i = 1; i < argc; i++) {
    command_line += " ";
    command_line += argv[i];
  }

  json["host_name"]    = my_host_name;
  json["start_time"]   = tvc_get_time_iso_string(analysis_start_time);
  json["version"]      = IonVersion::GetVersion() + "." + IonVersion::GetRelease();
  json["git_hash"]     = IonVersion::GetGitHash();
  json["build_number"] = IonVersion::GetBuildNum();
  json["command_line"] = command_line;
}

// ----------------------------------------------------------------

void SaveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    cerr << "Unable to write JSON file " << filename_json;
}
// ----------------------------------------------------------------

void * FlowSpaceConsensusWorker(void *input);

// ----------------------------------------------------------------
int ConsensusMain(int argc, char* argv[]) {
	pthread_mutex_init(&mutexbam, NULL);

	printf("consensus %s-%s (%s): Generate a consensus bam file by re-basecalling reads that are flow-synchronized\n\n",
	       IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());

	// stolen from "Analysis" to silence error messages from Armadillo library
	ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
	consensus::TheSilenceOfTheArmadillos(null_ostream);

	time_t program_start_time = time(NULL);
	Json::Value consensus_json(Json::objectValue);
	DumpStartingStateOfConsensus (argc, argv, program_start_time, consensus_json["consensus"]);

	// Initialize parameters
	ConsensusParameters parameters(argc, argv);

	mkdir(parameters.outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    parameters.program_flow.do_indel_assembly = false;

	// Initialize reference reader
	ReferenceReader ref_reader;
	ref_reader.Initialize(parameters.fasta);

	// Initialize targets manager
	TargetsManager targets_manager;
	targets_manager.Initialize(ref_reader, parameters.targets, parameters.min_cov_fraction, parameters.trim_ampliseq_primers);

	// Initialize consens bam walker engine
	ConsensusBAMWalkerEngine bam_walker;
	try {
		bam_walker.Initialize(ref_reader, targets_manager, parameters.bams, parameters.postprocessed_bam, parameters.prefixExclusion, parameters.consensus_bam);
		bam_walker.GetProgramVersions(parameters.basecaller_version, parameters.tmap_version);
	}
	catch(...){
		cerr << "Cannot open input-bam. " << endl;
		exit(1);
	}

	// Initialize sample manager
	SampleManager sample_manager;
	sample_manager.Initialize(bam_walker.GetBamHeader(), parameters.sampleName, parameters.force_sample_name, parameters.multisample);
	// No multi-sample analysis yet.

	// Initialize global_context
	InputStructures global_context;
	global_context.Initialize(parameters, ref_reader, bam_walker.GetBamHeader());

	// Initialize tag trimmer
	MolecularTagTrimmer tag_trimmer;
	tag_trimmer.InitializeFromSamHeader(parameters.tag_trimmer_parameters, bam_walker.GetBamHeader());

	cout << "consensus: Generating flow-space consensus bam "<< (tag_trimmer.HaveTags()? "with" : "without") << " molecular tags."<< endl;
	if (parameters.skip_consensus){
		cout << "consensus: Skip the calculation of consensus. Output targets_depth.txt only." << endl;
	}

	// Initialize molecular tag manager
	MolecularTagManager mol_tag_manager;
	mol_tag_manager.Initialize(&tag_trimmer, &sample_manager);


	OrderedVCFWriter vcf_writer;

	OrderedBAMWriter bam_writer;

	HotspotReader hotspot_reader;

	// set up producer of variants
	AlleleParser candidate_generator(parameters, ref_reader, sample_manager, vcf_writer, hotspot_reader);

	ConsensusContext vc;
	vc.ref_reader = &ref_reader;
	vc.targets_manager = &targets_manager;
	vc.bam_walker = &bam_walker;
	vc.consensus_bam_walker = &bam_walker;
	vc.parameters = &parameters;
	vc.consensus_parameters = &parameters;
	vc.global_context = &global_context;
	vc.candidate_generator = &candidate_generator;
	vc.vcf_writer = NULL;
	vc.bam_writer = &bam_writer;
	vc.metrics_manager = NULL;
	vc.sample_manager = &sample_manager;
	vc.mol_tag_manager = &mol_tag_manager;

	pthread_mutex_init(&vc.candidate_generation_mutex, NULL);
	pthread_mutex_init(&vc.read_loading_mutex, NULL);
	pthread_mutex_init(&vc.bam_walker_mutex, NULL);
	pthread_mutex_init(&vc.read_removal_mutex, NULL);
	pthread_mutex_init(&vc.read_filter_mutex, NULL);
	pthread_cond_init(&vc.memory_contention_cond, NULL);
	pthread_cond_init(&vc.alignment_tail_cond, NULL);

	//vc.bam_walker->openDepth(parameters.outputDir + "/depth.txt");
	vc.candidate_counter = 0;
	vc.dot_time = time(NULL) + 30;
	//vc.candidate_dot = 0;

	pthread_t worker_id[parameters.program_flow.nThreads];
	for (int worker = 0; worker < parameters.program_flow.nThreads; worker++){
		if (pthread_create(&worker_id[worker], NULL, FlowSpaceConsensusWorker, &vc)) {
			printf("*Error* - problem starting thread\n");
			exit(-1);
		}
	}
	for (int worker = 0; worker < parameters.program_flow.nThreads; worker++)
	{ pthread_join(worker_id[worker], NULL); }

	pthread_mutex_destroy(&vc.candidate_generation_mutex);
	pthread_mutex_destroy(&vc.read_loading_mutex);
	pthread_mutex_destroy(&vc.bam_walker_mutex);
	pthread_mutex_destroy(&vc.read_removal_mutex);
	pthread_mutex_destroy(&vc.read_filter_mutex);

	pthread_cond_destroy(&vc.memory_contention_cond);
	pthread_cond_destroy(&vc.alignment_tail_cond);

	vector<MergedTarget>::iterator depth_target = targets_manager.merged.begin();
	Alignment* save_list = vc.bam_writer->process_new_entries(vc.bam_walker->alignments_first_);
	//vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
	vc.bam_walker->FinishReadRemovalTask(save_list, -1);
	save_list = vc.bam_writer->flush();
	//vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
	vc.bam_walker->FinishReadRemovalTask(save_list, -1);

	vcf_writer.Close();
	bam_walker.Close();

	// write the target coverage txt
	vc.targets_manager->WriteTargetsCoverage(parameters.outputDir + "/targets_depth.txt" , *vc.ref_reader);

	pthread_mutex_destroy(&mutexbam);

    // Determine the most frequent tmap program group for alignment
	if (not parameters.skip_consensus){
      SamProgram most_popular_tmap_pg;
      if (not bam_walker.GetMostPopularTmap(most_popular_tmap_pg)){
          cerr <<"ERROR: Fail to get tmap program group from the input bam header!" << endl;
          exit(-1);
      }
	  consensus_json["tmap"]["command_line"] = most_popular_tmap_pg.CommandLine;
	}

    time_t program_end_time = time(NULL);
    consensus_json["consensus"]["end_time"] = tvc_get_time_iso_string(program_end_time);
    consensus_json["consensus"]["total_duration"] = (Json::Int)difftime(program_end_time,program_start_time);

	cerr << endl;
	cout << endl;
	cout << "[consensus] Processing time: " << (program_end_time-program_start_time) << " seconds." << endl << endl;

	SaveJson(consensus_json, parameters.outputDir + "/consensus.json");
	return 0;
}

// ----------------------------------------------------------------

void * FlowSpaceConsensusWorker(void *input) {
	ConsensusContext& vc = *static_cast<ConsensusContext*>(input);

	vector<MergedTarget>::iterator indel_target = vc.targets_manager->merged.begin();
	vector<MergedTarget>::iterator depth_target = vc.targets_manager->merged.begin();

	const static int kReadBatchSize = 40;
	Alignment * new_read[kReadBatchSize];
	bool success[kReadBatchSize];
	PersistingThreadObjects  thread_objects(*vc.global_context);

	list<PositionInProgress> ticket_temp_1(1), ticket_temp_2(1);

	// This is the consensus reads to be saved, no realignment needed.
	list<PositionInProgress>::iterator consensus_position_ticket = ticket_temp_1.begin();
	consensus_position_ticket->begin = NULL;
	consensus_position_ticket->end = NULL;
	// This is the consensus reads to be saved, realignment needed
	list<PositionInProgress>::iterator aln_needed_consensus_position_ticket = ticket_temp_2.begin();
	aln_needed_consensus_position_ticket->begin = NULL;
	aln_needed_consensus_position_ticket->end = NULL;

	list<PositionInProgress>::iterator target_ticket;

	bool more_positions = true;
	const bool use_molecular_tag = vc.mol_tag_manager->tag_trimmer->HaveTags();
    const int effective_min_family_size = use_molecular_tag? vc.parameters->tag_trimmer_parameters.min_family_size : 1;
    const int max_read_num_in_memory = use_molecular_tag? 250000 : 50000;

    int sample_num = 1;
    if (vc.consensus_parameters->multisample){
        sample_num = vc.sample_manager->num_samples_;
    }

	vector< vector< vector<MolecularFamily> > > my_molecular_families_multisample;
	MolecularFamilyGenerator my_family_generator;

	// Initialize flowspace_consensus_master
	string basecaller_ver, tmap_ver;
	vc.bam_walker->GetProgramVersions(basecaller_ver, tmap_ver);
	FlowSpaceConsensusMaster flowspace_consensus_master(&thread_objects, vc.ref_reader, vc.global_context, basecaller_ver);

	// Propagate parameters
	flowspace_consensus_master.PropagateFlowspaceConsensusParameters(*(vc.consensus_parameters), use_molecular_tag);
	if (not vc.consensus_parameters->skip_consensus){
    	flowspace_consensus_master.InitializeConsensusCounter(); // If not initialized, then I won't see the stat if no family is generated.
	}
	while (true /*more_positions*/) {

		// Opportunistic read removal
		// Note: The I need to make sure that I am not attempt to remove the reads in progress.
		pthread_mutex_lock(&vc.bam_walker_mutex);
		if (vc.consensus_bam_walker->EligibleForTargetBasedReadRemoval()) {
			if (pthread_mutex_trylock(&vc.read_removal_mutex) == 0) {
				Alignment *removal_list = NULL;
				vc.consensus_bam_walker->RequestTargetBasedReadRemovalTask(removal_list);
				pthread_mutex_unlock(&vc.bam_walker_mutex);
				//In rare case, the Eligible check pass, but another thread got to remove reads, then when this thread get the lock, it find there
				//is no reads to remove. The unexpected behavior of SaveAlignment() is that when NULL is passed in, it save all the reads and remove
				// ZM tags. To prevent that, we need to check for empty.
				if (removal_list) {
					Alignment* save_list = vc.bam_writer->process_new_entries(removal_list);
					//vc.bam_walker->SaveAlignments(save_list, vc, depth_target);
					pthread_mutex_lock(&vc.bam_walker_mutex);
					vc.bam_walker->FinishReadRemovalTask(save_list, max_read_num_in_memory + 5000);
					pthread_mutex_unlock(&vc.bam_walker_mutex);
				}
				pthread_mutex_unlock(&vc.read_removal_mutex);
				pthread_cond_broadcast(&vc.memory_contention_cond);
			}else{
				pthread_mutex_unlock(&vc.bam_walker_mutex);
			}
		}
		else{
			pthread_mutex_unlock(&vc.bam_walker_mutex);
		}

		// If too many reads in memory and at least one candidate evaluator in progress, pause this thread
		// Wake up when the oldest candidate evaluator task is completed.

		pthread_mutex_lock(&vc.bam_walker_mutex);
		if (vc.bam_walker->MemoryContention(max_read_num_in_memory)) {
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
			if (true/*pthread_mutex_trylock(&vc.candidate_generation_mutex) == 0*/) {
				pthread_mutex_lock(&vc.bam_walker_mutex);
				ready_for_next_position = vc.bam_walker->ReadyForNextPosition();
				if (not ready_for_next_position) {
					pthread_mutex_unlock(&vc.bam_walker_mutex);
//					pthread_mutex_unlock(&vc.candidate_generation_mutex);
				}
			}

		} else {
			// Greedy reading disallowed: if candidate generation in progress,
			// wait for it to finish before deciding what to do.
//			pthread_mutex_lock(&vc.candidate_generation_mutex);
			pthread_mutex_lock(&vc.bam_walker_mutex);
			ready_for_next_position = vc.bam_walker->ReadyForNextPosition();
			if (not ready_for_next_position) {
				pthread_mutex_unlock(&vc.bam_walker_mutex);
//				pthread_mutex_unlock(&vc.candidate_generation_mutex);
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
				{ break; }
			}
			pthread_mutex_unlock(&vc.read_loading_mutex);


			for (int i = 0; i < kReadBatchSize and success[i]; ++i) {
				// 1) Filling in read body information and do initial set of read filters
				if (not vc.candidate_generator->BasicFilters(*new_read[i]))
				{ continue; }

				// 2) Alignment information altering methods (can also filter a read)
				if (not vc.mol_tag_manager->tag_trimmer->GetTagsFromBamAlignment(new_read[i]->alignment, new_read[i]->tag_info)) {
					new_read[i]->filtered = true;
					continue;
				}

		        // Filter by read mismatch limit (note: NOT the filter for read-max-mismatch-fraction) here.
				// Note: Target override did not apply for consensus.
		        FilterByModifiedMismatches(new_read[i], vc.consensus_parameters->read_mismatch_limit, NULL);
		        if (new_read[i]->filtered)
		          continue;

				// 3) Filter by target
				// These two variables should be parameters of consensus.
				vc.targets_manager->FilterReadByRegion(new_read[i], vc.bam_walker->GetRecentUnmergedTarget());
				if (new_read[i]->filtered)
				{ continue; }
				// 4) Unpacking read meta data for flow-space consensus
				UnpackOnLoadLight(new_read[i], *vc.global_context);
				// 5) Count how many reads that use the program for alignment.
				vc.consensus_bam_walker->AddReadToPG(new_read[i]);
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
//			pthread_mutex_unlock(&vc.candidate_generation_mutex);
			pthread_cond_broadcast(&vc.memory_contention_cond);
			break;
		}

		pthread_mutex_unlock(&vc.bam_walker_mutex);
		pthread_mutex_lock(&vc.bam_walker_mutex);

		// I do consensus only if I travel to the position at the end of the target.
		bool do_consensus = vc.bam_walker->getPosition() == (vc.bam_walker->getEndPosition() - 1);

		if (not do_consensus){
			more_positions = vc.bam_walker->AdvancePosition(1);
			pthread_mutex_unlock(&vc.bam_walker_mutex);
			continue;
		}

		// Generating the target ticket that consists of the reads that cover the target.
		vc.consensus_bam_walker->BeginTargetProcessingTask(target_ticket);

		// The target_ticket has been generated. Now I can move to the next position (in fact, next target) to let other threads keep working.
		// But keep in mind, don't let other thread remove the reads that I am processing.
		more_positions = vc.bam_walker->AdvancePosition(1);
		pthread_mutex_unlock(&vc.bam_walker_mutex);

		// Now I am going to gather the reads to form consensus reads.
		my_molecular_families_multisample.resize(sample_num);

		// I need to make sure that every read won't be processed twice.
		// i.e., no read can appear in two or more consensus reads.
		// So I use read_filter_mutex to prevent any other thread is using the same read to generate another consensus read.
		pthread_mutex_lock(&vc.read_filter_mutex);
		// Generate families if I am using molecular tags.
		if (use_molecular_tag){
			for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
				int overloaded_sample_index = vc.consensus_parameters->multisample ? sample_index : -1;
				my_family_generator.GenerateMyMolecularFamilies(vc.mol_tag_manager, *target_ticket, overloaded_sample_index, my_molecular_families_multisample[sample_index]);
				// Label all the reads being processed here as filtered, so they won't be use again.
				for (vector< vector< MolecularFamily> >::iterator strand_it = my_molecular_families_multisample[sample_index].begin(); strand_it != my_molecular_families_multisample[sample_index].end(); ++strand_it) {
					for (vector< MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it !=  strand_it->end(); ++fam_it) {
						for (vector< Alignment *>::iterator read_it = (fam_it->all_family_members).begin(); read_it != (fam_it->all_family_members).end(); ++read_it) {
							(*read_it)->filtered = true;
						}
					}
				}
			}
		}
		// Generate "families" if I am NOT using molecular tags.
		else{
			for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
				// If not using molecular tag, all the reads on the strand form a family.
				// i.e., I will create 2 huge families, families of reads on the FWD/REV strands, respectively.
				my_molecular_families_multisample[sample_index].resize(2);
				for (int strand = 0; strand < 2; ++strand){
					if (my_molecular_families_multisample[sample_index][strand].empty()){
						string barcode_name = (strand == 0) ? "All_FWD_READS" : "All_REV_READS";
						my_molecular_families_multisample[sample_index][strand].push_back(MolecularFamily(barcode_name, strand));
						my_molecular_families_multisample[sample_index][strand][0].all_family_members.reserve(4096);
					}
					// I reuse all_family_members.
					my_molecular_families_multisample[sample_index][strand][0].all_family_members.resize(0);
				}
			}
			for (Alignment* rai = target_ticket->begin; rai != target_ticket->end; rai = rai->next) {
				if (rai == NULL) {
					target_ticket->end = NULL;
					break;
				}
				if (rai->filtered) {
					continue;
				}
				int strand_key =  (rai->is_reverse_strand)? 1 : 0;
				my_molecular_families_multisample[rai->sample_index][strand_key][0].AddNewMember(rai);
				// Label all the reads being processed here as filtered, so they won't be use again.
				rai->filtered = true;
			}
		}
		pthread_mutex_unlock(&vc.read_filter_mutex);

		// Now I can generate consensus reads from the families.
		GenerateFlowSpaceConsensusPositionTicket(my_molecular_families_multisample,
				flowspace_consensus_master,
				(unsigned int) effective_min_family_size,
				consensus_position_ticket,
				aln_needed_consensus_position_ticket,
				vc.targets_manager,
				vc.consensus_parameters->skip_consensus);

		if (not vc.consensus_parameters->skip_consensus){
			// Save consensus_position_ticket and aln_needed_consensus_position_ticket to the bam files
			vc.consensus_bam_walker->SaveConsensusAlignments(consensus_position_ticket->begin, aln_needed_consensus_position_ticket->begin);

			// delete consensus_position_ticket
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
			// delete aln_needed_consensus_position_ticket
			while ((aln_needed_consensus_position_ticket->begin != NULL) and (aln_needed_consensus_position_ticket->begin != aln_needed_consensus_position_ticket->end)) {
				Alignment* p = aln_needed_consensus_position_ticket->begin;
				aln_needed_consensus_position_ticket->begin = aln_needed_consensus_position_ticket->begin->next;
				delete p;
			}
			if (aln_needed_consensus_position_ticket->begin != NULL) {
				delete aln_needed_consensus_position_ticket->begin;
				aln_needed_consensus_position_ticket->begin = NULL;
				aln_needed_consensus_position_ticket->end = NULL;
			}
		}
//		pthread_mutex_unlock(&vc.candidate_generation_mutex);
		pthread_mutex_lock(&vc.bam_walker_mutex);
		bool signal_contention_removal = vc.bam_walker->IsEarlierstPositionProcessingTask(target_ticket);
		vc.bam_walker->FinishPositionProcessingTask(target_ticket);
		pthread_mutex_unlock(&vc.bam_walker_mutex);

		if (signal_contention_removal)
		{ pthread_cond_broadcast(&vc.memory_contention_cond); }

		if (time(NULL) > vc.dot_time) {
	    	cerr << '.';
	        vc.dot_time += 30;
	    }
	}
	return NULL;
}

