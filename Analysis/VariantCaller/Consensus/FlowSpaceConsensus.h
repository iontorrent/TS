/* Copyright (C) 2016 Thermo Fisher Scientific, Inc. All Rights Reserved */

#ifndef FLOWSPACECONSENSUS_H
#define FLOWSPACECONSENSUS_H

#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <map>
#include "InputStructures.h"
#include "BAMWalkerEngine.h"
#include "MolecularTag.h"
#include "ConsensusParameters.h"

using namespace std;

// ==============================================================================
// Pre-computed information for FlowSpaceCluster
struct PrecomputeForClustering
{
	PrecomputeForClustering(const Alignment* rai);
	vector<int> zc = {};                         //! bug-fixed zc tag of the read
	string base_seq = "";                        //! base sequence as called by BaseCaller
	vector <int> flowgram = {};                  //! flowgram of base_seq, will be computed if needed
	vector <int> base_index_to_flow_index = {};  //! the index mapping from base_seq to flowgram, will be computed if needed
};

// ==============================================================================

struct TrimInfo
{
	string prefix_bases = "";    //! the 5' hard-clipped prefix bases (KS + ZT + ZE)
	string suffix_bases = "";    //! the 3' hard-clipped suffix bases (YE + YT), 3' adapter not included
	int prefix_end_flow = -1;    //! Flow corresponding to to the last base of the 5' hard clipped prefix
    int prefix_end_hp = -1;	     //! Last hp length of prefix_bases
	int suffix_start_flow = -1;  //! The first incorporated flow of the suffix_bases. Set to the end flow of template if no suffix_bases.
    int suffix_start_hp = -1;    //! First hp length of suffix_bases
};

// ==============================================================================

//@TODO: optimize the heuristic assigned parameters kMaxNumDiffFlows_, kMaxNumLocalDiffFlows_, kStrictFlowSync_
class FlowSpaceCluster
{
private:
	const static int kMaxNumDiffFlows_ = 6;          //! Build-in parameter for bool IsFlowSynchronized(...)
	const static int kMaxNumLocalDiffFlows_ = 2;     //! Build-in parameter for bool IsFlowSynchronized(...)
	const static bool kStrictFlowSync_ = false;      //! Build-in parameter for bool IsFlowSynchronized(...)
	const ion::FlowOrder* const flow_order_;         //! the flow order corresponds to run_id
	string template_base_seq_ = "";                  //! The template base sequence that represents the cluster
	vector<int> template_flowgram_ = {};             //! The flowgram of template_base_seq_

	void InitializeCluster_(const Alignment *new_rai);
	void AddNewMember_(const Alignment* new_rai, PrecomputeForClustering& precomp_for_new_rai);

public:
	const string runid;                             //! Every read must come from the same run
	const short flow_order_index;                   //! flow order index used by of the cluster
	const bool is_reverse_strand;                   //! The reads in the cluster must be on the same strand.
	int adapter_start_flow = -1;                    //! The ZG tag (or ZC[0]) of the read in Ion BAM format
    int best_mapping_quality_index = -1;            //! The index of the read in cluster_members that has the best mapping quality
	uint16_t best_mapping_quality = 0;              //! The best mapping quality of the reads in cluster_members
	TrimInfo trim_info;                             //! Information for trimming prefix bases and suffix bases
	vector<const Alignment*> cluster_members = {};  //! the read members in the cluster

	FlowSpaceCluster(const Alignment* new_rai, const ion::FlowOrder* fo);  // Call me to construct an object if you want to ignore flow_order_index
	FlowSpaceCluster(const Alignment* new_rai, const vector<ion::FlowOrder>& flow_order_vector, PrecomputeForClustering& precomp_for_new_rai);

	bool AskToJoinMe(const Alignment* query_rai);
	bool AskToJoinMe(const Alignment* query_rai, PrecomputeForClustering& precomp_for_query_rai);
};

// ==============================================================================

class FlowSpaceConsensusMaster
{
private:
	short flow_order_index_;                           //! The flow order index associated with the reads that form the consensus
	const ion::FlowOrder* flow_order_;                 //! Flow order specified by flow_order_index_
	BasecallerRead consensus_read_;                    //! Consensus flow space information
	PersistingThreadObjects* const thread_objects_;    //! Used to call Treephaser and the realigner
	const ReferenceReader* const reference_reader_;    //! Reference reader for realignemnt
	const InputStructures* const global_context_;      //! Header information of the bam file
	bool is_suffix_counted_in_za_;                     //! true if the ZA tag counts the suffix bases (BaseCaller 5.2-0, ..., 5.2-20), else false.

public:
	bool filter_qt_reads  = false;                     //! filtered out quality trimmed reads
	bool need_3_end_adapter  = false;                  //! filtered out the reads w/o 3' end adapter found
	bool suppress_recalibration = true;                //! suppress recalibration when solving the consensus measurements
	bool consensus_for_molecular_tag = false;          //! consensus for molecular tags?
	bool filter_single_read_consensus = false;         //! filtered out single-read consensus
	FlowSpaceConsensusMaster(PersistingThreadObjects* thread_objects,
			const ReferenceReader* reference_reader,
			const InputStructures* global_context,
			const string& basecaller_ver);

	void PropagateFlowspaceConsensusParameters(const ConsensusParameters& my_param, bool use_mol_tag);
	void CalculateConsensusPhaseParams(const vector<const Alignment*>& cluster_members, vector<float>& consensus_phase_params, bool is_zero_droop = true);
	void GetMeasurements(const vector<const Alignment*>& cluster_members, vector<const vector <float> *>& flow_synchronized_measurements_ptr);
	void GetMeasurements(const vector<const Alignment*>& cluster_members, vector< vector <float> >& flow_synchronized_measurements);
	void InitializeForBaseCalling(const FlowSpaceCluster& my_cluster, const vector<float>& consensus_phase_params, bool suppress_recal, string &consensus_read_name);
	void InitializeConsensusCounter();
	bool CalculateConsensusMeasurements(const vector<vector <float> >& flow_synchronized_measurements,
									 const TrimInfo& trim_info,
									 vector<float>& consensus_measurements,
									 vector<float>& measurements_sd);
	bool CalculateConsensusMeasurements(const vector<const vector <float> *>& flow_synchronized_measurements_ptr,
									 const TrimInfo& trim_info,
									 vector<float>& consensus_measurements,
									 vector<float>& measurements_sd);
	void TrimPrefixSuffixBases(const TrimInfo& trim_info, string& trimmed_consensus_bases, int& start_flow);
	bool SaveToBamAlignment(const FlowSpaceCluster& my_cluster,
			const vector<float>& consensus_phase_params,
			const vector<float>& consensus_measurements,
			const vector<float>& measurements_sd,
			const string& trimmed_bases,
			const string& consensus_read_name,
			int start_flow,
			BamAlignment& alignment);
	unsigned int FlowSpaceConsensusOneFamily(vector<Alignment *>& family_members,
			list<PositionInProgress>::iterator& consensus_position_ticket,
			list<PositionInProgress>::iterator& aln_needed_consensus_position_ticket);
};

// ==============================================================================

void GenerateFlowSpaceConsensusPositionTicket(vector< vector< vector<MolecularFamily> > >& my_molecular_families_multisample,
                                     FlowSpaceConsensusMaster& flow_space_consensus_master,
									 unsigned int min_family_size,
                                     list<PositionInProgress>::iterator& consensus_position_ticket,
                                     list<PositionInProgress>::iterator& aln_needed_consensus_position_ticket,
									 TargetsManager* targets_manager,
									 bool skip_consensus);

#endif /* FLOWSPACECONSENSUS_H */
