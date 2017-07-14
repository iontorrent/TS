/* Copyright (C) 2016 Thermo Fisher Scientific, Inc. All Rights Reserved */

//! @file     FlowSpaceConsensus.cpp
//! @ingroup  VariantCaller
//! @brief    Cluster flow-synchronized reads; generate consensus measurements.

#include "FlowSpaceConsensus.h"

#include <assert.h>
#include "MiscUtil.h"
#include "ExtendedReadInfo.h"

class ConsensusCounter {
private:
    pthread_mutex_t mutextConsensusCounter_;
	unsigned int num_func_fam_ = 0;
	unsigned int num_consensus_reads_ = 0;
	unsigned int num_single_read_consensus_ = 0;
	unsigned int num_reads_in_func_fam_ = 0;
	unsigned int num_reads_in_consensus_bam_ = 0;
	unsigned int num_consensus_reads_need_aln_ = 0;
    void PrintSummary_()
    {
    	if (use_molecular_tag){
			cout << "[Stat] Summary of the flow-space consensus:" << endl
				 << "  - Number of functional families identified = " <<  num_func_fam_ << endl
				 << "  - Number of reads in the functional families identified = " <<  num_reads_in_func_fam_ << endl
				 << "  - Number of consensus reads generated = " <<  num_consensus_reads_ << endl
				 << "  - Number of consensus reads realigned by tmap = " <<  num_consensus_reads_need_aln_ << endl
				 << "  - Number of reads that consists of the consensus reads = " <<  num_reads_in_consensus_bam_ << endl
				 << "  - Number of single-read consensus = " <<  num_single_read_consensus_ << endl;
			cout << "  - Compression rate = "<< num_consensus_reads_ <<"/"<< num_reads_in_consensus_bam_ <<" = ";
			if (num_reads_in_consensus_bam_ == 0){
				 cout << "NaN"<< endl;
			}else{
				 cout << (double) num_consensus_reads_ / (double) num_reads_in_consensus_bam_ << endl;
			}
			cout << "  - Average number of consensus reads per functional family = ";
			if (num_func_fam_ == 0){
				cout << "NaN" << endl << endl << endl;
			}else{
				cout << (double) num_consensus_reads_ / (double) num_func_fam_ << endl << endl << endl;
			}
    	}
    	else{
			cout << "[Stat] Summary of the flow-space consensus:" << endl
				 << "  - Number of reads processed = " <<  num_reads_in_func_fam_ << endl
				 << "  - Number of reads that consists of the consensus reads = " <<  num_reads_in_consensus_bam_ << endl
				 << "  - Number of consensus reads generated = " <<  num_consensus_reads_ << endl
				 << "  - Number of consensus reads realigned by tmap = " <<  num_consensus_reads_need_aln_ << endl
				 << "  - Number of single-read consensus = " <<  num_single_read_consensus_ << endl
				 << "  - Compression rate = "<< num_consensus_reads_ <<"/"<< num_reads_in_consensus_bam_ << " = ";
			if (num_reads_in_consensus_bam_ == 0){
				 cout << "NaN" << endl << endl << endl;
			}else{
				 cout << (double) num_consensus_reads_ / (double) num_reads_in_consensus_bam_ << endl << endl << endl;
			}
    	}
    }
public:
    ConsensusCounter(){};
	ConsensusCounter(bool is_mol_tag){ use_molecular_tag = is_mol_tag; };
	bool use_molecular_tag = false;
	virtual ~ConsensusCounter() {PrintSummary_();}
    void Count(unsigned int num_func_fam,
    			unsigned int num_consensus_reads,
				unsigned int num_single_read_consensus,
				unsigned int num_reads_in_func_fam,
				unsigned int num_reads_in_consensus_bam,
				unsigned int num_consensus_reads_need_aln)
    {
        pthread_mutex_lock(&mutextConsensusCounter_);
        num_func_fam_ += num_func_fam;
        num_consensus_reads_ += num_consensus_reads;
        num_single_read_consensus_ += num_single_read_consensus;
        num_reads_in_func_fam_ += num_reads_in_func_fam;
        num_reads_in_consensus_bam_ += num_reads_in_consensus_bam;
        num_consensus_reads_need_aln_ += num_consensus_reads_need_aln;
        pthread_mutex_unlock(&mutextConsensusCounter_);
    }
};

// Inputs: base_seq (can be vector<char> or string), flow_order (any type with operator [] defined), num_flows (num_flows is the length of flow_order)
// Outputs: flowgram, base_idx_to_flow_idx
// Convert base_seq to flowgram associated with the flow_order
// base_idx_to_flow_idx[base_idx] is the flow index where base_seq[base_idx] incorporates.
// base_idx_to_flow_idx[base_idx] is set to -1 if incorporated flow of base_seq[base_idx] can not be specified by flow_order.
template <typename BaseSeqI, typename FlowOrderI>
void BaseSeqToFlowgram (const BaseSeqI &base_seq, const FlowOrderI &flow_order, int num_flows, vector<int> &flowgram, vector<int> &base_idx_to_flow_idx)
{
    int flow_idx = 0;
    unsigned int num_bases = base_seq.size();
    unsigned int base_idx = 0;

    base_idx_to_flow_idx.assign(base_seq.size(), -1);
    flowgram.assign(num_flows, 0);

    while (flow_idx < num_flows and base_idx < num_bases) {
        while (base_idx < num_bases and flow_order[flow_idx] == base_seq.at(base_idx)) {
        	base_idx_to_flow_idx[base_idx] = flow_idx;
        	++flowgram[flow_idx];
            ++base_idx;
        }
        ++flow_idx;
    }
}

// This function is used to fix the bug in the ZC tag where it is possible to get ZC[0] != ZC[1] while flow_order[ZC[0]] = flow_order[ZC[1]]
// The bug is fixed in 5.2.1.
bool FixZcBug(vector<int> &zc, const ion::FlowOrder& flow_order)
{
	bool is_fixed = false;
	if (zc[0] != zc[1] and flow_order[zc[0]] == flow_order[zc[1]]){
		if (zc[0] < zc[1])
			zc[1] = zc[0];
		else
			zc[0] = zc[1];
		is_fixed = true;
	}
	return is_fixed;
}

// A "softer" criterion to determine x_mer == y_mer that allows some hp error.
// I claim x_mer and y_mer are "very different" if max(x_mer, y_mer) - min(x_mer, y_mer) > kHpLenDiffAllowed[max(x_mer, y_mer)]
// return true if x_mer and y_mer are "very different".
// Here kHpLenDiffAllowed is set heuristically.
//@TODO: optimize kHpLenDiffAllowed or come up with a better algorithm (e.g. K-run).
bool SofterHpDisriminator(int x_mer, int y_mer)
{
	const static vector<int> kHpLenDiffAllowed = {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8};  // I hard code the criterion.
	const static int kMaxHpDefined = (int) kHpLenDiffAllowed.size() - 1;
	return (abs(x_mer - y_mer) > kHpLenDiffAllowed[min(max(x_mer, y_mer), kMaxHpDefined)]);
}

// base_seq = (KS + ZT + ZE) + template + (YE + YT)
// Here I do not reconstruct the 3' adapter because I don't have enough information.
void ReconstructBaseAsCalled(const Alignment* rai, string &base_seq)
{
	string read_base = rai->alignment.QueryBases;
	if (rai->is_reverse_strand){
		RevComplementInPlace(read_base);
	}
	base_seq = rai->prefix_bases + read_base + rai->suffix_bases;
}

// return the first/last hp length of base_seq if is_reverse_mode = true/false, respectively.
int FindFirstHpLen(const string& base_seq, bool is_reverse_mode = false)
{
	int base_seq_size = (int) base_seq.size();
	if (base_seq_size <= 1)
		return base_seq_size;
	int first_hp_len = 1;
	int current_idx = 1;
	int increment = 1;
	char first_nuc = base_seq.at(0);
	if (is_reverse_mode){
		current_idx = base_seq_size - 2;
		increment = -1;
		first_nuc =  base_seq.at(base_seq_size - 1);
	}
	while (current_idx >= 0 and current_idx < base_seq_size){
		if (base_seq.at(current_idx) == first_nuc)
			++first_hp_len;
		else
			return first_hp_len;
		current_idx += increment;
	}
	return first_hp_len;
}

// Find the smallest index i >= start_flow such that flowgram_0[i] = flowgram_1[i] > 0
// return -1 if not found
int GetNextIncorpFlowInCommon(const vector<int>& flowgram_0, const vector<int>& flowgram_1, unsigned int start_flow)
{
	unsigned int min_len = min(flowgram_0.size(), flowgram_1.size());
	int next_incorp_flow_in_common = -1;
	for (unsigned int flow_idx = start_flow; flow_idx < min_len; ++flow_idx){
		if (flowgram_0[flow_idx] == flowgram_1[flow_idx] and flowgram_0[flow_idx] > 0){
			next_incorp_flow_in_common = flow_idx;
			return next_incorp_flow_in_common;
		}
	}
	return next_incorp_flow_in_common;
}

// Find the largest index i <= start_flow such that flowgram_0[i] = flowgram_1[i] > 0
// return -1 if not found
int GetLastIncorpFlowInCommon(const vector<int>& flowgram_0, const vector<int>& flowgram_1, unsigned int start_flow)
{
	int last_incorp_flow_in_common = -1;
	unsigned int min_len = min(flowgram_0.size(), flowgram_1.size());
	if (min_len == 0){
		return last_incorp_flow_in_common;
	}
	if (start_flow >= min_len){
		start_flow = min_len - 1;
	}
	for (unsigned int flow_idx = start_flow; flow_idx != 0; --flow_idx){
		if (flowgram_0[flow_idx] == flowgram_1[flow_idx] and flowgram_0[flow_idx] > 0){
			last_incorp_flow_in_common = flow_idx;
			return last_incorp_flow_in_common;
		}
	}
	return last_incorp_flow_in_common;
}

// Find the largest index i <= start_flow such that flowgram_0[i] = flowgram_1[i] > 0
// return -1 if not found
int GetLastIncorpFlowAlmostInCommon(const vector<int>& flowgram_0, const vector<int>& flowgram_1, unsigned int start_flow)
{
	int last_incorp_flow_in_common = -1;
	unsigned int min_len = min(flowgram_0.size(), flowgram_1.size());
	if (min_len == 0){
		return last_incorp_flow_in_common;
	}
	if (start_flow >= min_len){
		start_flow = min_len - 1;
	}
	for (unsigned int flow_idx = start_flow; flow_idx != 0; --flow_idx){
		if ((not SofterHpDisriminator(flowgram_0[flow_idx], flowgram_1[flow_idx])) and flowgram_0[flow_idx] > 0 and flowgram_1[flow_idx] > 0){
			last_incorp_flow_in_common = flow_idx;
			return last_incorp_flow_in_common;
		}
	}
	return last_incorp_flow_in_common;
}

// This function determine whether flowgram_0[0:synchronized_len] and flowgram_1[0:synchronized_len] are flow-synchronized or not by simply looking at the discrepancy between the two flowgrams,
// where synchronized_len - 1 is the is last incorporated flow in common (i.e., the largest i such that flowgram_0[i] = flowgram_1[i] > 0).
// Alignment information is not needed!
// I claim the two flowsgrams are flow-synchronized if the following conditions are all satisfied.
// a) There exists no flow s.t. one flowgram is 0-mer while the other one is 2-mer or above. (i.e., 0-mer -> (>1-mer) is claimed to be not a sequencing error.)
// b) number of flows with "discrepancy" <= max_num_diff_flows, where discrepancy is defined by SofterHpDisriminator(...)
// c) number of flows with "discrepancy" in every window <= max_num_local_diff_flows
// where a window is a closed interval [window_start, window_end] where window_start, window_end are the only two incorporated flows in common in the window.
// If strict_flow_sync = true, then I don't allow any (0-mer) <-> (>0-mer) change.
// Consider y = H*x where y, x are the vectors of prediction and flowgram, respectively, H is the phasing matrix.
// strict_flow_sync guarantees the same structure of the phasing matrix.
bool IsFlowSynchronized(const vector<int>& flowgram_0, const vector<int>& flowgram_1, unsigned int& synchronized_len, unsigned int max_num_diff_flows,  unsigned int max_num_local_diff_flows, bool strict_flow_sync)
{
 	bool is_flow_synchronized = false;
	// Get the last incorporated flow where the two flowgrams are in common.
	synchronized_len = GetLastIncorpFlowAlmostInCommon(flowgram_0, flowgram_1, flowgram_0.size() - 1) + 1;

	if (synchronized_len == 0){
		return is_flow_synchronized;
	}

	vector<unsigned int> diff_flow_index;
	int max_0_mer_diff = strict_flow_sync? 0 : 1;
	diff_flow_index.reserve(max_num_diff_flows);
	// Find the diff_flow_index from the back since we can return the function faster if they are not flow-synchronized.
	for (unsigned int flow_idx = synchronized_len - 1; flow_idx != 0; --flow_idx){
		// Is there a delta at the flow?
		bool is_diff = false;
		if (flowgram_0[flow_idx] != flowgram_1[flow_idx]){
			if (flowgram_0[flow_idx] == 0 or flowgram_1[flow_idx] == 0){
				// A non-hp indel delta found.
				// I don't allow 0-mer <-> (>1-mer).
				// I don't allow 0-mer <-> (>0-mer) if strict_flow_sync = true.
				// Claim not flow-synchronized.
				if (abs((int) flowgram_0[flow_idx] - (int) flowgram_1[flow_idx]) > max_0_mer_diff){
					synchronized_len = 0;
					return is_flow_synchronized;
				}
				is_diff = true;
			}
			// hp-indel
			else if (SofterHpDisriminator((int) flowgram_0[flow_idx], (int) flowgram_1[flow_idx])){
				// The HP lengths of the two flowgram seem very different.
				is_diff = true;
			}

			if (is_diff){
				// The two flowgrams seem very different at the flow.
				diff_flow_index.push_back(flow_idx);
				if (diff_flow_index.size() > max_num_diff_flows){
					// The flows with discrepancy is greater than max_num_diff_flows.
					// Claim not flow-synchronized.
					synchronized_len = 0;
					return is_flow_synchronized;
				}
			}
		}
	}

	if (diff_flow_index.size() <= max_num_local_diff_flows){
		// num_local_diff won't be greater than max_num_local_diff_flows
		// I claim the two flowgrams are flow-synchronized.
		is_flow_synchronized = true;
		return is_flow_synchronized;
	}

	// Define window = closed interval of [window_start, window_end]
	int window_end = GetNextIncorpFlowInCommon(flowgram_0, flowgram_1, diff_flow_index[0]);
	int window_start = window_end;
	int i_diff = 0;

	// num_local_diff = the number of flows with discrepancy in the window
	// I claim the flowgrams are not flow-synchronized if num_local_diff > max_num_local_diff_flows in any window.
	// Sliding window approach to check num_local_diff for every window with complexity = Big O(diff_flow_index.size())
	// Note that diff_flow_index "must" be in "decreasing" order!
	while (i_diff < (int) diff_flow_index.size() and window_start > -1){
		if ((int) diff_flow_index[i_diff] < window_start){
			window_end = window_start;
			window_start = GetLastIncorpFlowInCommon(flowgram_0, flowgram_1, diff_flow_index[i_diff]);
			unsigned int num_local_diff = 0;
			int j_diff = i_diff;
			while ((int) diff_flow_index[j_diff] >= window_start and j_diff < (int) diff_flow_index.size()){
				++num_local_diff;
				if (num_local_diff > max_num_local_diff_flows){
					// I find one window with num_local_diff > max_num_local_diff_flows
					synchronized_len = 0;
					return is_flow_synchronized;
				}
				++j_diff;
			}
			i_diff = j_diff;
		}
	}

	// Now I have num_local_diff <= max_num_local_diff_flows in all windows.
	// Claim the two flowgrams are flow-synchronized.
	is_flow_synchronized = true;
	return is_flow_synchronized;
}

PrecomputeForClustering::PrecomputeForClustering(const Alignment* rai){
	if (not rai->alignment.GetTag("ZC", zc)){
		vector<uint32_t> u_zc;
		if (rai->alignment.GetTag("ZC", u_zc)){
			zc.assign(u_zc.size(), 0);
			for (unsigned int i = 0; i < u_zc.size(); ++i){
				zc[i] = (int) u_zc[i];
			}
		}
	}
	if ((not zc.empty()) and zc.size() < 3){
	    cerr << "ERROR: the ZC tag with number of fields < 3 in read " << rai->alignment.Name << endl;
	    exit(1);
	}
	ReconstructBaseAsCalled(rai, base_seq);
}

// ==============================================================================

FlowSpaceCluster::FlowSpaceCluster(const Alignment* new_rai, const ion::FlowOrder* fo)
	: flow_order_(fo), runid(new_rai->runid),  flow_order_index(new_rai->flow_order_index), is_reverse_strand(new_rai->is_reverse_strand)
{
	PrecomputeForClustering precomp_for_new_rai(new_rai);
	InitializeCluster_(new_rai);
	AddNewMember_(new_rai, precomp_for_new_rai);
}


FlowSpaceCluster::FlowSpaceCluster(const Alignment* new_rai, const vector<ion::FlowOrder>& flow_order_vector, PrecomputeForClustering& precomp_for_new_rai)
	: flow_order_(&(flow_order_vector[new_rai->flow_order_index])), runid(new_rai->runid),  flow_order_index(new_rai->flow_order_index), is_reverse_strand(new_rai->is_reverse_strand)
{
	InitializeCluster_(new_rai);
	AddNewMember_(new_rai, precomp_for_new_rai);
};


void FlowSpaceCluster::InitializeCluster_(const Alignment* new_rai)
{
	trim_info.prefix_bases = new_rai->prefix_bases;
	trim_info.suffix_bases = new_rai->suffix_bases;
	trim_info.prefix_end_flow = new_rai->prefix_flow;
	trim_info.prefix_end_hp = FindFirstHpLen(trim_info.prefix_bases, true);
	trim_info.suffix_start_hp = FindFirstHpLen(trim_info.suffix_bases, false);
}

void FlowSpaceCluster::AddNewMember_(const Alignment* new_rai, PrecomputeForClustering& precomp_for_new_rai)
{
	cluster_members.push_back(new_rai);
	// I use the cluster member with the highest mapping quality to represent this cluster.
	if (new_rai->alignment.MapQuality > best_mapping_quality){
		best_mapping_quality = new_rai->alignment.MapQuality;
		best_mapping_quality_index = (int) cluster_members.size() - 1;
		template_base_seq_ = precomp_for_new_rai.base_seq;

	    // Calculate the flowgram if I didn't do it before.
	    if (precomp_for_new_rai.flowgram.empty() or precomp_for_new_rai.base_index_to_flow_index.empty()){
		    BaseSeqToFlowgram(template_base_seq_, *flow_order_, flow_order_->num_flows(), precomp_for_new_rai.flowgram, precomp_for_new_rai.base_index_to_flow_index);
	    }
	    template_flowgram_ = precomp_for_new_rai.flowgram;

	    // Calculate suffix_start_flow
	    if (not trim_info.suffix_bases.empty()){
		    trim_info.suffix_start_flow = precomp_for_new_rai.base_index_to_flow_index[template_base_seq_.size() - trim_info.suffix_bases.size()];
	    }
	    else{
	    	//  Set to the end flow of template since there is no suffix_bases.
	    	trim_info.suffix_start_flow = precomp_for_new_rai.base_index_to_flow_index.back();
	    }

	    if (trim_info.suffix_start_flow < 0){
		    cerr << "ERROR: insert_base_end_flow outside of [0, num_flows) range in read " << new_rai->alignment.Name << endl;
		    exit(1);
	    }

	    if (not precomp_for_new_rai.zc.empty()){
	    	adapter_start_flow = precomp_for_new_rai.zc[0];
	    }
	    else{
	    	adapter_start_flow = -1;
	    }
	}
}


bool FlowSpaceCluster::AskToJoinMe(const Alignment* query_rai)
{
	PrecomputeForClustering precomp_for_query_rai(query_rai);
	return AskToJoinMe(query_rai, precomp_for_query_rai);
}

// The query_rai can join the cluster if query_rai is flow-synchronized with the cluster
// (Note): Sort the reads by the mapping quality in the decreasing order and then call this function is highly recommended!
// (Note): Don't waste your time inputting query_rai if IsIsolateRead(query_rai) is true, since it won't join me anyway.
bool FlowSpaceCluster::AskToJoinMe(const Alignment* query_rai, PrecomputeForClustering& precomp_for_query_rai)
{
	// I only cluster the reads with 3' adapter found for molecular tagging.
	// @TODO: It can be an optional constraint for general purpose.
	bool is_join = false;
	if (precomp_for_query_rai.zc.empty()){
		return is_join;
	}

	// The most common reason that a read can't join the cluster.
	if (precomp_for_query_rai.zc[0] != adapter_start_flow)   // I require the adapter start flow must be the same.
		return is_join;

	if (query_rai->runid != runid  // The query read must come from the same run
			or query_rai->is_reverse_strand != is_reverse_strand  // The query read must on the same strand
			or query_rai->prefix_bases != trim_info.prefix_bases  // The query read must have exactly the same prefix_bases (hp error not allowed)
			or query_rai->suffix_bases != trim_info.suffix_bases  // The query read must have exactly the same suffix_bases (hp error not allowed)
			or query_rai->prefix_flow != trim_info.prefix_end_flow  // Let's check the prefix end flow again, though it has been carried out by checking the prefix_bases
			or query_rai->flow_order_index != flow_order_index)     // I require the same flow order index (it is redundant since every run_id has a unique flow_order index)
		return is_join;

	// Easy task: Claim flow-synchronized if base_seq of the query_rai and this->template_base_seq_ are identical.
	is_join = precomp_for_query_rai.base_seq == template_base_seq_;
	if (is_join){
		AddNewMember_(query_rai, precomp_for_query_rai);
		return is_join;
	}

	// Calculate the flowgram for query_rai if we haven't done yet.
	if (precomp_for_query_rai.flowgram.empty() or precomp_for_query_rai.base_index_to_flow_index.empty()){
	    BaseSeqToFlowgram(precomp_for_query_rai.base_seq, *flow_order_, flow_order_->num_flows(), precomp_for_query_rai.flowgram, precomp_for_query_rai.base_index_to_flow_index);
	}

	unsigned int synchronized_len = 0;
	// Is precomp_for_query_rai.flowgram flow-synchronized with template_flowgram_?
	is_join = IsFlowSynchronized(template_flowgram_, precomp_for_query_rai.flowgram, synchronized_len, kMaxNumDiffFlows_, kMaxNumLocalDiffFlows_, kStrictFlowSync_);
	if (is_join){
		// Double check if synchronized_len covers suffix_start_flow
		// e.g., a quality trimmed read fails this test.
		if ((int) synchronized_len < trim_info.suffix_start_flow + 1){
    		is_join = false;
		}
		else{
			AddNewMember_(query_rai, precomp_for_query_rai);
		}
		return is_join;
	}
	return is_join;
}

// ==============================================================================

FlowSpaceConsensusMaster::FlowSpaceConsensusMaster(PersistingThreadObjects* thread_objects,
		const ReferenceReader* reference_reader,
		const InputStructures* global_context,
		const string& basecaller_ver):
		thread_objects_(thread_objects), reference_reader_(reference_reader), global_context_(global_context)
{
    flow_order_index_ = -1;
    flow_order_ = NULL;

    is_suffix_counted_in_za_ = false;
    // Note that the format of BaseCaller version is "5.2-x"
    if (basecaller_ver.substr(0, 4) == "5.2-"){
    	long release_ver = strtol(basecaller_ver.substr(4).c_str(), NULL, 0);
    	// The change of ZA definition starts from BaseCaller 5.2-21. Before that, suffix tags are counted in ZA.
    	is_suffix_counted_in_za_ = release_ver < 21;
    }else if (basecaller_ver.substr(0, 17) == "5.3-0-week160330-"){
    	is_suffix_counted_in_za_ = true;
    }
}


struct RecalGroup{
	vector<const Alignment*> group_members;
	uint16_t mapq_sum;
	RecalGroup(){ mapq_sum = 0;};
	RecalGroup(const Alignment* rai){
		group_members = {rai};
		mapq_sum = rai->alignment.MapQuality;
	};
	void AddNew(const Alignment* rai){
		group_members.push_back(rai);
		mapq_sum += rai->alignment.MapQuality;
	};
	// Compare two recalibration groups.
	// A group wins if it has more reads. If ties, the group with larger accumulated mapping quality wins.
	bool operator>(const RecalGroup& rhs){
	    if (group_members.size() == rhs.group_members.size()){
			return (mapq_sum > rhs.mapq_sum);
		}
		return (group_members.size() > rhs.group_members.size());
	};
};

void FlowSpaceConsensusMaster::InitializeForBaseCalling(const FlowSpaceCluster& my_cluster, const vector<float>& consensus_phase_params, bool suppress_recal, string &consensus_read_name){
	flow_order_index_ = my_cluster.flow_order_index;
	flow_order_ = &(global_context_->flow_order_vector.at(flow_order_index_));
	// Set phasing parameters
	thread_objects_->SetModelParameters(flow_order_index_, consensus_phase_params);
	// Disable use of a previously loaded recalibration model
	thread_objects_->DisableRecalibration(flow_order_index_);

	//@TODO: Need to come up with a more clever way to handle recalibration.
	bool recalibration_enabled = false;

	// Let's initialize recalibration
	if (not suppress_recal and global_context_->do_recal.recal_is_live()){
		recalibration_enabled = true;
		vector<pair<MultiAB, RecalGroup> > recal_groups_vec;
		recal_groups_vec.reserve(my_cluster.cluster_members.size());
		for (vector<const Alignment *>::const_iterator read_it = my_cluster.cluster_members.begin(); read_it != my_cluster.cluster_members.end(); ++read_it){
			string found_key = global_context_->do_recal.FindKey((*read_it)->runid, (*read_it)->well_rowcol[1], (*read_it)->well_rowcol[0]);
			MultiAB multi_ab;
		    global_context_->do_recal.getAB(multi_ab, found_key, (*read_it)->well_rowcol[1], (*read_it)->well_rowcol[0]);
		    if (not multi_ab.Valid()){
		    	recalibration_enabled = false;
		    	cerr << "Warning: The read "<< (*read_it)->alignment.Name << " has invalid recalibrartion MultiAB. Recalibration is disabled in the cluster."<< endl;
		    	break;
		    }
		    bool group_exists = false;
		    for (vector<pair<MultiAB, RecalGroup> >::iterator rc_group_it = recal_groups_vec.begin(); rc_group_it != recal_groups_vec.end(); ++rc_group_it){
		    	if (multi_ab.aPtr == rc_group_it->first.aPtr and multi_ab.bPtr == rc_group_it->first.bPtr){
		    		group_exists = true;
		    		rc_group_it->second.AddNew(*read_it);
		    		break;
		    	}
		    }
		    if (not group_exists){
		    	recal_groups_vec.push_back(pair<MultiAB, RecalGroup>(multi_ab, RecalGroup(*read_it)));
		    }
		}
		if (recalibration_enabled){
			vector<pair<MultiAB, RecalGroup> >::iterator best_rc_group_it = recal_groups_vec.begin();
		    for (vector<pair<MultiAB, RecalGroup> >::iterator rc_group_it = recal_groups_vec.begin(); rc_group_it != recal_groups_vec.end(); ++rc_group_it){
				// Find the "best" recalibration result, what I mean "best" is defined by the > operator
				if (rc_group_it->second > best_rc_group_it->second){
					best_rc_group_it = rc_group_it;
				}
			}
			// I use the best recalibration result as the recalibration for the consensus read.
	    	thread_objects_->SetAsBs(flow_order_index_, best_rc_group_it->first.aPtr, best_rc_group_it->first.bPtr);
			// query recalibration structure using row, column, entity.
	    	// The consensus_read_name set here will let tvc use the desired recal results.
	    	consensus_read_name = best_rc_group_it->second.group_members[0]->alignment.Name;
		}
	}

	// Use the read name with the best mapq as the consensus read name if recalibration is not enabled.
	if (not recalibration_enabled){
		consensus_read_name = my_cluster.cluster_members[my_cluster.best_mapping_quality_index]->alignment.Name;
	}
}


// this->InitializeForBaseCalling(short, const vector<float>&) must be done first.
// Input: flow_synchronized_measurements, trim_info
// Outputs: consensus_measurements, prefix_trimmed_consensus_base, measurements_sd
// Call this function if I use void GetMeasurements(const vector<const Alignment*>&, vector< vector <float> >&) to get flow_synchronized_measurements
bool FlowSpaceConsensusMaster::CalculateConsensusMeasurements(const vector<vector <float> >& flow_synchronized_measurements,
								 const TrimInfo& trim_info,
								 vector<float>& consensus_measurements,
								 vector<float>& measurements_sd)
{
    vector<const vector<float> *> flow_synchronized_measurements_ptr;
	flow_synchronized_measurements_ptr.assign(flow_synchronized_measurements.size(), NULL);
	for (unsigned int i_read = 0; i_read < flow_synchronized_measurements.size(); ++i_read){
		flow_synchronized_measurements_ptr[i_read] = &flow_synchronized_measurements[i_read];
	}
	return CalculateConsensusMeasurements(flow_synchronized_measurements_ptr, trim_info, consensus_measurements, measurements_sd);
}


// this->InitializeForBaseCalling(short, const vector<float>&) must be done first.
// Input: flow_synchronized_measurements, trim_info
// Outputs: consensus_measurements, prefix_trimmed_consensus_base, measurements_sd
// Call this function if I use void GetMeasurements(const vector<const Alignment*>&, vector< vector <float> *>&) to get flow_synchronized_measurements
bool FlowSpaceConsensusMaster::CalculateConsensusMeasurements(const vector<const vector <float> *>& flow_synchronized_measurements,
		 const TrimInfo& trim_info,
		 vector<float>& consensus_measurements,
		 vector<float>& measurements_sd)
{
	bool success = false;
	const unsigned int num_reads_in_cluster = flow_synchronized_measurements.size();
	const double num_reads_in_cluster_double = (double) num_reads_in_cluster;
    const double num_reads_in_cluster_minus_one = (double) ((int) num_reads_in_cluster - 1);

	// The flow space consensus is trivial for single read or no read
	// I don't attempt to generate consensus measurement for this guy.
	if (num_reads_in_cluster <= 1){
		return success;
	}

	int min_measurements_len = (int) flow_synchronized_measurements.at(0)->size();
	for (unsigned int i_read = 1; i_read < num_reads_in_cluster; ++i_read){
		min_measurements_len = min(min_measurements_len, (int) flow_synchronized_measurements.at(i_read)->size());
	}

	// consensus_measurements[flow] = mean(flow_synchronized_measurements[:][flow])
	consensus_measurements.resize(flow_order_->num_flows()); // Note that Treephaser requires the length of measurement = length of flow order
	// Taking the mean measurements of reads. Set to zero if the flow >= min_measurements_len
	for (int i_flow = 0; i_flow < min_measurements_len; ++i_flow){
		double mean_measuremet = 0.0; // Use double to get better numerical precision for cumulative sum
		for (unsigned int i_read = 0; i_read < num_reads_in_cluster; ++i_read){
			mean_measuremet += (double) flow_synchronized_measurements.at(i_read)->at(i_flow);
		}
		mean_measuremet /= num_reads_in_cluster_double;
		consensus_measurements.at(i_flow) = (float) mean_measuremet;
	}

	// Set data for base calling
	consensus_read_.SetData(consensus_measurements, flow_order_->num_flows());
	consensus_read_.sequence.resize(trim_info.prefix_bases.size());
	for (unsigned int i_base = 0; i_base < trim_info.prefix_bases.size(); ++i_base){
		consensus_read_.sequence[i_base] = trim_info.prefix_bases[i_base];
	}

	// Call Treephaser to solve the consensus_measurements
    int safe_start_flow = min(trim_info.prefix_end_flow, (int) consensus_read_.normalized_measurements.size() - 1);
    thread_objects_->SolveRead(flow_order_index_, consensus_read_, safe_start_flow, min_measurements_len);

    // Check Treephaser got any solution or not
    if (consensus_read_.sequence.size() <= trim_info.prefix_bases.size())
        return success;

    // Shrink the length of consensus_measurements to what we want to preserve
    consensus_measurements.resize(min_measurements_len);

    // Calculate the standard deviation of the measurements
	measurements_sd.resize(min_measurements_len);
	for (int i_flow = 0; i_flow < min_measurements_len; ++i_flow){
		double mean_squared_err = 0.0;
		for (unsigned int i_read = 0; i_read < num_reads_in_cluster; ++i_read){
			double err = (double) (flow_synchronized_measurements.at(i_read)->at(i_flow) - consensus_measurements.at(i_flow));
			mean_squared_err += (err * err);
		}
		mean_squared_err /= num_reads_in_cluster_minus_one; // It should be safe since I don't deal with any single read cluster.
		measurements_sd.at(i_flow) = (float) sqrt(mean_squared_err);
	}
	success = true;
    return success;
}

// Apply prefix and suffix trimming on consensus_read_.sequence according to trim_info
// Input: trim_info (contains the trimming information)
// Output: prefix and suffix trimmed base sequence, start_flow
// Note that I do the trimming in flow space -- the bases actually trimmed may be different from trim_info.prefix_base and trim.suffix_bases.
void FlowSpaceConsensusMaster::TrimPrefixSuffixBases(const TrimInfo& trim_info, string& trimmed_consensus_bases, int& start_flow)
{
	vector<int> bases_index_to_flow_index;
	vector<int> consensus_flowgram;
	BaseSeqToFlowgram(consensus_read_.sequence, *flow_order_, flow_order_->num_flows(), consensus_flowgram, bases_index_to_flow_index);
	int trim_prefix_len = 0;
	int trim_suffix_len = 0;

	//  I shall trim them all if the last incorporated flow is less than trim_info.prefix_end_flow.
	if (bases_index_to_flow_index.back() < trim_info.prefix_end_flow){
		trimmed_consensus_bases.resize(0);
		start_flow = -1;
		return;
	}

	for (int flow_idx = 0; flow_idx < trim_info.prefix_end_flow; ++flow_idx){
		trim_prefix_len += consensus_flowgram[flow_idx];
	}
	// I suppose to obtain at least (trim_info.prefix_end_hp)-mer at the (trim_info.prefix_end_flow)-th flow, but I don't.
	// So I don't trim more than (consensus_flowgram[trim_info.prefix_end_flow])-mer at this flow.
	trim_prefix_len += min(consensus_flowgram[trim_info.prefix_end_flow], trim_info.prefix_end_hp);
	// start_flow is the first incorporated flow of the template.
	start_flow = bases_index_to_flow_index[trim_prefix_len];

	if (bases_index_to_flow_index.back() >= trim_info.suffix_start_flow){
		for (int flow_idx = bases_index_to_flow_index.back(); flow_idx > trim_info.suffix_start_flow; --flow_idx){
			trim_suffix_len += consensus_flowgram[flow_idx];
		}
		// I suppose to obtain at least (trim_info.suffix_start_hp)-mer at the (trim_info.suffix_start_flow)-th flow, but I don't.
		// So I don't trim more than (consensus_flowgram[trim_info.suffix_start_flow])-mer at this flow.
		trim_suffix_len += min(consensus_flowgram[trim_info.suffix_start_flow], trim_info.suffix_start_hp);
	}

	int trimmed_consensus_bases_len = (int) consensus_read_.sequence.size() - (trim_prefix_len + trim_suffix_len);
	trimmed_consensus_bases.resize(trimmed_consensus_bases_len);
	int untrimmed_base_idx = trim_prefix_len;
	for (int trimmed_base_idx = 0; trimmed_base_idx < trimmed_consensus_bases_len; ++trimmed_base_idx){
		trimmed_consensus_bases[trimmed_base_idx] = consensus_read_.sequence[untrimmed_base_idx];
		++untrimmed_base_idx;
	}
}

// This function works for the case where the measurements have been written in Alignment
void FlowSpaceConsensusMaster::GetMeasurements(const vector<const Alignment*>& cluster_members, vector<const vector <float> *>& flow_synchronized_measurements_ptr)
{
	flow_synchronized_measurements_ptr.assign(cluster_members.size(), NULL);
	for (unsigned int read_idx = 0; read_idx < cluster_members.size(); ++read_idx)
	{
		if (cluster_members[read_idx]->measurements.empty()){
			cerr << "ERROR: Normalized measurements not initialized in read " << cluster_members[read_idx]->alignment.Name << endl;
			exit(1);
		}
		flow_synchronized_measurements_ptr[read_idx] = &(cluster_members[read_idx]->measurements);
	}
}

// This function works for the case where the measurements were not written in Alignment
void FlowSpaceConsensusMaster::GetMeasurements(const vector<const Alignment*>& cluster_members,  vector< vector <float> >& flow_synchronized_measurements)
{
	flow_synchronized_measurements.reserve(cluster_members.size());
	for (vector<const Alignment *>::const_iterator read_it = cluster_members.begin(); read_it != cluster_members.end(); ++read_it)
	{
	    vector<int16_t> quantized_measurements;
	    if (not (*read_it)->alignment.GetTag("ZM", quantized_measurements)) {
		    cerr << "ERROR: Normalized measurements ZM:tag is not present in read " << (*read_it)->alignment.Name << endl;
		    exit(1);
		}
	    flow_synchronized_measurements.push_back(vector<float>(quantized_measurements.size()));
	    for (unsigned int i_flow = 0; i_flow < quantized_measurements.size(); ++i_flow){
	    	flow_synchronized_measurements.back()[i_flow] = ((float) quantized_measurements[i_flow]) / 256.0f;
	    }
	}
}

// consensus_phase_params = mean of the phase_params
void FlowSpaceConsensusMaster::CalculateConsensusPhaseParams(const vector<const Alignment*>& cluster_members, vector<float>& consensus_phase_params, bool is_zero_droop)
{
	consensus_phase_params.assign(3, 0.0f);
	float num_reads = (float) cluster_members.size();
	assert(num_reads > 0.0f);
	for (vector<const Alignment *>::const_iterator read_it = cluster_members.begin(); read_it != cluster_members.end(); ++read_it){
	    consensus_phase_params[0] += ((*read_it)->phase_params[0]);
	    consensus_phase_params[1] += ((*read_it)->phase_params[1]);
	}
	consensus_phase_params[0] /= num_reads;
	consensus_phase_params[1] /= num_reads;

    if (not is_zero_droop){
    	for (vector<const Alignment *>::const_iterator read_it = cluster_members.begin(); read_it != cluster_members.end(); ++read_it){
	        consensus_phase_params[2] += ((*read_it)->phase_params[2]);
    	}
    	consensus_phase_params[2] /= num_reads;
    }
}


// Get the length of trimmed insert bases using the ZA tag where ZA = number of insert bases.
// Note that insert bases = template + YE + YT in 5.2.
// Note that insert bases = template in 5.2.1 and after.
// Input is_suffix_in_za = true if BaseCaller version <= 5.2; false if BaseCaller version >= 5.2.1
int LenTrimmedInsertBases(const Alignment* rai, bool is_suffix_in_za)
{
	int num_trimmed_bases = 0;
	int za = -1;
	string ye = "";
	int ye_len = 0;
	int len_QueryBases = (int) rai->alignment.QueryBases.size();

	if( not rai->alignment.GetTag("ZA", za)){
		uint32_t u_za = 0;
	    if (not rai->alignment.GetTag("ZA", u_za)) {
	        cerr << "ERROR: ZA tag not found in read " << rai->alignment.Name << endl;
	        exit(1);
	    }
	    za = (int) u_za;
	}
	if (za < len_QueryBases){
	    cerr << "Error: za < QueryBases.size() in read " << rai->alignment.Name << endl;
	    exit(1);
	}
	num_trimmed_bases = is_suffix_in_za? za - (len_QueryBases + rai->suffix_bases.size()) : za - len_QueryBases;
	return num_trimmed_bases;
}


// Input is_suffix_in_za = true if BaseCaller version <= 5.2; false if BaseCaller version >= 5.2.1
bool IsQualityTrimmedRead(const Alignment* rai, bool is_suffix_in_za){
	// It should be is_quality_trimmed = ( LenTrimmedInsertBases(rai) > 0)
	// But the bases additionally trimmed by with heal-tag-indel-hp as also counted in LenTrimmedInsertBases(rai).
	// The number 4 is safe against this issue.
	int max_num_trimmed_bases_allowed = is_suffix_in_za? 4 : 0;
	bool is_quality_trimmed = abs(LenTrimmedInsertBases(rai, is_suffix_in_za)) > max_num_trimmed_bases_allowed;
	return is_quality_trimmed;
}

// The return value indicates whether the consensus read needs realignment or not.
// Two cases where realignment is not needed
// (Case 1): The consensus query bases is exactly the same as the template read in the cluster.
// (Case 2): The consensus query bases all matches the reference.
// Usually, only a very small portion of consensus reads needs realignment.
bool FlowSpaceConsensusMaster::SaveToBamAlignment(const FlowSpaceCluster& my_cluster,
		const vector<float>& consensus_phase_params,
		const vector<float>& consensus_measurements,
		const vector<float>& measurements_sd,
		const string& trimmed_bases,
		const string& consensus_read_name,
		int start_flow,
		BamAlignment& alignment)
{
	// Use the read with the best mapping quality in the cluster as a template read.
	// The tags "ZT", "ZE", "YT", "YE", "RG", etc. follow the template read.
	// Note that The choice of the read name will affect the recaliration results when running tvc with the consensus bam.
	const BamAlignment& template_alignment = my_cluster.cluster_members[my_cluster.best_mapping_quality_index]->alignment;
	alignment = template_alignment;

	// Set the query bases
	alignment.Name = consensus_read_name;
	alignment.QueryBases = trimmed_bases;
	alignment.Length = (int32_t) alignment.QueryBases.size();
	if (my_cluster.is_reverse_strand){
		RevComplementInPlace(alignment.QueryBases);
	}

	// If the query bases of the consensus read is exactly the same as the template read, then I don't need to realign the consensus read.
	bool need_realign = (alignment.QueryBases != template_alignment.QueryBases);

	// If not exactly the same as the template, then let's try another trivial alignment where the consensus query bases all matches the reference
	if (need_realign){
		bool all_match_ref = false;
		string template_md = "";
		template_alignment.GetTag("MD", template_md);
		// If the template query bases all matches the reference, then no need to check again.
		if ((template_md != to_string((int) template_alignment.QueryBases.size()))
				    or (alignment.QueryBases.size() != template_alignment.QueryBases.size())){
			long try_start_position = template_alignment.Position;
			if (my_cluster.is_reverse_strand){
				int template_xm = (int) alignment.QueryBases.size(); // XM is the ion tag that specifies the number of ref bases spanned by the alignment.
				template_alignment.GetTag("XM", template_xm);
				// The actual "start" position of a reverse read of should be the end of the alignment.
				try_start_position = template_alignment.Position + (long) template_xm - (long) alignment.QueryBases.size();
			}

			string reference_bases = reference_reader_->substr((int) (template_alignment.RefID), try_start_position, (long) alignment.QueryBases.size());
			all_match_ref = alignment.QueryBases == reference_bases;

			if (all_match_ref){
				need_realign = false;
				alignment.Position = try_start_position;
				alignment.Qualities = "*";
				// I set MapQ super sloppily by adding 3 (since from not all matches to all matches it means that the mapping quality is getting better).
				// Also, according to the spec, no alignments should be assigned mapping quality 255.
				alignment.MapQuality = (uint16_t) (max(254 - 3, (int) alignment.MapQuality) + 3);
				alignment.CigarData.assign(1, CigarOp('M',  (uint32_t) alignment.QueryBases.size())); // All-match cigar
				alignment.EditTag("MD", "Z", to_string(alignment.QueryBases.size())); // All matches MD
				alignment.EditTag("NM", "i", 0); // no mismatch
				alignment.EditTag("XM", "i", (int) alignment.QueryBases.size());
			}
		}
	}

	if (need_realign){
		alignment.CigarData.assign(1, CigarOp('M',  (uint32_t) alignment.QueryBases.size())); // dummay cigar that claim all M
		alignment.Qualities = "*";
		alignment.MapQuality = 0;
		alignment.RemoveTag("MD");
		alignment.RemoveTag("NM");
		alignment.RemoveTag("XM");
	}

	// Add/Edit/Remove tags
	if (is_suffix_counted_in_za_){
		alignment.EditTag("ZA", "i", (int) (trimmed_bases.size() + my_cluster.trim_info.suffix_bases.size()));
	}else{
		alignment.EditTag("ZA", "i", (int) trimmed_bases.size());
	}
	alignment.EditTag("ZF", "i", start_flow);
	alignment.EditTag("ZG", "i", my_cluster.adapter_start_flow);
	alignment.EditTag("ZC", vector<int> {my_cluster.adapter_start_flow, -1, -1, -1}); // ZC[1], ZC[2], ZC[3] may vary members from members, so I don't save them for the consensus reads.
	alignment.EditTag("ZP", consensus_phase_params);
	alignment.RemoveTag("ZB"); // ZB may vary members from members, so I don't save them for the consensus reads.

	// Add the ZM tag
	vector<int16_t> vec_temp(consensus_measurements.size());
    for (unsigned int i_flow = 0; i_flow < consensus_measurements.size(); ++i_flow){
    	vec_temp[i_flow] = (int16_t) (consensus_measurements[i_flow] * 256.0f);
    }
	alignment.EditTag("ZM", vec_temp);

	// New tags "ZR", "ZN", "ZS" from consensus
	// Check the pre-existence of these tags.
	for (vector<const Alignment*>::const_iterator read_it = my_cluster.cluster_members.begin(); read_it != my_cluster.cluster_members.end(); ++read_it)
	{
		if ((*read_it)->alignment.HasTag("ZN") or (*read_it)->alignment.HasTag("ZR") or (*read_it)->alignment.HasTag("ZS")){
			cerr << "Warning: The tag ZN, ZR or ZS is found in the read "<< (*read_it)->alignment.Name << ". "
				 << "The input bam file may contain unrecognized tags or may be a consensus bam. "
				 << "Some information may be lost or the ZM, ZP, ZS, ZR, ZN tags may be calculated incorrectly." << endl;
		}
	}

	vec_temp.resize(measurements_sd.size());
	for (unsigned int i_flow = 0; i_flow < measurements_sd.size(); ++i_flow){
    	vec_temp[i_flow] = (int16_t) (measurements_sd[i_flow] * 256.0f);
    }
	alignment.EditTag("ZS", vec_temp);

	alignment.EditTag("ZR", "i", (int) my_cluster.cluster_members.size()); // ZR tag = read count, number of reads that form the consensus read
	string read_names = "";
	for (unsigned int i_member = 0; i_member < my_cluster.cluster_members.size(); ++i_member){
		read_names += my_cluster.cluster_members[i_member]->alignment.Name;
		if (i_member != my_cluster.cluster_members.size() - 1){
			read_names += ";";
		}
	}
	for (string::iterator c_it = read_names.begin(); c_it != read_names.end(); ++c_it)
		if (*c_it == ':')  {*c_it = '.';} // use "." to replace ":"
	alignment.EditTag("ZN", "Z", read_names);  // ZN tag = query names of the reads that from the consensus read

	return need_realign;
}

bool CompareMapQ(const Alignment* const rai_1, const Alignment* const rai_2){
	return rai_1->alignment.MapQuality > rai_2->alignment.MapQuality;
}

void AppendPositionTicket(list<PositionInProgress>::iterator& position_ticket, Alignment* const alignment)
{
	if (position_ticket->begin == NULL)
		position_ticket->begin = alignment;
	if (position_ticket->end != NULL)
		position_ticket->end->next = alignment;
	position_ticket->end = alignment;
	position_ticket->end->next = NULL;
}

// Input: family_members
// Output: consensus_position_ticket, aln_needed_consensus_position_ticket
unsigned int  FlowSpaceConsensusMaster::FlowSpaceConsensusOneFamily(vector<Alignment *>& family_members,
		list<PositionInProgress>::iterator& consensus_position_ticket,
		list<PositionInProgress>::iterator& aln_needed_consensus_position_ticket)
{
	// For ConsensusCounter
	static ConsensusCounter my_counter(consensus_for_molecular_tag);

	if (family_members.empty()){
		return 0;
	}

    unsigned int add_to_num_func_fam = family_members.empty()? 0 : 1;
    unsigned int add_to_num_consensus_reads = 0;
	unsigned int add_to_num_single_read_consensus = 0;
	unsigned int add_to_num_reads_in_func_fam = family_members.size();
	unsigned int add_to_num_reads_in_consensus_bam = 0;
	unsigned int add_to_num_consensus_reads_need_aln = 0;

	vector<FlowSpaceCluster> flow_space_clusters;
	vector<const Alignment *> isolated_reads;  // reads w/o 3' adapter found or quality trimmed are isolate reads.

	// This should be enough most of the time.
	if (consensus_for_molecular_tag){
		flow_space_clusters.reserve(4);
		if ((not filter_single_read_consensus) and (not (need_3_end_adapter and filter_qt_reads))){
			isolated_reads.reserve(8);
		}
	}else{
		flow_space_clusters.reserve(32);
		if ((not filter_single_read_consensus) and (not (need_3_end_adapter and filter_qt_reads))){
			isolated_reads.reserve(add_to_num_reads_in_func_fam / 8);
		}
	}

	// (Step 0): sort the reads by mapping quality
	sort(family_members.begin(), family_members.end(), CompareMapQ);

	// (Step 1): Flow space clustering
	for (vector<Alignment *>::const_iterator member_it = family_members.begin(); member_it != family_members.end(); ++member_it)
	{
		const Alignment* rai = *member_it;  // Now I do flowspace clustering for the read rai.

		// Did BaseCaller find 3' adapter in the read?
		bool is_adapter_found = (rai->alignment.HasTag("ZC") and rai->alignment.HasTag("ZA"));
		if (not is_adapter_found){
			if (not (need_3_end_adapter or filter_single_read_consensus)){
				isolated_reads.push_back(rai);
			}
			continue;
		}

		// Is the read quality trimmed?
		bool is_quality_trimmed_read = IsQualityTrimmedRead(rai, is_suffix_counted_in_za_);
		if (is_quality_trimmed_read){
			if (not (filter_qt_reads or filter_single_read_consensus)){
				isolated_reads.push_back(rai);
			}
			continue;
		}

		PrecomputeForClustering precomp_for_rai(rai);  // The information in precomp_for_rai may be used frequently
		bool is_in_existing_cluster = false; // Can I join any existing cluster

		for (vector<FlowSpaceCluster>::iterator cluster_it = flow_space_clusters.begin(); cluster_it != flow_space_clusters.end(); ++cluster_it){
			// Try to join any existing cluster if possible
			if (cluster_it->AskToJoinMe(rai, precomp_for_rai)){
				is_in_existing_cluster = true;
				break;  // The read join the cluster. No need to seek joining others.
			}
		}

		if (not is_in_existing_cluster){
			// The read can not join any existing cluster. So it forms a new cluster.
			flow_space_clusters.push_back(FlowSpaceCluster(rai, global_context_->flow_order_vector, precomp_for_rai));
		}
	}

	// How many consensus reads will be generated for the reads in family_members?
	add_to_num_consensus_reads = flow_space_clusters.size() + isolated_reads.size();
	add_to_num_reads_in_consensus_bam += isolated_reads.size();
	add_to_num_single_read_consensus += isolated_reads.size();

	// (Step 2a): BamAlignment for isolate reads: no extra action required
	for (vector<const Alignment *>::iterator read_it = isolated_reads.begin(); read_it != isolated_reads.end(); ++read_it)
	{
		// Append a new consensus alignment to consensus_position_ticket
		Alignment* consensus_alignment = new Alignment;
		consensus_alignment->alignment = (*read_it)->alignment;
		AppendPositionTicket(consensus_position_ticket, consensus_alignment);
	}

	// (Step 2b): BamAlignment for clusters
	for (vector<FlowSpaceCluster>::iterator cluster_it = flow_space_clusters.begin(); cluster_it != flow_space_clusters.end(); ++cluster_it)
	{
		// The cluster consists of single read or no read, no additional action required
		if (cluster_it->cluster_members.size() <= 1){
			if (cluster_it->cluster_members.empty() or filter_single_read_consensus){
				continue;
			}
			add_to_num_reads_in_consensus_bam += cluster_it->cluster_members.size();

			// Append a new consensus alignment to consensus_position_ticket
			Alignment* consensus_alignment = new Alignment;
			consensus_alignment->alignment = cluster_it->cluster_members[0]->alignment;
			AppendPositionTicket(consensus_position_ticket, consensus_alignment);
			++add_to_num_single_read_consensus;
			continue;
		}
		add_to_num_reads_in_consensus_bam += cluster_it->cluster_members.size();

		// Now I calculate the consensus_phase_params, consensus_measurements, trimmed_consensus_bases for the cluster
		vector<vector<float> > measurements_in_cluster;
		vector<float> consensus_phase_params;  // consensus phasing parameters for the cluster
		vector<float> consensus_measurements;  // consensus measurements for the cluster
		vector<float> measurements_sd;  // measurements deviation from consensus_measurements
		string trimmed_consensus_bases;  // prefix/suffix trimmed consensus base sequence as called by BaseCaller
		int start_flow;  // The first incorporated flow of trimmed_consensus_bases

		// Calculate consensus phase parameters and get the measurements of the reads
		CalculateConsensusPhaseParams(cluster_it->cluster_members, consensus_phase_params, true);

		// Initialize BaseCaller
		// Use the read with the best mapq as the template read of the cluster.
		string consensus_read_name;
		InitializeForBaseCalling(*cluster_it, consensus_phase_params, suppress_recalibration, consensus_read_name);

		// Calculate consensus measurements
		GetMeasurements(cluster_it->cluster_members, measurements_in_cluster);
		bool success = CalculateConsensusMeasurements(measurements_in_cluster, cluster_it->trim_info, consensus_measurements, measurements_sd);

		if (not success){
			cerr << "Warning: Fail to calculate the consensus measurements of the cluster (";
			for (vector<const Alignment *>::const_iterator read_it = cluster_it->cluster_members.begin(); read_it != cluster_it->cluster_members.end(); ++read_it){
				cerr << (*read_it)->alignment.Name <<", ";
			}
			cerr << "). The cluster will be ignored." << endl;
			continue;
		}

		// Trim prefix and suffix bases, note that trimmed_consensus_bases is the trimmed bases as called by BaseCaller
		TrimPrefixSuffixBases(cluster_it->trim_info, trimmed_consensus_bases, start_flow);

		// Write to the consensus information to a new Alignment
		Alignment* consensus_alignment = new Alignment;
		bool aln_needed = SaveToBamAlignment(*cluster_it, consensus_phase_params, consensus_measurements, measurements_sd, trimmed_consensus_bases, consensus_read_name, start_flow, consensus_alignment->alignment);

		if (aln_needed){
			++add_to_num_consensus_reads_need_aln;
			// Append the new consensus alignment to consensus_position_ticket
			AppendPositionTicket(aln_needed_consensus_position_ticket, consensus_alignment);
		}
		else{
			AppendPositionTicket(consensus_position_ticket, consensus_alignment);
		}
	}
	my_counter.Count(add_to_num_func_fam,
			add_to_num_consensus_reads,
			add_to_num_single_read_consensus,
			add_to_num_reads_in_func_fam,
			add_to_num_reads_in_consensus_bam,
			add_to_num_consensus_reads_need_aln);
	return add_to_num_reads_in_consensus_bam;
}

void FlowSpaceConsensusMaster::PropagateFlowspaceConsensusParameters(const ConsensusParameters& my_param, bool use_mol_tag){
	suppress_recalibration = my_param.program_flow.suppress_recalibration;
	filter_qt_reads = my_param.filter_qt_reads;
	filter_single_read_consensus = my_param.filter_single_read_consensus;
	need_3_end_adapter = my_param.need_3_end_adapter;
	consensus_for_molecular_tag = use_mol_tag;
}

void FlowSpaceConsensusMaster::InitializeConsensusCounter(){
    // Initialize the consensus counter using dummy inputs since now I know consensus_for_molecular_tag.
	list<PositionInProgress>::iterator dummy_iter;
    vector<Alignment *> dummy_family(0);
    FlowSpaceConsensusOneFamily(dummy_family, dummy_iter, dummy_iter);
}

void GenerateFlowSpaceConsensusPositionTicket(vector< vector< vector<MolecularFamily> > >& my_molecular_families_multisample,
                                     FlowSpaceConsensusMaster& flow_space_consensus_master,
									 unsigned int min_family_size,
                                     list<PositionInProgress>::iterator& consensus_position_ticket,
									 list<PositionInProgress>::iterator& aln_needed_consensus_position_ticket,
									 TargetsManager* targets_manager,
									 bool skip_consensus)
{
	// my_molecular_families_multisample is usually the famly pileup that cover one target.
	// Typcally, a read just covers one target.
	// map is a better container than vector to store the target stat.
	// stat_of_targets[i] is the coverage information for the i-th unmerged region generated here.
	// TODO: Should I split coverage stat for each sample?
	map<int, TargetStat> stat_of_targets;
	for (vector< vector< vector< MolecularFamily> > >::iterator sample_it = my_molecular_families_multisample.begin(); sample_it != my_molecular_families_multisample.end(); ++sample_it) {
		for (vector< vector< MolecularFamily> >::iterator strand_it = sample_it->begin(); strand_it != sample_it->end(); ++strand_it) {
			for (vector< MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it !=  strand_it->end(); ++fam_it) {
				// Is *fam_it functional?
				if (fam_it->SetFuncFromAll((unsigned int) min_family_size)) {
					unsigned int consensus_fam_size = 0;
					if (skip_consensus){
						consensus_fam_size = fam_it->GetFamSize();
					}else{
						// Generate consensus reads
						consensus_fam_size = flow_space_consensus_master.FlowSpaceConsensusOneFamily(fam_it->all_family_members, consensus_position_ticket, aln_needed_consensus_position_ticket);
					}
					// Count the coverage of the target.
					if (consensus_fam_size >= min_family_size and (not fam_it->all_family_members.empty())){
						// Important Assumption: is_split_families_by_region_ = true in MolecularFamilyGenerator.
						for (vector<int>::iterator target_it = fam_it->all_family_members[0]->target_coverage_indices.begin(); target_it != fam_it->all_family_members[0]->target_coverage_indices.end(); ++target_it){
							TargetStat& my_stat = stat_of_targets[*target_it];
							my_stat.read_coverage += consensus_fam_size;
							++(my_stat.family_coverage);
							++(my_stat.fam_size_hist[consensus_fam_size]);
						}
					}
				}
			}
		}
	}
	targets_manager->AddCoverageToRegions(stat_of_targets);

}
