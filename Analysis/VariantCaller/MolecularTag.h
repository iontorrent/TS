/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MOLECULARTAG_H
#define MOLECULARTAG_H


#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <Variant.h>
#include <sys/time.h>
#include <assert.h>
#include "InputStructures.h"
#include "ExtendParameters.h"
#include "BAMWalkerEngine.h"
#include "AlleleParser.h"
#include "MolecularTagTrimmer.h"

using namespace std;

class ConsensusAlignmentManager {
private:
	//
	ReferenceReader const* ref_reader_;
	// My private utilities
	void InsertToDict_(map<string, pair<unsigned int, unsigned int> >& read_count_dict, const string& my_key, unsigned int my_count, unsigned int my_tie_breaker) const;
	void GetMajority_(const map<string, pair<unsigned int, unsigned int> >& read_count_dict, string& majority_key) const ;
	void PartiallyCopyAlignmentFromAnother_(Alignment& alignment, const Alignment& template_alignment, int read_count) const;
	bool PrettyAlnToCigar_(const string& pretty_aln, vector<CigarOp>& cigar_data) const;
	char PrettyCharToCigarType_(char pretty_aln) const;
	// Private functions for determining the consensus alignment
	bool CalculateConsensusStrand_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const;
	bool CalculateConsensusPositions_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const;
	bool CalculateConsensusQueryAndCigar_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const;
	bool FinalizeConsensusRead_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const;
	// Debug related
	void PrintVerbose_(const vector<Alignment*>& family_members, const Alignment& cons_alignment) const;
	void PrintFailureMessage_(const vector<Alignment*>& family_members, const string& my_failure_reason) const;
	bool debug_;

public:
	~ConsensusAlignmentManager(){ref_reader_ = NULL;};
	ConsensusAlignmentManager(ReferenceReader const * const ref_reader = NULL, bool debug = false);
	void SetDebug(bool debug) {debug_ = debug;};
	void SetReferenceReader(ReferenceReader const * const ref_reader) { ref_reader_ = ref_reader; };
	// Interface for calculate basespace consensus read and alignment.
	bool CalculateConsensus(const vector<Alignment*>& family_members, Alignment& cons_alignment) const;
};

template <class MemberType>
class AbstractMolecularFamily {
public:
	//@TODO: Use enumerate for strand_key. Current the value of strand_key follows the convention in TVC evaluator.
	int strand_key;                          // -1: Bi-dir, 0: FWD, 1: REV
	string family_barcode;                   // prefix tag + suffix tag. If strand_key = -1, use the representation on the FWD strand.
	vector<MemberType> all_family_members;   // All reads identified in the family, no additional filtering applied.
	vector<MemberType> valid_family_members; // The reads from all_family_members that pass certain filtering criterion.

    AbstractMolecularFamily (const string &barcode = "", int strand_key = -1)
    	: strand_key(strand_key), family_barcode(barcode) { all_family_members.reserve(4); };

    virtual ~AbstractMolecularFamily(){};
    virtual int CountFamSizeFromAll() = 0;
    virtual int CountFamSizeFromValid() = 0;

	bool SetFuncFromAll(unsigned int min_fam_size, unsigned int min_fam_per_strand_cov = 0){
		if (not IsFamSizeCounted_()) {
			CountFamSizeFromAll();
		}
		is_func_from_family_members_ = (fam_size_ >= (int) min_fam_size);
		if (strand_key < 0){
			is_func_from_family_members_ *= (min(fam_cov_fwd_, fam_cov_rev_) >= (int) min_fam_per_strand_cov);
		}
		return is_func_from_family_members_;
	};

	bool SetFuncFromValid(unsigned int min_fam_size, unsigned int min_fam_per_strand_cov = 0){
		if (not IsValidFamSizeCounted_()) {
			CountFamSizeFromValid();
		}
		is_func_from_valid_family_members_ = (valid_fam_size_ >= (int) min_fam_size);
		if (strand_key < 0){
			is_func_from_valid_family_members_ *= (min(valid_fam_cov_fwd_, valid_fam_cov_rev_) >= (int) min_fam_per_strand_cov);
		}
		return is_func_from_valid_family_members_;
	};

	void AddNewMember(const MemberType& new_member)
		{ all_family_members.push_back(new_member); };

	bool GetFuncFromAll() const {
		assert(IsFamSizeCounted_());  // CountFamSizeFromAll must be done first.
		return is_func_from_family_members_;
	};

	bool GetFuncFromValid() const {
		assert(IsValidFamSizeCounted_());  // CountFamSizeFromValid must be done first.
		return is_func_from_valid_family_members_;
	};

	int GetFamSize(int strand_key = -1) {
		if (not IsFamSizeCounted_()) {
			CountFamSizeFromAll();
		}
		if (strand_key == 0){
			return fam_cov_fwd_;
		}else if (strand_key == 1){
			return fam_cov_rev_;
		}
		return fam_size_;
	};

	int GetValidFamSize(int strand_key = -1) {
		if (not IsValidFamSizeCounted_()) {
			CountFamSizeFromValid();
		}
		if (strand_key == 0){
			return valid_fam_cov_fwd_;
		}else if (strand_key == 1){
			return valid_fam_cov_rev_;
		}
		return valid_fam_size_;
	};

	void ResetFamily(){
    	all_family_members.resize(0);
    	fam_size_ = -1;
    	fam_cov_fwd_ = -1;
    	fam_cov_rev_ = -1;
    	is_func_from_family_members_ = false;
    	ResetValidFamilyMembers();
	}

    void ResetValidFamilyMembers() {
    	valid_family_members.resize(0);
    	valid_family_members.reserve(all_family_members.size());
    	valid_fam_size_ = -1;
    	valid_fam_cov_fwd_ = -1;
    	valid_fam_cov_rev_ = -1;
    	is_func_from_valid_family_members_ = false;
    };

    void ResetFamSize() {
    	fam_size_ = -1;
    	fam_cov_fwd_ = -1;
    	fam_cov_rev_ = -1;
    	is_func_from_family_members_ = false;
    	valid_fam_size_ = -1;
    	valid_fam_cov_fwd_ = -1;
    	valid_fam_cov_rev_ = -1;
    	is_func_from_valid_family_members_ = false;
    }

protected:
	bool is_func_from_family_members_ = false;
	bool is_func_from_valid_family_members_ = false;
    int fam_size_ = -1;  // Total read counts in all_family_members (a consensus read counted by its read_count)
    int fam_cov_fwd_ = -1; // Total fwd read counts in all_family_members (a consensus read counted by its read_count)
    int fam_cov_rev_ = -1; // Total rev read counts in all_family_members (a consensus read counted by its read_count)
    int valid_fam_size_ = -1; // Total read counts in valid_family_members (a consensus read counted by its read_count)
    int valid_fam_cov_fwd_ = -1; // Total fwd read counts in valid_family_members (a consensus read counted by its read_count)
    int valid_fam_cov_rev_ = -1; // Total rev read counts in valid_family_members (a consensus read counted by its read_count)
    bool IsFamSizeCounted_() const {
    	return (fam_size_ >= 0 && fam_cov_fwd_ >= 0 && fam_cov_rev_ >= 0);
    };
    bool IsValidFamSizeCounted_() const {
    	return (valid_fam_size_ >= 0 && valid_fam_cov_fwd_ >= 0 && valid_fam_cov_rev_ >= 0);
    };
};


class MolecularFamily : public AbstractMolecularFamily<Alignment*>
{
public:
	MolecularFamily(const string& barcode = "", int strand = -1)
		: AbstractMolecularFamily<Alignment*>(barcode, strand) {};
    int CountFamSizeFromAll();
    int CountFamSizeFromValid();
	void SortAllFamilyMembers();
	void SortValidFamilyMembers();
	void ResetValidFamilyMembers();
    virtual ~MolecularFamily(){};
	bool is_all_family_members_sorted = false;
	bool is_valid_family_members_sorted = false;
};


// An interface of handling the tag structures.
class MolecularTagManager
{
private:
	vector<string> multisample_prefix_tag_struct_;
	vector<string> multisample_suffix_tag_struct_;
	vector<vector<int> >    multisample_prefix_tag_random_bases_idx_;
	vector<vector<int> >    multisample_prefix_tag_stutter_bases_idx_;
	vector<vector<int> >    multisample_suffix_tag_random_bases_idx_;
	vector<vector<int> >    multisample_suffix_tag_stutter_bases_idx_;

public:
	MolecularTagManager();
	MolecularTagTrimmer* tag_trimmer;
	void Initialize(MolecularTagTrimmer* const input_tag_trimmer, const SampleManager* const sample_manager);
	//! Functions for tag strictness
	bool IsStrictTag(const string& prefix_tag, const string& suffix_tag, int sample_idx) const;
	bool IsStrictPrefixTag(const string& prefix_tag, int sample_idx) const;
	bool IsStrictSuffixTag(const string& suffix_tag, int sample_idx) const;
	//! Functions for getting tag structures of the sample
	string GetPrefixTagStruct(int sample_indx) const {return multisample_prefix_tag_struct_[max(0, sample_indx)];}; // sample_idx = -1 indicates no multisample
	string GetSuffixTagStruct(int sample_indx) const {return multisample_suffix_tag_struct_[max(0, sample_indx)];}; // sample_idx = -1 indicates no multisample
	unsigned int GetPrefixTagRandomBasesNum(int sample_indx) const { return multisample_prefix_tag_random_bases_idx_[max(0, sample_indx)].size();}; // sample_idx = -1 indicates no multisample
	unsigned int GetSuffixTagRandomBasesNum(int sample_indx) const { return multisample_suffix_tag_random_bases_idx_[max(0, sample_indx)].size();}; // sample_idx = -1 indicates no multisample
	//! Functions for hashing family information to integer or string
	void PreComputeForFamilyIdentification(Alignment* rai);
	bool FamilyInfoToLongLong(int strand_key, const string& zt, bool is_strict_zt, const string& yt, bool is_strict_yt, const vector<int>& target_indices, int sample_idx, unsigned long long& my_hash) const;
	static void LongLongToPrintableStr(unsigned long long v, string& s);
	static void FamilyInfoToReadableStr(int strand_key, const string& zt, const string& yt, const vector<int>& target_indices, string& my_readable_str);
	//! Functions for dehashing integer or string to family information
	string LongLongToFamilyInfo(unsigned long long my_hash, int sample_idx, int& strand_key, string& zt, string& yt, vector<int>& target_indices) const;
	static bool PrintableStrToLongLong(const string& s, unsigned long long& v);
	//! Functions for determining tag similarity
	bool IsPartialSimilarTags(string tag_1, string tag_2, bool is_prefix) const;
	bool IsFlowSynchronizedTags(string tag_1, string tag_2, bool is_prefix) const;
};

class MolecularFamilyGenerator
{
private:
	bool long_long_hashable_ = false;
	vector< map<unsigned long long, unsigned int> > long_long_tag_lookup_table_;
	vector< map<string, unsigned int> > string_tag_lookup_table_;
	bool FindFamilyForOneRead_(Alignment* rai, vector< vector<MolecularFamily> >& my_molecular_families);

public:
	MolecularFamilyGenerator() {};
	void GenerateMyMolecularFamilies(const MolecularTagManager* const mol_tag_manager,
			PositionInProgress& bam_position,
			int sample_index,
            vector< vector<MolecularFamily> >& my_molecular_families);
};

class ConsensusPositionTicketManager {
private:
	static bool debug_;
	static pthread_mutex_t mutex_CPT_;
	static int kNumAppended_;
	static int kNumDeleted_;
public:
	ConsensusPositionTicketManager() {};
	~ConsensusPositionTicketManager() {
		if (debug_){
			cout << "ConsensusPositionTicketManager::kNumAppended_ = " << kNumAppended_ << endl
				 << "ConsensusPositionTicketManager::kNumDeleted_ = " << kNumDeleted_ << endl;
		}
	};
	static void ClearConsensusPositionTicket(list<PositionInProgress>::iterator &consensus_position_ticket);
	static void AppendConsensusPositionTicket(list<PositionInProgress>::iterator& consensus_position_ticket, Alignment* const alignment);
	static void CloseConsensusPositionTicket(list<PositionInProgress>::iterator& consensus_position_ticket);
	static void ReopenConsensusPositionTicket(list<PositionInProgress>::iterator& consensus_position_ticket);
	static int ConsensusPositionTicketCounter(const list<PositionInProgress>::iterator& consensus_position_ticket);
};

void GenerateConsensusPositionTicket(vector< vector< vector<MolecularFamily> > > &my_molecular_families_multisample,
		                             VariantCallerContext &vc,
									 const ConsensusAlignmentManager &consensus,
		                             list<PositionInProgress>::iterator &consensus_position_ticket,
									 bool filter_all_reads_after_done = false);

#endif /* MOLECULARTAG_H */
