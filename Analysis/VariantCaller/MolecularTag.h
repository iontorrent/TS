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

class Timer {
private:
  string tag_;
  double run_time_;
  timeval start_time_;
  timeval end_time_;
  pthread_mutex_t mutextimer_;
public:
Timer(const string& tag) {tag_ = tag; run_time_ = 0; pthread_mutex_init(&mutextimer_, NULL);}
virtual ~Timer() {print();}
void start() {gettimeofday(&start_time_, NULL);}
void end() {
   gettimeofday(&end_time_, NULL);
   pthread_mutex_lock (&mutextimer_);
   run_time_ += ((end_time_.tv_sec - start_time_.tv_sec) + ((end_time_.tv_usec - start_time_.tv_usec) / 1000000.0));
   pthread_mutex_unlock (&mutextimer_);
}
void print() {cerr << tag_ << " run_time = " << run_time_ << " seconds." << endl;}
};

class Consensus {
private:
   	bool reverse_;
	string name_;
	string flow_order_;
	unsigned int min_start_position_;
	std::map<unsigned int, unsigned int> insertions_;
	std::map<unsigned int, unsigned int> new_insertions_;
	vector<string> aligned_bases_;
	vector<int> read_counts_;
	vector<vector<int> > flow_indexes_;
	vector<vector<float> > measurement_vector_;
	int start_flow_;
	vector<int> flow_index_;
	vector<float> measurements_;
	vector<vector<float> > phase_params_;
	std::basic_string<char>::size_type max_read_length_;
	string insertion_bases_;
	string consensus_;
	vector<CigarOp> new_cigar_;
	bool debug_;
	bool flow_consensus_;
    int flow_order_index_;
    vector<int> soft_clip_offset_;
    bool error_;
    bool stitch_;
    float iupac_cutoff_;
	
	unsigned int GetAlignedFamily(vector<Alignment*>& family_members);
	void GetConsensus(ReferenceReader& ref_reader, unsigned int RefID);
	unsigned int TrimConsensus();
	void CalculateCigar();
	void PartiallyResetAlignment_(Alignment& alignment);

public:	
	Consensus();
	virtual ~Consensus();
	
	void SetIUPACCutoff(float f) {iupac_cutoff_ = f;}
	void SetStitch(bool b) {stitch_ = true;}
	void SetFlowConsensus(bool b) {flow_consensus_ = b;}
	void SetDebug(bool b) {debug_ = b;}
	void GetAlignedBases(vector<string>& v) {v = aligned_bases_;}
	void GetInsertionBases(string& str) {str = insertion_bases_;}
	void GetConsensus(string& str) {str = consensus_;}
	bool CalculateConsensus(ReferenceReader& ref_reader, vector<Alignment*>& family_members, Alignment& alignment, const string& flow_order = "");
};

template <class MemberType>
class AbstractMolecularFamily {
public:
	int strand_key;
	string family_barcode;
	vector<MemberType> all_family_members;   // All reads identified in the family, no additional filtering applied.
	vector<MemberType> valid_family_members; // The reads from all_family_members that pass certain filtering criterion.

    AbstractMolecularFamily (const string &barcode = "", int strand = -1)
    	: strand_key(strand), family_barcode(barcode) { all_family_members.reserve(4); };

    virtual ~AbstractMolecularFamily(){};
    virtual int CountFamSizeFromAll() = 0;
    virtual int CountFamSizeFromValid() = 0;

	bool SetFuncFromAll(unsigned int min_fam_size){
		if (fam_size_ < 0) {CountFamSizeFromAll(); }
		is_func_from_family_members = (fam_size_ >= (int) min_fam_size);
		return is_func_from_family_members;
	};

	bool SetFuncFromValid(unsigned int min_fam_size){
		if (valid_fam_size_ < 0) {CountFamSizeFromValid();}
		is_func_from_valid_family_members = (valid_fam_size_ >= (int) min_fam_size);
		return is_func_from_valid_family_members;
	};

	void AddNewMember(const MemberType& new_member)
		{ all_family_members.push_back(new_member); };

	bool GetFuncFromAll() const {
		assert(fam_size_ > -1);  // CountFamSizeFromAll must be done first.
		return is_func_from_family_members;
	};

	bool GetFuncFromValid() const {
		assert(valid_fam_size_ > -1);  // CountFamSizeFromValid must be done first.
		return is_func_from_valid_family_members;
	};

	int GetFamSize() {
		if (fam_size_ < 0) { CountFamSizeFromAll(); }
		return fam_size_;
	};

	int GetValidFamSize() {
		if (valid_fam_size_ < 0) { CountFamSizeFromValid(); }
		return valid_fam_size_;
	};

    void ResetValidFamilyMembers() {
    	valid_family_members.resize(0);
    	valid_family_members.reserve(all_family_members.size());
    	valid_fam_size_ = -1;
    	is_func_from_valid_family_members = false;
    };

protected:
	bool is_func_from_family_members = false;
	bool is_func_from_valid_family_members = false;
    int fam_size_ = -1;  // Total read counts in all_family_members (a consensus read counted by its read_count)
    int valid_fam_size_ = -1; // Total read counts in valid_family_members (a consensus read counted by its read_count)
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

public:
	MolecularTagManager();
	MolecularTagTrimmer* tag_trimmer;
	void Initialize(MolecularTagTrimmer* const input_tag_trimmer, const SampleManager* const sample_manager);
	bool IsStrictTag(const string& prefix_tag, const string& suffix_tag, int sample_idx) const;
	bool IsStrictPrefixTag(const string& prefix_tag, int sample_idx) const;
	bool IsStrictSuffixTag(const string& suffix_tag, int sample_idx) const;
	string GetPrefixTagStruct(int sample_indx) const {return multisample_prefix_tag_struct_[max(0, sample_indx)];}; // sample_idx = -1 indicates no multisample
	string GetSuffixTagStruct(int sample_indx) const {return multisample_suffix_tag_struct_[max(0, sample_indx)];}; // sample_idx = -1 indicates no multisample
	bool IsPartialSimilarTags(string tag_1, string tag_2, bool is_prefix) const;
	bool IsFlowSynchronizedTags(string tag_1, string tag_2, bool is_prefix) const;
};

class MolecularFamilyGenerator
{
private:
	bool long_long_hashable_ = false;
	const bool is_split_families_by_region_ = true; // I will always split families by region.
	vector< map<long long, unsigned int> > long_long_tag_lookup_table_;
	vector< map<string, unsigned int> > string_tag_lookup_table_;
	void SplitFamiliesByRegion_(vector< vector<MolecularFamily> >& my_molecular_families) const;
	void FindFamilyForOneRead_(Alignment* rai, vector< vector<MolecularFamily> >& my_molecular_families);
	long long BaseSeqToLongLong_(const string& base_seq) const;
	char NucTo0123_(char nuc) const;
public:
	MolecularFamilyGenerator() {};
	void GenerateMyMolecularFamilies(const MolecularTagManager* const mol_tag_manager,
			PositionInProgress& bam_position,
			int sample_index,
            vector< vector<MolecularFamily> >& my_molecular_families);
};


void GenerateConsensusPositionTicket(vector< vector< vector<MolecularFamily> > > &my_molecular_families_multisample,
		                             VariantCallerContext &vc,
		                             Consensus &consensus,
		                             list<PositionInProgress>::iterator &consensus_position_ticket,
									 bool filter_all_reads_after_done = false);

void GenerateCandidatesFromConsensusPositionTicket(AlleleParser* candidate_generator,
		                                           const BAMWalkerEngine* bam_walker,
		                                           deque<VariantCandidate>& variant_candidates,
												   list<PositionInProgress>::iterator& consensus_position_ticket,
												   int haplotype_length);

#endif /* MOLECULARTAG_H */
