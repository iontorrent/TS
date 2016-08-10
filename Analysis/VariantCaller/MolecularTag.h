/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef READFAMILY_H
#define READFAMILY_H


#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <Variant.h>
#include "ExtendParameters.h"
#include "BAMWalkerEngine.h"
#include <sys/time.h>

using namespace std;

class Consensus {
private:
	unsigned int min_start_position_;
	std::map<unsigned int, unsigned int> insertions_;
	std::map<unsigned int, unsigned int> new_insertions_;
	vector<string> aligned_bases_;
	std::basic_string<char>::size_type max_read_length_;
	string insertion_bases_;
	string consensus_;
	vector<CigarOp> new_cigar_;
	bool debug_;
    //double alignment_time = 0;
    //double consensus_time = 0;
    //double trim_time = 0;
    //double cigar_time = 0;
	
	unsigned int GetAlignedFamily(vector<Alignment*>& family_members);
	void GetConsensus(ReferenceReader& ref_reader, unsigned int RefID);
	unsigned int TrimConsensus();
	void CalculateCigar();
	
public:	
	Consensus();
	virtual ~Consensus();
	
	void SetDebug(bool b) {debug_ = b;}
	void GetAlignedBases(vector<string>& v) {v = aligned_bases_;}
	void GetInsertionBases(string& str) {str = insertion_bases_;}
	void GetConsensus(string& str) {str = consensus_;}
	
	bool CalculateConsensus(ReferenceReader& ref_reader, vector<Alignment*>& family_members, Alignment& alignment);
};

template <class MemberType>
class MolecularFamily {
public:
	int strand_key;
	string family_barcode;
	vector<MemberType> family_members;
	vector<MemberType> family_members_temp; // in case we want to apply additional rules to filter out some reads
	Alignment consensus_alignment;
    bool operator<(const MolecularFamily &rhs) const { return this->family_members_temp.size() < rhs.family_members_temp.size(); } // use for std::sort
    bool is_func_family_temp; // is_func_family_temp indicates the functionality of this family if the family members are family_members_temp

	MolecularFamily(const string &barcode, int strand): family_barcode(barcode){
		strand_key = strand;
		ResetFamily();
	};

	void ResetFamily(){
		family_members.clear();
		family_members.reserve(16);
		family_members_temp.clear();
		is_func_family_ = false;
		is_func_family_temp = false;
		consensus_alignment.Reset();
	};
	void AddNewMember(const MemberType &new_member) { family_members.push_back(new_member); };
	bool SetFunctionality(unsigned int min_fam_size){
		is_func_family_ = family_members.size() >= min_fam_size;
		return is_func_family_;
	};
	bool GetFunctionality() const { return is_func_family_; };
	
private:
	bool is_func_family_;
};

// The task of this class is done by MolecularTagTrimmer object whihc is being shared by BaseCaller and TVC
// XXX ======== REMOVE ME! ===============
class MolecularTagClassifier {
public:
	MolecularTagClassifier();
	//void PropagateTagParameters(const MolecularBarcodeParameters &my_tag_params);
	void InitializeTag(const string &a_handle_fwd, const string &prefix_barcode_format_fwd, const string &suffix_barcode_format_fwd);
	bool ClassifyOneRead(string &prefix_barcode, string &suffix_barcode, int &len_prefix_base, int &len_suffix_bases, const string &read_seq);
	bool IsStratsWithAHandle(unsigned int &barcode_start_index, const string &base_seq);
	bool StrictBarcodeClassifier(string &prefix_barcode, string &suffix_barcode, int &prefix_base_len, int &suffix_base_len, const string &base_seq, unsigned int prefix_barcode_start_index);
	bool SloppyBarcodeClassifier(string &prefix_barcode, string &suffix_barcode, int &prefix_base_len, int &suffix_base_len, const string &base_seq, unsigned int prefix_barcode_start_index);

private:
	string a_handle_;
	string prefix_barcode_format_;
	string suffix_barcode_format_;
	unsigned int allow_a_handle_error_num_ = 2;
	bool is_use_strict_barcode_classifier = true;

	// Heal Hp indel related
	bool use_heal_hp_indel_ = false;
	vector<int> heal_indel_len_ = {-1, 1, 2}; // Must be in ascending order! -x means x-mer hp deletion, +x means x-mer hp insertion.
	void SetUseHealHpIndel_();
	bool HealHpIndelOneSegment_(const string &base_segment, const string &flag_segment, int random_base_len, string &barcode_segment, int &indel_len);
	bool HealHpIndel_(string base_seq, string barcode_format, bool is_reverse_mode, string &barcode, int &total_offset);

    // Smith-Waterman related.
	float matchScore_ = 19.0f;
    float mismatchScore_ = -9.0f;
    float gapOpenPenalty_ = 15.0f;
    float gapExtendPenalty_ = 6.66f;
    float entropyGapOpenPenalty_ = 0.0f;
    bool useRepeatGapExtendPenalty_ = false;
    float repeatGapExtendPenalty_ = 1.0f;
};
// =============================================

void GenerateMyMolecularFamilies(PositionInProgress &bam_position,
		                         vector< vector< MolecularFamily<Alignment*> > > &my_molecular_families,
								 const ExtendParameters &parameters,
								 int sample_index = -1);

void RemoveNonFuncFamilies(vector< MolecularFamily<Alignment*> > &my_molecular_families, unsigned int min_fam_size);

#endif /* READFAMILY_H */
