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

void GenerateMyMolecularFamilies(PositionInProgress &bam_position,
		                         vector< vector< MolecularFamily<Alignment*> > > &my_molecular_families,
								 const ExtendParameters &parameters,
								 int sample_index = -1);

void RemoveNonFuncFamilies(vector< MolecularFamily<Alignment*> > &my_molecular_families, unsigned int min_fam_size);

#endif /* MOLECULARTAG_H */
