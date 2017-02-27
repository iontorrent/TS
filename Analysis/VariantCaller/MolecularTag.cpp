/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MolecularTag.h"
#include "MiscUtil.h"

Consensus::Consensus() {
	min_start_position_ = 0;
	max_read_length_ = 0;
	debug_ = false;
}

Consensus::~Consensus() {
}

unsigned int Consensus::GetAlignedFamily(vector<Alignment*>& family_members) {
	min_start_position_ = 999999999;
	insertions_.clear();
	aligned_bases_.clear();
	insertion_bases_.clear();
	max_read_length_ = 0;
	
	unsigned int map_quality = 0;
	for (vector<Alignment*>::iterator iter = family_members.begin(); (iter != family_members.end()); ++iter) {
		if (debug_) {
			cerr << "RefID:Position " << (*iter)->alignment.RefID << ":" << (*iter)->alignment.Position << endl;
			cerr << "start          " << (*iter)->start << endl;
			cerr << "end            " << (*iter)->end << endl;
			cerr << "QueryBases     " << (*iter)->alignment.QueryBases << endl;
			cerr << "AlignedBases   " << (*iter)->alignment.AlignedBases << endl;
		}
		map_quality += (*iter)->alignment.MapQuality;
		min_start_position_ = min(min_start_position_, (unsigned int)((*iter)->alignment.Position));
		unsigned int offset = 0;
		for (unsigned int position = min_start_position_; (position < (unsigned int)((*iter)->alignment.Position)); ++position) {offset++;}
		max_read_length_ = max(max_read_length_, (std::basic_string<char>::size_type)(((*iter)->alignment.AlignedBases.length() + offset)));
		unsigned int pos = 0;
		for (vector<CigarOp>::iterator iterCigar = (*iter)->alignment.CigarData.begin(); (iterCigar != (*iter)->alignment.CigarData.end()); ++iterCigar) {
			if (debug_) {cerr << iterCigar->Type << " " << iterCigar->Length << endl;}
			if (iterCigar->Type == 'I') {
				std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.find(pos + offset);
				if (iterInsertions == insertions_.end()) {insertions_[pos + offset] = iterCigar->Length;}
				else {insertions_[pos + offset] = max(iterInsertions->second, iterCigar->Length);}
			}
			if (iterCigar->Type == 'M') {pos += iterCigar->Length;}
		}
	}
	map_quality /= family_members.size();
	unsigned int count = 0;
	for (std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.begin(); (iterInsertions != insertions_.end()); ++iterInsertions) {
		count += iterInsertions->second;
	}
	max_read_length_ += count;
	insertion_bases_.assign(max_read_length_, '-');
	for (vector<Alignment*>::iterator iter = family_members.begin(); (iter != family_members.end()); ++iter) {
		string str = (*iter)->alignment.AlignedBases;
        str.reserve(max_read_length_);
		unsigned int insertion_count = 0;
		unsigned int offset = 0;
		for (unsigned int position = min_start_position_; (position < (unsigned int)((*iter)->alignment.Position)); ++position) {str = '-' + str; insertion_count++; offset++;}
		while (str.length() < max_read_length_) {str += "-";}
		std::map<unsigned int, unsigned int> read_insertions;
		unsigned int pos = 0;
		for (vector<CigarOp>::iterator iterCigar = (*iter)->alignment.CigarData.begin(); (iterCigar != (*iter)->alignment.CigarData.end()); ++iterCigar) {
			if (iterCigar->Type == 'I') {
				read_insertions[pos + offset] = iterCigar->Length;
			}
			if (iterCigar->Type == 'M') {pos += iterCigar->Length;}
		}
		for (std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.begin(); (iterInsertions != insertions_.end()); ++iterInsertions) {
			for (unsigned int pos = iterInsertions->first + insertion_count - offset; (pos < (iterInsertions->first + insertion_count - offset + iterInsertions->second)); ++pos) {
				if (pos < insertion_bases_.length()) {insertion_bases_[pos] = '*';}
			}
			std::map<unsigned int, unsigned int>::iterator iterReadInsertions = read_insertions.find(iterInsertions->first);
			if (iterReadInsertions == read_insertions.end()) {
				string part1 = str;
				if (iterInsertions->first + insertion_count - offset < str.length()) {part1 = str.substr(0, iterInsertions->first + insertion_count - offset);}
				string part2 = "";
				if (iterInsertions->first + insertion_count - offset < str.length()) {part2 = str.substr(iterInsertions->first + insertion_count - offset);}
				str = part1;
				str.reserve(max_read_length_);
				for (int index = 0; (index < (int)iterInsertions->second); ++index) {
					str += "*";
					insertion_count++;
				}
				str += part2;
			}
			else {
				string part1 = "";
				if (iterInsertions->first + iterReadInsertions->second + insertion_count - offset < str.length()) {part1 = str.substr(0, iterInsertions->first + iterReadInsertions->second + insertion_count - offset);}
				string part2 = "";
				if (iterReadInsertions->first + iterReadInsertions->second + insertion_count - offset < str.length()) {part2 = str.substr(iterReadInsertions->first + iterReadInsertions->second + insertion_count - offset);}
				str = part1;
				str.reserve(max_read_length_);
				for (int index = 0; (index < (int)(iterInsertions->second - iterReadInsertions->second)); ++index) {
					str += "*";
				}
				insertion_count += iterInsertions->second;
				str += part2;
				
			}
		}
		aligned_bases_.push_back(str);
	}
	
	max_read_length_ = 0;
	for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
		max_read_length_ = max(max_read_length_, (std::basic_string<char>::size_type)(iter->length()));
		if (debug_) {cerr << *iter << endl;}
	}
	if (debug_) {cerr << endl << insertion_bases_ << endl;}
	return map_quality;
}

void Consensus::GetConsensus(ReferenceReader& ref_reader, unsigned int RefID) {
	consensus_ = "";
    consensus_.reserve(max_read_length_);
	int A_count = 0;
	int C_count = 0;
	int G_count = 0;
	int T_count = 0;
	int N_count = 0;
	int D_count = 0;
    int I_count = 0;
	unsigned int position = min_start_position_;
	for (unsigned int index = 0; (index < max_read_length_); ++index) {
		A_count = 0;
		C_count = 0;
		G_count = 0;
		T_count = 0;
		N_count = 0;
		D_count = 0;
        I_count = 0;
		for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
			if (index >= iter->length()) {D_count++;}
			else {
				switch (iter->at(index)) {
					case 'A': A_count++; break;
					case 'C': C_count++; break;
					case 'G': G_count++; break;
					case 'T': T_count++; break;
					case '-': D_count++; break;
					case '*': I_count++; break;
					default: N_count++;
				}
			}
		}
		char ref_base = 0;
		if (insertion_bases_[index] != '*') {ref_base = (char)(*(ref_reader.iter(RefID, position)));}
		int max_count = max(max(max(max(max(max(A_count, C_count), G_count), T_count), N_count), D_count), I_count);
		if ((ref_base == 'G') and (max_count == G_count)) {C_count = 0; T_count = 0; A_count = 0; D_count = 0;}
		else if ((ref_base == 'C') and (max_count == C_count)) {G_count = 0; T_count = 0; A_count = 0; D_count = 0;}
		else if ((ref_base == 'T') and (max_count == T_count)) {C_count = 0; G_count = 0; A_count = 0; D_count = 0;}
		else if ((ref_base == 'A') and (max_count == A_count)) {C_count = 0; G_count = 0; T_count = 0; D_count = 0;}
		if (max_count == I_count) {consensus_ += '-';}
		else if (max_count == G_count) {consensus_ += 'G';}
		else if (max_count == C_count) {consensus_ += 'C';}
		else if (max_count == T_count) {consensus_ += 'T';}
		else if (max_count == A_count) {consensus_ += 'A';}
		else if (max_count == D_count) {consensus_ += '-';}
		else if (max_count == N_count) {consensus_ += 'N';}
		if (insertion_bases_[index] != '*') {position++;}
	}
}

unsigned int Consensus::TrimConsensus() {
	std::vector<unsigned int> offsets;
	unsigned int insertion_count = 0;
	for (std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.begin(); (iterInsertions != insertions_.end()); ++iterInsertions) {
		offsets.push_back(insertion_count);			
		insertion_count += iterInsertions->second;
	}
	new_insertions_.clear();
	// trim insertions
	int offset_index = insertions_.size();
	for (std::map<unsigned int, unsigned int>::reverse_iterator iterInsertions = insertions_.rbegin(); (iterInsertions != insertions_.rend()); ++iterInsertions) {
		offset_index--;
		for (unsigned int pos = (min((unsigned int)consensus_.length(), iterInsertions->first + iterInsertions->second + offsets[offset_index]) - 1); (pos >= iterInsertions->first + offsets[offset_index]); --pos) {
            if (consensus_[pos] == '-') {
                if (pos == consensus_.length() - 1) {consensus_ = consensus_.substr(0, pos);}
                else {
				    consensus_ = consensus_.substr(0, pos) + consensus_.substr(pos + 1);
                }
				iterInsertions->second--;
			}
		}
	}	
	// trim begining of consensus
	unsigned int new_position = min_start_position_;
	unsigned int removed_count = 0;
	for (unsigned int index = 0; (index < consensus_.length()); ++index) {
        if (index >= consensus_.length()) {break;}
		if (consensus_[index] == '-') {
			if (index + 1 >= consensus_.length()) {consensus_ = "";}
			else {consensus_ = consensus_.substr(index + 1);}
			removed_count++;
			new_position++;
			index--;
		} else {break;}
	}	
	// adjust insertion positions	
	for (std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.begin(); (iterInsertions != insertions_.end()); ++iterInsertions) {
		if (iterInsertions->second != 0) {new_insertions_[iterInsertions->first - removed_count] = iterInsertions->second;}
	}
	// trim end of consensus
	for (int index = (consensus_.length() - 1); (index >= 0); --index) {
		if (consensus_[index] == '-') {
			consensus_ = consensus_.substr(0, index);
		} else {break;}
	}
	return new_position;
}

void Consensus::CalculateCigar() {
	// calculate new cigar
	new_cigar_.clear();
	unsigned int pos = 0;
	unsigned int offset = 0;
	bool end = false;
	for (std::map<unsigned int, unsigned int>::iterator iterInsertions = new_insertions_.begin(); (iterInsertions != new_insertions_.end()); ++iterInsertions) {
		if (iterInsertions->first > 0) {
			unsigned int m_count = 0;
			unsigned int deletion_count = 0;
			for (unsigned int index = 0; (index < (iterInsertions->first + offset - pos)); ++index) {
				if (pos + index >= consensus_.length()) {end = true; break;}
				if (consensus_[pos + index] == '-') {
					deletion_count++; offset++;
				}
				else {
					if (deletion_count > 0) {
						if (m_count > 0) {new_cigar_.push_back(CigarOp('M', m_count)); m_count = 0;}
						new_cigar_.push_back(CigarOp('D', deletion_count)); 
						deletion_count = 0;
					}
					m_count++;
				}
			}
			if (m_count > 0) {new_cigar_.push_back(CigarOp('M', m_count)); m_count = 0;}
			pos = iterInsertions->first + offset;
		}
		if (!end) {new_cigar_.push_back(CigarOp('I', iterInsertions->second)); offset += iterInsertions->second; pos += iterInsertions->second;}
	}
	unsigned int m_count = 0;
	unsigned int deletion_count = 0;
	for (unsigned int index = 0; (index < (consensus_.length() - pos)); ++index) {
		if (pos + index >= consensus_.length()) {break;}
		if (consensus_[pos + index] == '-') {
			deletion_count++;
		}
		else {
			if (deletion_count > 0) {
				if (m_count > 0) {new_cigar_.push_back(CigarOp('M', m_count)); m_count = 0;}
				new_cigar_.push_back(CigarOp('D', deletion_count));
				deletion_count = 0;
			}
			m_count++;
		}
	}
	if (m_count > 0) {new_cigar_.push_back(CigarOp('M', m_count)); m_count = 0;}
}

bool Consensus::CalculateConsensus(ReferenceReader& ref_reader, vector<Alignment*>& family_members, Alignment& alignment) {
	if (family_members.size() < 1) {return false;}
	alignment = *(family_members[0]);
    //cerr << endl << family_barcode << " " << family_members.size() << " " << family_members[0]->alignment.RefID << " " << family_members[0]->alignment.Position << endl;
	//debug_ = false;
	//if ((family_members[0]->alignment.Position <= 55259524) and (family_members[0]->alignment.GetEndPosition() >= 55259524)) {debug_ = true;}
	//if (alignment.alignment.Name == "TSJAM:02884:02345") {debug_ = true;}
	//if ((family_barcode == "TTGACTTTGTGATCATAAAGTCAA") and (family_members.size() == 15) and (family_members[0]->alignment.Position == 29443576)) {debug_ = true;}
	if (debug_) {cerr << endl << family_members.size() << " " << family_members[0]->alignment.RefID << " " << family_members[0]->alignment.Position << endl;}
	
	min_start_position_ = 0;
	insertions_.clear();
	aligned_bases_.clear();
	max_read_length_ = 0;
	consensus_.clear();
	insertion_bases_.clear();
	new_insertions_.clear();
	new_cigar_.clear();
	
    //timeval start_time;
    //timeval end_time;
    //gettimeofday(&start_time, NULL);
	unsigned int map_quality = GetAlignedFamily(family_members);
    //gettimeofday(&end_time, NULL);
    //alignment_time += ((end_time.tv_sec - start_time.tv_sec) + ((end_time.tv_usec - start_time.tv_usec) / 1000000.0));
	if (debug_) {cerr << "GetConsensus" << endl;}
    //gettimeofday(&start_time, NULL);
	GetConsensus(ref_reader, alignment.alignment.RefID);
    //gettimeofday(&end_time, NULL);
    //consensus_time += ((end_time.tv_sec - start_time.tv_sec) + ((end_time.tv_usec - start_time.tv_usec) / 1000000.0));
	if (debug_) {cerr << "TrimConsensus" << endl;}
    //gettimeofday(&start_time, NULL);
	unsigned int new_position = TrimConsensus();
    //gettimeofday(&end_time, NULL);
    //trim_time += ((end_time.tv_sec - start_time.tv_sec) + ((end_time.tv_usec - start_time.tv_usec) / 1000000.0));
	if (debug_) {cerr << "CalculateCigar" << endl;}
    //gettimeofday(&start_time, NULL);
	CalculateCigar();
    //gettimeofday(&end_time, NULL);
    //cigar_time += ((end_time.tv_sec - start_time.tv_sec) + ((end_time.tv_usec - start_time.tv_usec) / 1000000.0));
	string query_bases = "";
	query_bases.reserve(max_read_length_);
	for (unsigned int index = 0; (index < consensus_.length()); ++index) {
		if (consensus_[index] != '-') {query_bases += consensus_[index];}
	}
	long int start = new_position;
	long int end = new_position;
	for (vector<CigarOp>::iterator iterCigar = new_cigar_.begin(); (iterCigar != new_cigar_.end()); ++iterCigar) {
		if (iterCigar->Type == 'M') {end += iterCigar->Length;}
	}
	
	// output new consensus and cigar
	if (debug_) {
		bool run_exit = false;
		cerr << endl;
        cerr << alignment.alignment.Name << endl;
		cerr << start << endl;
		cerr << end << endl;
		cerr << new_position << endl;
		cerr << query_bases << endl;
		cerr << consensus_ << endl;
		for (vector<CigarOp>::iterator iterCigar = new_cigar_.begin(); (iterCigar != new_cigar_.end()); ++iterCigar) {
			cerr << iterCigar->Type << " " << iterCigar->Length << endl;
			if (iterCigar->Type == 'D') {run_exit = true;}
		}
		//if (new_position == 115256481) {run_exit = true;}
		if (run_exit) {
			//exit(1);
		}
        //cerr << "alignment " << alignment_time << endl;
        //cerr << "consensus " << consensus_time << endl;
        //cerr << "trim " << trim_time << endl;
        //cerr << "cigar " << cigar_time << endl;
        cerr << endl;
		cerr << "Done" << endl;
	}
	alignment.next = NULL;
	alignment.processing_prev = NULL;
	alignment.processing_next = NULL;
	alignment.start = start;
	alignment.end = end;
	alignment.alignment.Position = new_position;
	alignment.alignment.MapQuality = map_quality;
	alignment.alignment.QueryBases = query_bases;
	alignment.alignment.Length = query_bases.length();
	alignment.alignment.AlignedBases = consensus_;
	alignment.alignment.CigarData = new_cigar_;
	alignment.refmap_start.clear();
	alignment.refmap_code.clear();
	alignment.refmap_has_allele.clear();
	alignment.refmap_allele.clear();
	alignment.read_count = family_members.size();
	return true;
}


long long int NucToLongLong(char nuc) {
    if (nuc=='A' or nuc=='a') return 0;
    if (nuc=='C' or nuc=='c') return 1;
    if (nuc=='G' or nuc=='g') return 2;
    if (nuc=='T' or nuc=='t') return 3;
    return -1;
}

//@TODO: This can be done even faster by using memset
long long BarcodeStrToLongLong(const string &barcode_str){
	// I currently support the molecular barcode with total length < 32.
	if(barcode_str.size() > 31){
		cerr<<"Warning: Length of the molecular barcode "<< barcode_str << " > 31 !"<<endl;
		return -1;
	}
    long long barcode_long_long = 0;
    for(string::const_iterator it = barcode_str.begin(); it != barcode_str.end(); ++it){
    	long long nuc_long_long = NucToLongLong(*it);
    	if(nuc_long_long < 0){
    		return -1; // I see something not TACG
    	}
    	barcode_long_long <<= 2;
        barcode_long_long |= nuc_long_long;
    }
    return barcode_long_long;
}


void GenerateMyMolecularFamilies(PositionInProgress &bam_position,
                                 vector< vector< MolecularFamily<Alignment*> > > &my_molecular_families,
								 const ExtendParameters &parameters,
		                         int sample_index){
	//MolecularTagClassifier mol_tag_classifier;
	vector< map<long long, unsigned int> > barcode_lookup_table; // lookup table for barcodes
	barcode_lookup_table.resize(2);
	//mol_tag_classifier.PropagateTagParameters(parameters.my_mol_param);
	// my_molecular_families[0] for fwd strand, my_molecular_families[1] for rev strand
	my_molecular_families.clear();
	my_molecular_families.resize(2);
	my_molecular_families[0].reserve(6000); // reserve 6000 families on each strand
	my_molecular_families[1].reserve(6000);

	for(Alignment* rai = bam_position.begin; rai != bam_position.end; rai = rai->next){
		if (rai == NULL) {bam_position.end = NULL; break;}
	    // skip the read if it is not for this sample
		if(parameters.multisample){
		    if(rai->sample_index != sample_index){
			    continue;
	        }
		}
	    if(rai->filtered or (not rai->tag_info.HasTags())){
	      continue;
	    }

	    string barcode_of_read = rai->tag_info.prefix_mol_tag + rai->tag_info.suffix_mol_tag;
	    long long barcode_long_long = BarcodeStrToLongLong(barcode_of_read);

	    if(barcode_long_long < 0){
			cerr<<"Warning: Skip the unsupported molecular barcode "<< barcode_of_read << " !"<< endl;
			rai->filtered = true;
			continue;
	    }

	    int strand_key = (rai->is_reverse_strand)? 1 : 0;

		pair< map<long long, unsigned int>::iterator, bool> find_barcode;
		// map barcode_of_read to the index of my_family_[strand_key] for barcode_of_read
		find_barcode = barcode_lookup_table[strand_key].insert(pair<long long, unsigned int>(barcode_long_long, my_molecular_families[strand_key].size()));
		// Note that
		// find_barcode.first->first == barcode_of_read
		// find_barcode.first->second is the index of my_family_[strand_key] for barcode_of_read
		// find_barcode.second == true if we got barcode_of_read previously and hence we don't change barcode_lookup_table
		// find_barcode.second == false if this is the first time we get barcode_of_read and we insert the barcode_lookup_table into barcode_lookup_table[strand_key]

		// Initialize if this is the first time we get the barcode
		if(find_barcode.second){
			my_molecular_families[strand_key].push_back(MolecularFamily<Alignment*>(barcode_of_read, strand_key));
		}
		// add the read to the family
		// note that find_barcode.first->second == barcode_lookup_table[strand_key][barcode_of_read]
		my_molecular_families[strand_key][find_barcode.first->second].AddNewMember(rai);
    }
}

// Remove the non-functional families in the vector
void RemoveNonFuncFamilies(vector< MolecularFamily<Alignment*> > &my_molecular_families, unsigned int min_fam_size){
	vector< MolecularFamily<Alignment*> > molecular_families_temp;
	molecular_families_temp.reserve(my_molecular_families.size());
	for(vector< MolecularFamily<Alignment*> >::iterator fam_it = my_molecular_families.begin();
			fam_it != my_molecular_families.end(); ++fam_it){
		if(fam_it->SetFunctionality(min_fam_size)){
			molecular_families_temp.push_back(*fam_it);
		}
	}
	my_molecular_families.swap(molecular_families_temp);
}
