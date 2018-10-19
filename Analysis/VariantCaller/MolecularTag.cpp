/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MolecularTag.h"
#include "MiscUtil.h"

int MolecularFamily::CountFamSizeFromAll()
{
	fam_size_ = 0;
	fam_cov_fwd_ = 0;
	fam_cov_rev_ = 0;
	for (vector<Alignment*>::iterator read_it = all_family_members.begin(); read_it != all_family_members.end(); ++read_it){
		if ((*read_it)->is_reverse_strand){
			fam_cov_rev_ += (*read_it)->read_count;
		}
		else{
			fam_cov_fwd_ += (*read_it)->read_count;
		}
	}
	fam_size_ = fam_cov_fwd_ + fam_cov_rev_;
	return fam_size_;
}

int MolecularFamily::CountFamSizeFromValid()
{
	valid_fam_size_ = 0;
	valid_fam_cov_fwd_ = 0;
	valid_fam_cov_rev_ = 0;
	for (vector<Alignment*>::iterator read_it = valid_family_members.begin(); read_it != valid_family_members.end(); ++read_it){
		if ((*read_it)->is_reverse_strand){
			valid_fam_cov_rev_ += (*read_it)->read_count;
		}
		else{
			valid_fam_cov_fwd_ += (*read_it)->read_count;
		}
	}
	valid_fam_size_ = valid_fam_cov_fwd_ + valid_fam_cov_rev_;
	return valid_fam_size_;
}

bool CompareByReadCounts(const Alignment* const rai_1, const Alignment* const rai_2){
	return rai_1->read_count > rai_2->read_count;
}

void MolecularFamily::SortAllFamilyMembers(){
	sort(all_family_members.begin(), all_family_members.end(), CompareByReadCounts);
	is_all_family_members_sorted = true;
}

void MolecularFamily::SortValidFamilyMembers(){
	sort(valid_family_members.begin(), valid_family_members.end(), CompareByReadCounts);
	is_valid_family_members_sorted = true;
}

void MolecularFamily::ResetValidFamilyMembers(){
	AbstractMolecularFamily::ResetValidFamilyMembers();
	is_valid_family_members_sorted = false;
}

ConsensusAlignmentManager::ConsensusAlignmentManager(ReferenceReader const * const ref_reader, bool debug){
	SetReferenceReader(ref_reader);
	SetDebug(debug);
}

void ConsensusAlignmentManager::GetMajority_(const map<string, pair<unsigned int, unsigned int> >& read_count_dict, string& majority_key) const{
	assert(not read_count_dict.empty());
	map<string, pair<unsigned int, unsigned int> >::const_iterator current_best_it = read_count_dict.begin();
	map<string, pair<unsigned int, unsigned int> >::const_iterator my_it = read_count_dict.begin();
	for (++my_it; my_it != read_count_dict.end(); ++my_it){
		if (my_it->second.first > current_best_it->second.first){
			current_best_it = my_it;
		}else if (my_it->second.first == current_best_it->second.first){
			// Tie in the read count. I use the smallest "read_number" as the tie breaker to get a consistent alignment.
			// For example, two reads with the same read counts
			// REF:     T A C G T A C G
			// Read #1: T A C A T A C G
			// Read #2: T A C G C A C G
			// I want to get a consensus that is either Read #1 or #2
			// Not      T A C G T A C G
			// or       T A C A C A C G
			if (my_it->second.second < current_best_it->second.second){
				current_best_it = my_it;
			}
		}
	}
	majority_key = current_best_it->first;
}

void ConsensusAlignmentManager::InsertToDict_(map<string, pair<unsigned int, unsigned int> >& read_count_dict, const string& my_key, unsigned int my_count, unsigned int my_read_number) const {
	pair<string, pair<unsigned int, unsigned int> > insert_me(my_key, {my_count, my_read_number});
	pair< map<string, pair<unsigned int, unsigned int> >::iterator, bool> key_finder = read_count_dict.insert(insert_me);
	if (not key_finder.second){
		key_finder.first->second.first += my_count;
		// my_read_number will be used to be the tie breaker (my_read_number is unique).
		key_finder.first->second.second = min(my_read_number, key_finder.first->second.second);
	}
}

bool ConsensusAlignmentManager::CalculateConsensusStrand_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const{
	if (family_members.empty()){
		return false;
	}
	cons_alignment.is_reverse_strand = family_members[0]->is_reverse_strand;
	cons_alignment.alignment.SetIsReverseStrand(cons_alignment.is_reverse_strand);
	for (unsigned int idx = 1; idx < family_members.size(); ++idx){
		if (family_members[idx]->is_reverse_strand != cons_alignment.is_reverse_strand){
			return false;
		}
	}
	return true;
}

// Calculate the smallest possible align_start and the largest possible align_end. Note that the aligned interval may be shrinked later.
bool ConsensusAlignmentManager::CalculateConsensusPositions_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const{
	if (family_members.empty()){
		return false;
	}
	cons_alignment.alignment.RefID = family_members[0]->alignment.RefID;
	int32_t start_position = -1;
	int end_position = 0;
	for (vector<Alignment*>::const_iterator read_it = family_members.begin(); read_it != family_members.end(); ++read_it){
		start_position = (start_position >= 0)? min((*read_it)->alignment.Position, start_position) : (*read_it)->alignment.Position;
		end_position = max(end_position, (*read_it)->alignment.GetEndPosition(false, true));
		// Every read should have the same RefID.
		if (cons_alignment.alignment.RefID != (*read_it)->alignment.RefID){
			return false;
		}
	}

	cons_alignment.alignment.Position = start_position;
	cons_alignment.original_position = (int) start_position;
	cons_alignment.align_start = (int) start_position;
	cons_alignment.align_end = end_position;
	cons_alignment.right_sc = 0;
	cons_alignment.left_sc = 0;
	// Note that the consensus query sequence and its length is not ready yet.
	bool success = start_position >= 0
					and end_position >= (int) start_position
					and end_position < ref_reader_->chr_size(cons_alignment.alignment.RefID);
	return success;
}


bool ConsensusAlignmentManager::CalculateConsensus(const vector<Alignment*>& family_members, Alignment& cons_alignment) const {
	// Empty family_members
	if (family_members.empty()) {
		return false;
	}

	// Calculate the family size and find the read w/ max read count.
	unsigned int max_read_count_idx = 0;
	int max_read_count = family_members[0]->read_count;
	int fam_size = max_read_count;
	for (unsigned int read_idx = 1; read_idx < family_members.size(); ++read_idx){
		int read_count = family_members[read_idx]->read_count;
		fam_size += read_count;
		if (read_count > max_read_count){
			max_read_count = read_count;
			max_read_count_idx = read_idx;
		}
	}

	// Trivial consensus alignment: use the alignment of the read that dominants the family
	// Check if family_members has a consensus read that dominates all the others.
	// It should be the case most of the time if the input bam is a consensus bam.
	if (2 * max_read_count >= fam_size){
		int template_idx = max_read_count_idx;
		if (family_members.size() == 2 and 2 * max_read_count == cons_alignment.read_count){
			// The family consists of two reads that have equal read count.
			// I pick the read with the better mapping quality to be the basespace consensus.
			// In this case, I implicitly prefer to get the reference alleles.
			template_idx = (family_members[0]->alignment.MapQuality > family_members[1]->alignment.MapQuality)? 0 : 1;
		}
		PartiallyCopyAlignmentFromAnother_(cons_alignment, *(family_members[template_idx]), fam_size);
		return true;
	}

	// Now I calculate the non-trivial consensus alignment.
	cons_alignment = Alignment(); // Create a new Alignment object.
	cons_alignment.read_count = fam_size;

	if (not CalculateConsensusStrand_(family_members, cons_alignment)){
		cons_alignment.Reset();
		PrintFailureMessage_(family_members, "Fail to calculate consensus strand direction.");
		return false;
	}
	if (not CalculateConsensusPositions_(family_members, cons_alignment)){
		cons_alignment.Reset();
		PrintFailureMessage_(family_members, "Fail to calculate consensus positions.");
		return false;
	}
	if (not CalculateConsensusQueryAndCigar_(family_members, cons_alignment)){
		cons_alignment.Reset();
		PrintFailureMessage_(family_members, "Fail to calculate consensus query bases and cigar.");
		return false;
	}
	if (not FinalizeConsensusRead_(family_members, cons_alignment)){
		cons_alignment.Reset();
		PrintFailureMessage_(family_members, "Fail to finalize the consensus read.");
		return false;
	}

	PrintVerbose_(family_members, cons_alignment);
	return true;
}

bool ConsensusAlignmentManager::FinalizeConsensusRead_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const{
	if (family_members.empty()
			or cons_alignment.alignment.QueryBases.empty()
			or cons_alignment.alignment.CigarData.empty()
			or cons_alignment.pretty_aln.empty()
			or cons_alignment.read_count < 1
			or cons_alignment.alignment.Position != cons_alignment.align_start
			or cons_alignment.align_end != cons_alignment.alignment.GetEndPosition(false, true)){
		return false;
	}

	// Deal with the rest of the tags
	cons_alignment.alignment.AlignedBases = cons_alignment.alignment.QueryBases;  // No soft/hard-clipping => AlignedBases = QueryBases
	cons_alignment.alignment.Length = (int32_t) cons_alignment.alignment.QueryBases.length();
	cons_alignment.alignment.Qualities = "*";

	// MAPQ = weighted (by ZR) average of the MAPQ of reads
	int mapq = 0;
	for (vector<Alignment*>::const_iterator read_it = family_members.begin(); read_it != family_members.end(); ++read_it){
		mapq += ((*read_it)->read_count * (int) (*read_it)->alignment.MapQuality);
	}
	cons_alignment.alignment.MapQuality = (uint16_t) (mapq / cons_alignment.read_count);

	// Remove the tags
	const static vector<string> tags_to_be_removed = {"XM", "NM", "MD", "ZF", "ZP", "ZM", "ZS", "ZR", "ZN", "ZC", "ZG", "ZA", "RG"};
	for (vector<string>::const_iterator tag_it = tags_to_be_removed.begin(); tag_it != tags_to_be_removed.end(); ++tag_it){
		cons_alignment.alignment.RemoveTag(*tag_it);
	}

	// Add RG tag
	cons_alignment.alignment.AddTag("RG", "Z", family_members[0]->read_group);  // Use the RG of the first read which will be used to identify the sample.

	// Add ZR tag
	cons_alignment.alignment.AddTag("ZR", "i", cons_alignment.read_count); // ZR tag = read count, number of reads that form the consensus read

	// Add ZN tag
	string read_names = "";
	read_names.reserve(family_members.size() * (family_members[0]->alignment.Name.size() + 1));
	for (unsigned int i_member = 0; i_member < family_members.size(); ++i_member){
		read_names += (family_members[i_member]->alignment.Name + ";");
	}
	read_names.resize(read_names.size() - 1); // remove the last ";"
	for (string::iterator c_it = read_names.begin(); c_it != read_names.end(); ++c_it){
		if (*c_it == ':'){
			*c_it = '.';
		}
	}
	cons_alignment.alignment.AddTag("ZN", "Z", read_names);  // ZN tag = query names of the reads that from the consensus read

	// Add NM (Number of Mismatches) tag
	int nm = 0;
	ReferenceReader::iterator ref_it = ref_reader_->iter(cons_alignment.alignment.RefID, cons_alignment.alignment.Position);
	string::const_iterator query_it = cons_alignment.alignment.QueryBases.begin();
	bool check_point = true;
	for (string::const_iterator pretty_it = cons_alignment.pretty_aln.begin(); pretty_it != cons_alignment.pretty_aln.end(); ++pretty_it){
		switch(*pretty_it){
			case '+':
				++nm;
				++query_it;
				break;
			case '-':
				++nm;
				++ref_it;
				break;
			case '|':
				nm += (*query_it != * ref_it);
				++query_it;
				++ref_it;
				break;
			default:
				check_point = false;
				break;
		}
	}
	cons_alignment.alignment.AddTag("NM", "i", nm); // NM tag = Number of mismatches, i.e., edit distance between query and reference.

	// Sanity check: Make sure Cigar, QureyBases, Positions, align_start/end are consistent.
	int qlen_from_pretty = 0;
	int alen_from_pretty = 0;
	for (vector<CigarOp>::const_iterator cigar_it = cons_alignment.alignment.CigarData.begin(); cigar_it != cons_alignment.alignment.CigarData.end(); ++cigar_it){
		if (cigar_it->Type == 'M' or cigar_it->Type == 'D'){
			alen_from_pretty += cigar_it->Length;
		}
		if (cigar_it->Type == 'M' or cigar_it->Type == 'I'){
			qlen_from_pretty += cigar_it->Length;
		}
	}
	check_point *= (qlen_from_pretty == (int) cons_alignment.alignment.QueryBases.size() and alen_from_pretty == (cons_alignment.align_end - cons_alignment.align_start + 1));
	if (not check_point){
		if (debug_){
			assert(check_point);
		}else{
			cerr << "Non-Fatal: Error while generating consensus alignment for the reads " << read_names << endl;
			return false;
		}
	}
	return true;
}

bool ConsensusAlignmentManager::CalculateConsensusQueryAndCigar_(const vector<Alignment*>& family_members, Alignment& cons_alignment) const {
	const unsigned int num_members = family_members.size();
	int read_counts_at_pos = 0;
	vector<unsigned int> query_idx_vec(num_members, 0);
	vector<unsigned int> pretty_idx_vec(num_members, 0);
	vector<unsigned int> right_end_del(num_members, 0);
	map<string, pair<unsigned int, unsigned int> > ins_dict;
	map<string, pair<unsigned int, unsigned int> > match_or_del_dict;
	string cons_query_bases;
	string cons_pretty_aln;
	int cons_align_start = -1; // No left soft-clipping
	int cons_align_end = -1;   // No right soft-clipping

	if (cons_alignment.align_end <= cons_alignment.align_start){ // align_end == align_start makes no sense at all!
		return false;
	}

	// Reserve memory because I will do a lot of push_pack later.
	cons_query_bases.reserve(cons_alignment.align_end - cons_alignment.align_start + 16);
	cons_pretty_aln.reserve(cons_alignment.align_end - cons_alignment.align_start + 16);

	for (unsigned int read_idx = 0; read_idx < num_members; ++read_idx){
		// The query index start from the first aligned base of the reads.
		query_idx_vec[read_idx] = family_members[read_idx]->left_sc;
		// Calculate the SC or DEL at the end of the
		unsigned int non_sc_end_idx = family_members[read_idx]->alignment.CigarData.size() - 1;
		if (family_members[read_idx]->right_sc > 0 and non_sc_end_idx > 0){
			--non_sc_end_idx;
		}
		right_end_del[read_idx] = (family_members[read_idx]->alignment.CigarData[non_sc_end_idx].Type == 'D'? family_members[read_idx]->alignment.CigarData[non_sc_end_idx].Length : 0);
	}

	// Calculate consensus pretty aln and consensus bases
	// Important: Must iterate over the positions spanned by the aligned positions of all reads.
	// Note 1: The aligned interval is [align_start, align_end], i.e., right-closed and left-closed interval.
	// Note 2: Start/End INSERTION bases are NOT counted as aligned position!

	for (int pos = cons_alignment.align_start; pos <= cons_alignment.align_end; ++pos){
		// (Step 1): Generate the ins_dict and match_or_del_dict for all reads at the position
		ins_dict.clear();
		match_or_del_dict.clear();
		read_counts_at_pos = 0;

		// Gather the alignment of reads at the position.
		for (unsigned int read_idx = 0; read_idx < num_members; ++read_idx){
			Alignment const * const rai = family_members[read_idx];
			// Is pos covered by the read?
			// Note: Do NOT use rai->align_start and rai->align_end because they are calculated before primer trimming.
			if (pos < rai->alignment.Position or pos > rai->alignment.GetEndPosition(false, true)){
				// Note that, if a read ends with INS, then the end INS will not be processed since end INS is not covered by the aligned interval.
				// This results that the INS at the end of the reads won't be in consensus alignment which does not really matter because TVC can't evaluate such INSs anyway.
				continue;
			}

			// Safety check for (Step 1a):
			bool check_point = pretty_idx_vec[read_idx] < rai->pretty_aln.size() and query_idx_vec[read_idx] < (rai->alignment.QueryBases.size() + right_end_del[read_idx] - (unsigned int) rai->right_sc);
			if (not check_point){
				if (debug_){
					assert(check_point);
				}else{
					cerr << "Non-Fatal: Error while generating consensus alignment for the read " << rai->alignment.Name << endl;
					return false;
				}
			}
			// End of safety check for (Step 1a)

			// (Step 1a): deal with INSERTION
			string ins_bases = ""; // If not an insertion, an empty string will be inserted to ins_dict.
			// Is the read has an INSERTION?
			if (rai->pretty_aln[pretty_idx_vec[read_idx]] == '+'){
				// Got an INSERTION! Process all consecutive INSERTION bases together.
				for(; pretty_idx_vec[read_idx] < rai->pretty_aln.size(); ++query_idx_vec[read_idx], ++pretty_idx_vec[read_idx]){
					if (rai->pretty_aln[pretty_idx_vec[read_idx]] != '+'){
						break;
					}
					ins_bases.push_back(toupper(rai->alignment.QueryBases[query_idx_vec[read_idx]]));
				}
			}
			// Insert the INSERTION to ins_dict
			InsertToDict_(ins_dict, ins_bases, rai->read_count, rai->read_number);

			// Safety check for (Step 1a):
			check_point = pretty_idx_vec[read_idx] < rai->pretty_aln.size() and query_idx_vec[read_idx] < (rai->alignment.QueryBases.size() + right_end_del[read_idx] - (unsigned int) rai->right_sc);
			if (not check_point){
				if (debug_){
					assert(check_point);
				}else{
					cerr << "Non-Fatal: Error while generating consensus alignment for the read " << rai->alignment.Name << endl;
					return false;
				}
			}
			check_point = rai->pretty_aln[pretty_idx_vec[read_idx]] == '-' or rai->pretty_aln[pretty_idx_vec[read_idx]] == '|';
			if (not check_point){
				if (debug_){
					assert(check_point);
				}else{
					cerr << "Non-Fatal: Error while generating consensus alignment for the read " << rai->alignment.Name << endl;
					return false;
				}
			}
			// End of safety check for (Step 1a)

			// (Step 1b): deal with the case of MATCH or DELETION.
			string my_base = (rai->pretty_aln[pretty_idx_vec[read_idx]] == '|')? string(1, toupper(rai->alignment.QueryBases[query_idx_vec[read_idx]])) : "";
			// Insert the 1bp MATCH/DELETION to match_or_del_dict.
			InsertToDict_(match_or_del_dict, my_base, rai->read_count, rai->read_number);

			// read_idx finished
			read_counts_at_pos += rai->read_count;
			if (my_base != ""){
				// Not a deletion => query indx moving forward.
				++query_idx_vec[read_idx];
			}
			++pretty_idx_vec[read_idx];
		}

		// Safety check for (Step 1):
		bool check_point = not (ins_dict.empty() or match_or_del_dict.empty());
		if (not check_point){
			if (debug_){
				assert(check_point);
			}else{
				cerr << "Non-Fatal: Error while generating consensus alignment for the reads ";
				for (unsigned int read_idx = 0; read_idx < num_members; ++read_idx){
					cerr << family_members[read_idx]->alignment.Name << ", ";
				}
				cerr << endl;
				return false;
			}
		}
		// End of safety check for (Step 1)

		// (Step 2): Handle the coverage related issue.
		// The aligned positions of the consensus read is the first interval whose positions that have coverage at least half of the total read counts.
		// Example: Read #1: [0, 100), Read #2: [0, 60), Read #3: [50, 100), Read #4: [70, 100) where the reads have equal read counts.
		// Then the consensus alignment covers [50, 60).
		// TODO: Strand specific seems more reasonable. E.g. [70, 100) if the reads are on the REV strand in the example (though highly fragmented coverage like this is rare).
		if (2 * read_counts_at_pos < cons_alignment.read_count){
			// The coverage is less than half
			if (cons_align_start < 0){
				// Not yet reach the start position of the consensus alignment yet.
				continue;
			}else{
				// Equivalently, hard clipping from the right.
				break;
			}
		}
		// The start position of the consensus alignment.
		if (cons_align_start < 0){
			cons_align_start = pos;
		}
		// The end position of the consensus alignment.
		cons_align_end = pos;

		// (Step 3): Applying majority rule
		string majority_key;
		char cons_pretty_aln_base = 0;

		// (Step 3a): Get majority among ins_dict
		GetMajority_(ins_dict, majority_key);
		if (not majority_key.empty()){
			// INSERTION
			cons_pretty_aln_base = '+';
			cons_query_bases += majority_key;
			cons_pretty_aln += string(majority_key.size(), cons_pretty_aln_base);
		}

		// (Step 3b): Get majority among match_or_del_dict
		GetMajority_(match_or_del_dict, majority_key);
		if (majority_key.empty()){
			// DELETION
			cons_pretty_aln_base = '-';
			cons_pretty_aln += string(1, cons_pretty_aln_base);
		}else{
			// MATCH
			cons_pretty_aln_base = '|';
			cons_pretty_aln += string(majority_key.size(), cons_pretty_aln_base);
			cons_query_bases += majority_key;
		}
	}

	// (Step 4): Save the consensus alignment.
	cons_alignment.alignment.QueryBases.swap(cons_query_bases);
	cons_alignment.alignment.Position = cons_align_start;
	cons_alignment.align_start = cons_align_start;
	cons_alignment.align_end = cons_align_end;
	cons_alignment.pretty_aln.swap(cons_pretty_aln);
	bool success = PrettyAlnToCigar_(cons_alignment.pretty_aln, cons_alignment.alignment.CigarData)
			and (cons_alignment.alignment.Position >= 0)
			and (cons_alignment.align_start <= cons_alignment.align_end)
			and (not cons_alignment.alignment.QueryBases.empty())
			and (not cons_alignment.pretty_aln.empty());

	return success;
}

char ConsensusAlignmentManager::PrettyCharToCigarType_(char pretty_aln) const{
	switch (pretty_aln){
		case '|':
			return 'M';
		case '+':
			return 'I';
		case '-':
			return 'D';
	}
	return 0;
}

// return true if success else false.
bool ConsensusAlignmentManager::PrettyAlnToCigar_(const string& pretty_aln, vector<CigarOp>& cigar_data) const{
	cigar_data.resize(0);
	if (pretty_aln.empty()){
		return true;
	}
	char my_cigar_type = PrettyCharToCigarType_(pretty_aln[0]);
	if (my_cigar_type == 0){
		return false;
	}
	cigar_data.push_back(CigarOp(my_cigar_type, 1));
	for (unsigned int pretty_idx = 1; pretty_idx < pretty_aln.size(); ++pretty_idx){
		if (pretty_aln[pretty_idx] == pretty_aln[pretty_idx - 1]){
			++(cigar_data.back().Length);
		}else{
			my_cigar_type = PrettyCharToCigarType_(pretty_aln[pretty_idx]);
			if (my_cigar_type == 0){
				cigar_data.resize(0);
				return false;
			}
			cigar_data.push_back(CigarOp(my_cigar_type, 1));
		}
	}
	return true;
}

void ConsensusAlignmentManager::PrintFailureMessage_(const vector<Alignment*>& family_members, const string& my_failure_reason) const{
	if (not debug_){
		if (not my_failure_reason.empty()){
			cerr << "Warning (non-fatal): Fail to generate consensus alignment for the reads ";
			for (vector<Alignment*>::const_iterator read_it = family_members.begin(); read_it != family_members.end(); ++read_it){
				cerr << (*read_it)->alignment.Name << ", ";
			}
			cerr << endl;
		}
		return;
	}
	int total_read_counts = 0;
	for (vector<Alignment*>::const_iterator read_it = family_members.begin(); read_it != family_members.end(); ++read_it){
		total_read_counts += (*read_it)->read_count;
	}
	cout << "+ Calculating consensus alignments for " << family_members.size() << " reads:" << endl
		 << "  - Status: ";
	if (my_failure_reason.empty()){
		cout << "SUCCESS." << endl;
	}else{
		cout << "FAILURE (" << my_failure_reason << ")" << endl;
	}
	cout << "  - Total read counts = " << total_read_counts << endl
	     << "  + Reads for generating the consensus alignment:" << endl;
	for (vector<Alignment*>::const_iterator read_it = family_members.begin(); read_it != family_members.end(); ++read_it){
		cout << "    + " << (*read_it)->alignment.Name
			 << " (x" << (*read_it)->read_count
			 << ((*read_it)->is_reverse_strand? " REV reads)" : " FWD reads)" ) << endl;
		cout << "      - Aligned to " << ref_reader_->chr_str((*read_it)->alignment.RefID)
			 << ":[" << (*read_it)->alignment.Position << ", " << (*read_it)->alignment.GetEndPosition(false, true) + 1 << ")." << endl
			 << "      - query bases = "<< (*read_it)->alignment.QueryBases << endl
			 << "      - cigar = (";
		for (vector<CigarOp>::const_iterator cigar_it = (*read_it)->alignment.CigarData.begin(); cigar_it != (*read_it)->alignment.CigarData.end(); ++cigar_it){
			cout << "(" << cigar_it->Length << ", " << cigar_it->Type << "), ";
		}
		cout << ")" << endl;
	}
}


void ConsensusAlignmentManager::PrintVerbose_(const vector<Alignment*>& family_members, const Alignment& cons_alignment) const {
	if (not debug_){
		return;
	}
	PrintFailureMessage_(family_members, "");
	cout << "  + Consensus read:" << endl
		 << "    - query bases = " << cons_alignment.alignment.QueryBases << endl
		 << "    - length = " <<  cons_alignment.alignment.QueryBases.size() << endl
		 << "    - aligned to " << ref_reader_->chr_str(cons_alignment.alignment.RefID)
		 << ":[" << cons_alignment.align_start << ", " << cons_alignment.align_end + 1 << ") in 0-based coordinate." << endl
		 << "    - cigar = (";
	for (vector<CigarOp>::const_iterator cigar_it = cons_alignment.alignment.CigarData.begin(); cigar_it != cons_alignment.alignment.CigarData.end(); ++cigar_it){
		cout << "(" << cigar_it->Length << ", " << cigar_it->Type << "), ";
	}
	cout << ")" << endl;
}

// I want to copy the information needed for candidate gen. No need to copy those flowspace related stuff.
void ConsensusAlignmentManager::PartiallyCopyAlignmentFromAnother_(Alignment& alignment, const Alignment& template_alignment, int read_count) const{
	// Create a new Alignment object.
	alignment = Alignment();
	alignment.read_count = read_count;
	alignment.alignment = template_alignment.alignment;
	// Copy all the member variables determined in AlleleParser::UnpackReadAllele
	alignment.original_position = template_alignment.original_position;
	alignment.original_end_position = template_alignment.original_end_position;
	alignment.start = template_alignment.start;
	alignment.end = template_alignment.end;
	alignment.snp_count = template_alignment.snp_count;
	alignment.is_read_allele_unpacked = template_alignment.is_read_allele_unpacked;
	alignment.refmap_start = template_alignment.refmap_start;
	alignment.refmap_code = template_alignment.refmap_code;
	alignment.refmap_has_allele = template_alignment.refmap_has_allele;
	alignment.refmap_allele = template_alignment.refmap_allele;
	// By TrimAmpliseqPrimer
	alignment.target_coverage_indices = template_alignment.target_coverage_indices;
	alignment.best_coverage_target_idx = template_alignment.best_coverage_target_idx;
	// Copy some useful (to FreeBayes) member variables determined in UnpackOnLoad
	// measurements, measurements_sd, phase_params, pretty_aln, etc are not needed.
	alignment.is_reverse_strand = template_alignment.is_reverse_strand;
	alignment.left_sc = template_alignment.left_sc;
	alignment.right_sc = template_alignment.right_sc;
	alignment.start_sc = template_alignment.start_sc;
	alignment.align_start = template_alignment.align_start;
	alignment.align_end = template_alignment.align_end;
	alignment.runid = template_alignment.runid;
	alignment.read_group = template_alignment.read_group;
	alignment.sample_index = template_alignment.sample_index;
	alignment.primary_sample = template_alignment.primary_sample;
}


void MolecularTagManager::Initialize(MolecularTagTrimmer* const input_tag_trimmer, const SampleManager* const sample_manager)
{
	tag_trimmer = input_tag_trimmer;
	if (not tag_trimmer->HaveTags()){
		multisample_prefix_tag_struct_.clear();
		multisample_suffix_tag_struct_.clear();
		return;
	}

	// Initialize
	multisample_prefix_tag_struct_.assign(sample_manager->num_samples_, "");
    multisample_prefix_tag_random_bases_idx_.resize(sample_manager->num_samples_);
    multisample_prefix_tag_stutter_bases_idx_.resize(sample_manager->num_samples_);
	multisample_suffix_tag_struct_.assign(sample_manager->num_samples_, "");
    multisample_suffix_tag_random_bases_idx_.resize(sample_manager->num_samples_);
    multisample_suffix_tag_stutter_bases_idx_.resize(sample_manager->num_samples_);
    cout << "MolecularTagManager: Found "<< tag_trimmer->NumTaggedReadGroups() << " read group(s) with molecular tags." << endl;

    // Print warning message if not all RG have tags.
    if (tag_trimmer->NumReadGroups() > tag_trimmer->NumTaggedReadGroups()){
        cerr << "Warning: MolecularTagManager: Molecular tags not found in read group(s) {";
        for (map<string, int>::const_iterator rg_it = sample_manager->read_group_to_sample_idx_.begin(); rg_it != sample_manager->read_group_to_sample_idx_.end(); ++rg_it){
        	const string& read_group = rg_it->first;
        	if (not tag_trimmer->HasTags(read_group)){
      		    cerr << read_group << ", ";
      	    }
        }
        cerr << "}. The reads in these group(s) may be filtered out." << endl;
    }

    // Check the uniqueness of tag structures cross the read groups in each sample.
    for (map<string, int>::const_iterator rg_it = sample_manager->read_group_to_sample_idx_.begin(); rg_it != sample_manager->read_group_to_sample_idx_.end(); ++rg_it){
    	const int& sample_idx = rg_it->second;
    	const string& read_group = rg_it->first;
    	if (not tag_trimmer->HasTags(read_group)){
    		// I implicitly allow a sample has a read group has tag but another read group doesn't.
    		continue;
    	}
    	if (multisample_prefix_tag_struct_[sample_idx].empty() and multisample_suffix_tag_struct_[sample_idx].empty()){
    		multisample_prefix_tag_struct_[sample_idx] = tag_trimmer->GetPrefixTag(read_group);
    		multisample_suffix_tag_struct_[sample_idx] = tag_trimmer->GetSuffixTag(read_group);
    	}
    	else{
    		if (multisample_prefix_tag_struct_[sample_idx] != tag_trimmer->GetPrefixTag(read_group)
    				or multisample_suffix_tag_struct_[sample_idx] != tag_trimmer->GetSuffixTag(read_group)){
        	    cerr << "ERROR: MolecularTagManager: Variable tag structures found in the sample " << sample_manager->sample_names_[sample_idx] << " !" << endl;
          	    exit(-1);
          	    return;
    		}
    	}
    }

    // Get the indicies of random/stutter bases.
    for (int sample_idx = 0; sample_idx < sample_manager->num_samples_; ++sample_idx){
    	for (int xfix = 0; xfix < 2; ++xfix){
    		string* my_tag_struct = xfix == 0? &(multisample_prefix_tag_struct_[sample_idx]) : &(multisample_suffix_tag_struct_[sample_idx]);
    		vector<int>* my_random_idx = xfix == 0? &(multisample_prefix_tag_random_bases_idx_[sample_idx]) : &(multisample_suffix_tag_random_bases_idx_[sample_idx]);
    		vector<int>* my_stutter_idx = xfix == 0? &(multisample_prefix_tag_stutter_bases_idx_[sample_idx]) : &(multisample_suffix_tag_stutter_bases_idx_[sample_idx]);
    		my_random_idx->resize(0);
    		my_random_idx->reserve(8);
    		my_stutter_idx->resize(0);
    		my_stutter_idx->reserve(8);
        	for (int nuc_idx = 0; nuc_idx < (int) my_tag_struct->size(); ++nuc_idx){
        		if (my_tag_struct->at(nuc_idx) == 'N' or my_tag_struct->at(nuc_idx) == 'n'){
        			my_random_idx->push_back(nuc_idx);
        		}else{
        			my_stutter_idx->push_back(nuc_idx);
        		}
        	}
    	}

    	cout <<"MolecularTagManager: Found the unique molecular tag structures in the sample "<< sample_manager->sample_names_[sample_idx] << ": ";
    	cout <<"prefix tag = "<< (multisample_prefix_tag_struct_[sample_idx].empty()? "NULL" : multisample_prefix_tag_struct_[sample_idx]) << ", ";
    	cout <<"suffix tag = "<< (multisample_suffix_tag_struct_[sample_idx].empty()? "NULL" : multisample_suffix_tag_struct_[sample_idx]) << endl << endl;
    }
}

bool IsStrictness(const string& mol_tag, const string& tag_struct)
{
	if ((mol_tag.size() != tag_struct.size())){
		return false;
	}

	string::const_iterator tag_it = mol_tag.begin();
	for (string::const_iterator struct_it = tag_struct.begin(); struct_it != tag_struct.end(); ++struct_it, ++tag_it){
		if ((*struct_it != 'N') and (*tag_it != *struct_it)){
			return false;
		}
	}
	return true;
}

bool IsStrictness(const string& mol_tag, const string& tag_struct, const vector<int>& stutter_idx)
{
	if ((mol_tag.size() != tag_struct.size())){
		return false;
	}
	for (vector<int>::const_iterator idx_it = stutter_idx.begin(); idx_it != stutter_idx.end(); ++idx_it){
		if (tag_struct.at(*idx_it) != mol_tag.at(*idx_it)){
			return false;
		}
	}
	return true;
}

// return true if the tags match the tag structures.
bool MolecularTagManager::IsStrictTag(const string& prefix_tag, const string& suffix_tag, int sample_idx) const
{
	// Check suffix first because it has higher chance to be non-strict.
	if (not IsStrictSuffixTag(suffix_tag, sample_idx)){
		return false;
	}
	if (not IsStrictPrefixTag(prefix_tag, sample_idx)){
		return false;
	}

	return true;
}

// sample_idx = -1 indicates no multisample
bool MolecularTagManager::IsStrictSuffixTag(const string& suffix_tag, int sample_idx) const
{
	sample_idx = max(0, sample_idx); // sample_idx = -1 indicates no multisample
	return IsStrictness(suffix_tag, multisample_suffix_tag_struct_[sample_idx], multisample_suffix_tag_stutter_bases_idx_[sample_idx]);
}

// sample_idx = -1 indicates no multisample
bool MolecularTagManager::IsStrictPrefixTag(const string& prefix_tag, int sample_idx) const
{
	sample_idx = max(0, sample_idx); // sample_idx = -1 indicates no multisample
	return IsStrictness(prefix_tag, multisample_prefix_tag_struct_[sample_idx], multisample_prefix_tag_stutter_bases_idx_[sample_idx]);
}

MolecularTagManager::MolecularTagManager()
{
	multisample_prefix_tag_struct_.clear();
	multisample_suffix_tag_struct_.clear();
	tag_trimmer = NULL;
}

// I stole the terms flat and run from Earl's indel assembly algorithm.  a) flat = nuc of homopolymer, b) run = length of homopolymer
// Example: base_seq = "TAACGGG" gives flat_and_run = {('T', 1), ('A', 2), ('C', 1), ('G': 3)}
void BaseSeqToFlatAndRun(const string& base_seq, vector<pair<char, int> >& flat_and_run)
{
    flat_and_run.clear();
    if (base_seq.empty()){
        return;
    }
    flat_and_run.reserve(base_seq.size());
    flat_and_run.push_back(pair<char, int>(base_seq[0], 1));
    for (int base_idx = 1; base_idx < (int) base_seq.size(); ++base_idx){
        if (base_seq[base_idx] == base_seq[base_idx - 1]){
            ++(flat_and_run.back().second);
        }else{
            flat_and_run.push_back(pair<char, int>(base_seq[base_idx], 1));
        }
    }
}

// Convert the vector of flat&run pair to the base sequence.
string FlatAndRunToBaseSeq(const vector<pair<char, int> >::const_iterator& it_begin, const vector<pair<char, int> >::const_iterator& it_end)
{
    string base_seq("");
    for (vector<pair<char, int> >::const_iterator it = it_begin; it != it_end; ++it){
        base_seq += string(it->second, it->first);
    }
    return base_seq;
}

// Claim the two vectors of flat&run pairs synchronized if the following two criteria are both satisfied
// a) they have the same flat (shrank to the smaller length)
// b) the discrepancy between the runs <= allowed_hp_indel_len.
bool IsSyncFlatAndRun(const vector<pair<char, int> >::const_iterator it_1_begin,
                         const vector<pair<char, int> >::const_iterator it_1_end,
                         const vector<pair<char, int> >::const_iterator it_2_begin,
                         const vector<pair<char, int> >::const_iterator it_2_end,
                         int allowed_hp_indel_len)
{
    vector<pair<char, int> >::const_iterator it_1 = it_1_begin;
    vector<pair<char, int> >::const_iterator it_2 = it_2_begin;

    while (it_1 != it_1_end and it_2 != it_2_end){
        if (it_1->first != it_2->first
            or abs(it_1->second - it_2->second) > allowed_hp_indel_len){
            return false;
        }
        ++it_1;
        ++it_2;
    }
    return true;
}

bool MolecularTagManager::IsFlowSynchronizedTags(string tag_1, string tag_2, bool is_prefix) const
{
    if (tag_1.size() != tag_2.size()){
        return false;
    }

    if (not is_prefix){
    	// Be consistent with the tag trimming direction in BaseCaller.
        reverse(tag_1.begin(), tag_1.end());
        reverse(tag_2.begin(), tag_2.end());
    }
    int allowed_hp_indel_len = 1;
    vector<pair<char, int> > flat_and_run_1;
    vector<pair<char, int> > flat_and_run_2;
    BaseSeqToFlatAndRun(tag_1, flat_and_run_1);
    BaseSeqToFlatAndRun(tag_2, flat_and_run_2);
    return IsSyncFlatAndRun(flat_and_run_1.begin(), flat_and_run_1.end(), flat_and_run_2.begin(), flat_and_run_2.end(), allowed_hp_indel_len);
}

// Determine whether 2 tags are "partial similar" in the sense that the molecular tag may suffer from PCR error and sequencing error.
// By adjusting the length of the homopolymers appropriately, if the delta between the two tags is nothing or just "one" SNP, then I claim they are similar.
bool MolecularTagManager::IsPartialSimilarTags(string tag_1, string tag_2, bool is_prefix) const
{
    if (tag_1.size() != tag_2.size()){
        return false;
    }

    if (not is_prefix){
    	// Be consistent with the tag trimming direction in BaseCaller.
        reverse(tag_1.begin(), tag_1.end());
        reverse(tag_2.begin(), tag_2.end());
    }

    // I allow "one" SNP and (allowed_hp_indel_len)-mer HP-INDELs.
    int allowed_hp_indel_len = 1;
    const static vector<string> kAllNucs = {"T", "A", "C", "G"};
    vector<pair<char, int> > flat_and_run_1;
    vector<pair<char, int> > flat_and_run_2;
    BaseSeqToFlatAndRun(tag_1, flat_and_run_1);
    BaseSeqToFlatAndRun(tag_2, flat_and_run_2);
    vector<pair<char, int> >::iterator it_1 = flat_and_run_1.begin();
    vector<pair<char, int> >::iterator it_2 = flat_and_run_2.begin();
    bool is_not_begin = false;

    while (it_1 != flat_and_run_1.end() and it_2 != flat_and_run_2.end()){
        // Is "flat" the same?
    	if (it_1->first == it_2->first){
            if (is_not_begin){
            	// For similar tags, every large HP-INDEL must be explained by a SNP. Otherwise, I claim not similar.
                if (abs((it_1 - 1)->second - (it_2 - 1)->second) > allowed_hp_indel_len){
                    return false;
                }
            }
            ++it_1;
            ++it_2;
        }else{
        	// Now the "flats" of the two tag are different.
            // Since I only allow one SNP, the brute-force algorithm can be very efficient.
            // Let's try all 6 possible combinations for altering the first base of the homopolymer at the flats. See if I can obtain two synchronized vectors of flat&run pairs.

        	// Set the anchors
        	string tag_1_anchor = "";
            string tag_2_anchor = "";
            vector<pair<char, int> >::iterator anchor_it_1 = it_1;
            vector<pair<char, int> >::iterator anchor_it_2 = it_2;
            if (is_not_begin){
                --anchor_it_1;
                --anchor_it_2;
                tag_1_anchor = string(anchor_it_1->second, anchor_it_1->first);
                tag_2_anchor = string(anchor_it_2->second, anchor_it_2->first);
            }

            // Set the paddings
            --(it_1->second);
            --(it_2->second);
            string tag_1_padding = FlatAndRunToBaseSeq(it_1, flat_and_run_1.end());
            string tag_2_padding = FlatAndRunToBaseSeq(it_2, flat_and_run_2.end());
            ++(it_1->second);
            ++(it_2->second);

            for (int try_nuc_idx = 0; try_nuc_idx < 4; ++try_nuc_idx){
                if (kAllNucs[try_nuc_idx][0] != it_1->first){
                	// Let's alter one base (where the base is the first base of the flat&run pair *it_1) of tag_1 to kAllNucs[try_nuc_idx][0]
                    string try_base_seq_1 = tag_1_anchor + kAllNucs[try_nuc_idx] + tag_1_padding;
                    vector<pair<char, int> > try_flat_and_run_1;
                    BaseSeqToFlatAndRun(try_base_seq_1, try_flat_and_run_1);
                    if (IsSyncFlatAndRun(try_flat_and_run_1.begin(), try_flat_and_run_1.end(), anchor_it_2 ,flat_and_run_2.end(), allowed_hp_indel_len)){
                        return true;
                    }
                }
                if (kAllNucs[try_nuc_idx][0] != it_2->first){
                	// Let's alter one base (where the base is the first base of the flat&run pair *it_2) of tag_2 to kAllNucs[try_nuc_idx][0]
                    string try_base_seq_2 = tag_2_anchor + kAllNucs[try_nuc_idx] + tag_2_padding;
                    vector<pair<char, int> > try_flat_and_run_2;
                    BaseSeqToFlatAndRun(try_base_seq_2, try_flat_and_run_2);
                    if (IsSyncFlatAndRun(try_flat_and_run_2.begin(), try_flat_and_run_2.end(), anchor_it_1, flat_and_run_1.end(), allowed_hp_indel_len)){
                        return true;
                    }
                }
            }
            // One SNP + HP-INDELs can't explain the delta between the two tags. Thus I claim not partial similar.
            return false;
        }
        is_not_begin = true;
    }
    return true;
}



char NucTo0123(char nuc)
{
    switch (nuc){
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        case 'a':
            return 0;
        case 'c':
            return 1;
        case 'g':
            return 2;
        case 't':
            return 3;
    }
    return -1;
}

char NumToNuc(char nuc_in_0123){
    switch (nuc_in_0123){
        case 0:
            return 'A';
        case 1:
            return 'C';
        case 2:
            return 'G';
        case 3:
            return 'T';
    }
    return -1;
}

//TODO: CZB: Implement it using strategy design pattern
bool MolecularFamilyGenerator::FindFamilyForOneRead_(Alignment* rai, vector< vector<MolecularFamily> >& my_molecular_families)
{
	int strand_key_idx = (rai->tag_info.is_bi_directional_tag)? 0 : (rai->is_reverse_strand? 2 : 1);
	bool is_new_tag = true;
	unsigned int tag_index_in_my_molecular_families = 0;
	bool success = true;

	// Hashing mol_tag to a long long integer facilitates the mapping between the mol_tag to the index of my_molecular_families.
	if (long_long_hashable_){
		pair< map<unsigned long long, unsigned int>::iterator, bool> tag_finder;
		// map mol_tag_long_long to the index of my_family_[strand_key] for mol_tag
		tag_finder = long_long_tag_lookup_table_[strand_key_idx].insert(pair<unsigned long long, unsigned int>(rai->tag_info.tag_hash, my_molecular_families[strand_key_idx].size()));
		// Note that map::insert will not insert value into the key of the map if key is pre-existed.
		// tag_finder.first->first = mol_tag_long_long
		// tag_finder.first->second = the index of my_molecular_families[strand_key] for tag
		// tag_finder.second indicates inserted or not.
		// tag_finder.second = false if I previously got the family, and hence long_long_tag_lookup_table_ is not updated.
		// tag_finder.second = true if this is the first time we get mol_tag, and hence mol_tag_long_long is inserted into long_long_tag_lookup_table_[strand_key]
		is_new_tag = tag_finder.second;
		tag_index_in_my_molecular_families = tag_finder.first->second;
		// Detect the collision of the hash if it is not strict
		if ((not rai->tag_info.is_strict_tag) and (not is_new_tag)){
			if ( rai->tag_info.readable_fam_info != my_molecular_families[strand_key_idx][tag_index_in_my_molecular_families].all_family_members[0]->tag_info.readable_fam_info){
				success = false;
				return success;
			}
		}
	}
	// Map a string to the index of my_molecular_families, slow but always safe.
	else{
		pair< map<string, unsigned int>::iterator, bool> tag_finder;
		tag_finder = string_tag_lookup_table_[strand_key_idx].insert(pair<string, unsigned int>(rai->tag_info.readable_fam_info, my_molecular_families[strand_key_idx].size()));
		is_new_tag = tag_finder.second;
		tag_index_in_my_molecular_families = tag_finder.first->second;
	}

	if (is_new_tag){
		// The first char of rai->tag_info.readable_fam_info is for strand direction.
		string my_tag = rai->tag_info.readable_fam_info.substr(1, rai->tag_info.prefix_mol_tag.size() + rai->tag_info.suffix_mol_tag.size());
		// Generate a new family since this is the first time I get the mol_tag
		my_molecular_families[strand_key_idx].push_back(MolecularFamily(my_tag, strand_key_idx - 1));
	}
	// Add the read to the family
	my_molecular_families[strand_key_idx][tag_index_in_my_molecular_families].AddNewMember(rai);
	return success;
}

// Generate molecular families
// Inputs: bam_position, sample_index,
// Output: my_molecular_families
// Set sample_index = -1 if not multi-sample.
void MolecularFamilyGenerator::GenerateMyMolecularFamilies(const MolecularTagManager* const mol_tag_manager,
		PositionInProgress& bam_position,
		int sample_index,
		vector< vector<MolecularFamily> >& my_molecular_families)
{
	// No tags, no families.
	if (not mol_tag_manager->tag_trimmer->HaveTags()){
		return;
	}
	bool reset = false; // In case I detect any collision when I use long long hash. If happened, reset and use string as the key.
	bool is_consensus_bam = false;
	unsigned int prefix_tag_len = (mol_tag_manager->GetPrefixTagStruct(sample_index)).size();
	unsigned int suffix_tag_len = (mol_tag_manager->GetSuffixTagStruct(sample_index)).size();
    long_long_hashable_ = mol_tag_manager->GetPrefixTagRandomBasesNum(sample_index) <= 12 and mol_tag_manager->GetSuffixTagRandomBasesNum(sample_index) <= 12;

    // my_molecular_families[0] for BI-DIR Families (Bi-dir Ampliseq UMT)
    // my_molecular_families[1] for FWD Families (Tagseq or Uni-dir Ampliseq UMT)
    // my_molecular_families[2] for REV Families (Tagseq or Uni-dir Ampliseq UMT)
	my_molecular_families.resize(3);
	long_long_tag_lookup_table_.resize(my_molecular_families.size());
	string_tag_lookup_table_.resize(my_molecular_families.size());

	// Initialize the family and table container
	for (int i_strand = 0; i_strand < (int) my_molecular_families.size(); ++i_strand){
		my_molecular_families[i_strand].resize(0);
		my_molecular_families[i_strand].reserve(32768); // Reverse for 2^15 families (including non-functional ones) per strand should be enough most of the time.
		long_long_tag_lookup_table_[i_strand].clear();
		string_tag_lookup_table_[i_strand].clear();
	}

	// max_first_target_idx and min_first_target_idx are used to determine uniquely hashable.
	int max_first_target_idx = -1;
	int min_first_target_idx = -1;
	// Iterate over reads
	Alignment* rai = bam_position.begin;
	while (rai != bam_position.end){
		if (rai == NULL) {
			bam_position.end = NULL;
			return;
		}
		// Skip the read if filtered or has no tag
		if (rai->filtered or (not rai->tag_info.HasTags())) {
			rai = rai->next;
			continue;
		}

		// skip the read if it is not for this sample
		if (sample_index >= 0 and rai->sample_index != sample_index) {
			rai = rai->next;
			continue;
		}
		// Tag length check
		if ((rai->tag_info.prefix_mol_tag.size() > 0 and rai->tag_info.prefix_mol_tag.size() != prefix_tag_len)
				or (rai->tag_info.suffix_mol_tag.size() > 0 and rai->tag_info.suffix_mol_tag.size() != suffix_tag_len)){
			cerr << "MolecularFamilyGenerator: Warning: The length of the molecular tag of the read "<< rai->alignment.Name << " doesn't match the tag structure provided in the bam header." << endl;
			rai = rai->next;
			continue;
		}

		// Are the reads in the pileup hased by unsigned long long with no collision?
		if (long_long_hashable_){
			if (not rai->target_coverage_indices.empty()){
				max_first_target_idx = max(max_first_target_idx, rai->target_coverage_indices[0]);
				min_first_target_idx = (min_first_target_idx) < 0 ? rai->target_coverage_indices[0] : min(rai->target_coverage_indices[0], min_first_target_idx);
			}
			// Does my hash method have collision?
			long_long_hashable_ = rai->tag_info.is_uniquely_hashable and (max_first_target_idx - min_first_target_idx < 127);
			// If it turns out not hashable, start over and I will use string as keys.
			reset = not long_long_hashable_;
		}
		// If any read has read_count > 1 then I am dealing with a consensus bam
		is_consensus_bam += (rai->read_count > 1);

		// Find my family
		reset = not FindFamilyForOneRead_(rai, my_molecular_families);
		// Clear and start over if the long long hash has collision.
		if (reset){
			reset = false;
			long_long_hashable_ = false;
			// Clear the family and table containers
			for (int i_strand = 0; i_strand < (int) my_molecular_families.size(); ++i_strand){
				my_molecular_families[i_strand].resize(0);
				long_long_tag_lookup_table_[i_strand].clear();
				string_tag_lookup_table_[i_strand].clear();
			}
			// Start from the first read
			rai = bam_position.begin;
			continue;
		}
		rai = rai->next;
	}

	// Finally, sort the members in each family by the read counts if it is a consensus bam
	for (vector<vector<MolecularFamily> >::iterator strand_it = my_molecular_families.begin(); strand_it != my_molecular_families.end(); ++strand_it){
		for (vector<MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it != strand_it->end(); ++fam_it){
			if (is_consensus_bam){
				fam_it->SortAllFamilyMembers();
			}
			else{
				fam_it->is_all_family_members_sorted = true; // Sort no needed means sorted.
			}
		}
	}
}

// Reversed operation of MolecularTagManager::LongLongToPrintableStr
bool MolecularTagManager::PrintableStrToLongLong(const string& s, unsigned long long& v) {
    const static unsigned long long multiplier = 62; // number of alphanumeric char

    v = 0;
    if (s.size() != 11){
        return false;
    }

    for (int idx = 10; idx >= 0; --idx){
    	char offset = 0;
    	if (s[idx] >= 'a' and s[idx] <= 'z'){
    		offset = s[idx] - 61; // 61 = 'a' - 36
    	}else if (s[idx] >= 'A' and s[idx] <= 'Z'){
    		offset = s[idx] - 55; // 55 = 'A' - 10
    	}else if (s[idx] >= '0' and s[idx] <= '9'){
    		offset = s[idx] - '0';
    	}else{
    		return false;
    	}

        v *= multiplier;
        v += (unsigned long long) offset;
    }
    return true;
}

// Use a string of 11 (i.e., 64*log(2)/log(62) = 10.74872...) alphanumeric characters (0-9, A-Z, a-z) to represent a 64 bits unsigned integer.
void MolecularTagManager::LongLongToPrintableStr(unsigned long long v, string& s){
    const unsigned long long divisor = 62; // number of allowed printable char (no space)
    const static string alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    s.assign(11, '0'); // 64 bits data can be uniquely hashable by 11 alphanumeric characters

    for (int idx = 0; v != 0; ++idx){
        // lldiv might not handle "unsigned" long long properly, so I calculate the quotient and reminder in two operations.
        unsigned long long quotient = v / divisor;
        char reminder = (char) (v - quotient * divisor); // Not ideal, but I can afford an additional multiplication
        s[idx] = alphanumeric.at(reminder);
        v = quotient;
    }
}

void MolecularTagManager::FamilyInfoToReadableStr(int strand_key, const string& zt, const string& yt, const vector<int>& target_indices, string& my_readable_str){
	my_readable_str.resize(0);
	my_readable_str.reserve(zt.size() + yt.size() + 8);
	switch(strand_key){
		case -1:
			my_readable_str += "B";
			break;
		case 0:
			my_readable_str += "F";
			break;
		case 1:
			my_readable_str += "R";
			break;
		default:
			return;
	}
	my_readable_str += zt;
	my_readable_str += yt;

	// Add target coverage information to the key
	for (vector<int>::const_iterator target_idx_it = target_indices.begin(); target_idx_it != target_indices.end(); ++target_idx_it){
		my_readable_str += (to_string(*target_idx_it) + ",");
	}
	if (my_readable_str[my_readable_str.size() - 1] == ','){
		my_readable_str.resize(my_readable_str.size() - 1);
	}
}


// De-hash the long long integer to the family, i.e., the reverse operation of FamilyToLongLong.
string MolecularTagManager::LongLongToFamilyInfo(unsigned long long my_hash, int sample_idx, int& strand_key, string& zt, string& yt, vector<int>& target_indices) const {
	string fail_reason = "";

	// Handle the case of failure first.
	if (my_hash == 0){
		fail_reason = "Hash value = 0 is preserved for error.";
		return fail_reason;
	}
	if (multisample_prefix_tag_random_bases_idx_[sample_idx].size() > 12){
		fail_reason = "Number of prefix random bases > 12.";
		return fail_reason;
	}
	if (multisample_suffix_tag_random_bases_idx_[sample_idx].size() > 12){
		fail_reason = "Number of suffix random bases > 12.";
		return fail_reason;
	}
	// The 26-th bit from the left is the strictness of YT
	if (((my_hash >> 25) & 1) != 1){
		fail_reason = "Unable to dehash a non-strict YT.";
		return fail_reason;
	}
	// The 52-th bit from the left is the strictness of ZT
	if (((my_hash >> 51) & 1) != 1){
		fail_reason = "Unable to dehash a non-strict ZT.";
		return fail_reason;
	}

    // Decode zt, yt
	zt = multisample_prefix_tag_struct_[sample_idx];
    yt = multisample_suffix_tag_struct_[sample_idx];
    char* long_long_tail_byte = NULL;

    // I only need to fill the random part of zt and yt because I can't de-hash the non-strict tags.
    for (int tag_key = 1; tag_key >=0 ; --tag_key){
        unsigned long long my_umt_hash = my_hash & 33554431; // 33554431 = 2^25 - 1
        long_long_tail_byte = (char*) &my_umt_hash;
        string * my_tag = NULL;
        vector<int> const * my_idx_vec = NULL;
        // tag_key = 0 for ZT
        if (tag_key == 0){
        	my_tag = &zt;
        	my_idx_vec = &(multisample_prefix_tag_random_bases_idx_[sample_idx]);
        }
        // tag_key = 1 for YT
        else{
        	my_tag = &yt;
        	my_idx_vec = &(multisample_suffix_tag_random_bases_idx_[sample_idx]);
        }

        for (int vec_idx = (int) my_idx_vec->size() - 1; vec_idx >= 0; --vec_idx){
            char nuc_in_0123 = (*long_long_tail_byte) & 3; // Bit mask: get the last 2 tail bits
            my_tag->at(my_idx_vec->at(vec_idx)) = NumToNuc(nuc_in_0123);
            // I use 2 bits to represent 1 nuc.
            my_umt_hash >>= 2;
        }
        my_hash >>= 26; // 25 bits of tag + 1 bit of tag strictness.
    }

    // Decode Target Coverage
    long_long_tail_byte = (char*) &my_hash;
    char bit_mapped_additional_covered_idx = (*long_long_tail_byte) & 7; // Get the 3 tail bits
    my_hash >>= 3;
    char first_target_idx = ((*long_long_tail_byte) & 127) - 1; // Get the 7 tail bits
    target_indices.resize(0);
    target_indices.reserve(4);
    if (first_target_idx < 0 and bit_mapped_additional_covered_idx > 0){
    	fail_reason = "Fail to decode the target indices.";
    	return fail_reason;
    }
    if (first_target_idx >= 0){
        target_indices.push_back((int) first_target_idx);
        if (bit_mapped_additional_covered_idx > 0){
            for (int i_offset = 1; i_offset < 4; ++i_offset){
                if ((bit_mapped_additional_covered_idx & 1)){
                    target_indices.push_back(i_offset + (int) first_target_idx);
                }
                bit_mapped_additional_covered_idx >>= 1;
            }
        }
    }else if (bit_mapped_additional_covered_idx > 0){
        // I can't have no first target but have an additional target index.
    	fail_reason = "Fail to decode the target indices.";
    	return fail_reason;
    }
    // Shift 7 bits to the right.
    my_hash >>= 7;

    // Decode strand key
    switch (*long_long_tail_byte){
        case 1:
            strand_key = -1;
            break;
        case 2:
            strand_key = 0;
            break;
        case 3:
            strand_key = 1;
            break;
        default:{
        	fail_reason = "Unknown strand key.";
        	return fail_reason;
        }
    }
    return fail_reason;
}


void MolecularTagManager::PreComputeForFamilyIdentification(Alignment* rai){
	if (not rai->tag_info.HasTags()){
		return;
	}
	int strand_key = (rai->alignment.IsReverseStrand())? 1 : 0;
	string zt = rai->tag_info.prefix_mol_tag;
	string yt = rai->tag_info.suffix_mol_tag;

	// Handling BI-DIR UMT
	if (rai->tag_info.is_bi_directional_tag){
		if (strand_key == 1){
			// In bi-dir UMT, the mol tag of a REV read needs to be RevComp.
			zt.swap(yt);
			RevComplementInPlace(zt);
			RevComplementInPlace(yt);
		}
		strand_key = -1; // -1 means bi-dir family
	}
	// Also check and filter on tag strictness
	rai->tag_info.is_strict_prefix_tag = IsStrictPrefixTag(zt, rai->sample_index);
	rai->tag_info.is_strict_suffix_tag = IsStrictSuffixTag(yt, rai->sample_index);
	rai->tag_info.is_strict_tag = rai->tag_info.is_strict_prefix_tag and rai->tag_info.is_strict_suffix_tag;
	// Calculate the hash for family identification
	rai->tag_info.is_uniquely_hashable = FamilyInfoToLongLong(strand_key, zt, rai->tag_info.is_strict_prefix_tag, yt, rai->tag_info.is_strict_suffix_tag, rai->target_coverage_indices, rai->sample_index, rai->tag_info.tag_hash);
	// Also add the readable family information for family identification in case not uniquely hashable
	FamilyInfoToReadableStr(strand_key, zt, yt, rai->target_coverage_indices, rai->tag_info.readable_fam_info);
	// Note mol_tag_manager->tag_trimmer->tag_trim_method_ = 0, 1 indicates strict, sloppy, respectively.
	if (tag_trimmer->GetTagTrimMethod() == 0 and (not rai->tag_info.is_strict_tag)){
		rai->filtered = true;
	}
}

hash<string> string_hash;

// Hash a family to a 64-bit unsigned long long integer
// The hash key contains the following information
// 1) strand key (2 bits): 00, 01, 10, 11 mean ERROR, BI-DIR, FWD, REV, respectively.
// 2) covered target regions (10 bits where 7 bits indicates the first covered target, 3 bits used to indicate the next three targets are covered or not.
// 3) ZT strictness (1 bit)
// 4) ZT (25 bits): TACT->0123 hash if strict, else random hash.
// 5) YT strictness (1 bit)
// 6) YT (25 bits): TACT->0123 hash if strict, else random hash.
bool MolecularTagManager::FamilyInfoToLongLong(int strand_key, const string& zt, bool is_strict_zt, const string& yt, bool is_strict_yt, const vector<int>& target_indices, int sample_idx, unsigned long long& my_hash) const {
    my_hash = 0;  // my_hash = 0 is reserved for error.
	bool uniquely_hashable = target_indices.size() <= 4 and multisample_prefix_tag_random_bases_idx_[sample_idx].size() <= 12 and + multisample_suffix_tag_random_bases_idx_[sample_idx].size() <= 12;
    if (zt.size() != multisample_prefix_tag_struct_[sample_idx].size() or yt.size() != multisample_suffix_tag_struct_[sample_idx].size()){
    	return false;
    }
    char* long_long_tail_byte = (char*) &my_hash;
    //strand key (2 bits)
    switch (strand_key){
        case 0:
        	// FWD strand
            *long_long_tail_byte = 2;
            break;
        case 1:
        	// REV strand
            *long_long_tail_byte = 3;
            break;
        case -1:
        	// Bi-direction
            *long_long_tail_byte = 1;
            break;
        default:
        	// invalid strand key
            my_hash = 0;
            return false;
    }
    // Shift 7 bits to the left
    my_hash <<= 7;

    // Target coverage
    // Use 7 bits to represent the first covered target idx
    if (not target_indices.empty()){
        *long_long_tail_byte |= (char) ((target_indices[0] % 127) + 1);  // takes values from 1 to 127, 0 is reserved for empty target_indices
    }
    // Shift 3 bits to the left
    my_hash <<= 3;

    // 3 additional nearby covered target indicies
    char bit_mapped_additional_covered_idx = 0;
    for (int idx = 1; idx < min(4, (int) target_indices.size()); ++idx){
        assert(target_indices[idx] > target_indices[idx - 1]);
        int idx_diff_to_first = target_indices[idx] - target_indices[0];
        if (idx_diff_to_first > 3){
            uniquely_hashable = false;
            continue;
        }
        char one = 1;
        bit_mapped_additional_covered_idx += (one <<= (idx_diff_to_first -1));
    }
    // Set the last 3 bits of my_hash to bit_mapped_additional_covered_idx
    *long_long_tail_byte |= bit_mapped_additional_covered_idx;

    for (int tag_key = 0; tag_key < 2; ++tag_key){
        string const * my_tag = NULL;
        vector<int> const * my_idx_vec = NULL;
        bool is_my_tag_strict = false;
        unsigned long long my_tag_hash = 0;
        long_long_tail_byte = (char*) &my_tag_hash;
        // tag_key = 0 for ZT
        if (tag_key == 0){
        	my_tag = &zt;
        	my_idx_vec = &(multisample_prefix_tag_random_bases_idx_[sample_idx]);
        	is_my_tag_strict = is_strict_zt;
        }
        // tag_key = 1 for YT
        else{
        	my_tag = &yt;
        	my_idx_vec = &(multisample_suffix_tag_random_bases_idx_[sample_idx]);
        	is_my_tag_strict = is_strict_yt;
        }
        // Shift 1 bit to the left
        my_hash <<= 1;

        if (is_my_tag_strict){
        	//  Strictness of tag (1 bit) 1 means strict
        	my_hash |= 1;
            // 25 bits for my_tag (ZT or YT)
            my_hash <<= 25;
            // Iterate over random bases of my_tag
            for (vector<int>::const_iterator idx_it = my_idx_vec->begin(); idx_it != my_idx_vec->end(); ++idx_it){
                char my_nuc_in_2_bits = NucTo0123(my_tag->at(*idx_it));
                // Got something other than TACG?
                if (my_nuc_in_2_bits < 0){
                    my_hash = 0;
                    return false;
                }
                // Set the last 2 bits of my_umt_hash to my_nuc_in_2_bits
                (*long_long_tail_byte) |= my_nuc_in_2_bits;
                // Shift 2 bits to the left
                my_tag_hash <<= 2;
            }
            // I shifted 2 bits to the left at the last iteration previously. Must shift 2 bits back to the right.
            my_tag_hash >>= 2;
        }else{
            // 25 bits for my_tag (ZT or YT)
            my_hash <<= 25;
        	// Hash a non-strict zt to a 25 bits: a) hash to a 32 bits integer, b) use a 25-bit mask.
        	my_tag_hash = (unsigned long long) string_hash(*my_tag) & 33554431; // 33554431 = 2^25 - 1, i.e., 25 1's in binary
        }
        my_hash |= my_tag_hash;
    }

    return uniquely_hashable;
}



int ConsensusPositionTicketManager::kNumAppended_ = 0;
int ConsensusPositionTicketManager::kNumDeleted_ = 0;
pthread_mutex_t ConsensusPositionTicketManager::mutex_CPT_;
bool ConsensusPositionTicketManager::debug_ = false;

// Clear all reads in consensus_position_ticket
void ConsensusPositionTicketManager::ClearConsensusPositionTicket(list<PositionInProgress>::iterator &consensus_position_ticket) {
	// Close the consensus_position_ticket if not.
	if (consensus_position_ticket->end){
		CloseConsensusPositionTicket(consensus_position_ticket);
	}
	Alignment* rai = consensus_position_ticket->begin;
	//if (debug_) pthread_mutex_lock(&mutex_CPT_);
	while (rai != consensus_position_ticket->end){
		Alignment* temp_rai = rai;
		rai = rai->next;
		delete temp_rai;
		//++kNumDeleted_;
	}
	//if (debug_) pthread_mutex_unlock(&mutex_CPT_);
	//else kNumDeleted_ = 0;
	consensus_position_ticket->begin = NULL;
	consensus_position_ticket->end = NULL;
}

// Call this function if you want to append a read to a closed consensus_position_ticket
void ConsensusPositionTicketManager::ReopenConsensusPositionTicket(list<PositionInProgress>::iterator &consensus_position_ticket){
	if (not consensus_position_ticket->begin){
		return;
	}
	Alignment* rai = consensus_position_ticket->begin;
	while (rai->next){
		rai = rai->next;
	}
	// Sanity check for the case where consensus_position_ticket is opened.
	// I.e., in AppendConsensusPositionTicket, consensus_position_ticket->end is the last read being appended.
	if (consensus_position_ticket->end){
		assert(consensus_position_ticket->end == rai);
	}
	// Now close the ticket.
	consensus_position_ticket->end = rai;
}

// Append a read to consensus_position_ticket.
void ConsensusPositionTicketManager::AppendConsensusPositionTicket(list<PositionInProgress>::iterator& consensus_position_ticket, Alignment* const alignment){
	alignment->next = NULL;
	if (not consensus_position_ticket->end){
		ReopenConsensusPositionTicket(consensus_position_ticket);
	}
	if (consensus_position_ticket->begin){
		consensus_position_ticket->end->next = alignment;
	}else{
		// empty consensus_position_ticket
		consensus_position_ticket->begin = alignment;
	}
	consensus_position_ticket->end = alignment;
	/*
	if (debug_){
		pthread_mutex_lock(&mutex_CPT_);
		++kNumAppended_;
		pthread_mutex_unlock(&mutex_CPT_);
	}
	*/
}

// Call this function after you finish the task of appending the reads.
void ConsensusPositionTicketManager::CloseConsensusPositionTicket(list<PositionInProgress>::iterator &consensus_position_ticket){
	consensus_position_ticket->end = NULL;
}

int ConsensusPositionTicketManager::ConsensusPositionTicketCounter(const list<PositionInProgress>::iterator &consensus_position_ticket){
	int rai_counts = 0;
	assert(consensus_position_ticket->end == NULL);
	for (Alignment const * rai = consensus_position_ticket->begin; rai != consensus_position_ticket->end; rai = rai->next){
		++rai_counts;
	}
	return rai_counts;
}

// Note that consensus_position_ticket must be generated from all samples!
// I.e., if one sample produces a candidate, then the candidate will be applied to all other samples as well.
void GenerateConsensusPositionTicket(vector< vector< vector<MolecularFamily> > > &my_molecular_families_multisample,
                                     VariantCallerContext &vc,
									 const ConsensusAlignmentManager &consensus,
                                     list<PositionInProgress>::iterator &consensus_position_ticket,
									 bool filter_all_reads_after_done)
{
	unsigned int min_family_size = (unsigned int) vc.parameters->tag_trimmer_parameters.min_family_size;
	unsigned int min_fam_per_strand_cov = (unsigned int) vc.parameters->tag_trimmer_parameters.min_fam_per_strand_cov;
	// First clear consensus_position_ticket
	ConsensusPositionTicketManager::ClearConsensusPositionTicket(consensus_position_ticket);

	// Iterate over samples
	for (vector< vector< vector< MolecularFamily> > >::iterator sample_it = my_molecular_families_multisample.begin(); sample_it != my_molecular_families_multisample.end(); ++sample_it) {
		// Iterate over strands
		for (vector< vector< MolecularFamily> >::iterator strand_it = sample_it->begin(); strand_it != sample_it->end(); ++strand_it) {
			// Iterate over families
			for (vector< MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it !=  strand_it->end(); ++fam_it) {
				// Is the family functional?
				if (not fam_it->SetFuncFromAll(min_family_size, min_fam_per_strand_cov)) {
					continue;
				}
				// Generate strand-specific basespace consensus reads for one family.
				vector<Alignment*> fwd_reads(0);
				vector<Alignment*> rev_reads(0);
				vector< vector<Alignment*>* > my_stranded_reads = {&fwd_reads, &rev_reads};

				if (fam_it->strand_key < 0){
					// strand_key < 0 => Bi-directional family
					fwd_reads.reserve(fam_it->all_family_members.size());
					rev_reads.reserve(fam_it->all_family_members.size());
					for (vector<Alignment*>::iterator read_it = fam_it->all_family_members.begin(); read_it != fam_it->all_family_members.end(); ++read_it){
						my_stranded_reads[((*read_it)->is_reverse_strand? 1 : 0)]->push_back(*read_it);
					}
				}else{
					// strand_key >= 0 => Uni-directional family that has reads on the same strand.
					my_stranded_reads[fam_it->strand_key] = &(fam_it->all_family_members);
				}
				for (vector< vector<Alignment*>* >::iterator my_reads_it = my_stranded_reads.begin(); my_reads_it != my_stranded_reads.end(); ++my_reads_it){
					// Skip if no reads on the strand.
					if ((*my_reads_it)->empty()){
						continue;
					}

					// Generate basespace consensus
					Alignment* alignment = new Alignment;
					bool success = consensus.CalculateConsensus(**my_reads_it, *alignment);
					// Basic filtering
					if (success){
						success = vc.candidate_generator->BasicFilters(*alignment);
					}
					// TrimAmpliseqPrimers (not alignment->target_coverage_indices.empty() implies it has been carried out)
					if (success and alignment->target_coverage_indices.empty()){
						vc.targets_manager->TrimAmpliseqPrimers(alignment, vc.bam_walker->GetRecentUnmergedTarget());
						success = not alignment->filtered;
					}
					// UnpackReadAlleles if not done
					if (success and (not alignment->is_read_allele_unpacked)){
						vc.candidate_generator->UnpackReadAlleles(*alignment);
					}
					// Append to position ticket if success, else release the memory
					if (success){
						ConsensusPositionTicketManager::AppendConsensusPositionTicket(consensus_position_ticket, alignment);
					}else{
						delete alignment;
					}
				}
				if (filter_all_reads_after_done){
					for (vector<Alignment*>::iterator read_it = fam_it->all_family_members.begin(); read_it != fam_it->all_family_members.end(); ++read_it)
						(*read_it)->filtered = true;
				}
			}
		}
	}
	// Note that the next of the last alignemnt is always NULL. MUST close the end of consensus_position_ticket.
	ConsensusPositionTicketManager::CloseConsensusPositionTicket(consensus_position_ticket);
}
