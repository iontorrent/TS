/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

#include "MolecularTag.h"
#include "MiscUtil.h"

float median(vector<float>& v) {
	sort(v.begin(), v.end());
	int n = v.size();
	if (n == 0) {return 0;}
	if ((n % 2) == 1) {return v[n/2];}
	else {int i = n/2; return (v[i-1] + v[i])/2;}
}

float mean(vector<float>& v) {
	unsigned int size = v.size();
	if (size == 0) {return 0;}
	else {
		float sum = 0;
		for (vector<float>::iterator iter = v.begin(); (iter != v.end()); ++iter) {
			sum += *iter;
		}
		return (sum / size);
	}
}

int MolecularFamily::CountFamSizeFromAll()
{
	fam_size_ = 0;
	for (vector<Alignment*>::iterator read_it = all_family_members.begin(); read_it != all_family_members.end(); ++read_it){
		fam_size_ += ((*read_it)->read_count);
	}
	return fam_size_;
}

int MolecularFamily::CountFamSizeFromValid()
{
	valid_fam_size_ = 0;
	for (vector<Alignment*>::iterator read_it = valid_family_members.begin(); read_it != valid_family_members.end(); ++read_it){
		valid_fam_size_ += ((*read_it)->read_count);
	}
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

Consensus::Consensus() {
	min_start_position_ = 0;
	max_read_length_ = 0;
	debug_ = false;
	flow_consensus_ = true;
	error_ = false;
	stitch_ = false;
	iupac_cutoff_ = 1;
}

Consensus::~Consensus() {
}

unsigned int Consensus::GetAlignedFamily(vector<Alignment*>& family_members) {
	min_start_position_ = 999999999;
	start_flow_ = 99999999;
	insertions_.clear();
	aligned_bases_.clear();
	insertion_bases_.clear();
	flow_indexes_.clear();
	measurement_vector_.clear();
	phase_params_.clear();
	flow_index_.clear();
	measurements_.clear();
	soft_clip_offset_.clear();
	max_read_length_ = 0;
	error_ = false;

	unsigned int map_quality = 0;
	for (vector<Alignment*>::iterator iter = family_members.begin(); (iter != family_members.end()); ++iter) {
		if (iter == family_members.begin()) {name_ = (*iter)->alignment.Name;}
		if (debug_) {
			cerr << "Name           " << (*iter)->alignment.Name << endl;
			cerr << "RefID:Position " << (*iter)->alignment.RefID << ":" << (*iter)->alignment.Position << endl;
			cerr << "align_start    " << (*iter)->align_start << endl;
			cerr << "start          " << (*iter)->start << endl;
			cerr << "end            " << (*iter)->end << endl;
			cerr << "QueryBases     " << (*iter)->alignment.QueryBases << endl;
			cerr << "AlignedBases   " << (*iter)->alignment.AlignedBases << endl;
		}
		phase_params_.push_back((*iter)->phase_params);
		map_quality += (*iter)->alignment.MapQuality;
		min_start_position_ = min(min_start_position_, (unsigned int)((*iter)->alignment.Position));
		unsigned int offset = 0;
		for (unsigned int position = min_start_position_; (position < (unsigned int)((*iter)->alignment.Position)); ++position) {offset++;}
		max_read_length_ = max(max_read_length_, (std::basic_string<char>::size_type)(((*iter)->alignment.AlignedBases.length() + offset)));
		unsigned int pos = 0;
		int soft_clip = 0;
		unsigned int cigar_index = 0;
		for (vector<CigarOp>::iterator iterCigar = (*iter)->alignment.CigarData.begin(); (iterCigar != (*iter)->alignment.CigarData.end()); ++iterCigar) {
			if (debug_) {cerr << iterCigar->Type << " " << iterCigar->Length << endl;}
			if (iterCigar == (*iter)->alignment.CigarData.begin()) {
				if (iterCigar->Type == 'S') {
					soft_clip = iterCigar->Length;
				}
			}
			if (iterCigar->Type == 'I') {
				std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.find(pos + offset);
				if (iterInsertions == insertions_.end()) {insertions_[pos + offset] = iterCigar->Length;}
				else {insertions_[pos + offset] = max(iterInsertions->second, iterCigar->Length);}
			}
			if (iterCigar->Type == 'M') {pos += iterCigar->Length;}
			if (iterCigar->Type == 'D') {pos += iterCigar->Length;}
			cigar_index++;
		}
		soft_clip_offset_.push_back(soft_clip);
	}
	map_quality /= family_members.size();
	unsigned int count = 0;
	for (std::map<unsigned int, unsigned int>::iterator iterInsertions = insertions_.begin(); (iterInsertions != insertions_.end()); ++iterInsertions) {
		count += iterInsertions->second;
	}
	max_read_length_ += count;
	insertion_bases_.assign(max_read_length_, '-');
	int read_index = -1;
	for (vector<Alignment*>::iterator iter = family_members.begin(); (iter != family_members.end()); ++iter) {
		read_index++;
		string str = (*iter)->alignment.AlignedBases;
		if (flow_consensus_) {
			flow_indexes_.push_back((*iter)->flow_index);
			if (iter == family_members.begin()) {
				flow_order_index_ = (*iter)->flow_order_index;
				measurements_ = (*iter)->measurements;
				for (vector<float>::iterator init = measurements_.begin(); (init != measurements_.end()); ++init) {*init = 0;}
			}
			int start_flow = 0;
			(*iter)->alignment.GetTag("ZF", start_flow);
			start_flow_ = min(start_flow_, start_flow);
			measurement_vector_.push_back((*iter)->measurements);
			vector<int> hp_counts = (*iter)->flow_index;
			for (unsigned int index = 0; (index < (*iter)->flow_index.size()); ++index) {
				if (index == (*iter)->flow_index.size() - 1) {hp_counts[index] = 1;}
				else {
					unsigned int index2 = index + 1;
					for (; (index2 < (*iter)->flow_index.size()); ++index2) {
						if ((*iter)->flow_index[index2] != (*iter)->flow_index[index]) {break;}
					}
					for (unsigned int index3 = index; (index3 < index2); ++index3) {
						if (index3 == index) {
							hp_counts[index3] = index2 - index;
						} else {
							hp_counts[index3] = 1;
						}
					}
					index += index2 - index - 1;
				}
			}
			int index = 0;
			for (vector<int>::iterator iter2 = (*iter)->flow_index.begin(); (iter2 != (*iter)->flow_index.end()); ++iter2) {
				measurement_vector_[measurement_vector_.size() - 1][*iter2] /= hp_counts[index];
				index++;
			}
		}
		str.reserve(max_read_length_);
		unsigned int insertion_count = 0;
		unsigned int offset = 0;
		unsigned int length = 0;
		str.reserve(max_read_length_);
		char padding = '-';
		if (stitch_) {padding = ' ';}
		for (unsigned int position = min_start_position_; (position < (unsigned int)((*iter)->alignment.Position)); ++position) {str = padding + str; insertion_count++; offset++; length++;}
		std::map<unsigned int, unsigned int> read_insertions;
		unsigned int pos = 0;
		for (vector<CigarOp>::iterator iterCigar = (*iter)->alignment.CigarData.begin(); (iterCigar != (*iter)->alignment.CigarData.end()); ++iterCigar) {
			if (iterCigar->Type == 'I') {
				read_insertions[pos + offset] = iterCigar->Length; length += iterCigar->Length;
			}
			if (iterCigar->Type == 'M') {pos += iterCigar->Length; length += iterCigar->Length;}
			if (iterCigar->Type == 'D') {pos += iterCigar->Length; length += iterCigar->Length;}
		}
		while (str.length() < max_read_length_) {str += padding;}
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
			} else {
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
		read_counts_.push_back((*iter)->read_count);
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
	// base space consensus
	consensus_ = "";
	consensus_.reserve(max_read_length_);
	int A_count = 0;
	int C_count = 0;
	int G_count = 0;
	int T_count = 0;
	int N_count = 0;
	int D_count = 0;
	int I_count = 0;
	int R_count = 0;
	int Y_count = 0;
	int S_count = 0;
	int W_count = 0;
	int K_count = 0;
	int M_count = 0;
	int total_count = 0;
	unsigned int position = min_start_position_;
	while (insertion_bases_.length() < max_read_length_) {insertion_bases_ += '-';}
	for (unsigned int index = 0; (index < max_read_length_); ++index) {
		A_count = 0;
		C_count = 0;
		G_count = 0;
		T_count = 0;
		N_count = 0;
		D_count = 0;
		I_count = 0;
		R_count = 0;
		Y_count = 0;
		S_count = 0;
		W_count = 0;
		K_count = 0;
		M_count = 0;
		total_count = 0;
		vector<int>::iterator read_counts_iter = read_counts_.begin();
		for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
			if (index >= iter->length()) {D_count+=(*read_counts_iter);}
			else {
				switch (iter->at(index)) {
				case ' ': break;
				case 'A': total_count+=(*read_counts_iter); A_count+=(*read_counts_iter); break;
				case 'C': total_count+=(*read_counts_iter); C_count+=(*read_counts_iter); break;
				case 'G': total_count+=(*read_counts_iter); G_count+=(*read_counts_iter); break;
				case 'T': total_count+=(*read_counts_iter); T_count+=(*read_counts_iter); break;
				case 'R': total_count+=(*read_counts_iter); R_count+=(*read_counts_iter); break;
				case 'Y': total_count+=(*read_counts_iter); Y_count+=(*read_counts_iter); break;
				case 'S': total_count+=(*read_counts_iter); S_count+=(*read_counts_iter); break;
				case 'W': total_count+=(*read_counts_iter); W_count+=(*read_counts_iter); break;
				case 'K': total_count+=(*read_counts_iter); K_count+=(*read_counts_iter); break;
				case 'M': total_count+=(*read_counts_iter); M_count+=(*read_counts_iter); break;
				case '-': total_count+=(*read_counts_iter); D_count+=(*read_counts_iter); break;
				case '*': total_count+=(*read_counts_iter); I_count+=(*read_counts_iter); break;
				default: total_count+=(*read_counts_iter); N_count+=(*read_counts_iter);
				}
			}
			++read_counts_iter;
		}
		if (total_count == 0) {D_count = 1; total_count = 1;}
		char ref_base = 0;
		if ((index < insertion_bases_.length()) and (insertion_bases_[index] != '*')) {ref_base = (char)(*(ref_reader.iter(RefID, position)));}

		int allele_count = 0;
		float D_rate = (D_count / (double)total_count);
		float A_rate = (A_count / (double)total_count);
		float C_rate = (C_count / (double)total_count);
		float G_rate = (G_count / (double)total_count);
		float T_rate = (T_count / (double)total_count);
		if (A_rate > iupac_cutoff_) {allele_count++;}
		if (C_rate > iupac_cutoff_) {allele_count++;}
		if (G_rate > iupac_cutoff_) {allele_count++;}
		if (T_rate > iupac_cutoff_) {allele_count++;}
		if ((allele_count == 1) and (D_rate > iupac_cutoff_)) {D_count = 0;}
		if (allele_count > 1) {
			if (debug_) {
				cerr << A_rate << " " << C_rate << " " << G_rate << " " << T_rate << " " << D_rate << endl;
			}
			if ((A_rate > iupac_cutoff_) and (G_rate > iupac_cutoff_)) {
				//consensus_ += "R";
				if (ref_base == 'A') { consensus_ += "G"; } else { consensus_ += "A"; }
			} else if ((C_rate > iupac_cutoff_) and (T_rate > iupac_cutoff_)) {
				//consensus_ += "Y";
				if (ref_base == 'C') { consensus_ += "T"; } else { consensus_ += "C"; }
			} else if ((G_rate > iupac_cutoff_) and (C_rate > iupac_cutoff_)) {
				//consensus_ += "S";
				if (ref_base == 'G') { consensus_ += "C"; } else { consensus_ += "G"; }
			} else if ((A_rate > iupac_cutoff_) and (T_rate > iupac_cutoff_)) {
				//consensus_ += "W";
				if (ref_base == 'A') { consensus_ += "T"; } else { consensus_ += "A"; }
			} else if ((G_rate > iupac_cutoff_) and (T_rate > iupac_cutoff_)) {
				//consensus_ += "K";
				if (ref_base == 'G') { consensus_ += "T"; } else { consensus_ += "G"; }
			} else if ((A_rate > iupac_cutoff_) and (C_rate > iupac_cutoff_)) {
				//consensus_ += "M";
				if (ref_base == 'A') { consensus_ += "C"; } else { consensus_ += "A"; }
			} else {consensus_ += "N";}
		} else {
			int max_count = max(max(max(max(max(max(max(max(max(max(max(max(A_count, C_count), G_count), T_count), N_count), D_count), I_count), R_count), Y_count), S_count), W_count), K_count), M_count);
			if ((ref_base == 'G') and (max_count == G_count)) {C_count = 0; T_count = 0; A_count = 0; D_count = 0;}
			else if ((ref_base == 'C') and (max_count == C_count)) {G_count = 0; T_count = 0; A_count = 0; D_count = 0;}
			else if ((ref_base == 'T') and (max_count == T_count)) {C_count = 0; G_count = 0; A_count = 0; D_count = 0;}
			else if ((ref_base == 'A') and (max_count == A_count)) {C_count = 0; G_count = 0; T_count = 0; D_count = 0;}
			if (max_count == I_count) {consensus_ += '-';}
			else if (max_count == R_count) {consensus_ += 'R';}
			else if (max_count == Y_count) {consensus_ += 'Y';}
			else if (max_count == S_count) {consensus_ += 'S';}
			else if (max_count == W_count) {consensus_ += 'W';}
			else if (max_count == K_count) {consensus_ += 'K';}
			else if (max_count == M_count) {consensus_ += 'M';}
			else if (max_count == G_count) {consensus_ += 'G';}
			else if (max_count == C_count) {consensus_ += 'C';}
			else if (max_count == T_count) {consensus_ += 'T';}
			else if (max_count == A_count) {consensus_ += 'A';}
			else if (max_count == D_count) {consensus_ += '-';}
			else if (max_count == N_count) {consensus_ += 'N';}
		}
		if (insertion_bases_[index] != '*') {position++;}
	}
	// flow space consensus
	if (flow_consensus_) {
		if (reverse_) {
			RevComplementInPlace(consensus_);
			for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
				RevComplementInPlace(*iter);
			}
		}
		vector<unsigned int> positions;
		unsigned int read_index = 0;
		for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {positions.push_back(soft_clip_offset_[read_index]); read_index++;}
		// setup flow prefix
		vector<float> values;
		values.reserve(flow_indexes_.size());
		unsigned int flow_index = start_flow_;
		for (unsigned int flow = 0; (flow < (unsigned int)start_flow_); ++flow) {
			values.clear();
			read_index = 0;
			for (vector<vector<int> >::iterator iter = flow_indexes_.begin(); (iter != flow_indexes_.end()); ++iter) {
				values.push_back(measurement_vector_[read_index][flow]);
				read_index++;
			}
			if (flow >= measurements_.size()) {measurements_.resize(flow + 1);}
			measurements_[flow] = median(values);
		}
		vector<float> v;
		v.reserve(flow_indexes_.size());
		flow_index_.reserve(max_read_length_);
		for (unsigned int index = 0; (index < consensus_.length()); ++index) {
			if (consensus_[index] != '-') {
				while ((flow_index < flow_order_.length()) and (consensus_[index] != flow_order_[flow_index])) {flow_index++;}
				if (flow_index >= flow_order_.length()) {consensus_ = consensus_.substr(0, index);}
				else {
					read_index = 0;
					v.clear();
					for (vector<vector<int> >::iterator iter = flow_indexes_.begin(); (iter != flow_indexes_.end()); ++iter) {
						if ((positions[read_index] < (*iter).size()) and (index < aligned_bases_[read_index].length())) {
							if (aligned_bases_[read_index][index] == consensus_[index]) {
								unsigned int flow = iter->at(positions[read_index]);
								v.push_back(measurement_vector_[read_index][flow]);
							}
						}
						read_index++;
					}
					flow_index_.push_back(flow_index);
					measurements_[flow_index] += median(v);
				}
			}
			read_index = 0;
			for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
				if (index < iter->length()) {
					if ((iter->at(index) != ' ') and (iter->at(index) != '-') and (iter->at(index) != '*')) {positions[read_index]++;}
				}
				read_index++;
			}
		}
		if (reverse_) {
			RevComplementInPlace(consensus_);
			for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
				RevComplementInPlace(*iter);
			}
		}
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
		for (int pos = (min((unsigned int)consensus_.length(), iterInsertions->first + iterInsertions->second + offsets[offset_index]) - 1); (pos >= (int)(iterInsertions->first + offsets[offset_index])); --pos) {
			if (pos >= (int)consensus_.size()) {continue;}
			if (pos < 0) {break;}
			if (consensus_[pos] == '-') {
				if (pos == (int)consensus_.length() - 1) {consensus_ = consensus_.substr(0, pos);}
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
					deletion_count++;
				} else {
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
		} else {
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
void Consensus::PartiallyResetAlignment_(Alignment& alignment)
{
	long int start = alignment.start;
	long int end = alignment.end;
	alignment.Reset();
	alignment.start = start;
	alignment.end = end;
}


bool Consensus::CalculateConsensus(ReferenceReader& ref_reader, vector<Alignment*>& family_members, Alignment& alignment, const string& flow_order) {
	// Empty family_members
	if (family_members.empty()) {
		return false;
	}

	// Trivial tie case
	if (family_members.size() == 2){
		if (family_members[0]->read_count == family_members[1]->read_count){
			// I pick the read with the better mapping quality to be the consensus.
			// In this case, I implicitly prefer to get the reference alleles.
			alignment = (family_members[0]->alignment.MapQuality > family_members[1]->alignment.MapQuality)? *(family_members[0]) : *(family_members[1]);
			alignment.read_count = family_members[0]->read_count + family_members[1]->read_count;
			PartiallyResetAlignment_(alignment);
			return true;
		}
	}

	int fam_size = 0;
	int max_read_count = -1;
	unsigned int max_read_count_idx = 0;

	// Calculate the family size and find the read w/ max read count.
	for (unsigned int read_idx = 0; read_idx < family_members.size(); ++read_idx){
		int read_count = family_members[read_idx]->read_count;
		fam_size += read_count;
		if (read_count > max_read_count){
			max_read_count = read_count;
			max_read_count_idx = read_idx;
		}
	}
	// Check if family_members has a consensus read that dominates all the others.
	// If true, I am actually calculating the consensus of consensus reads, which is trivial in this case.
	// It should be the case most of the time if the input bam is a consensus bam.
	if (2 * max_read_count >= fam_size){
		alignment = *(family_members[max_read_count_idx]);
		alignment.read_count = fam_size;
		PartiallyResetAlignment_(alignment);
		return true;
	}

	// If I didn't input flow_order, I can't generate flow consensus. So I go through the base space version instead.
	bool flow_consensus_temp = flow_consensus_;
	if (flow_order.empty()){
		SetFlowConsensus(false);
	}

	flow_order_ = flow_order;
	alignment = Alignment(*(family_members[0]));
	alignment.read_count = fam_size;
	PartiallyResetAlignment_(alignment);
	reverse_ = alignment.is_reverse_strand;
	debug_ = false;
	bool show_variants = false;
	int show_chr = 1;
	long int show_pos = 2488153;
	//if ((family_members[0]->alignment.RefID == (show_chr - 1)) and (family_members[0]->alignment.Position <= show_pos) and (family_members[0]->alignment.GetEndPosition() >= show_pos)) {show_variants = true;}
	//if (alignment.alignment.Name == "7XBVF:02889:02037") {debug_ = true;}
	if (debug_) {cerr << endl << family_members.size() << " " << family_members[0]->alignment.RefID << " " << family_members[0]->alignment.Position << endl;}

	min_start_position_ = 0;
	insertions_.clear();
	aligned_bases_.clear();
	max_read_length_ = 0;
	consensus_.clear();
	insertion_bases_.clear();
	new_insertions_.clear();
	new_cigar_.clear();
//static Timer timer1("GetAlignedFamly");
//timer1.start();
	unsigned int map_quality = GetAlignedFamily(family_members);
	if (map_quality == 0) {
		SetFlowConsensus(flow_consensus_temp);
		return false;
	}
//timer1.end();

//static Timer timer2("GetConsensus");
//timer2.start();
	if (debug_) {cerr << "GetConsensus" << endl;}
	GetConsensus(ref_reader, alignment.alignment.RefID);
//timer2.end();

//static Timer timer3("TrimConsensus");
//timer3.start();
	if (debug_) {cerr << "TrimConsensus" << endl;}
	unsigned int new_position = TrimConsensus();
//timer3.end();

//static Timer timer4("CalculateCigar");
//timer4.start();
	if (debug_) {cerr << "CalculateCigar" << endl;}
	CalculateCigar();
//timer4.end();

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
	alignment.flow_index = flow_index_;
	alignment.measurements = measurements_;
	alignment.read_bases = query_bases;
	if (debug_) {
		bool run_exit = false;
		cerr << endl;
		cerr << alignment.alignment.Name << endl;

		unsigned int index = 0;
		for (vector<float>::iterator iter = alignment.measurements.begin(); (iter != alignment.measurements.end()); ++iter) {
			cerr << "measurements " << index << " " << *iter << endl;
			index++;
		}
		vector<int> hp_counts = alignment.flow_index;
		for (index = 0; (index < alignment.flow_index.size()); ++index) {
			if (index == alignment.flow_index.size() - 1) {hp_counts[index] = 1;}
			else {
				unsigned int index2 = index + 1;
				for (; (index2 < alignment.flow_index.size()); ++index2) {
					if (alignment.flow_index[index2] != alignment.flow_index[index]) {break;}
				}
				for (unsigned int index3 = index; (index3 < index2); ++index3) {
					hp_counts[index3] = index2 - index;
				}
				index += index2 - index - 1;
			}
		}
		index = 0;
		for (vector<int>::iterator iter = alignment.flow_index.begin(); (iter != alignment.flow_index.end()); ++iter) {
			if (*iter < (int)flow_order_.length()) {
				cerr << "flow_index " << index << " " << *iter << " " << alignment.measurements[*iter] << " " << alignment.read_bases[index] << " " << flow_order_[*iter] << " " << hp_counts[index] << " " << (alignment.measurements[*iter] / hp_counts[index]) << endl;
			}
			index++;
		}

		cerr << start << endl;
		cerr << end << endl;
		cerr << new_position << endl;
		cerr << query_bases << endl;
		cerr << consensus_ << endl;
		for (vector<CigarOp>::iterator iterCigar = new_cigar_.begin(); (iterCigar != new_cigar_.end()); ++iterCigar) {
			cerr << iterCigar->Type << " " << iterCigar->Length << endl;
			if (iterCigar->Type == 'D') {run_exit = true;}
		}
		if (run_exit) {
			//exit(1);
		}
		cerr << endl;
	}
	alignment.start = start;
	alignment.end = end;
	alignment.alignment.Position = new_position;
	alignment.alignment.MapQuality = map_quality;
	alignment.alignment.QueryBases = query_bases;
	alignment.alignment.Length = query_bases.length();
	alignment.alignment.Qualities = "*";
	alignment.alignment.Length = query_bases.length();
	alignment.alignment.AlignedBases = consensus_;
	alignment.alignment.CigarData.swap(new_cigar_);

	alignment.alignment.RemoveTag("NM");
	alignment.alignment.RemoveTag("XM");
	alignment.alignment.RemoveTag("MD");
	alignment.alignment.RemoveTag("ZF");
	alignment.alignment.RemoveTag("ZP");
	alignment.alignment.RemoveTag("ZM");
	alignment.alignment.RemoveTag("ZN");
	alignment.alignment.RemoveTag("ZR");
	// calculate ZM tag
	vector<int16_t> measurements;
	for (vector<float>::iterator iter = measurements_.begin(); (iter != measurements_.end()); ++iter) {
		measurements.push_back((int)(*iter * 256));
	}
	// calculate MD tag
	//bool snp = false;
	string md = "";
	int match_count = 0;
	int mismatch_count = 0;
	int total_match_count = 0;
	unsigned int ref_pos = alignment.start;
	unsigned int seq_pos = alignment.start;
	for (vector<CigarOp>::iterator iterCigar = new_cigar_.begin(); (iterCigar != new_cigar_.end()); ++iterCigar) {
		if (iterCigar->Type == 'M') {
			for (unsigned int i = 0; (i < iterCigar->Length); ++i) {
				if ((unsigned int)(seq_pos - alignment.start) < alignment.alignment.AlignedBases.length()) {
					if (alignment.alignment.AlignedBases[seq_pos - alignment.start] == (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)))) {
						match_count++;
						total_match_count++;
					} else {
						//snp = true;
						mismatch_count++;
						if (match_count > 0) {md += std::to_string(match_count);}
						md += alignment.alignment.AlignedBases[seq_pos - alignment.start];
						match_count = 0;
					}
				}
				ref_pos++;
				seq_pos++;
			}
		} else if (iterCigar->Type == 'D') {
			for (unsigned int i = 0; (i < iterCigar->Length); ++i) {
				if ((unsigned int)(seq_pos - alignment.start) < alignment.alignment.AlignedBases.length()) {
					if (alignment.alignment.AlignedBases[seq_pos - alignment.start] == '-') {
						if (match_count > 0) {md += std::to_string(match_count);}
						md += "^";
						md += (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)));
						match_count = 0;
					}
				}
				ref_pos++;
				seq_pos++;
			}
		} else if (iterCigar->Type == 'I') {
			seq_pos += iterCigar->Length;
		}
	}
	if (match_count > 0) {md += std::to_string(match_count);}
	// calculate ZP tag
	vector<float> phase_params;
	unsigned int max_size = 0;
	for (vector<vector<float> >::iterator iter = phase_params_.begin(); (iter != phase_params_.end()); ++iter) {
		if (iter->size() > max_size) {max_size = iter->size();}
	}
	vector<float> values;
	unsigned int index = 0;
	while (index < max_size) {
		values.clear();
		for (vector<vector<float> >::iterator iter = phase_params_.begin(); (iter != phase_params_.end()); ++iter) {
			if (index < iter->size()) {values.push_back(iter->at(index));}
		}
		if (values.size() > 0) {
			phase_params.push_back(mean(values));
		}
		index++;
	}
	alignment.alignment.AddTag("NM", "i", mismatch_count);
	alignment.alignment.AddTag("XM", "i", total_match_count);
	alignment.alignment.AddTag("MD", "Z", md);
	alignment.alignment.AddTag("ZF", "i", start_flow_);
	alignment.alignment.AddTag("ZP", phase_params);
	alignment.alignment.AddTag("ZM", measurements);
	
	// New tags "ZR", "ZN", "ZS" from consensus
	/*
	vec_temp.resize(measurements_sd.size());
	for (unsigned int i_flow = 0; i_flow < measurements_sd.size(); ++i_flow){
    	vec_temp[i_flow] = (int16_t) (measurements_sd[i_flow] * 256.0f);
    }
	alignment.EditTag("ZS", vec_temp);
	*/
	alignment.alignment.AddTag("ZR", "i", (int) family_members.size()); // ZR tag = read count, number of reads that form the consensus read
	string read_names = "";
	for (unsigned int i_member = 0; i_member < family_members.size(); ++i_member){
		read_names += family_members[i_member]->alignment.Name;
		if (i_member != family_members.size() - 1){
			read_names += ";";
		}
	}
	for (string::iterator c_it = read_names.begin(); c_it != read_names.end(); ++c_it)
		if (*c_it == ':')  {*c_it = '.';}
	alignment.alignment.AddTag("ZN", "Z", read_names);  // ZN tag = query names of the reads that from the consensus read

	if (flow_consensus_) {
		if (alignment.read_bases.length() != alignment.flow_index.size()) {
			error_ = true;
			cerr << "consensus " << alignment.alignment.Name << " read_bases.length() " << alignment.read_bases.length() << " flow_index.size() " << alignment.flow_index.size() << endl;
		} else {
			if (reverse_) {RevComplementInPlace(alignment.read_bases);}
			unsigned int index = 0;
			for (vector<int>::iterator iter = alignment.flow_index.begin(); (iter != alignment.flow_index.end()); ++iter) {
				if (((*iter) < (int)flow_order_.length()) and (alignment.read_bases[index] != flow_order_[*iter])) {
					error_ = true;
					cerr << "consensus " << alignment.alignment.Name << " base mismatch at " << index << endl;
					break;
				}
				index++;
			}
			unsigned int flow = start_flow_;
			unsigned int base_index = 0;
			while (base_index < query_bases.length() and flow < flow_order_.length()) {
				while (flow < flow_order_.length() and flow_order_[flow] != alignment.read_bases[base_index]) {
					flow++;
				}
				if (flow >= flow_order_.length()) {break;}
				base_index++;
			}
			if (reverse_) {RevComplementInPlace(alignment.read_bases);}
			if (base_index != query_bases.length()) {
				error_ = true;
				cerr << "WARNING in MolecularTag::CalculateConsensus: There are more bases in the read than fit into the flow order.\t" << alignment.alignment.Name << endl;
				cerr << query_bases << endl;
			}
		}
	}
	if (error_) {cerr << "Read(s) have flow errors. Consensus rejected.\t" << alignment.alignment.Name << endl;}
	if ((not error_) and ((debug_) or (show_variants))) {
		cerr << alignment.alignment.Name << " " << (alignment.is_reverse_strand ? "reverse" : "forward") << endl << endl;
		string ref_seq = "";
		string allele = "";
		string ref = "";
		string gt = "";
		unsigned int pos = 0;
		unsigned int ref_pos = alignment.start;
		unsigned int seq_pos = alignment.start;
		unsigned int variant_count = 0;
		for (vector<CigarOp>::iterator iterCigar = alignment.alignment.CigarData.begin(); (iterCigar != alignment.alignment.CigarData.end()); ++iterCigar) {
			if (iterCigar->Type == 'M') {
				for (unsigned int i = 0; (i < iterCigar->Length); ++i) {
					ref = (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)));
					if ((unsigned int)(seq_pos - alignment.start) < alignment.alignment.AlignedBases.length()) {
						if (alignment.alignment.AlignedBases[seq_pos - alignment.start] != (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)))) {
							pos = ref_pos;
							allele = alignment.alignment.AlignedBases[seq_pos - alignment.start];
							gt = "0/1";
							if ((allele == "A") or (allele == "C") or (allele == "G") or (allele == "T")) {gt = "1/1";}
							if ((allele == "R") and (ref == "A")) {allele = "G";}
							else if ((allele == "R") and (ref == "G")) {allele = "A";}
							else if (allele == "R") {allele = "A,G"; gt = "1/2";}
							else if ((allele == "Y") and (ref == "C")) {allele = "T";}
							else if ((allele == "Y") and (ref == "T")) {allele = "C";}
							else if (allele == "Y") {allele = "C,T"; gt = "1/2";}
							else if ((allele == "S") and (ref == "G")) {allele = "C";}
							else if ((allele == "S") and (ref == "C")) {allele = "G";}
							else if (allele == "S") {allele = "C,G"; gt = "1/2";}
							else if ((allele == "W") and (ref == "A")) {allele = "T";}
							else if ((allele == "W") and (ref == "T")) {allele = "A";}
							else if (allele == "W") {allele = "A,T"; gt = "1/2";}
							else if ((allele == "K") and (ref == "G")) {allele = "T";}
							else if ((allele == "K") and (ref == "T")) {allele = "G";}
							else if (allele == "K") {allele = "G,T"; gt = "1/2";}
							else if ((allele == "M") and (ref == "A")) {allele = "C";}
							else if ((allele == "M") and (ref == "C")) {allele = "A";}
							else if (allele == "M") {allele = "A,C"; gt = "1/2";}
							if (alignment.alignment.RefID + 1 == 23) {
								cerr << "chrX" << "\t" << (pos + 1) << "\t" << "." << "\t" << (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos))) << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=snp" << "\t" << "GT" << "\t" << gt << ":" << endl;
							} else {
								cerr << "chr" << (alignment.alignment.RefID + 1) << "\t" << (pos + 1) << "\t" << "." << "\t" << (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos))) << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=snp" << "\t" << "GT" << "\t" << gt << ":" << endl;
							}
							std::transform(ref.begin(), ref.end(), ref.begin(), ::tolower);
							variant_count++;
						}
					}
					ref_seq += ref;
					ref_pos++;
					seq_pos++;
				}
			} else if (iterCigar->Type == 'D') {
				pos = ref_pos - 1;
				ref = (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos - 1)));
				allele = alignment.alignment.AlignedBases[seq_pos - alignment.start - 1];
				for (unsigned int i = 0; (i < iterCigar->Length); ++i) {
					ref += (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)));
					ref_seq += (char)(*(ref_reader.iter(alignment.alignment.RefID, ref_pos)));
					ref_pos++;
					seq_pos++;
				}
				if (alignment.alignment.RefID + 1 == 23) {
					cerr << "chrX" << "\t" << (pos + 1) << "\t" << "." << "\t" << ref << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=del" << "\t" << "GT" << "\t" << "0/1:" << endl;
				} else {
					cerr << "chr" << (alignment.alignment.RefID + 1) << "\t" << (pos + 1) << "\t" << "." << "\t" << ref << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=del" << "\t" << "GT" << "\t" << "0/1:" << endl;
				}
				variant_count++;
			} else if (iterCigar->Type == 'I') {
				pos = ref_pos - 1;
				allele = alignment.alignment.AlignedBases[seq_pos - alignment.start - 1];
				for (unsigned int i = 0; (i < iterCigar->Length); ++i) {
					allele += alignment.alignment.AlignedBases[seq_pos - alignment.start];
					ref_seq += "-";
					seq_pos++;
				}
				if (alignment.alignment.RefID + 1 == 23) {
					cerr << "chrX" << "\t" << (pos + 1) << "\t" << "." << "\t" << (char)(*(ref_reader.iter(alignment.alignment.RefID, pos))) << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=ins" << "\t" << "GT" << "\t" << "0/1:" << endl;
				} else {
					cerr << "chr" << (alignment.alignment.RefID + 1) << "\t" << (pos + 1) << "\t" << "." << "\t" << (char)(*(ref_reader.iter(alignment.alignment.RefID, pos))) << "\t" << allele << "\t" << 100 << "\t" << "PASS" << "\t" << "TYPE=ins" << "\t" << "GT" << "\t" << "0/1:" << endl;
				}
				variant_count++;
			}
		}
		cerr << endl;
		for (vector<string>::iterator iter = aligned_bases_.begin(); (iter != aligned_bases_.end()); ++iter) {
			cerr << *iter << endl;
		}
		cerr << endl << ref_seq << endl;
		cerr << alignment.alignment.AlignedBases << endl;
		cerr << endl << "Done" << endl;
	}
	SetFlowConsensus(flow_consensus_temp);
	return (not error_);
}

void MolecularTagManager::Initialize(MolecularTagTrimmer* const input_tag_trimmer, const SampleManager* const sample_manager)
{
	tag_trimmer = input_tag_trimmer;
	if (not tag_trimmer->HaveTags()){
		multisample_prefix_tag_struct_.clear();
		multisample_suffix_tag_struct_.clear();
		return;
	}
	multisample_prefix_tag_struct_.assign(sample_manager->num_samples_, "");
	multisample_suffix_tag_struct_.assign(sample_manager->num_samples_, "");
    cout << "MolecularTagManager: Found "<< tag_trimmer->NumTaggedReadGroups() << " read group(s) with molecular tags." << endl;
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

    for (unsigned int sample_idx = 0; sample_idx < sample_manager->sample_names_.size(); ++sample_idx){
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
	for (string::const_iterator struct_it = tag_struct.begin(); struct_it != tag_struct.end(); ++struct_it){
		if ((*struct_it != 'N') and (*tag_it != *struct_it)){
			return false;
		}
		++tag_it;
	}
	return true;
}

// return true if the tags match the tag structures.
bool MolecularTagManager::IsStrictTag(const string& prefix_tag, const string& suffix_tag, int sample_idx) const
{
	sample_idx = max(0, sample_idx); // sample_idx = -1 indicates no multisample
	if (not IsStrictness(prefix_tag, multisample_prefix_tag_struct_[sample_idx])){
		return false;
	}
	if (not IsStrictness(suffix_tag, multisample_suffix_tag_struct_[sample_idx])){
		return false;
	}
	return true;
}

// sample_idx = -1 indicates no multisample
bool MolecularTagManager::IsStrictSuffixTag(const string& suffix_tag, int sample_idx) const
{
	return IsStrictness(suffix_tag, multisample_suffix_tag_struct_[max(0, sample_idx)]);
}

// sample_idx = -1 indicates no multisample
bool MolecularTagManager::IsStrictPrefixTag(const string& prefix_tag, int sample_idx) const
{
	return IsStrictness(prefix_tag, multisample_prefix_tag_struct_[max(0, sample_idx)]);
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



char MolecularFamilyGenerator::NucTo0123_(char nuc) const
{
	if (nuc == 'A') { return 0; }
	if (nuc == 'C') { return 1; }
	if (nuc == 'G') { return 2; }
	if (nuc == 'T') { return 3; }
	if (nuc == 'a') { return 0; }
	if (nuc == 'c') { return 1; }
	if (nuc == 'g') { return 2; }
	if (nuc == 't') { return 3; }
	return -1;
}

// Isomorphisim mapping between a string of TACG and a long long integer.
// I.e., I use 2 bits to represent 1 nuc. The sign of the long long integer is preserved for error handling.
// I currently support base_seq of length < 32.
// return -1 if the hash is not successful.
long long MolecularFamilyGenerator::BaseSeqToLongLong_(const string& base_seq) const
{
	if (base_seq.size() > 31){
		cerr << "ERROR: Cannot hash a string of char TACG of length >= 32 to a 64-bit long long integer." << endl;
		exit(1);
		return -1;
	}

	long long base_seq_long_long = 0;
	char* long_long_tail_byte = (char*) &base_seq_long_long; // pointer to the first byte of base_seq_long_long, like a bit mask.
	bool non_valid_string = false;
	for (string::const_iterator it = base_seq.begin(); it != base_seq.end(); ++it) {
		char nuc_in_0123 = NucTo0123_(*it);
		non_valid_string += (nuc_in_0123 < 0);
		base_seq_long_long <<= 2; // shift 2 bits to the left
		*long_long_tail_byte |= nuc_in_0123; // set the first 2 bits of barcode_long_long from the right to be nuc_in_0123
	}
	if (non_valid_string){
		cerr << "ERROR: Invalid molecular tag that contains a non-TACG character." << endl;
		exit(1);
		return -1;
	}
	return base_seq_long_long;
}

void MolecularFamilyGenerator::FindFamilyForOneRead_(Alignment* rai, vector< vector<MolecularFamily> >& my_molecular_families)
{
	string mol_tag = rai->tag_info.prefix_mol_tag + rai->tag_info.suffix_mol_tag;
	int strand_key = (rai->is_reverse_strand)? 1 : 0;
	bool is_new_tag = true;
	unsigned int tag_index_in_my_molecular_families = 0;

	// Hashing mol_tag to a long long integer facilitates the mapping between the mol_tag to the index of my_molecular_families.
	// Note that the length of the mol_tag must be < 32.
	if (long_long_hashable_){
		long long long_long_tag = BaseSeqToLongLong_(mol_tag);
		pair< map<long long, unsigned int>::iterator, bool> tag_finder;
		// map mol_tag_long_long to the index of my_family_[strand_key] for mol_tag
		tag_finder = long_long_tag_lookup_table_[strand_key].insert(pair<long long, unsigned int>(long_long_tag, my_molecular_families[strand_key].size()));
		// Note that map::insert will not insert value into the key of the map if key is pre-existed.
		// tag_finder.first->first = mol_tag_long_long
		// tag_finder.first->second = the index of my_molecular_families[strand_key] for tag
		// tag_finder.second indicates inserted or not.
		// tag_finder.second = false if I got mol_tag previously, and hence long_long_tag_lookup_table_ is not updated.
		// tag_finder.second = true if this is the first time we get mol_tag, and hence mol_tag_long_long is inserted into long_long_tag_lookup_table_[strand_key]

		is_new_tag = tag_finder.second;
		tag_index_in_my_molecular_families = tag_finder.first->second;
	}
	// Map a string to the index of my_molecular_families, slow but always safe.
	else{
		pair< map<string, unsigned int>::iterator, bool> tag_finder;
		tag_finder = string_tag_lookup_table_[strand_key].insert(pair<string, unsigned int>(mol_tag, my_molecular_families[strand_key].size()));
		is_new_tag = tag_finder.second;
		tag_index_in_my_molecular_families = tag_finder.first->second;
	}

	if (is_new_tag){
		// Generate a new family since this is the first time I get the mol_tag
		my_molecular_families[strand_key].push_back(MolecularFamily(mol_tag, strand_key));
	}
	// Add the read to the family
	my_molecular_families[strand_key][tag_index_in_my_molecular_families].AddNewMember(rai);
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
	bool is_consensus_bam = false;
	unsigned int prefix_tag_len = (mol_tag_manager->GetPrefixTagStruct(sample_index)).size();
	unsigned int suffix_tag_len = (mol_tag_manager->GetSuffixTagStruct(sample_index)).size();
    long_long_hashable_ = (prefix_tag_len + suffix_tag_len) < 32;

	my_molecular_families.resize(2); 	// my_molecular_families[0] for fwd strand, my_molecular_families[1] for rev strand
	long_long_tag_lookup_table_.resize(2);
	string_tag_lookup_table_.resize(2);

	for (int i_strand = 0; i_strand < 2; ++i_strand){
		my_molecular_families[i_strand].resize(0);
		my_molecular_families[i_strand].reserve(20000); // Reverse for 20000 families (including non-functional ones) per strand should be enough most of the time.
		long_long_tag_lookup_table_[i_strand].clear();
		string_tag_lookup_table_[i_strand].clear();
	}

	for (Alignment* rai = bam_position.begin; rai != bam_position.end; rai = rai->next) {
		if (rai == NULL) {
			bam_position.end = NULL;
			return;
		}

		if (rai->filtered or (not rai->tag_info.HasTags())) {
			continue;
		}

		// skip the read if it is not for this sample
		if( sample_index >=0 and rai->sample_index != sample_index) {
			continue;
		}

		// Tag length check
		if ((rai->tag_info.prefix_mol_tag.size() > 0 and rai->tag_info.prefix_mol_tag.size() != prefix_tag_len)
				or (rai->tag_info.suffix_mol_tag.size() > 0 and rai->tag_info.suffix_mol_tag.size() != suffix_tag_len)){
			cerr << "MolecularFamilyGenerator: Warning: The molecular tag length of the read "<< rai->alignment.Name << " doesn't match the tag structure provided in the bam header." << endl;
			continue;
		}

		// Skip the read whose tags don't exactly match the tag structures if I use strict trim.
		// Note mol_tag_manager->tag_trimmer->tag_trim_method_ = 0, 1 indicates strict, sloppy, respectively.
		if (mol_tag_manager->tag_trimmer->GetTagTrimMethod() == 0){
			if (not mol_tag_manager->IsStrictTag(rai->tag_info.prefix_mol_tag, rai->tag_info.suffix_mol_tag, sample_index)){
				continue;
			}
		}

		// If any read has read_count > 1 then I am dealing with a consensus bam
		is_consensus_bam += (rai->read_count > 1);
		FindFamilyForOneRead_(rai, my_molecular_families);
	}

	if (is_split_families_by_region_){
		SplitFamiliesByRegion_(my_molecular_families);
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

// Split a family into multiple if the reads cover different target regions.
void MolecularFamilyGenerator::SplitFamiliesByRegion_(vector< vector<MolecularFamily> >& my_molecular_families) const
{
	if (not is_split_families_by_region_){
		return;
	}
	for (vector<vector<MolecularFamily> >::iterator strand_it = my_molecular_families.begin(); strand_it != my_molecular_families.end(); ++strand_it){
		unsigned int original_fam_num_on_strand = strand_it->size();
		for (unsigned int i_fam = 0; i_fam < original_fam_num_on_strand; ++i_fam){
			if (strand_it->at(i_fam).all_family_members.size() < 2){
				continue;
			}
			bool need_split = false;
			vector<Alignment*>::iterator read_it_0 = strand_it->at(i_fam).all_family_members.begin();
			for (vector<Alignment*>::iterator read_it = (read_it_0 + 1); read_it !=  strand_it->at(i_fam).all_family_members.end(); ++read_it){
				if ((*read_it)->target_coverage_indices != (*read_it_0)->target_coverage_indices){
					// Need to split the family if there is one read has a different target_coverage_indices.
					need_split = true;
					break;
				}
			}
			if (need_split){
				vector<MolecularFamily> splitted_families_from_one_family;
				splitted_families_from_one_family.clear();
				for (vector<Alignment*>::iterator read_it = strand_it->at(i_fam).all_family_members.begin(); read_it != strand_it->at(i_fam).all_family_members.end(); ++read_it){
					bool is_target_coverage_indices_exists = false;
					vector<MolecularFamily>::iterator splitted_fam_it = splitted_families_from_one_family.begin();
					for (; splitted_fam_it != splitted_families_from_one_family.end(); ++splitted_fam_it){
						if (splitted_fam_it->all_family_members[0]->target_coverage_indices == (*read_it)->target_coverage_indices){
							is_target_coverage_indices_exists = true;
							break;
						}
					}

					if (is_target_coverage_indices_exists){
						splitted_fam_it->AddNewMember(*read_it);
					}else{
						splitted_families_from_one_family.push_back(MolecularFamily(strand_it->at(i_fam).family_barcode, strand_it->at(i_fam).strand_key));
						splitted_families_from_one_family.back().AddNewMember(*read_it);
					}
				}
				strand_it->at(i_fam).all_family_members.swap(splitted_families_from_one_family.begin()->all_family_members);
				for (vector<MolecularFamily>::iterator splitted_fam_it = splitted_families_from_one_family.begin() + 1; splitted_fam_it != splitted_families_from_one_family.end(); ++splitted_fam_it){
					strand_it->push_back(*splitted_fam_it);
				}
			}
		}
	}
}

/*
// Note that consensus_position_ticket must be generated from all samples!
// I.e., if one sample produces a candidate, then the candidate will be applied to all other samples as well.
void GenerateConsensusPositionTicket(vector< vector< vector<MolecularFamily> > > &my_molecular_families_multisample,
                                     VariantCallerContext &vc,
                                     Consensus &consensus,
                                     list<PositionInProgress>::iterator &consensus_position_ticket,
									 bool filter_all_reads_after_done) {
	unsigned int min_family_size = (unsigned int) vc.parameters->tag_trimmer_parameters.min_family_size;
	for (vector< vector< vector< MolecularFamily> > >::iterator sample_it = my_molecular_families_multisample.begin(); sample_it != my_molecular_families_multisample.end(); ++sample_it) {
		for (vector< vector< MolecularFamily> >::iterator strand_it = sample_it->begin(); strand_it != sample_it->end(); ++strand_it) {
			for (vector< MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it !=  strand_it->end(); ++fam_it) {
				// Is *fam_it functional?
				if (fam_it->SetFuncFromAll(min_family_size)) {
					if (fam_it->all_family_members.size() >= 1) {
						map<int, vector<Alignment*> > alignment_map;
						for (vector<Alignment*>::iterator iter = fam_it->all_family_members.begin(); (iter != fam_it->all_family_members.end()); ++iter) {
							if (((*iter)->start > vc.bam_walker->getEndPosition()) or ((*iter)->end < vc.bam_walker->getStartPosition())) {continue;}
							if ((*iter)->is_reverse_strand) {
								alignment_map[1000000 + floor(((*iter)->end - vc.bam_walker->getEndPosition()) / 10.0)].push_back(*iter);
							} else {
								alignment_map[floor(((*iter)->start - vc.bam_walker->getStartPosition()) / 10.0)].push_back(*iter);
							}
						}
						for (map<int, vector<Alignment*> >::iterator iter = alignment_map.begin(); (iter != alignment_map.end()); ++iter) {
							Alignment* alignment = new Alignment;
							const ion::FlowOrder & flow_order = vc.global_context->flow_order_vector.at(iter->second[0]->flow_order_index);
							string flow_order_str;
							flow_order_str.reserve(flow_order.num_flows());
							for (int i = 0; (i < flow_order.num_flows()); ++i) {
								flow_order_str += flow_order.nuc_at(i);
							}
							bool success = true;
							success = consensus.CalculateConsensus(*vc.ref_reader, iter->second, *alignment, flow_order_str);
							if ((not success) or (not vc.candidate_generator->BasicFilters(*alignment))) {
								delete alignment;
							} else {
								vc.targets_manager->TrimAmpliseqPrimers(alignment, vc.bam_walker->GetRecentUnmergedTarget());
								if (alignment->filtered) {delete alignment;}
								else {
									if (consensus_position_ticket->begin == NULL) {consensus_position_ticket->begin = alignment;}
									if (consensus_position_ticket->end != NULL) {consensus_position_ticket->end->next = alignment;}
									consensus_position_ticket->end = alignment;
								}
							}
						}
					}
				}
				if (filter_all_reads_after_done){
					for (vector<Alignment*>::iterator read_it = fam_it->all_family_members.begin(); read_it != fam_it->all_family_members.end(); ++read_it)
						(*read_it)->filtered = true;
				}
			}
		}
	}
	consensus_position_ticket->end = NULL;
}
*/


// Note that consensus_position_ticket must be generated from all samples!
// I.e., if one sample produces a candidate, then the candidate will be applied to all other samples as well.
void GenerateConsensusPositionTicket(vector< vector< vector<MolecularFamily> > > &my_molecular_families_multisample,
                                     VariantCallerContext &vc,
                                     Consensus &consensus,
                                     list<PositionInProgress>::iterator &consensus_position_ticket,
									 bool filter_all_reads_after_done)
{
	unsigned int min_family_size = (unsigned int) vc.parameters->tag_trimmer_parameters.min_family_size;
	for (vector< vector< vector< MolecularFamily> > >::iterator sample_it = my_molecular_families_multisample.begin(); sample_it != my_molecular_families_multisample.end(); ++sample_it) {
		for (vector< vector< MolecularFamily> >::iterator strand_it = sample_it->begin(); strand_it != sample_it->end(); ++strand_it) {
			for (vector< MolecularFamily>::iterator fam_it = strand_it->begin(); fam_it !=  strand_it->end(); ++fam_it) {
				// Is *fam_it functional?
				if (fam_it->SetFuncFromAll(min_family_size)) {
					Alignment* alignment = new Alignment;
					bool success = consensus.CalculateConsensus(*vc.ref_reader, fam_it->all_family_members, *alignment);
					if ((not success) or (not vc.candidate_generator->BasicFilters(*alignment))) {
						delete alignment;
					}
					else {
						vc.targets_manager->TrimAmpliseqPrimers(alignment, vc.bam_walker->GetRecentUnmergedTarget());
						if (alignment->filtered) {delete alignment;}
						else {
							if (consensus_position_ticket->begin == NULL) {consensus_position_ticket->begin = alignment;}
							if (consensus_position_ticket->end != NULL) {consensus_position_ticket->end->next = alignment;}
							consensus_position_ticket->end = alignment;
						}
					}
					if (filter_all_reads_after_done){
						for (vector<Alignment*>::iterator read_it = fam_it->all_family_members.begin(); read_it != fam_it->all_family_members.end(); ++read_it)
							(*read_it)->filtered = true;
					}
				}
			}
		}
	}
	consensus_position_ticket->end = NULL;
}
