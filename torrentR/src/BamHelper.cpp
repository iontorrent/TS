/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */



#include "BamHelper.h"

using namespace std;


void MyBamGroup::ReadGroup(char *bamFile){

	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		 errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
	} else {
		BamTools::SamHeader samHeader = bamReader.GetHeader();
		for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
			if(itr->HasID()) {
				ID.push_back(itr->ID);
			} else {
				ID.push_back("");
			}
			if(itr->HasFlowOrder()) {
				FlowOrder.push_back(itr->FlowOrder);
			} else {
				FlowOrder.push_back("");
			}
			if(itr->HasKeySequence()) {
				KeySequence.push_back(itr->KeySequence);
			} else {
				KeySequence.push_back("");
			}
			if(itr->HasDescription()) {
				Description.push_back(itr->Description);
			} else {
				Description.push_back("");
			}
			if(itr->HasLibrary()) {
				Library.push_back(itr->Library);
			} else {
				Library.push_back("");
			}
			if(itr->HasPlatformUnit()) {
				PlatformUnit.push_back(itr->PlatformUnit);
			} else {
				PlatformUnit.push_back("");
			}
			if(itr->HasPredictedInsertSize()) {
				PredictedInsertSize.push_back(itr->PredictedInsertSize);
			} else {
				PredictedInsertSize.push_back("");
			}
			if(itr->HasProductionDate()) {
				ProductionDate.push_back(itr->ProductionDate);
			} else {
				ProductionDate.push_back("");
			}
			if(itr->HasProgram()) {
				Program.push_back(itr->Program);
			} else {
				Program.push_back("");
			}
			if(itr->HasSample()) {
				Sample.push_back(itr->Sample);
			} else {
				Sample.push_back("");
			}
			if(itr->HasSequencingCenter()) {
				SequencingCenter.push_back(itr->SequencingCenter);
			} else {
				SequencingCenter.push_back("");
			}
			if(itr->HasSequencingTechnology()) {
				SequencingTechnology.push_back(itr->SequencingTechnology);
			} else {
				SequencingTechnology.push_back("");
			}
		}
		bamReader.Close();
	}

}



void BamHeaderHelper::GetRefID(BamTools::BamReader &bamReader)
{ 
  BamTools::SamHeader samHeader = bamReader.GetHeader();
  for (BamTools::SamSequenceIterator itr = samHeader.Sequences.Begin(); itr != samHeader.Sequences.End(); ++itr) {
    string bamseq = itr->Name;
    bam_sequence_names.push_back(bamseq);
   }
}

int BamHeaderHelper::IdentifyRefID(string &sequenceName){
     int chrIndex = -1;
    for (size_t i = 0; i < bam_sequence_names.size(); i++) {
      if (sequenceName.compare(bam_sequence_names[i]) == 0) {
        chrIndex = i;
        break;
      }
    }
    return(chrIndex);
}

void BamHeaderHelper::GetFlowOrder(BamTools::BamReader &bamReader){
  BamTools::SamHeader samHeader = bamReader.GetHeader();
  if (!samHeader.HasReadGroups()) {
    //bamReader.Close();
    cerr << "ERROR: there is no read group in " << "this file" << endl;
    //exit(1);
  }
    for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {
    if (itr->HasFlowOrder()) {
      flow_order_set.push_back(itr->FlowOrder);
      //flowKey = itr->KeySequence;
    }
  }
}


void OpenMyBam(BamTools::BamReader &bamReader, char *bamFile){

   if (!bamReader.Open(bamFile)) {
    cerr << " ERROR: fail to open bam" << bamFile << endl;
    //exit(-1); throw exception instead
  }
  //find the index
  string bamIndex(bamFile);
  // replace last character to find index
  bamIndex.append(".bai"); // bam->bai
  if (!bamReader.OpenIndex(bamIndex)) {
    cerr << "ERROR: fail to open bam index " << bamIndex << endl;
    // throw exception
  }
}

bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value) {
	char tagType = ' ';
	if(alignment.GetTagType(tag, tagType)) {
		switch(tagType) {
			case BamTools::Constants::BAM_TAG_TYPE_INT8: {
				int8_t value_int8 = 0;
				alignment.GetTag(tag, value_int8);
				value = value_int8;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT8: {
				uint8_t value_uint8 = 0;
				alignment.GetTag(tag, value_uint8);
				value = value_uint8;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_INT16: {
				int16_t value_int16 = 0;
				alignment.GetTag(tag, value_int16);
				value = value_int16;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT16: {
				uint16_t value_uint16 = 0;
				alignment.GetTag(tag, value_uint16);
				value = value_uint16;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_INT32: {
				int32_t value_int32 = 0;
				alignment.GetTag(tag, value_int32);
				value = value_int32;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT32: {
				uint32_t value_uint32 = 0;
				alignment.GetTag(tag, value_uint32);
				value = value_uint32;
			} break;
			default: {
				alignment.GetTag(tag, value);
			} break;
		}
		return(true);
	} else {
		return(false);
	}
}

//Ported from BamUtils

//this could probably be faster -- maybe with an std::transform
void reverse_comp(std::string& c_dna) {
    for (unsigned int i = 0; i<c_dna.length(); i++) {
        switch (c_dna[i]) {
            case 'A':
                c_dna[i] = 'T';
                break;
            case 'T':
                c_dna[i] = 'A';
                break;
            case 'C':
                c_dna[i] = 'G';
                break;
            case 'G':
                c_dna[i] = 'C';
                break;
            case '-':
                c_dna[i] = '-';
                break;

            default:
                break;
        }
    }
    std::reverse(c_dna.begin(), c_dna.end());

}

void dna( string& qDNA, const vector<BamTools::CigarOp>& cig, const string& md, string& tDNA) {

    int position = 0;
    string seq;
    string::const_iterator qDNA_itr = qDNA.begin();

    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i != cig.end(); ++i) {
        if ( i->Type == 'M') {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {

                if (count >= i->Length) {
                    break;
                } else {
                    seq += *qDNA_itr;
                    ++qDNA_itr;
                    ++count;
                }
            }
        } else if ((i->Type == 'I') || (i->Type == 'S')) {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {
                if (count >= i->Length) {
                    break;
                }
                ++qDNA_itr;
                ++count;
            }
            //bool is_error = false;

//            if (i->Type == 'S') {
//                soft_clipped_bases += i->Length;
//                //is_error = true;
//            }
        }
        position++;
    }

    tDNA.reserve(seq.length());
    int start = 0;
    string::const_iterator md_itr = md.begin();
    std::string num;
    int md_len = 0;
    char cur;

    while (md_itr != md.end()) {

        cur = *md_itr;

        if (std::isdigit(cur)) {
            num+=cur;
            //md_itr.next();
        }
        else {
            if (num.length() > 0) {
                md_len = strtol(num.c_str(),NULL, 10);
                num.clear();

                tDNA += seq.substr(start, md_len);
                start += md_len;
            }
        }

        if (cur == '^') {
            //get nuc
            ++md_itr;
            char nuc = *md_itr;
            while (std::isalpha(nuc)) {
                tDNA += nuc;
                ++md_itr;
                nuc = *md_itr;
            }
            num += nuc; //it's a number now will
                        //lose this value if i don't do it here
            //cur = nuc;

        } else if (std::isalpha(cur)) {
            tDNA += cur;
            start++;

        }
        ++md_itr;
    }

    //clean up residual num if there is any
    if (num.length() > 0) {
        md_len = strtol(num.c_str(),NULL, 10);
        num.clear();
        tDNA += seq.substr(start, md_len);
        start += md_len;
    }
}


void padded_alignment(const vector<BamTools::CigarOp>& cig, string& qDNA, string& tDNA,  string& pad_query, string& pad_target, string& pad_match, bool isReversed) {

    int sdna_pos = 0;
    unsigned int tdna_pos = 0;
    pad_target.reserve(tDNA.length());
    pad_query.reserve(tDNA.length());
    pad_match.reserve(tDNA.length());
    string::iterator tdna_itr = tDNA.begin();
    unsigned int tot = 0;

    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i!=cig.end(); ++i) {

        if (i->Type == 'I' ) {
            pad_target.append(i->Length, '-');

            unsigned int count = 0;

            tdna_itr = qDNA.begin();
            advance(tdna_itr, sdna_pos);

            while (tdna_itr != tDNA.end() ) {
                if (count >= i->Length) {
                    break;
                } else {
                    pad_query += *tdna_itr;
                    ++tdna_itr;
                    //++tdna_pos;
                    ++sdna_pos;
                    ++count;
                }
            }
            pad_match.append(i->Length, '+');
        }
        else if(i->Type == 'D' || i->Type == 'N') {
            pad_target.append( tDNA.substr(tdna_pos, i->Length));
            sdna_pos += i->Length;
            tdna_pos += i->Length;
            pad_query.append(i->Length, '-');
            pad_match.append(i->Length, '-');
        }
        else if(i->Type == 'P') {
            pad_target.append(i->Length, '*');
            pad_query.append(i->Length, '*');
            pad_match.append(i->Length, ' ');
        } else if (i->Type == 'S') {

//            if (!truncate_soft_clipped) {

//                    pad_source.append(i->Length, '-');
//                    pad_match.append(i->Length, '+');
//                    pad_target.append(i->Length, '+');

//            }
//            int count = 0;
//            while (tdna_itr != tDNA.end()) {
//                if (count >= i->Length) {
//                    break;
//                }
//                ++tdna_pos;
//                ++tdna_itr;
//                ++count;
//            }
        }

        else if (i->Type == 'H') {
            //nothing for clipped bases
        }else {
            std::string ps, pt, pm;
            ps.reserve(i->Length);
            pm.reserve(i->Length);

            ps = qDNA.substr(sdna_pos,i->Length); //tdna is really qdna

            tdna_itr = tDNA.begin();
            advance(tdna_itr, tdna_pos);

            unsigned int count = 0;

            while (tdna_itr != tDNA.end()) {
                if (count < i->Length) {
                    pt += *tdna_itr;
                } else {
                    break;
                }

                ++tdna_itr;
                ++count;

            }
            for (unsigned int z = 0; z < ps.length(); z++) {
                if (ps[z] == pt[z]) {
                    pad_match += '|';
                } else {
                    pad_match += ' ';
                }
            }//end for loop
            pad_target += pt;
            pad_query += ps;

            sdna_pos += i->Length;
            tdna_pos += i->Length;
            if( tdna_pos >= tDNA.size() )
                break;
        }
        tot++;
    }
    /*
    std::cerr << "pad_source: " << pad_source << std::endl;
    std::cerr << "pad_target: " << pad_target << std::endl;
    std::cerr << "pad_match : " << pad_match << std::endl;
    */
}

std::vector<int> score_alignments(string& pad_source, string& pad_target, string& pad_match ){

    int n_qlen = 0;
    int t_len = 0;
    int t_diff = 0;
    int match_base = 0;
    int num_slop = 0;

    int consecutive_error = 0;

    //using namespace std;
    for (int i = 0; (unsigned int)i < pad_source.length(); i++) {
        //std::cerr << " i: " << i << " n_qlen: " << n_qlen << " t_len: " << t_len << " t_diff: " << t_diff << std::endl;
        if (pad_source[i] != '-') {
            t_len = t_len + 1;
        }

        if (pad_match[i] != '|') {
            t_diff = t_diff + 1;

            if (i > 0 && pad_match[i-1] != '|' && ( ( pad_target[i] == pad_target[i - 1] ) || pad_match[i] == '-' ) ) {
                consecutive_error = consecutive_error + 1;
            } else {
                consecutive_error = 1;
            }
        } else {
            consecutive_error = 0;
            match_base = match_base + 1;
        }
        if (pad_target[i] != '-') {
            n_qlen = n_qlen + 1;
        }
    }


    //get qual vals from  bam_record
    std::vector<double> Q;

    //setting acceptable error rates for each q score, defaults are
    //7,10,17,20,47
    //phred_val == 7
    Q.push_back(0.2);
    //phred_val == 10
    Q.push_back(0.1);
    //phred_val == 17
    Q.push_back(0.02);
    //phred_val == 20
    Q.push_back(0.01);
    //phred_val == 47
    Q.push_back(0.00002);

    std::vector<int> q_len_vec(Q.size(), 0);

    int prev_t_diff = 0;
    int prev_loc_len = 0;
    int i = pad_source.length() - 1;

    for (std::vector<std::string>::size_type k =0; k < Q.size(); k++) {
        int loc_len = n_qlen;
        int loc_err = t_diff;
        if (k > 0) {
            loc_len = prev_loc_len;
            loc_err = prev_t_diff;
        }

        while ((loc_len > 0) && (static_cast<int>(i) >= num_slop) && i > 0) {

            if (q_len_vec[k] == 0 && (((loc_err / static_cast<double>(loc_len))) <= Q[k]) /*&& (equivalent_length(loc_len) != 0)*/) {

                q_len_vec[k] = loc_len;

                prev_t_diff = loc_err;
                prev_loc_len = loc_len;
                break;
            }
            if (pad_match[i] != '|') {
                loc_err--;
            }
            if (pad_target[i] != '-') {

                loc_len--;
            }
            i--;
        }
    }
    return q_len_vec;
}

bool getNextAlignment(BamTools::BamAlignment &alignment, BamTools::BamReader &bamReader, const std::map<std::string, int> &groupID, std::vector< BamTools::BamAlignment > &alignmentSample, std::map<std::string, int> &wellIndex, unsigned int nSample) {
	if(nSample > 0) {
		// We are randomly sampling, so next read should come from the sample that was already taken from the bam file
		if(alignmentSample.size() > 0) {
			alignment = alignmentSample.back();
			alignmentSample.pop_back();
			alignment.BuildCharData();
			return(true);
		} else {
			return(false);
		}
	} else {
		// No random sampling, so we're either returning everything or we're looking for specific read names
		bool storeRead = false;
		while(bamReader.GetNextAlignment(alignment)) {
			if(groupID.size() > 0) {
				std::string thisReadGroupID = "";
				if( !alignment.GetTag("RG", thisReadGroupID) || (groupID.find(thisReadGroupID)==groupID.end()) ) {
					continue;
				}
			}
			storeRead=true;
			if(wellIndex.size() > 0) {
				// We are filtering by position, so check if we should skip or keep the read
				int thisCol,thisRow;
				if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
					std::cerr << "Error parsing read name: " << alignment.Name << "\n";
				std::stringstream wellIdStream;
				wellIdStream << thisCol << ":" << thisRow;
				std::map<std::string, int>::iterator wellIndexIter;
				wellIndexIter = wellIndex.find(wellIdStream.str());
				if(wellIndexIter != wellIndex.end()) {
					// If the read ID matches we should keep, unless its a duplicate
					if(wellIndexIter->second >= 0) {
						storeRead=true;
						wellIndexIter->second=-1;
					} else {
						storeRead=false;
						std::cerr << "WARNING: found extra instance of readID " << wellIdStream.str() << ", keeping only first\n";
					}
				} else {
					// read ID is not one we should keep
					storeRead=false;
				}
			}
			if(storeRead)
				break;
		}
		return(storeRead);
	}
}




std::string getQuickStats(const std::string &bamFile, std::map< std::string, int > &keyLen, unsigned int &nFlowFZ, unsigned int &nFlowZM) {
	std::string errMsg = "";
	BamTools::BamReader bamReader;
	if(!bamReader.Open(bamFile)) {
		errMsg += "Failed to open bam " + bamFile + "\n";
		return(errMsg);
	}
	BamTools::SamHeader samHeader = bamReader.GetHeader();
	for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
		if(itr->HasID())
			keyLen[itr->ID] = itr->HasKeySequence() ? itr->KeySequence.length() : 0;
		if(itr->HasFlowOrder())
			nFlowZM = std::max(nFlowZM,(unsigned int) itr->FlowOrder.length());
	}
	BamTools::BamAlignment alignment;
	std::vector<uint16_t> flowIntFZ;
	while(bamReader.GetNextAlignment(alignment)) {
		if(alignment.GetTag("FZ", flowIntFZ))
			nFlowFZ = flowIntFZ.size();
		break;
	}
	bamReader.Close();
	if(nFlowFZ==0)
		std::cout << "NOTE: bam file has no flow signals in FZ tag: " + bamFile + "\n";
	if(nFlowZM==0)
		std::cout << "NOTE: bam file has no flow signals in ZM tag: " + bamFile + "\n";
	return(errMsg);
}

