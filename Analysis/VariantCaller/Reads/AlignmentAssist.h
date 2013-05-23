/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     AlignmentAssist.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef ALIGNMENTASSIST_H
#define ALIGNMENTASSIST_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <levmar.h>
#include <Variant.h>


#include "BaseCallerUtils.h"

using namespace std;
using namespace BamTools;
using namespace ion;


int retrieve_flowpos(string readBase, const string & local_contig_sequence, bool strand, string ref_aln, string seq_aln, int start_pos, int start_flow,
                     int startSC, int endSC, vector<int> &flowIndex, string &flowOrder, int hotPos, int DEBUG);

void get_alignments(string base_seq, const string & local_contig_sequence, int start_pos, vector<CigarOp> cigar, string & ref_aln, string & seq_aln);
int get_next_event_pos(string cigar, int pos);
void parse_cigar(vector<BamTools::CigarOp> cigar, bool strand, int &, int, int &, int &, int &, int &, int &, int &);
void getHardClipPos(vector<BamTools::CigarOp> cigar, int & startHC, int & startSC, int & endHC, int & endSC);

#endif //ALIGNMENTASSIST_H
