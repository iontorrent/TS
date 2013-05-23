/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     SpliceVariantsToReads.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef SPLICEVARIANTSTOREADS_H
#define SPLICEVARIANTSTOREADS_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>


#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <levmar.h>
#include <Variant.h>

#include "MiscUtil.h"
#include "ClassifyVariant.h"
#include "ExtendedReadInfo.h"


using namespace std;


class TinyAlignCell {
  public:
    int start_window, end_window;
    string *inAlleleString;
    string *delAlleleString;
    vector<string *> *altAllelesFound;
    bool inAlleleFound, delAlleleFound;
    int DEBUG;

    TinyAlignCell() {
      inAlleleString=NULL;
      delAlleleString = NULL;
      altAllelesFound = NULL;
      inAlleleFound = false;
      delAlleleFound = false;
      DEBUG = 0;
      start_window = 0;
      end_window = 0;
    };

    ~TinyAlignCell();

    void Init(int _DEBUG) {
      altAllelesFound = new vector<string*>();
      inAlleleFound = false;
      delAlleleFound = false;
      inAlleleString = new string();
      delAlleleString= new string();
      DEBUG=_DEBUG;
    };

    void SetWindow(int _start_window, int _end_window) {
      start_window = _start_window;
      end_window = _end_window;
    }

    void MatchBase(
      char base_seq, char base_ref,
      string &refSequence, int &refSeqAtIndel,
      int &pos_ref, int &pos_seq);

    void InsertBase(
      char base_seq, char base_ref,
      string &refSequence, int &refSeqAtIndel,
      int &pos_ref, int &pos_seq);

    void DeleteBase(
      char base_seq, char base_ref,
      string &refSequence, int &refSeqAtIndel,
      int &pos_ref, int &pos_seq);

    bool CheckValid(bool isInsertion, bool isDeletion, string &varAllele, string &refAllele);

    void DoAlignmentForSplicing(string &ref_aln, string &seq_aln,
                                int &pos_ref, int &pos_seq,
                                int &finalIndelPos,
                                const string &local_contig_sequence, int start_pos,
                                int hotPos, int refSeqAtIndel,
                                string &refSequence) ;
};

int ConstructRefVarSeq(int DEBUG, string readBase, const string &local_contig_sequence, bool strand,
                       bool isSNP, bool isMNV, bool isInsertion, bool isDeletion, int inDelSize,
                       string ref_aln, string seq_aln,
                       string refAllele, string varAllele,
                       int start_pos, int startSC, int endSC,
                       int hotPos, int start_window, int end_window,
                       string &refSequence, string &varSequence, string &readSequence);

bool SpliceMeNow(bool check_valid, string readBase, bool isSNP, bool isMNV,
                 bool isInsertion, bool isDeletion, bool isHpIndel, int inDelLength,
                 int pos_seq, int finalIndelPos, int startSC, int endSC,
                 string &refAllele, string &varAllele, string &refSequence, string &varSequence,
                 string &readSequence, int DEBUG);

void LoadTripleHypotheses(vector<string> &hypotheses, string &firstSeq, string &secondSeq, string &thirdSeq, int strand, int DEBUG);
void LoadOneHypothesis(vector<string> &hypotheses, string &target, int strand);
int HypothesisSpliceSNP(vector<string> &hypotheses, string readBase, const string &local_contig_sequence, bool strand, bool isInsertion, bool isDeletion, int inDelLength, string ref_aln, string seq_aln,
                        string refAllele, string varAllele, int start_pos, int startSC, int endSC, int hotPos, int start_window, int end_window, int DEBUG);
int HypothesisSpliceNonHP(vector<string> &hypotheses, string readBase, const string &local_contig_sequence, bool strand, bool isInsertion, bool isDeletion, int inDelLength, string ref_aln, string seq_aln,
                          string refAllele, string varAllele, int start_pos, int startSC, int endSC, int hotPos, int start_window, int end_window, int DEBUG);

int HypothesisSpliceVariant(vector<string> &hypotheses, ExtendedReadInfo &current_read, AlleleIdentity &variant_identity,
                            string refAllele, const string &local_contig_sequence, int DEBUG);

#endif //SPLICEVARIANTSTOREADS_H
