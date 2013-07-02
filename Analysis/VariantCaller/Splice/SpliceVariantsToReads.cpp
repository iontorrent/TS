/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     SpliceVariantsToReads.cpp
//! @ingroup  VariantCaller
//! @brief    Functions in this file are all retired from ensemble evaluator as of May 24 2013

#include "SpliceVariantsToReads.h"

// xxx Match
void TinyAlignCell::MatchBase(
  char base_seq, char base_ref,
  string &refSequence, int &refSeqAtIndel,
  int &pos_ref, int &pos_seq) {
  // end of window is exclusive in AlleleIdentity::CalculateWindowForVariant
  if (pos_ref < start_window || pos_ref >= end_window) {
    refSequence += base_seq;
    refSeqAtIndel++;
  } else {
    refSequence += base_ref;
    refSeqAtIndel++;
    // Deleting inAllele and delAllele and pushing the content onto altAlleles...
    if (inAlleleFound) {
      inAlleleFound = false;
      if (inAlleleString != NULL && inAlleleString->length() > 0) {
        altAllelesFound->push_back(inAlleleString);
        inAlleleString = NULL;
      }
    }
    if (delAlleleFound) {
      delAlleleFound = false;
      if (delAlleleString != NULL && delAlleleString->length() > 0) {
        altAllelesFound->push_back(delAlleleString);
        if (this->DEBUG > 1)
          cout << "Found match " << base_ref << " " << *delAlleleString << endl;
        delAlleleString = NULL;
      }
    }
  }

  pos_ref++;
  pos_seq++;
}

void TinyAlignCell::InsertBase(
  char base_seq, char base_ref,
  string &refSequence, int &refSeqAtIndel,
  int &pos_ref, int &pos_seq) {
  if (pos_ref < start_window || pos_ref >= end_window) {
    refSequence += base_seq;
    refSeqAtIndel++;
  } else {
    if (inAlleleFound && inAlleleString != NULL) {
      (*inAlleleString) += base_seq;
    } else {
      inAlleleFound = true;
      if (inAlleleString != NULL && inAlleleString->length() > 0) {
        altAllelesFound->push_back(inAlleleString);
        if (this->DEBUG > 1)
          cout << "Found Insertion allele " << *inAlleleString << endl;

      }
      inAlleleString = new string(); // xxx memory leak
      (*inAlleleString) += base_seq;
    }
  }
  //if there is an insertion of base in the read just skip it
  //else
  // referenceSequence += base_ref;


  pos_seq++;
}

void TinyAlignCell::DeleteBase(
  char base_seq, char base_ref,
  string &refSequence, int &refSeqAtIndel,

  int &pos_ref, int &pos_seq) {
  //if there is deletion just skip the base
  if (pos_ref < start_window || pos_ref >= end_window) {
    //referenceSequence += base_seq;
  } else {
    refSequence += base_ref;
    refSeqAtIndel++;
    if (delAlleleFound && delAlleleString != NULL) {
      (*delAlleleString) += base_ref;
    } else {
      if (this->DEBUG >1)
        cout << "Base_seq  = " << base_seq << " " << delAlleleFound << endl;
      delAlleleFound = true;
      if (delAlleleString != NULL && delAlleleString->length() > 0) {
        altAllelesFound->push_back(delAlleleString);
        if (this->DEBUG >1)
          cout << "Found deletion allele " <<  *delAlleleString << endl;

      }
      delAlleleString = new string(); // xxx memory leak
      (*delAlleleString) += base_ref;
    }
  }

  pos_ref++;
}

// This function returns 'false' if the variant allele is found within the window by the tiny alignment,
// also when it is found at a position not specified in the vcf. ???
bool TinyAlignCell::CheckValid(bool isInsertion, bool isDeletion, string &varAllele, string &refAllele) {
  bool constructVarSeq = true;
  for (unsigned int i = 0; i < altAllelesFound->size(); i++) {
    if (this->DEBUG >1)
      cout << "InDel allele : " << *(altAllelesFound->at(i)) << endl;
    if (isInsertion) {
      if (altAllelesFound->at(i)->compare(varAllele.substr(1)) == 0)
        constructVarSeq = false;
    } else
      if (isDeletion) {
        if (altAllelesFound->at(i)->compare(refAllele.substr(1)) == 0)
          constructVarSeq = false;
      }
  }
  return(constructVarSeq);
}

// Destructor
TinyAlignCell::~TinyAlignCell() {
  if (delAlleleString!=NULL)
    delete delAlleleString;
  if (inAlleleString!=NULL)
    delete inAlleleString;
  if (altAllelesFound!=NULL) {
    for (unsigned int i=0; i<altAllelesFound->size(); i++)
      if (altAllelesFound->at(i)!=NULL)
        delete(string*)altAllelesFound->at(i);
    delete altAllelesFound;
  }
}


// What am I doing? xxx
// pos_ref = start_pos
// ref_aln: padded ref alignment
// ref_aln: padded query alignment
// pos_ref: pos_ref = start_pos
// pos_seq: pos_seq = startSC
// finalIndelPos = 0
// local_contig_sequence: reference sequence
// start_pos: vcf position
// hotPos: variant_identity.modified_start_pos
// refSeqAtIndel = 0 -> index counter
// refSequence: empty string
void TinyAlignCell::DoAlignmentForSplicing(string &ref_aln, string &seq_aln,
    int &pos_ref, int &pos_seq,
    int &finalIndelPos,
    const string &local_contig_sequence, int start_pos,
    int hotPos, int refSeqAtIndel,
    string &refSequence) {

  // Iterating over whole alignment sequence
  for (unsigned int i=0; i< ref_aln.length() && i<seq_aln.length();i++) {
    if (pos_ref>= (int)local_contig_sequence.length()) {
      cout << "WARN: Mapping found outside the chromosome: " << ". Start position: " << start_pos << " Chromosome Length = " << local_contig_sequence.length()   << endl;
      break;
    }

    char base_ref = toupper(ref_aln.at(i));
    char base_seq = toupper(seq_aln.at(i));

    //cout << "PosRef = " << pos_ref << " HotPos = " << hotPos << endl;

    if (pos_ref == hotPos)
      finalIndelPos = refSeqAtIndel;

    if (base_ref != '-' && base_seq !='-') {
// one base each
      MatchBase(
        base_seq, base_ref,
        refSequence, refSeqAtIndel,
        pos_ref,pos_seq);
    } else
      if (base_seq != '-' && base_ref == '-') {
// insertion case
        InsertBase(
          base_seq, base_ref,
          refSequence, refSeqAtIndel,
          pos_ref,pos_seq);
      } else
        if (base_seq == '-' && base_ref != '-') {
          // insertion case
          DeleteBase(
            base_seq, base_ref,
            refSequence, refSeqAtIndel,
            pos_ref,pos_seq);
        }
  }
}

bool SpliceMeNow(bool check_valid,
                 string readBase,
                 bool isSNP,
                 bool isMNV,
                 bool isInsertion,
                 bool isDeletion,
                 bool isHpIndel,
                 int inDelLength,
                 int pos_seq, int finalIndelPos,
                 int startSC, int endSC,
                 string &refAllele, string &varAllele,
                 string &refSequence,
                 string &varSequence,
                 string &readSequence,
                 int DEBUG) {
 //DEBUG=true;

  if (isSNP) {
    varSequence = refSequence;
    varSequence[finalIndelPos] = varAllele.at(0);
    // Ensure proper choice of substring without soft clipped bases
    if ((startSC + pos_seq + endSC) == (int)readBase.length())
      readSequence = readBase.substr(startSC, pos_seq);
    else {
      if (DEBUG>0)
        cout << "Error in splicing read. Query length: " << readBase.length()
             << " startSC: " << startSC << " pos_seq: " << pos_seq << " endSC: " << endSC << endl;
      return false;
    }
    //if (pos_seq < (int)readBase.length()-1)
    //  readSequence = readBase.substr(0, pos_seq);
    //else
    //  readSequence = readBase;
  }
  else if (isMNV) {
    varSequence = refSequence;
    size_t len = varAllele.length();
    for (size_t i = 0; i < len; i++ ){
      varSequence[finalIndelPos + i] = varAllele.at(i);
    }
    if ((startSC + pos_seq + endSC) == (int)readBase.length())
      readSequence = readBase.substr(startSC, pos_seq);
    else {
      if (DEBUG>0)
        cout << "Error in splicing read. Query length: " << readBase.length()
             << " startSC: " << startSC << " pos_seq: " << pos_seq << " endSC: " << endSC << endl;
      return false;
    }
    //if (pos_seq < (int)readBase.length()-1)
    //          readSequence = readBase.substr(0, pos_seq);
    //else
    // readSequence = readBase;
  }
  else { // InDels now...
    //cout << "SpliceMeNow : " << check_valid << " " << finalIndelPos << " " << refSequence.length() << " " << isDeletion << " " << isInsertion << endl;

    if (!check_valid) {
      if ((startSC + pos_seq + endSC) == (int)readBase.length())
        readSequence = readBase.substr(startSC, pos_seq);
      else
        return false;
      varSequence = readSequence;  // CK: ???
    } else {
      // we do this if we have found a "valid allele", i.e. the
      if (isDeletion) {

        if (refSequence.size() < finalIndelPos+refAllele.length()-1) {
          if (DEBUG>0)
            cout << "Error in splicing: Deletion size test failed." << endl;
          return false; // Safety so that we can splice the whole variant
        }

        varSequence += refSequence.substr(0,finalIndelPos);
        //need to splice complex deletions such as REF=CAGAGAGA, VAR=CAGAGAG,CAGAGA,CAGA....
        if (varAllele.size() > 1)
          varSequence += varAllele.substr(1);
        varSequence += refSequence.substr(finalIndelPos+refAllele.length()-1);
        //cout << "final del var seq = " << varSequence << endl;

      } else if (isInsertion) {
          if (refSequence.size() < finalIndelPos+refAllele.size()-1) {
            if (DEBUG>0)
              cout << "Error in splicing: Deletion size test failed." << endl;
            return false;
          }

          //cout << "Ref::" << refSequence << endl;
          varSequence += refSequence.substr(0,finalIndelPos-1);
          //cout << varSequence << endl;
          //Example:CAGAG, VAR=CAGAGAG
          varSequence += varAllele;
          //cout << varSequence << endl;
          if (refAllele.size() > 1)
            varSequence += refSequence.substr(finalIndelPos+refAllele.size()-1);
          else
          varSequence += refSequence.substr(finalIndelPos);
          //cout << varSequence << endl;
        }
      if ((startSC + pos_seq + endSC) == (int)readBase.length())
        readSequence = readBase.substr(startSC, pos_seq);
      else {
        if (DEBUG>0)
          cout << "Error in splicing read. Query length: " << readBase.length()
               << " startSC: " << startSC << " pos_seq: " << pos_seq << " endSC: " << endSC << endl;
        return false;
      }
      //if (pos_seq < (int)readBase.length()-1)
      //  readSequence = readBase.substr(0, pos_seq);
      //else
      //  readSequence = readBase;
    } // end of the "no valid check" else statement

  } // InDel else
  if (DEBUG >1) {
    cout << "RefS: " << refSequence << endl;
    cout << "VarS: " << varSequence << endl;
    cout << "RedS: " << readSequence << endl;
  }

  return true;
}

// -------------------------------------------------------------------------

int ConstructRefVarSeq(int DEBUG, string readBase, const string &local_contig_sequence, bool strand, bool isSNP, bool isMNV,
                       bool isInsertion, bool isDeletion, bool isHpIndel, int inDelSize,
                       string ref_aln, string seq_aln,
                       string refAllele, string varAllele,
                       int start_pos, int startSC, int endSC,
                       int hotPos, int start_window, int end_window,
                       string &refSequence, string &varSequence, string &readSequence) {

  //DEBUG = true;
  if (DEBUG > 1) {
    cout << "Ref : " << ref_aln << endl;
    cout << "Base: " << seq_aln << endl;
    cout << "Start_window = " << start_window << endl;
    cout << "End_window = " << end_window << endl;
    cout << "Reference Seq = " << refSequence << endl; // Should be empty string, why are we printing this?
  }

  TinyAlignCell my_align_cell;
  my_align_cell.Init(DEBUG);
  my_align_cell.SetWindow(start_window, end_window);
  int refSeqAtIndel = 0;
  int finalIndelPos = -1;    // Index of variant start position within generated hypothesis string
  int pos_ref = start_pos;  // reference index
  int pos_seq = 0;          // read sequence index w.r.t aligned bases
  //int pos_seq = startSC; // Old version assumed that softclips had been added to read hypothesis

  // What does this exactly do? xxx
  my_align_cell.DoAlignmentForSplicing(ref_aln, seq_aln, pos_ref, pos_seq, finalIndelPos,
                                       local_contig_sequence, start_pos, hotPos, refSeqAtIndel,
                                       refSequence);

  //bool constructVarSeq = my_align_cell.CheckValid(isInsertion,isDeletion,refAllele,varAllele);
  //check to make sure that the read spans the entire variant allele, if not skip this read
  if (finalIndelPos == -1 || finalIndelPos > (int)refSequence.length() - 2
      || (isMNV && (finalIndelPos + (int)varAllele.length()) > (int)refSequence.length()  )
      || (isDeletion && (finalIndelPos+ (int)refAllele.length()-1) > (int)refSequence.length())
      || (isInsertion && (finalIndelPos+ (int)refAllele.length()-1) > (int)refSequence.length()) ) {
    if (DEBUG >0) {
      cout << "ERROR: final IndelPos = " << finalIndelPos << " Reference Length = " << refSequence.length() << endl;
      cout << "Ref = " << refSequence << endl;
      cout << "RefA= " << ref_aln << endl;
      cout << "BasA= " << seq_aln << endl;
      cout << "Var Pos = " << hotPos << endl;
      cout << "Start   = " << start_pos << endl;
      cout << "IndelType = " << isInsertion << " " << isDeletion << "  " << refAllele << "  " << varAllele << endl;
    }
    return -1;
  }

  //return SpliceMeNow(constructVarSeq, readBase,isSNP, isMNV,
  //            isInsertion, isDeletion, isHpIndel, inDelSize,
  //            pos_seq, finalIndelPos, startSC, endSC, refAllele, varAllele,
  //            refSequence,varSequence,readSequence,DEBUG);
  return SpliceMeNow(true, readBase,isSNP, isMNV,
                isInsertion, isDeletion, isHpIndel, inDelSize,
                pos_seq, finalIndelPos, startSC, endSC, refAllele, varAllele,
                refSequence,varSequence,readSequence,DEBUG);
  //return 0;
}

void LoadOneHypothesis(vector<string> &hypotheses, string &target, int strand) {

  if (!strand) {

    char *ref_targ = new char[target.length()+1]; // xxx
    reverseComplement(target, ref_targ);

    hypotheses.push_back(string((const char*) ref_targ));

    delete[] ref_targ;

  } else {
    hypotheses.push_back(target);
  }
}

void LoadTripleHypotheses(vector<string> &hypotheses, string &firstSeq, string &secondSeq, string &thirdSeq, int strand, int DEBUG) {
  LoadOneHypothesis(hypotheses,firstSeq, strand);
  LoadOneHypothesis(hypotheses,secondSeq, strand);
  LoadOneHypothesis(hypotheses,thirdSeq, strand);
  //DEBUG=true;
  if (DEBUG >1) {
    cout << "hyp. 0 = " << hypotheses[0] << endl;
    cout << "hyp. 1 = " << hypotheses[1] << endl;
    cout << "hyp. 2 = " << hypotheses[2] << endl;
  }
}


int HypothesisSpliceSNP(vector<string> &hypotheses, string readBase, const string &local_contig_sequence, bool strand, bool isInsertion, bool isDeletion, int inDelLength, string ref_aln, string seq_aln,
                        string refAllele, string varAllele, int start_pos, int startSC, int endSC, int hotPos, int start_window, int end_window, int DEBUG) {

  string refSeq;
  string varSeq;
  string readSeq;
  int retValue = 0;

  retValue = ConstructRefVarSeq(DEBUG, readBase,  local_contig_sequence, strand, true, false, false, false, false, inDelLength,  ref_aln,  seq_aln, refAllele,
                                varAllele, start_pos, startSC, endSC, hotPos,  start_window, end_window, refSeq, varSeq, readSeq);

  if (retValue == -1)
    return -1;

  LoadTripleHypotheses(hypotheses, refSeq,varSeq,readSeq, strand, DEBUG);
  return(0);
}


int HypothesisSpliceNonHP(vector<string> &hypotheses, string readBase, const string &local_contig_sequence, bool strand, bool isInsertion, bool isDeletion, int inDelLength, string ref_aln, string seq_aln,
                          string refAllele, string varAllele, int start_pos, int startSC, int endSC, int hotPos, int start_window, int end_window, int DEBUG) {
  string refSeq;
  string varSeq;
  string readSeq;
  int retValue = 0;

  retValue = ConstructRefVarSeq(DEBUG, readBase,  local_contig_sequence, strand, false, false, isInsertion, isDeletion, false, inDelLength,  ref_aln,  seq_aln, refAllele,
                                varAllele, start_pos, startSC, endSC, hotPos,  start_window, end_window, refSeq, varSeq, readSeq);

  if (retValue == -1)
    return retValue;

  LoadTripleHypotheses(hypotheses, refSeq,varSeq,readSeq, strand, DEBUG);
  return(0);
}

// xxx CK: Yet another wrapper inside a wrapper. Splicing operations start here.
int HypothesisSpliceVariant(vector<string> &hypotheses,
                            ExtendedReadInfo &current_read,
                            AlleleIdentity &variant_identity,
                            string refAllele,
                            const string &local_contig_sequence,
                            int DEBUG) {

  string refSeq;
  string varSeq;
  string readSeq;
  int retValue = 0;

  // This function needs to go...
  retValue = ConstructRefVarSeq(DEBUG, current_read.alignment.QueryBases,  local_contig_sequence,
                                current_read.is_forward_strand, variant_identity.status.isSNP,
                                variant_identity.status.isMNV, variant_identity.status.isInsertion,
                                variant_identity.status.isDeletion, variant_identity.status.isHPIndel,
                                variant_identity.inDelLength,  current_read.ref_aln,  current_read.seq_aln,
                                refAllele, variant_identity.altAllele, current_read.start_pos,
                                current_read.startSC, current_read.endSC, variant_identity.modified_start_pos,
                                variant_identity.start_window, variant_identity.end_window,
                                refSeq, varSeq, readSeq);

  if (retValue == -1)
    return -1;

  LoadTripleHypotheses(hypotheses, refSeq,varSeq,readSeq, current_read.is_forward_strand, DEBUG);
  return(0);
}
