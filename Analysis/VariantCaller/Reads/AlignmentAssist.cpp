/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     AlignmentAssist.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "AlignmentAssist.h"

// Used in MultiFlowDist
int retrieve_flowpos(string readBase, const string &local_contig_sequence, bool strand, string ref_aln, string seq_aln, int start_pos, int start_flow,
                     int startSC, int endSC, vector<int> &flowIndex, string &flowOrder,int hotPos, int DEBUG) {

  bool flowSigPresent = true;

  vector<int> flowIndexRev;

  //if flow present and mapped to negative strand reverse the flows
  if (!strand) {
    vector<int>::reverse_iterator flowIndexRevItr;

    for (flowIndexRevItr = flowIndex.rbegin(); flowIndexRevItr < flowIndex.rend(); flowIndexRevItr++) {
      flowIndexRev.push_back(*flowIndexRevItr);

    }
  }

  int pos_ref = start_pos;
  int pos_seq = 0;
  int flowPos = 0;
  int flowIndexPos = 0;


  if (strand)
    flowPos =  startSC + 1;//1-based flow positions
  else
    flowPos = 1 + startSC; //1-based flow positions

  //cout << "Read seq length = " << readBase.length() << " flowIndex length = " << flowIndex->size() << "start flow = " << start_flow << "startHC = " << startHC+startSC << endl;
  //unsigned int refLength = ref_aln.length();
  char readbase;
  char base_ref;
  char base_seq;
  for (unsigned int i=0; i< ref_aln.length() && i<seq_aln.length();i++) {
    if (pos_ref>=(int)local_contig_sequence.length()) {
      cout << "WARN: Mapping found outside the chromosome: " << ". Start position: " << start_pos << " Chromosome Length = " << local_contig_sequence.length()  << endl;
      break;
    }
    base_ref = tolower(ref_aln.at(i));
    base_seq = tolower(seq_aln.at(i));
    readbase = tolower(readBase.at(pos_seq+startSC)); //base seq has SoftClips present in them.


    if (flowSigPresent && strand) {
      if (flowPos < 1 || flowPos > (int)flowIndex.size()) {
        cout << "Fatal ERR: Position out of range of vector FlowSigValue: flowPos = " << flowPos << " vector size = " << flowIndex.size() << endl;
        exit(-1);
      }

      flowIndexPos = flowIndex.at(flowPos-1);


    } else
      if (flowSigPresent && !strand) {
        if (flowPos < 1 || flowPos > (int)flowIndexRev.size()) {
          cout << "Fatal ERR: Position out of range of vector FlowSigRev: flowPos = " << flowPos << " vector size = " << flowIndexRev.size() << endl;
          exit(-1);
        }

        flowIndexPos = flowIndexRev.at(flowPos-1);
      }
    //if (DEBUG)
    //  cout << " Base pos = " << i << " base = " << base_seq << " flowPos = " << flowPos << " flowIndex = " << flowIndexPos << "Flow = " << flowOrder[flowIndexPos] << " ref pos = " << pos_ref << " base pos = " << pos_seq << endl;


    if (pos_ref == hotPos) { //exact position of the indel
      if (base_seq == base_ref)
        return flowIndexPos;
      else
        if (base_seq == '-' && base_ref == readbase)
          return flowIndexPos;
        else
          if (base_ref == '-' && base_seq != '-') {
            return flowIndexPos; //assuming that the next base in Ref will match the base, not true in case of single base insertions
          } else
            if (base_seq != base_ref) {
              while (readbase != base_ref && (pos_seq+startSC < (int)readBase.length()-1)) {
                pos_seq++;
                readbase = tolower(readBase.at(pos_seq+startSC)); //NOTE: Check this logic more carefully
                flowPos++;
              }
              if (strand)
                flowIndexPos = flowIndex.at(flowPos-1);
              else
                flowIndexPos = flowIndexRev.at(flowPos-1);

              return flowIndexPos;
            }

    } else {
      if (base_ref != '-' && base_seq !='-') {

        pos_ref++;
        pos_seq++;
        flowPos++;
      } else
        if (base_seq != '-' && base_ref == '-') {


          pos_seq++;
          flowPos++;
        } else
          if (base_seq == '-' && base_ref != '-') {

            pos_ref++;
            //continue;
          }
    }


  }
  return flowIndexPos;

}

// Was only used in ExtendedReadInfo; sets object properties ref_aln and seq_aln.
// Replaced with ExtendedReadInfo::UnpackAlignmentInfo
void get_alignments(string base_seq, const string &local_contig_sequence, int start_pos, vector<BamTools::CigarOp> cigar, string & ref_aln, string & seq_aln) {
  unsigned int ref_pos = start_pos;
  unsigned int seq_pos = 0;
  unsigned int cigar_pos = 0;
  ref_aln.assign("");
  seq_aln.assign("");
  BamTools::CigarOp cigarop;
  char event;
  int l;
  while (seq_pos < base_seq.length() && ref_pos < local_contig_sequence.length() && cigar_pos < cigar.size()) {
    //cout << "cigar_pos " << cigar_pos << endl;
    //int event_pos = get_next_event_pos(cigar,cigar_pos);
    cigarop = cigar.at(cigar_pos);
    event = cigarop.Type;
    l = cigarop.Length;

    if (event == 'M') {
      seq_aln.append(base_seq,seq_pos,l);
      ref_aln.append(local_contig_sequence,ref_pos,l);
      seq_pos +=l;
      ref_pos += l;
    } else
      if (event == 'D' || event == 'N') {
        seq_aln.append(l,'-');
        ref_aln.append(local_contig_sequence,ref_pos,l);
        ref_pos += l;
      } else
        if (event == 'I') {
          ref_aln.append(l,'-');
          seq_aln.append(base_seq,seq_pos,l);
          seq_pos +=l;
        } else
          if (event == 'S') {
            seq_pos += l;
          }

    cigar_pos++;


  }


}

// Only used in ExtendedReadInfo; sets object properties startHC, endHC, startSC, endSC.
// Replaced with ExtendedReadInfo::UnpackAlignmentInfo
void getHardClipPos(vector<BamTools::CigarOp> cigar, int & startHC, int & startSC, int & endHC, int & endSC) {
  bool matchFound = false;
  unsigned int cigar_pos = 0;
  CigarOp cigarop;
  while (cigar_pos < cigar.size()) {
    cigarop = cigar.at(cigar_pos);
    char event = cigarop.Type;
    int l = cigarop.Length;
    if (event == 'M')
      matchFound = true;
    else
      if (event == 'H') {
        if (matchFound)
          endHC = l;
        else
          startHC = l;
      } else
        if (event == 'S') {
          if (matchFound)
            endSC = l;
          else
            startSC = l;
        }
    cigar_pos++;

  }
}

// Seems to not be called anywhere...
void parse_cigar(vector<BamTools::CigarOp> cigar, bool strand, int & offset,  int seqLen, int & softCPos, int & totIns, int & totDel, int & totMatch, int & maxMatch, int & totSC) {

  unsigned int cigar_pos = 0;
  int matchLength = 0;
  int totLength = 0;
  int numInsertions = 0;
  int numDeletions = 0;
  int maxMatchLen = 0;
  int totMatchLen = 0;
  CigarOp cigarop;
  while (cigar_pos < cigar.size()) {
    cigarop = cigar.at(cigar_pos);
    char event = cigarop.Type;
    int l = cigarop.Length;
    //cout << " event = " << event << endl;
    //cout << " length = " << l << endl;
    if ((event == 'H' || event == 'S') && cigar_pos == 0) {
      offset = l;
      if (event == 'S') {
        softCPos = l;
        totSC += l;
        totLength += l;
      }
    } else
      if (event == 'M') {
        matchLength = l;
        if (matchLength > maxMatchLen)
          maxMatchLen = matchLength;
        totMatchLen += l;
        totLength += l;
      } else
        if (event == 'H' || event == 'S') {
          if (event == 'S') {
            softCPos = l;
            totSC += l;
            totLength += l;
          }
        } else
          if (event == 'I') {
            numInsertions++;
            //cout << "numInsertions = " << numInsertions << endl;

            totLength += l;
          } else
            if (event == 'D') {
              numDeletions++;

            } else
              if (event == 'N') {

              }

    cigar_pos++;

  } //end while loop

  if (totLength != seqLen) {
    //cout << "Warning total CIGAR length is not matching SEQ length " << cigar << endl;
  }



  //return offset;
}
