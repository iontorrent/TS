/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

// Compare the basecalls for the same named reads in two different alignment files.
//
// Consider a chip with two different basecalling methods.  The reads have the same names (each read mapping to a well),
// but the read sequences may differ and their genomic alignment may differ.
//
// This program takes 2 SAM files, both sorted by read (query) name.  It iterates through the set of reads,
// comparing the pairs (A and B) of query sequences and the alignments for each.
//
// Tallies are written to stdout as follow
//
// errors.txt:
//   	Pos	Nuc	HPLen	HPLenA	HPLenB	Errs	Count
//	20	T	2	2	3	1	513
//
// Pos - position in A
// Nuc - A, C, G, T
// HPLen -- length of homopolymer in genomic (truth)
// HPLenA - length of homopolymer in read A
// HPLenB - length of homopolymer in read B
// Errs - number of A errors in current segment (1..50,51..100,etc.)
// Count - the number of instances of this tuple

#include <cassert>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include "BAMReader.h"

using namespace std;

typedef unsigned int uint;

// quality (as number of errors - indels and mismatches) is reported in segments of SEGMENT_SIZE
const uint SEGMENT_SIZE = 50;

// Quality to consider.  This implies an alignment length.
// Since the analysis is producing errors as a function of position,
// choose a low quality so we have a reasonably long alignment.
const uint MIN_QUAL = 7;

// alignment should be at least MIN_LEN at MIN_QUAL
const uint MIN_LEN = 50;

// maximum number of indels and mismatches in a segment.  indels > MAX_ERRS are coalesced.
const uint MAX_ERRS = 10;

// maximum length of an alignment (max length of a read plus gaps)
const uint MAX_ALIGN_LEN = 400;

// maximum length of an HP (larger are coalesced)
const uint MAX_HP_LEN = 30;

const char tab = '\t';

// array storing tallies
uint tallies[MAX_ERRS][4][MAX_ALIGN_LEN][MAX_HP_LEN][MAX_HP_LEN][MAX_HP_LEN];


string extract_dna(const string& dna) {
  string retdna;
  for (uint i=0; i < dna.size(); i++) {
    if (dna[i]=='A' || dna[i]=='C' || dna[i]=='G' || dna[i]=='T')
      retdna += dna[i];
  }
  return retdna;
}

// return the next HP, including gaps on both sides
void get_HP(const string& dna, uint& start, uint& end, uint& len, char& nuc) {
  nuc=dna[start];
  while (start>0 && dna[start-1]=='-')
    start--;

  end = start;
  len = 0;
  while (end < dna.size() && (dna[end] == nuc || dna[end] == '-')) {
    if (dna[end] != '-') { 
      if (nuc=='-')
	nuc=dna[end];
      len++;
    }
    end++;
  }
}

// return length of the specified nuc in the given range.
// It's possible, but in the noise, for there to be 2 HPs of the same nuc within an alignment region
uint get_HP_len(char nuc, const string& read, uint start, uint end) {

  uint len = 0;
  for (uint i=start; i < end; i++) {
    if (read[i] == nuc)
      len++;
  }
  return len;
}

uint toInt(char nuc) {
  switch(nuc) {
  case 'A': return 0;
  case 'C': return 1;
  case 'G': return 2;
  default: return 3;
  }
}

// create a vector of the number of errors in segments of SEGMENT_SIZE.
// returns => [ 0, 1, 1, 2 ]   use pos / SEGMENT_SIZE to dereference.
void countErrs(BAMUtils& utils, vector<int>& errs, uint start_col) {
  const string tdna = utils.get_tdna();
  const string qdna = utils.get_qdna();

  errs.assign(tdna.size() / SEGMENT_SIZE + 1, 0);

  for (uint i=start_col; i < tdna.size(); i+=SEGMENT_SIZE) {
    
    int errCount = 0;
    uint last_col = tdna.size()<i+SEGMENT_SIZE?tdna.size():i+SEGMENT_SIZE;
    for (uint j=i; j < last_col; j++)
      if (tdna[j] != qdna[j]) {
	// skip successive indels.  Count as single indel event.
	if (tdna[j]=='-' && j>0 && tdna[j-1]=='-')
	  continue;
	if (qdna[j]=='-' && j>0 && qdna[j-1]=='-')
	  continue;
	errCount++;
      }

    errs[i/SEGMENT_SIZE]=errCount;
  }
}

// count the errors along two alignment pairs and store according to position, nucleotide, HP len, and quality
void tally(BAMUtils& utils1, BAMUtils& utils2) {

  string dna1 = utils1.get_tdna();
  string dna2 = utils2.get_tdna();

  uint hp_start1 = 0, hp_start2 = 0;	// first column in alignment

  // count the number of errors (mismatches and indels) in, arbitrarily, read1.
  vector<int> errs;
  while (dna1[hp_start1]=='-') hp_start1++;
  countErrs(utils1,errs,hp_start1);
  while (dna2[hp_start2]=='-') hp_start2++;
  //hp_start1=0;

  while (hp_start1 < dna1.size() && hp_start2 < dna2.size()) {

    // iterate through genomic HPs in alignment
    uint hp_end1 = hp_start1;
    uint hp_end2 = hp_start2;
    uint hp_len1 = 1;
    uint hp_len2 = 1;
    char t_nuc1, t_nuc2;

    // given the HP beginning at hp_start1, expand to columns up and down stream to include
    // all columns of gaps or of the nucleotide at hp_start1.
    // sets hp_start1 (may go down), hp_end1, hp_len1, and t_nuc1.
    get_HP(dna1, hp_start1, hp_end1, hp_len1, t_nuc1);
    get_HP(dna2, hp_start2, hp_end2, hp_len2, t_nuc2);
    if (hp_end1 < dna1.size() && hp_end2 < dna2.size()) {
      
      if (hp_len1 != hp_len2) {
	cerr << hp_start1 << tab << hp_end1 << endl;
	cerr << hp_start2 << tab << hp_end2 << endl;
	cerr << utils1.to_string() << endl << utils2.to_string() << endl;
      }
      assert(hp_len1 == hp_len2);
      assert(t_nuc1==t_nuc2);

      // All I care about is the relationship of r1 and r2 to the single genomic HP.
      // I'm ignoring other rare complications, e.g. alignment differences like this are ignored.
      //
      // r1  C-A   r2 A-G
      // dna -T-  dna -T-
      //
      // Those other HPs in r1 and r2 are ignored.  If I had a reliable mapping of
      // flow order then it would be easy, but I'm just using the alignments given by BAMUtils
      // as a black box where that mapping is lost.  (Even if I tried to reconstruct the alignments 
      // myself, there's not enough information in the SAM/BAM to make the correct correspondence
      // between flow number at sequence position.)


      // Find the length of the HP in r1 and r2.
      uint r1len = get_HP_len(t_nuc1, utils1.get_qdna(), hp_start1, hp_end1);
      uint r2len = get_HP_len(t_nuc2, utils2.get_qdna(), hp_start2, hp_end2);

      // truncate large values
      uint err = errs[hp_start1/SEGMENT_SIZE];
      if (err >= MAX_ERRS) err = MAX_ERRS-1;
      if (hp_start1 >= MAX_ALIGN_LEN) hp_start1 = MAX_ALIGN_LEN-1;
      if (hp_len1 >= MAX_HP_LEN) hp_len1 = MAX_HP_LEN-1;
      if (r1len >= MAX_HP_LEN) r1len = MAX_HP_LEN-1;
      if (r2len >= MAX_HP_LEN) r2len = MAX_HP_LEN-1;

      // store tally of tuple
      tallies[err][toInt(t_nuc1)][hp_start1][hp_len1][r1len][r2len]++;

    }
    hp_start1 = hp_end1;
    hp_start2 = hp_end2;
  }
}

string spaces(int num) {
  string s;
  s.assign(num,' ');
  return s;
}

int strnum_cmp(const char *a, const char *b)
{
	char *pa, *pb;
	pa = (char*)a; pb = (char*)b;
	while (*pa && *pb) {
		if (isdigit(*pa) && isdigit(*pb)) {
			long ai, bi;
			ai = strtol(pa, &pa, 10);
			bi = strtol(pb, &pb, 10);
			if (ai != bi) return ai<bi? -1 : ai>bi? 1 : 0;
		} else {
			if (*pa != *pb) break;
			++pa; ++pb;
		}
	}
	if (*pa == *pb)
		return (pa-a) < (pb-b)? -1 : (pa-a) > (pb-b)? 1 : 0;
	return *pa<*pb? -1 : *pa>*pb? 1 : 0;
}

int main(int argc, char* argv[])
{
  // initializes tally array to 0
  bzero(tallies, MAX_ERRS*4*MAX_ALIGN_LEN*MAX_HP_LEN*MAX_HP_LEN*MAX_HP_LEN);

  // Open two BAM files
  char* bam1 = argv[1];
  char* bam2 = argv[2];

  BAMReader reader1(bam1);
  reader1.open();
  assert(reader1);
  BAMReader::iterator r1_iter=reader1.get_iterator();

  BAMReader reader2(bam2);
  reader2.open();
  assert(reader2);
  BAMReader::iterator r2_iter=reader2.get_iterator();

  // work through parallel sorted lists of reads
  int read_count=0;
  for(; r1_iter.good(); r1_iter.next()) {

    if (!r2_iter.good())
      continue;

    BAMRead r1 = r1_iter.get();
    BAMRead r2 = r2_iter.get();

    if (strnum_cmp(r1.get_qname(), r2.get_qname()) == -1)
      continue;  // r1_iter is behind.  advance r1_iter

    // r2_iter is behind.  advance r2_iter.
    while (strnum_cmp(r2.get_qname(),r1.get_qname()) == -1) {
      r2_iter.next();
      if (!r2_iter.good())
	continue;
      r2 = r2_iter.get();
    }

    // if we're still not equal then r1 is missing in r2 or
    // we've run out of entries in r2.  Just move on.
    if (strnum_cmp(r1.get_qname(), r2.get_qname())!=0)
      continue;

    // Now we have a BAMRead for the same well in both files...

    if (r1.get_rname() && r2.get_rname() && strcmp(r1.get_rname(),r2.get_rname())==0) {
      if (r1.mapped_reverse_strand() == r2.mapped_reverse_strand()) {

	// use utils to get the 5'-3' oriented alignment with clipping removed
	BAMUtils utils1(r1), utils2(r2);

	// extract just the nucs (not dashes) to compare
	string dna1 = extract_dna(utils1.get_tdna());
	string dna2 = extract_dna(utils2.get_tdna());

	if (dna1.find(dna2,0)==0 || dna2.find(dna1,0)==0) { // this is the same template -- possibly one longer than the other

	  if (utils1.get_phred_len(MIN_QUAL) > MIN_LEN && utils2.get_phred_len(MIN_QUAL) > MIN_LEN) {
	  /* DEBUG
	    string t1(utils1.get_tdna()), t2(utils2.get_tdna()),
	      q1(utils1.get_qdna()), q2(utils2.get_qdna()),
	      rn(r1.get_rname());
	    char strand = (r1.mapped_reverse_strand()?'-':'+');
	    cerr << rn.substr(1,10) << tab << ' ' << t1 << endl << 
	      qname1 << tab << strand << q1 << endl << 
	      qname1 << tab << strand << q2 << endl << 
	      rn.substr(1,10) << tab << ' ' << t2 << endl << endl;
	  */
	    read_count++;
	    if (read_count % 10000 == 0)
	      cerr << read_count << endl;
	    tally(utils1, utils2);
	  }
	}

      } // both same strand
    } // both mapped to same ref
  } // r1 loop
  cerr << read_count << endl;
  
  // dump non-zero tallies
  cout << "ERRS"<<SEGMENT_SIZE<<"\tNUC\tPOS\tHP_LEN\tR1_LEN\tR2_LEN\tCOUNT\n";
  for (uint err=0; err < MAX_ERRS; err++)
    for (uint nuc=0; nuc < 4; nuc++)
      for (uint pos=0; pos < MAX_ALIGN_LEN; pos++)
	for (uint hp_len=0; hp_len < MAX_HP_LEN; hp_len++)
	  for (uint r1_hp_len=0; r1_hp_len < MAX_HP_LEN; r1_hp_len++)
	    for (uint r2_hp_len=0; r2_hp_len < MAX_HP_LEN; r2_hp_len++) {
	      uint count = tallies[err][nuc][pos][hp_len][r1_hp_len][r2_hp_len];
	      if (count > 0)
		cout << err << tab << nuc << tab << pos << tab << hp_len << tab << 
		  r1_hp_len << tab << r2_hp_len << tab << count << endl;
	    }

} // main

