/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     Realigner.h
//! @brief    Perform local realignment of read.


#ifndef REALIGNER_H
#define REALIGNER_H

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>

#include "api/BamReader.h"
#include "api/BamWriter.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

//#include "BaseCallerUtils.h"
//#include "SystemMagicDefines.h"

using namespace std;
using namespace BamTools;


// ==================================================================

struct AlignmentCell {

  void initialize(int init_score);

  bool            is_match;
  int             best_score;
  int             best_path_direction;
  vector<int>     scores;
  vector<int>     in_directions;
};

struct MDelement {
  string Type;
  int    Length;
};

struct ClippedAnchors {
  vector<CigarOp>   cigar_left;
  vector<CigarOp>   cigar_right;
  vector<MDelement> md_left;
  vector<MDelement> md_right;
};


void reverseString(string& S);

template <class T>
void reverseVector (vector<T>& myvector);

// ==================================================================

class Realigner {

public:

  //! @brief  Constructor.
  Realigner();
  Realigner(unsigned int reserve_size, unsigned int clipping_size);

  //! @brief  Initialized the flow order.
  //! @param[in]  flow_order     Flow order object, also stores number of flows
  //void SetFlowOrder(const ion::FlowOrder& flow_order);

  //! @brief  Set the scores for the Smith-Waterman aligner
  //! @param[in]  score_vec      integer vector of length 4 containing scores
  bool SetScores(const vector<int> score_vec);

  //! @brief  Computes a Smith-Waterman alignment.
  //! @param[out]  CigarData         Cigar Vector of newly computed alignment
  //! @param[out]  MD_Data           MD Vector of newly computed alignment
  //! @param[out]  start_pos_update  Indicates by how many bases the start position shifted
  bool computeSWalignment(vector<CigarOp>& CigarData, vector<MDelement>& MD_data,
         unsigned int& start_pos_update);

  //! @brief  Creates the Reference from the bases in the read and the cigar / md tag in the BAM
  //! @param[in]  algn_query_bases       read bases as written in input BAM
  //! @param[in]  algn_cigar_data        cigar string as obtained from input BAM
  //! @param[in]  md_tag                 md tag as obtained from input BAM
  //! @param[in]  clip_matches_at_ends   switch indicating whether a substring should be used for realignment
  //! @param[in]  print_bam_alignment    switch print existing alignment in input BAM
  //! @param[out] clip_cigar             cigar vector storing info about clipped bases
  bool CreateRefFromQueryBases(const string& algn_query_bases, const vector<CigarOp>& algn_cigar_data,
         const string& md_tag, const bool clip_matches_at_ends);
	 
  enum CREATE_REF_ERR_CODE
  {
    CR_SUCCESS,
    CR_ERR_RECREATE_REF,
    CR_ERR_CLIP_ANCHOR
  };
  
  CREATE_REF_ERR_CODE GetCreateRefError () const
  {
    return cr_error;
  }
  

  //! @brief  Sets the query / target sequences and the strand in the object
  //! @param[in]  q_seq             query sequence
  //! @param[in]  t_seq             target sequence
  //! @param[in]  isForwardStrand   boolean variable indicating whether the read is from the forward strand
  void SetSequences(const string& q_seq, const string& t_seq, const string& aln_path, const bool isForwardStrand);

  //! @brief  Set strand of read
  void SetStrand(bool isForward) { isForwardStrandRead_ = isForward; };

  //! @brief  Set the ends to apply soft clipping
  void SetClipping(int clipping, bool is_forward_strand);

  //! @brief  Set the one sided alignment bandwidth
  void SetAlignmentBandwidth(int bandwidth) { alignment_bandwidth_ = bandwidth; };

  //! @brief  Set score for match
  void SetMatchScore(int match) { kMatchScore = match; };

  //! @brief  Set score for mismatch
  void SetMisatchScore(int mmatch) { kMismatchScore = mmatch; };

  //! @brief  Set gap open score
  void SetGapOpenScore(int g_open) { kGapOpen = g_open; };

  //! @brief  Set gap Extension Score
  void SetGapExtScore(int g_ext) { kGapExtend = g_ext; };


  //! @brief  Combines the tag data of the newly found alignment with previously clipped bases
  //! @brief[in/out]  cigar_data  in: cigar data of new alignment out: cigar data of new alignment + clipped bases
  //! @brief[in/out]  MD_data     in: MD data of new alignment out: MD data of new alignment + clipped bases
  //! @brief[in]      cigar_data  cigar data of bases clipped by call to "CreateRefFromQueryBases"
  bool addClippedBasesToTags(vector<CigarOp>& cigar_data, vector<MDelement>& MD_data, unsigned int nr_read_bases) const;

  //! @brief  Updates the position of the first matching base in the read.
  //! @brief  [in]  original_cigar       Cigar data from input BAM
  //! @brief  [in]  start_position_shift Shift in the start position as determined by realignment
  //! @brief  [in]  org_position         Start position as reported in input BAM
  int updateReadPosition(const vector<CigarOp>& original_cigar, const int& start_position_shift,
                 const int& org_position) const;

  //! @brief  Converts a MD vector into a string representation
  string GetMDstring(vector<MDelement>& MD_data) const;

  //! @brief  Complement a nucleotide
  char NucComplement (char nuc) const;

  //! @brief  Complement and potentially reverse a string of nucleotides
  string ComplementSequence(string seq, bool reverse) const;

  //! @brief  Print out content of dynamic programming matrix
  void PrintScores() const;

  //! @brief  Indicated whether the left anchor of the alignment has been clipped prior to realignment
  bool LeftAnchorClipped() const { return (clipped_anchors_.md_left.size() > 0); };

  //! @brief  Indicated whether the left anchor of the alignment has been clipped prior to realignment
  bool RightAnchorClipped() const { return (clipped_anchors_.md_right.size() > 0); };

  //! @brief  Returns aligned target sequence
  string pretty_tseq()  const { return pretty_tseq_; };

  //! @brief  Returns aligned query sequence
  string pretty_qseq()  const { return pretty_qseq_; };

  //! @brief  Returns aligned target sequence
  string pretty_aln()  const { return pretty_aln_; };

  bool             verbose_;                //!< Print detailed information about realignment to screen
  bool             debug_;
  bool             invalid_cigar_in_input;  //!< Gets set to true if invalid cigar / md pairs are encountered in the input



protected:

  void InitializeRealigner(unsigned int reserve_size, unsigned int clipping_size);

  //! @brief  Read alignment from filled dynamic programming matrix
  //! @param[in]  t_idx       row index (target seq) of cell to start the readout
  //! @param[in]  q_idx       column index (query seq) of cell to start
  //! @param[out] cigar_data  cigar vector of newly read alignment
  //! @param[out] MD_data     cigar vector of newly read alignment
  void backtrackAlignment(unsigned int t_idx, unsigned int q_idx, vector<CigarOp>& CigarData,
		  vector<MDelement>& MD_data, unsigned int& start_pos_update);

  //! @brief Functions clips the anchors to obtain a shorter substring for realignment.
  bool ClipAnchors(bool perform_clipping);

  //! @brief Resets the anchors if an error in ClipAnchors() occurs.
  void RestoreAnchors();

  //! @brief  Reverses the clipping settings
  void ReverseClipping();

  //! @brief  Computes the boundaries of a tubed alignment around the previously found one
  bool ComputeTubedAlignmentBoundaries();

  //! @brief  Updates or adds another element in a partial cigar vector
  void addCigarElement(int align_type, int last_move,
          CigarOp& current_cigar_element, vector<CigarOp>& CigarData);

  //! @brief  Updates or adds another element in a partial md vector
  void addMDelement(int align_type, bool is_match, int last_move, unsigned int t_idx,
          MDelement& current_MD_element, vector<MDelement>& MD_data);

  //! @brief  Do query and target (complex symbol) nucleotide produce a match?
  bool isMatch(char nuc1, char nuc2);

  //! @brief  Which nucleotides match this complex symbol?
  vector<bool> getNucMatches(char nuc);

  //! @ brief  Spells out the constants representing alignment types
  string PrintAlignType(int align_type);

  //! @ brief - Create a dummy pretty alignment string if none is available
  void create_dummy_pretty(unsigned int target_length, unsigned int query_length);


  bool             start_anywhere_in_ref_;  //!< Allow aligned read to start at any point in the reference
  bool             stop_anywhere_in_ref_;   //!< Allow aligned read to end at any point in the reference
  bool             soft_clip_left_;         //!< Softclip bases on the key adapter end of the read
  bool             soft_clip_right_;        //!< Softclip bases on the bead adapter end of the read
  bool             isForwardStrandRead_;    //!< Direction of the read

  string           q_seq_;                  //!< Query sequence internal representation
  string           t_seq_;                  //!< Target sequence internal representation
  string           pretty_tseq_;            //!< Pretty alignment - padded target sequence
  string           pretty_qseq_;            //!< Pretty alignment - padded query sequence
  string           pretty_aln_;             //!< Pretty alignment - string indicating match operations
  //string           aln_path_;               //!< previously computed alignment of query and target

  //ion::FlowOrder   flow_order_;             //!< Sequence of nucleotide flows
  vector<vector<AlignmentCell> > DP_matrix; //!< Dynamic programming matrix
  unsigned int     alignment_bandwidth_;    //!< Diagonal bandwidth of tubed alignment around previously found one
  vector<unsigned int>   q_limit_minus_;    //!< Lower (inclusive) limit on the query index for each target index
  vector<unsigned int>   q_limit_plus_;     //!< Upper (exclusive) limit on the query index for each target index
  ClippedAnchors         clipped_anchors_;  //!< Stores information of bases the are not realigned

  int              kMatchScore;
  int              kMismatchScore;
  int              kGapOpen;
  int              kGapExtend;
  const static int kNotApplicable = -1000000;
  
  
  CREATE_REF_ERR_CODE cr_error;

  const static int      FROM_MATCH   = 0;   //!< The alignment was extended from a match.
  const static int      FROM_MISM    = 1;   //!< The alignment was extended from a mismatch.
  const static int      FROM_I       = 2;   //!< The alignment was extended from an insertion.
  const static int      FROM_D       = 3;   //!< The alignment was extended from an deletion.
  const static int      FROM_NOWHERE = 4;   //!< No valid incoming alignment move.

  const static char     ALN_DEL      = '-'; //!< A base deletion in the alignment string.
  const static char     ALN_INS      = '+'; //!< A base insertion in the alignment string.
  const static char     ALN_MATCH    = '|'; //!< A matching base in the alignment string.
  const static char     ALN_MISMATCH = ' '; //!< A mismatched base in the alignment string.

};

#endif // REALIGNER_H
