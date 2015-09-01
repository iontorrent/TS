/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BAMWalkerEngine.h
//! @ingroup  VariantCaller
//! @brief    Streamed BAM reader


#ifndef BAMWALKERENGINE_H
#define BAMWALKERENGINE_H

#include <list>
#include <map>
#include <vector>
#include <string>
#include "api/BamMultiReader.h"
#include "api/BamWriter.h"
#include "TargetsManager.h"

using namespace std;
using namespace BamTools;



enum AlleleType {
  ALLELE_UNKNOWN = 1,
  ALLELE_REFERENCE = 2,
  ALLELE_MNP = 4,
  ALLELE_SNP = 8,
  ALLELE_INSERTION = 16,
  ALLELE_DELETION = 32,
  ALLELE_COMPLEX = 64,
  ALLELE_NULL = 128
};

struct Allele {
  Allele() : type(ALLELE_UNKNOWN), position(0), ref_length(0), alt_length(0), alt_sequence(NULL) {}
  Allele(AlleleType t, long int pos, unsigned int ref_len, unsigned int alt_len, const char *seq_ptr)
      : type(t), position(pos), ref_length(ref_len), alt_length(alt_len), alt_sequence(seq_ptr) {}

  AlleleType      type;             //! type of the allele
  long int        position;         //! 0-based position within current chromosome
  unsigned int    ref_length;       //! allele length relative to the reference
  unsigned int    alt_length;       //! allele length relative to the read
  const char *    alt_sequence;     //! allele sequence within the read, need not be 0-terminated
};


// structure to encapsulate registered reads and alleles
struct Alignment {

  Alignment() { Reset(); }
  void Reset() {
    next = NULL;
    read_number = 0;
    original_position = 0;
    processed = false;
    processing_prev = NULL;
    processing_next = NULL;
    filtered = false;
    start = 0;
    end = 0;
    sample_index = 0;
    primary_sample = false;
    snp_count = 0;
    refmap_start.clear();
    refmap_code.clear();
    refmap_has_allele.clear();
    refmap_allele.clear();
    is_reverse_strand = false;
    evaluator_filtered = false;
    measurements.clear();
    measurements_length = 0;
    phase_params.clear();
    runid.clear();
    well_rowcol.clear();
    read_bases.clear();
    pretty_aln.clear();
    left_sc = 0;
    right_sc = 0;
    start_sc = 0;
    align_start = 0;
    align_end = 0;
    start_flow = 0;
    prefix_flow = -1;
    flow_index.clear();
    flow_order_index = -1;
    read_group.clear();

    worth_saving = false;
  }

  BamAlignment          alignment;          //! Raw BamTools alignment
  Alignment*            next;               //! Singly-linked list for durable alignments iterator
  int                   read_number;        //! Sequential number of this read
  int                   original_position;  //! Alignment position, before primer trimming

  // Processing state
  bool                  processed;          //! Is candidate generator's pre-processing finished?
  Alignment*            processing_prev;    //! Previous in a list of alignments being processed
  Alignment*            processing_next;    //! Next in a list of alignments being processed

  // Candidate generation information
  bool                  filtered;           //! Is unusable for candidate generator?
  long int              start;              //! Start of the first usable allele
  long int              end;                //! End of the last usable allele
  int                   sample_index;       //! Sample associated with this read
  bool                  primary_sample;     //! This sample is being called by evaluator
  int                   snp_count;
  vector<const char*>   refmap_start;
  vector<char>          refmap_code;
  vector<char>          refmap_has_allele;
  vector<Allele>        refmap_allele;

  // Candidate evaluator information
  bool                  is_reverse_strand;  //! Indicates whether read is from the forward or reverse strand
  bool                  evaluator_filtered; //! Is unusable for candidate evaluator?
  vector<float>         measurements;       //! The measurement values for this read blown up to the length of the flow order
  int                   measurements_length;//! Original trimmed length of the ZM measurements vector
  vector<float>         phase_params;       //! cf, ie, droop parameters of this read
  string                runid;              //! Identify the run from which this read came: used to find run-specific parameters
  vector<int>           well_rowcol;        //! 2 element int vector 0-based row, col in that order mapping to row,col in chip
  string                read_bases;         //! Read sequence as base called (minus hard but including soft clips)
  string                pretty_aln;         //! pretty alignment string displaying matches, insertions, deletions
  int                   left_sc;            //! Number of soft clipped bases at the start of the alignment
  int                   right_sc;           //! Number of soft clipped bases at the end of the alignment
  int                   start_sc;           //! Number of soft clipped bases at the read beginning
  int                   align_start;        //! genomic 0-based end position of soft clipped untrimmed read
  int                   align_end;          //! genomic 0-based end position of soft clipped untrimmed read
  int                   start_flow;         //! Flow corresponding to the first base in read_bases
  int                   prefix_flow;        //! Flow corresponding to to the last base of the 5' hard clipped prefix
  vector<int>           flow_index;         //! Main incorporating flow for each base in read_bases
  short                 flow_order_index;   //! Index of the flow order belonging to this read
  string                read_group;         //! Read group of this read

  // Post-processing information
  bool                  worth_saving;
};


struct PositionInProgress {

  int                   chr;                //! Chromosome index of this variant position
  long                  pos;                //! Position within chromosome
  long                  target_end;         //! End of current target region
  Alignment *           begin;              //! First read covering this position
  Alignment *           end;                //! Last read coverint this position
  time_t                start_time;
};


class ReferenceReader;

class BAMWalkerEngine {
public:

  // Initialization
  BAMWalkerEngine();
  ~BAMWalkerEngine();
  void Initialize(const ReferenceReader& ref_reader, TargetsManager& targets_manager,
      const vector<string>& bam_filenames, const string& postprocessed_bami, int px);
  void Close();
  const SamHeader& GetBamHeader() { return bam_header_; }

  // Job dispatch
  bool EligibleForReadRemoval();
  bool EligibleForGreedyRead();
  bool ReadyForNextPosition();

  // Memory contention prevention
  bool MemoryContention();
  bool IsEarlierstPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);

  // Loading new reads
  void RequestReadProcessingTask(Alignment*& new_read);
  bool GetNextAlignmentCore(Alignment* new_read);
  void FinishReadProcessingTask(Alignment* new_read, bool success);

  // Processing genomic position
  void BeginPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);
  bool AdvancePosition(int position_increment, int next_hotspot_chr = -1, long next_hotspot_position = -1);
  void FinishPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);

  // Deleting or saving used up reads
  void RequestReadRemovalTask(Alignment*& removal_list);
  void SaveAlignments(Alignment* removal_list);
  void FinishReadRemovalTask(Alignment* removal_list);

  void PrintStatus();
  int GetRecentUnmergedTarget();

  bool HasMoreAlignments() { return has_more_alignments_; }
  bool ReadProcessingTasksInProgress() { return processing_first_; }

  void GetProgramVersions(string& basecaller_version, string& tmap_version) {
    basecaller_version = basecaller_version_;
    tmap_version = tmap_version_;
  }

private:
  void InitializeBAMs(const ReferenceReader& ref_reader, const vector<string>& bam_filenames);

  TargetsManager *          targets_manager_;       //! Manages targets loaded from BED file
  BamMultiReader            bam_reader_;            //! BamTools mulit-bam reader
  SamHeader                 bam_header_;            //! Bam header
  string                    basecaller_version_;    //! BaseCaller version retrieved from BAM header
  string                    tmap_version_;          //! TMAP version retrieved from BAM header

  MergedTarget *            next_target_;           //! Target containing next position
  long int                  next_position_;         //! Next position (chr in the target)

  int                       last_processed_chr_;    //! Reads up to this chr+pos are guaranteed to be processed
  long int                  last_processed_pos_;    //! Reads up to this chr+pos are guaranteed to be processed
  bool                      has_more_alignments_;   //! Are there still more reads in BAM?
  bool                      has_more_positions_;    //! Are there still more positions within the target region to process?
public:
  Alignment *               alignments_first_;      //! First in a list of all alignments in memory
private:
  Alignment *               alignments_last_;       //! Last in a list of all alignments in memory
  int                       read_counter_;          //! Total # of reads retrieved so far

  Alignment *               recycle_;               //! Stack of allocated, reusable Alignment objects
  int                       recycle_size_;          //! Size of the the recycle stack
  pthread_mutex_t           recycle_mutex_;         //! Mutex controlling access to the recycle stack

  Alignment *               tmp_begin_;             //! Starts read window of most recent position task
  Alignment *               tmp_end_;               //! Ends read window of most recent position task

  Alignment *               processing_first_;      //! First in a list of alignments being processed
  Alignment *               processing_last_;       //! Last in a list of alignments being processed
  list<PositionInProgress>  positions_in_progress_; //! List of positions being processed
  int                       first_excess_read_;     //! Index of the earliest read beyond the current position

  int                       first_useful_read_;     //! Index of the earliest read that may still be in use

  bool                      bam_writing_enabled_;
  BamWriter                 bam_writer_;

  int                       prefix_exclude;

  int                       temp_read_size;
  vector<BamAlignment>      temp_reads;
  BamAlignment              *next_temp_read;
  vector<BamAlignment *>    temp_heap;

};


#endif //BAMWALKERENGINE_H

