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
#include "MolecularTagTrimmer.h"

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

  Alignment() { Reset(); read_count = 1;}
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
    target_coverage_indices.clear();
    is_reverse_strand = false;
    measurements.clear();
    measurements_length = 0;
    phase_params.clear();
    measurements_sd.clear();
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
    prefix_bases.clear();
    suffix_bases.clear();
    tag_info.Clear();
//	read_count = 1;
  }

  BamAlignment          alignment;          //! Raw BamTools alignment
  Alignment*            next;               //! Singly-linked list for durable alignments iterator
  int                   read_number;        //! Sequential number of this read
  int                   original_position;  //! Alignment position, before primer trimming

  // Processing state
  bool                  processed;          //! Is candidate generator's pre-processing finished?
  Alignment*            processing_prev;    //! Previous in a list of alignments being processed
  Alignment*            processing_next;    //! Next in a list of alignments being processed

  // Read Tag information
  MolTag                tag_info;           //! Structure to store tag information for this read

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
  vector<int>           target_coverage_indices;

  // Candidate evaluator information
  bool                  is_reverse_strand;  //! Indicates whether read is from the forward or reverse strand
  vector<float>         measurements;       //! The measurement values for this read blown up to the length of the flow order
  int                   measurements_length;//! Original trimmed length of the ZM measurements vector
  vector<float>         phase_params;       //! cf, ie, droop parameters of this read
  vector<float>         measurements_sd;    //! The standard deviation of measurements for consensus reads
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
  string                prefix_bases;       //! hard clipped start of the read
  string                suffix_bases;       //! hard clipped end of the read

  // Post-processing information
  vector<CigarOp>       old_cigar;          //! Cigar information before primer trimming
  int                   read_count;         //! The number of reads associated with this alignment. Used to capture consensus depth.
};


struct PositionInProgress {

  int                   chr;                //! Chromosome index of this variant position
  long                  pos;                //! Position within chromosome
  long                  target_begin;       //! Begin of current target region
  long                  target_end;         //! End of current target region
  Alignment *           begin;              //! First read covering this position
  Alignment *           end;                //! Last read coverint this position
  time_t                start_time;
};


class VariantCallerContext;
class ReferenceReader;
class IndelAssembly;

class BAMWalkerEngine {
public:

  // Initialization
  BAMWalkerEngine();
  ~BAMWalkerEngine();
  void Initialize(const ReferenceReader& ref_reader, TargetsManager& targets_manager,
      const vector<string>& bam_filenames, const string& postprocessed_bam, int px);
  void Close();
  const SamHeader& GetBamHeader() { return bam_header_; }

  // Job dispatch
  bool EligibleForReadRemoval();
  bool EligibleForGreedyRead();
  bool ReadyForNextPosition();

  // Memory contention prevention
  bool MemoryContention(int max_num_reads = 50000);
  bool IsEarlierstPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);

  // Loading new reads
  void RequestReadProcessingTask(Alignment*& new_read);
  bool GetNextAlignmentCore(Alignment* new_read, VariantCallerContext& vc, vector<MergedTarget>::iterator& indel_target);
  void FinishReadProcessingTask(Alignment* new_read, bool success);

  // Processing genomic position
  void SetupPositionTicket(list<PositionInProgress>::iterator& position_ticket) const;
  void BeginPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);
  bool AdvancePosition(int position_increment, int next_hotspot_chr = -1, long next_hotspot_position = -1);
  void FinishPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket);

  // Deleting or saving used up reads
  void RequestReadRemovalTask(Alignment*& removal_list);
  void SaveAlignments(Alignment*& removal_list, VariantCallerContext& vc, vector<MergedTarget>::iterator& depth_target);
  void FinishReadRemovalTask(Alignment* removal_list, int recycle_limit = 55000);

  void PrintStatus();
  int GetRecentUnmergedTarget();

  bool HasMoreAlignments() { return has_more_alignments_; }
  bool ReadProcessingTasksInProgress() { return processing_first_; }

  void GetProgramVersions(string& basecaller_version, string& tmap_version) {
    basecaller_version = basecaller_version_;
    tmap_version = tmap_version_;
  }
  void AddReadToPG(Alignment *rai);
  bool GetMostPopularTmap(SamProgram& most_popular_tmap);

  void processDepth(BamAlignment& alignment, TargetsManager* targets_manager, vector<MergedTarget>::iterator& curr_target);
  void openDepth(const string& filename);
  void writeDepth(ReferenceReader* reference_reader, std::map<long int, int>::size_type offset = 0);
  void closeDepth(ReferenceReader* reference_reader);
  int getChrIndex() const {return next_target_->chr;}
  long int getStartPosition() const {return next_target_->begin;}
  long int getEndPosition() const {return next_target_->end;}
  long int getPosition() const {return next_position_;}
  Alignment *               alignments_first_;      //! First in a list of all alignments in memory

private:
  void InitializeBAMs(const ReferenceReader& ref_reader, const vector<string>& bam_filenames);

  TargetsManager *          targets_manager_;       //! Manages targets loaded from BED file
  string                    basecaller_version_;    //! BaseCaller version retrieved from BAM header
  string                    tmap_version_;          //! TMAP version retrieved from BAM header

  Alignment *               recycle_;               //! Stack of allocated, reusable Alignment objects
  int                       recycle_size_;          //! Size of the the recycle stack
  pthread_mutex_t           recycle_mutex_;         //! Mutex controlling access to the recycle stack

  bool                      bam_writing_enabled_;
  BamWriter                 bam_writer_;

  int                       prefix_exclude;

  int                       temp_read_size;
  vector<BamAlignment>      temp_reads;
  BamAlignment              *next_temp_read;
  vector<BamAlignment *>    temp_heap;

  std::map<long int, int> depth_map;
  ofstream depth_out;
  pthread_mutex_t mutexdepth;
  int prevRefID;
  long int prevEndPos;

protected:
  BamMultiReader            bam_reader_;            //! BamTools mulit-bam reader
  SamHeader                 bam_header_;            //! Bam header

  MergedTarget *            next_target_;           //! Target containing next position
  long int                  next_position_;         //! Next position (chr in the target)

  int                       last_processed_chr_;    //! Reads up to this chr+pos are guaranteed to be processed
  long int                  last_processed_pos_;    //! Reads up to this chr+pos are guaranteed to be processed
  bool                      has_more_alignments_;   //! Are there still more reads in BAM?
  bool                      has_more_positions_;    //! Are there still more positions within the target region to process?
  Alignment *               alignments_last_;       //! Last in a list of all alignments in memory
  int                       read_counter_;          //! Total # of reads retrieved so far

  Alignment *               tmp_begin_;             //! Starts read window of most recent position task
  Alignment *               tmp_end_;               //! Ends read window of most recent position task

  Alignment *               processing_first_;      //! First in a list of alignments being processed
  Alignment *               processing_last_;       //! Last in a list of alignments being processed
  list<PositionInProgress>  positions_in_progress_; //! List of positions being processed
  int                       first_excess_read_;     //! Index of the earliest read beyond the current position

  int                       first_useful_read_;     //! Index of the earliest read that may still be in use
  map<string, unsigned int> read_counts_of_pg_;     //!  Count how many reads that use the program for alignment.
};

class ConsensusBAMWalkerEngine : public BAMWalkerEngine
{
private:
    BamWriter               aln_no_needed_consensus_bam_writer_;
    BamWriter               aln_needed_consensus_bam_writer_;
    pthread_mutex_t         aln_no_needed_consensus_bam_writer_mutex_;
    pthread_mutex_t         aln_needed_consensus_bam_writer_mutex_;
    bool                    write_consensus_bam_ = true;
public:
    ConsensusBAMWalkerEngine() : BAMWalkerEngine(){};
	void SaveConsensusAlignments(Alignment* const & read_list, Alignment* const & aln_needed_read_list);
	void Initialize(const ReferenceReader& ref_reader, TargetsManager& targets_manager,
	        const vector<string>& bam_filenames, const string& postprocessed_bam, int px, const string& consensus_bam);
	void Close();
    void RequestTargetBasedReadRemovalTask(Alignment*& removal_list);
    void BeginTargetProcessingTask(list<PositionInProgress>::iterator& position_ticket);
    bool EligibleForTargetBasedReadRemoval();
};

#endif //BAMWALKERENGINE_H

