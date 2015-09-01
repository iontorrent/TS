/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BAMWalkerEngine.h
//! @ingroup  VariantCaller
//! @brief    Streamed BAM reader

#include "BAMWalkerEngine.h"
#include <algorithm>

#include <errno.h>
#include <limits.h>
#include <set>
#include "ReferenceReader.h"



BAMWalkerEngine::BAMWalkerEngine()
{
  targets_manager_ = NULL;
  next_target_ = NULL;
  next_position_ = 0;
  last_processed_chr_ = 0;
  last_processed_pos_ = 0;
  has_more_alignments_ = true;
  has_more_positions_ = true;
  tmp_begin_ = NULL;
  tmp_end_ = NULL;
  processing_first_ = NULL;
  processing_last_ = NULL;
  alignments_first_ = NULL;
  alignments_last_ = NULL;
  read_counter_ = 0;
  recycle_ = NULL;
  recycle_size_ = 0;
  first_excess_read_ = 0;
  first_useful_read_ = 0;
  bam_writing_enabled_ = false;
  pthread_mutex_init(&recycle_mutex_, NULL);
  temp_read_size = 100;
  temp_reads.resize(temp_read_size);
  next_temp_read = NULL;
}


BAMWalkerEngine::~BAMWalkerEngine()
{
  pthread_mutex_destroy(&recycle_mutex_);
}


void BAMWalkerEngine::Initialize(const ReferenceReader& ref_reader, TargetsManager& targets_manager,
    const vector<string>& bam_filenames, const string& postprocessed_bam, int px)
{

  InitializeBAMs(ref_reader, bam_filenames);

  targets_manager_ = &targets_manager;
  next_target_ = &targets_manager_->merged.front();
  next_position_ = next_target_->begin;
  prefix_exclude = px;

  // BAM writing init
  if (not postprocessed_bam.empty()) {
    bam_writing_enabled_ = true;
    SamHeader tmp_header = bam_header_;
    tmp_header.Comments.clear();
    tmp_header.Programs.Clear();
    bam_writer_.SetCompressionMode(BamWriter::Compressed);
    bam_writer_.SetNumThreads(4);
    if (not bam_writer_.Open(postprocessed_bam, tmp_header, bam_reader_.GetReferenceData())) {
      cerr << "ERROR: Could not open postprocessed BAM file for writing : " << bam_writer_.GetErrorString();
      exit(1);
    }
  }
}

void BAMWalkerEngine::Close()
{
  if (bam_writing_enabled_)
    bam_writer_.Close();
  bam_reader_.Close();
}



// open BAM input file
void BAMWalkerEngine::InitializeBAMs(const ReferenceReader& ref_reader, const vector<string>& bam_filenames)
{
  if (not bam_reader_.SetExplicitMergeOrder(BamMultiReader::MergeByCoordinate)) {
    cerr << "ERROR: Could not set merge order to BamMultiReader::MergeByCoordinate" << endl;
    exit(1);
  }

  if (not bam_reader_.Open(bam_filenames)) {
    cerr << "ERROR: Could not open input BAM file(s) : " << bam_reader_.GetErrorString() << endl;
    exit(1);
  }
  if (not bam_reader_.LocateIndexes()) {
    cerr << "ERROR: Could not open BAM index file(s) : " << bam_reader_.GetErrorString() << endl;
    exit(1);
  }

  // BAM multi reader combines the read group information of the different BAMs but does not merge comment sections
  bam_header_ = bam_reader_.GetHeader();
  if (!bam_header_.HasReadGroups()) {
    cerr << "ERROR: there is no read group in BAM files specified" << endl;
    exit(1);
  }

  // Manually merge comment sections of BAM files if we have more than one BAM file
  if (bam_filenames.size() > 1) {

    unsigned int num_duplicates = 0;
    unsigned int num_merged = 0;

    for (unsigned int bam_idx = 0; bam_idx < bam_filenames.size(); bam_idx++) {

      BamReader reader;
      if (not reader.Open(bam_filenames.at(bam_idx))) {
        cerr << "TVC ERROR: Failed to open input BAM file " << reader.GetErrorString() << endl;
    	 exit(1);
      }
      SamHeader header = reader.GetHeader();

      for (unsigned int i_co = 0; i_co < header.Comments.size(); i_co++) {

        // Step 1: Check if this comment is already part of the merged header
    	unsigned int m_co = 0;
    	while (m_co < bam_header_.Comments.size() and bam_header_.Comments.at(m_co) != header.Comments.at(i_co))
    	  m_co++;

    	if (m_co < bam_header_.Comments.size()){
          num_duplicates++;
          continue;
    	}

    	// Add comment line to merged header if it is a new one
    	num_merged++;
    	bam_header_.Comments.push_back(header.Comments.at(i_co));
      }
    }
    // Verbose what we did
    cout << "Merged " << num_merged << " unique comment lines into combined BAM header. Encountered " << num_duplicates << " duplicate comments." << endl;
  }

  //
  // Reference sequences in the bam file must match that in the fasta file
  //

  vector<RefData> referenceSequences = bam_reader_.GetReferenceData();

  if ((int)referenceSequences.size() != ref_reader.chr_count()) {
    cerr << "ERROR: Reference in BAM file does not match fasta file" << endl
         << "       BAM has " << referenceSequences.size()
         << " chromosomes while fasta has " << ref_reader.chr_count() << endl;
    exit(1);
  }

  for (int chr_idx = 0; chr_idx < ref_reader.chr_count(); ++chr_idx) {
    if (referenceSequences[chr_idx].RefName != ref_reader.chr_str(chr_idx)) {
      cerr << "ERROR: Reference in BAM file does not match fasta file" << endl
           << "       Chromosome #" << (chr_idx+1) << "in BAM is " << referenceSequences[chr_idx].RefName
           << " while fasta has " << ref_reader.chr_str(chr_idx) << endl;
      exit(1);
    }
    if (referenceSequences[chr_idx].RefLength != ref_reader.chr_size(chr_idx)) {
      cerr << "ERROR: Reference in BAM file does not match fasta file" << endl
           << "       Chromosome " << referenceSequences[chr_idx].RefName
           << "in BAM has length " << referenceSequences[chr_idx].RefLength
           << " while fasta has " << ref_reader.chr_size(chr_idx) << endl;
      exit(1);
    }
  }


  //
  // Retrieve BaseCaller and TMAP version strings from BAM header
  //

  set<string> basecaller_versions;
  set<string> tmap_versions;
  for (SamProgramIterator I = bam_header_.Programs.Begin(); I != bam_header_.Programs.End(); ++I) {
    if (I->ID.substr(0,2) == "bc")
      basecaller_versions.insert(I->Version);
    if (I->ID.substr(0,4) == "tmap")
      tmap_versions.insert(I->Version);
  }
  basecaller_version_ = "";
  for (set<string>::const_iterator I = basecaller_versions.begin(); I != basecaller_versions.end(); ++I) {
    if (not basecaller_version_.empty())
      basecaller_version_ += ", ";
    basecaller_version_ += *I;
  }
  tmap_version_ = "";
  for (set<string>::const_iterator I = tmap_versions.begin(); I != tmap_versions.end(); ++I) {
    if (not tmap_version_.empty())
      tmap_version_ += ", ";
    tmap_version_ += *I;
  }

}



bool  BAMWalkerEngine::EligibleForReadRemoval()
{
  return alignments_first_ and alignments_first_->read_number+100 < first_useful_read_;
}


void BAMWalkerEngine::RequestReadRemovalTask(Alignment*& removal_list)
{
  removal_list = alignments_first_;

  Alignment *list_end = removal_list;

  while (alignments_first_ and alignments_first_->read_number < first_useful_read_) {
    list_end = alignments_first_;
    alignments_first_ = alignments_first_->next;
  }
  if (list_end == removal_list)
    removal_list = NULL;
  else
    list_end->next = NULL;
}


void BAMWalkerEngine::SaveAlignments(Alignment* removal_list)
{
  if (not bam_writing_enabled_)
    return;
  
//  if (!removal_list) removal_list = alignments_first_;
    
  for (Alignment *current_read = removal_list; current_read; current_read = current_read->next) {
    if (not current_read->worth_saving)
      continue;
    current_read->alignment.RemoveTag("ZM");
    current_read->alignment.RemoveTag("ZP");
    current_read->alignment.RemoveTag("PG");
    bam_writer_.SaveAlignment(current_read->alignment);
  }
}


void BAMWalkerEngine::FinishReadRemovalTask(Alignment* removal_list)
{
  pthread_mutex_lock(&recycle_mutex_);

  while (removal_list) {

    Alignment *excess = removal_list;
    removal_list = removal_list->next;
    if (recycle_size_ > 55000) {
      delete excess;
    } else {
      excess->next = recycle_;
      recycle_ = excess;
      recycle_size_++;
    }
  }
  pthread_mutex_unlock(&recycle_mutex_);
}





bool BAMWalkerEngine::EligibleForGreedyRead()
{
  if (not has_more_alignments_)
    return false;

  return read_counter_ < (first_excess_read_ + 10000);

}


bool BAMWalkerEngine::ReadyForNextPosition()
{
  // Try Getting More Reads - if just starting
  if (not alignments_last_)
    return false;

  // Try Getting More Reads - if target regions positions have been processed, but there are more reads
  if (not has_more_positions_) // and has_more_alignments_)
    return false;

  // Try Generating a Variant - if processed reads are ahead of current position
  if (last_processed_chr_ > next_target_->chr or
      (last_processed_chr_ == next_target_->chr and last_processed_pos_ > next_position_))
    return true;

  // Try Generating a Variant (or quitting) - if all reads have been loaded and processed
  if (not has_more_alignments_ and not processing_first_)
    return true;

  return false;
}



void BAMWalkerEngine::RequestReadProcessingTask(Alignment* & new_read)
{

  pthread_mutex_lock(&recycle_mutex_);
  if (recycle_) {
    new_read = recycle_;
    recycle_ = recycle_->next;
    recycle_size_--;
    pthread_mutex_unlock(&recycle_mutex_);
    new_read->Reset();
  } else {
    pthread_mutex_unlock(&recycle_mutex_);
    try {
      new_read = new Alignment;
    }
    catch(std::bad_alloc& exc)
    {
      cerr << "ERROR: failed to allocate memory in reading BAM in BAMWalkerEngine::RequestReadProcessingTask" << endl;
      exit(1);	
    }
  }
  new_read->read_number = read_counter_++;

  if (alignments_last_)
    alignments_last_->next = new_read;
  else
    alignments_first_ = new_read;
  alignments_last_ = new_read;

  // Add new read to the end of the "processing" list
  if (processing_last_) {
    processing_last_->processing_next = new_read;
    new_read->processing_prev = processing_last_;
    processing_last_ = new_read;
  } else {
    processing_first_ = new_read;
    processing_last_ = new_read;
  }

}
static int prefix_exclude_ = 6;

static bool myorder(BamAlignment *A, BamAlignment *B)
{
    return (A->Name.substr(prefix_exclude_).compare(B->Name.substr(prefix_exclude_)) > 0);
}
bool BAMWalkerEngine::GetNextAlignmentCore(Alignment* new_read)
{
  //return has_more_alignments_ = (bam_reader_.GetNextAlignmentCore(new_read->alignment) && new_read!=NULL && new_read->alignment.RefID>=0);
  // maintain a list of all reads that are in order of read name if the position is the same, ZZ
  if (temp_heap.size() == 0) {
    int i = 0;
    if (next_temp_read) {
      temp_reads[0] = *next_temp_read;
      i = 1;
      next_temp_read = NULL;
    }
    temp_heap.clear();
    do {
	if (!bam_reader_.GetNextAlignmentCore(temp_reads[i])) break;
	temp_reads[i].BuildCharData();
	if (temp_reads[i].RefID < 0) break;
	if (temp_reads[i].Position != temp_reads[0].Position or temp_reads[i].RefID != temp_reads[0].RefID) {
	    next_temp_read = &temp_reads[i];
	    break;
	} 
	i++;
	if (i >= temp_read_size) {
	    temp_read_size *= 2;
	    temp_reads.resize(temp_read_size);
	}
    } while (1);
    if (i == 0) return has_more_alignments_ = false;
    for (int j = 0; j < i; j++) {
	temp_heap.push_back(&temp_reads[j]);
    }
    if (temp_heap.size() > 1) {
	prefix_exclude_ = prefix_exclude;
        std::sort(temp_heap.begin(), temp_heap.end(), myorder); // sort into descending order by read Name
    }
  }
  new_read->alignment = *(temp_heap.back()); // output read by ascending order of read Name using popback.
  temp_heap.pop_back();
  return has_more_alignments_ = true;
}


void BAMWalkerEngine::FinishReadProcessingTask(Alignment* new_read, bool success)
{
  new_read->processed = success;
  if (success and new_read == processing_first_) {
    last_processed_chr_ = new_read->alignment.RefID;
    last_processed_pos_ = new_read->original_position;

    // Special case. If no positions are being processed and this read ends before the next position, trigger a cleanup
    if (positions_in_progress_.empty() and (new_read->alignment.RefID < next_target_->chr
        or (new_read->alignment.RefID == next_target_->chr and new_read->end <= next_position_))) {

      Alignment *useful_read = alignments_first_;
      while (useful_read != new_read and (useful_read->alignment.RefID < next_target_->chr
          or (useful_read->alignment.RefID == next_target_->chr and useful_read->end <= next_position_))) {
        useful_read = useful_read->next;
      }
      first_useful_read_ = max(first_useful_read_,useful_read->read_number);

      tmp_begin_ = NULL;
      tmp_end_ = NULL;
    }

    if (not has_more_positions_ and positions_in_progress_.empty())
      first_useful_read_ = max(first_useful_read_,new_read->read_number);

  }
  if (new_read->processing_prev)
    new_read->processing_prev->processing_next = new_read->processing_next;
  else
    processing_first_ = new_read->processing_next;

  if (new_read->processing_next)
    new_read->processing_next->processing_prev = new_read->processing_prev;
  else
    processing_last_ = new_read->processing_prev;

}


void BAMWalkerEngine::BeginPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket)
{
  if (not tmp_begin_)
    tmp_begin_ = alignments_first_;

  while (tmp_begin_ and (
      (tmp_begin_->alignment.RefID == next_target_->chr and tmp_begin_->end <= next_position_)
      or tmp_begin_->alignment.RefID < next_target_->chr)
      and tmp_begin_->processed)
    tmp_begin_ = tmp_begin_->next;

  if (not tmp_end_)
    tmp_end_ = tmp_begin_;

  while (tmp_end_ and (
      (tmp_end_->alignment.RefID == next_target_->chr and tmp_end_->original_position <= next_position_)
      or tmp_end_->alignment.RefID < next_target_->chr)
      and tmp_end_->processed)
    tmp_end_ = tmp_end_->next;

  positions_in_progress_.push_back(PositionInProgress());
  position_ticket = positions_in_progress_.end();
  --position_ticket;
  position_ticket->chr = next_target_->chr;
  position_ticket->pos = next_position_;
  position_ticket->target_end = next_target_->end;
  position_ticket->begin = tmp_begin_;
  position_ticket->end = tmp_end_;
  position_ticket->start_time = time(NULL);


  first_excess_read_ = tmp_end_->read_number;
}


bool BAMWalkerEngine::AdvancePosition(int position_increment, int next_hotspot_chr, long next_hotspot_position)
{
  next_position_ += position_increment;

  // Skip-ahead logic for sparse BAMs
  if (tmp_begin_) {
    int closest_chr = tmp_begin_->alignment.RefID;
    int closest_pos = tmp_begin_->original_position;
    if (next_hotspot_chr >= 0) {
      if (next_hotspot_chr < closest_chr or (next_hotspot_chr == closest_chr and next_hotspot_position < closest_pos)) {
        // actually the hotspot is closer
        closest_chr = next_hotspot_chr;
        closest_pos = next_hotspot_position;
      }
    }

    if (next_target_->chr < closest_chr)
      next_position_ = next_target_->end; // Just force transition to next target
    else if (next_target_->chr == closest_chr and next_position_ < closest_pos)
      next_position_ = closest_pos;
  }

  if (next_position_ >= next_target_->end) {
    if (next_target_ == &targets_manager_->merged.back()) {// Can't go any further
      has_more_positions_ = false;
      return false;
    }
    next_target_++;
    next_position_ = next_target_->begin;
  }
  return true;
}


void BAMWalkerEngine::FinishPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket)
{
  time_t delta = time(NULL) - position_ticket->start_time;
  if (delta > 60) {
    cerr<< "WARNING: Variant " << bam_reader_.GetReferenceData()[position_ticket->chr].RefName << ":" << (position_ticket->pos+1)
        <<" has unexpected processing time of " << delta << " seconds." << endl;
  }

  if (position_ticket == positions_in_progress_.begin()) {
    first_useful_read_ = max(first_useful_read_,position_ticket->begin->read_number);
    positions_in_progress_.erase(position_ticket);
    if (not positions_in_progress_.empty()) {
      first_useful_read_ = max(first_useful_read_,positions_in_progress_.begin()->begin->read_number);
    }
  } else {
    positions_in_progress_.erase(position_ticket);
  }
}



int BAMWalkerEngine::GetRecentUnmergedTarget()
{
  MergedTarget *my_next_target = next_target_;
  if (my_next_target)
    return my_next_target->first_unmerged;
  else
    return 0;
}


bool BAMWalkerEngine::MemoryContention()
{
  if (positions_in_progress_.empty())
    return false;
  if (not alignments_first_)
    return false;
  if (read_counter_ - alignments_first_->read_number < 50000)
    return false;
  return true;
}

bool BAMWalkerEngine::IsEarlierstPositionProcessingTask(list<PositionInProgress>::iterator& position_ticket)
{
  return position_ticket == positions_in_progress_.begin();
}

void BAMWalkerEngine::PrintStatus()
{
  cerr<< "BAMWalkerEngine:"
      << " start=" << alignments_first_->read_number
      << " in_memory="   << read_counter_ - alignments_first_->read_number
      << " deleteable=" << first_useful_read_ - alignments_first_->read_number
      << " read_ahead=" << read_counter_ - first_excess_read_
      << " recycle=" << recycle_size_ << endl;
}




