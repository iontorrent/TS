/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef IONSTATS_ALIGNMENT_SUMMARY_H
#define IONSTATS_ALIGNMENT_SUMMARY_H

#include "ionstats.h"
#include "ionstats_data.h"
#include "ionstats_alignment.h"

using namespace std;

class AlignmentSummary {
public:

  AlignmentSummary() : 
    n_read_(0), 
    n_skip_rg_(0), 
    n_skip_min_map_qual_(0), 
    n_skip_max_map_qual_(0), 
    n_aligned_(0), 
    n_aligned_with_MD_(0), 
    n_invalid_cigar_(0), 
    n_invalid_read_bases_(0), 
    n_invalid_ref_bases_(0), 
    n_no_flow_data_(0), 
    n_barcode_error_(0), 
    n_no_read_group_(0), 
    n_unmatched_read_group_(0), 
    n_no_region_(0), 
    n_out_of_bounds_(0)
  {}

  void Initialize(map< string, int > & read_groups, IonstatsAlignmentOptions & opt);

  void IncrementReadCount(void)               { n_read_++; };
  void IncrementSkipReadGroupCount(void)      { n_skip_rg_++; };
  void IncrementSkipMinMapQualCount(void)     { n_skip_min_map_qual_++; };
  void IncrementSkipMaxMapQualCount(void)     { n_skip_max_map_qual_++; };
  void IncrementAlignedCount(void)            { n_aligned_++; };
  void IncrementAlignedWithMdCount(void)      { n_aligned_with_MD_++; };
  void IncrementInvalidCigarCount(void)       { n_invalid_cigar_++; };
  void IncrementInvalidReadBasesCount(void)   { n_invalid_read_bases_++; };
  void IncrementInvalidRefBasesCount(void)    { n_invalid_ref_bases_++; };
  void IncrementNoFlowDataCount(void)         { n_no_flow_data_++; };
  void IncrementBarcodeErrorCount(void)       { n_barcode_error_++; };
  void IncrementNoReadGroupCount(void)        { n_no_read_group_++; };
  void IncrementUnmatchedReadGroupCount(void) { n_unmatched_read_group_++; };
  void IncrementNoRegionCount(void)           { n_no_region_++; };
  void IncrementOutOfBoundsCount(void)        { n_out_of_bounds_++; };

  uint64_t ReadCount(void)               { return(n_read_); };
  uint64_t SkipReadGroupCount(void)      { return(n_skip_rg_); };
  uint64_t SkipMinMapQualCount(void)     { return(n_skip_min_map_qual_); };
  uint64_t SkipMaxMapQualCount(void)     { return(n_skip_max_map_qual_); };
  uint64_t AlignedCount(void)            { return(n_aligned_); };
  uint64_t AlignedWithMdCount(void)      { return(n_aligned_with_MD_); };
  uint64_t InvalidCigarCount(void)       { return(n_invalid_cigar_); };
  uint64_t InvalidReadBasesCount(void)   { return(n_invalid_read_bases_); };
  uint64_t InvalidRefBasesCount(void)    { return(n_invalid_ref_bases_); };
  uint64_t NoFlowDataCount(void)         { return(n_no_flow_data_); };
  uint64_t BarcodeErrorCount(void)       { return(n_barcode_error_); };
  uint64_t NoReadGroupCount(void)        { return(n_no_read_group_); };
  uint64_t UnmatchedReadGroupCount(void) { return(n_unmatched_read_group_); };
  uint64_t NoRegionCount(void)           { return(n_no_region_); };
  uint64_t OutOfBoundsCount(void)        { return(n_out_of_bounds_); };

  // reference accessors
  ReadLengthHistogram &           CalledHistogram(void) { return(called_histogram_); };
  ReadLengthHistogram &           AlignedHistogram(void) { return(aligned_histogram_); };
  vector< ReadLengthHistogram > & AqHistogram(void) { return(aq_histogram_); };
  ReadLengthHistogram &           TotalInsertHistogram(void) { return(total_insert_histo_); };
  ReadLengthHistogram &           TotalQ17Histogram(void) { return(total_Q17_histo_); };
  ReadLengthHistogram &           TotalQ20Histogram(void) { return(total_Q20_histo_); };
  MetricGeneratorSNR &            SystemSnr(void) { return(system_snr_); };
  BaseQVHistogram &               QvHistogram(void) { return(qv_histogram_); };
  ReadLengthHistogram &           CalledHistogramBc(void) { return(called_histogram_bc_); };
  ReadLengthHistogram &           AlignedHistogramBc(void) { return(aligned_histogram_bc_); };
  vector< ReadLengthHistogram > & AqHistogramBc(void) { return(aq_histogram_bc_); };
  SimpleHistogram &               BasePositionErrorCount(void) { return(base_position_error_count_); };
  PerReadFlowMatrix &             PerReadFlow(void) { return(per_read_flow_); };
  map< string, ErrorData > &      ReadGroupBasePosition(void) { return(read_group_base_position_); };
  map< string, ErrorData > &      ReadGroupFlowPosition(void) { return(read_group_flow_position_); };
  map< string, HpData > &         ReadGroupPerHp(void) { return(read_group_per_hp_); };
  vector< RegionalSummary > &     GetRegionalSummary(void) { return(regional_summary_); };

  // Accessors returning copies of objects
  ErrorData                           & BasePosition(void) { return(base_position_); };
  ErrorData                           & FlowPosition(void) { return(flow_position_); };
  HpData                              & PerHp(void) { return(per_hp_); };

  void AddCalledLength(unsigned int len)               { called_histogram_.Add(len); };
  void AddAlignedLength(unsigned int len)              { aligned_histogram_.Add(len); };
  void AddAqLength(unsigned int len, unsigned int i)   { aq_histogram_[i].Add(len); };
  void AddAqLength(unsigned int region_idx, unsigned int len, unsigned int i)   {regional_summary_[region_idx].AddAqLength(len,i); };
  void AddInsertLength(unsigned int len)               { total_insert_histo_.Add(len); };
  void AddQ17Length(unsigned int len)                  { total_Q17_histo_.Add(len); };
  void AddQ20Length(unsigned int len)                  { total_Q20_histo_.Add(len); };
  void AddSystemSNR(const vector<uint16_t>& fs, const string &key, const string& fo) { system_snr_.Add(fs, key, fo); };
  void AddSystemSNR(const vector<int16_t>&  fs, const string &key, const string& fo) { system_snr_.Add(fs, key, fo); };
  void AddQVHistogram(const string &qual)              { qv_histogram_.Add(qual); };
  void AddCalledLengthBc(unsigned int len)             { called_histogram_bc_.Add(len); };
  void AddAlignedLengthBc(unsigned int len)            { aligned_histogram_bc_.Add(len); };
  void AddAqLengthBc(unsigned int len, unsigned int i) { aq_histogram_bc_[i].Add(len); };
  void AddBasePosition(ReadAlignmentErrors &e)         { base_position_.Add(e); };
  void AddFlowPosition(ReadAlignmentErrors &e)         { flow_position_.Add(e); };
  void AddPerHp(vector<char> &ref_hp_nuc, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, bool ignore_terminal_hp=true) {
    per_hp_.Add(ref_hp_nuc, ref_hp_len, ref_hp_err, ignore_terminal_hp);
  };
  void AddPerHp(vector<char> &ref_hp_nuc, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> & ref_hp_flow, vector<uint16_t> & zeromer_insertion_flow, vector<uint16_t> & zeromer_insertion_len, string &flow_order, bool ignore_terminal_hp=true) {
    per_hp_.Add(ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, flow_order, ignore_terminal_hp);
  };
  int AddPerReadFlow(string &id, ReadAlignmentErrors &e, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> &ref_hp_flow, bool ignore_terminal_hp=true) {
    return(per_read_flow_.Add(id,e, ref_hp_len, ref_hp_err, ref_hp_flow, ignore_terminal_hp));
  };
  int AddPerReadFlow(string &id, ReadAlignmentErrors &e, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> &ref_hp_flow, vector<uint16_t> & zeromer_insertion_flow, vector<uint16_t> & zeromer_insertion_len, bool ignore_terminal_hp=true) {
    return(per_read_flow_.Add(id,e, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, ignore_terminal_hp));
  };
  bool PerReadFlowBufferFull(void)  { return(per_read_flow_.BufferFull()); };
  bool PerReadFlowBufferEmpty(void) { return(per_read_flow_.BufferEmpty()); };
  void PerReadFlowBufferFlush(void) { per_read_flow_.FlushToH5Buffered(); };
  void PerReadFlowForcedFlush(void) { per_read_flow_.FlushToH5Forced(); };
  void PerReadFlowCloseH5(void) { per_read_flow_.CloseH5(); };
  void AddReadGroupBasePosition(string &read_group, ReadAlignmentErrors &e) { read_group_base_position_[read_group].Add(e); };
  void AddReadGroupFlowPosition(string &read_group, ReadAlignmentErrors &e) { read_group_flow_position_[read_group].Add(e); };
  void AddReadGroupPerHp(string &read_group, vector<char> &ref_hp_nuc, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> & ref_hp_flow, vector<uint16_t> & zeromer_insertion_flow, vector<uint16_t> & zeromer_insertion_len, string &flow_order, bool ignore_terminal_hp=true) {
    read_group_per_hp_[read_group].Add(ref_hp_nuc, ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, zeromer_insertion_len, flow_order, ignore_terminal_hp);
  };
  void AddReadGroupPerHp(string &read_group, vector<char> &ref_hp_nuc, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, bool ignore_terminal_hp) {
    read_group_per_hp_[read_group].Add(ref_hp_nuc, ref_hp_len, ref_hp_err, ignore_terminal_hp);
  };
  void AddRegionalSummaryBasePosition(unsigned int region_idx, ReadAlignmentErrors &e) { regional_summary_[region_idx].Add(e); };
  void AddRegionalSummaryPerHp(unsigned int region_idx, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> &ref_hp_flow, bool ignore_terminal_hp=true) {
    regional_summary_[region_idx].Add(ref_hp_len, ref_hp_err, ref_hp_flow, ignore_terminal_hp);
  };
  void AddRegionalSummaryPerHp(unsigned int region_idx, vector<uint16_t> &ref_hp_len, vector<int16_t> &ref_hp_err, vector<uint16_t> &ref_hp_flow, vector<uint16_t> & zeromer_insertion_flow, bool ignore_terminal_hp=true) {
    regional_summary_[region_idx].Add(ref_hp_len, ref_hp_err, ref_hp_flow, zeromer_insertion_flow, ignore_terminal_hp);
  };

  void AddBasePositionErrorCount(const vector<uint16_t> &err_pos, const vector<uint16_t> &err_len) {
    vector<uint16_t>::const_iterator pos=err_pos.begin();
    vector<uint16_t>::const_iterator len=err_len.begin();
    while(pos != err_pos.end() && len != err_len.end()) {
      base_position_error_count_.Add(*pos,*len);
      ++pos;
      ++len;
    }
  }

  void WriteWarnings(
    ostream &outstream,
    const string &prefix,
    const string &skip_rg_suffix,
    int min_map_qual,
    int max_map_qual,
    bool spatial_stratify,
    bool evaluate_flow,
    bool bc_adjust
  );

  void FillBasePositionDepths(void) {
    base_position_.ComputeDepth();
    for(map< string, ErrorData >::iterator it = read_group_base_position_.begin(); it != read_group_base_position_.end(); ++it)
      it->second.ComputeDepth();
  }

  void MergeFrom(AlignmentSummary &other);

private:

  uint64_t n_read_;
  uint64_t n_skip_rg_;
  uint64_t n_skip_min_map_qual_;
  uint64_t n_skip_max_map_qual_;
  uint64_t n_aligned_;
  uint64_t n_aligned_with_MD_;
  uint64_t n_invalid_cigar_;
  uint64_t n_invalid_read_bases_;
  uint64_t n_invalid_ref_bases_;
  uint64_t n_no_flow_data_;
  uint64_t n_barcode_error_;
  uint64_t n_no_read_group_;
  uint64_t n_unmatched_read_group_;
  uint64_t n_no_region_;
  uint64_t n_out_of_bounds_;

  // Called & aligned lengths
  ReadLengthHistogram called_histogram_;
  ReadLengthHistogram aligned_histogram_;
  vector< ReadLengthHistogram > aq_histogram_;
  ReadLengthHistogram total_insert_histo_;
  ReadLengthHistogram total_Q17_histo_;
  ReadLengthHistogram total_Q20_histo_;
  MetricGeneratorSNR system_snr_;
  BaseQVHistogram qv_histogram_;
  // Called & aligned lengths including barcodes
  ReadLengthHistogram called_histogram_bc_;
  ReadLengthHistogram aligned_histogram_bc_;
  vector< ReadLengthHistogram > aq_histogram_bc_;
  // Per-base and per-flow error data
  SimpleHistogram base_position_error_count_;
  ErrorData base_position_;
  ErrorData flow_position_;
  HpData per_hp_;
  PerReadFlowMatrix per_read_flow_;
  // Read Group per-base and per-flow error data
  map< string, ErrorData > read_group_base_position_;
  map< string, ErrorData > read_group_flow_position_;
  map< string, HpData > read_group_per_hp_;
  // Regional summary data
  vector< RegionalSummary > regional_summary_;

};


#endif // IONSTATS_ALIGNMENT_SUMMARY_H
