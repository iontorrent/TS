/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ionstats_alignment_summary.h"

void AlignmentSummary::Initialize(map< string, int > & read_groups, IonstatsAlignmentOptions & opt) {
  // Called & aligned lengths
  called_histogram_.Initialize(opt.HistogramLength());
  aligned_histogram_.Initialize(opt.HistogramLength());
  aq_histogram_.resize(opt.NErrorRates());
  for(unsigned int i=0; i < opt.NErrorRates(); ++i)
    aq_histogram_[i].Initialize(opt.HistogramLength());
  total_insert_histo_.Initialize(opt.HistogramLength());;
  total_Q17_histo_.Initialize(opt.HistogramLength());;
  total_Q20_histo_.Initialize(opt.HistogramLength());;
  // Called & aligned lengths including barcodes
  if(opt.BcAdjust()) {
    called_histogram_bc_.Initialize(opt.HistogramLength());
    aligned_histogram_bc_.Initialize(opt.HistogramLength());
    aq_histogram_bc_.resize(opt.NErrorRates());
    for(unsigned int i=0; i < opt.NErrorRates(); ++i)
      aq_histogram_bc_[i].Initialize(opt.HistogramLength());
  }
  // Per-base and per-flow error data
  base_position_error_count_.Initialize(opt.HistogramLength());
  base_position_.Initialize(opt.HistogramLength());
  if(opt.EvaluateFlow())
    flow_position_.Initialize(opt.NFlow());
  if(opt.EvaluateHp())
    per_hp_.Initialize(opt.MaxHp());
  if(opt.EvaluatePerReadPerFlow()) {
    per_read_flow_.Initialize(opt.NFlow());
    per_read_flow_.InitializeNewH5(opt.OutputH5Filename());
  }
  if(opt.SpatialStratify()) {
    unsigned int n_subregions = opt.NSubregions();
    regional_summary_.resize(n_subregions);
    for(unsigned int i=0; i < n_subregions; ++i)
      regional_summary_[i].Initialize(opt.MaxSubregionHp(),opt.NFlow(),opt.RegionSpecificOrigin(i),opt.RegionSpecificDim(i),opt.NErrorRates(),opt.HistogramLength());
  }

  // Initialize per-read-group structures for any new read groups
  for(map< string, int >::iterator it = read_groups.begin(); it != read_groups.end(); ++it) {
    // Check if the read group has already been seen
    map< string, ErrorData >::iterator rg_it = read_group_base_position_.find(it->first);
    if(rg_it != read_group_base_position_.end())
      continue;
    // It is a new read group, need to initialize
    ErrorData temp_error_data;
    read_group_base_position_[it->first] = temp_error_data;
    read_group_base_position_[it->first].Initialize(opt.HistogramLength());
    if(opt.EvaluateFlow()) {
      read_group_flow_position_[it->first] = temp_error_data;
      read_group_flow_position_[it->first].Initialize(opt.NFlow());
    }
    if(opt.EvaluateHp()) {
      HpData temp_hp_data;
      read_group_per_hp_[it->first] = temp_hp_data;
      read_group_per_hp_[it->first].Initialize(opt.MaxHp());
    }
  }
}

void AlignmentSummary::WriteWarnings(
  ostream &outstream,
  const string &prefix,
  const string &skip_rg_suffix,
  int min_map_qual,
  int max_map_qual,
  bool spatial_stratify,
  bool evaluate_flow,
  bool bc_adjust
) {
  if(skip_rg_suffix != "" && SkipReadGroupCount() > 0)
    outstream << "NOTE: " << prefix << ": " << SkipReadGroupCount() << " reads skipped due to not matching read group suffix " << skip_rg_suffix << endl;
  if(min_map_qual > -1)
    outstream << "NOTE: " << prefix << ": " << SkipMinMapQualCount() << " reads skipped due to having map qual less than " << min_map_qual << endl;
  if(max_map_qual > -1)
    outstream << "NOTE: " << prefix << ": " << SkipMaxMapQualCount() << " reads skipped due to having map qual greater than " << max_map_qual << endl;
  if(spatial_stratify && ( (NoRegionCount() > 0) || (OutOfBoundsCount() > 0) ) )
    outstream << "WARNING: " << prefix << ": of " << ReadCount() << " reads, " << NoRegionCount() << " have no region and " << OutOfBoundsCount() << " are out-of-bounds." << endl;
  if(evaluate_flow && NoFlowDataCount() > 0)
    outstream << "WARNING: " << prefix << ": " << NoFlowDataCount() << " of " << AlignedCount() << " aligned reads have no flow data" << endl;
  if(AlignedWithMdCount() < AlignedCount())
    outstream << "WARNING: " << prefix << ": " << (AlignedCount() - AlignedWithMdCount()) << " of " << AlignedCount() << " aligned reads have no MD tag" << endl;
  if(InvalidCigarCount() > 0)
    outstream << "WARNING: " << prefix << ": " << InvalidCigarCount() << " of " << AlignedWithMdCount() << " aligned reads with MD tag have incompatible CIGAR and SEQ entries" << endl;
  if(InvalidReadBasesCount() > 0)
    outstream << "NOTE: " << prefix << ": " << InvalidReadBasesCount() << " of " << AlignedWithMdCount() << " aligned reads with MD tag have one or more non-[ACGT] bases in the read" << endl;
  if(InvalidRefBasesCount() > 0)
    outstream << "NOTE: " << prefix << ": " << InvalidRefBasesCount() << " of " << AlignedWithMdCount() << " aligned reads with MD tag have one or more non-[ACGT] bases in the reference" << endl;
  if(bc_adjust)
    outstream << "NOTE: " << prefix << ": " << BarcodeErrorCount() << " of " << ReadCount() << " reads have bc errors." << endl;
  if( (NoReadGroupCount() > 0) || (UnmatchedReadGroupCount() > 0) )
    outstream << "WARNING: of " << prefix << ": " << AlignedCount() << " aligned reads, " << NoReadGroupCount() << " have no RG tag " << UnmatchedReadGroupCount() << " have an RG tag that does not match the header." << endl;
}

void AlignmentSummary::MergeFrom(AlignmentSummary &other) {
  n_read_                 += other.ReadCount();
  n_skip_rg_              += other.SkipReadGroupCount();
  n_skip_min_map_qual_    += other.SkipMinMapQualCount();
  n_skip_max_map_qual_    += other.SkipMaxMapQualCount();
  n_aligned_              += other.AlignedCount();
  n_aligned_with_MD_      += other.AlignedWithMdCount();
  n_invalid_cigar_        += other.InvalidCigarCount();
  n_invalid_read_bases_   += other.InvalidReadBasesCount();
  n_invalid_ref_bases_    += other.InvalidRefBasesCount();
  n_no_flow_data_         += other.NoFlowDataCount();
  n_barcode_error_        += other.BarcodeErrorCount();
  n_no_read_group_        += other.NoReadGroupCount();
  n_unmatched_read_group_ += other.UnmatchedReadGroupCount();
  n_no_region_            += other.NoRegionCount();
  n_out_of_bounds_        += other.OutOfBoundsCount();

  // Called & aligned lengths
  called_histogram_.MergeFrom(other.CalledHistogram());
  aligned_histogram_.MergeFrom(other.AlignedHistogram());
  if(aq_histogram_.size() != other.AqHistogram().size()) {
    cerr << "ERROR: unable to merge alignment summary data, AQ histograms have different size" << endl;
    exit(EXIT_FAILURE);
  } else {
    for(unsigned int i=0; i<aq_histogram_.size(); ++i)
      aq_histogram_[i].MergeFrom(other.AqHistogram()[i]);
  }
  total_insert_histo_.MergeFrom(other.TotalInsertHistogram());
  total_Q17_histo_.MergeFrom(other.TotalQ17Histogram());
  total_Q20_histo_.MergeFrom(other.TotalQ20Histogram());
  system_snr_.MergeFrom(other.SystemSnr());
  qv_histogram_.MergeFrom(other.QvHistogram());
  // Called & aligned lengths including barcodes
  called_histogram_bc_.MergeFrom(other.CalledHistogramBc());
  aligned_histogram_bc_.MergeFrom(other.AlignedHistogramBc());
  if(aq_histogram_bc_.size() != other.AqHistogramBc().size()) {
    cerr << "ERROR: unable to merge alignment summary data, AQ histograms have different size" << endl;
    exit(EXIT_FAILURE);
  } else {
    for(unsigned int i=0; i<aq_histogram_bc_.size(); ++i)
      aq_histogram_bc_[i] = other.AqHistogramBc()[i];
  }
  // Per-base and per-flow error data
  base_position_error_count_.MergeFrom(other.BasePositionErrorCount());
  base_position_.MergeFrom(other.BasePosition());
  flow_position_.MergeFrom(other.FlowPosition());
  per_hp_.MergeFrom(other.PerHp());
  // Read Group per-base and per-flow error data
  for(map< string, ErrorData >::iterator other_it=other.ReadGroupBasePosition().begin(); other_it != other.ReadGroupBasePosition().end(); ++other_it) {
    map< string, ErrorData >::iterator it = read_group_base_position_.find(other_it->first);
    if(it == read_group_base_position_.end()) {
      cerr << "ERROR: unable to merge base position data for read group " << other_it->first << endl;
      exit(EXIT_FAILURE);
    } else {
      read_group_base_position_[it->first].MergeFrom(other_it->second);
    }
  }
  for(map< string, ErrorData >::iterator other_it=other.ReadGroupFlowPosition().begin(); other_it != other.ReadGroupFlowPosition().end(); ++other_it) {
    map< string, ErrorData >::iterator it = read_group_flow_position_.find(other_it->first);
    if(it == read_group_flow_position_.end()) {
      cerr << "ERROR: unable to merge flow position data for read group " << other_it->first << endl;
      exit(EXIT_FAILURE);
    } else {
      read_group_flow_position_[it->first].MergeFrom(other_it->second);
    }
  }
  for(map< string, HpData >::iterator other_it=other.ReadGroupPerHp().begin(); other_it != other.ReadGroupPerHp().end(); ++other_it) {
    map< string, HpData >::iterator it = read_group_per_hp_.find(other_it->first);
    if(it == read_group_per_hp_.end()) {
      cerr << "ERROR: unable to merge per hp data for read group " << other_it->first << endl;
      exit(EXIT_FAILURE);
    } else {
      read_group_per_hp_[it->first].MergeFrom(other_it->second);
    }
  }
  // Regional summary data
  if(regional_summary_.size() != other.GetRegionalSummary().size()) {
    cerr << "ERROR: unable to merge regional summary data, different numbers of regions" << endl;
    exit(EXIT_FAILURE);
  } else {
    for(unsigned int i=0; i<regional_summary_.size(); ++i)
      regional_summary_[i].MergeFrom(other.GetRegionalSummary()[i]);
  }
}
