/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerFilters.h
//! @ingroup  BaseCaller
//! @brief    BaseCallerFilters. Filtering and trimming algorithms, configuration, and accounting

#ifndef BASECALLERFILTERS_H
#define BASECALLERFILTERS_H

#include <string>
#include <vector>
#include "json/json.h"

#include "BaseCallerUtils.h"
#include "Mask.h"
#include "mixed.h"
#include "OptArgs.h"

class  DPTreephaser;
struct BasecallerRead;
struct ProcessedRead;
struct ReadFilteringHistory;

using namespace std;



//! @brief    Filtering+trimming algorithms, configuration, and accounting
//! @ingroup  BaseCaller
//! @details
//! BaseCallerFilters contains implementations of all read filters applied by BaseCaller.
//! It knows which filters are enabled/disabled and what are their parameters/thresholds.
//! It also keeps track of filtering outcomes for all reads and can produce
//! detailed statistics and update Mask.

class BaseCallerFilters {
public:
  // *** API for Setup and final results

  //! @brief    Constructor.
  //!
  //! @param    opts                Command line options
  //! @param    flow_order          Flow order object, also stores number of flows
  //! @param    keys                Key sequences in use
  //! @param    mask                Mask object
  BaseCallerFilters(OptArgs& opts, vector<string> & bam_comments, const string & run_id,
         const ion::FlowOrder& flow_order, const vector<KeySequence>& keys, const Mask& mask);

  //! @brief    Print usage
  static void PrintHelp();

  //! @brief    Trains polyclonal filter, if necessary.
  //!
  //! @param    output_directory    Directory where optional log files can be placed
  //! @param    wells               Wells file reader object, source of filter training reads
  //! @param    max_sample_size     Max number of reads to sample for training
  //! @param    mask                Mask for determining which reads are eligible for training set
  //! @param    opts                User options for polyclonal filter and flows
  void TrainClonalFilter(const string& output_directory, RawWells& wells, Mask& mask, const PolyclonalFilterOpts & opts);

  //! @brief    Once filtering is complete, transfer filtering outcomes to Mask object.
  //!
  //! @param    mask                Mask object to which filtering results will be added
  void TransferFilteringResultsToMask(Mask &mask) const;

  //! @brief    Helper function calculating median absolute value of residuals, over a range of flows.
  //!
  //! @param    residual            Vector of residual values
  //! @param    use_flows           Max number of flows to use in the calculation
  static double MedianAbsoluteCafieResidual(const vector<float> &residual, unsigned int use_flows);

  //! @brief    Helper function returning total number of visited reads.
  //! @return   Number of visited reads, filtered or unfiltered
  int NumWellsCalled() const;

  //! @ brief   Provides access to the (3') library bead adapters
  const vector<string> & GetLibBeadAdapters() const {return trim_adapter_;};

  //! @ brief   Provides access to the (3') test-fragment bead adapters
  const vector<string> & GetTFBeadAdapters() const {return trim_adapter_tf_;};


  // *** API for applying filters to individual reads

  //! @brief    Touch a read and mark it as valid. Filters can now be applied to it.
  //! @param    read_index          Read index
  void SetValid                     (int read_index);

  void SetFiltered                  (int read_index, int read_class, ReadFilteringHistory& filter_history);

  //! @brief    Unconditionally mark a valid read as polyclonal, as determined in background model.
  //! @param    read_index          Read index
  void SetBkgmodelPolyclonal        (int read_index, ReadFilteringHistory& filter_history);

  //! @brief    Unconditionally mark a valid read as high percent positive flows, as determined in background model.
  //! @param    read_index          Read index
  void SetBkgmodelHighPPF           (int read_index, ReadFilteringHistory& filter_history);

  //! @brief    Unconditionally mark a valid read as failed keypass, as determined in background model.
  //! @param    read_index          Read index
  void SetBkgmodelFailedKeypass     (int read_index, ReadFilteringHistory& filter_history);

  //! @brief    Apply polyclonal and ppf filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    measurements        Key-normalized flow signal from wells
  //! @param    opts                User options for how filters work
  void FilterHighPPFAndPolyclonal   (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& measurements, const PolyclonalFilterOpts & opts);

  //! @brief    Apply zero-length filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void FilterZeroBases              (int read_index, int read_class, ReadFilteringHistory& filter_history);

  //! @brief    Apply short-length filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void FilterShortRead              (int read_index, int read_class, ReadFilteringHistory& filter_history);

  //! @brief    Apply keypass filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    solution            Homopolymer calls for this read
  void FilterFailedKeypass          (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<char>& sequence);

  //! @brief    Apply high-residual filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    residual            Vector of phasing residuals
  void FilterHighResidual           (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& residual);

  //! @brief    Apply Beverly trimmer and filter to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    read                Signals for this read
  //! @param    sff_entry           Basecalling results for this read
  void FilterBeverly                (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& scaled_residual,
                                     const vector<int>& base_to_flow);

  //! @brief    Entry point for quality trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void FilterQuality                (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Entry point for quality trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void TrimQuality                  (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Sliding window quality trimming algorithm.
  //! @param    read_index          Read index
  //! @param    sff_entry           Basecalling results for this read
  //! @param    return value        Read trimming point
  int TrimQuality_Windowed          (int read_index, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Trimming by expected number of errors calculated from quality values.
  //! @param    read_index          Read index
  //! @param    sff_entry           Basecalling results for this read
  //! @param    return value        Read trimming point
  int TrimQuality_ExpectedErrors    (int read_index, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Apply adapter trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void TrimAdapter                  (int read_index, int read_class, ProcessedRead& processed_read, const vector<float>& scaled_residual,
                                     const vector<int>& base_to_flow, DPTreephaser& treephaser, const BasecallerRead& read);

  //! @brief    setup Avalanche trimmer parameters
  //! @param    maxflows            max num flows
  void TrimAvalanche_setup          (int maxflows);

  //! @brief    Apply Avalanche trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void TrimAvalanche                (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Check if a read is marked as valid.
  //! @param    read_index          Read index
  bool IsValid(int read_index) const;

  //! @brief    Check if a read is marked as polyclonal.
  //! @param    read_index          Read index
  bool IsPolyclonal(int read_index) const;


protected:

  // Write the bead adapters to comments for BAM header
  void WriteAdaptersToBamComments(vector<string> &comments, const string & run_id);

  //! @brief    Check input strings from non-ACGT characters
  void ValidateBaseStringVector(vector<string>& string_vector);

  //! @brief    Adapter trimmer using the predicted signal to determine the adapter position
  bool TrimAdapter_PredSignal(float& best_metric, int& best_start_flow, int& best_start_base, int& best_adapter_overlap,
           const string& effective_adapter, DPTreephaser& treephaser, const BasecallerRead& read);

  //! @brief    Adapter trimmer using a flowspace alignment to determine the adapter position
  bool TrimAdapter_FlowAlign(float& best_metric, int& best_start_flow, int& best_start_base, int& best_adapter_overlap,
  		   const string& effective_adapter, const vector<float>& scaled_residual,
  		   const vector<int>& base_to_flow, const BasecallerRead& read);

  // General information
  ion::FlowOrder      flow_order_;                        //!< Flow order object, also stores number of flows
  vector<int>         filter_mask_;                       //!< Vector indicating filtering decision for each well on the chip
  int                 num_classes_;                       //!< Number of read classes. Currently must be 2: library and TFs.
  vector<KeySequence> keys_;                              //!< Key sequences for read classes.

  // Primary filters
  bool                filter_keypass_enabled_;            //!< Is keypass filter enabled?
  int                 filter_min_read_length_;            //!< If basecaller produces read shorter than this, the read is filtered
  bool                filter_residual_enabled_;           //!< Is residual filter enabled for library reads?
  bool                filter_residual_enabled_tfs_;       //!< Is residual filter enabled for TFs?
  double              filter_residual_max_value_;         //!< Residual filter threshold
  bool                filter_clonal_enabled_;             //!< Is polyclonal filter enabled for library reads?
  bool                filter_clonal_enabled_tfs_;         //!< Is polyclonal filter enabled for TFs?
  int                 filter_clonal_maxreads_;            //!< Number of reads to be used for clonal filter training
  clonal_filter       clonal_population_;                 //!< Object implementing clonal filter
  int                 extra_trim_right_;                  //!< Make quality trimming delete some extra bases

  // Beverly filter
  bool                filter_beverly_enabled_;            //!< Is Beverly filter enabled?
  vector<double>      filter_beverly_filter_trim_ratio_;  //!< Fractions of one-plus-two-mer outliers, onemer outliers before Beverly filter trims

  // Quality Filter
  bool                filter_quality_enabled_;            //!< Is quality read filter enabled?
  double              filter_quality_offset_;             //!< Errors allowed per base for filtering based on expected errors
  double              filter_quality_slope_;              //!< Error offset for filtering based on expected errors
  double              filter_quality_quadr_;              //!< Extra Error offset for filtering based on expected errors


  // Adapter and quality trimming
  int                 trim_min_read_len_;                 //!< If adapter or quality trimming makes the read shorter than this, the read is filtered

  vector<string>      trim_adapter_;                      //!< Adapter sequences
  double              trim_adapter_cutoff_;               //!< Adapter detection threshold
  double              trim_adapter_separation_;           //!< Minimum separation between found adapter sequences
  int                 trim_adapter_min_match_;            //!< Minimum number of overlapping adapter bases for detection
  int                 trim_adapter_mode_;                 //!< Selects algorithm and metric used for adapter detection
  vector<string>      trim_adapter_tf_;                   //!< Test Fragment adapter sequences. If empty, do not perform adapter trimming on TFs.

  string              trim_qual_mode_;                    //!< Set quality trimming mode
  int                 trim_qual_mode_enum_;               //!< Enumerator type of quality trimming mode
  int                 trim_qual_window_size_;             //!< Size of averaging window used by sliding window quality trimmer
  double              trim_qual_cutoff_;                  //!< Quality cutoff used by sliding window quality trimmer
  double              trim_qual_slope_;                   //!< Expected number of errors allowed per base for expected error qv trimmer
  double              trim_qual_offset_;                  //!< Offset for expected errors to allow for variation in expected qv trimmer
  double              trim_qual_quadr_;                   //!< Extra  expected errors to allow for variation in expected qv trimmer

  // Avalanche filter (sort readlength filter, higher QV on shorter reads, and lower QV for longer reads)
  bool                filter_avalanche_enabled_;          //!< Is Avalanche filter enabled?
  int                 avalanche_max_pos_;                 //!< max QV threshold pos
  int                 avalanche_mid_pos_;                 //!< mid QV threshold pos
  int                 avalanche_min_pos_;                 //!< min QV threshold pos
  int                 avalanche_win_;                     //!< Avalanche window used by avalanche trimmer
  double              trim_qual_avalanche_max_;           //!< Quality cutoff used by avalanche trimmer
  double              trim_qual_avalanche_hi_;            //!< Quality cutoff used by avalanche trimmer
  double              trim_qual_avalanche_lo_;            //!< Quality cutoff used by avalanche trimmer
  double              avalanche_sum_delta_;              //!<delta subtracted from minimum_sum before avalanche_mid_pos_
  double              avalanche_ava_delta_;              //!<delta subtracted from minimum_ava after avalanche_mid_pos_
  double              avalanche_min_delta_;              //!<delta subtracted from minimum_min after avalanche_min_pos_
  double              delta_ava_;                        //!<delta between avalanche_ava_delta_ & avalanche_sum_delta_
  double              delta_min_;                        //!<delta between avalanche_min_delta_ & avalanche_ava_delta_
  int                 trim_min_read_len_avalanch_;       //!< diff from trim_min_read_len_
  int                 qv_sum_thresh_max;                 //!< threshold on the sum of QV for pos < trim_qual_avalanche_max_
  int                 qv_sum_thresh_min;                 //!< threshold on the sum of QV for pos >= trim_qual_avalanche_lo_
  int                 min_bases_passed_test;            //!< min number of bases that the reads have to pass the qv test
  vector<double>      qv_sum_thresh;                     //!< threshold on the sum of QV
  //long                count_filtered;                    //!<counter mainly for debugging
  //long                count_trimmed;                     //!<counter mainly for debugging
  //long                count_passed;                      //!<counter mainly for debugging

};



#endif // BASECALLERFILTERS_H
