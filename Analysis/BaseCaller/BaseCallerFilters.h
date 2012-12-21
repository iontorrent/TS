/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerFilters.h
//! @ingroup  BaseCaller
//! @brief    BaseCallerFilters. Filtering and trimming algorithms, configuration, and accounting

#ifndef BASECALLERFILTERS_H
#define BASECALLERFILTERS_H

#include <string>
#include <vector>
#include <set>
#include "json/json.h"

#include "BaseCallerUtils.h"
#include "DPTreephaser.h"
#include "OrderedDatasetWriter.h"
#include "Mask.h"
#include "mixed.h"
#include "OptArgs.h"

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
  BaseCallerFilters(OptArgs& opts, const ion::FlowOrder& flow_order,
      const vector<KeySequence>& keys, const Mask& mask);

  //! @brief    Print usage
  static void PrintHelp();

  //! @brief    Trains polyclonal filter, if necessary.
  //!
  //! @param    output_directory    Directory where optional log files can be placed
  //! @param    wells               Wells file reader object, source of filter training reads
  //! @param    sample_size         Max number of reads to sample for training
  //! @param    mask                Mask for determining which reads are eligible for training set
  void TrainClonalFilter(const string& output_directory, RawWells& wells, int sample_size, Mask& mask);

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


  // *** API for applying filters to individual reads

  //! @brief    Touch a read and mark it as valid. Filters can now be applied to it.
  //! @param    read_index          Read index
  void SetValid                     (int read_index);

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
  void FilterHighPPFAndPolyclonal   (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<float>& measurements);

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

  //! @brief    Apply quality trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void TrimQuality                  (int read_index, int read_class, ReadFilteringHistory& filter_history, const vector<uint8_t>& quality);

  //! @brief    Apply adapter trimmer to a valid read.
  //! @param    read_index          Read index
  //! @param    read_class          Read class, 0=library, 1=TFs
  //! @param    sff_entry           Basecalling results for this read
  void TrimAdapter                  (int read_index, int read_class, ProcessedRead& processed_read, const vector<float>& scaled_residual,
                                     const vector<int>& base_to_flow, DPTreephaser& treephaser, const BasecallerRead& read);

  //! @brief    Check if a read is marked as valid.
  //! @param    read_index          Read index
  bool IsValid(int read_index) const;

  //! @brief    Check if a read is marked as polyclonal.
  //! @param    read_index          Read index
  bool IsPolyclonal(int read_index) const;



protected:

  // General information
  ion::FlowOrder      flow_order_;                        //!< Flow order object, also stores number of flows
  vector<int>         filter_mask_;                       //!< Vector indicating filtering decision for each well on the chip
  int                 num_classes_;                       //!< Number of read classes. Currently must be 2: library and TFs.
  vector<KeySequence> keys_;                              //!< Key sequences for read classes.
  bool                generate_bead_summary_;             //!< If true, beadSummary.filtered.txt will be generated.

  // Primary filters
  bool                filter_keypass_enabled_;            //!< Is keypass filter enabled?
  int                 filter_min_read_length_;            //!< If basecaller produces read shorter than this, the read is filtered
  bool                filter_residual_enabled_;           //!< Is residual filter enabled for library reads?
  bool                filter_residual_enabled_tfs_;       //!< Is residual filter enabled for TFs?
  double              filter_residual_max_value_;         //!< Residual filter threshold
  bool                filter_clonal_enabled_;             //!< Is polyclonal filter enabled for library reads?
  bool                filter_clonal_enabled_tfs_;         //!< Is polyclonal filter enabled for TFs?
  clonal_filter       clonal_population_;                 //!< Object implementing clonal filter

  // Beverly filter
  bool                filter_beverly_enabled_;            //!< Is Beverly filter enabled?
  float               filter_beverly_filter_ratio_;       //!< Fraction of  one-plus-two-mer outliers before Beverly filter considers trimming
  float               filter_beverly_trim_ratio_;         //!< Fraction of onemer outliers before Beverly filter trims
  int                 filter_beverly_min_read_length_;    //!< If Beverly filter trims and makes the read shorter than this, the read is filtered

  // Adapter and quality trimming
  string              trim_adapter_;                      //!< Adapter sequence
  double              trim_adapter_cutoff_;               //!< Adapter detection threshold
  int                 trim_adapter_min_match_;            //!< Minimum number of overlapping adapter bases for detection
  int                 trim_adapter_mode_;                 //!< Selects algorithm and metric used for adapter detection
  int                 trim_qual_window_size_;             //!< Size of averaging window used by quality trimmer
  double              trim_qual_cutoff_;                  //!< Quality cutoff used by quality trimmer
  int                 trim_min_read_len_;                 //!< If adapter or quality trimming makes the read shorter than this, the read is filtered

};



#endif // BASECALLERFILTERS_H
