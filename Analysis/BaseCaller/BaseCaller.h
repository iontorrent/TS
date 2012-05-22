/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BASECALLER_H
#define BASECALLER_H

#include <string>
#include <vector>
#include "json/json.h"

#include "Mask.h"
#include "RawWells.h"
#include "CommandLineOpts.h"
#include "mixed.h"
#include "TrackProgress.h"
#include "PerBaseQual.h"
#include "OrderedRegionSFFWriter.h"
#include "OptArgs.h"
#include "PhaseEstimator.h"
#include "BaseCallerUtils.h"

using namespace std;



// The max number of flows to be evaluated for "percent positive flows" (ppf) metric
#define PERCENT_POSITIVE_FLOWS_N 60
// The max number of flows to be evaluated for Cafie residual metrics
#define CAFIE_RESIDUAL_FLOWS_N 60

typedef uint32_t     well_index_t;  // To hold well indexes.  32-bit ints will allow us up to 4.3G wells



// BaseCallerFilters keeps track of filter-related stuff, like:
//   - which filters are enabled/disabled
//   - filter parameters/thresholds
//   - statistics of which filters were used

class BaseCallerFilters {
public:
  BaseCallerFilters(OptArgs& opts, const string& _flowOrder, int _numFlows, const vector<KeySequence>& _keys,
      Mask *_maskPtr);

  static void PrintHelp();

  void TransferFilteringResultsToMask(Mask *myMask);
  void GenerateFilteringStatistics(Json::Value &filterSummary);

  // Filter settings:
  double cafieResMaxValue;
  bool keypassFilter;
  int minReadLength;
  bool clonalFilterSolving;
  bool clonalFilterTraining;
  bool cafieResFilterCalling;
  bool percentPositiveFlowsFilterTFs;
  bool cafieResFilterTFs;

  bool generate_bead_summary_;

  // Filtering flags and stats
  int                     numClasses;
  vector<KeySequence>     keys;

  vector<bool>                    classFilterPolyclonal;
  vector<bool>                    classFilterHighResidual;

  // Beverly filter
  bool                filter_beverly_enabled_;            //!< Is Beverly filter enabled?
  float               filter_beverly_filter_ratio_;       //!< Fraction of  one-plus-two-mer outliers before Beverly filter considers trimming
  float               filter_beverly_trim_ratio_;         //!< Fraction of onemer outliers before Beverly filter trims
  int                 filter_beverly_min_read_length_;    //!< If Beverly filter trims and makes the read shorter than this, the read is filtered

  // SFFTrim
  string  trim_adapter;
  double  trim_adapter_cutoff;
  bool    trim_adapter_closest;
  int     trim_qual_wsize;
  double  trim_qual_cutoff;
  int     trim_min_read_len;

  Mask  *maskPtr;
  int numFlows;
  string flowOrder;

  clonal_filter   clonalPopulation;

  vector<int>     filterMask;


  void FindClonalPopulation(const string& outputDirectory, RawWells *wellsPtr, int nUnfilteredLib);

  void markReadAsValid(int readIndex);
  bool isValid(int readIndex);
  bool isPolyclonal(int readIndex);

  void forceBkgmodelPolyclonal          (int readIndex);
  void forceBkgmodelHighPPF             (int readIndex);
  void forceBkgmodelFailedKeypass       (int readIndex);

  void tryFilteringHighPPFAndPolyclonal (int readIndex, int iClass, const vector<float>& measurements);
  void tryFilteringZeroBases            (int readIndex, int iClass, const SFFWriterWell& readResults);
  void tryFilteringShortRead            (int readIndex, int iClass, const SFFWriterWell& readResults);
  void tryFilteringFailedKeypass        (int readIndex, int iClass, const vector<char> &solution);
  void tryFilteringHighResidual         (int readIndex, int iClass, const vector<float>& residual);
  void tryFilteringBeverly              (int readIndex, int iClass, const BasecallerRead &read, SFFWriterWell& readResults);

  void tryTrimmingQuality               (int readIndex, int iClass, SFFWriterWell& readResults);
  void tryTrimmingAdapter               (int readIndex, int iClass, SFFWriterWell& readResults);

  int getNumWellsCalled();

  static double getMedianAbsoluteCafieResidual(const vector<weight_t> &residual, unsigned int nFlowsToAssess);

};





struct BaseCaller {

  // General run parameters
  string                  run_id;
  string                  dephaser;
  string                  phred_table_file;
  string                  output_directory;
  string                  filename_wells;
  ChipIdEnum              chip_id;
  int                     chip_size_y;
  int                     chip_size_x;
  int                     region_size_x;
  int                     region_size_y;
  int                     num_flows;
  string                  flow_order;
  vector<KeySequence>     keys;

  // Important outside entities accessed by BaseCaller
  Mask                    *mask;
  BaseCallerFilters       *filters;
  PhaseEstimator          estimator;
  vector<int>             class_map;

  // Threaded processing
  pthread_mutex_t         mutex;
  int                     next_region;
  int                     next_begin_x;
  int                     next_begin_y;

  // Basecalling results saved here
  RawWells                *residual_file;
  FILE                    *well_stat_file;
  OrderedRegionSFFWriter  lib_sff;
  OrderedRegionSFFWriter  tf_sff;
  set<well_index_t>       unfiltered_set;
  OrderedRegionSFFWriter  unfiltered_sff;
  OrderedRegionSFFWriter  unfiltered_trimmed_sff;

  void BasecallerWorker();
};


FILE * OpenWellStatFile(const string& wellStatFile);
void WriteWellStatFileEntry(FILE *wellStatFileFP, const KeySequence& key,
    SFFWriterWell & readResults, BasecallerRead & read, weight_vec_t & residual,
    int x, int y, double cf, double ie, double dr, bool clonal);


#endif // BASECALLER_H
