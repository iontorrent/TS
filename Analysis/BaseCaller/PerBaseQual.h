/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PerBaseQual.h
//! @ingroup  BaseCaller
//! @brief    PerBaseQual. Determination of base qualities from predictors

#ifndef PERBASEQUAL_H
#define PERBASEQUAL_H

#include <string>
#include <vector>
#include <fstream>
#include <stdint.h>
#include "OptArgs.h"

using namespace std;

//! @brief    Determination of base qualities from predictors
//! @ingroup  BaseCaller
//! @details
//! PerBaseQual assigns quality values to bases produced by the BaseCaller.
//! For each base it assembles a set of predictor values, some passed directly from
//! DPTreephaser, some custom-calculated from other outputs.
//! The predictors are then compared against rows of a phred table loaded from an outside file.
//! The earliest row for which all predictors exceed the thresholds contains the quality value for the base.

class PerBaseQual {
public:
  //! Constructor.
  PerBaseQual();
  //! Destructor.
  ~PerBaseQual();

  //! @brief  Print usage
  static void PrintHelp();

  //! @brief  Initialize the object and load phred table.
  //! @param[in]  opts                Command line options
  //! @param[in]  chip_type           Chip type, may determine default phred table
  //! @param[in]  output_directory    Directory where predictor dump file may be saved
  void Init(OptArgs& opts, const string& chip_type, const string &output_directory);

  //! @brief  Generate quality values for all bases in a read.
  //! @param[in]  read_name           Read name used in predictor dump
  //! @param[in]  num_bases           Number of bases that need quality values calculated
  //! @param[in]  num_flows           Number of flows. Predictors are not available beyond this base
  //! @param[in]  predictor1          Vector of quality value predictors no.1
  //! @param[in]  predictor2          Vector of quality value predictors no.2
  //! @param[in]  predictor3          Vector of quality value predictors no.3
  //! @param[in]  predictor4          Vector of quality value predictors no.4
  //! @param[in]  predictor5          Vector of quality value predictors no.5
  //! @param[in]  predictor6          Vector of quality value predictors no.6
  //! @param[in]  base_to_flow        Flow number corresponding to each called base
  //! @param[out] quality             Quality values
  void GenerateBaseQualities(const string& read_name, int num_bases, int num_flows,
      const vector<float> &predictor1, const vector<float> &predictor2,
      const vector<float> &predictor3, const vector<float> &predictor4,
      const vector<float> &predictor5, const vector<float> &predictor6,
      const vector<int>& base_to_flow, vector<uint8_t> &quality,
      const vector<float> &candidate1,
      const vector<float> &candidate2,
      const vector<float> &candidate3);

  //! @brief  Calculate Local Noise predictor for all bases in a read
  //! @param[out] local_noise         Local Noise predictor
  //! @param[in]  max_base            Number of bases for which predictor should be calculated
  //! @param[in]  base_to_flow        Flow number corresponding to each called base
  //! @param[in]  corrected_ionogram  Estimated ionogram after dephasing
  static void PredictorLocalNoise(vector<float>& local_noise, int max_base, const vector<int>& base_to_flow,
      const vector<float>& corrected_ionogram);

  //! @brief  Calculate Noise Overlap predictor shared by all bases in a read
  //! @param[in]  corrected_ionogram  Estimated ionogram after dephasing
  //! @return Noise Overlap predictor
  static void PredictorNoiseOverlap(vector<float>& minus_noise_overlap, int max_base, const vector<float>& corrected_ionogram);

  //! @brief  Calculate Homopolymer Rank predictor for all bases in a read
  //! @param[out] homopolymer_rank    Homopolymer Rank predictor
  //! @param[in]  max_base            Number of bases for which predictor should be calculated
  //! @param[in]  flow_index          Flow increment for each base
  static void PredictorHomopolymerRank(vector<float>& homopolymer_rank, int max_base, const vector< uint8_t >& flow_index);

  //! @brief  Calculate Neighborhood Noise predictor for all bases in a read
  //! @param[out] neighborhood_noise  Neighborhood Noise predictor
  //! @param[in]  max_base            Number of bases for which predictor should be calculated
  //! @param[in]  base_to_flow        Flow number corresponding to each called base
  //! @param[in]  corrected_ionogram  Estimated ionogram after dephasing
  static void PredictorNeighborhoodNoise(vector<float>& neighborhood_noise, int max_base, const vector<int>& base_to_flow,
      const vector<float>& corrected_ionogram);

protected:

  //! @brief  Use phred table to determine quality value from predictors
  //! @param[in]  pred                Array of predictor values. May be modified in place
  //! @return Quality value
  uint8_t CalculatePerBaseScore(float* pred) const;

  const static int        kNumPredictors = 6;         //!< Number of predictors used for quality value determination
  const static int        kMinQuality = 5;            //!< Lowest possible quality value

  vector<vector<float> >  phred_thresholds_;          //!< Predictor threshold table, kNumPredictors x num_phred_cuts.
  vector<float>           phred_thresholds_max_;      //!< Maximum threshold for each predictor
  vector<uint8_t>         phred_quality_;             //!< Quality value associated with each predictor cut.

  bool                    save_predictors_;           //!< If true, dump predictor values for each processed read and base
  ofstream                predictor_dump_;            //!< File to which predictor values are dumped
  pthread_mutex_t         predictor_mutex_;           //!< Mutex protecting writes to predictor dump file


private:
  float transform_P1(float p);
  float transform_P5(float p);
  float transform_P6(float p);
  float transform_P7(float p);
  float transform_P8(float p);
  float transform_P9(float p);

};



#endif // PERBASEQUAL_H
