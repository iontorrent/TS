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

#include "json/json.h"


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
  //! @param[in]  input_directory     Directory where 1.wells could be found
  //! @param[in]  output_directory    Directory where predictor dump file may be saved
  void Init(OptArgs& opts, const string& chip_type, const string &input_directory, const string &output_directory, bool recalib);

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
      const vector<float> &predictor1, const vector<float> &predictor2, const vector<float> &predictor3,
      const vector<float> &predictor4, const vector<float> &predictor5, const vector<float> &predictor6,
      const vector<int>& base_to_flow, vector<uint8_t> &quality,
      const vector<float> &candidate1, const vector<float> &candidate2, const vector<float> &candidate3,
      const vector<float> &predictor1_flow, const vector<float> &predictor5_flow, const vector<float> &predictor4_flow,
      const vector<int>& flow_to_base, const bool flow_predictors_=false);

  void DumpPredictors(const string& read_name, int num_bases, int num_flows,
      const vector<float> &predictor1, const vector<float> &predictor2, const vector<float> &predictor3,
      const vector<float> &predictor4, const vector<float> &predictor5, const vector<float> &predictor6,
      const vector<int>& base_to_flow, vector<uint8_t> &quality,
      const vector<float> &candidate1, const vector<float> &candidate2, const vector<float> &candidate3,
      const vector<float> &predictor1_flow, const vector<float> &predictor5_flow, const vector<float> &predictor4_flow,
      const vector<int>& flow_to_base, const bool flow_predictors_=false);

  //! @brief  Calculate Local Noise predictor for all bases in a read
  //! @param[out] local_noise               Local Noise predictor
  //! @param[in]  max_base                  Number of bases for which predictor should be calculated
  //! @param[in]  base_to_flow              Flow number corresponding to each called base
  //! @param[in]  normalized_measurements   Normalized flow signal from wells file
  //! @param[in]  prediction                Model-predicted flow signal
  static void PredictorLocalNoise(vector<float>& local_noise, int max_base, const vector<int>& base_to_flow,
      const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_);

  //! @brief  Calculate Noise Overlap predictor shared by all bases in a read
  //! @param[out] minus_noise_overlap       Noise Overlap predictor
  //! @param[in]  max_base                  Number of bases for which predictor should be calculated
  //! @param[in]  normalized_measurements   Normalized flow signal from wells file
  //! @param[in]  prediction                Model-predicted flow signal
  //! @return Noise Overlap predictor
  static void PredictorNoiseOverlap(vector<float>& minus_noise_overlap, int max_base,
      const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_);

  //! @brief  Calculate Homopolymer Rank predictor for all bases in a read
  //! @param[out] homopolymer_rank          Homopolymer Rank predictor
  //! @param[in]  max_base                  Number of bases for which predictor should be calculated
  //! @param[in]  sequence                  Called bases
  static void PredictorHomopolymerRank(vector<float>& homopolymer_rank, int max_base, const vector<char>& sequence, vector<float>& homopolymer_rank_flow, const vector<int>& flow_to_base, int flow_predictors_=false);

  //! @brief  Calculate Neighborhood Noise predictor for all bases in a read
  //! @param[out] neighborhood_noise        Neighborhood Noise predictor
  //! @param[in]  max_base                  Number of bases for which predictor should be calculated
  //! @param[in]  base_to_flow              Flow number corresponding to each called base
  //! @param[in]  normalized_measurements   Normalized flow signal from wells file
  //! @param[in]  prediction                Model-predicted flow signal
  static void PredictorNeighborhoodNoise(vector<float>& neighborhood_noise, int max_base, const vector<int>& base_to_flow,
      const vector<float>& normalized_measurements, const vector<float>& prediction, const bool flow_predictors_);


  static void PredictorBeverlyEvents(vector<float>& beverly_events, int max_base, const vector<int>& base_to_flow,
      const vector<float>& scaled_residual, const bool flow_predictors_);

  bool toSavePredictors() {return (save_predictors_ ? true:false);}

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

  vector<vector<float> >  phred_cuts_;				  //!< Predictor threshold table, kNumPredictors x num_phred_cuts.
  unsigned char*		  phred_table_;				  //!< Predictor table of QV values.
  vector<size_t>		  offsets_;					  //!< Indexing offsets.

  bool                    save_predictors_;           //!< If true, dump predictor values for each processed read and base
  ofstream                predictor_dump_;            //!< File to which predictor values are dumped
  pthread_mutex_t         predictor_mutex_;           //!< Mutex protecting writes to predictor dump file
  //string                  enzyme_name_;               //!< Name of the "enzyme"

private:
  float transform_P1(float p);
  float transform_P2(float p);
  float transform_P5(float p);
  float transform_P5_v34(float p);
  float transform_P6(float p);
  float transform_P7(float p);
  float transform_P8(float p);
  float transform_P9(float p);

public:
  bool hasBinaryExtension(string &filename);
  char *get_KnownAlternate_PhredTable(string chip_type, bool recalib, string enzymeName="", bool binTable=true);
  char *get_phred_table_name(string chip_type, bool recalib, string enzymeName="");

private:
  string add_Recal_to_phredTableName(string phred_table_file, bool recalib=true);
  bool startswith(const string &fullString, const string &teststring);
  bool endswith(const string &fullString, const string &teststring);
  bool contains(const string &fullString, const string &teststring);
};



#endif // PERBASEQUAL_H
