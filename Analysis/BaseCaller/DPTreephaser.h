/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     DPTreephaser.h
//! @ingroup  BaseCaller
//! @brief    DPTreephaser. Perform dephasing and call base sequence by tree search

#ifndef DPTREEPHASER_H
#define DPTREEPHASER_H

#include <string>
#include <vector>
#include "BaseCallerUtils.h"

using namespace std;


//! @brief    Input/output data structure for DPTreephaser
//! @ingroup  BaseCaller

struct BasecallerRead {
  void SetDataAndKeyNormalize(const float *measurements, int num_flows, const int *key_flows, int num_key_flows);

  float           key_normalizer;           //!< Scaling factor used for initial key normalization
  vector<float>   raw_measurements;         //!< Measured, key-normalized flow signal
  vector<float>   normalized_measurements;  //!< Measured flow signal with best normalization so far
  vector<char>    solution;                 //!< HP-sequence determined by the solver. All entries are integer
  vector<float>   prediction;               //!< Model-based phased signal predicted for the "solved" sequence

  // For QV metrics
  vector<float>   additive_correction;      //!< Additive correction applied to get normalized measurements
  vector<float>   multiplicative_correction;//!< Multiplicative correction applied to get normalized measurements
  vector<float>   state_inphase;            //!< Fraction of live in-phase polymerase
  vector<float>   state_total;              //!< Fraction of live polymerase
  vector<float>   penalty_residual;         //!< Absolute score of the called nuc hypothesis
  vector<float>   penalty_mismatch;         //!< Score difference to second-best nuc hypothesis
};



//! @brief    Performs dephasing and base calling by tree search
//! @ingroup  BaseCaller
//! @details
//! DPTreephaser is responsible for determining base sequence from phased signal output by background model.
//! It uses a generative phasing model that can produce expected signal (prediction) for a partial
//! or complete base sequence. It further uses tree search to find a base sequence with prediction
//! matching the background model signal (measurements) the closest.
//! Additionally, DPTreephaser contains signal normalization procedures that can correct for additive
//! and multiplicative distortion using earlier predicted signal.
//! This allows dephasing and normalization to be performed iteratively (adaptive normalization)

class DPTreephaser {

public:
  //! @brief  Constructor.
  //! @param[in] flow_order   Flow order object, also stores number of flows
  DPTreephaser(const ion::FlowOrder& flow_order);

  //! @brief  Initializes phasing model using specific phasing parameters.
  //! @param[in]  cf          Carry forward rate, how much nuc from previous flow is encountered
  //! @param[in]  ie          Incomplete extension rate, how much polymerase fails to incorporate
  //! @param[in]  dr          Droop, how much polymerase deactivates during an incorporation
  void  SetModelParameters(double cf, double ie, double dr);

  //! @brief  Perform adaptive normalization using WindowedNormalize (slow on long reads)
  //! @param[in,out]  read      Input and output information for the read
  //! @param[in]      max_flows Number of flows to process
  void  NormalizeAndSolve3(BasecallerRead& read, int max_flows);

  //! @brief  Perform adaptive normalization using Normalize (not as accurate)
  //! @param[in,out]  read      Input and output information for the read
  //! @param[in]      max_flows Number of flows to process
  void  NormalizeAndSolve4(BasecallerRead& read, int max_flows);

  //! @brief  Perform adaptive normalization using WindowedNormalize and solving with sliding window
  //! @param[in,out]  read      Input and output information for the read
  //! @param[in]      max_flows Number of flows to process
  void  NormalizeAndSolve5(BasecallerRead& read, int max_flows);

  //! @brief  Tree-search-based dephasing.
  //! @param[in]  read.normalized_measurements    Normalized measurements
  //! @param[out] read.solution   Predicted base sequence in homopolymer space
  //! @param[out] read.prediction Predicted signal
  //! @param[in]  max_flows       Number of flows to process
  //! @param[in]  restart_flows   Number of flows to simulate, rather than solve
  void  Solve(BasecallerRead& read, int max_flows, int restart_flows = 0);

  //! @brief  Generate predicted signal from base sequence
  //! @param[in]  read.solution     Base sequence in homopolymer space
  //! @param[out] read.prediction   Predicted signal
  //! @param[in]  max_flows         Number of flows to process
  void  Simulate(BasecallerRead& read, int max_flows);

  //! @brief  Perform a more advanced simulation to generate QV predictors
  //! @param[in]  read.solution         Base sequence in homopolymer space
  //! @param[out] read.onemer_height    Expected 1-mer signal, used for scaling residuals
  //! @param[out] read.penalty_residual Absolute score of the called nuc hypothesis
  //! @param[out] read.penalty_mismatch Score difference to second-best nuc hypothesis
  int   ComputeQVmetrics(BasecallerRead& read); // Computes "oneMerHeight" and "deltaPenalty"

  //! @brief  Correct for uniform multiplicative scaling
  //! @param[in]  read.prediction               Model-predicted signal
  //! @param[in]  read.raw_measurements         Flow signal before normalization
  //! @param[out] read.normalized_measurements  Flow signal after normalization
  //! @param[in]  start_flow,end_flow           Range of flows to process
  float Normalize(BasecallerRead& read, int start_flow, int end_flow) const;

  //! @brief  Correct for flow-varying multiplicative and additive distortion
  //! @param[in]  read.prediction               Model-predicted signal
  //! @param[in]  read.raw_measurements         Flow signal before normalization
  //! @param[out] read.normalized_measurements  Flow signal after normalization
  //! @param[in]  num_steps                     Number of windows-worth of predictions to use
  //! @param[in]  window_size                   Size of a window in flows
  void  WindowedNormalize(BasecallerRead& read, int num_steps, int window_size) const;


protected:
  //! @brief    Treephaser's slot for partial base sequence, complete with tree search metrics and state for extending
  struct TreephaserPath {
    bool              in_use;                   //!< Is this slot in use?

    // Phasing and input-output state of this path
    int               flow;                     //!< In phase flow of last incorporated base
    vector<float>     state;                    //!< Histogram of flows at which last base was incorporated
    int               window_start;             //!< Start flow (inclusive) of meaningful state values
    int               window_end;               //!< End flow (noninclusive) of meaningful state values
    vector<char>      solution;                 //!< Path sequence in homopolymer space
    vector<float>     prediction;               //!< Model-based phased signal predicted for this path

    // Path metrics and related values
    float             path_metric;              //!< Primary tree search metric
    float             residual_left_of_window;  //!< Residual left of the state window
    float             per_flow_metric;          //!< Auxiliary tree search metric, useful for stack pruning
    int               dot_counter;              //!< Number of extreme mismatch flows encountered so far
  };

  //! @brief  Set path to an empty sequence, a starting point for phasing simulation
  //! @param[out]  state    Path slot
  void InitializeState(TreephaserPath *state) const;

  //! @brief  Perform a path extension by one nucleotide
  //! @param[out]  child     Path slot to store the extended path
  //! @param[in]   parent    Path to be extended
  //! @param[in]   nuc       Nucleotide (integer) to extend the path by
  //! @param[in]   max_flow  Do not read/write past this flow
  void AdvanceState(TreephaserPath *child, const TreephaserPath *parent, int nuc, int max_flow) const;

  //! @brief  Perform a path extension by one nucleotide
  //! @param[in,out] state     Path to be extended in place
  //! @param[in]     nuc       Nucleotide (integer) to extend the path by
  //! @param[in]     max_flow  Do not read/write past this flow
  void AdvanceStateInPlace(TreephaserPath *state, int nuc, int max_flow) const;

  ion::FlowOrder      flow_order_;                //!< Sequence of nucleotide flows
  vector<float>       transition_base_[4];        //!< Probability of polymerase incorporating and staying active
  vector<float>       transition_flow_[4];        //!< Probability of polymerase not incorporating and staying active
  vector<TreephaserPath> path_;                   //!< Preallocated space for partial path slots

  const static int    kNumPaths = 8;              //!< Maximum number of paths considered
  const static float  kExtendThreshold = 0.2;     //!< Threshold for extending paths
  const static float  kNegativeMultiplier = 2.0;  //!< Extra weight on the negative residuals
  const static float  kDotThreshold = 0.3;        //!< percentage of expected Signal that constitutes a "dot"
  const static int    kMaxHP = 11;                //!< Maximum callable homopolymer length
  const static float  kStateWindowCutoff = 1e-6;  //!< Minimum fraction to be taken into account
  const static int    kMaxPathDelay = 40;         //!< Paths that are delayed more are killed

};

#endif // DPTREEPHASER_H
