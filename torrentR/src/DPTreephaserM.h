/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     DPTreephaserM.h
//! @ingroup  BaseCaller
//! @brief    DPTreephaserM. Perform dephasing and call base sequence by tree search on potentially more than one raw signal jointly

#ifndef DPTREEPHASERM_H
#define DPTREEPHASERM_H

#include <string>
#include <vector>
#include "BaseCallerUtils.h"
#include "DPTreephaser.h"
#include <Rcpp.h>

using namespace std;

// ---------------------------------------------------------------------------
//! @brief    Input/output data structure for DPTreephaserM - storing multiple reads of the same sequence
//! @ingroup  BaseCaller
struct BasecallerMultiRead {

	vector<int>              active_until_flow;  //!< Flow value until a read is being used (non-inclusive)
	vector<BasecallerRead>   read_vector;        //!< Vector of Basecaller reads
	vector<char>             solution;           //!< Base sequence called
	unsigned int             bases_called;       //!< Number of bases called
};


// ---------------------------------------------------------------------------
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

class DPTreephaserM {

public:
  //! @brief  Constructor.
  //! @param[in] flow_order   vector of flow order objects, one for each read to be solved jointly
  DPTreephaserM(const vector<ion::FlowOrder> &flow_orders);

  //! @brief  Initializes phasing model using specific phasing parameters.
  //! @param[in]  cf          Carry forward rate, how much nuc from previous flow is encountered
  //! @param[in]  ie          Incomplete extension rate, how much polymerase fails to incorporate
  //! @param[in]  dr          Droop, how much polymerase deactivates during an incorporation
  void  SetPhasingParameters(vector<double> cf, vector<double> ie, vector<double> dr);

  //! @brief  Perform adaptive normalization using WindowedNormalize (slow on long reads)
  //! @param[in,out]  read      Input and output information for the read
  //! @param[in]      max_flows Number of flows to process
  void  NormalizeAndSolveAN(BasecallerMultiRead& read, int max_flows);

  //! @brief  Perform adaptive normalization using WindowedNormalize and solving with sliding window
  //! @param[in,out]  read      Input and output information for the read
  //! @param[in]      max_flows Number of flows to process
  void  NormalizeAndSolveSWAN(BasecallerMultiRead& read, int max_flows);

  //! @brief  Tree-search-based dephasing.
  //! @param[in]  read.normalized_measurements    Normalized measurements
  //! @param[out] read.solution   Predicted base sequence in homopolymer space
  //! @param[out] read.prediction Predicted signal
  //! @param[in]  max_flows       Number of flows to process
  //! @param[in]  restart_flows   Number of flows to simulate, rather than solve
  void  Solve(BasecallerMultiRead& read, int max_flows, int restart_flows = 0);

  //! @brief  Generate predicted signal from base sequence
  //! @param[in]  read.solution     Base sequence in homopolymer space
  //! @param[out] read.prediction   Predicted signal
  //! @param[in]  max_flows         Number of flows to process
  //void  Simulate(BasecallerMultiRead& read, int max_flows);

  //! @brief  Perform a more advanced simulation to generate QV predictors
  //! @param[in]  read.solution         Base sequence in homopolymer space
  //! @param[out] read.onemer_height    Expected 1-mer signal, used for scaling residuals
  //! @param[out] read.penalty_residual Absolute score of the called nuc hypothesis
  //! @param[out] read.penalty_mismatch Score difference to second-best nuc hypothesis
  // int   ComputeQVmetrics(BasecallerRead& read); // Computes "oneMerHeight" and "deltaPenalty"

  //! @brief  Correct for flow-varying multiplicative and additive distortion
  //! @param[in]  read.prediction               Model-predicted signal
  //! @param[in]  read.raw_measurements         Flow signal before normalization
  //! @param[out] read.normalized_measurements  Flow signal after normalization
  //! @param[in]  num_steps                     Number of windows-worth of predictions to use
  //! @param[in]  window_size                   Size of a window in flows
  void  WindowedNormalize(BasecallerRead& read,  int flow_limit, int num_flows, int num_steps, int window_size) const;

  int verbose; // Sets whether, and how much, DPTreePhaser should print to screen


protected:
  //! @brief    Treephaser's slot for partial base sequence, complete with tree search metrics and state for extending
  struct TreephaserPath {
	int            active_until_flow;                //!< Flow number marking the end of the period where the read is used

	// Phasing and input-output state of this path, one per read
    int             flow;                     //!< In phase flow of last incorporated base *
    vector<float>   state;                    //!< Histogram of flows at which last base was incorporated *
    int             window_start;             //!< Start flow (inclusive) of meaningful state values *
    int             window_end;               //!< End flow (noninclusive) of meaningful state values *
    vector<float>   prediction;               //!< Model-based phased signal predicted for this path *

    // Path metrics and related values
    float           read_path_metric;         //!< Tree search metric for individual read *
    float           residual_left_of_window;  //!< Residual left of the state window *
    float           read_per_flow_metric;     //!< Auxiliary tree search metric for individual read *
    int             dot_counter;              //!< Number of extreme mismatch flows encountered so far *
  };

  struct TreephaserMultiPath {
	  bool                    path_in_use;              //!< Is this slot in use? *

	  int                     num_active_reads;         //!< Number of active reads for this path segment *
	  int                     min_flow;                 //!< Smallest value of active incorporating flows *
	  int                     max_flow;                 //!< Largest value of active incorporating flows *
	  vector<TreephaserPath>  read_data_;               //!< state information for each read
	  vector<char>            solution;                 //!< Path sequence in BASE space
	  unsigned int            bases_called;             //!< Number of bases that have been called in that path *
	  int                     current_hp;               //!< Value of current Homopolymer being processed *
	  float                   path_metric;              //!< Primary tree search metric
	  float                   per_flow_metric;          //!< Auxiliary tree search metric, useful for stack pruning
  };

  //! @brief  Set path to an empty sequence, a starting point for phasing simulation
  //! @param[out]  state    Path slot
  void InitializeState(TreephaserMultiPath *state) const;

  //! @brief  Perform a path extension by one nucleotide for one read
  //! @param[out]  child     Path slot to store the extended path
  //! @param[in]   parent    Path to be extended
  //! @param[in]   nuc       Nucleotide (integer) to extend the path by
  //! @param[in]     max_flow  Do not read/write past this flow
  void AdvanceStates(TreephaserMultiPath *child, const TreephaserMultiPath *parent, int nuc, int max_flow) const;

  //! @brief  Perform a path extension by one nucleotide
  //! @param[in,out] state     Path to be extended in place
  //! @param[in]     nuc       Nucleotide (integer) to extend the path by
  //! @param[in]     max_flow  Do not read/write past this flow
  void AdvanceStatesInPlace(TreephaserMultiPath *state, int nuc, int max_flow) const;

  vector<ion::FlowOrder>      flow_orders_;   //!< Sequence of nucleotide flows for each read
  unsigned int max_num_flows;                 //!< Largest number of flows among the different flow orders

  vector< vector<float> >       transition_base_[4];        //!< Probability of polymerase incorporating and staying active
  vector< vector<float> >       transition_flow_[4];        //!< Probability of polymerase not incorporating and staying active

  vector< TreephaserMultiPath > path_;                   //!< Preallocated space for partial path slots

  const static int    kNumPaths = 8;              //!< Maximum number of paths considered
  const static int    kMaxHP = 11;                //!< Maximum callable homopolymer length
  const static int    kMaxPathDelay = 40;         //!< Paths that are delayed more are killed

  static constexpr float  kExtendThreshold = 0.2;     //!< Threshold for extending paths
  static constexpr float  kNegativeMultiplier = 2.0;  //!< Extra weight on the negative residuals
  static constexpr float  kDotThreshold = 0.3;        //!< percentage of expected Signal that constitutes a "dot"
  static constexpr float  kStateWindowCutoff = 1e-6;  //!< Minimum fraction to be taken into account

};

#endif // DPTREEPHASERM_H
