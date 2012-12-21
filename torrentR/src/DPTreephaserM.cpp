/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     DPTreephaserM.cpp
//! @ingroup  BaseCaller
//! @brief    DPTreephaser. Perform dephasing and call base sequence by tree search on potentially more than one raw signal jointly

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "DPTreephaser.h"
#include "DPTreephaserM.h"


// -------------------------------------------------------------------------

DPTreephaserM::DPTreephaserM(const vector<ion::FlowOrder> &flow_orders)
{
	flow_orders_ = flow_orders;
	int max_flows = 0;
	for (unsigned int read = 0; read < flow_orders_.size(); read++) {
		if (flow_orders_[read].num_flows() > max_flows)
			max_flows = flow_orders_[read].num_flows();
	}
	max_num_flows = max_flows;
    verbose = 0;
	for (int nuc = 0; nuc < 4; nuc++) {
        transition_base_[nuc].resize(flow_orders_.size());
        transition_flow_[nuc].resize(flow_orders_.size());
		for (unsigned int read = 0; read < flow_orders_.size(); read++) {
		    transition_base_[nuc][read].resize(flow_orders_[read].num_flows());
		    transition_flow_[nuc][read].resize(flow_orders_[read].num_flows());
		}
	}
	path_.resize(kNumPaths);
	for (int p = 0; p < kNumPaths; ++p) {
		path_[p].solution.resize(max_flows);
		path_[p].read_data_.resize(flow_orders_.size());
		for (unsigned int read = 0; read < flow_orders_.size(); read++) {
			path_[p].read_data_[read].state.resize(flow_orders_[read].num_flows());
			path_[p].read_data_[read].prediction.resize(flow_orders_[read].num_flows());
		}
	}
}


//-------------------------------------------------------------------------

void DPTreephaserM::SetPhasingParameters(vector<double> carry_forward_rate,
		     vector<double> incomplete_extension_rate, vector<double> droop_rate)
{
	assert(carry_forward_rate.size() == incomplete_extension_rate.size());
	assert(carry_forward_rate.size() == droop_rate.size());
	assert(carry_forward_rate.size() == flow_orders_.size());

	for (unsigned int read = 0; read < flow_orders_.size(); read++) {
        double nuc_avaliability[4] = { 0, 0, 0, 0 };
        for (int flow = 0; flow < flow_orders_[read].num_flows(); ++flow) {
            nuc_avaliability[flow_orders_[read].int_at(flow)] = 1;
            for (int nuc = 0; nuc < 4; nuc++) {
                transition_base_[nuc][read][flow] =
            		nuc_avaliability[nuc] * (1-droop_rate[read]) * (1-incomplete_extension_rate[read]);
                transition_flow_[nuc][read][flow] =
            		(1-nuc_avaliability[nuc]) + nuc_avaliability[nuc] * (1-droop_rate[read]) * incomplete_extension_rate[read];
                nuc_avaliability[nuc] *= carry_forward_rate[read];
            }
        }
	}
}

//-------------------------------------------------------------------------
// Windowed normalization strategy that fits an additive and multiplicative offset

void DPTreephaserM::WindowedNormalize(BasecallerRead& read, int flow_limit, int num_flows, int num_steps, int window_size) const
{
  float median_set[window_size];

  // Estimate and correct for additive offset

  float next_normalizer = 0;
  int estim_flow = 0;
  int apply_flow = 0;

  for (int step = 0; step < num_steps; ++step) {

    int window_end = estim_flow + window_size;
    int window_middle = estim_flow + window_size / 2;
    if (window_middle > flow_limit)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < flow_limit; ++estim_flow)
      if (read.prediction[estim_flow] < 0.3)
        median_set[median_set_size++] = read.raw_measurements[estim_flow] - read.prediction[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / window_size;

    for (; apply_flow < window_middle and apply_flow < flow_limit; ++apply_flow) {
      read.normalized_measurements[apply_flow] = read.raw_measurements[apply_flow] - normalizer;
      read.additive_correction[apply_flow] = normalizer;
      normalizer += delta;
    }
  }

  for (; apply_flow < num_flows; ++apply_flow) {
    read.normalized_measurements[apply_flow] = read.raw_measurements[apply_flow] - next_normalizer;
    read.additive_correction[apply_flow] = next_normalizer;
  }

  // Estimate and correct for multiplicative scaling

  next_normalizer = 1;
  estim_flow = 0;
  apply_flow = 0;

  for (int step = 0; step < num_steps; ++step) {

    int window_end = estim_flow + window_size;
    int window_middle = estim_flow + window_size / 2;
    if (window_middle > flow_limit)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < flow_limit; ++estim_flow)
      if (read.prediction[estim_flow] > 0.5 and read.normalized_measurements[estim_flow] > 0)
        median_set[median_set_size++] = read.normalized_measurements[estim_flow] / read.prediction[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / window_size;

    for (; apply_flow < window_middle and apply_flow < flow_limit; ++apply_flow) {
      read.normalized_measurements[apply_flow] /= normalizer;
      read.multiplicative_correction[apply_flow] = normalizer;
      normalizer += delta;
    }
  }

  for (; apply_flow < num_flows; ++apply_flow) {
    read.normalized_measurements[apply_flow] /= next_normalizer;
    read.multiplicative_correction[apply_flow] = next_normalizer;
  }
}


// ----------------------------------------------------------------------------
// Joint normalization and solving functions

// New improved normalization strategy: fitting additive and multiplicative offset
void DPTreephaserM::NormalizeAndSolveAN(BasecallerMultiRead& well, int max_flows)
{
  int window_size = 50;
  int solve_flows = 0;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {
    solve_flows = min((num_steps+1) * window_size, max_flows);

    Solve(well, solve_flows);
    // Normalize each read individually
    for (unsigned int read = 0; read < flow_orders_.size(); read++)
      WindowedNormalize(well.read_vector[read], well.active_until_flow[read], flow_orders_[read].num_flows(),
    		            num_steps, window_size);
  }

  Solve(well, max_flows);
}


// Sliding window adaptive normalization
void DPTreephaserM::NormalizeAndSolveSWAN(BasecallerMultiRead& well, int max_flows)
{
  int window_size = 50;
  int solve_flows = 0;
  int restart_flows = 0;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {

    solve_flows = min((num_steps+1) * window_size, max_flows);
    restart_flows = max((solve_flows - 2*window_size), 0);

    Solve(well, solve_flows, restart_flows);
    // Normalize each read individually
    for (unsigned int read = 0; read < flow_orders_.size(); read++)
      WindowedNormalize(well.read_vector[read], well.active_until_flow[read], flow_orders_[read].num_flows(),
    		            num_steps, window_size);
  }
}


//-------------------------------------------------------------------------

void DPTreephaserM::InitializeState(TreephaserMultiPath *state) const
{
	state->path_in_use = true;
	state->num_active_reads = 0;
	state->min_flow = 0;
	state->max_flow = 0;
	state->bases_called = 0;
	state->current_hp = 0;
	state->per_flow_metric = 0;

	// Initialize state for individual reads
	for (unsigned int read = 0; read < state->read_data_.size(); read++) {
    	state->read_data_[read].flow = 0;
    	state->read_data_[read].state[0] = 1;
    	state->read_data_[read].window_start = 0;
    	state->read_data_[read].window_end = 1;
    	state->read_data_[read].prediction.assign(flow_orders_[read].num_flows(), 0);
    	state->read_data_[read].read_path_metric = 0;
    	state->read_data_[read].residual_left_of_window = 0;
    	state->read_data_[read].read_per_flow_metric = 0;
    	state->read_data_[read].dot_counter = 0;
    }
}


//-------------------------------------------------------------------------
// State progression functions for multiple reads

void DPTreephaserM::AdvanceStates(TreephaserMultiPath *child, const TreephaserMultiPath *parent, int nuc, int max_flow) const
{
  assert (child != parent);

  child->path_metric = parent->path_metric;
  child->per_flow_metric = 0;
  child->num_active_reads = 0;
  child->min_flow = 100000;
  child->max_flow = 0;

  // Loop over all the reads, i.e., the length of the flow orders vector
  for (unsigned int read = 0; read < flow_orders_.size(); read++) {

    // Copy flow  and read specific information
    child->read_data_[read].flow = parent->read_data_[read].flow;
    child->read_data_[read].active_until_flow = parent->read_data_[read].active_until_flow;
    child->read_data_[read].residual_left_of_window = parent->read_data_[read].residual_left_of_window;
    child->read_data_[read].read_path_metric = parent->read_data_[read].residual_left_of_window;
    child->read_data_[read].read_per_flow_metric = parent->read_data_[read].read_per_flow_metric;

    // Check whether read is available for extension
    if (child->read_data_[read].active_until_flow > child->read_data_[read].flow) {

  	  // --- Advance state of read ---
      // Advance flow
      while (child->read_data_[read].flow < child->read_data_[read].active_until_flow and
    		  flow_orders_[read].int_at(child->read_data_[read].flow) != nuc)
        child->read_data_[read].flow++;

      if (child->read_data_[read].active_until_flow > child->read_data_[read].flow)
        child->num_active_reads++;

      if (child->read_data_[read].flow < child->min_flow)
        child->min_flow = child->read_data_[read].flow;
      if (child->read_data_[read].flow > child->max_flow)
        child->max_flow = child->read_data_[read].flow;

      // Initialize window
      child->read_data_[read].window_start = parent->read_data_[read].window_start;
      child->read_data_[read].window_end = parent->read_data_[read].window_end;

      if (parent->read_data_[read].flow != child->read_data_[read].flow or parent->read_data_[read].flow == 0) {

        // This nuc begins a new homopolymer
        float alive = 0;
        for (int flow = parent->read_data_[read].window_start; flow < child->read_data_[read].window_end; ++flow) {

          // State progression according to phasing model
          if (flow < parent->read_data_[read].window_end)
            alive += parent->read_data_[read].state[flow];
          child->read_data_[read].state[flow] = alive * transition_base_[nuc][read][flow];
          alive *= transition_flow_[nuc][read][flow];

          // Window maintenance
          if (flow == child->read_data_[read].window_start and (child->read_data_[read].state[flow] < kStateWindowCutoff)) // or flow < child->flow-60))
            child->read_data_[read].window_start++;

          if (flow == child->read_data_[read].window_end-1 and
        		  child->read_data_[read].window_end < child->read_data_[read].active_until_flow and
        		  alive > kStateWindowCutoff) // and flow < child->flow+60)
            child->read_data_[read].window_end++;
        }
        child->current_hp = 1;

      } else {
        // This nuc simply prolongs current homopolymer, inherits state from parent
        memcpy(&child->read_data_[read].state[child->read_data_[read].window_start], &parent->read_data_[read].state[child->read_data_[read].window_start],
            (child->read_data_[read].window_end-child->read_data_[read].window_start)*sizeof(float));
        child->current_hp = parent->current_hp + 1;
      }

      for (int flow = parent->read_data_[read].window_start; flow < child->read_data_[read].window_end; ++flow)
          child->read_data_[read].prediction[flow] = parent->read_data_[read].prediction[flow] + child->read_data_[read].state[flow];
      // --- Sate has been advanced ---

  	} // if parent->read_in_use bracket
  } // looping over reads
  // If the incorporating flow of the read with the minimum flow is equal or larger than max_flow,
  // path falls off a cliff
  if (child->min_flow >= max_flow) {
    child->num_active_reads = 0;
  }
}


void DPTreephaserM::AdvanceStatesInPlace(TreephaserMultiPath *state, int nuc, int max_flow) const
{
  state->num_active_reads  = 0;
  state->min_flow = 100000;
  state->max_flow = 0;

  for (unsigned int read = 0; read < flow_orders_.size(); read++) {

	int up_to_flow = min( max_flow, state->read_data_[read].active_until_flow );
	// Check whether read is available for extension
	if (state->read_data_[read].flow < up_to_flow) {

      // --- Advance in-phase flow ---
      int old_flow = state->read_data_[read].flow;
      int old_window_start = state->read_data_[read].window_start;
      int old_window_end = state->read_data_[read].window_end;

      // Advancing flow
      while (state->read_data_[read].flow < flow_orders_[read].num_flows() and
              flow_orders_[read].int_at(state->read_data_[read].flow) != nuc)
        state->read_data_[read].flow++;

      // Check whether end of read was reached or whether read is still active
      if (state->read_data_[read].flow < state->read_data_[read].active_until_flow)
        state->num_active_reads++;

      if (state->read_data_[read].flow < state->min_flow)
        state->min_flow = state->read_data_[read].flow;
      if (state->read_data_[read].flow > state->max_flow)
        state->max_flow = state->read_data_[read].flow;

      if (old_flow != state->read_data_[read].flow or old_flow == 0) {

        // This nuc begins a new homopolymer, need to adjust state
        float alive = 0;
        for (int flow = old_window_start; flow < state->read_data_[read].window_end; flow++) {

          // State progression according to phasing model
          if (flow < old_window_end)
            alive += state->read_data_[read].state[flow];
          state->read_data_[read].state[flow] = alive * transition_base_[nuc][read][flow];
          alive *= transition_flow_[nuc][read][flow];

          // Window maintenance
          if (flow == state->read_data_[read].window_start and
                  (state->read_data_[read].state[flow] < kStateWindowCutoff)) // or flow < state->flow-60))
          state->read_data_[read].window_start++;

          if (flow == state->read_data_[read].window_end-1 and
                  state->read_data_[read].window_end < up_to_flow and alive > kStateWindowCutoff) // and flow < state->flow+60)
            state->read_data_[read].window_end++;
        }
        // --- State advanced ---
        state->current_hp = 1;
      }
      else
        state->current_hp++;

      // Update prediction
      for (int flow = state->read_data_[read].window_start; flow < state->read_data_[read].window_end; ++flow)
        state->read_data_[read].prediction[flow] += state->read_data_[read].state[flow];

	} // if read_in_use bracket
  } // looping over reads
}


/*/ ------------------------------------------------------------------------------
// Let's deal with this function later XXX

void DPTreephaserM::Simulate(BasecallerMultiRead& multi_read, int max_flows)
{
  // Should probably be in base space
  InitializeState(&path_[0]);

  while(path_[0].min_flow < max_flows and path_[0].bases_called < multi_read.bases_called) {
      AdvanceStatesInPlace(&path_[0], multi_read.solution[path_[0].bases_called], max_flows);

      if (path_[0].num_active_reads > 0 and path_[0].bases_called < path_[0].solution.size()) {
        path_[0].solution[path_[0].bases_called] = multi_read.solution[path_[0].bases_called];
        path_[0].bases_called++;
      }
      else {
        // All reads ended before restart_flows. No point resolving it.
        for (int read = 0; read < flow_orders_.size(); read++) {
          multi_read.read_vector[read].prediction.swap(path_[0].read_data_[read].prediction);
        }
        // Solution might be a subset of what was originally provided if sequence longer than flow space allows
        multi_read.solution.swap(path_[0].solution);
        multi_read.bases_called = path_[0].bases_called;
        return;
      }
  } // end while statement
  multi_read.read_vector[read].prediction.swap(path_[0].read_data_[read].prediction);
  multi_read.solution.swap(path_[0].solution);
  multi_read.bases_called = path_[0].bases_called;
}
// ------------------------------------------------------------------------------ */


// Solve - main tree search procedure that determines the base sequence.
// Another temporary version, uses external class for storing read data

void DPTreephaserM::Solve(BasecallerMultiRead& multi_read, int max_flows, int restart_flows)
{
  // Check and adjust sizing of input values
  assert( multi_read.read_vector.size() == flow_orders_.size() );
  for (unsigned int read = 0; read < flow_orders_.size(); read++) {
    assert( multi_read.active_until_flow[read] <= flow_orders_[read].num_flows() );
    path_[0].read_data_[read].active_until_flow = multi_read.active_until_flow[read];
    if (multi_read.read_vector[read].prediction.size() != 2*max_num_flows)
      multi_read.read_vector[read].prediction.resize(flow_orders_[read].num_flows());
  }
  if (multi_read.solution.size() != 2*max_num_flows)
    multi_read.solution.resize(2*max_num_flows);

  // Initialize stack: just one root path
    for (int p = 1; p < kNumPaths; ++p)
      path_[p].path_in_use = false;

  InitializeState(&path_[0]);
  path_[0].path_metric = 0;
  path_[0].per_flow_metric = 0;

  // check size of solution and adjust if necessary
  if (path_[0].solution.size() != multi_read.solution.size()) {
	  path_[0].solution.resize(multi_read.solution.size());
  }

  int space_on_stack = kNumPaths - 1;
  float sum_of_squares_upper_bound = 1e20;  //max_flows; // Squared distance of solution to measurements

  if (restart_flows > 0) {
    // The solver will not attempt to solve initial restart_flows (read farthest along surpassing restart_flows!)
    // - Simulate restart_flows instead of solving
    // - If it turns out that solving was finished before restart_flows, simply exit without any changes to the read.

    while(path_[0].max_flow < restart_flows and path_[0].bases_called < multi_read.bases_called) {
      AdvanceStatesInPlace(&path_[0], multi_read.solution[path_[0].bases_called], max_flows);

      if (path_[0].num_active_reads > 0 and path_[0].bases_called < path_[0].solution.size()) {
        path_[0].solution[path_[0].bases_called] = multi_read.solution[path_[0].bases_called];
        path_[0].bases_called++;
      }
      else {
        // All reads ended before restart_flows. No point resolving it.
        for (unsigned int read = 0; read < flow_orders_.size(); read++) {
          multi_read.read_vector[read].prediction.swap(path_[0].read_data_[read].prediction);
        }
        // Solution might be a subset of what was originally provided if sequence longer than flow space allows
        multi_read.solution.swap(path_[0].solution);
        multi_read.bases_called = path_[0].bases_called;
        return;
      }
    } // end resolve while statement

    // Compute "left of window" metric for paths
    for (unsigned int read = 0; read < flow_orders_.size(); read++) {
      for (int flow = 0; flow < path_[0].read_data_[read].window_start; ++flow) {
        float residual = multi_read.read_vector[read].normalized_measurements[flow]
                           - path_[0].read_data_[read].prediction[flow];
        path_[0].read_data_[read].residual_left_of_window += residual * residual;
      }
      // If read is still active at restart point, reset flow_limit to maximum number of flows
      // so it will potentially be solved beyond last exit point
      if (path_[0].read_data_[read].active_until_flow > path_[0].read_data_[read].flow) {
      	path_[0].read_data_[read].active_until_flow = flow_orders_[read].num_flows();
      }
    }
  }  //end if-restart block

  // ------- Main loop to select / expand / delete paths -------
  while (1) {

    // ------------------------------------------
    // Step 1: Prune the content of the stack and make sure there are at least 4 empty slots

    // Remove paths that are more than 'maxPathDelay' behind the longest one
    if (space_on_stack < kNumPaths-3) {
      int longest_path = 0;
      for (int p = 0; p < kNumPaths; ++p)
        if (path_[p].path_in_use)
          longest_path = max(longest_path, path_[p].max_flow);

      if (longest_path > kMaxPathDelay) {
        for (int p = 0; p < kNumPaths; ++p) {
          if (path_[p].path_in_use and path_[p].max_flow < longest_path-kMaxPathDelay) {
            path_[p].path_in_use = false;
            space_on_stack++;
          }
        }
      }
    }

    // If necessary, remove paths with worst perFlowMetric
    while (space_on_stack < 4) {
      // find maximum per flow metric
      float max_per_flow_metric = -0.1;
      int max_metric_path = kNumPaths;
      for (int p = 0; p < kNumPaths; ++p) {
        if (path_[p].path_in_use and path_[p].per_flow_metric > max_per_flow_metric) {
          max_per_flow_metric = path_[p].per_flow_metric;
          max_metric_path = p;
        }
      }

      // killing path with largest per flow metric
      if (!(max_metric_path < kNumPaths)) {
        printf("Failed assertion in Treephaser\n");
        for (int p = 0; p < kNumPaths; ++p) {
          if (path_[p].path_in_use)
            printf("Path %d, in_use = true, per_flow_metric = %f\n", p, path_[p].per_flow_metric);
          else
            printf("Path %d, in_use = false, per_flow_metric = %f\n", p, path_[p].per_flow_metric);
        }
        fflush(NULL);
      }
      assert (max_metric_path < kNumPaths);

      path_[max_metric_path].path_in_use = false;
      space_on_stack++;
    }

    // ------------------------------------------
    // Step 2: Select a path to expand or break if there is none

    TreephaserMultiPath *parent = NULL;
    float min_path_metric = 1000;
    for (int p = 0; p < kNumPaths; ++p) {
      if (path_[p].path_in_use and path_[p].path_metric < min_path_metric and path_[p].bases_called < path_[p].solution.size()) {
        min_path_metric = path_[p].path_metric;
        parent = &path_[p];
      }
    }
    if (!parent)
      break;

    // ------------------------------------------
    // Step 3: Construct four expanded paths and calculate feasibility metrics
    assert (space_on_stack >= 4);

    TreephaserMultiPath *children[4];

    for (int nuc = 0, p = 0; nuc < 4; ++p)
      if (not path_[p].path_in_use)
        children[nuc++] = &path_[p];

    float penalty[4] = { 0, 0, 0, 0 };

    for (int nuc = 0; nuc < 4; ++nuc) {

      TreephaserMultiPath *child = children[nuc];

      AdvanceStates(child, parent, nuc, max_flows);

      // Apply easy termination rules

      if (child->current_hp > kMaxHP) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      if (child->num_active_reads < 1) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }
      else {
        // Looping over different reads to update metrics
        for (unsigned int read = 0; read < flow_orders_.size(); read++) {

          // Compute metrics if read is active
          if (child->read_data_[read].flow < child->read_data_[read].active_until_flow) {
            float penaltyN = 0;
            float penalty1 = 0;
            int up_to_flow = min(max_flows, flow_orders_[read].num_flows());

            for (int flow = parent->read_data_[read].window_start; flow < child->read_data_[read].window_end; ++flow) {

        	  float residual = multi_read.read_vector[read].normalized_measurements[flow] -
        	         child->read_data_[read].prediction[flow];
              float residual_squared = residual * residual;

              // Update metrics for each read; path metric until specified maximum
              if (flow < child->read_data_[read].window_start) {
                child->read_data_[read].residual_left_of_window += residual_squared;
                child->read_data_[read].read_path_metric += residual_squared;
              } else if (residual <= 0 and flow < up_to_flow)
                child->read_data_[read].read_path_metric += residual_squared;

              if (residual <= 0)
                penaltyN += residual_squared;
              else if (flow < child->read_data_[read].flow)
                penalty1 += residual_squared;
            }

            penalty[nuc] += (penalty1 + kNegativeMultiplier * penaltyN) / child->num_active_reads;
            penalty1 += penaltyN;

            if (child->read_data_[read].flow > 0)
              child->read_data_[read].read_per_flow_metric =
        		  (child->read_data_[read].read_path_metric + 0.5 * penalty1) / child->read_data_[read].flow;
            else
              child->read_data_[read].read_per_flow_metric = 0;
            child->path_metric += (child->read_data_[read].read_path_metric
        		  - parent->read_data_[read].read_path_metric) / child->num_active_reads;

          } // end block - if read is active

        child->per_flow_metric += child->read_data_[read].read_per_flow_metric;
        } // looping over reads
      } // end else statement
    } //looping over nucs

    // Find out which nuc has the least penalty (the greedy choice nuc)
    int best_nuc = 0;
    if (penalty[best_nuc] > penalty[1])
      best_nuc = 1;
    if (penalty[best_nuc] > penalty[2])
      best_nuc = 2;
    if (penalty[best_nuc] > penalty[3])
      best_nuc = 3;

    // ------------------------------------------
    // Step 4: Use calculated metrics to decide which paths are worth keeping

    for (int nuc = 0; nuc < 4; ++nuc) {

      TreephaserMultiPath *child = children[nuc];

      // Path termination rules

      if (penalty[nuc] >= 20)
        continue;

      float read_path_metric_sum = 0;
      for (unsigned int read = 0; read < flow_orders_.size(); read++)
        read_path_metric_sum += child->read_data_[read].read_path_metric;

      if (read_path_metric_sum > sum_of_squares_upper_bound)
        continue;

      // This is the only rule that depends on finding the "best nuc"
      if (penalty[nuc] - penalty[best_nuc] >= kExtendThreshold)
        continue;

      // Discontinue reads that have more than one severe signal mismatch ("dot") in a  row
      // XXX Should I be less stringent for multi reads?
      for (unsigned int read = 0; read < flow_orders_.size(); read++) {
        if (child->read_data_[read].active_until_flow > child->read_data_[read].flow) {
          float dot_signal = (multi_read.read_vector[read].normalized_measurements[child->read_data_[read].flow]
                                - parent->read_data_[read].prediction[child->read_data_[read].flow])
                                / child->read_data_[read].state[child->read_data_[read].flow];
          child->read_data_[read].dot_counter = (dot_signal < kDotThreshold) ? (parent->read_data_[read].dot_counter + 1) : 0;
          if (child->read_data_[read].dot_counter > 1) {
            child->read_data_[read].active_until_flow = child->read_data_[read].flow;
            child->num_active_reads--;
          }
        }
      }

      if (child->num_active_reads <= 0)
        continue;

      // Path survived termination rules and will be kept on stack
      child->path_in_use = true;
      space_on_stack--;

      // --- Updates only prediction of active reads XXX
      // Fill out the remaining portion of the prediction
      for (unsigned int read = 0; read < flow_orders_.size(); read++) {
        if (child->read_data_[read].flow < child->read_data_[read].active_until_flow) {

          memcpy(&child->read_data_[read].prediction[0], &parent->read_data_[read].prediction[0], parent->read_data_[read].window_start*sizeof(float));
          for (int flow = child->read_data_[read].window_end; flow < flow_orders_[read].num_flows(); ++flow)
            child->read_data_[read].prediction[flow] = 0;
        } else {
          // Just copy prediction from parent if read is not active (any more)
          child->read_data_[read].prediction = parent->read_data_[read].prediction;
        }
      }

      // Fill out the solution in base space
      child->solution = parent->solution;
      child->solution[parent->bases_called] = nuc;
      child->bases_called = parent->bases_called + 1;
    } // Looping over nucs

    // ------------------------------------------
    // Step 5. Check if the selected path is in fact the best path so far

    float sum_of_squares = 0;
    for (unsigned int read = 0; read < flow_orders_.size(); read++) {
      // If read has ever been active, involve it in the computation of the squared distance
      if (parent->read_data_[read].active_until_flow > 0) {

        int up_to_flow = min(max_flows, flow_orders_[read].num_flows());
    	sum_of_squares += parent->read_data_[read].residual_left_of_window;

        for (int flow = parent->read_data_[read].window_start; flow < up_to_flow; flow++) {
          float residual = multi_read.read_vector[read].normalized_measurements[flow] - parent->read_data_[read].prediction[flow];
          sum_of_squares += residual * residual;
        }
      }
    }
    // Updating best path
    if (sum_of_squares < sum_of_squares_upper_bound) {
      multi_read.solution.swap(parent->solution);
      sum_of_squares_upper_bound = sum_of_squares;
      multi_read.bases_called = parent->bases_called;

      for (unsigned int read = 0; read < flow_orders_.size(); read++) {
    	multi_read.active_until_flow[read] = parent->read_data_[read].active_until_flow;
    	multi_read.read_vector[read].prediction.swap(parent->read_data_[read].prediction);
      }
    }

    parent->path_in_use = false;
    space_on_stack++;

  } // main decision loop
}



/*/ ------------------------------------------------------------------------
// Commenting that one out to start with

// Compute quality metrics
int DPTreephaserM::ComputeQVmetrics(BasecallerRead& read)
{

  read.state_inphase.assign(flow_order_.num_flows(), 1);
  read.state_total.assign(flow_order_.num_flows(), 1);

  int num_bases = 0;
  for (int flow = 0; flow < flow_order_.num_flows(); ++flow)
    num_bases += read.solution[flow];

  if (num_bases == 0)
    return 0;

  read.penalty_mismatch.assign(num_bases, 0);
  read.penalty_residual.assign(num_bases, 0);

  int max_flows = flow_order_.num_flows();

  TreephaserPath *parent = &path_[0];
  TreephaserPath *children[4] = { &path_[1], &path_[2], &path_[3], &path_[4] };

  InitializeState(parent);

  int base = 0;
  float recent_state_inphase = 1;
  float recent_state_total = 1;

  // main loop for base calling
  for (int solution_flow = 0; solution_flow < max_flows; ++solution_flow) {
    for (int hp = 0; hp < read.solution[solution_flow]; ++hp) {

      float penalty[4] = { 0, 0, 0, 0 };

      for (int nuc = 0; nuc < 4; nuc++) {

        TreephaserPath *child = children[nuc];

        AdvanceState(child, parent, nuc, max_flows);

        // Apply easy termination rules

        if (child->flow >= max_flows) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        if (parent->solution[child->flow] >= kMaxHP) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        for (int flow = parent->window_start; flow < child->window_end; ++flow) {
          float residual = read.normalized_measurements[flow] - child->prediction[flow];
          if (residual <= 0 or flow < child->flow)
            penalty[nuc] += residual*residual;
        }
      } //looping over nucs


      // find current incorporating base
      int called_nuc = flow_order_.int_at(solution_flow);
      assert(children[called_nuc]->flow == solution_flow);

      recent_state_inphase = children[called_nuc]->state[solution_flow];
      recent_state_total = 0;
      for (int flow = children[called_nuc]->window_start; flow < children[called_nuc]->window_end; ++flow)
        recent_state_total += children[called_nuc]->state[flow];

      // Get delta penalty to next best solution
      read.penalty_mismatch[base] = -1; // min delta penalty to earlier base hypothesis
      read.penalty_residual[base] = 0;

      if (solution_flow - parent->window_start > 0)
        read.penalty_residual[base] = penalty[called_nuc] / (solution_flow - parent->window_start);

      for (int nuc = 0; nuc < 4; ++nuc) {
        if (nuc == called_nuc)
            continue;
        float penalty_mismatch = penalty[called_nuc] - penalty[nuc];
        read.penalty_mismatch[base] = max(read.penalty_mismatch[base], penalty_mismatch);
      }

      // Fill out the remaining portion of the prediction
      for (int flow = 0; flow < parent->window_start; ++flow)
        children[called_nuc]->prediction[flow] = parent->prediction[flow];

      for (int flow = children[called_nuc]->window_end; flow < max_flows; ++flow)
        children[called_nuc]->prediction[flow] = 0;

      // Called state is the starting point for next base
      TreephaserPath *swap = parent;
      parent = children[called_nuc];
      children[called_nuc] = swap;

      base++;
    }

    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
    read.state_total[solution_flow] = max(recent_state_total, 0.01f);
  }

  return num_bases;
}

// ------------------------------------------------------------------------ */


