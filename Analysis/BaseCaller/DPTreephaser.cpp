/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     DPTreephaser.cpp
//! @ingroup  BaseCaller
//! @brief    DPTreephaser. Perform dephasing and call base sequence by tree search

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "DPTreephaser.h"

DPTreephaser::DPTreephaser(const ion::FlowOrder& flow_order)
  : flow_order_(flow_order)
{
  for (int i = 0; i < 4; i++) {
    transition_base_[i].resize(flow_order_.num_flows());
    transition_flow_[i].resize(flow_order_.num_flows());
  }
  path_.resize(kNumPaths);
  for (int p = 0; p < kNumPaths; ++p) {
    path_[p].state.resize(flow_order_.num_flows());
    path_[p].solution.resize(flow_order_.num_flows());
    path_[p].prediction.resize(flow_order_.num_flows());
  }
}

//-------------------------------------------------------------------------

void DPTreephaser::SetModelParameters(double carry_forward_rate, double incomplete_extension_rate, double droop_rate)
{
  double nuc_avaliability[4] = { 0, 0, 0, 0 };
  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    nuc_avaliability[flow_order_.int_at(flow)] = 1;
    for (int nuc = 0; nuc < 4; nuc++) {
      transition_base_[nuc][flow] = nuc_avaliability[nuc] * (1-droop_rate) * (1-incomplete_extension_rate);
      transition_flow_[nuc][flow] = (1-nuc_avaliability[nuc]) + nuc_avaliability[nuc] * (1-droop_rate) * incomplete_extension_rate;
      nuc_avaliability[nuc] *= carry_forward_rate;
    }
  }
}

//-------------------------------------------------------------------------


void BasecallerRead::SetDataAndKeyNormalize(const float *measurements, int num_flows, const int *key_flows, int num_key_flows)
{
  raw_measurements.resize(num_flows);
  normalized_measurements.resize(num_flows);
  solution.assign(num_flows, 0);
  prediction.assign(num_flows, 0);
  additive_correction.assign(num_flows, 0);
  multiplicative_correction.assign(num_flows, 1.0);

  float onemer_sum = 0.0;
  float onemer_count = 0.0;
  for (int flow = 0; flow < num_key_flows; ++flow) {
    if (key_flows[flow] == 1) {
      onemer_sum += measurements[flow];
      onemer_count += 1.0;
    }
  }

  key_normalizer = 1;
  if (onemer_sum and onemer_count)
    key_normalizer = onemer_count / onemer_sum;

  for (int flow = 0; flow < num_flows; ++flow) {
    raw_measurements[flow] = measurements[flow] * key_normalizer;
    normalized_measurements[flow] = raw_measurements[flow];
  }
}



// ----------------------------------------------------------------------
// New normalization strategy

void DPTreephaser::WindowedNormalize(BasecallerRead& read, int num_steps, int window_size) const
{
  int num_flows = read.raw_measurements.size();
  float median_set[window_size];

  // Estimate and correct for additive offset

  float next_normalizer = 0;
  int estim_flow = 0;
  int apply_flow = 0;

  for (int step = 0; step < num_steps; ++step) {

    int window_end = estim_flow + window_size;
    int window_middle = estim_flow + window_size / 2;
    if (window_middle > num_flows)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < num_flows; ++estim_flow)
      if (read.prediction[estim_flow] < 0.3)
        median_set[median_set_size++] = read.raw_measurements[estim_flow] - read.prediction[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / window_size;

    for (; apply_flow < window_middle and apply_flow < num_flows; ++apply_flow) {
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
    if (window_middle > num_flows)
      break;

    float normalizer = next_normalizer;

    int median_set_size = 0;
    for (; estim_flow < window_end and estim_flow < num_flows; ++estim_flow)
      if (read.prediction[estim_flow] > 0.5 and read.normalized_measurements[estim_flow] > 0)
        median_set[median_set_size++] = read.normalized_measurements[estim_flow] / read.prediction[estim_flow];

    if (median_set_size > 5) {
      std::nth_element(median_set, median_set + median_set_size/2, median_set + median_set_size);
      next_normalizer = median_set[median_set_size / 2];
      if (step == 0)
        normalizer = next_normalizer;
    }

    float delta = (next_normalizer - normalizer) / window_size;

    for (; apply_flow < window_middle and apply_flow < num_flows; ++apply_flow) {
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



// New improved normalization strategy
void DPTreephaser::NormalizeAndSolve3(BasecallerRead& well, int max_flows)
{
  int window_size = 50;
  int solve_flows = 0;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {
    solve_flows = min((num_steps+1) * window_size, max_flows);

    Solve(well, solve_flows);
    WindowedNormalize(well, num_steps, window_size);
  }

  Solve(well, max_flows);
}



// Old normalization, but uses BasecallerRead object
void DPTreephaser::NormalizeAndSolve4(BasecallerRead& well, int max_flows)
{
  for (int iter = 0; iter < 7; ++iter) {
    int solve_flow = 100 + 20 * iter;
    if (solve_flow < max_flows) {
      Solve(well, solve_flow);
      Normalize(well, 11, solve_flow-20);
    }
  }
  Solve(well, max_flows);
}




// Sliding window adaptive normalization
void DPTreephaser::NormalizeAndSolve5(BasecallerRead& well, int max_flows)
{
  int window_size = 50;
  int solve_flows = 0;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {
    solve_flows = min((num_steps+1) * window_size, max_flows);
    int restart_flows = max(solve_flows-100, 0);

    Solve(well, solve_flows, restart_flows);
    WindowedNormalize(well, num_steps, window_size);
  }

  Solve(well, max_flows);
}




float DPTreephaser::Normalize(BasecallerRead& read, int start_flow, int end_flow) const
{
  const static float kHomopolymerWeight[5] = { 0, 1, 0.8, 0.64, 0.37 };

  float xy = 0;
  float yy = 0;
  int num_flows = read.raw_measurements.size();

  for (int flow = start_flow; flow < end_flow and flow < num_flows; ++flow) {
    if (read.solution[flow] > 0 and read.solution[flow] < 5) {
      xy += kHomopolymerWeight[(int)read.solution[flow]] * read.prediction[flow] * read.raw_measurements[flow];
      yy += kHomopolymerWeight[(int)read.solution[flow]] * read.prediction[flow] * read.prediction[flow];
    }
  }

  float divisor = 1;
  if (xy > 0 and yy > 0)
    divisor = xy / yy;

  for (int flow = 0; flow < num_flows; ++flow)
    read.normalized_measurements[flow] = read.raw_measurements[flow] / divisor;

  read.additive_correction.assign(num_flows, 0);
  read.multiplicative_correction.assign(num_flows, divisor);

  return divisor;
}


//-------------------------------------------------------------------------

void DPTreephaser::InitializeState(TreephaserPath *state) const
{
  state->flow = 0;
  //state->state.resize(flow_order_.num_flows()); // Superfluous
  state->state[0] = 1;
  state->window_start = 0;
  state->window_end = 1;
  state->prediction.assign(flow_order_.num_flows(), 0);
  state->solution.assign(flow_order_.num_flows(), 0);
}


void DPTreephaser::AdvanceState(TreephaserPath *child, const TreephaserPath *parent, int nuc, int max_flow) const
{
  assert (child != parent);

  // Advance flow
  child->flow = parent->flow;
  while (child->flow < max_flow and flow_order_.int_at(child->flow) != nuc)
    child->flow++;

  // Initialize window
  child->window_start = parent->window_start;
  child->window_end = parent->window_end;

  if (parent->flow != child->flow or parent->flow == 0) {

    // This nuc begins a new homopolymer
    float alive = 0;
    for (int flow = parent->window_start; flow < child->window_end; ++flow) {

      // State progression according to phasing model
      if (flow < parent->window_end)
        alive += parent->state[flow];
      child->state[flow] = alive * transition_base_[nuc][flow];
      alive *= transition_flow_[nuc][flow];

      // Window maintenance
      if (flow == child->window_start and (child->state[flow] < kStateWindowCutoff)) // or flow < child->flow-60))
        child->window_start++;

      if (flow == child->window_end-1 and child->window_end < max_flow and alive > kStateWindowCutoff) // and flow < child->flow+60)
        child->window_end++;
    }

  } else {
    // This nuc simply prolongs current homopolymer, inherits state from parent
    //for (int flow = child->window_start; flow < child->window_end; ++flow)
    //  child->state[flow] = parent->state[flow];
    memcpy(&child->state[child->window_start], &parent->state[child->window_start],
        (child->window_end-child->window_start)*sizeof(float));
  }

  for (int flow = parent->window_start; flow < child->window_end; ++flow)
    child->prediction[flow] = parent->prediction[flow] + child->state[flow];
}

void DPTreephaser::AdvanceStateInPlace(TreephaserPath *state, int nuc, int max_flow) const
{
  // Advance in-phase flow
  int old_flow = state->flow;
  int old_window_start = state->window_start;
  int old_window_end = state->window_end;
  while (state->flow < max_flow and flow_order_.int_at(state->flow) != nuc)
    state->flow++;

  if (old_flow != state->flow or old_flow == 0) {

    // This nuc begins a new homopolymer, need to adjust state
    float alive = 0;
    for (int flow = old_window_start; flow < state->window_end; flow++) {

      // State progression according to phasing model
      if (flow < old_window_end)
        alive += state->state[flow];
      state->state[flow] = alive * transition_base_[nuc][flow];
      alive *= transition_flow_[nuc][flow];

      // Window maintenance
      if (flow == state->window_start and (state->state[flow] < kStateWindowCutoff)) // or flow < state->flow-60))
        state->window_start++;

      if (flow == state->window_end-1 and state->window_end < max_flow and alive > kStateWindowCutoff) // and flow < state->flow+60)
        state->window_end++;
    }
  }

//  for (int flow = old_window_start; flow < state->window_end; ++flow)
  for (int flow = state->window_start; flow < state->window_end; ++flow)
    state->prediction[flow] += state->state[flow];
}



void DPTreephaser::Simulate(BasecallerRead& data, int max_flows)
{
  max_flows = min(max_flows,flow_order_.num_flows());
  InitializeState(&path_[0]);

  for (int solution_flow = 0; solution_flow < max_flows; ++solution_flow)
    for (int hp = 0; hp < data.solution[solution_flow]; ++hp)
      AdvanceStateInPlace(&path_[0], flow_order_.int_at(solution_flow), flow_order_.num_flows());

  data.prediction.swap(path_[0].prediction);
}



// Solve3 - main tree search procedure that determines the base sequence.
// Another temporary version, uses external class for storing read data

void DPTreephaser::Solve(BasecallerRead& read, int max_flows, int restart_flows)
{
  assert(max_flows <= flow_order_.num_flows());

  // Initialize stack: just one root path
  for (int p = 1; p < kNumPaths; ++p)
    path_[p].in_use = false;

  InitializeState(&path_[0]);
  path_[0].path_metric = 0;
  path_[0].per_flow_metric = 0;
  path_[0].residual_left_of_window = 0;
  path_[0].dot_counter = 0;
  path_[0].in_use = true;

  int space_on_stack = kNumPaths - 1;
  float sum_of_squares_upper_bound = 1e20;  //max_flows; // Squared distance of solution to measurements

  if (restart_flows > 0) {
    // The solver will not attempt to solve initial restart_flows
    // - Simulate restart_flows instead of solving
    // - If it turns out that solving was finished before restart_flows, simply exit without any changes to the read.

    restart_flows = min(restart_flows, flow_order_.num_flows());
    int last_incorporating_flow = 0;

    for (int solution_flow = 0; solution_flow < restart_flows; ++solution_flow) {

      path_[0].solution[solution_flow] = read.solution[solution_flow];

      for (int hp = 0; hp < path_[0].solution[solution_flow]; ++hp) {
        AdvanceStateInPlace(&path_[0], flow_order_.int_at(solution_flow), max_flows);
        last_incorporating_flow = solution_flow;
      }
    }

    if (last_incorporating_flow < restart_flows-10) { // This read ended before restartFlows. No point resolving it.
      read.prediction.swap(path_[0].prediction);
      return;
    }

    for (int flow = 0; flow < path_[0].window_start; ++flow) {
      float residual = read.normalized_measurements[flow] - path_[0].prediction[flow];
      path_[0].residual_left_of_window += residual * residual;
    }
  }

  // Initializing variables
  read.solution.assign(flow_order_.num_flows(), 0);
  read.prediction.assign(flow_order_.num_flows(), 0);

  // Main loop to select / expand / delete paths
  while (1) {

    // ------------------------------------------
    // Step 1: Prune the content of the stack and make sure there are at least 4 empty slots

    // Remove paths that are more than 'maxPathDelay' behind the longest one
    if (space_on_stack < kNumPaths-3) {
      int longest_path = 0;
      for (int p = 0; p < kNumPaths; ++p)
        if (path_[p].in_use)
          longest_path = max(longest_path, path_[p].flow);

      if (longest_path > kMaxPathDelay) {
        for (int p = 0; p < kNumPaths; ++p) {
          if (path_[p].in_use and path_[p].flow < longest_path-kMaxPathDelay) {
            path_[p].in_use = false;
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
        if (path_[p].in_use and path_[p].per_flow_metric > max_per_flow_metric) {
          max_per_flow_metric = path_[p].per_flow_metric;
          max_metric_path = p;
        }
      }

      // killing path with largest per flow metric
      if (!(max_metric_path < kNumPaths)) {
        printf("Failed assertion in Treephaser\n");
        for (int p = 0; p < kNumPaths; ++p) {
          if (path_[p].in_use)
            printf("Path %d, in_use = true, per_flow_metric = %f\n", p, path_[p].per_flow_metric);
          else
            printf("Path %d, in_use = false, per_flow_metric = %f\n", p, path_[p].per_flow_metric);
        }
        fflush(NULL);
      }
      assert (max_metric_path < kNumPaths);

      path_[max_metric_path].in_use = false;
      space_on_stack++;
    }

    // ------------------------------------------
    // Step 2: Select a path to expand or break if there is none

    TreephaserPath *parent = NULL;
    float min_path_metric = 1000;
    for (int p = 0; p < kNumPaths; ++p) {
      if (path_[p].in_use and path_[p].path_metric < min_path_metric) {
        min_path_metric = path_[p].path_metric;
        parent = &path_[p];
      }
    }
    if (!parent)
      break;

    // ------------------------------------------
    // Step 3: Construct four expanded paths and calculate feasibility metrics
    assert (space_on_stack >= 4);

    TreephaserPath *children[4];

    for (int nuc = 0, p = 0; nuc < 4; ++p)
      if (not path_[p].in_use)
        children[nuc++] = &path_[p];

    float penalty[4] = { 0, 0, 0, 0 };

    for (int nuc = 0; nuc < 4; ++nuc) {

      TreephaserPath *child = children[nuc];

      AdvanceState(child, parent, nuc, max_flows);

      // Apply easy termination rules

      if (child->flow >= max_flows) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      int currentHP = parent->solution[child->flow];
      if (currentHP == kMaxHP) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      child->path_metric = parent->residual_left_of_window;
      child->residual_left_of_window = parent->residual_left_of_window;

      float penaltyN = 0;
      float penalty1 = 0;

      for (int flow = parent->window_start; flow < child->window_end; ++flow) {

        float residual = read.normalized_measurements[flow] - child->prediction[flow];
        float residual_squared = residual * residual;

        // Metric calculation
        if (flow < child->window_start) {
          child->residual_left_of_window += residual_squared;
          child->path_metric += residual_squared;
        } else if (residual <= 0)
          child->path_metric += residual_squared;

        if (residual <= 0)
          penaltyN += residual_squared;
        else if (flow < child->flow)
          penalty1 += residual_squared;
      }


      penalty[nuc] = penalty1 + kNegativeMultiplier * penaltyN;
      penalty1 += penaltyN;

      child->per_flow_metric = (child->path_metric + 0.5 * penalty1) / child->flow;

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

      TreephaserPath *child = children[nuc];

      // Path termination rules

      if (penalty[nuc] >= 20)
        continue;

      if (child->path_metric > sum_of_squares_upper_bound)
        continue;

      // This is the only rule that depends on finding the "best nuc"
      if (penalty[nuc] - penalty[best_nuc] >= kExtendThreshold)
        continue;

      float dot_signal = (read.normalized_measurements[child->flow] - parent->prediction[child->flow]) / child->state[child->flow];
      child->dot_counter = (dot_signal < kDotThreshold) ? (parent->dot_counter + 1) : 0;
      if (child->dot_counter > 1)
        continue;

      // Path survived termination rules and will be kept on stack
      child->in_use = true;
      space_on_stack--;

      // Fill out the remaining portion of the prediction
//      for (int flow = 0; flow < parent->window_start; ++flow)
//        child->prediction[flow] = parent->prediction[flow];
      memcpy(&child->prediction[0], &parent->prediction[0], parent->window_start*sizeof(float));

      for (int flow = child->window_end; flow < max_flows; ++flow)
        child->prediction[flow] = 0;

      // Fill out the solution
      child->solution = parent->solution;
      child->solution[child->flow]++;
    }

    // ------------------------------------------
    // Step 5. Check if the selected path is in fact the best path so far

    // Computing sequence squared distance
    float sum_of_squares = parent->residual_left_of_window;
    for (int flow = parent->window_start; flow < max_flows; flow++) {
      float residual = read.normalized_measurements[flow] - parent->prediction[flow];
      sum_of_squares += residual * residual;
    }

    // Updating best path
    if (sum_of_squares < sum_of_squares_upper_bound) {
      read.prediction.swap(parent->prediction);
      read.solution.swap(parent->solution);
      sum_of_squares_upper_bound = sum_of_squares;
    }

    parent->in_use = false;
    space_on_stack++;

  } // main decision loop
}



// ------------------------------------------------------------------------
// Compute quality metrics
int DPTreephaser::ComputeQVmetrics(BasecallerRead& read)
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




