/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     DPTreephaser.cpp
//! @ingroup  BaseCaller
//! @brief    DPTreephaser. Perform dephasing and call base sequence by tree search

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "DPTreephaser.h"
#include "BaseCallerUtils.h"
#include "PIDloop.h"

// CLANG fix
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2937.html#454
// 454. When is a definition of a static data member required?
const int  DPTreephaser::kMinWindowSize_;
const int  DPTreephaser::kMaxWindowSize_;
constexpr char DPTreephaser::nuc_int_to_char_[];

DPTreephaser::DPTreephaser()
    : my_cf_(-1.0), my_ie_(-1.0), my_dr_(-1.0), flow_order_("TCAG", 4)
{
  SetNormalizationWindowSize(kWindowSizeDefault_);
}


DPTreephaser::DPTreephaser(const ion::FlowOrder& flow_order, const int windowSize){
  SetFlowOrder(flow_order);
  SetNormalizationWindowSize(windowSize);
}

//-------------------------------------------------------------------------

void DPTreephaser::SetFlowOrder(const ion::FlowOrder& flow_order)
{
  flow_order_ = flow_order;
  for (int i = 0; i < 8; i++) {
    transition_base_[i].resize(flow_order_.num_flows());
    transition_flow_[i].resize(flow_order_.num_flows());
  }
  path_.resize(kNumPaths);
  for (int p = 0; p < kNumPaths; ++p) {
    path_[p].state.resize(flow_order_.num_flows());
    path_[p].prediction.resize(flow_order_.num_flows());
    path_[p].sequence.reserve(2*flow_order_.num_flows());
    path_[p].calibA.assign(flow_order_.num_flows(), 1.0f);
  }
  pm_model_available_              = false;
  recalibrate_predictions_         = false;
  skip_recal_during_normalization_ = false;
  diagonal_states_        = false;
  
  my_cf_ = -1.0;
  my_ie_ = -1.0;
  my_dr_ = -1.0;
}

//-------------------------------------------------------------------------

void  DPTreephaser::ResetRecalibrationStructures() {
  for (int p = 0; p < kNumPaths; ++p) {
	path_[p].calibA.assign(flow_order_.num_flows(), 1.0f);
  }
}

//-------------------------------------------------------------------------

void DPTreephaser::SetModelParameters(double carry_forward_rate, double incomplete_extension_rate, double droop_rate)
{
  if (carry_forward_rate == my_cf_ and incomplete_extension_rate == my_ie_ and droop_rate == my_dr_)
    return;
  
  double nuc_avaliability[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    nuc_avaliability[flow_order_[flow]&7] = 1;
    for (int nuc = 0; nuc < 8; nuc++) {
      transition_base_[nuc][flow] = nuc_avaliability[nuc] * (1-droop_rate) * (1-incomplete_extension_rate);
      transition_flow_[nuc][flow] = (1-nuc_avaliability[nuc]) + nuc_avaliability[nuc] * (1-droop_rate) * incomplete_extension_rate;
      nuc_avaliability[nuc] *= carry_forward_rate;
    }
  }
  my_cf_ = carry_forward_rate;
  my_ie_ = incomplete_extension_rate;
  my_dr_ = droop_rate;
}

//-------------------------------------------------------------------------

void DPTreephaser::SetModelParameters(double carry_forward_rate, double incomplete_extension_rate)
{
  if (carry_forward_rate == my_cf_ and incomplete_extension_rate == my_ie_ and my_dr_ == 0.0)
    return;
  
  double nuc_avaliability[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    nuc_avaliability[flow_order_[flow]&7] = 1;
    for (int nuc = 0; nuc < 8; nuc++) {
      transition_base_[nuc][flow] = nuc_avaliability[nuc] * (1-incomplete_extension_rate);
      transition_flow_[nuc][flow] = 1 - transition_base_[nuc][flow];
      nuc_avaliability[nuc] *= carry_forward_rate;
    }
  }
  my_cf_ = carry_forward_rate;
  my_ie_ = incomplete_extension_rate;
  my_dr_ = 0.0;
}


//-------------------------------------------------------------------------
// Load input data without any normalization

void BasecallerRead::SetData(const vector<float> &measurements, int num_flows) {

  raw_measurements = measurements;
  raw_measurements.resize(num_flows, 0);
  for (int iFlow = 0; iFlow < num_flows; iFlow++) {
    if (isnan(measurements[iFlow])) {
      std::cerr << "Warning: Basecaller Read: NAN in measurements!"<< std::endl;
      raw_measurements.at(iFlow) = 0;
    }
  }

  key_normalizer = 1.0f;
  normalized_measurements = raw_measurements;
  sequence.clear();
  sequence.reserve(2*num_flows);
  prediction.assign(num_flows, 0);
  state_inphase.assign(num_flows, 1.0);
  state_total.assign(num_flows, 1.0);
  additive_correction.assign(num_flows, 0);
  multiplicative_correction.assign(num_flows, 1.0);
}

//-------------------------------------------------------------------------

bool BasecallerRead::SignalKeyPass(const vector<float> & measurements, int num_flows, const int *key_flows, int num_key_flows){

   if (num_flows < num_key_flows)
     return false;

   bool key_pass = false;
   int flow = 0;
   while (flow < num_key_flows and (int)round(measurements[flow]) == key_flows[flow])
     ++flow;

   return (flow == num_key_flows);
}

//-------------------------------------------------------------------------

bool BasecallerRead::SetDataAndKeyPass(const vector<float> &measurements, int num_flows, const int *key_flows, int num_key_flows)
{
  SetData(measurements, num_flows);
  return SignalKeyPass(raw_measurements, num_flows, key_flows, num_key_flows);
}

//-------------------------------------------------------------------------

bool BasecallerRead::SetDataAndKeyNormalize(const float *measurements, int num_flows, const int *key_flows, int num_key_flows)
{
  raw_measurements.resize(num_flows);
  normalized_measurements.resize(num_flows);
  prediction.assign(num_flows, 0);
  state_inphase.assign(num_flows, 1.0);
  state_total.assign(num_flows, 1.0);
  additive_correction.assign(num_flows, 0);
  multiplicative_correction.assign(num_flows, 1.0);
  sequence.clear();
  sequence.reserve(2*num_flows);

  int   zeromer_count = 0;
  float zeromer_sum   = 0.0f;
  float onemer_sum   = 0.0f;
  int   onemer_count = 0;

  for (int flow=0; flow < num_key_flows; ++flow)
  {
      if (key_flows[flow] == 0) {
          zeromer_sum += measurements[flow];
          ++zeromer_count;
      }
      if (key_flows[flow] == 1) {
          onemer_sum += measurements[flow];
          ++onemer_count;
      }
  }

  key_normalizer = 1.0f;
  if (onemer_sum > 0.3 and onemer_count)
    key_normalizer = static_cast<float>(onemer_count) / onemer_sum;

  // Safeguard against crazy values
  for (int flow = 0; flow < num_flows; ++flow) {
    raw_measurements[flow] = min(max(measurements[flow] * key_normalizer, -1.0f), (float)MAX_HPXLEN);
    normalized_measurements[flow] = raw_measurements[flow];
  }

  return SignalKeyPass(raw_measurements, num_flows, key_flows, num_key_flows);
}

//-------------------------------------------------------------------------

bool BasecallerRead::SetDataAndKeyNormalizeNew(const float *measurements, int num_flows, const int *key_flows, int num_key_flows, const bool phased)
{
    raw_measurements.resize(num_flows);
    normalized_measurements.resize(num_flows);
    prediction.assign(num_flows, 0.0f);
    state_inphase.assign(num_flows, 1.0f);
    state_total.assign(num_flows, 1.0);
    additive_correction.assign(num_flows, 0.0f);
    multiplicative_correction.assign(num_flows, 1.0f);
    sequence.clear();
    sequence.reserve(2*num_flows);

    // New key normalization
    float zeromer_sum   = 0.0f;
    int   zeromer_count = 0;
    float onemer_sum    = 0.0f;
    int   onemer_count  = 0;
    int   flow          = 0;
    for (; flow < num_key_flows; ++flow)
    {
        if (key_flows[flow] == 0)
        {
            zeromer_sum += measurements[flow];
            ++zeromer_count;
        }
        if (key_flows[flow] == 1)
        {
            onemer_sum += measurements[flow];
            ++onemer_count;
        }
    }

    float zeromerMean = (zeromer_count ? zeromer_sum / static_cast<float>(zeromer_count) : 0.0f);
    float onemerMean  = (onemer_count ? onemer_sum / static_cast<float>(onemer_count) : 1.0f);
    key_normalizer = (onemerMean - zeromerMean) > 0.25f ? 1.0f / (onemerMean - zeromerMean) : 1.0f;  // Guard against silly values

    // Key-normalize entire flow using global averages
    for (flow = 0; flow < num_flows; ++flow)
    {
        raw_measurements[flow] = min(max(((measurements[flow] - zeromerMean) * key_normalizer), -1.0f), (float)MAX_HPXLEN);
        normalized_measurements[flow] = raw_measurements[flow];
    }

    // Ben's code
    if (phased)
    {
        // Calculate statistics for flow zeromers and onemers from first 32 flows post key
        int maxIdx = min(num_flows, num_key_flows + 32);
        int zeromerIdxs[32];
        int onemerIdxs[32];
        int zeromerCount = 0;
        int onemerCount  = 0;
        for (flow = num_key_flows; flow < maxIdx; ++flow)
        {
            if ((raw_measurements[flow] > kZeromerMin) && (raw_measurements[flow] < kZeromerMax))
            {
                zeromerIdxs[zeromerCount] = flow;
                ++zeromerCount;
            }
            if ((raw_measurements[flow] > kOnemerMin) && (raw_measurements[flow] < kOnemerMax))
            {
                onemerIdxs[onemerCount] = flow;
                ++onemerCount;
            }
        }
        // Calculate means
        zeromer_sum = 0.0f;
        for (flow = 0; flow < zeromerCount; ++flow)
        {
            zeromer_sum += raw_measurements[zeromerIdxs[flow]];
        }
        zeromerMean = (zeromerCount ? zeromer_sum / static_cast<float>(zeromerCount) : kZeromerMean);
        onemer_sum = 0.0f;
        for (flow = 0; flow < onemerCount; ++flow)
        {
            onemer_sum += raw_measurements[onemerIdxs[flow]];
        }
        onemerMean = (onemerCount ? onemer_sum / static_cast<float>(onemerCount) : kOnemerMean);
        // Calculate sigma squareds
        float zeromerSigSq = kRunZeroSigSq;
        float onemerSigSq  = kRunOneSigSq;
        float delta;
        for (flow = 0; flow < zeromerCount; ++flow)
        {
            delta = raw_measurements[zeromerIdxs[flow]] - zeromerMean;
            zeromerSigSq += delta * delta;
        }
        if (zeromerCount)
            zeromerSigSq /= static_cast<float>(zeromerCount);
        for (flow = 0; flow < onemerCount; ++flow)
        {
            delta = raw_measurements[onemerIdxs[flow]] - onemerMean;
            onemerSigSq += delta * delta;
        }
        if (onemerCount)
            onemerSigSq /= static_cast<float>(onemerCount);
        // Correct zeromer and onemer estimates
        float oneOnSigSq = (zeromerSigSq > 0.0001f ? 1.0f / zeromerSigSq : 0.0f);
        zeromerMean = ((kZeromerMean * kInvZeroSigSq) + (zeromerMean * oneOnSigSq)) / (kInvZeroSigSq + oneOnSigSq);
        oneOnSigSq  = (onemerSigSq > 0.0001f ? 1.0f / onemerSigSq : 0.0f);
        onemerMean  = ((kOnemerMean * kInvOneSigSq) + (onemerMean * oneOnSigSq)) / (kInvOneSigSq + oneOnSigSq);
        // Normalize all non-key flows
        float flowGain = (onemerMean > 0.3f ? 1.0f / onemerMean : 1.0f);
        for (flow = num_key_flows; flow < num_flows; ++flow)
        {
            raw_measurements[flow] = (raw_measurements[flow] - kZeromerMean) * flowGain;
            normalized_measurements[flow] = raw_measurements[flow];
        }
    }
    return SignalKeyPass(raw_measurements, num_flows, key_flows, num_key_flows);
}

// ----------------------------------------------------------------------
// New normalization strategy

void DPTreephaser::WindowedNormalize(BasecallerRead& read, int num_steps, int window_size, const bool normalize_predictions) const
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
      if (normalize_predictions) {
        read.prediction[apply_flow] = (read.prediction[apply_flow] * normalizer) + read.additive_correction[apply_flow];
        read.normalized_measurements[apply_flow] = read.raw_measurements[apply_flow];
      }
      else
        read.normalized_measurements[apply_flow] /= normalizer;

      read.multiplicative_correction[apply_flow] = normalizer;
      normalizer += delta;
    }
  }

  for (; apply_flow < num_flows; ++apply_flow) {
    if (normalize_predictions) {
      read.prediction[apply_flow] = (read.prediction[apply_flow] * next_normalizer) + read.additive_correction[apply_flow];
      read.normalized_measurements[apply_flow] = read.raw_measurements[apply_flow];
    }
    else
      read.normalized_measurements[apply_flow] /= next_normalizer;

    read.multiplicative_correction[apply_flow] = next_normalizer;
  }
}

//-------------------------------------------------------------------------
// PID loop based normalization
void DPTreephaser::PIDNormalize(BasecallerRead& read, const int num_samples)
{
    int   num_flows = read.raw_measurements.size();
    int   idx       = 0;
    float rawVal, preVal, filtVal, normVal;

    pidOffset_.Initialize(0.0f);
    pidGain_.Initialize(1.0f);

    // Cacluate and apply offset and gain corrections
    for (idx = 0; idx < num_samples; ++idx)
    {
        rawVal = read.raw_measurements[idx];
        preVal = read.prediction[idx];
        // Offset correction
        filtVal = (preVal < 0.3f ? pidOffset_.Step(rawVal - preVal) : pidOffset_.Step());
        normVal = rawVal - filtVal;
        read.additive_correction[idx] = filtVal;
        // Gain correction
        filtVal = (preVal > 0.5f && preVal <= 4.0f && normVal > 0.0f ? pidGain_.Step(normVal / preVal) : pidGain_.Step());
        read.normalized_measurements[idx]   = normVal / filtVal;
        read.multiplicative_correction[idx] = filtVal;
    }

    // Copy any un-corrected samples
    for (; idx < num_flows; ++idx)
    {
        read.normalized_measurements[idx]   = read.raw_measurements[idx];
        read.additive_correction[idx]       = 0.0f;
        read.multiplicative_correction[idx] = 1.0f;
    }
}

// PID loop based normalization used during phase estimation (gain only)
float DPTreephaser::PIDNormalize(BasecallerRead& read, const int start_flow, const int end_flow)
{
    int   num_flows = end_flow - start_flow;
    int   idx       = start_flow;
    float rawVal, preVal, filtVal;
    float sumGain = 0.0f;

    // Find the first "good" flow before the target window and use to initialize gain PID
    pidGain_.Initialize(1.0f);

    // Cacluate mean gain correction for window and gain correct data
    for (idx = 0; idx < (int)read.raw_measurements.size(); ++idx)
    {
        rawVal   = read.raw_measurements[idx];
        preVal   = read.prediction[idx];
        filtVal  = (preVal > 0.5f && preVal <= 4.0f && rawVal > 0.0f ? pidGain_.Step(rawVal / preVal) : pidGain_.Step());
        if (idx >= start_flow && idx < end_flow)
            sumGain += filtVal;
        read.additive_correction[idx]       = 0.0f;
        read.normalized_measurements[idx]   = rawVal / filtVal;
        read.multiplicative_correction[idx] = filtVal;
    }

    return (num_flows ? sumGain / static_cast<float>(num_flows) : 1.0f);
}

//-------------------------------------------------------------------------
// New improved normalization strategy
void DPTreephaser::NormalizeAndSolve_Adaptive(BasecallerRead& read, int max_flows)
{
  int window_size = windowSize_;
  int solve_flows = 0;
  // Disable recalibration during normalization stage if requested
  if (skip_recal_during_normalization_)
    recalibrate_predictions_ = false;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {
    solve_flows = min((num_steps+1) * window_size, max_flows);

    Solve(read, solve_flows);
    WindowedNormalize(read, num_steps, window_size);
  }

  // And turn it back on (if available) for the final solving part
  EnableRecalibration();
  Solve(read, max_flows);
}


// Old normalization, but uses BasecallerRead object
void DPTreephaser::NormalizeAndSolve_GainNorm(BasecallerRead& read, int max_flows)
{
  // Disable recalibration during normalization stage if requested
  if (skip_recal_during_normalization_)
    recalibrate_predictions_ = false;

  for (int iter = 0; iter < 7; ++iter) {
    int solve_flow = 100 + 20 * iter;
    if (solve_flow < max_flows) {
      Solve(read, solve_flow);
      Normalize(read, 11, solve_flow-20);
    }
  }
  // And turn it back on (if available) for the final solving part
  EnableRecalibration();
  Solve(read, max_flows);
}


// Sliding window adaptive normalization
void DPTreephaser::NormalizeAndSolve_SWnorm(BasecallerRead& read, int max_flows)
{
  int window_size = windowSize_;
  int solve_flows = 0;

  // Disable recalibration during normalization stage if requested
  if (skip_recal_during_normalization_)
    recalibrate_predictions_ = false;

  for (int num_steps = 1; solve_flows < max_flows; ++num_steps) {
    solve_flows = min((num_steps+1) * window_size, max_flows);
    int restart_flows = max(solve_flows-100, 0);

    Solve(read, solve_flows, restart_flows);
    WindowedNormalize(read, num_steps, window_size);
  }

  // And turn it back on (if available) for the final solving part
  EnableRecalibration();
  Solve(read, max_flows);
}


//-------------------------------------------------------------------------

float DPTreephaser::Normalize(BasecallerRead& read, int start_flow, int end_flow) const
{
  float xy = 0;
  float yy = 0;
  int num_flows = read.raw_measurements.size();
  int norm_flows = 0;

  for (int flow = start_flow; flow < end_flow and flow < num_flows; ++flow) {
    if (read.prediction[flow] > 0.5 and read.prediction[flow] <= 4) {
      xy += read.raw_measurements[flow];
      yy += read.prediction[flow];
      ++norm_flows;
    }
  }

  float divisor = 1;
  if (xy > 0 and yy > 0 and norm_flows > 4)
    divisor = xy / yy;

  // range sanity check - key normalization should not be too bad
  if (divisor < 0.1 or divisor > 10)
    divisor = 1;


  for (int flow = 0; flow < num_flows; ++flow)
    read.normalized_measurements[flow] = min(max(read.raw_measurements[flow] / divisor, -1.0f), (float)MAX_HPXLEN);

  read.additive_correction.assign(num_flows, 0);
  read.multiplicative_correction.assign(num_flows, divisor);

  return divisor;
}



//-------------------------------------------------------------------------
// XXX

void DPTreephaser::InitializeState(TreephaserPath *state) const
{
  state->flow = 0;
  state->state[0] = 1;
  state->window_start = 0;
  state->window_end = 1;
  state->prediction.assign(flow_order_.num_flows(), 0.0);
  state->sequence.clear();
  state->sequence.reserve(2*flow_order_.num_flows());
  state->last_hp = 0;
}

//-------------------------------------------------------------------------

void DPTreephaser::AdvanceFlow(int & flow, char nuc, int max_flow) const
{
  while (flow < max_flow and flow_order_[flow] != nuc)
    ++flow;
}

//-------------------------------------------------------------------------

// This is not meant to incorporate a diagonal shift.
void DPTreephaser::AdvanceOnlyState(TreephaserPath *child, const TreephaserPath *parent, char nuc, int max_flow) const
{
  assert (child != parent);
  if (child->state.size() != parent->state.size())
    child->state.assign(parent->state.size(), 0.0f);

  // enable diagonal state movement if we want to limit HPs to size 1
  int diagonal_shift = 0;
  if (diagonal_states_ and parent->sequence.size()>0)
    diagonal_shift = 1;

  child->flow = parent->flow + diagonal_shift;
  AdvanceFlow(child->flow, nuc, max_flow);

  // Initialize window
  child->window_start = parent->window_start + diagonal_shift;
  child->window_end   = min(parent->window_end + diagonal_shift, max_flow);

  float max_pop = 0.0;
  child->max_population = 0;

  if (parent->flow != child->flow or parent->last_hp<1) {

    // This nuc begins a new homopolymer
    child->last_hp = 1;
    float alive = 0;
    child->state[parent->window_start] = 0;

    for (int flow = parent->window_start+diagonal_shift; flow < child->window_end; ++flow) {

      // State progression according to phasing model
      if ((flow-diagonal_shift) < parent->window_end)
        alive += parent->state[flow-diagonal_shift];
      child->state[flow] = alive * transition_base_[nuc&7][flow];
      alive *= transition_flow_[nuc&7][flow];
      if (child->state[flow] > max_pop){
        max_pop = child->state[flow];
        child->max_population = flow;
      }

      // Window maintenance
      if (flow == child->window_start and child->state[flow] < kStateWindowCutoff)
        child->window_start++;

      if (flow == child->window_end-1 and child->window_end < max_flow and alive > kStateWindowCutoff)
        child->window_end++;
    }

  } else {
    // This nuc simply prolongs the current homopolymer, it inherits the state from it's parent
    child->last_hp = parent->last_hp + 1;
    memcpy(&child->state[child->window_start], &parent->state[child->window_start],
        (child->window_end-child->window_start)*sizeof(float));
  }
}


//-------------------------------------------------------------------------

void DPTreephaser::AdvanceState(TreephaserPath *child, const TreephaserPath *parent, char nuc, int max_flow) const
{
  assert (child != parent);
  AdvanceOnlyState(child, parent, nuc, max_flow);

  // --- Maintaining recalibration data structures & logging coefficients for this path
  // Difference to sse version: Here we potentially recalibrate all HPs.
  int calib_hp   = min(child->last_hp, MAX_HPXLEN);
  if (recalibrate_predictions_) {
    child->calibA = parent->calibA;
    // Log zero mer flow coefficients
    for (int flow = parent->flow+1; flow < child->flow; flow++)
      child->calibA.at(flow) = (*As_).at(flow).at(flow_order_.int_at(flow)).at(0);
    if (child->flow < max_flow)
      child->calibA.at(child->flow) = (*As_).at(child->flow).at(flow_order_.int_at(child->flow)).at(calib_hp);
  }
  // ---

  // Transforming recalibration model into incremental HP model through inversion.
  // We assume here we don't ever have an offset coefficient for 0-mers.
  // XXX Note:
  // This recalibration method application differs slightly from the sse version,
  // where the residuals are calculated based on:
  //   recalibrated(parent->prediction[flow]) + child->state[flow] instead of
  //   recalibrated(parent->prediction[flow] + child->state[flow]) here

  for (int flow = parent->window_start; flow < parent->window_end; ++flow) {
    if (recalibrate_predictions_ and flow <= child->flow) {
      if (flow < child->flow  or child->last_hp>MAX_HPXLEN) {
        child->prediction[flow] = parent->prediction[flow] + (child->calibA[flow] * child->state[flow]);
      }
      else {
        // Inverse recalibration operation for active flow
        float original_prediction = parent->prediction.at(flow);
        if (child->last_hp > 1 and (*As_).at(flow).at(flow_order_.int_at(flow)).at(child->last_hp-1) > 0) {
        original_prediction = (parent->prediction.at(flow) - (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(child->last_hp-1))
                              / (*As_).at(flow).at(flow_order_.int_at(flow)).at(child->last_hp-1);
        }
        // Apply recalibration for the flow where we changed a base
        child->prediction[flow] = ( (original_prediction + child->state[flow]) * child->calibA.at(flow) )
   		                          + (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(calib_hp);
      }
    }
    else {
      // The simple no HP recalibration case
      child->prediction[flow] = parent->prediction[flow] + child->state[flow];
    }
  }
  for (int flow = parent->window_end; flow < child->window_end; ++flow) {
    if (recalibrate_predictions_ and flow <= child->flow) {
      child->prediction[flow] = child->state[flow] * child->calibA.at(flow);
      if (flow == child->flow)
        child->prediction[flow] += (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(calib_hp);
    }
    else {
      // The simple no HP recalibration case
      child->prediction[flow] = child->state[flow];
    }
  }
}

//-------------------------------------------------------------------------
void DPTreephaser::AdvanceStateInPlace(TreephaserPath *state, char nuc, int max_flow) const
{
  int old_flow = state->flow;
  // enable diagonal state movement
  if (diagonal_states_) {
	  if( state->sequence.size()>0){
		state->flow++;
		state->window_end = min(state->window_end + 1, max_flow);
		for (int flow=state->window_end-1; flow>state->window_start; flow--)
		  state->state[flow] = state->state[flow-1];
		state->state[state->window_start] = 0;
		state->window_start++;
	  }
  }
  int old_window_start = state->window_start;
  int old_window_end   = state->window_end;
  int sf = state->flow;


  // Advance in-phase flow
  while (sf < max_flow and flow_order_[sf] != nuc)
    sf++;
  state->flow=sf;

  if (sf == max_flow) // Immediately return if base does not fit any more
    return;

  if (old_flow == sf)
    state->last_hp++;
  else
    state->last_hp = 1;
  int calib_hp = min(state->last_hp, MAX_HPXLEN);

  // --- Maintaining recalibration data structures & logging coefficients for this path
  if (recalibrate_predictions_) {
    for (int flow = old_flow+1; flow < state->flow; flow++)
      state->calibA.at(flow) = (*As_).at(flow).at(flow_order_.int_at(flow)).at(0);
    state->calibA.at(state->flow) = (*As_).at(state->flow).at(flow_order_.int_at(state->flow)).at(calib_hp);

  // ---


	  if (old_flow != sf or old_flow == 0) {

		// This nuc begins a new homopolymer, need to adjust state
		float alive = 0;
		for (int flow = old_window_start; flow < state->window_end; flow++) {

		  // State progression according to phasing model
		  if (flow < old_window_end)
			alive += state->state[flow];
		  state->state[flow] = alive * transition_base_[nuc&7][flow];
		  alive *= transition_flow_[nuc&7][flow];

		  // Window maintenance
		  if (flow == state->window_start and state->state[flow] < kStateWindowCutoff)
			state->window_start++;

		  if (flow == state->window_end-1 and state->window_end < max_flow and alive > kStateWindowCutoff)
			state->window_end++;
		}
	  }

  // Create predictions through incremental homopolymer recalibration model
    for (int flow = old_window_start; flow < state->window_end; ++flow) {
		if (recalibrate_predictions_ and flow <= state->flow) {
		  if (flow < state->flow  or state->last_hp>MAX_HPXLEN) {
			  state->prediction[flow] += state->calibA[flow] * state->state[flow];
		  }
		  else {
			float original_prediction = state->prediction[flow];
			if (state->last_hp > 1 and (*As_).at(flow).at(flow_order_.int_at(flow)).at(state->last_hp-1) > 0) {
			  // Invert re-calibration operation for active flow
			  original_prediction = ( state->prediction[flow] - (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(state->last_hp-1) )
									/ (*As_).at(flow).at(flow_order_.int_at(flow)).at(state->last_hp-1);
			}
			// Apply recalibration for the flow where we changed a base
			state->prediction[flow] = ( (original_prediction + state->state[flow]) * state->calibA.at(flow) )
									  + (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(calib_hp);
		  }
		}
		else
		  state->prediction[flow] += state->state[flow];
	  }
  }
  else{
		float alive = 0;
		const float *tb=&transition_base_[nuc&7][0];
		const float *tf=&transition_flow_[nuc&7][0];
		float *cst =&state->state[0];
		float *csp =&state->prediction[0];
		int swe=state->window_end;
		int sws=state->window_start;
	  if (old_flow != sf or old_flow == 0) {

		// This nuc begins a new homopolymer, need to adjust state
		for (int flow = old_window_start; flow < swe; flow++) {

		  // State progression according to phasing model
		  if (flow < old_window_end)
			alive += cst[flow];
		  cst[flow] = alive * tb[flow];
		  alive *= tf[flow];
		  csp[flow] += cst[flow];

		  // Window maintenance
		  if (flow == sws and cst[flow] < kStateWindowCutoff)
			sws++;

		  if (flow == swe-1 and swe < max_flow and alive > kStateWindowCutoff)
			swe++;
		}
		state->window_end=swe;
		state->window_start=sws;
	  }
	  else{
	    for (int flow = old_window_start; flow < swe; ++flow) {
		  csp[flow] += cst[flow];
	    }
	  }
  }

}

//-------------------------------------------------------------------------

void DPTreephaser::Simulate(BasecallerRead& data, int max_flows,bool state_inphase)
{
	SimulateStateMatrix(data, max_flows, state_inphase, NULL, 0.0f);
}

//-------------------------------------------------------------------------

void DPTreephaser::SimulateStateMatrix(BasecallerRead& data, int max_flows,bool state_inphase, vector<vector<pair<int, float> > >* row_linked_state_matrix, float amp_cut_off)
{
  InitializeState(&path_[0]);
  int   recent_flow = 0;
  float recent_state_inphase = 1.0;
  bool is_output_state_matrix = (row_linked_state_matrix != NULL);
  if (is_output_state_matrix){
	  row_linked_state_matrix->reserve(data.sequence.size());
  }

  for (vector<char>::iterator nuc = data.sequence.begin(); nuc != data.sequence.end()
       and path_[0].flow < max_flows; ++nuc) {
    AdvanceStateInPlace(&path_[0], *nuc, flow_order_.num_flows());
    path_[0].sequence.push_back(*nuc); // Needed to simulate diagonal states correctly

    if (is_output_state_matrix){
      vector<pair<int, float> > sparse_matrix_row;
      sparse_matrix_row.reserve(path_[0].window_end - path_[0].window_start);
      for (int flow = path_[0].window_start; flow < path_[0].window_end; ++flow){
        if (path_[0].state[flow] > amp_cut_off){
    	  sparse_matrix_row.push_back(pair<int, float>(flow, path_[0].state[flow]));
        }
      }
      row_linked_state_matrix->push_back(sparse_matrix_row);
    }

    if (state_inphase and path_[0].flow < max_flows) {
      for (int iFlow=recent_flow+1; iFlow<path_[0].flow; iFlow++)
        data.state_inphase.at(iFlow) = recent_state_inphase;

      data.state_inphase.at(path_[0].flow) = path_[0].state.at(path_[0].flow);
      recent_flow = path_[0].flow;
      recent_state_inphase = path_[0].state.at(path_[0].flow);
    }
  }

  if (state_inphase){
    for (int iFlow=recent_flow+1; iFlow<max_flows; iFlow++)
      data.state_inphase.at(iFlow) = recent_state_inphase;
  }

  data.prediction.swap(path_[0].prediction);
}

//-------------------------------------------------------------------------
// Simluate all states including 1st gen. dead children XXX

void DPTreephaser::SimulateAllStates(vector<TreephaserPath> & states, BasecallerRead& read, int max_flows, bool debug)
{
  assert(max_flows <= flow_order_.num_flows());
  InitializeState(&path_[0]);
  TreephaserPath *parent = &path_[0];

  states.resize(flow_order_.num_flows());
  for (int iflow=0; iflow < flow_order_.num_flows(); ++iflow)
    states.at(iflow).Initialize();
  int child_flow = 0;

  for (vector<char>::const_iterator nuc = read.sequence.begin(); (nuc != read.sequence.end()) and (child_flow < max_flows); ++nuc) {

    int child_flow = parent->flow;
    AdvanceFlow(child_flow, *nuc, max_flows);

    // Update parent HP if we simply add another nuc
    if (child_flow == parent->flow and parent->last_hp > 0){
      ++parent->last_hp;
      continue;
    }

    for (int next_nuc_idx=0; next_nuc_idx<4; ++next_nuc_idx) {

      int temp_flow = parent->flow;
      AdvanceFlow(temp_flow, nuc_int_to_char_[next_nuc_idx], max_flows);
      if (temp_flow >= max_flows or states.at(temp_flow).last_hp > 0)
        continue; // Do not mess with parent HP

      if (temp_flow <= child_flow) {
        AdvanceOnlyState(&(states.at(temp_flow)), parent, nuc_int_to_char_[next_nuc_idx], max_flows);
        states.at(temp_flow).in_use = true;
        if (temp_flow < child_flow)
          states.at(temp_flow).last_hp  =  0; // Mark dead children
      }

    }

    if (child_flow < max_flows and child_flow < (int)states.size()) {
      parent = &(states.at(child_flow));
      parent->sequence.push_back(*nuc);
    }
  }

  // Now generate predictions from individual states
  // Need to initialize predictions since it may already contain values
  read.prediction.assign(flow_order_.num_flows(), 0.0f);
  int   my_hp = 0;
  float recent_state_inphase = 1.0;
  float recent_total_state = 1.0;

  for (int inphase_flow=0; inphase_flow < flow_order_.num_flows(); ++inphase_flow) {

    my_hp = states.at(inphase_flow).last_hp;
    if (my_hp == 0){
      continue;
    } else if (!states.at(inphase_flow).in_use){
      cout << inphase_flow << "SimulateAllStates: state " << inphase_flow <<"NOT IN USE" << endl;
      exit(EXIT_FAILURE);
    }

    if (my_hp > 0){
      recent_total_state = 0.0f;
      for (int flow = states.at(inphase_flow).window_start; flow < states.at(inphase_flow).window_end; ++flow){
        read.prediction.at(flow) += my_hp * states.at(inphase_flow).state.at(flow);
        recent_total_state += states.at(inphase_flow).state.at(flow);
      }
      recent_state_inphase = states.at(inphase_flow).state.at(states.at(inphase_flow).flow);
    }

    read.state_inphase.at(inphase_flow) = max(recent_state_inphase, 0.01f);
    read.state_total.at(inphase_flow)   = max(recent_total_state, 0.01f);

  }

  // And in a second pass do linear model calibration
  if (recalibrate_predictions_) {

    float calibA, calibB;
    int calib_hp;
    for (int flow = 0; flow < max_flows; ++flow){
      calib_hp = min(states.at(flow).last_hp, MAX_HPXLEN);
      calibA = (*As_).at(flow).at(flow_order_.int_at(flow)).at(calib_hp);
      calibB = (*Bs_).at(flow).at(flow_order_.int_at(flow)).at(calib_hp);
      read.prediction.at(flow) = (read.prediction[flow] * calibA) + calibB;
    }
  }

}

//-------------------------------------------------------------------------


void DPTreephaser::QueryState(BasecallerRead& data, vector<float>& query_state, int& current_hp, int max_flows, int query_flow)
{
  max_flows = min(max_flows,flow_order_.num_flows());
  assert(query_flow < max_flows);
  InitializeState(&path_[0]);
  query_state.assign(max_flows,0);
  char myNuc = 'N';

  for (vector<char>::iterator nuc = data.sequence.begin(); nuc != data.sequence.end() and path_[0].flow <= query_flow; ++nuc) {
    if (path_[0].flow == query_flow and myNuc != 'N' and myNuc != *nuc)
      break;
    AdvanceStateInPlace(&path_[0], *nuc, flow_order_.num_flows());
    if (path_[0].flow == query_flow and myNuc == 'N')
      myNuc = *nuc;
  }

  // Catching cases where a query_flow without incorporation or query_flow after end of sequence was given
  int until_flow = min(path_[0].window_end, max_flows);
  if (path_[0].flow == query_flow) {
    current_hp = path_[0].last_hp;
    for (int flow = path_[0].window_start; flow < until_flow; ++flow)
      query_state[flow] = path_[0].state[flow];
  }
  else
    current_hp = 0;
}


void DPTreephaser::QueryAllStates(BasecallerRead& data, vector< vector<float> >& query_states, vector<int>& hp_lengths, int max_flows)
{
  max_flows = min(max_flows,flow_order_.num_flows());
  InitializeState(&path_[0]);
  max_flows = min(max_flows, flow_order_.num_flows());
  query_states.reserve(data.sequence.size());
  query_states.resize(0);
  vector<float> zero_vec(flow_order_.num_flows(), 0.0);
  hp_lengths.assign(data.sequence.size(), 0);
  char last_nuc = 'N';
  int hp_count = 0;
  int old_window_start = 0;

  for (vector<char>::iterator nuc = data.sequence.begin(); nuc != data.sequence.end() and path_[0].flow < max_flows; ++nuc) {
    if (last_nuc != *nuc and last_nuc != 'N') {
      hp_lengths.at(hp_count) = path_[0].last_hp;
      //query_states.push_back(path_[0].state);
      query_states.push_back(zero_vec);
      for(int iFlow = old_window_start; iFlow<path_[0].window_end; iFlow++)
        query_states.back().at(iFlow) = path_[0].state.at(iFlow);
      hp_count++;
    }
    old_window_start = path_[0].window_start;
    AdvanceStateInPlace(&path_[0], *nuc, max_flows);
    last_nuc = *nuc;
  }
  hp_lengths[hp_count] = path_[0].last_hp;
  //query_states.push_back(path_[0].state);
  query_states.push_back(zero_vec);
  for(int iFlow = old_window_start; iFlow<path_[0].window_end; iFlow++)
    query_states.back().at(iFlow) = path_[0].state.at(iFlow);
  hp_lengths.resize(query_states.size());
  data.prediction.swap(path_[0].prediction);
}


//-------------------------------------------------------------------------

void DPTreephaser::Solve(BasecallerRead& read, int max_flows, int restart_flows)
{
  //static const char nuc_int_to_char[5] = "ACGT";
  assert(max_flows <= flow_order_.num_flows());
  assert(max_flows > 0);

  // Initialize stack: just one root path
  if(recalibrate_predictions_)
    ResetRecalibrationStructures();
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

    restart_flows = min(restart_flows, max_flows);

    for (vector<char>::iterator nuc = read.sequence.begin();
          nuc != read.sequence.end() and path_[0].flow < restart_flows; ++nuc) {
      AdvanceStateInPlace(&path_[0], *nuc, flow_order_.num_flows());
      if (path_[0].flow < max_flows)
        path_[0].sequence.push_back(*nuc);
    }

    if (path_[0].flow < restart_flows-10 or path_[0].flow >= max_flows) { // This read ended before restart_flows. No point resolving it.
      read.prediction.swap(path_[0].prediction);
      return;
    }

    for (int flow = 0; flow < path_[0].window_start; ++flow) {
      float residual = read.normalized_measurements[flow] - path_[0].prediction[flow];
      path_[0].residual_left_of_window += residual * residual;
    }
  }

  // Initializing variables
  read.sequence.clear();
  read.sequence.reserve(2*flow_order_.num_flows());
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

      AdvanceState(child, parent, nuc_int_to_char_[nuc], max_flows);

      // Apply easy termination rules

      if (child->flow >= max_flows) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      if (child->last_hp > kMaxHP) {
        penalty[nuc] = 25; // Mark for deletion
        continue;
      }

      if ((int)parent->sequence.size() >= (2 * flow_order_.num_flows() - 10)) {
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

      if (child->flow>0)
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
      memcpy(&child->prediction[0], &parent->prediction[0], (parent->window_start)*sizeof(float));

      for (int flow = child->window_end; flow < max_flows; ++flow)
        child->prediction[flow] = 0;

      // Fill out the solution
      child->sequence = parent->sequence;
      child->sequence.push_back(nuc_int_to_char_[nuc]);
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
      read.sequence.swap(parent->sequence);
      sum_of_squares_upper_bound = sum_of_squares;
    }

    parent->in_use = false;
    space_on_stack++;

  } // main decision loop
}


// ------------------------------------------------------------------------
// Compute quality metrics

void  DPTreephaser::ComputeQVmetrics(BasecallerRead& read)
{
  //static const char nuc_int_to_char[5] = "ACGT";
  //read.state_total.assign(flow_order_.num_flows(), 1);

  if (read.sequence.empty())
    return;
  int sz = (int)read.sequence.size();
	  
  read.penalty_mismatch.assign(sz, 0);
  read.penalty_residual.assign(sz, 0);

  TreephaserPath *parent = &path_[0];
  TreephaserPath *children[4] = { &path_[1], &path_[2], &path_[3], &path_[4] };

  if(recalibrate_predictions_)
    ResetRecalibrationStructures();
  InitializeState(parent);

  float recent_state_inphase = 1;
  float recent_state_total = 1;

  // main loop for base calling
  for (int solution_flow = 0, base = 0; solution_flow < flow_order_.num_flows(); ++solution_flow) {
    for (; ; ++base) {
      float penalty[4] = { 0, 0, 0, 0 };

      int called_nuc = 0;

      for (int nuc = 0; nuc < 4; nuc++) {

        TreephaserPath *child = children[nuc];

        AdvanceState(child, parent, nuc_int_to_char_[nuc], flow_order_.num_flows());

        if (nuc_int_to_char_[nuc] == flow_order_[solution_flow])
          called_nuc = nuc;

        // Apply easy termination rules

        if (child->flow >= flow_order_.num_flows()) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        if (child->last_hp > kMaxHP) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        if ((int)parent->sequence.size() >= (2 * flow_order_.num_flows() - 10)) {
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

      for (int flow = children[called_nuc]->window_end; flow < flow_order_.num_flows(); ++flow)
        children[called_nuc]->prediction[flow] = 0;

      // Called state is the starting point for next base
      TreephaserPath *swap = parent;
      parent = children[called_nuc];
      children[called_nuc] = swap;

    }

    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
    read.state_total[solution_flow] = max(recent_state_total, 0.01f);
  }

  read.prediction.swap(parent->prediction);

}


void  DPTreephaser::ComputeQVmetrics_flow(BasecallerRead& read, vector<int>& flow_to_base, const bool flow_predictors_)
{
  //static const char nuc_int_to_char[5] = "ACGT";
  //read.state_total.assign(flow_order_.num_flows(), 1);

  if (read.sequence.empty())
    return;
  int sz = (int)read.sequence.size();
  if (flow_predictors_)
      sz = flow_order_.num_flows();

  read.penalty_mismatch.assign(sz, 0);
  read.penalty_residual.assign(sz, 0);

  TreephaserPath *parent = &path_[0];
  TreephaserPath *children[4] = { &path_[1], &path_[2], &path_[3], &path_[4] };

  if(recalibrate_predictions_)
    ResetRecalibrationStructures();
  InitializeState(parent);

  float recent_state_inphase = 1;
  float recent_state_total = 1;

  // main loop for base calling
  for (int solution_flow = 0, base = 0; solution_flow < flow_order_.num_flows(); ++solution_flow) {
    for (; ; ++base) {
        if (flow_predictors_)
            break;
        else if (not (base < sz and read.sequence[base] == flow_order_[solution_flow]))
            break;

      float penalty[4] = { 0, 0, 0, 0 };

      int called_nuc = 0;

      for (int nuc = 0; nuc < 4; nuc++) {

        TreephaserPath *child = children[nuc];

        AdvanceState(child, parent, nuc_int_to_char_[nuc], flow_order_.num_flows());

        if (nuc_int_to_char_[nuc] == flow_order_[solution_flow])
          called_nuc = nuc;

        // Apply easy termination rules

        if (child->flow >= flow_order_.num_flows()) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        if (child->last_hp > kMaxHP) {
          penalty[nuc] = 25; // Mark for deletion
          continue;
        }

        if ((int)parent->sequence.size() >= (2 * flow_order_.num_flows() - 10)) {
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

      for (int flow = children[called_nuc]->window_end; flow < flow_order_.num_flows(); ++flow)
        children[called_nuc]->prediction[flow] = 0;

      // Called state is the starting point for next base
      TreephaserPath *swap = parent;
      parent = children[called_nuc];
      children[called_nuc] = swap;

    }

    read.state_inphase[solution_flow] = max(recent_state_inphase, 0.01f);
    read.state_total[solution_flow] = max(recent_state_total, 0.01f);
  }

  read.prediction.swap(parent->prediction);

}


// ----------------------------------------------------------------------------

void DPTreephaser::TreephaserPath::Initialize()
{
  in_use = false;
  flow = 0;
  state.clear();
  window_start = 0;
  window_end   = 1;
  prediction.clear();
  calibA.clear();
  sequence.clear();
  last_hp = 0;
  max_population = -1;
}


// =========================================================================== // XXX

DPPhaseSimulator::DPPhaseSimulator(const ion::FlowOrder flow_order)
: flow_order_(flow_order), ready_to_go_(false)
{

  num_hp_simulated_          = 0;
  kStateWindowCutoff_        = 1e-6;
  max_flow_                 = flow_order_.num_flows();


  // Initalize null state
  null_state_.flow = 0;
  null_state_.nuc  = 'X';
  null_state_.hp_length = 0;
  null_state_.window_start = 0;
  null_state_.window_end = 1;
  null_state_.state.assign(flow_order_.num_flows(), 0);
  null_state_.state.at(0) = 1.0;
  null_state_.index = -1;

  state_space_.resize(flow_order_.num_flows()+1);
  for (unsigned int iState=0; iState<state_space_.size(); iState++) {
    state_space_.at(iState).state.assign(flow_order_.num_flows(), 0);
    state_space_.at(iState).index = iState;
  }

  // Initalize transition probability vectors
  transition_base_.resize(flow_order_.num_flows());
  transition_flow_.resize(flow_order_.num_flows());
  for (unsigned int base=0;base<transition_base_.size(); base++){
    transition_base_.at(base).resize(4);
    transition_flow_.at(base).resize(4);
    for (unsigned int nuc=0; nuc<4; nuc++){
    	transition_base_.at(base).at(nuc).resize(flow_order_.num_flows());
    	transition_flow_.at(base).at(nuc).resize(flow_order_.num_flows());
    }
  }


  // default nuc availability - diagonal matrix
  nuc_availability_.resize(4);
  for (unsigned int nuc=0; nuc<4; nuc++){
    nuc_availability_.at(nuc).assign(4,0.0);
    nuc_availability_.at(nuc).at(nuc) = 0.0;
  }

}


// ---------------------------------------------------------------------------

void DPPhaseSimulator::SetBasePhasingParameters(const unsigned int base, const vector<double>& carry_forward_rates,
                                 const vector<double>& incomplete_extension_rates, const vector<double>& droop_rates)
{
  vector<double> nuc_avaliability(4,0.0);

  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    for (int nuc = 0; nuc < 4; nuc++) {
      nuc_avaliability.at(nuc) += nuc_availability_.at(flow_order_.int_at(flow)).at(nuc);
      nuc_avaliability.at(nuc) = min(nuc_avaliability.at(nuc), 1.0);

      transition_base_.at(base).at(nuc).at(flow) = nuc_avaliability[nuc] * (1-droop_rates.at(flow)) * (1-incomplete_extension_rates.at(flow));
      transition_flow_.at(base).at(nuc).at(flow) = (1-nuc_avaliability[nuc]) + nuc_avaliability[nuc] * (1-droop_rates.at(flow)) * incomplete_extension_rates.at(flow);

      nuc_avaliability.at(nuc) *= carry_forward_rates.at(flow);
    }
  }
  int sim_hp = 0;
  while(sim_hp < num_hp_simulated_ and state_space_.at(sim_hp).flow < (int)base)
    sim_hp++;
  num_hp_simulated_ = sim_hp;
}

void DPPhaseSimulator::SetBasePhasingParameters(const unsigned int base, const double carry_forward_rate,
		                             const double incomplete_extension_rate, const double droop_rate)
{
  vector<double> cf_vec(flow_order_.num_flows(), carry_forward_rate);
  vector<double> ie_vec(flow_order_.num_flows(), incomplete_extension_rate);
  vector<double> dr_vec(flow_order_.num_flows(), droop_rate);

  SetBasePhasingParameters(base, cf_vec, ie_vec, dr_vec);
}


// ---------------------------------------------------------------------------

void DPPhaseSimulator::SetPhasingParameters_TimeVarying(const vector<double>& carry_forward_rates,
                              const vector<double>& incomplete_extension_rates, const vector<double>& droop_rates)
{
  SetBasePhasingParameters(0, carry_forward_rates, incomplete_extension_rates, droop_rates);

  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    for (int nuc = 0; nuc < 4; nuc++) {
      // In basic model there is no change in transition probabilities
      for (unsigned int base=1; base<transition_base_.size(); base++){
        transition_base_.at(base).at(nuc).at(flow) = transition_base_.at(0).at(nuc).at(flow);
        transition_flow_.at(base).at(nuc).at(flow) = transition_flow_.at(0).at(nuc).at(flow);
      }
    }
  }
  ready_to_go_         = true;
  num_hp_simulated_ = 0;
}


// ---------------------------------------------------------------------------

void DPPhaseSimulator::SetPhasingParameters_Basic(double carry_forward_rate, double incomplete_extension_rate, double droop_rate)
{
  SetBasePhasingParameters(0, carry_forward_rate, incomplete_extension_rate, droop_rate);

  for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
    for (int nuc = 0; nuc < 4; nuc++) {
      // In basic model there is no change in transition probabilities
      for (unsigned int base=1; base<transition_base_.size(); base++){
        transition_base_.at(base).at(nuc).at(flow) = transition_base_.at(0).at(nuc).at(flow);
        transition_flow_.at(base).at(nuc).at(flow) = transition_flow_.at(0).at(nuc).at(flow);
      }
    }
  }
  ready_to_go_         = true;
  num_hp_simulated_ = 0;
}


// ---------------------------------------------------------------------------

void DPPhaseSimulator::SetPhasingParameters_Full(const vector<vector<double> >& carry_forward_rates,
                                                 const vector<vector<double> >& incomplete_extension_rates,
                                                 const vector<vector<double> >& droop_rates)
{
  for (unsigned int base=0; base<transition_base_.size(); base++) {
    SetBasePhasingParameters(base, carry_forward_rates.at(base), incomplete_extension_rates.at(base), droop_rates.at(base));
  }
  ready_to_go_ = true;
  num_hp_simulated_ = 0;
}


// ---------------------------------------------------------------------------

void DPPhaseSimulator::UpdateNucAvailability(const vector<vector<double> >& nuc_availability)
{
  assert(nuc_availability.size()==4);
  for (int nuc = 0; nuc < 4; nuc++) {
    assert(nuc_availability.at(nuc).size()==4);
  }
  nuc_availability_ = nuc_availability;
  ready_to_go_ = false;
  num_hp_simulated_ = 0;
}


//-------------------------------------------------------------------------

void DPPhaseSimulator::SetBaseSequence(const string& my_sequence) {
    base_sequence_ = my_sequence;
    num_hp_simulated_ = 0;
};


//-------------------------------------------------------------------------

void DPPhaseSimulator::GetStates(vector<vector<float> > & states, vector<int> & hp_lengths)
{
    states.resize(num_hp_simulated_);
    hp_lengths.resize(num_hp_simulated_);
    for (int iHP=0; iHP<num_hp_simulated_; iHP++){
      states.at(iHP).assign(flow_order_.num_flows(), 0.0);
      hp_lengths.at(iHP) = state_space_.at(iHP).hp_length;
      int iFlow = 0;
      if (iHP >0)
        iFlow = state_space_.at(iHP-1).window_start;

      for (; iFlow < state_space_.at(iHP).window_end; iFlow++)
    	  states.at(iHP).at(iFlow) = state_space_.at(iHP).state.at(iFlow);
    }
};


//-------------------------------------------------------------------------

void DPPhaseSimulator::SetMaxFlows(int max_flows) {
    max_flow_ = min(max_flow_, flow_order_.num_flows());
    int sim_hp = 0;
    while(sim_hp < num_hp_simulated_ and state_space_.at(sim_hp).flow < max_flow_)
      sim_hp++;
    num_hp_simulated_ = sim_hp;
};

//-------------------------------------------------------------------------

void DPPhaseSimulator::UpdateStates(int max_flow)
{
  if (not ready_to_go_) {
	cerr << "DPPhaseSimulator: Need a phasing model before simulation can happen." << endl;
	exit(1);
  }

  unsigned int nuc_index = 0;
  SimPathElement *parent, *child;


  if(num_hp_simulated_ == 0)
	parent = &null_state_;
  else
	parent = &state_space_.at(num_hp_simulated_-1);

  child = &state_space_.at(num_hp_simulated_);


  for(int iHP=0; iHP<num_hp_simulated_; iHP++)
    nuc_index += state_space_.at(iHP).hp_length;


  for (; nuc_index<base_sequence_.length(); nuc_index++) {

    if (base_sequence_.at(nuc_index) == parent->nuc) {
      parent->hp_length++;
    }
    else {

      AdvanceState(child, parent, base_sequence_.at(nuc_index), max_flow);

      if(child->flow < max_flow) {
        num_hp_simulated_++;
        child->nuc = base_sequence_.at(nuc_index);
        parent = child;
        child  = &state_space_.at(num_hp_simulated_);
      }
      else
        break;
    }

  }
};

//-------------------------------------------------------------------------

void DPPhaseSimulator::GetPredictions(vector<float>& predictions)
{
  predictions.assign(flow_order_.num_flows(), 0.0);
  for (int iHP=0; iHP<num_hp_simulated_; iHP++){

	int iFlow = 0;
	if (iHP>0)
	  iFlow = state_space_.at(iHP-1).window_start;

    for (; iFlow<state_space_.at(iHP).window_end; iFlow++)
      predictions.at(iFlow) += state_space_.at(iHP).hp_length * state_space_.at(iHP).state.at(iFlow);
  }
};

//-------------------------------------------------------------------------

void DPPhaseSimulator::GetSimSequence(string& sim_sequence)
{
  sim_sequence.clear();
  sim_sequence.reserve(base_sequence_.length());
  for (int iHP=0; iHP<num_hp_simulated_; iHP++)
    for (int iNuc=0; iNuc<state_space_.at(iHP).hp_length ; iNuc++)
      sim_sequence += state_space_.at(iHP).nuc;
};


//-------------------------------------------------------------------------

void DPPhaseSimulator::Simulate(string& sim_sequence, vector<float>& predictions, int max_flows)
{
  SetBaseSequence(sim_sequence);
  SetMaxFlows(max_flows);
  UpdateStates(max_flow_);
  GetSimSequence(sim_sequence);
  GetPredictions(predictions);
};


//-------------------------------------------------------------------------

void DPPhaseSimulator::AdvanceState(SimPathElement *child, const SimPathElement *parent, char nuc, int max_flow) const
{
  assert (child != parent);

  // Advance flow
  child->flow = parent->flow;
  while (child->flow < max_flow and flow_order_[child->flow] != nuc)
    child->flow++;

  if (child->flow == parent->flow)
    child->hp_length = parent->hp_length + 1;
  else
    child->hp_length = 1;

  // Initialize window
  child->window_start = parent->window_start;
  child->window_end   = min(parent->window_end, max_flow);

  if (parent->flow != child->flow or parent->flow == 0) {

    // This nuc begins a new homopolymer
    float alive = 0;
    child->state.at(parent->window_start) = 0;

    for (int flow = parent->window_start; flow < child->window_end; ++flow) {

      // State progression according to phasing model
      if ((flow) < parent->window_end)
        alive += parent->state.at(flow);
      child->state.at(flow) = alive * transition_base_.at(child->flow).at(flow_order_.NucToInt(nuc)).at(flow);
      alive *= transition_flow_.at(child->flow).at(flow_order_.NucToInt(nuc)).at(flow);

      // Window maintenance
      if (flow == child->window_start and child->state.at(flow) < kStateWindowCutoff_)
        child->window_start++;

      if (flow == child->window_end-1 and child->window_end < max_flow and alive > kStateWindowCutoff_)
        child->window_end++;
    }

  } else {
    // This nuc simply prolongs the current homopolymer, it inherits the state from it's parent
    memcpy(&child->state.at(child->window_start), &parent->state.at(child->window_start),
        (child->window_end-child->window_start)*sizeof(float));
  }
};

