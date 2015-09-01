/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#ifndef ION_ANALYSIS_VITERBI_HPP
#define ION_ANALYSIS_VITERBI_HPP

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <boost/math/distributions/poisson.hpp>
#include <boost/algorithm/minmax_element.hpp>

#define MAX_STATE_VALUE 100000

bool markov_chain_comparator::operator()(const markov_state& lhs, const markov_state& rhs) {
  // special comparator functor to implement state selection in building markov chain
  if (current) {
    double lhs_cost = lhs.prev + current->cost(lhs),
        rhs_cost = rhs.prev + current->cost(rhs);
    return lhs_cost > rhs_cost;
  }
  return lhs.current > rhs.current;
}

double markov_state::cost(double v) {
  // cost of being in the state under Poisson distribution log pdf to make is additive
  double rounded_v = min(v, (double)MAX_STATE_VALUE);
  // increase cost of small values to report spikes down in gvcf
  return rounded_v < 0.75 * dist.mean() ? -numeric_limits<double>::infinity() : log(boost::math::pdf(dist, rounded_v));
}

template<typename T>
template<typename _ForwardIterator>
markov_chain<T>::markov_chain(_ForwardIterator begin, _ForwardIterator end) {
  pair<_ForwardIterator, _ForwardIterator> m = boost::minmax_element(begin, end);
  initialize(begin, end, *(m.first), *(m.second));
}

template<typename T>
template<typename _ForwardIterator>
markov_chain<T>::markov_chain(_ForwardIterator begin, _ForwardIterator end, T min, T max) {
  initialize(begin, end, min, max);
}

template<typename T>
template<typename _ForwardIterator>
void markov_chain<T>::initialize(_ForwardIterator begin, _ForwardIterator end, T min, T max) {
  // initialize a constructor delegation method
  size_t s = (size_t)distance(begin, end);
  T dmax = (T) max < MAX_STATE_VALUE ? max : MAX_STATE_VALUE,
    dmin = (T) min < MAX_STATE_VALUE ? min : MAX_STATE_VALUE;
  if (dmin == dmax) {
    T avg = (T) round(accumulate(begin, end, 0.0) / s);
    items.push_back(make_pair(depth_info<T>(avg, min, max), s - 1));
    return;
  }
  values.reserve(s);
  reserve(s);
  generate_states(dmin, dmax);
  for (_ForwardIterator it = begin; it != end; it++)
    process_next_value(*it);
  optimal_path();
}

template<typename T>
void markov_chain<T>::generate_states(T min, T max) {
  // prepare state values and state objects for viterbi execution
  double dmax = (double) max,
      dmin = (double) min,
      alpha = 2,
      val;
  val = ceil(pow(dmin, 1 / alpha));

  states.push_back(dmin);
  for (double st = pow(val, alpha); st < dmax; st = pow(val, alpha), val++) {
    states.push_back(st);
  }
  states.push_back(dmax);
}

template<typename T>
void markov_chain<T>::process_next_value(T val) {
  // accept next value and fill out column in table of states
  vector<long> states_trace;
  states_trace.reserve(states.size());
  for (vector<markov_state>::iterator st = states.begin(); st != states.end(); st++) {
    markov_state& current = *st;
    markov_chain_comparator mccmp = markov_chain_comparator(current);
    vector<markov_state>::iterator min_prev = min_element(states.begin(), states.end(), mccmp);
    states_trace.push_back(min_prev - states.begin());
    double new_penalty = min_prev->prev + current.cost(*min_prev) + current.cost(val);
    st->current = new_penalty;
  }
  for (vector<markov_state>::iterator st = states.begin(); st != states.end(); st++) st->step_forward();
  push_back(states_trace);
  values.push_back(val);
}

template<typename T>
void markov_chain<T>::optimal_path() {
  // find optimal state switches and generate items for a chain

  double mean = 0;
  int i = 0;

  items.reserve(size());
  long state_index = min_element(states.begin(), states.end(), markov_chain_comparator()) - states.begin();

  markov_chain::reverse_iterator it = rbegin();
  typename vector<T>::reverse_iterator vit = values.rbegin();
  items.push_back(make_pair(depth_info<T>(*vit), size() - 1));
  for (; it != rend(), vit != values.rend(); it++, vit++) {
    mean += *vit;
    i++;
    if (vit == values.rend() - 1) {
      items.back().first.dp = (T)round(mean / i);
    } else if ((*it)[state_index] != state_index){
      items.back().first.dp = (T)round(mean / i);
      mean = 0;
      i = 0;
      items.push_back(make_pair(depth_info<T>(*(vit+1)), rend() - it - 2));
    } else {
      depth_info<T>& cdi = items.back().first;
      cdi.max_dp = max(*vit, cdi.max_dp);
      cdi.min_dp = min(*vit, cdi.min_dp);
    }
    state_index = (*it)[state_index];
  }
}

#endif //ION_ANALYSIS_VITERBI_HPP
