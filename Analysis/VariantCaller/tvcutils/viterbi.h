/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

/* Author: Alex Artyomenko <aartyomenko@cs.gsu.edu> */

#ifndef ION_ANALYSIS_VITERBI_H
#define ION_ANALYSIS_VITERBI_H

using namespace std;

struct markov_state {
  double value, prev, current;
  boost::math::poisson dist;

  markov_state() : value(0), prev(0), current(0), dist(0) { }
  markov_state(double value) : value(value), prev(0), current(0), dist(value) { }

  double cost(double v);

  double cost(const markov_state& st) { return st.value == value ? 0 : -5E2; }

  void step_forward() { prev = current; }
};

struct markov_chain_comparator {
  markov_state * current;
  markov_chain_comparator() : current(NULL) { }
  markov_chain_comparator(markov_state& current) : current(&current) { }
  bool operator()(const markov_state& lhs, const markov_state& rhs);
};

template<typename T>
struct depth_info {
  T dp, min_dp, max_dp;
  depth_info() : dp(0), min_dp(numeric_limits<T>::max()), max_dp(numeric_limits<T>::min()) { }
  depth_info(T dp, T min_dp, T max_dp) : dp(dp), min_dp(min_dp), max_dp(max_dp) { }
  depth_info(T dp) : dp(0), min_dp(dp), max_dp(dp) { }
};

template<typename T>
class markov_chain : vector< vector<long> > {
  vector<markov_state> states;
  vector<pair<depth_info<T>, long> > items;
  vector<T> values;
  markov_chain() { }
public:
  template<typename _ForwardIterator>
  markov_chain(_ForwardIterator begin, _ForwardIterator end);
  template<typename _ForwardIterator>
  markov_chain(_ForwardIterator begin, _ForwardIterator end, T min, T max);
  ~markov_chain() { };

  markov_state state(long i) { return states[i]; }

  typename vector<pair<depth_info<T>, long> >::reverse_iterator ibegin() { return items.rbegin(); }
  typename vector<pair<depth_info<T>, long> >::reverse_iterator iend()   { return items.rend(); }
private:
  void generate_states(T min, T max);
  void process_next_value(T val);
  void optimal_path();
  template<typename _ForwardIterator>
  void initialize(_ForwardIterator begin, _ForwardIterator end, T min, T max);
};

#include "viterbi.hpp"

#endif //ION_ANALYSIS_VITERBI_H
