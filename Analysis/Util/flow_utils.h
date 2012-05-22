/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOW_UTILS_H
#define FLOW_UTILS_H

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include "flowgram_it.h"

template <class Ran>
long flow2base(int flow, Ran flow_index, int nbases)
{
	++flow;
	int base = 0;
	for(; base<nbases and flow>0; ++base)
		flow -= flow_index[base];
	return base;
}

template <class FlowIt>
void seq2flow(const std::string& seq, const std::string& flow_order, FlowIt dest)
{
	// Convert sequence to flowspace:
	for(flowgram_it i(flow_order, seq); i.good(); i.next(), ++dest)
		*dest = i.hplen();
}

// Return flow position of a particular base position doing it the naive way
int getFlowNum(std::string& seq, std::string& flow_order, int seq_position);

#endif // FLOW_UTILS_H

