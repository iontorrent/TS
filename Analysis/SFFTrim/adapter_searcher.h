/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ADAPTER_SEARCHER_H
#define ADAPTER_SEARCHER_H

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "flow_utils.h"

class adapter_searcher {
public:
	// Build an adpater_searcher by supplying flow order, key sequence, and reverse compelement of
	// adapter sequence:
	adapter_searcher(const std::string& flow_order, const std::string& key, const std::string& adapter);
	
	// The following methods search an ionogram for matches to the adapter.
	// The return value is the number of candidate matches found.

	// Search for adapter in ionogram stored as a sequence of floating point values in the range [first, last):
	template <class Ran>
	int find_matches(Ran first, Ran last, float cutoff);

	// Search for adapter in ionogram stored as a squence of integers, as in an sff file:
	template <class Ran>
	int find_matches_sff(Ran flowgram, int nflows, float cutoff);

	struct match {
		match() : _flow(0), _score(0.0), _len(0) {} 
		match(int flow, float score, int len) : _flow(flow), _score(score), _len(len) {}
		bool operator<(const match& rhs) const {return _len < rhs._len;}

		int   _flow;
		float _score;
		int   _len;
	};

	match pick_longest() const;
	match pick_closest() const;

	// Convert flow number to base number using flow_index field from sff:
	template <class Ran1, class Ran2>
	long flow2pos(Ran1 flow_index, Ran2 base, int nbases, int flow) const;

private:
	typedef float                    float_t;
	typedef std::vector<float_t>     ionogram;
	typedef ionogram::const_iterator ion_it;
	typedef std::deque<ionogram>     ion_deq;
	typedef std::deque<int>          off_deq;
	typedef std::deque<match>        match_deq;

	const std::string& _flow_order;
	const std::string& _adapter;
	ion_deq            _adapter_flow;
	off_deq            _flow_off;
	match_deq          _matches;
	int                _n_key_flows;

	template <class Ran>
	Ran search_start(Ran first, int flow_off) const;

	template <class Ran1, class Ran2>
	float distance(Ran1 abeg, Ran2 rbeg, int len) const;
};

adapter_searcher::adapter_searcher(const std::string& flow_order, const std::string& key, const std::string& adapter)
	: _flow_order(flow_order)
	, _adapter(adapter)
	, _n_key_flows(0)
{
	// Look through the flow order, finding each possible start
	// position for the adapter:
	for(unsigned int pos=0; pos<_flow_order.length(); ++pos){
		if(_flow_order[pos] == _adapter[0]){
			_flow_off.push_back(pos);
			// What would the adapter ionogram look like, if it started at this position?
			_adapter_flow.push_back(ionogram());
			ionogram& ion     = _adapter_flow.back();
			std::string order = _flow_order.substr(pos) + _flow_order.substr(0, pos);
			seq2flow(_adapter, order, back_inserter(ion));
		}
	}

	// How many flows consumed in the key?
	ionogram key_flow;
	seq2flow(key, _flow_order, back_inserter(key_flow));
	_n_key_flows = key_flow.size();

	// flow2pos() assumes _adapter begins with a 1-mer:
	assert(_adapter[0] != _adapter[1]);
}

template <class Ran>
int adapter_searcher::find_matches_sff(Ran flowgram, int nflows, float cutoff)
{
	// Copy flows, and divide by 100:
	ionogram read_flow(nflows);
	std::transform(flowgram, flowgram+nflows, read_flow.begin(), std::binder2nd<std::divides<float_t> >(std::divides<float_t>(),100));

	return find_matches(read_flow.begin(), read_flow.end(), cutoff);
}

template <class Ran>
Ran adapter_searcher::search_start(Ran first, int flow_off) const
{
	// First place in read where a match could start:
	int remainder = _n_key_flows % _flow_order.length();
	first += _n_key_flows - remainder + flow_off;
	if(flow_off < remainder)
		first += _flow_order.length();
	
	return first;
}

template <class Ran1, class Ran2>
float adapter_searcher::distance(Ran1 abeg, Ran2 rbeg, int len) const
{
	// Ignore first flow:
	++abeg;
	++rbeg;
	--len;
	float diff[len];
	std::transform(abeg, abeg+len, rbeg, diff, std::minus<float>());
	return std::inner_product(diff, diff+len, diff, 0.0);
}

template <class Ran>
int adapter_searcher::find_matches(Ran first, Ran last, float cutoff)
{
	// Find potential matches to adapter:
	_matches.clear();
	int n_starts = _flow_off.size();
	int cycl_len = _flow_order.length(); 
	for(int i=0; i<n_starts; ++i){
		ion_it    adap_beg = _adapter_flow[i].begin(); 
		ptrdiff_t adap_len = _adapter_flow[i].end() - adap_beg;
		Ran       read_beg = search_start(first, _flow_off[i]); 
		ptrdiff_t read_len = last - read_beg;
		for(int offset=0; offset<read_len; offset+=cycl_len){
			int   match_len = std::min(read_len-offset, adap_len);
			float score     = distance(adap_beg, read_beg+offset, match_len) * adap_len / match_len;
			if(score < cutoff and *max_element(adap_beg+1,adap_beg+match_len))
				_matches.push_back(match(read_beg-first+offset, score, match_len));
		}
	}

	return _matches.size();
}

adapter_searcher::match adapter_searcher::pick_longest() const
{
	// Pick longest match among the candidates:
	assert(not _matches.empty());
	return *std::max_element(_matches.begin(), _matches.end());
}

adapter_searcher::match adapter_searcher::pick_closest() const
{
	// Pick closest match among the candidates:
	assert(not _matches.empty());
	match_deq::const_iterator closest = _matches.begin();
	for(match_deq::const_iterator m=_matches.begin(); m!=_matches.end(); ++m){
		if(m->_score < closest->_score)
			closest = m;
	}
	return *closest;
}

template <class Ran1, class Ran2>
long adapter_searcher::flow2pos(Ran1 flow_index, Ran2 base, int nbases, int flow) const
{
	// see assert() at end of ctor.
	long pos = flow2base(flow, flow_index, nbases);
	while(pos < nbases and base[pos+1] == _adapter[0])
		++pos;
	
	return pos;
}

#endif // ADAPTER_SEARCHER_H

