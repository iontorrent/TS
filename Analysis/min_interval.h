/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef MIN_INTERVAL_H
#define MIN_INTERVAL_H

#include <iterator>
#include <utility>

// Given a sequence of numbers, find the shortest sub-interval
// that sums to a specified minimum.

template <class Ran, class T>
std::pair<Ran,Ran> min_interval(Ran first, Ran last, T min_sum)
{
	// Find out the type of the numbers in the sequence:
	typedef typename std::iterator_traits<Ran>::value_type      num_t;
	typedef typename std::iterator_traits<Ran>::difference_type diff_t;

	// Record shortest interval found so far:
	Ran    best_beg = first;
	Ran    best_end = last;
	diff_t best_len = best_end - best_beg;

	// For each interval start position, find the first possible
	// interval end position.
	Ran   beg = first;
	Ran   end = first;
	num_t sum = 0;
	num_t tmp = 0;

	for(; end<last; sum-=*beg++){
		// Find the first possible end position:
		while(end<last and sum<min_sum)
			sum += *end++;
		
		// Done if none is found:
		if(sum < min_sum)
			break;

		// Find last possible start position:
		while(beg<end and (tmp=sum-*beg)>=min_sum){
			sum = tmp;
			++beg;
		}

		// Length of interval found:
		diff_t len = end - beg;

		// Is this the best length?
		if(len < best_len){
			best_beg = beg;
			best_end = end;
			best_len = len;
		}
	}

	// Return the shortest interval:
	return std::make_pair(best_beg, best_end);
}

#endif // MIN_INTERVAL_H
