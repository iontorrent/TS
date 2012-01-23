/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef QSCORETRIM_H
#define QSCORETRIM_H

#include <algorithm>
#include <numeric>

template <class Ran>
Ran QualTrim(Ran qbeg, Ran qend, float cutoff, int window_size)
{
	// Given a seqeunce of Q scores in [qbeg,qend), find the point
	// at which the sliding window mean of the scores drops below
	// cutoff.

	// Initialize the sliding window:
	Ran   wbeg = qbeg;
	Ran   wend = std::min(wbeg+window_size, qend);
	float wsum = std::accumulate(wbeg, wend, 0.0);

	// Find the first window with mean Q below cutoff:
	float minsum = window_size * cutoff;
	while(wend < qend and wsum >= minsum){
		wsum += *wend++;
		wsum -= *wbeg++;
	}

	// Return pointer to the center of this window:
	return wbeg + (wend - wbeg)/2;
}

#endif // QSCORETRIM_H

