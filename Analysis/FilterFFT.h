/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILTERFFT_H
#define FILTERFFT_H

#include "Filter.h"

class FilterFFT : public Filter {
	public:
		FilterFFT(int w, int h) : Filter(w, h) {}
		virtual ~FilterFFT() {};
	private:
		FilterFFT();
};

#endif // FILTERFFT_H

