/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FILTER_H
#define FILTER_H

class Filter {
	public:
		Filter(int _w, int _h) {w = _w; h = _h;}
		virtual ~Filter() {};

	protected:
		int	w;
		int	h;

	private:
		Filter();
};

#endif // FILTER_H

