/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ACQ_H
#define ACQ_H

#include <stdint.h>
#include "Image.h"

class Acq {
	public:
		Acq();
		virtual ~Acq();

		void	SetSize(int w, int h, int numFrames, int uncompFrames);
		void	SetData(Image *_image);
		void	SetWellTrace(double *trace, int x, int y);
		void	Clear();
		void	Write();
		bool	Write(const char *name, int ox, int oy, int ow, int oh);
		bool    WriteVFC(const char *name, int ox, int oy, int ow, int oh);
		bool	WriteAscii(const char *acqName, int ox, int oy, int ow, int oh);
		int	PinnedLow() {return pinnedLow;}
		int	PinnedHigh() {return pinnedHigh;}
		uint16_t get_val(uint32_t x, uint32_t y, uint32_t rframe, uint32_t numFrames);
		int     PrevFrameSubtract(int elems, int16_t *framePtr, int16_t *prevFramePtr, int16_t *results, uint64_t *out_len, uint32_t ow, uint32_t oh);

	protected:

		Image *image;
		int	numFrames;
		int uncompFrames;
		int	w, h;
		unsigned short *data;
		static int counter;
		int	frameStride;
		int	pinnedLow;
		int	pinnedHigh;
		int *timestamps;
		int x_region_size;
		int y_region_size;
};

#endif // ACQ_H

