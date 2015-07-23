/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ACQ_H
#define ACQ_H

#include <stdint.h>
#include <vector>
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
        bool    WriteVFC(const char *name, int ox, int oy, int ow, int oh, bool verbose=false);
        bool    WriteThumbnailVFC(const char *acqName, int cropx, int cropy, int kernx, int kerny, int region_len_x, int region_len_y, int marginx, int marginy, int ow, int oh, bool verbose);
        bool	WriteAscii(const char *acqName, int ox, int oy, int ow, int oh);
		int	PinnedLow() {return pinnedLow;}
		int	PinnedHigh() {return pinnedHigh;}
		uint16_t get_val(uint32_t x, uint32_t y, uint32_t rframe, uint32_t numFrames);
//		int     PrevFrameSubtract(int elems, int16_t *framePtr, int16_t *prevFramePtr, int16_t *results, uint64_t *out_len, uint32_t ow, uint32_t oh);

                void CalculateNumOfRegions(uint32_t ow, uint32_t oh);
                void PopulateRegionalAcquisitionWindow(
                    const char* t0InfoFile, 
                    const char* regionAcqT0File,
                    int ow, 
                    int oh);
                bool CheckForRegionAcqTimeWindow(uint32_t baseInterval, uint32_t timeStamp, int regionNum, int* framesToAvg = NULL);
                int DeltaCompressionOnRegionalAcquisitionWindow(
                    int elems, 
		    uint32_t baseFrameRate,
		    uint32_t timeStamp,
		    unsigned int frameNum,
		    int16_t* firstFramePtr,
		    int16_t *framePtr, 
		    int16_t* prevFramePtr, 
		    int16_t *results, 
		    uint64_t *out_len, 
		    uint32_t ow, 
		    uint32_t oh); 
                bool WriteRegionBasedAcq(char *acqName, int ox, int oy, int ow, int oh);
                bool WriteFrameAveragedRegionBasedAcq(char *acqName, int ox, int oy, int ow, int oh);
                unsigned int GetCompressedFrameNum(unsigned int unCompressedFrameNum);
                unsigned int GetUncompressedFrames(unsigned int compressedFrames);
                bool WriteTimeBasedAcq(char *acqName, int ox, int oy, int ow, int oh);
                bool WritePFV(char *acqName, int ox, int oy, int ow, int oh, char *options);
                void PopulateCroppedRegionalAcquisitionWindow(
                    const char* t0InfoFile, 
		    const char* regionAcqT0File,
		    int ox,
		    int oy,
		    int ow, 
		    int oh,
                    unsigned int baseframerate);
		void ParseT0File(
                    const char* t0InfoFile, 
		    const char* regionAcqT0File,
		    int ox,
		    int oy,
		    int ow, 
		    int oh,
                    unsigned int baseframerate);
                void GenerateExcludeMaskRegions(const char* excludeMaskFile);
                int FramesBeforeT0Averaged(
		    int elems, 
		    uint32_t baseFrameRate,
		    uint32_t timeStamp,
		    unsigned int frameNum,
		    int* framesToAvg,
		    int32_t* avgFramePtr,
		    int16_t* firstFramePtr,
		    int16_t *framePtr, 
		    int16_t* prevFramePtr, 
		    int16_t *results, 
		    uint64_t *out_len, 
		    uint32_t ow, 
		    uint32_t oh); 

            // to compress image
            void doT0Compression();
            void getAverageTrace(double *avg_trace, FILE *fp);

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
                uint32_t num_regions_x;
                uint32_t num_regions_y;
                std::vector<float> region_acq_start;
                std::vector<float> region_acq_end;
                std::vector<int> excludedRegions;

        static int t0est_start;
        static int t0est_end;
        static int slower_frames;
};
extern int PrevFrameSubtract(uint32_t w, uint32_t h, int16_t *framePtr, int16_t *prevFramePtr,
		int16_t *results, uint64_t *out_len, uint32_t *comprType);

#endif // ACQ_H

