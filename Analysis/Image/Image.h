/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#include "Mask.h"
#include "datahdr.h"
#include "ByteSwapUtils.h"
#include "Region.h"
#include "ChipIdDecoder.h"
#include "ChannelXTCorrection.h"
#include "TikhonovSmoother.h"
#include "RawImage.h"
#include "ImageTransformer.h"

#include <vector>
#include <string>

class PinnedInFlow;

enum InterlaceType {
    FlatImage = 0,
    Interlaced314 = 1,
    Interlaced314Q = 2,
    Interlaced316C = 3,
    InterlacedComp = 4
};

class Image;
class ImageCropping; // forward declaration of this global state
class SynchDat;

class AcqMovie {
 public:
  virtual ~AcqMovie() {}
  virtual int GetRows() const = 0;
  virtual int GetCols() const  = 0;
  virtual int GetFrames() const = 0;
  virtual short At ( int well, int frame ) = 0;
};


/* shared mem sem.h */
typedef struct Image_semaphore {
    pthread_mutex_t lock;
    pthread_cond_t nonzero;
    unsigned count;
    pthread_t owner;
}Image_semaphore_t;

class Image : public AcqMovie
{
public:
    Image();
    virtual ~Image();
    bool LoadRaw ( const char *rawFileName, int frames = 0, bool allocate = true, bool headerOnly = false );
    bool LoadRaw ( const char *rawFileName, TikhonovSmoother *tikSmoother );
    bool LoadRaw ( const char *rawFileName, int frames, bool allocate, bool headerOnly,
                   TikhonovSmoother *tikSmoother );
    int ActuallyLoadRaw ( const char *rawFileName, int frames,  bool headerOnly );
    /** Not perfect as sdat has a particular time compression per region but good workaround for some use cases. */
    void InitFromSdat(SynchDat *sdat);
    int GetRows() const {
        return raw->rows;
    }
    int GetCols() const {
        return raw->cols;
    }
    int GetFrames() const {
        return raw->frames;
    }

    short &At ( int row, int col, int frame ) {
        return raw->image[raw->cols * row + col + frame*raw->frameStride];
    }

    const short &At ( int row, int col, int frame ) const {
        return raw->image[raw->cols * row + col + frame*raw->frameStride];
    }

    const short &At ( int well, int frame ) const {
        return raw->image[well + frame*raw->frameStride];
    }

    short At ( int well, int frame ) {
        return raw->image[well + frame*raw->frameStride];
    }

    //
    // LoadSlice returns subsets of a dat, allowing for:
    //   seleciton of arbitrary collections of wells (via col & row args)
    //   seleciton of rectangular region of wells (via minCol, minRow, maxCol, maxRow args)
    //   seleciton of subset of frames in a timespan (via loadMinTime & loadMaxTime)
    //   uncompression of VFC data (via uncompress)
    //   baselining (via baselineMinTime & baselineMaxTime)
    //   control over what is returned (via returnSignal, returnMean, returnSD)
    // Returns true upon successful completion.
    //
    bool LoadSlice (
        // Inputs
        std::vector<std::string> rawFileName,  // datFiles to read
        std::vector<unsigned int> col,         // 0-based column indexes to load
        std::vector<unsigned int> row,         // 0-based row indexes to load
        int minCol,                            // 0-based min col to load
        int maxCol,                            // 1-based max col to load
        int minRow,                            // 0-based min row to load
        int maxRow,                            // 1-based max row to load
        bool returnSignal,                     // If true, signal object is populated upon return
        bool returnMean,                       // If true, per-well mean is populated upon return
        bool returnSD,                         // If true, per-well sd is populated upon return
        bool returnLag,                         // If true, per-well sd of lag-one data is populated upon return
        bool uncompress,                       // If true, VFC-compressed data will be interpolated before return
        bool doNormalize,                      // If true, normalize data
        int normStart,                         // First frame to use for normalization
        int normEnd,                           // Last frame to use for normalization
        bool XTCorrect,                        // If true, do electrical xtalk correct with XTChannelCorrect
        std::string chipType,                  // Optionally set chip type, or empty string to try auto-detect
        double baselineMinTime,                // Min time (in seconds) for range of flows to use for baseline subtraction
        double baselineMaxTime,                // Max time for baseline.  Set to less than baselineMinTime to disable.
        double loadMinTime,                    // Min time (in seconds) for range of flows to return
        double loadMaxTime,                    // Max time for return flows.  Set to less than loadMinTime for all flows.
        // Outputs
        unsigned int &nCol,                    // Number of cols in full dat
        unsigned int &nRow,                    // Number of rows in full dat
        std::vector<unsigned int> &colOut,     // Col indices of returned data
        std::vector<unsigned int> &rowOut,     // Row indices of returned data
        unsigned int &nFrame,                  // Number of frames in full dat
        std::vector< std::vector<double> > &frameStart,           // frameStart[dat][frame] is time in seconds for start of frame
        std::vector< std::vector<double> > &frameEnd,             // frameStart[dat][frame] is time in seconds for end of frame
        std::vector< std::vector< std::vector<short> > > &signal, // signal[dat][well][frame] is returned signal
        std::vector< std::vector<short> > &mean,                  // mean[dat][well] is per-well mean signal
        std::vector< std::vector<short> > &sd,                     // sd[dat][well] is per-well sd signal
        std::vector< std::vector<short> > &lag                     // sd[dat][well] is per-well sd signal
    );

    void Close(); //Frees image memory only
    void SetDir ( const char* directory );
    int  FilterForPinned ( Mask *mask, MaskType these, int markBead = 0 );

    void SetMeanOfFramesToZero ( int startPos, int endPos );
    // void SubtractImage(Image *sub);
    void SubtractLocalReferenceTrace ( Mask *mask, MaskType apply_to_these, MaskType derive_from_these, int inner, int outer, bool bkg = false, bool onlyBkg = false, bool replaceWBkg = false );
    void SubtractLocalReferenceTrace ( Mask *mask, MaskType apply_to_these, MaskType derive_from_these, int innerx, int innery, int outerx, int outery, bool bkg = false, bool onlyBkg = false, bool replaceWBkg = false );
    void SubtractLocalReferenceTraceInRegion (Region &reg,  Mask *mask, MaskType apply_to_these, MaskType derive_from_these, int innerx, int innery,
            int outerx, int outery, bool saveBkg = false, bool onlyBkg = false, bool replaceWBkg = false );
    void  GenerateCumulativeSumMatrix(int64_t *workTotal, unsigned int *workNum, uint16_t *MaskPtr, MaskType derive_from_these, int frame);
    int WholeFrameMean(int64_t *workTotal, unsigned int *workNum);
    void ApplyLocalReferenceToWholeChip(int64_t *workTotal, unsigned int *workNum, 
                                        uint16_t *MaskPtr, MaskType apply_to_these,
                                        int innerx, int innery, int outerx, int outery,
                                        bool saveBkg, bool onlyBkg, bool replaceWBkg, int frame);
    void SetUpBkgSave(bool saveBkg);

    void  GenerateCumulativeSumMatrixInRegion(Region &reg, int64_t *workTotal, unsigned int *workNum, uint16_t *MaskPtr, MaskType derive_from_these, int frame);
    void ApplyLocalReferenceInRegion(Region &reg, int64_t *workTotal, unsigned int *workNum,
                                     uint16_t *MaskPtr, MaskType apply_to_these,
                                     int innerx, int innery, int outerx, int outery,
                                     bool saveBkg, bool onlyBkg, bool replaceWBkg, int frame);
    int FindMeanValueInRegion ( Region &reg, int64_t *workTotal, unsigned int *workNum );
    unsigned int GetUnCompFrames() {
        return raw->uncompFrames;
    }

    void CalcBeadfindMetric_1 ( Mask *mask, Region r, char *tr, int frameStart=-1, int frameEnd=-1 );
    void CalcBeadfindMetricIntegral (Mask *mask, Region region, char *idStr, int frameStart, int frameEnd);
    void CalcBeadfindMetricRegionMean ( Mask *mask, Region region, char *idStr, int frameStart, int frameEnd);
    void FindPeak ( Mask *mask, MaskType these );
    void SetMaxFrames ( int max ) {
        maxFrames = max;
    }
    int  GetMaxFrames() {
        return raw->frames;
    }
    float GetInterpolatedValue ( int frame, int x, int y );
    void GetUncompressedTrace ( float *val, int last_frame, int x, int y );

    void SetTimeout ( int _total_timeout,int _retry_interval ) {
        retry_interval = _retry_interval;
        total_timeout = _total_timeout;
    }
    int GetRetryInterval() {return retry_interval;}
    int GetTotalTimeout() {return total_timeout;}
    bool WaitForMyFileToWakeMeFromSleep ( const char *rawFileName );
    void SetImgLoadImmediate ( bool flag ) {
        recklessAbandon = flag;
    }
    void IntegrateRaw ( Mask *mask, MaskType these, int start, int end );
    void IntegrateRawBaseline ( Mask *mask, MaskType these, int start, int end, int baselineStart, int baselineEnd, double *minval=NULL, double *maxval=NULL );
    void DumpHdr();
    void DebugDumpResults ( char *fileName, Region region );
    void DebugDumpResults ( char *fileName );
    void DumpTrace ( int r, int c, char *fileName );
    double DumpStep ( int c, int r, int w, int h, std::string regionName, char nucChar, std::string nucStepDir, Mask *mask, PinnedInFlow *pinnedInFlow, int flowNumber );
    int DumpDcOffset ( int nSample, std::string dcOffsetDir, char nucChar, int flowNumber );
    void Cleanup();
    void JustCacheOneImage(const char *name);
    static void ImageSemGive();
    static void ImageSemTake();

    const double *GetResults() {
        return results;
    }
    const short *GetBkg() {
        return bkg;
    }
    const RawImage *GetImage() {
        return raw;
    }
    void SetImage ( RawImage *img );

    int  GetFrame ( int time );

    void    SetFlowOffset ( int _flowOffset ) {
        flowOffset = _flowOffset;
    }
    int     GetFlowOffset ( void ) {
        return ( flowOffset );
    }
    void    SetNoFlowTime ( int _noFlowTime ) {
        noFlowTime = _noFlowTime;
    }
    int     GetNoFlowTime ( void ) {
        return ( noFlowTime );
    }
    bool    BeadfindType();

    static bool ReadyToLoad ( const char * );
    void SetNumAcqFiles ( int num ) {
        numAcqFiles = num;
    }
    bool VFREnabled() {
        if ( raw->uncompFrames == raw->frames ) return false;
        else return true;
    }
    void SetIgnoreChecksumErrors ( int flag ) {
        ignoreChecksumErrors = flag;
    }
    void SmoothMeTikhonov ( TikhonovSmoother *tikSmoother, bool dont_smooth_me_bro, const char *rawFileName );

    void SetOffsetFromChipOrigin ( const char * );


    bool doLocalRescaleRegionByEmptyWells()
    {
        return (smooth_max_amplitude != NULL);
    }
    short CalculateCharacteristicValue(short *prow, int ax);
    void FindCharacteristicValuesInReferenceWells(short *tmp, short *pattern_flag,
                                                  MaskType reference_type, Mask *mask_ptr, PinnedInFlow &pinnedInFlow,
                                                  int flow);
    void SlowSmoothPattern(short *tmp, short *pattern_flag, int smooth_span);
    void FastSmoothPattern(short *tmp, short *pattern_flag, int smooth_span);
    
    void GenerateCumulativeSumMatrixFromPattern ( int64_t *workTotal, unsigned int *workNum, short *my_values, short *my_pattern );
     void SmoothMaxByPattern ( int64_t *workTotal, unsigned int *workNum,int smooth_span);
    void CalculateEmptyWellLocalScaleForFlow ( PinnedInFlow& pinnedInFlow,Mask *bfmask,int flow,MaskType referenceMask, int smooth_span );
    float GetEmptyWellAmplitudeRegionAverage ( Region *region );
    void LocalRescaleRegionByEmptyWells ( Region *region );
    float getEmptyWellAmplitude ( int row,int col ) {
        return ( smooth_max_amplitude[row*raw->cols + col] );
    }
    bool isEmptyWellAmplitudeAvailable ( void ) {
        return ( smooth_max_amplitude != NULL );
    }

    void TimeStampCalculation();
    void TimeStampReporting ( int rc );

    RawImage *raw;    // the raw image
    char   *results_folder;  // really the directory
    static Image_semaphore_t *Image_SemPtr;

protected:
    double  *results;   // working memory filled with results from various methods
    int16_t  *bkg;    // average background value per pixel
    int   maxFrames;   // fax frames this object will attempt to load
    int   flowOffset;  // time in milliseconds between start of nuc flow and start of image acquisition
    int   noFlowTime;  // time in milliseconds for which it can be assumed nuc flow has not yet hit
    int         retry_interval;
    int         total_timeout;
    void    cleanupRaw();
    int  numAcqFiles;
    bool recklessAbandon;
    int     ignoreChecksumErrors;   // can be set to force loading of possibly corrupt images

    // nn-smoothed maximum amplitude of empty wells
    float *smooth_max_amplitude;

};

// prototype for deInterlace.cpp

int deInterlace_c (
    char *fname, short **_out, int **_timestamps,
    int *_rows, int *_cols, int *_frames, int *_compFrames, int start_frame, int end_frame,
    int mincols, int minrows, int maxcols, int maxrows, int ignoreErrors );


#endif // IMAGE_H
