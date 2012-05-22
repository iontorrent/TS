/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#include "Mask.h"
#include "datahdr.h"
#include "ByteSwapUtils.h"
#include "Filter.h"
#include "Region.h"
#include "ChipIdDecoder.h"
#include "ChannelXTCorrection.h"

#include <vector>
#include <string>

// #include "sgfilter/SGFilter.h"

class SGFilter;
class PinnedInFlow;

enum InterlaceType
{
  FlatImage = 0,
  Interlaced314 = 1,
  Interlaced314Q = 2,
  Interlaced316C = 3,
  InterlacedComp = 4
};

typedef struct
{
  ChipIdEnum id;
  ChannelXTCorrectionDescriptor descr;
} ChipXtVectArrayType;

struct RawImage
{
  int rows, cols, frames;   // vars for image size
  int chip_offset_x, chip_offset_y;   // offset of image origin relative to chip origin
  int uncompFrames,compFrames;
  int channels;    // number of channels in this image
  int interlaceType;   // 0 is not interlaced, 1 and up are various interlace schemes
  int frameStride;   // rows * cols
  short *image;    // raw image (loaded in as unsigned short, and byte-swapped, masked with 0x3fff, and promoted to short)
  int *timestamps;
  short *compImage; 
  int *compTimestamp;
  int baseFrameRate;
  int *interpolatedFrames;
  float *interpolatedMult;
  RawImage()
  {
    rows = 0;
    cols = 0;
    frames = 0;
    chip_offset_x = 0;
    chip_offset_y = 0;
    channels = 0;
    interlaceType = 0;
    frameStride = 0;
    image = NULL;
    timestamps = NULL;
    compImage=NULL;
    compTimestamp=NULL;
    interpolatedFrames=NULL;
    interpolatedMult=NULL;
    uncompFrames=0;
    compFrames=0;
    baseFrameRate=0;
  }
};

class Image;

class Image
{
 public:
  Image();
  virtual ~Image();
  bool LoadRaw(const char *rawFileName, int frames = 0, bool allocate = true, bool headerOnly = false);
//  bool LoadRaw (const char *rawFileName);
  int GetRows() { return raw->rows; }
  int GetCols() { return raw->cols; }
  int GetFrames() { return raw->frames; }
  short At(int row, int col, int frame) { return raw->image[raw->cols * row + col + frame*raw->frameStride]; }
  short At(int well, int frame) { return raw->image[well + frame*raw->frameStride]; }
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
  bool LoadSlice(
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
  void SetDir(const char* directory);
  int  FilterForPinned(Mask *mask, MaskType these, int markBead = 0);
  void ReportPinnedWells(const std::string& strDatFileName);
  void Normalize(int startPos, int endPos);
  // void SubtractImage(Image *sub);
  void BackgroundCorrect(Mask *mask, MaskType these, MaskType usingThese, int inner, int outer, Filter *filter = NULL, bool bkg = false, bool onlyBkg = false, bool replaceWBkg = false);
  void BackgroundCorrect(Mask *mask, MaskType these, MaskType usingThese, int innerx, int innery, int outerx, int outery, Filter *filter = NULL, bool bkg = false, bool onlyBkg = false, bool replaceWBkg = false);
  void BackgroundCorrectRegion(Mask *mask, Region &reg, MaskType these, MaskType usingThese, int innerx, int innery, 
                               int outerx, int outery, Filter *f = NULL, bool saveBkg = false, bool onlyBkg = false, bool replaceWBkg = false);

  unsigned int GetUnCompFrames() { return raw->uncompFrames; }

  void SGFilterSet(int spread, int coeff);
  void SGFilterApply(Mask *mask, MaskType these);
  void SGFilterApply(short *source, short *target);
  void SGFilterApply(float *source, float *target);

  void CalcBeadfindMetric_1(Mask *mask, Region r, char *tr, int frameStart=-1, int frameEnd=-1);
  void FindPeak(Mask *mask, MaskType these);
  void SetMaxFrames(int max) { maxFrames = max; }
  int  GetMaxFrames() { return raw->frames; }
  float GetInterpolatedValue(int frame, int x, int y);
  void GetUncompressedTrace (float *val, int last_frame, int x, int y);
  void GetInterpolatedValueAvg4(int16_t *ptr, int frame, int x, int y, int numInt);

  void SetTimeout(int _total_timeout,int _retry_interval)
  {
    retry_interval = _retry_interval;
    total_timeout = _total_timeout;
  }
  void SetImgLoadImmediate(bool flag) { recklessAbandon = flag; }
  void IntegrateRaw(Mask *mask, MaskType these, int start, int end);
  void IntegrateRawBaseline(Mask *mask, MaskType these, int start, int end, int baselineStart, int baselineEnd, double *minval=NULL, double *maxval=NULL);
  void DumpHdr();
  void DebugDumpResults(char *fileName, Region region);
  void DebugDumpResults(char *fileName);
  void DumpTrace(int r, int c, char *fileName);
  double DumpStep(int c, int r, int w, int h, std::string regionName, char nucChar, std::string nucStepDir, Mask *mask, PinnedInFlow *pinnedInFlow, int flowNumber);
  int DumpDcOffset(int nSample, std::string dcOffsetDir, char nucChar, int flowNumber);
  void Cleanup();

  const double *GetResults() { return results; }
  const short *GetBkg() { return bkg; }
  const RawImage *GetImage() { return raw; }
  void SetImage(RawImage *img);

  int  GetFrame(int time);

  void    SetFlowOffset(int _flowOffset) { flowOffset = _flowOffset; }
  int     GetFlowOffset(void) { return(flowOffset); }
  void    SetNoFlowTime(int _noFlowTime) { noFlowTime = _noFlowTime; }
  int     GetNoFlowTime(void) { return(noFlowTime); }
  bool    BeadfindType();

  bool ReadyToLoad(const char *);
  void SetNumAcqFiles(int num) { numAcqFiles = num; }
  bool VFREnabled()
  {
    if (raw->uncompFrames == raw->frames) return false;
    else return true;
  }
  void SetIgnoreChecksumErrors(int flag) { ignoreChecksumErrors = flag; }
  static Region chipSubRegion;

  // void    XTChannelCorrect(Mask *mask);
  void    XTChannelCorrect();
  static  void SetCroppedRegionOrigin(int x,int y)
  {
    cropped_region_offset_x = x;
    cropped_region_offset_y = y;
  }
  void SetOffsetFromChipOrigin(const char *);
  static  void    CalibrateChannelXTCorrection(const char *exp_dir,const char *filename, bool wait_for_prerun=true);


protected:
  RawImage *raw;    // the raw image
  double  *results;   // working memory filled with results from various methods
  int16_t  *bkg;    // average background value per pixel
  int   maxFrames;   // fax frames this object will attempt to load
  SGFilter *sgFilter;   // the SG-Filter class
  int  sgSpread;   // # of points on either side of primary point used to smooth
  int  sgCoeff;   // # polynomial coefficients
  char   *experimentName;
  int   flowOffset;  // time in milliseconds between start of nuc flow and start of image acquisition
  int   noFlowTime;  // time in milliseconds for which it can be assumed nuc flow has not yet hit
  int         retry_interval;
  int         total_timeout;
  void    cleanupRaw();
  int  numAcqFiles;
  bool recklessAbandon;
  int     ignoreChecksumErrors;   // can be set to force loading of possibly corrupt images
  int dump_XTvects_to_file;


  static ChipXtVectArrayType default_chip_xt_vect_array[];
  static int chan_xt_column_offset[];
  static ChannelXTCorrectionDescriptor selected_chip_xt_vectors;
  static ChannelXTCorrection *custom_correction_data;

  // allow the user to specify the origin of a cropped data set so that cross-talk correction
  // works properly
  static int cropped_region_offset_x;
  static int cropped_region_offset_y;


};

// prototype for deInterlace.cpp

int deInterlace_c(
  char *fname, short **_out, int **_timestamps,
  int *_rows, int *_cols, int *_frames, int *_compFrames, int start_frame, int end_frame,
  int mincols, int minrows, int maxcols, int maxrows, int ignoreErrors);


#endif // IMAGE_H
