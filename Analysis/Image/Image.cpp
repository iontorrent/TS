/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>   // for sysconf ()
#include <memory>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h> //  for debug time interval
#include <fcntl.h>
#include <libgen.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <limits>
#include "MathOptim.h"

#include "IonErr.h"
#include "Image.h"
#include "SampleStats.h"

#include "sgfilter/SGFilter.h"
#include "Histogram.h"
#include "Utils.h"
#include "deInterlace.h"
#include "LinuxCompat.h"
#include "ChipIdDecoder.h"
#include "LSRowImageProcessor.h"

#include "dbgmem.h"
#include "IonErr.h"
#include "PinnedWellReporter.h"
#include "PinnedInFlow.h"

using namespace std;


#define RETRY_INTERVAL 15 // 15 seconds wait time.
#define TOTAL_TIMEOUT 3600 // 1 hr before giving up.

//Initialize chipSubRegion in Image class
Region Image::chipSubRegion = {0,0,0,0,0};

// this is perhaps a prime candidate for something that is a json set of parameters to read in
// inversion vector + coordinates of the offsets
// we do only a single pass because the second-order correction is too small to notice
#define DEFAULT_VECT_LEN  7
static float chan_xt_vect_316[DEFAULT_VECT_LEN]      = {0.0029,-0.0632,-0.0511,1.1114,0.0000,0.0000,0.0000};
static float *default_316_xt_vectors[] = {chan_xt_vect_316};

static float chan_xt_vect_318_even[DEFAULT_VECT_LEN] = {0.0132,-0.1511,-0.0131,1.1076,0.0404,0.0013,0.0018};
static float chan_xt_vect_318_odd[DEFAULT_VECT_LEN]  = {0.0356,-0.1787,-0.1732,1.3311,-0.0085,-0.0066,0.0001};
static float *default_318_xt_vectors[] = {chan_xt_vect_318_even, chan_xt_vect_318_even,chan_xt_vect_318_even, chan_xt_vect_318_even,
    chan_xt_vect_318_odd, chan_xt_vect_318_odd,chan_xt_vect_318_odd, chan_xt_vect_318_odd
                                         };
int Image::chan_xt_column_offset[DEFAULT_VECT_LEN]   = {-12,-8,-4,0,4,8,12};

ChipXtVectArrayType Image::default_chip_xt_vect_array[] =
{
  {ChipId316, {default_316_xt_vectors, 1, DEFAULT_VECT_LEN, chan_xt_column_offset} },
  {ChipId318, {default_318_xt_vectors, 8, DEFAULT_VECT_LEN, chan_xt_column_offset} },
  {ChipIdUnknown, {NULL, 0,0,NULL} },
};

ChannelXTCorrectionDescriptor Image::selected_chip_xt_vectors = {NULL, 0,0,NULL};

// default constructor
Image::Image()
{
  results = NULL;
  raw = new RawImage;
  memset (raw,0,sizeof (*raw));
  maxFrames = 0;

  // set up the default SG-Filter class
  // MGD note - may want to ensure this becomes thread-safe or create one per region/thread
  sgSpread = 2;
  sgCoeff = 1;
  sgFilter = new SGFilter();
  sgFilter->SetFilterParameters (sgSpread, sgCoeff);
  experimentName = NULL;
  flowOffset = 1000;
  noFlowTime = 1350;
  bkg = NULL;



  retry_interval = 15;  // 15 seconds wait time.
  total_timeout = 36000;  // 10 hours before giving up.
  numAcqFiles = 0;    // total number of acq files in the dataset
  recklessAbandon = true; // false: use file availability testing during loading
  ignoreChecksumErrors = 0;
  dump_XTvects_to_file = 1;
}

Image::~Image()
{
  cleanupRaw();
  delete raw;
  delete sgFilter;
  if (results)
    delete [] results;
  if (experimentName)
    free (experimentName);
  if (bkg)
    free (bkg);

}

void Image::Close()
{
  cleanupRaw();

  if (results)
    delete [] results;
  results = NULL;

  if (bkg)
    free (bkg);
  bkg = NULL;

  maxFrames = 0;
}

void Image::cleanupRaw()
{
  if (raw->image)
  {
    free (raw->image);
    raw->image = NULL;
  }
  if (raw->timestamps)
  {
    free (raw->timestamps);
    raw->timestamps = NULL;
  }
  if (raw->interpolatedFrames)
  {
    free (raw->interpolatedFrames);
    raw->interpolatedFrames = NULL;
  }
  if (raw->interpolatedMult)
  {
    free (raw->interpolatedMult);
    raw->interpolatedMult = NULL;
  }

  memset (raw,0,sizeof (*raw));
}


//@TODO: this function has become unwieldy and needs refactoring.
bool Image::LoadSlice (
  vector<string> rawFileName,
  vector<unsigned int> col,
  vector<unsigned int> row,
  int minCol,
  int maxCol,
  int minRow,
  int maxRow,
  bool returnSignal,
  bool returnMean,
  bool returnSD,
  bool returnLag,
  bool uncompress,
  bool doNormalize,
  int normStart,
  int normEnd,
  bool XTCorrect,
  std::string chipType,
  double baselineMinTime,
  double baselineMaxTime,
  double loadMinTime,
  double loadMaxTime,
  unsigned int &nColFull,
  unsigned int &nRowFull,
  vector<unsigned int> &colOut,
  vector<unsigned int> &rowOut,
  unsigned int &nFrame,
  vector< vector<double> > &frameStart,
  vector< vector<double> > &frameEnd,
  vector< vector< vector<short> > > &signal,
  vector< vector<short> > &mean,
  vector< vector<short> > &sd,
  vector< vector<short> > &lag
)
{

  // Determine how many wells are sought
  unsigned int nSavedWells=0;
  bool cherryPickWells=false;
  if (col.size() > 0 || row.size() > 0)
  {
    if (row.size() != col.size())
    {
      cerr << "number of requested rows and columns should be the same" << endl;
      return (false);
    }
    nSavedWells = col.size();
    cherryPickWells=true;
    // Figure out min & max cols to load less data
    minCol = col[0];
    maxCol = minCol+1;
    minRow = row[0];
    maxRow = minRow+1;
    for (unsigned int iWell=1; iWell < col.size(); iWell++)
    {
      if (col[iWell] < (unsigned int) minCol)
        minCol = col[iWell];
      if (col[iWell] >= (unsigned int) maxCol)
        maxCol = 1+col[iWell];
      if (row[iWell] < (unsigned int) minRow)
        minRow = row[iWell];
      if (row[iWell] >= (unsigned int) maxRow)
        maxRow = 1+row[iWell];
    }
  }

  // Read header to determine full-chip rows & cols
  if (!LoadRaw (rawFileName[0].c_str(), 0, true, true))
  {
    cerr << "Problem loading dat file " << rawFileName[0] << endl;
    return (false);
  }
  nColFull = raw->cols;
  nRowFull = raw->rows;

  // This next block of ugliness will not be needed when we encode the chip type in the dat file
  char *chipID = NULL;
  if(chipType != "") {
    chipID = strdup(chipType.c_str());
  } else {
    if (nColFull == 1280 && nRowFull == 1152)
    {
      chipID = strdup ("314");
    }
    else if (nColFull == 2736 && nRowFull == 2640)
    {
      chipID = strdup ("316");
    }
    else if (nColFull == 3392 && nRowFull == 3792)
    {
      chipID = strdup ("318");
    }
    else
    {
      ION_WARN ("Unable to determine chip type from dimensions");
    }
  }

  // Only allow for XTCorrection on 316 and 318 chips
  if (XTCorrect)
  {
    if ( (chipID == NULL) || (strcmp (chipID,"318") && strcmp (chipID,"316")))
    {
      XTCorrect = false;
    }
  }

  // Variables to handle cases where we need to expand for proper
  // XTCorrection at boundaries
  int minColOuter;
  int maxColOuter;
  int minRowOuter;
  int maxRowOuter;
  if (minCol > -1 || minRow > -1 || maxCol > -1 || maxRow > -1)
  {
    // First do a few boundary checks
    bool badBoundary=false;
    if (minCol >= (int) nColFull)
    {
      cerr << "Error in Image::LoadSlice() - minCol is " << minCol << " which should be less than nColFull which is " << nColFull << endl;
      badBoundary=true;
    }
    if (maxCol <= 0)
    {
      cerr << "Error in Image::LoadSlice() - maxCol is " << maxCol << " which is less than 1" << endl;
      badBoundary=true;
    }
    if (minCol >= maxCol)
    {
      cerr << "Error in Image::LoadSlice() - maxCol is " << maxCol << " which is not greater than minCol which is " << minCol << endl;
      badBoundary=true;
    }
    if (minRow >= (int) nRowFull)
    {
      cerr << "Error in Image::LoadSlice() - minRow is " << minRow << " which should be less than nRowFull which is " << nRowFull << endl;
      badBoundary=true;
    }
    if (maxRow <= 0)
    {
      cerr << "Error in Image::LoadSlice() - maxRow is " << maxRow << " which is less than 1" << endl;
      badBoundary=true;
    }
    if (minRow >= maxRow)
    {
      cerr << "Error in Image::LoadSlice() - maxRow is " << maxRow << " which is not greater than minRow which is " << minRow << endl;
      badBoundary=true;
    }
    if (badBoundary)
      return (false);
    // Expand boundaries as necessary for XTCorrection
    minColOuter = minCol;
    maxColOuter = maxCol;
    minRowOuter = minRow;
    maxRowOuter = maxRow;
    if (XTCorrect)
    {
      minColOuter = std::max (0, minCol + chan_xt_column_offset[0]);
      maxColOuter = std::min ( (int) nColFull, maxCol + chan_xt_column_offset[DEFAULT_VECT_LEN-1]);
    }
  }
  else
  {
    if (minCol < 0)
      minCol = 0;
    if (minRow < 0)
      minRow = 0;
    if (maxCol < 0)
      maxCol = nColFull;
    if (maxRow < 0)
      maxRow = nRowFull;
    minColOuter = minCol;
    maxColOuter = maxCol;
    minRowOuter = minRow;
    maxRowOuter = maxRow;
  }
  // Set up Image class to read only the sub-range
  chipSubRegion.col = minColOuter;
  chipSubRegion.row = minRowOuter;
  chipSubRegion.w   = maxColOuter-minColOuter;
  chipSubRegion.h   = maxRowOuter-minRowOuter;
  // Set region origin for proper XTCorrection
  SetCroppedRegionOrigin (minColOuter,minRowOuter);

  unsigned int nPrevCol=0;
  unsigned int nPrevRow=0;
  unsigned int nPrevFramesCompressed=0;
  unsigned int nPrevFramesUncompressed=0;
  unsigned int nDat = rawFileName.size();
  bool problem = false;
  int *uncompressedTimestamps = NULL;
  vector<unsigned int> wellIndex;
  vector< vector<SampleStats<float> > > signalStats;
  vector< vector<SampleStats<float> > > lagStats;
  for (unsigned int iDat=0; iDat < nDat; iDat++)
  {
    if (!LoadRaw (rawFileName[iDat].c_str(), 0, true, false))
    {
      cerr << "Problem loading dat file " << rawFileName[iDat] << endl;
      return (false);
    }
    unsigned int nCol = raw->cols;
    unsigned int nRow = raw->rows;

    // We normalize if asked
    if (doNormalize)
    {
      Normalize (normStart,normEnd);
    }

    // cross-channel correction
    if (XTCorrect)
    {
      // Mask tempMask (raw->cols, raw->rows);
      if (chipID != NULL)
      {
        ChipIdDecoder::SetGlobalChipId (chipID);
        const char *rawDir = dirname ( (char *) rawFileName[iDat].c_str());
        CalibrateChannelXTCorrection (rawDir,"lsrowimage.dat");
        // XTChannelCorrect (&tempMask);
        XTChannelCorrect ();
      }
    }

    unsigned int nLoadedWells = nRow * nCol;
    unsigned int nFramesCompressed = raw->frames;
    unsigned int nFramesUncompressed = raw->uncompFrames;
    if (iDat==0)
    {
      nPrevCol=nCol;
      nPrevRow=nRow;
      nPrevFramesCompressed=nFramesCompressed;
      nPrevFramesUncompressed=nFramesUncompressed;
      // Size the return objects depending on whether or not we're returning compressed data
      if (uncompress)
      {
        nFrame = nFramesUncompressed;
      }
      else
      {
        nFrame = nFramesCompressed;
      }
      frameStart.resize (nDat);
      frameEnd.resize (nDat);
      for (unsigned int jDat=0; jDat<nDat; jDat++)
      {
        frameStart[jDat].resize (nFrame);
        frameEnd[jDat].resize (nFrame);
      }
      if (!cherryPickWells)
      {
        // If not using a specified set of row,col coordinates then return a rectangular region
        nSavedWells = (maxCol-minCol) * (maxRow-minRow);
      }
      if (returnSignal)
      {
        signal.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          signal[jDat].resize (nSavedWells);
          // Will resize for number of frames to return later, when that has been determined
        }
      }
      if (returnMean)
      {
        mean.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          mean[jDat].resize (nSavedWells);
        }
      }
      if (returnSD)
      {
        sd.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          sd[jDat].resize (nSavedWells);
        }
      }
      if (returnLag)
      {
        lag.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          lag[jDat].resize (nSavedWells);
        }
      }
      if (returnMean || returnSD)
      {
        signalStats.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          signalStats[jDat].resize (nSavedWells);
        }
      }
      if (returnLag)
      {
        lagStats.resize (nDat);
        for (unsigned int jDat=0; jDat<nDat; jDat++)
        {
          lagStats[jDat].resize (nSavedWells);
        }
      }
    }
    else
    {
      if (nCol != nPrevCol)
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of cols to first dat file " << rawFileName[0] << endl;
        return (false);
      }
      if (nRow != nPrevRow)
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of rows to first dat file " << rawFileName[0] << endl;
        return (false);
      }
      if (nFramesCompressed != nPrevFramesCompressed)
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of compressed frames to first dat file " << rawFileName[0] << endl;
        return (false);
      }
      if (nFramesUncompressed != nPrevFramesUncompressed)
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of uncompressed frames to first dat file " << rawFileName[0] << endl;
        return (false);
      }
    }

    // Determine well offsets to collect
    if (iDat==0)
    {
      wellIndex.resize (nSavedWells);
      colOut.resize (nSavedWells);
      rowOut.resize (nSavedWells);
      if (cherryPickWells)
      {
        for (unsigned int iWell=0; iWell < nSavedWells; iWell++)
        {
          wellIndex[iWell] = (row[iWell]-minRowOuter) *nCol + (col[iWell]-minColOuter);
          colOut[iWell] = col[iWell];
          rowOut[iWell] = row[iWell];
        }
      }
      else
      {
        for (unsigned int iWell=0, iRow=minRow; iRow < (unsigned int) maxRow; iRow++)
        {
          for (unsigned int iCol=minCol; iCol < (unsigned int) maxCol; iCol++, iWell++)
          {
            wellIndex[iWell] = (iRow-minRowOuter) *nCol + (iCol-minColOuter);
            colOut[iWell] = iCol;
            rowOut[iWell] = iRow;
          }
        }
      }
    }

    // Check if we need to uncompress
    // Make rawTimestamps point to the timestamps we need to report
    int *rawTimestamps = NULL;
    if (iDat==0 && uncompress && (nFramesUncompressed != nFramesCompressed))
    {
      int baseTime = raw->timestamps[0];
      uncompressedTimestamps = new int[sizeof (int) * nFramesUncompressed];
      uncompressedTimestamps[0] = 0;
      for (int iFrame=1; iFrame < raw->uncompFrames; iFrame++)
        uncompressedTimestamps[iFrame] = baseTime+uncompressedTimestamps[iFrame-1];
      rawTimestamps = uncompressedTimestamps;
    }
    else
    {
      rawTimestamps = raw->timestamps;
    }

    // Determine timeframe starts & stops
    bool oldMode = (rawTimestamps[0]==0);
    for (unsigned int iFrame=0; iFrame<nFrame; iFrame++)
    {
      if (oldMode)
      {
        frameStart[iDat][iFrame] = rawTimestamps[iFrame] / FRAME_SCALE;
        if (iFrame < nFrame-1)
          frameEnd[iDat][iFrame] = ( (double) rawTimestamps[iFrame+1]) / FRAME_SCALE;
        else if (iFrame > 0)
          frameEnd[iDat][iFrame] = (2.0 * (double) rawTimestamps[iFrame] - (double) rawTimestamps[iFrame-1]) / FRAME_SCALE;
        else
          frameEnd[iDat][iFrame] = 0;
      }
      else
      {
        frameEnd[iDat][iFrame] = ( (double) rawTimestamps[iFrame]) / FRAME_SCALE;
        frameStart[iDat][iFrame] = ( (double) ( (iFrame > 0) ? rawTimestamps[iFrame-1] : 0)) / FRAME_SCALE;
      }
    }

    // Determing per-well baseline values to subtract
    bool doBaseline=false;
    int baselineMinFrame = -1;
    int baselineMaxFrame = -1;
    std::vector<double> baselineWeight;
    if (baselineMinTime < baselineMaxTime)
    {
      double baselineWeightSum = 0;
      for (unsigned int iFrame=0; iFrame < nFrame; iFrame++)
      {
        if ( (frameStart[iDat][iFrame] > (baselineMinTime-numeric_limits<double>::epsilon())) && frameEnd[iDat][iFrame] < (baselineMaxTime+numeric_limits<double>::epsilon()))
        {
          // frame is in our baseline timeframe
          if (baselineMinFrame < 0)
            baselineMinFrame = iFrame;
          baselineMaxFrame = iFrame;
          baselineWeight.push_back (frameEnd[iDat][iFrame]-frameStart[iDat][iFrame]);
          baselineWeightSum += frameEnd[iDat][iFrame]-frameStart[iDat][iFrame];
        }
      }
      if (baselineWeightSum > 0)
      {
        unsigned int nBaselineFrame = baselineWeight.size();
        for (unsigned int iFrame=0; iFrame < nBaselineFrame; iFrame++)
          baselineWeight[iFrame] /= baselineWeightSum;
        doBaseline=true;
      }
    }
    vector<double> baselineVal;
    short bVal=0;
    if (doBaseline)
    {
      baselineVal.resize (nSavedWells,0);
      for (unsigned int iWell=0; iWell < nSavedWells; iWell++)
      {
        for (int iFrame=baselineMinFrame; iFrame <= baselineMaxFrame; iFrame++)
        {
          if (uncompress)
            bVal = GetInterpolatedValue (iFrame, colOut[iWell]-minColOuter, rowOut[iWell]-minRowOuter);
          else
            bVal = raw->image[iFrame * nLoadedWells + wellIndex[iWell]];
          baselineVal[iWell] += baselineWeight[iFrame-baselineMinFrame] * bVal;
        }
      }
    }

    // Determine which frames to return
    int loadMinFrame = 0;
    int loadMaxFrame = nFrame;
    unsigned int nLoadFrame = nFrame;
    bool loadFrameSubset=false;
    if (loadMinTime < loadMaxTime)
    {
      loadMinFrame = -1;
      for (unsigned int iFrame=0; iFrame < nFrame; iFrame++)
      {
        if ( (frameStart[iDat][iFrame] > (loadMinTime-numeric_limits<double>::epsilon())) && frameEnd[iDat][iFrame] < (loadMaxTime+numeric_limits<double>::epsilon()))
        {
          if (loadMinFrame < 0)
            loadMinFrame = iFrame;
          loadMaxFrame = iFrame+1;
        }
      }
      if (loadMinFrame == -1)
      {
        cerr << "Image::LoadSlice - no frames found in requested timeframe\n";
        problem=true;
        break;
      }
      else
      {
        nLoadFrame = loadMaxFrame-loadMinFrame;
        loadFrameSubset=true;
      }
    }

    // resize return signal object
    if (returnSignal)
    {
      for (unsigned int iWell=0; iWell < nSavedWells; iWell++)
        signal[iDat][iWell].resize (nLoadFrame);
    }
    // subset frameStart/frameEnd for frame range requested
    if (loadFrameSubset)
    {
      if (loadMaxFrame < (int) nFrame)
      {
        unsigned int nToDrop = nFrame - loadMaxFrame;
        frameStart[iDat].erase (frameStart[iDat].end()-nToDrop,frameStart[iDat].end());
        frameEnd[iDat].erase (frameEnd[iDat].end()-nToDrop,frameEnd[iDat].end());
      }
      if (loadMinFrame > 0)
      {
        unsigned int nToDrop = loadMinFrame+1;
        frameStart[iDat].erase (frameStart[iDat].begin(),frameStart[iDat].begin() +nToDrop);
        frameEnd[iDat].erase (frameEnd[iDat].begin(),frameEnd[iDat].begin() +nToDrop);
      }
    }

    short val=0;
    for (unsigned int iWell=0; iWell < nSavedWells; iWell++)
    {
      // do frames
      short oldval =0;
      for (int iFrame=loadMinFrame; iFrame < loadMaxFrame; iFrame++)
      {
        if (uncompress)
          val = GetInterpolatedValue (iFrame, colOut[iWell]-minColOuter, rowOut[iWell]-minRowOuter);
        else
          val = raw->image[iFrame * nLoadedWells + wellIndex[iWell]];
        if (doBaseline)
        {
          val = (short) ( (double) val - baselineVal[iWell]);
        }
        if (returnSignal)
          signal[iDat][iWell][iFrame-loadMinFrame] = val;
        if (returnMean || returnSD)
          signalStats[iDat][iWell].AddValue ( (float) val);
        if (returnLag & (iFrame>loadMinFrame))
          lagStats[iDat][iWell].AddValue ( (float) (val-oldval));
        oldval = val;
      }
    }
    for (unsigned int iWell=0; iWell < nSavedWells; iWell++)
    {
      if (returnMean)
      {
        mean[iDat][iWell] = (short) (signalStats[iDat][iWell].GetMean());
      }
      if (returnSD)
      {
        sd[iDat][iWell] = (short) (signalStats[iDat][iWell].GetSD());
      }
      if (returnLag)
      {
        lag[iDat][iWell] = (short) (lagStats[iDat][iWell].GetSD()); // sd on lagged signal
      }
    }

    cleanupRaw();
  }

  if (chipID != NULL)
    free (chipID);

  if (uncompressedTimestamps != NULL)
    delete [] uncompressedTimestamps;

  if (problem)
    return (false);
  else
    return (true);

};

//
// LoadRaw
// loads raw image data for one experiment, and byte-swaps as appropriate
// returns a structure with header data and a pointer to the allocated image data with timesteps removed
// A couple prototypes are defined so all the old applications work with the old argument lists
// the argumant defaults are in the .h file.  
// this is the original prototype




// This is the actual function
bool Image::LoadRaw (const char *rawFileName, int frames, bool allocate, bool headerOnly)
{
//  _file_hdr hdr;
//  int offset=0;
  int rc;
  (void) allocate;

  cleanupRaw();

  //set default name only if not already set
  if (!experimentName)
  {
    experimentName = (char *) malloc (3);
    strncpy (experimentName, "./", 3);
  }

  //DEBUG: monitor file access time
  struct timeval tv;
  double startT;
  double stopT;
  gettimeofday (&tv, NULL);
  startT = (double) tv.tv_sec + ( (double) tv.tv_usec/1000000);

  FILE *fp = NULL;
  if (recklessAbandon)
  {
    fopen_s (&fp, rawFileName, "rb");
  }
  else    // Try open and wait until open or timeout
  {

    uint32_t waitTime = retry_interval;
    int32_t timeOut = total_timeout;
    //--- Wait up to 3600 seconds for a file to be available
    while (timeOut > 0)
    {
      //--- Is the file we want available?
      if (ReadyToLoad (rawFileName))
      {
        //--- Open the file we want
        fopen_s (&fp, rawFileName, "rb");
        break;  // any error will be reported below
      }
      //DEBUG
      fprintf (stdout, "Waiting to load %s\n", rawFileName);
      sleep (waitTime);
      timeOut -= waitTime;
    }

  }
  if (fp == NULL)
  {
    perror (rawFileName);
    return false;
  }

  printf ("\nLoading raw file: %s...\n", rawFileName);
  fflush (stdout);
//  size_t rdSize;

  fclose (fp);

  raw->channels = 4;
  raw->interlaceType = 0;
  raw->image = NULL;
  if (frames)
    raw->frames = frames;


  if (headerOnly)
  {
    rc = deInterlace_c ( (char *) rawFileName,NULL,NULL,
                         &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                         0,0,chipSubRegion.col,chipSubRegion.row,chipSubRegion.col+chipSubRegion.w,chipSubRegion.row+chipSubRegion.h,ignoreChecksumErrors);
  }
  else
  {
    rc = deInterlace_c ( (char *) rawFileName,&raw->image,&raw->timestamps,
                         &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                         0,0,chipSubRegion.col,chipSubRegion.row,chipSubRegion.col+chipSubRegion.w,chipSubRegion.row+chipSubRegion.h,ignoreChecksumErrors);


    if (chipSubRegion.h != 0)
      raw->rows = chipSubRegion.h;
    if (chipSubRegion.w != 0)
      raw->cols = chipSubRegion.w;



     raw->baseFrameRate=raw->timestamps[0];
    if (raw->baseFrameRate == 0)
      raw->baseFrameRate=raw->timestamps[1];
    if (raw->uncompFrames != raw->frames)
    {

      // create a temporary set of timestamps that indicate the centroid of each averaged data point
      // from the ones in the file (which indicate the end of each data point)
      float *centr_timestamps = (float *) malloc (sizeof (float) *raw->frames);
      float last_timestamp = 0.0;
      for (int j=0;j<raw->frames;j++)
      {
        centr_timestamps[j] = (raw->timestamps[j] + last_timestamp) /2.0;
        last_timestamp = raw->timestamps[j];
      }

      raw->interpolatedFrames = (int *) malloc (sizeof (int) *raw->uncompFrames);
      raw->interpolatedMult = (float *) malloc (sizeof (float) *raw->uncompFrames);
      for (int i=0;i<raw->uncompFrames;i++)
      {
        int curTime=raw->baseFrameRate*i;
        int prevTime,nextTime;

        if (curTime > centr_timestamps[raw->frames-1])
        {
          // the last several points must actually be extrapolated
          nextTime = centr_timestamps[raw->frames-1];
          prevTime = centr_timestamps[raw->frames-2];

          raw->interpolatedFrames[i] = raw->frames-1;
          raw->interpolatedMult[i] = (float) (nextTime-curTime) / (float) (nextTime-prevTime);
        }
        else
          for (int j=0;j<raw->frames;j++)
          {
            if (centr_timestamps[j] >= curTime)
            {
              nextTime = centr_timestamps[j];

              if (j)
                prevTime = centr_timestamps[j-1];
              else
                prevTime = 0;

              raw->interpolatedFrames[i] = j;
              raw->interpolatedMult[i] = (float) (nextTime-curTime) / (float) (nextTime-prevTime);

              break;
            }
          }
      }

      free (centr_timestamps);
    }
  }

  raw->frameStride = raw->rows * raw->cols;

  printf ("Loading raw file: %s...done\n", rawFileName);

  if (rc && raw->timestamps)
  {
    uint32_t prev = 0;
    float avgTimestamp = 0;
    double fps;

    // read the raw data, and convert it into image data
    int i;

    for (i=0;i<raw->frames;i++)
    {
      avgTimestamp += (raw->timestamps[i] - prev);
      prev = raw->timestamps[i];
    }
    avgTimestamp = avgTimestamp / (raw->frames - 1);  // milliseconds
    fps = (1000.0/avgTimestamp);  // convert to frames per second


    // Subtle hint to users of "old" cropped datasets that did not have real timestamps written
    /*if (rint(fps) == 10) {
      fprintf (stdout, "\n\nWARNING: if this is a cropped dataset, it may have incorrect frame timestamp!\n");
      fprintf (stdout, "Your results will not be valid\n\n");
    }*/

    //DEBUG
    fprintf (stdout, "Avg Image Time = %f ", avgTimestamp);
    fprintf (stdout, "Frames = %d ", raw->frames);
    fprintf (stdout, "FPS = %f\n", fps);
    gettimeofday (&tv, NULL);
    stopT = (double) tv.tv_sec + ( (double) tv.tv_usec/1000000);
    fprintf (stdout, "File access = %0.2lf sec.\n", stopT - startT);
    fflush (stdout);
  }

  ReportPinnedWells (std::string (rawFileName));


  return true;
}

void Image::ReportPinnedWells (const std::string& strDatFileName)
{
  // Create a PinnedWellReporter object instance.
  // To disable the reporter, pass false into Instance()
  // by calling PinnedWellReporter::Instance( false ).
  //
  // Note:  If PinnedWellReporter::Instance() is called before this
  // call, below, then the state, either enabled or disabled, is already
  // set and cannot be changed for the life of the application.
  //
  // For example, this call to Instance() passes false to disable
  // this instance.  However, if a previous call to Instance was
  // made before this call to Instance(), then the enable/disable state
  // of the previous call is used.  This is because the enable/disable
  // state is stored in a private static flag that is set on the first call to
  // PinnedWellReporter::Instance().  Any subsequent calls to Instance()
  // will not change the enable/disable state.
  //
  // Therefore, to enable PinnedWellReporter::Instance(), be sure
  // to call Instance() early on, preferrably in main() so that there
  // will be no mysteries as to why the PinnedWellReporter is or is not
  // working.
  PWR::PinnedWellReporter& pwr = *PWR::PinnedWellReporter::Instance (false);

  // Do only if the well is pinned and the reporter object is enabled.
  if (PWR::PinnedWellReporter::IsEnabled())
  {
    // Pinned value constants.
    const int PIN_LOW_VALUE = 0;
    const int PIN_HIGH_VALUE = 0x3fff;

    // Linear position index into the raw->image array.
    int i = 0;

    // Do for each row in the well matrix.
    for (int y = 0; y < raw->rows; y++)
    {
      // Do for each column in the well matrix.
      for (int x = 0; x < raw->cols; x++)
      {
        // Accumulate data for well x,y here...
        PWR::PinnedWell pinnedWell;

        // Count of pinned frames in this well.
        int numPinnedFramesInWell = 0;

        // Is this frame pinned?
        bool bIsPinned = false;

        // Assume a pinned low state
        PWR::PinnedWellReporter::PinnedStateEnum pinnedState
        = PWR::PinnedWellReporter::LOW;

        // Do for each frame in this well.
        for (int frame = 0; frame < raw->frames; frame++)
        {
          // Pre-calculate the index into the image.
          const int iImgIndex = frame * raw->frameStride + i;
          // Do if the value is pinned.
          if (PIN_LOW_VALUE == raw->image[iImgIndex]
              || PIN_HIGH_VALUE == raw->image[iImgIndex])
          {
            // This pixel value is pinned either high or low.
            bIsPinned = true;

            // Accumulate the number of pinned frames in this well.
            numPinnedFramesInWell++;

            // If pinned high, then set the state accordingly.
            if (PIN_HIGH_VALUE == raw->image[iImgIndex])
              pinnedState = PWR::PinnedWellReporter::HIGH;
          }
        } // END for( int frame = 0; frame < raw->frames; frame++ )

        // Do only if this frame is pinned.
        if (bIsPinned)
        {
          // Calculate the value in this frame.
          unsigned int valueInFrame = raw->image[i];

          // Accumulate information at this well.
          pinnedWell.Add (x, y, valueInFrame, pinnedState);

        } // END if( bIsPinned )

        // Add the pinned well data into the reporter instance.
        pwr.Write (strDatFileName, pinnedWell, numPinnedFramesInWell);

        // Increment the linear position index
        i++;
      } // END for( int x = 0; x < raw->cols; x++ )
    } // END for( int y = 0; y < raw->rows; y++ )
  } // END if( PWR::PinnedWellReporter::IsEnabled() )
} // END Image::ReportedFilterForPinned()

void Image::SetDir (const char *directory)
{
  if (experimentName)
    free (experimentName);
  experimentName = (char *) malloc (strlen (directory) + 1);
  strncpy (experimentName, directory, strlen (directory) + 1);
  return;
}

int Image::FilterForPinned (Mask *mask, MaskType these, int markBead)
{

  int x, y, frame;
  int pinnedCount = 0;
  int i = 0;
  const short pinLow = GetPinLow();
  const short pinHigh = GetPinHigh();

  printf ("Filtering for pinned pixels between %d & %d.\n", pinLow, pinHigh);

  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      if ( (*mask) [i] & these)
      {
        for (frame=0;frame<raw->frames;frame++)
        {
          if (raw->image[frame*raw->frameStride + i] <= pinLow ||
              raw->image[frame*raw->frameStride + i] >= pinHigh)
          {
            (*mask) [i] = MaskPinned; // this pixel is pinned high or low
            if (markBead)
              (*mask) [i] |= MaskBead;
            pinnedCount++;
            break;
          }
        }
      }
      i++;
    }
  }
  fprintf (stdout, "FilterForPinned: found %d\n", pinnedCount);
  return pinnedCount;
}

void Image::Normalize (int startPos, int endPos)
{
  // normalize trace data to the input frame per well
  printf ("Normalizing from frame %d to %d\n",startPos,endPos);
  int frame, x, y;
  int ref;
  int i = 0;
  short *imagePtr;
  int nframes = raw->frames;

  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      int pos;
      ref = 0;
      for (pos=startPos;pos<=endPos;pos++)
        ref += raw->image[pos*raw->frameStride+i];
      ref /= (endPos-startPos+1);
      imagePtr = &raw->image[i];
      for (frame=0;frame<nframes;frame++)
      {
        *imagePtr -= ref;
        imagePtr += raw->frameStride;
      }
      i++;
    }
  }
  printf ("Normalizing...done\n");
}

void Image::IntegrateRaw (Mask *mask, MaskType these, int start, int end)
{
  printf ("Integrating Raw...\n");
  if (!results)
    results = new double[raw->rows * raw->cols];
  memset (results, 0, raw->rows * raw->cols * sizeof (double));

  int x, y, frame, k;
  double buf;
  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      if ( (*mask) [x+raw->cols*y] & these)
      {
        k = y*raw->cols + x + start*raw->frameStride;
        buf = 0;
        for (frame=start;frame<=end;frame++)
        {
          buf += raw->image[k];
          k += raw->frameStride;
        }
        results[x+y*raw->cols] = buf; // /(double)(end-start+1.0);
      }
    }
  }
}

void Image::IntegrateRawBaseline (Mask *mask, MaskType these, int start, int end, int baselineStart, int baselineEnd, double *_minval, double *_maxval)
{
  printf ("Integrating Raw Baseline...\n");
  if (!results)
    results = new double[raw->rows * raw->cols];
  memset (results, 0, raw->rows * raw->cols * sizeof (double));

  int x, y, frame, k;
  double buf;
  double minval = 99999999999.0, maxval = -99999999999.0;
  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      if ( (*mask) [x+raw->cols*y] & these)
      {
        k = y*raw->cols + x + start*raw->frameStride;
        buf = 0;
        int pos;
        int ref = 0;
        for (pos=baselineStart;pos<=baselineEnd;pos++)
          ref += raw->image[pos*raw->frameStride+x+y*raw->cols];
        ref /= (baselineEnd-baselineStart+1);
        for (frame=start;frame<=end;frame++)
        {
          buf += (raw->image[k] - ref);
          k += raw->frameStride;
        }
        results[x+y*raw->cols] = buf; // /(double)(end-start+1.0);
        if (buf > maxval) maxval = buf;
        if (buf < minval) minval = buf;
      }
    }
  }
  if (_minval) *_minval = minval;
  if (_maxval) *_maxval = maxval;
}


/*
void Image::SubtractFile(RawImage *ref)
{
  printf("Subtracting reference file from image...\n");

  if (ref->rows == raw->rows && ref->cols == raw->cols && ref->frames == raw->frames) {
    int frame, x, y;
    int k;
    for(frame=0;frame<raw->frames;frame++) {
      for(y=0;y<raw->rows;y++) {
        k = frame*raw->frameStride + y*raw->cols;
        for(x=0;x<raw->cols;x++) {
          raw->image[k] -= ref->image[k];
          k++;
        }
      }
    }
  }
}
*/

void Image::FindPeak (Mask *mask, MaskType these)
{
  if (!results)
    results = new double[raw->rows * raw->cols];
  memset (results, 0, raw->rows * raw->cols * sizeof (double));

  int frame, x, y;
  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      if ( (*mask) [x+raw->cols*y] & these)
      {
        for (frame=0;frame<raw->frames;frame++)
        {
          if (frame == 0 || (raw->image[frame*raw->frameStride + x+y*raw->cols] > results[x+y*raw->cols]))
          {
            results[x+y*raw->cols] = frame; // only need to save what frame we found the peak
          }
        }
      }
    }
  }
}
void Image::BackgroundCorrect (Mask *mask, MaskType these, MaskType usingThese, int inner, int outer, Filter *f, bool saveBkg, bool onlyBkg, bool replaceWBkg)
{
  BackgroundCorrect (mask, these, usingThese, inner, inner, outer, outer, f, saveBkg, onlyBkg,replaceWBkg);
}

void Image::BackgroundCorrect (Mask *mask, MaskType these, MaskType usingThese, int innerx, int innery, int outerx, int outery, Filter *f, bool saveBkg, bool onlyBkg, bool replaceWBkg)
{
//  return BackgroundCorrectMulti(mask,these,usingThese,innerx,innery,outerx,outery,f,saveBkg);
  // BackgroundCorrect - Algorithm is as follows:
  //   grabs an NxN area around a bead,
  //   only looks for empty wells,
  //   averages those traces,
  //   subtracts from bead well
  //
  // Assumptions:  data has all been normalized to some frame
  // Improvements: could add an NxN weighting matrix, applied only on the normalization pass (so its fast), may correct for cross-talk this way?
  // Improvements: with lots of additional memory (or maybe not, could use a buf[frames] thing to temp store to), could store the avg trace for each bead's background result, then sg-filter prior to subtraction

  // allocate a temporary one-frame buffer
  int64_t *workTotal = (int64_t *) malloc (sizeof (int64_t) *raw->rows*raw->cols);
  unsigned int *workNum   = (unsigned int *) malloc (sizeof (unsigned int) *raw->rows*raw->cols);

  int x, y, frame;
  int64_t sum,innersum;
  int count,innercount;
  short *fptr;
  short *Rfptr;

  uint16_t *MaskPtr = (uint16_t *) mask->GetMask();
  register uint16_t lmsk,*rMaskPtr;
  unsigned int *lWorkNumPtr;
  int64_t *lWorkTotalPtr;

  if (saveBkg)
  {
    if (bkg)
      free (bkg);
    bkg = (int16_t *) malloc (sizeof (int16_t) * raw->rows*raw->cols*raw->frames);
    memset (bkg,0,sizeof (int16_t) *raw->rows*raw->cols*raw->frames);
  }

  for (frame=0;frame<raw->frames;frame++)
  {

    memset (workNum  ,0,sizeof (unsigned int) *raw->rows*raw->cols);
    memset (workTotal,0,sizeof (int64_t) *raw->rows*raw->cols);
    fptr = &raw->image[frame*raw->frameStride];
    lWorkTotalPtr = workTotal;
    lWorkNumPtr = workNum;
//    int skipped = 0;

    // sum empty wells on the whole image
    for (y=0;y<raw->rows;y++)
    {
      rMaskPtr = &MaskPtr[raw->cols*y];
      Rfptr = &fptr[raw->cols*y];
      for (x=0;x<raw->cols;x++)
      {
        lmsk = rMaskPtr[x];

        if ( (lmsk & usingThese) // look only at our beads...
             /*!(lmsk & MaskWashout)*/) // Skip any well marked ignore
        {
          *lWorkTotalPtr = Rfptr[x];
          *lWorkNumPtr   = 1;
        }
        else
        {
          *lWorkTotalPtr = 0;
          *lWorkNumPtr   = 0;
        }
        if (x)
        {
          *lWorkNumPtr   += * (lWorkNumPtr-1);  // the one to the left
          *lWorkTotalPtr += * (lWorkTotalPtr-1); // the one to the left
        }
        if (y)
        {
          *lWorkNumPtr   += * (lWorkNumPtr   - raw->cols); // the one above
          *lWorkTotalPtr += * (lWorkTotalPtr - raw->cols); // the one above
        }
        if (x && y)
        {
          *lWorkNumPtr   -= * (lWorkNumPtr   - raw->cols - 1); // add the common area
          *lWorkTotalPtr -= * (lWorkTotalPtr - raw->cols - 1); // add the common area
        }
        lWorkNumPtr++;
        lWorkTotalPtr++;
      }
    }

    // compute the whole chip subtraction coefficient
    int yi1,yi2,xi1,xi2,tli,bli,tri,bri;
    int yo1 = (y-outery) < 0 ? 0 : (y-outery);
    int yo2 = (y+outery) >= raw->rows ? raw->rows-1 : (y+outery);
    int xo1 = (x-outerx) < 0 ? 0 : (x-outerx);
    int xo2 = (x+outerx) >= raw->cols ? raw->cols-1 : (x+outerx);
    int tlo = 0;
    int blo = (raw->rows-1) *raw->cols;
    int tro = raw->cols-1;
    int bro = (raw->rows-1) *raw->cols + raw->cols-1;
    sum    = workTotal[bro] + workTotal[tlo];
    sum   -= workTotal[tro] + workTotal[blo];
    count  = workNum[bro] + workNum[tlo];
    count -= workNum[tro] + workNum[blo];
    int WholeBkg = 0;
    if (sum != 0 && count != 0)
    {
      WholeBkg = sum/count;
    }

    // now, compute background for each live bead
    for (y=0;y<raw->rows;y++)
    {
      rMaskPtr = &MaskPtr[raw->cols*y];
      Rfptr = &fptr[raw->cols*y];
      for (x=0;x<raw->cols;x++)
      {
        lmsk = rMaskPtr[x];
        if ( (lmsk & these))
        {
          yo1 = (y-outery) < 0 ? 0 : (y-outery);
          yo2 = (y+outery) >= raw->rows ? raw->rows-1 : (y+outery);
          xo1 = (x-outerx) < 0 ? 0 : (x-outerx);
          xo2 = (x+outerx) >= raw->cols ? raw->cols-1 : (x+outerx);
          yi1 = (y-innery) < 0 ? 0 : (y-innery);
          yi2 = (y+innery) >= raw->rows ? raw->rows-1 : (y+innery);
          xi1 = (x-innerx) < 0 ? 0 : (x-innerx);
          xi2 = (x+innerx) >= raw->cols ? raw->cols-1 : (x+innerx);

          tli = (yi1?yi1-1:0) *raw->cols + (xi1?xi1-1:0);
          bli = yi2*raw->cols + (xi1?xi1-1:0);
          tri = (yi1?yi1-1:0) *raw->cols + xi2;
          bri = yi2*raw->cols + xi2;
          tlo = (yo1?yo1-1:0) *raw->cols + (xo1?xo1-1:0);
          blo = yo2*raw->cols + (xo1?xo1-1:0);
          tro = (yo1?yo1-1:0) *raw->cols + xo2;
          bro = yo2*raw->cols + xo2;
          sum    = workTotal[bro] + workTotal[tlo];
          sum   -= workTotal[tro] + workTotal[blo];
          innersum  = workTotal[bri] + workTotal[tli];
          innersum -= workTotal[tri] + workTotal[bli];
          sum -= innersum;
          count  = workNum[bro] + workNum[tlo];
          count -= workNum[tro] + workNum[blo];
          innercount  = workNum[bri] + workNum[tli];
          innercount -= workNum[tri] + workNum[bli];
          count -= innercount;
          if (count > 0)
          {
            sum /= count;
          }
          else
          {
            sum = WholeBkg;
          }
          // update the value (background subtract) in the work buffer
          if (!onlyBkg)
          {
            if (replaceWBkg)
              Rfptr[x] = sum;
            else
              Rfptr[x] -= sum;
          }

          if (saveBkg)
          {
            bkg[frame*raw->frameStride+y*raw->cols+x] = sum;
          }
        }
//        else
//          skipped++;
      }
    }
  }

  free (workNum);
  free (workTotal);

#if 0
  static int FileNum = 0;
  static int ThisIsIt = 0;

  {
    char name[256];
    int x,y;

    sprintf (name,"/home/proton/tstNew/img%d.txt",FileNum++);
    FILE *fp = fopen (name,"wb");
    if (fp)
    {
#if 0
      for (frame=0;frame<raw->frames;frame++)
      {
        for (y=0;y<raw->rows;raw++)
        {
          //          fprintf(fp,"%d) ",y);
          for (x=0;x<raw->cols;x++)
          {
            fprintf (fp,"%f, ",raw->image[y*raw->cols + x]);
          }
          fprintf (fp,"\n");
        }
      }
#else
      fwrite (raw->image,raw->frameStride*raw->frames,2,fp);
#endif
      fclose (fp);
    }
  }
#endif

}

void Image::BackgroundCorrectRegion (Mask *mask, Region &reg, MaskType these, MaskType usingThese, int innerx, int innery,
                                     int outerx, int outery, Filter *f, bool saveBkg, bool onlyBkg, bool replaceWBkg)
{
  // BackgroundCorrect - Algorithm is as follows:
  //   grabs an NxN area around a bead,
  //   only looks for empty wells,
  //   averages those traces,
  //   subtracts from bead well
  //
  // Assumptions:  data has all been normalized to some frame
  // Improvements: could add an NxN weighting matrix, applied only on the normalization pass (so its fast), may correct for cross-talk this way?
  // Improvements: with lots of additional memory (or maybe not, could use a buf[frames] thing to temp store to), could store the avg trace for each bead's background result, then sg-filter prior to subtraction

  // allocate a temporary one-frame buffer
  int64_t *workTotal = (int64_t *) malloc (sizeof (int64_t) *raw->rows*raw->cols);
  unsigned int *workNum   = (unsigned int *) malloc (sizeof (unsigned int) *raw->rows*raw->cols);

  int x, y, frame;
  int64_t sum,innersum;
  int count,innercount;
  short *fptr;
  uint16_t *MaskPtr = (uint16_t *) mask->GetMask();
  register uint16_t lmsk;
  unsigned int *lWorkNumPtr;
  int64_t *lWorkTotalPtr;

  if (saveBkg)
  {
    if (bkg)
      free (bkg);
    bkg = (int16_t *) malloc (sizeof (int16_t) * raw->rows*raw->cols*raw->frames);
    memset (bkg,0,sizeof (int16_t) *raw->rows*raw->cols*raw->frames);
  }

  for (frame=0;frame<raw->frames;frame++)
  {

    memset (workNum  ,0,sizeof (unsigned int) *raw->rows*raw->cols);
    memset (workTotal,0,sizeof (int64_t) *raw->rows*raw->cols);
    fptr = &raw->image[frame*raw->frameStride];
    lWorkTotalPtr = workTotal;
    lWorkNumPtr = workNum;
    //    int skipped = 0;
    int rowStart = reg.row;
    int rowEnd = reg.row+reg.h;
    int colStart = reg.col;
    int colEnd = reg.col+reg.w;

    // calculate cumulative sum once so fast to calculate sum  empty wells on the whole image
    for (y=rowStart;y<rowEnd;y++)
    {
      for (x=colStart;x<colEnd;x++)
      {
        int wellIx = y * raw->cols + x;
        lmsk = MaskPtr[wellIx];
        if ( (lmsk & usingThese)) // look only at our beads...
        {
          lWorkTotalPtr[wellIx] = fptr[wellIx];
          lWorkNumPtr[wellIx]   = 1;
        }
        else
        {
          lWorkTotalPtr[wellIx] = 0;
          lWorkNumPtr[wellIx]   = 0;
        }
        if (x != colStart)
        {
          lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx-1];   // the one to the left
          lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx-1]; // the one to the left
        }
        if (y != rowStart)
        {
          lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx - raw->cols]; // the one below
          lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx - raw->cols]; // the one below
        }
        if (x != colStart && y != rowStart)
        {
          lWorkNumPtr[wellIx]   -= lWorkNumPtr[wellIx - raw->cols - 1]; // add the common area
          lWorkTotalPtr[wellIx] -= lWorkTotalPtr[wellIx - raw->cols - 1]; // add the common area
        }
      }
    }

    // Compute the whole chip subtraction coefficient
    int yi1,yi2,xi1,xi2,tli,bli,tri,bri;
    int yo1 = (y-outery) < rowStart ? rowStart : (y-outery);
    int yo2 = (y+outery) >= rowEnd ? rowEnd-1 : (y+outery);
    int xo1 = (x-outerx) < colStart ? colStart : (x-outerx);
    int xo2 = (x+outerx) >= colEnd ? colEnd-1 : (x+outerx);
    int tlo = rowStart * raw->cols + colStart;
    int blo = (rowEnd-1) *raw->cols + colStart;
    int tro = rowStart * raw->cols + colEnd-1;
    int bro = (rowEnd-1) *raw->cols + colEnd-1;

    sum    = workTotal[bro];
    count = workNum[bro];
    int WholeBkg = 0;
    if (sum != 0 && count != 0)
    {
      WholeBkg = sum/count;
    }

    // now, compute background for each live bead
    for (y=rowStart;y<rowEnd;y++)
    {
      //      rMaskPtr = &MaskPtr[raw->cols*y];
      //      Rfptr = &fptr[raw->cols*y];
      for (x=colStart;x<colEnd;x++)
      {
        int wellIx = y * raw->cols + x;
        lmsk = MaskPtr[wellIx];        //if ( (lmsk & these) != 0 )
        if ( (lmsk & these))
        {
          yo1 = (y-outery) < rowStart ? rowStart : (y-outery);
          yo2 = (y+outery) >= rowEnd ? rowEnd-1 : (y+outery);
          xo1 = (x-outerx) < colStart ? colStart : (x-outerx);
          xo2 = (x+outerx) >= colEnd ? colEnd-1 : (x+outerx);
          yi1 = (y-innery) < rowStart ? rowStart : (y-innery);
          yi2 = (y+innery) >= rowEnd ? rowEnd-1 : (y+innery);
          xi1 = (x-innerx) < colStart ? colStart : (x-innerx);
          xi2 = (x+innerx) >= colEnd ? colEnd-1 : (x+innerx);

          tli = (yi1 != rowStart ? yi1-1: rowStart) * raw->cols + (xi1 != colStart ? xi1-1 : colStart);
          bli = yi2*raw->cols + (xi1 != colStart ? xi1-1 : colStart);
          tri = (yi1 != rowStart ? yi1-1 : rowStart) *raw->cols + xi2;
          bri = yi2*raw->cols + xi2;
          tlo = (yo1 != rowStart ? yo1-1 : rowStart) *raw->cols + (xo1 != colStart? xo1-1 : colStart);
          blo = yo2*raw->cols + (xo1 != colStart ? xo1-1 : colStart);
          tro = (yo1 != rowStart ? yo1-1:rowStart) *raw->cols + xo2;
          bro = yo2*raw->cols + xo2;

          sum = workTotal[bro];
          count  = workNum[bro];
          if (yo1 != rowStart)
          {
            sum -= workTotal[tro];
            count -= workNum[tro];
          }
          if (xo1 != colStart)
          {
            sum -= workTotal[blo];
            count -= workNum[blo];
          }
          if (xo1 != colStart && yo1 != rowStart)
          {
            sum += workTotal[tlo];
            count += workNum[tlo];
          }

          innersum = workTotal[bri];
          innercount  = workNum[bri];
          if (yi1 != rowStart)
          {
            innersum -= workTotal[tri];
            innercount -= workNum[tri];
          }
          if (xi1 != colStart)
          {
            innersum -= workTotal[bli];
            innercount -= workNum[bli];
          }
          if (xi1 != colStart && yi1 != rowStart)
          {
            innersum += workTotal[tli];
            innercount += workNum[tli];
          }

          if (count > 0)
          {
            sum /= count;
          }
          else
          {
            sum = WholeBkg;
          }
          // update the value (background subtract) in the work buffer
          if (!onlyBkg)
          {
            if (replaceWBkg)
              fptr[wellIx] = sum;
            else
              fptr[wellIx] -= sum;
          }
          if (saveBkg)
          {
            bkg[frame*raw->frameStride+y*raw->cols+x] = sum;
          }
        }
      }
    }
  }

  free (workNum);
  free (workTotal);
}

void Image::SGFilterSet (int spread, int coeff)
{
  if ( (spread != sgSpread) || (coeff != sgCoeff))
  {
    sgSpread = spread;
    sgCoeff = coeff;
    sgFilter->SetFilterParameters (sgSpread, sgCoeff);
  }
}

void Image::SGFilterApply (Mask *mask, MaskType these)
{
  printf ("SG-Filtering...\n");
  float *trace = new float[raw->frames];
  float *tracesg = new float[raw->frames];
  int x, y, frame;
  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      if (!mask || ( (*mask) [x+raw->cols*y] & these))
      {
        for (frame=0;frame<raw->frames;frame++)
        {
          trace[frame] = raw->image[frame*raw->frameStride + x + y*raw->cols];
        }
        sgFilter->FilterDataImpl (trace, tracesg, raw->frames, 0);
        for (frame=0;frame<raw->frames;frame++)
        {
          raw->image[frame*raw->frameStride + x + y*raw->cols] = (short) (tracesg[frame] + 0.5);
        }
      }
    }
  }
  delete [] trace;
  delete [] tracesg;
}

void Image::SGFilterApply (short *source, short *target)
{
  float *trace = new float[raw->frames];
  float *tracesg = new float[raw->frames];
  int frame;
  for (frame=0;frame<raw->frames;frame++)
    trace[frame] = source[frame];
  sgFilter->FilterDataImpl (trace, tracesg, raw->frames, 0);
  for (frame=0;frame<raw->frames;frame++)
    target[frame] = (short) (tracesg[frame] + 0.5);
  delete [] trace;
  delete [] tracesg;
}

void Image::SGFilterApply (float *source, float *target)
{
  sgFilter->FilterDataImpl (source, target, raw->frames, 0);
}

// Metric 1 is, for each trace, the max - abs(min)
// Arguments are only used for generating the metrics histogram which is only
// used for reporting purposes.  Note that the actual clustering bead finding
// happens in the Separator - find beads.
void Image::CalcBeadfindMetric_1 (Mask *mask, Region region, char *idStr, int frameStart, int frameEnd)
{
  printf ("gathering metric 1...\n");
  if (!results)
  {
    results = new double[raw->rows * raw->cols];
    memset (results, 0, raw->rows * raw->cols * sizeof (double));
    //fprintf (stdout, "Image::CalcBeadfindMetric_1 allocated: %lu\n",raw->rows * raw->cols * sizeof(double) );
  }

  if (frameStart == -1)
  {
    frameStart = GetFrame (12); // Frame 15;
  }
  if (frameEnd == -1)
  {
    frameEnd = GetFrame (2374); // Frame 50;
  }
  fprintf (stdout, "Image: CalcBeadFindMetric_1 %d %d\n", frameStart, frameEnd);
  int frame, x, y;
  int k;
  int min, max;
  //int printCnt = 0;
  for (y=0;y<raw->rows;y++)
  {
    for (x=0;x<raw->cols;x++)
    {
      k = x + y*raw->cols;
      min = 65536;
      max = -65536;
      int minK = 0;
      int maxK = 0;
      // for(frame=0;frame<raw->frames;frame++) {
      k += frameStart*raw->frameStride;
      for (frame=frameStart;frame<frameEnd;frame++)
      {
        if (frame == frameStart || (raw->image[k] > max))
        {
          max = raw->image[k];
          maxK = frame;
        }
        if (frame == frameStart || (raw->image[k] < min))
        {
          min = raw->image[k];
          minK = frame;
        }
        k += raw->frameStride;
      }
      results[x+y*raw->cols] = max - abs (min);
#if 0
      if (results[x+y*raw->cols] > -1 && results[x+y*raw->cols] < 1)
      {
        if ( (*mask) [x+y*raw->cols] & (MaskPinned | MaskIgnore | MaskWashout))
        {
          continue;
        }
        int j = x + y*raw->cols;
        j += frameStart*raw->frameStride;
        for (int l=frameStart;l<frameEnd;l++)
        {
          fprintf (stdout, "%d ", raw->image[j]);
          j += raw->frameStride;
        }
        fprintf (stdout, "\n");
        printCnt++;
      }
#endif
    }
  }
}


void Image::Cleanup()
{
  if (results)
    delete [] results;
  results = NULL;
}

void Image::SetImage (RawImage *img)
{
  cleanupRaw();
  delete raw;
  raw = img;
}

void Image::DebugDumpResults (char *fileName, Region region)
{
  if (!results)
    return;

  int x,y,inx;
  FILE *fp = NULL;
  fopen_s (&fp, fileName, "w");
  if (!fp)
  {
    fprintf (fp, "%s: %s\n", fileName, strerror (errno));
    return;
  }
  for (y = region.row; y < (region.row+region.h); y++)
    for (x = region.col; x < (region.col+region.w); x++)
    {
      inx = x + (y * raw->cols);
      fprintf (fp, "%d\n", (int) results[inx]);
    }
  fclose (fp);
  return;
}

void Image::DebugDumpResults (char *fileName)
{
  if (!results)
    return;

  int x,y,inx;
  FILE *fp = NULL;
  fopen_s (&fp, fileName, "w");
  if (!fp)
  {
    fprintf (fp, "%s: %s\n", fileName, strerror (errno));
    return;
  }
  for (y = 0; y < raw->rows; y++)
    for (x = 0; x < raw->cols; x++)
    {
      inx = x + (y * raw->cols);
      fprintf (fp, "%d\n", (int) results[inx]);
    }
  fclose (fp);
  return;
}

//
void Image::DumpTrace (int r, int c, char *fileName)
{
  FILE *fp = NULL;
  fopen_s (&fp, fileName, "w");
  if (!fp)
  {
    printf ("%s: %s\n", fileName, strerror (errno));
    return;
  }
  int frameStart = 0;
  int frameEnd = raw->frames;
  int k = (frameStart * raw->frameStride) + (c + (raw->cols * r));

  for (int frame=frameStart;frame<frameEnd;frame++)
  {
    //Prints a column
    fprintf (fp, "%d\n", raw->image[k]);
    //Prints comma delimited row
    //fprintf (fp, "%d", raw->image[k]);
    //if (frame < (frameEnd - 1))
    //  fprintf (fp, ",");
    k += raw->frameStride;
  }
  fprintf (fp, "\n");
  fclose (fp);
}

int Image::DumpDcOffset (int nSample, string dcOffsetDir, char nucChar, int flowNumber) {

  // DC-offset using time 0 through noFlowTime milliseconds
  int dcStartFrame = GetFrame(   0 - GetFlowOffset());
  int dcEndFrame   = GetFrame(GetNoFlowTime() - GetFlowOffset());
  dcStartFrame = std::min(raw->frames-1,std::max(dcStartFrame,0));
  dcEndFrame   = std::min(raw->frames-1,std::max(dcEndFrame,  0));
  dcEndFrame   = std::max(dcStartFrame,dcEndFrame);
  float nDcFrame = dcEndFrame-dcStartFrame+1;
  
  // Init random seed
  int random_seed=0;
  srand(random_seed);

  // Set sample size
  int nWells = raw->cols * raw->rows;
  nSample = std::min(nSample,nWells);

  // Get the random sample of wells and compute dc offsets
  vector<int> dcOffset(nSample);
  for(int iSample=0; iSample<nSample; iSample++) {
    int maskIndex = rand() % nWells;
    int k = raw->frameStride*dcStartFrame + maskIndex;
    float sum = 0;
    for (int frame = dcStartFrame; frame <= dcEndFrame; frame++, k += raw->frameStride)
      sum += raw->image[k];
    sum /= nDcFrame;
    dcOffset[iSample] = (int) sum;
  }

  // Sort results so we can return percentiles
  sort(dcOffset.begin(),dcOffset.end());

  // Open results file for append
  string dcOffsetFileName = dcOffsetDir + string ("/dcOffset.txt");
  FILE *dcOffsetFP  = NULL;
  fopen_s (&dcOffsetFP, dcOffsetFileName.c_str(), "a");
  if (!dcOffsetFP)
  {
    printf ("Could not open/append to %s, err %s\n", dcOffsetFileName.c_str(), strerror (errno));
    return EXIT_FAILURE;
  }

  // Write percentiles and close file
  fprintf (dcOffsetFP, "%d\t%c", flowNumber, nucChar);
  fprintf (dcOffsetFP, "\t%d\t%d", dcOffset.front(),dcOffset.back()); // min and max
  float nQuantiles=100;
  float jump = dcOffset.size() / nQuantiles;
  for(int i=1; i<nQuantiles; i++)
    fprintf (dcOffsetFP, "\t%d", dcOffset[floor(i*jump)]);
  fprintf (dcOffsetFP, "\n");
  fclose (dcOffsetFP);

  return(EXIT_SUCCESS);
}

double Image::DumpStep (int c, int r, int w, int h, string regionName, char nucChar, string nucStepDir, Mask *mask, PinnedInFlow *pinnedInFlow, int flowNumber)
{
  // make sure user calls us with sane bounds
  if (w < 1) w = 1;
  if (h < 1) h = 1;
  if (r < 0) r = 0;
  if (c < 0) c = 0;
  if (r >= raw->rows) r = raw->rows-1;
  if (c >= raw->cols) c = raw->cols-1;
  if (r+h > raw->rows) h = raw->rows - r;
  if (c+w > raw->cols) w = raw->cols - c;
  if (h < 1 || w < 1) // ok, nothing left to compute?
    return 0.0;

  // DC-offset using time 0 through noFlowTime milliseconds
  int dcStartFrame = GetFrame(   0 - GetFlowOffset());
  int dcEndFrame   = GetFrame(GetNoFlowTime() - GetFlowOffset());
  dcStartFrame = std::min(raw->frames-1,std::max(dcStartFrame,0));
  dcEndFrame   = std::min(raw->frames-1,std::max(dcEndFrame,  0));
  dcEndFrame   = std::max(dcStartFrame,dcEndFrame);
  
  string baseName = nucStepDir + string ("/NucStep_") + regionName;
  string nucStepSizeFileName    = baseName + string ("_step.txt");
  string nucStepBeadFileName    = baseName + string ("_bead.txt");
  string nucStepEmptyFileName   = baseName + string ("_empty.txt");
  string nucStepEmptySdFileName = baseName + string ("_empty_sd.txt");

  string frameTimeFileName = nucStepDir + string ("/NucStep_frametime.txt");
  FILE *fpFrameTime  = NULL;
  fopen_s (&fpFrameTime, frameTimeFileName.c_str(), "w");
  if (!fpFrameTime)
  {
    printf ("Could not open to %s, err %s\n", frameTimeFileName.c_str(), strerror (errno));
    return 0.0;
  }
  if (fpFrameTime)
  {
    fprintf (fpFrameTime, "0");
    for (int iFrame=1; iFrame<raw->frames; iFrame++)
    {
      fprintf (fpFrameTime, "\t%.3f", raw->timestamps[iFrame-1] / FRAME_SCALE);
    }
    fprintf (fpFrameTime, "\n");
    fclose (fpFrameTime);
  }

  FILE *fpSize  = NULL;
  fopen_s (&fpSize, nucStepSizeFileName.c_str(), "a");
  if (!fpSize)
  {
    printf ("Could not open/append to %s, err %s\n", nucStepSizeFileName.c_str(), strerror (errno));
    return 0.0;
  }

  FILE *fpBead  = NULL;
  fopen_s (&fpBead, nucStepBeadFileName.c_str(), "a");
  if (!fpBead)
  {
    printf ("Could not open/append to %s, err %s\n", nucStepBeadFileName.c_str(), strerror (errno));
    return 0.0;
  }

  FILE *fpEmpty = NULL;
  fopen_s (&fpEmpty, nucStepEmptyFileName.c_str(), "a");
  if (!fpEmpty)
  {
    printf ("Could not open/append to %s, err %s\n", nucStepEmptyFileName.c_str(), strerror (errno));
    return 0.0;
  }

  FILE *fpEmptySd = NULL;
  fopen_s (&fpEmptySd, nucStepEmptySdFileName.c_str(), "a");
  if (!fpEmptySd)
  {
    printf ("Could not open/append to %s, err %s\n", nucStepEmptySdFileName.c_str(), strerror (errno));
    return 0.0;
  }

  vector<float> frameWeight(raw->frames,0);
  float weightSum=0;
  for (int frame = dcStartFrame; frame <= dcEndFrame; frame++) {
    float thisWeight = raw->timestamps[frame] - ( (frame > 0) ? raw->timestamps[frame-1] : 0 );
    frameWeight[frame] = thisWeight;
    weightSum += thisWeight;
  }
  for (int frame = dcStartFrame; frame <= dcEndFrame; frame++) {
    frameWeight[frame] /= weightSum;
  }

  unsigned int nBead=0;
  unsigned int nEmpty=0;
  vector<float> valBead (raw->frames,0);
  vector<float> valEmpty (raw->frames,0);
  vector<float> valEmptySd (raw->frames,0);
  for (int y=r;y<(r+h);y++)
  {
    int maskIndex = raw->cols * y + c;
    for (int x=c;x< (c+w);x++, maskIndex++)
    {
      bool unPinned = ! (pinnedInFlow->IsPinned(flowNumber, maskIndex));
      bool notBad   = unPinned && (!mask->Match(maskIndex, (MaskType) (MaskExclude | MaskIgnore)));
      bool isBead   = notBad && mask->Match(maskIndex, (MaskType) MaskBead);
      bool isEmpty  = notBad && mask->Match(maskIndex, (MaskType) MaskEmpty);
      if(isBead || isEmpty) {
        // First subtract dc offsets from the bead
        int k = raw->frameStride*dcStartFrame + maskIndex;
        float sum = 0;
        for (int frame = dcStartFrame; frame <= dcEndFrame; frame++, k += raw->frameStride)
          sum += raw->image[k] * frameWeight[frame];

        if(isBead)
          nBead++;
        else
          nEmpty++;
        k = maskIndex;
        for (int frame = 0; frame < raw->frames; frame++, k += raw->frameStride)
        {
          float thisVal = (float) raw->image[k] - sum;
          if (isBead)
            valBead[frame] += thisVal;
          else {
            valEmpty[frame] += thisVal;
            valEmptySd[frame] += thisVal*thisVal;
          }
        }
      }
    }
  }

  if (nBead > 0)
  {
    for (int frame=0; frame < raw->frames; frame++)
    {
      valBead[frame] /= (float) nBead;
    }
  }
  if (nEmpty > 0)
  {
    for (int frame=0; frame < raw->frames; frame++)
    {
      valEmpty[frame] /= (float) nEmpty;
    }
    if (nEmpty > 1) {
      for (int frame=0; frame < raw->frames; frame++)
      {
        valEmptySd[frame] /= (float) nEmpty;
        valEmptySd[frame] -= valEmpty[frame]*valEmpty[frame];
        valEmptySd[frame] *= ((float) nEmpty) / ((float) (nEmpty-1));
        valEmptySd[frame] = sqrt(valEmptySd[frame]);
      }
    } else {
      for (int frame=0; frame < raw->frames; frame++)
      {
        valEmpty[frame] = 0;
      }
    }
  }

  // Compute step size in empty wells
  double minVal = 0.0;
  double maxVal = 0.0;
  if (nEmpty > 0)
  {
    minVal = maxVal = valEmpty[0];
    for (int frame=1; frame < raw->frames; frame++)
    {
      double thisVal = valEmpty[frame];
      if (thisVal < minVal)
      {
        minVal = thisVal;
      }
      if (thisVal > maxVal)
      {
        maxVal = thisVal;
      }
    }
  }
  double stepSize = maxVal-minVal;

  if (fpSize)
  {
    fprintf (fpSize, "%d\t%c\t%.2lf\n", flowNumber, nucChar, stepSize);
    fclose (fpSize);
  }
  if (fpBead)
  {
    fprintf (fpBead, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nBead);
    for (int frame = 0; frame < raw->frames; frame++)
    {
      fprintf (fpBead, "\t%.3f", valBead[frame]);
    }
    fprintf (fpBead, "\n");
    fclose (fpBead);
  }
  if (fpEmpty)
  {
    fprintf (fpEmpty, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nEmpty);
    for (int frame = 0; frame < raw->frames; frame++)
    {
      fprintf (fpEmpty, "\t%.3f", valEmpty[frame]);
    }
    fprintf (fpEmpty, "\n");
    fclose (fpEmpty);
  }
  if (fpEmptySd)
  {
    fprintf (fpEmptySd, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nEmpty);
    for (int frame = 0; frame < raw->frames; frame++)
    {
      fprintf (fpEmptySd, "\t%.3f", valEmptySd[frame]);
    }
    fprintf (fpEmptySd, "\n");
    fclose (fpEmptySd);
  }

  return stepSize;
}

//
//  input time is in milliseconds
//  returns frame number corresponding to that time
int Image::GetFrame (int time)
{
  int frame = 0;
  int prev=0;
  //flowOffset is time between image start and nuke flow.
  //all times provided are relative to the nuke flow.
  time = time + flowOffset;
  for (frame=0;frame < raw->frames;frame++)
  {
    if (raw->timestamps[frame] >= time)
      break;
    prev = frame;
  }

#if 0
  frame = (int) rint ( ( (float) time / 1000) * fps);
  //fprintf (stdout, "GetFrame: fps = %f\n", img->GetFPS());
  if (frame < 0)
  {
    fprintf (stdout, "Calculated negative frame!  Setting to 0\n");
    frame = 0;
  }
  return (frame);
#endif
  return (prev);
}

// Special:  we usually get >all< the values for a given trace and send them to the bkgmodel.
// more efficient to keep variables around than repeatedly calling GetInterpolatedVal
void Image::GetUncompressedTrace (float *val, int last_frame, int x, int y)
{
  // set up:  index by well, make sure last_frame is within range
  int l_coord = y*raw->cols+x;

  if (last_frame>raw->uncompFrames)
    last_frame = raw->uncompFrames;
// if compressed
  if (raw->uncompFrames != raw->frames)
  {
    int my_frame = 0;
    val[my_frame] = raw->image[l_coord];

    float prev=raw->image[l_coord];
    float next=0.0f;

    for (my_frame=1; my_frame<last_frame; my_frame++)
    {
      // need to make this faster!!!
      int interf= raw->interpolatedFrames[my_frame];

      int f_coord = l_coord+raw->frameStride*interf;
      next = raw->image[f_coord];
      prev = raw->image[f_coord-raw->frameStride];

      // interpolate
      float mult = raw->interpolatedMult[my_frame];
      val[my_frame] = (prev-next) *mult + next;
    }
  }
  else
  {
    // the rare "uncompressed" case
    for (int my_frame=0; my_frame<last_frame; my_frame++)
    {
      val[my_frame] = raw->image[l_coord+my_frame*raw->frameStride];
    }
  }
}



float Image::GetInterpolatedValue (int frame, int x, int y)
{
  float rc;
  if (frame < 0)
  {
    printf ("asked for negative frame!!!  %d\n",frame);
    return 0.0f;
  }
  if (raw->uncompFrames == raw->frames)
  {
    rc = raw->image[raw->frameStride*frame+y*raw->cols+x];
  }
  else
  {
    if (frame==0)
    {
      rc = raw->image[y*raw->cols+x];
    }
    else
    {
      if (frame >= raw->uncompFrames)
        frame = raw->uncompFrames-1;

      // need to make this faster!!!
      int interf=raw->interpolatedFrames[frame];
      float mult = raw->interpolatedMult[frame];

      float prev=0.0f;
      float next=0.0f;

      next = raw->image[raw->frameStride*interf+y*raw->cols+x];
      if (interf)
        prev = raw->image[raw->frameStride* (interf-1) +y*raw->cols+x];

      // interpolate
      rc = (prev-next) *mult + next;
    }
  }

  return rc;
}


void Image::GetInterpolatedValueAvg4 (int16_t *ptr, int frame, int x, int y, int num)
{
  int i;
//  float sum=0;
#if 0
  for (i=0;i<num;i++,frame++)
  {
    sum += GetInterpolatedValue (frame,x,y);
  }
  return sum/num;
#else
  short rc;
  int interf,idx,partial;
//  int oframe=0;
//  int fournum=0;
  Vec4 multV;
  Vec4 prevV;
  Vec4 nextV;
  Vec4 sumV={{0,0,0,0}};
  Vec4 zeroV={{0,0,0,0}};
  Vec4 oneV={{1,1,1,1}};
  Vec4 divV;
  static int doneOnce=0/*,doneTwice=0*/;

  if (frame < 0)
  {
    printf ("asked for negative frame!!!  %d\n",frame);
    return;
  }
  if (raw->uncompFrames == raw->frames)
  {
    rc = raw->image[raw->frameStride*frame+y*raw->cols+x];
  }
  else
  {
    frame++;

    if (frame >= raw->uncompFrames)
      frame = raw->uncompFrames-1;
    if ( (num + frame) >= raw->uncompFrames)
      num = raw->uncompFrames - frame;

//    fournum = num & ~0x3;
//    oframe = frame;
    divV.e[0] = divV.e[1] = divV.e[2] = divV.e[3] = num;

    partial = y*raw->cols+x;

    for (i=0;i<num;i++,frame++)
    {
      prevV.v = zeroV.v;

      // need to make this faster!!!
      interf=raw->interpolatedFrames[frame];
      idx = raw->frameStride*interf+partial;

      multV.e[0] = multV.e[1] = multV.e[2] = multV.e[3] = raw->interpolatedMult[frame];
      nextV.e[0] = raw->image[idx+0];
      nextV.e[1] = raw->image[idx+1];
      nextV.e[2] = raw->image[idx+2];
      nextV.e[3] = raw->image[idx+3];
      if (interf)
      {
        prevV.e[0] = raw->image[idx+0-raw->frameStride];
        prevV.e[1] = raw->image[idx+1-raw->frameStride];
        prevV.e[2] = raw->image[idx+2-raw->frameStride];
        prevV.e[3] = raw->image[idx+3-raw->frameStride];
      }
      // interpolate
      sumV.v += prevV.v*multV.v + nextV.v* (oneV.v-multV.v);
      if (!doneOnce)
      {
        doneOnce=1;
        printf ("sumV.v = %f %f %f %f\n",sumV.e[0],sumV.e[1],sumV.e[2],sumV.e[3]);
      }

    }

    sumV.v /= divV.v;
    ptr[0] = (int16_t) sumV.e[0];
    ptr[1] = (int16_t) sumV.e[1];
    ptr[2] = (int16_t) sumV.e[2];
    ptr[3] = (int16_t) sumV.e[3];
//    sum = sumV.e[0] + sumV.e[1] + sumV.e[2] + sumV.e[3];


#if 0
    {
      float sum2=0;
      for (i=0;i<num;i++)
      {
        sum2 += GetInterpolatedValue (oframe+i,x,y);
        if (num >= 4 && !doneTwice)
        {
          doneTwice=1;
          printf ("real(%d) = %f %f %f %f\n",num,GetInterpolatedValue (oframe+0,x,y),
                  GetInterpolatedValue (oframe+1,x,y),
                  GetInterpolatedValue (oframe+2,x,y),
                  GetInterpolatedValue (oframe+3,x,y));
        }
      }
      if (sum2 != sum)
      {
        printf ("both don't match %f %f\n",sum,sum2);
        exit (0);
      }
    }
#endif
  }
#endif
}


//
//  For any given image file, return true if the image file can be loaded for processing.
//
//  Algorithm is:
//    If explog_final.txt exists, index file can be loaded
//    If beadfind_post_0000 exists, index file can be loaded
//    for a given file's index, if the index+1 file exists, then the index file can be loaded
//
bool Image::ReadyToLoad (const char *filename)
{

  char thisFileName[PATH_MAX] = {'\0'};
  char nextFileName[PATH_MAX] = {'\0'};
  char thisPath[PATH_MAX] = {'\0'};
  char *path = strdup (filename);
  strcpy (thisPath, dirname (path));
  free (path);
  // This method is only for acq image files
  strcpy (thisFileName, filename);
  if (strncmp (basename (thisFileName), "acq", 3) != 0)
  {
    return true;
  }

  // If explog_final.txt exists, the run is done and all files should load
  // Block datasets will find explog_final.txt in parent directory
  sprintf (nextFileName, "%s/explog_final.txt", thisPath);
  //fprintf (stdout, "Looking for %s\n", nextFileName);
  if (isFile (nextFileName))
  {
    return true;
  }
  /*
  // Block datasets will find explog_final.txt in parent directory
  char *parent = NULL;
  parent = strdup (thisPath);
  char *parent2 = dirname (parent);
  sprintf (nextFileName, "%s/explog_final.txt", parent2);
  //fprintf (stdout, "And now Looking for %s\n", nextFileName);
  free (parent);
  if (isFile (nextFileName))
  {
    return true;
  }
  */
  // If beadfind_post_0000.txt exists, the run is done and all files should load
  sprintf (nextFileName, "%s/beadfind_post_0000.dat", thisPath);
  if (isFile (nextFileName))
  {
    return true;
  }

  // If subsequent image file exists, this image file should load
  //--- Get the index of this file
  int idxThisFile = -1;
  strncpy (thisFileName, filename, strlen (filename));
  sscanf (basename (thisFileName), "acq_%d.dat", &idxThisFile);
  assert (idxThisFile >= 0);
  sprintf (nextFileName, "%s/acq_%04d.dat", thisPath, idxThisFile + 1);
  if (isFile (nextFileName))
  {
    return true;
  }

  return false;

}

int Image::cropped_region_offset_x = 0;
int Image::cropped_region_offset_y = 0;

// XTChannelCorrect:
// For the 316 and 318, corrects cross-talk due to incomplete analog settling within the 316 and 318, and also
// residual uncorrected incomplete setting at the output of the devices.
// works along each row of the image and corrects cross talk that occurs
// within a single acquisition channel (every fourth pixel is the same channel on the 316/318)
// This method has no effect on the 314.
// The following NOTE is out-of-date, something like this may come back sometime
// NOTE:  A side-effect of this method is that the data for all pinned pixels will be replaced with
// the average of the surrounding neighbor pixels.  This helps limit the spread of invalid data in the pinned
// pixels to neighboring wells
// void Image::XTChannelCorrect (Mask *mask)
void Image::XTChannelCorrect ()
{
  short tmp[raw->cols];

  float **vects = NULL;
  int nvects = 0;
  int *col_offset = NULL;
  float *vect;
  int vector_len;

  // If no correction has been configured for (by a call to CalibrateChannelXTCorrection), the try to find the default
  // correction using the chip id as a guide.
  if (selected_chip_xt_vectors.xt_vector_ptrs == NULL)
    for (int nchip = 0;default_chip_xt_vect_array[nchip].id != ChipIdUnknown;nchip++)
      if (default_chip_xt_vect_array[nchip].id == ChipIdDecoder::GetGlobalChipId())
      {
        memcpy (&selected_chip_xt_vectors,& (default_chip_xt_vect_array[nchip].descr),sizeof (selected_chip_xt_vectors));
        break;
      }

  // if the chip type is unsupported, silently return and do nothing
  if (selected_chip_xt_vectors.xt_vector_ptrs == NULL)
    return;

  vects = selected_chip_xt_vectors.xt_vector_ptrs;
  nvects = selected_chip_xt_vectors.num_vectors;
  col_offset = selected_chip_xt_vectors.vector_indicies;
  vector_len = selected_chip_xt_vectors.vector_len;

  // fill in pinned pixels with average of surrounding valid wells
  //BackgroundCorrect(mask, MaskPinned, (MaskType)(MaskAll & ~MaskPinned & ~MaskExclude),0,5,NULL,false,false,true);

  for (int frame = 0;frame < raw->frames;frame++)
  {
    short *pfrm = & (raw->image[frame*raw->frameStride]);
    for (int row = 0;row < raw->rows;row++)
    {
      short *prow = pfrm + row*raw->cols;
      for (int col = 0;col < raw->cols;col++)
      {
        int vndx = ( (col+cropped_region_offset_x) % nvects);
        vect = vects[vndx];

        float sum = 0.0;
        for (int vn = 0;vn < vector_len;vn++)
        {
          int ndx = col + col_offset[vn];
          if ( (ndx >= 0) && (ndx < raw->cols))
            sum += prow[ndx]*vect[vn];
        }
        tmp[col] = (short) (sum);
      }
      // copy result back into the image
      memcpy (prow,tmp,sizeof (short[raw->cols]));
    }
  }

  //Dump XT vectors to file
  if (dump_XTvects_to_file)
  {
    char xtfname[512];
    sprintf (xtfname,"%s/cross_talk_vectors.txt", experimentName);
    FILE* xtfile = fopen (xtfname, "wt");

    if (xtfile !=NULL)
    {
      //write vector length and number of vectors on top
      fprintf (xtfile, "%d\t%d\n", vector_len, nvects);
      //write offsets in single line
      for (int nl=0; nl<vector_len; nl++)
        fprintf (xtfile, "%d\t", col_offset[nl]);
      fprintf (xtfile, "\n");
      //write vectors tab-separated one line per vector
      for (int vndx=0; vndx < nvects; vndx++)
      {
        vect = vects[vndx];
        for (int vn=0; vn < vector_len; vn++)
          fprintf (xtfile, "%4.6f\t", vect[vn]);
        fprintf (xtfile, "\n");
      }
      fclose (xtfile);
    }
    dump_XTvects_to_file = 0;
  }
}

ChannelXTCorrection *Image::custom_correction_data = NULL;

// checks to see if the special lsrowimage.dat file exists in the experiment directory.  If it does,
// this image is used to generate custom channel correction coefficients.  If not, the method silently
// returns (and subsequent analysis uses the default correction).
void Image::CalibrateChannelXTCorrection (const char *exp_dir,const char *filename, bool wait_for_prerun)
{
  // only allow this to be done once
  if (custom_correction_data != NULL)
    return;

  // LSRowImageProcessor can generate a correction for the 314, but application of the correction is much more
  // difficult than for 316/318, and the expected benefit is not as high, so for now...we're skipping the 314
  if ( (ChipIdDecoder::GetGlobalChipId() != ChipId316) && (ChipIdDecoder::GetGlobalChipId() != ChipId318))
    return;

  int len = strlen (exp_dir) +strlen (filename) + 2;
  char full_fname[len];

  sprintf (full_fname,"%s/%s",exp_dir,filename);

  if (wait_for_prerun)
  {
    std::string preRun = exp_dir;
    preRun = preRun + "/prerun_0000.dat";
    std::string acq0 = exp_dir;
    acq0 = acq0 + "/acq_0000.dat";

    uint32_t waitTime = RETRY_INTERVAL;
    int32_t timeOut = TOTAL_TIMEOUT;
    //--- Wait up to 3600 seconds for a file to be available
    bool okToProceed = false;
    while (timeOut > 0)
    {
      //--- do our checkpoint files exist?
      if (isFile (preRun.c_str()) || isFile (acq0.c_str()))
      {
        okToProceed = true;
        break;
      }
      fprintf (stdout, "Waiting to load crosstalk params in %s\n",  full_fname);
      sleep (waitTime);
      timeOut -= waitTime;
    }
    if (!okToProceed)
    {
      ION_ABORT ("Couldn't find gateway files for: " + ToStr (full_fname));
    }
    // We got the files we expected so if the xtalk file isn't there then warn.
    if (!isFile (full_fname))
    {
      ION_WARN ("Didn't find xtalk file: " + ToStr (full_fname));
    }
  }
  LSRowImageProcessor lsrowproc;
  custom_correction_data = lsrowproc.GenerateCorrection (full_fname);
  if (custom_correction_data != NULL)
    selected_chip_xt_vectors = custom_correction_data->GetCorrectionDescriptor();

}

// determines offset from Chip origin of this image data.  Only needed for block
// datasets.
void Image::SetOffsetFromChipOrigin (const char *filepath)
{
  // Block datasets are stored in subdirectories named for the x and y coordinates
  // of the origin of the block data, i.e. "X256_Y1024"
  // extract the subdirectory and parse the coordinates from that
  char *path = NULL;
  path = strdup (filepath);

  char *dir = NULL;
  dir = dirname (path);

  char *coords = NULL;
  coords = basename (dir);

  int val = -1;
  int chip_offset_x = 0;
  int chip_offset_y = 0;
  val = sscanf (coords, "X%d_Y%d", &chip_offset_x, &chip_offset_y);
  if (val != 2)
  {
    // ERROR
    raw->chip_offset_x = -1;
    raw->chip_offset_y = -1;
    fprintf (stdout, "Could not determine this image's chip origin offset from directory name\n");
  }
  else
  {
    // SUCCESS
    raw->chip_offset_x = chip_offset_x;
    raw->chip_offset_y = chip_offset_y;
    fprintf (stdout, "Determined this image's chip origin offset:\n X: %d Y: %d\n", chip_offset_x,chip_offset_y);
  }
}






