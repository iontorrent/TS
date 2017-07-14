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
#include <algorithm>
#include "MathOptim.h"

#include "IonErr.h"
#include "Image.h"
#include "SampleStats.h"

#include "Utils.h"
#include "deInterlace.h"
#include "LinuxCompat.h"
#include "ChipIdDecoder.h"

#include "ImageTransformer.h"

#include "IonErr.h"

#include "PinnedInFlow.h"
#include "IonImageSem.h"
#include "PCACompression.h"

using namespace std;


// default constructor
Image::Image()
{
  results = NULL;
  raw = new RawImage;
  memset ( raw,0,sizeof ( *raw ) );
  maxFrames = 0;

  // set up the default SG-Filter class
  // MGD note - may want to ensure this becomes thread-safe or create one per region/thread
//  sgSpread = 2;
//  sgCoeff = 1;
//  sgFilter = new SGFilter();
//  sgFilter->SetFilterParameters (sgSpread, sgCoeff);
  results_folder = NULL;
  // revise when we have command line values
  acqPrefix = strdup("acq_");
  datPostfix = strdup("dat");

  flowOffset = 1000;
  noFlowTime = 1350;
  bkg = NULL;

  smooth_max_amplitude = NULL;

  retry_interval = 15;  // 15 seconds wait time.
  total_timeout = 36000;  // 10 hours before giving up.
  numAcqFiles = 0;    // total number of acq files in the dataset
  recklessAbandon = true; // false: use file availability testing during loading
  ignoreChecksumErrors = 0;

  CacheAccessTime=0;
  SemaphoreWaitTime=0;
  FileLoadTime=0;
}

Image::~Image()
{
  cleanupRaw();
  delete raw;
//  delete sgFilter;
  if ( results )
    delete [] results;
  if ( results_folder )
  {
      try
      {
        free ( results_folder );
      }
      catch (...)
      {
        std::cerr << "~Image error: free results_folder " << results_folder << std::endl << std::flush;
        exit(1);
      }
  }
  if ( bkg )
    free ( bkg );

  if ( smooth_max_amplitude )
    delete [] smooth_max_amplitude;
}

void Image::Close()
{
  cleanupRaw();

  if ( results )
    delete [] results;
  results = NULL;

  if (acqPrefix!=NULL){
    free(acqPrefix);
    acqPrefix = NULL;
  }
  if (datPostfix!=NULL){
    free(datPostfix);
    datPostfix = NULL;
  }
  if ( bkg )
    free ( bkg );
  bkg = NULL;

  maxFrames = 0;
}

void Image::cleanupRaw()
{
  if ( raw->image )
  {
    free ( raw->image );
    raw->image = NULL;
  }
  if ( raw->timestamps )
  {
    free ( raw->timestamps );
    raw->timestamps = NULL;
  }
  if ( raw->interpolatedFrames )
  {
    free ( raw->interpolatedFrames );
    raw->interpolatedFrames = NULL;
  }
  if ( raw->interpolatedMult )
  {
    free ( raw->interpolatedMult );
    raw->interpolatedMult = NULL;
  }
  if ( raw->interpolatedDiv )
  {
    free ( raw->interpolatedDiv );
    raw->interpolatedDiv = NULL;
  }
  if ( raw->compToUncompFrames ) 
    { 
      free ( raw->compToUncompFrames );
      raw->compToUncompFrames = NULL;
    }
  memset ( raw,0,sizeof ( *raw ) );
}


void Image::SetDir ( const char *directory, const char *_acqPrefix, const char *_datPostfix )
{
  if ( results_folder )
    free ( results_folder );
  results_folder = ( char * ) malloc ( strlen ( directory ) + 1 );
  strncpy ( results_folder, directory, strlen ( directory ) + 1 );

  if (acqPrefix!=NULL)
    free(acqPrefix);
  acqPrefix = strdup(_acqPrefix);
  if (datPostfix!=NULL)
    free(datPostfix);
  datPostfix = strdup(_datPostfix);

  return;
}
//
//  For any given image file, return true if the image file can be loaded for processing.
//
//  Algorithm is:
//    If explog_final.txt exists, index file can be loaded
//    If beadfind_post_0000 exists, index file can be loaded
//    for a given file's index, if the index+1 file exists, then the index file can be loaded
//
bool Image::ReadyToLoad ( const char *filename , const char *_acqPrefix, const char * _datPostfix)
{

  char thisFileName[PATH_MAX] = {'\0'};
  char nextFileName[PATH_MAX] = {'\0'};
  char thisPath[PATH_MAX] = {'\0'};
  char *path = strdup ( filename );
  strcpy ( thisPath, dirname ( path ) );
  free ( path );
  // This method is only for acq image files
  strcpy ( thisFileName, filename );
  // this should probably be acqPrefix length
  if ( strncmp ( basename ( thisFileName ), _acqPrefix, 3 ) != 0 )
  {
    return true;
  }

  // If explog_final.txt exists, the run is done and all files should load
  // Block datasets will find explog_final.txt in parent directory
  sprintf ( nextFileName, "%s/explog_final.txt", thisPath );
  //fprintf (stdout, "Looking for %s\n", nextFileName);
  if ( isFile ( nextFileName ) )
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
  sprintf ( nextFileName, "%s/beadfind_post_0000.%s", thisPath, _datPostfix );
  if ( isFile ( nextFileName ) )
  {
    return true;
  }

  // If subsequent image file exists, this image file should load
  //--- Get the index of this file
  int idxThisFile = -1;
  strncpy ( thisFileName, filename, strlen ( filename ) );
  char format_dat[1024];
  // note %%d writes %d to the format string, which is waht we want
  sprintf(format_dat, "%s%%d.%s", _acqPrefix, _datPostfix);
  sscanf ( basename ( thisFileName ), format_dat, &idxThisFile );
  assert ( idxThisFile >= 0 );
  sprintf ( nextFileName, "%s/%s%04d.%s", thisPath, _acqPrefix, idxThisFile + 1, _datPostfix );
  if ( isFile ( nextFileName ) )
  {
    return true;
  }

  return false;

}

// Block datasets are stored in subdirectories named for the x and y coordinates
// of the origin of the block data, i.e. "X256_Y1024"
// extract the subdirectory and parse the coordinates from that
bool Image::GetOffsetFromChipPath (const char * filepath, int &x_offset, int &y_offset) {
  char *path = NULL;
  path = strdup ( filepath );

  char *dir = NULL;
  dir = dirname ( path );

  char *coords = NULL;
  coords = basename ( dir );

  int val = -1;
  x_offset = 0;
  y_offset = 0;
  bool success = false;
  val = sscanf ( coords, "X%d_Y%d", &x_offset, &y_offset );
  if (val == 2) {
    success = true;
  }
  if (path)
     free (path);
  return success;
}

// determines offset from Chip origin of this image data.  Only needed for block
// datasets.
void Image::SetOffsetFromChipOrigin ( const char *filepath )
{
  int chip_offset_x, chip_offset_y;
  bool success = GetOffsetFromChipPath(filepath, chip_offset_x, chip_offset_y);
  if (success) {  
    fprintf ( stdout, "Determined this image's chip origin offset:\n X: %d Y: %d\n", chip_offset_x,chip_offset_y ); 
  }
  else {
    chip_offset_x = chip_offset_y = -1;
    fprintf ( stdout, "Could not determine this image's chip origin offset from directory name\n" );
  }
  raw->chip_offset_x = chip_offset_x;
  raw->chip_offset_y = chip_offset_y;
}


//
// LoadRaw
// loads raw image data for one experiment, and byte-swaps as appropriate
// returns a structure with header data and a pointer to the allocated image data with timesteps removed
// A couple prototypes are defined so all the old applications work with the old argument lists
// the argumant defaults are in the .h file.  tikSmoother defaults to NULL, which we count on later.

// this is the original prototype
bool Image::LoadRaw ( const char *rawFileName, int frames, bool allocate, bool headerOnly, bool timeTransform )
{
  return ( LoadRaw ( rawFileName, frames, allocate, headerOnly, timeTransform, static_cast<TikhonovSmoother*> ( NULL ) ) );
}

// this is for calling LoadRaw in ImageLoader
bool Image::LoadRaw ( const char *rawFileName, TikhonovSmoother *tikSmoother )
{
  return ( LoadRaw ( rawFileName, 0, true, 0, tikSmoother ) );
}

double TinyTimer()
{
  struct timeval tv;

  gettimeofday ( &tv, NULL );
  double curT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
  return ( curT );
}

void Image::JustCacheOneImage(const char *name)
{
	int totalLen = 0;
    double startT = TinyTimer();
	char buf[1024 * 1024];
	int len;

	// just read the file in...  It will be stored in cache
	int fd = open(name, O_RDONLY);
	if (fd >= 0)
	{
		while ((len = read(fd, &buf[0], sizeof(buf))) > 0)
			totalLen += len;
		close(fd);
	}
	else
	{
		printf("failed to open %s\n", name);
	}

	CacheAccessTime = TinyTimer()-startT;
//	fprintf ( stdout, "File %s cache access = %0.2lf sec. with %d bytes\n", name,stopT - startT,totalLen );
//	fflush ( stdout );
}


// This is the actual function
bool Image::LoadRaw_noWait ( const char *rawFileName, int frames, bool allocate, bool headerOnly)
{
    struct stat buffer;
    //if ( !WaitForMyFileToWakeMeFromSleep ( rawFileName ) )
    if (stat(rawFileName, &buffer)) // file does not exist
    {
        fprintf (stdout, "LoadRaw_noWait warning... file %s doest not exist\n", rawFileName);
        fflush (stdout);
        return (false);
    }

  ( void ) allocate;
  cleanupRaw();

  //set default name only if not already set
  if ( !results_folder )
  {
    results_folder = ( char * ) malloc ( 3 );
    strncpy ( results_folder, "./", 3 );
  }

  //int rc =
  ActuallyLoadRaw ( rawFileName,frames,headerOnly );
  return true;
}


bool Image::LoadRaw_noWait_noSem ( const char *rawFileName, int frames, bool allocate, bool headerOnly)
{
    struct stat buffer;
    //if ( !WaitForMyFileToWakeMeFromSleep ( rawFileName ) )
    if (stat(rawFileName, &buffer)) // file does not exist
    {
        fprintf (stdout, "LoadRaw_noWait warning... file %s doest not exist\n", rawFileName);
        fflush (stdout);
        return (false);
    }

    ( void ) allocate;
    cleanupRaw();

    //set default name only if not already set
    if ( !results_folder )
    {
        results_folder = ( char * ) malloc ( 3 );
        strncpy ( results_folder, "./", 3 );
    }

    //int rc =
    ActuallyLoadRaw_noSem ( rawFileName,frames,headerOnly );
    return true;
}



// This is the actual function
bool Image::LoadRaw ( const char *rawFileName, int frames, bool allocate, bool headerOnly, bool timeTransform, TikhonovSmoother *tikSmoother )
{
//  _file_hdr hdr;
//  int offset=0;
  ( void ) allocate;

  cleanupRaw();

  //set default name only if not already set
  if ( !results_folder )
  {
    results_folder = ( char * ) malloc ( 3 );
    strncpy ( results_folder, "./", 3 );
  }

  //DEBUG: monitor file access time
//  double startT = TinyTimer();

  if ( !WaitForMyFileToWakeMeFromSleep ( rawFileName ) )
    return ( false );

  //int rc =
  ActuallyLoadRaw ( rawFileName,frames,headerOnly, timeTransform );
  
  //TimeStampReporting ( rc );

  // No magic post-processing that is invisible
  // static variables are bugs waiting to happen.
  // >explicitly< do transformations of images


//  double stopT = TinyTimer();
//  fprintf ( stdout, "File access = %0.2lf sec.\n", stopT - startT );
//  fflush ( stdout );

  return true;
}

void Image::SmoothMeTikhonov ( TikhonovSmoother *tikSmoother, bool dont_smooth_me_bro , const char *rawFileName )
{
  if ( !dont_smooth_me_bro )
  {
    // this is where we impliment the --smoothing-file or --smoothing command switch
    // tikSmoother was constructed in ImageLoader.cpp
    // Now that the RawImage has been uncompressed, we optionally smooth it to minimize the
    // RMS second differences with minimal perturbation of the data.  We do it here, early in the
    // image processing to mimic what should be the final situation, i.e., the smoothing takes
    // place on the proton to decrease the entropy of the data to facilitate compression.  And,
    // if we're lucky, improve the SNR by removing noise. (APB)
    // SmoothTrace smooths all the traces in raw->image, leaving the result in raw->image.  Original data is lost
    // See TikhonovSmother.cpp for source and Image/MakeTikMatrix.m for a Matlab script to make the smoothing
    // matrix
    //@TODO: this is the wrong place for this operation, obviously, as everything else here belongs in "raw" class
    if ( tikSmoother != NULL && tikSmoother->doSmoothing )
    {
      tikSmoother->SmoothTrace ( raw->image, raw->rows, raw->cols, raw->frames, rawFileName );
    }
    // End of smoothing
  }
}

int Image::ActuallyLoadRaw ( const char *rawFileName, int frames,  bool headerOnly, bool timeTransform )
{
  int rc;
  raw->channels = 4;
  raw->interlaceType = 0;
  raw->image = NULL;
  if ( frames )
    raw->frames = frames;

  if ( headerOnly )
  {
    rc = deInterlace_c ( ( char * ) rawFileName,NULL,NULL,
                         &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                         0,0,
                         ImageCropping::chipSubRegion.col,ImageCropping::chipSubRegion.row,
                         ImageCropping::chipSubRegion.col+ImageCropping::chipSubRegion.w,ImageCropping::chipSubRegion.row+ImageCropping::chipSubRegion.h,
                         ignoreChecksumErrors, &raw->imageState );
  }
  else
  {

	  if(	  ImageCropping::chipSubRegion.col == 0 && ImageCropping::chipSubRegion.w == 0 &&
			  ImageCropping::chipSubRegion.row == 0 && ImageCropping::chipSubRegion.h == 0 &&
			  raw->frames == 0)
	  { // only cache whole-file reads...
		  double startT = TinyTimer();
		  IonImageSem::Take();
		  SemaphoreWaitTime = TinyTimer()-startT;

		  JustCacheOneImage(rawFileName);

		  IonImageSem::Give();
	  }
	  double startT = TinyTimer();
	  int saved_rows=raw->rows;
	  int saved_cols=raw->cols;
	  int saved_frames=raw->frames;
	  int saved_uncompframes=raw->uncompFrames;
	  int saved_ImageState=raw->imageState;

    rc = deInterlace_c ( ( char * ) rawFileName,&raw->image,&raw->timestamps,
                         &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                         0,0,
                         ImageCropping::chipSubRegion.col,ImageCropping::chipSubRegion.row,
                         ImageCropping::chipSubRegion.col+ImageCropping::chipSubRegion.w,ImageCropping::chipSubRegion.row+ImageCropping::chipSubRegion.h,
                         ignoreChecksumErrors, &raw->imageState );

    // testing of lossy compression
//    printf("loaded %s ts[0]=%d\n",rawFileName,raw->timestamps[0]);
//    if(!headerOnly && timeTransform && rc != 0 && raw->timestamps[0] > 0 && (raw->timestamps[0] < 60 || raw->timestamps[0] > 72))
//    {
//        // more than 10% off from 15fps
//        // use PCA to time-transform this file to 15fps
//        AdvComprTest(rawFileName,this,(char *)"7"); // 7 is write a file that's pca compressed and time transformed
//        free(raw->timestamps);
//        raw->timestamps=NULL;
//        free(raw->image);
//        raw->image=NULL;
//        raw->rows=saved_rows;
//        raw->cols=saved_cols;
//        raw->frames=saved_frames;
//        raw->uncompFrames=saved_uncompframes;
//        raw->imageState = saved_ImageState;
//        char newFname[2048];
//        strcpy(newFname,rawFileName);
//      char *ptr = strstr(newFname,datPostfix);
//        if(ptr)
//        sprintf(ptr,"_testPCA.%s", datPostfix);
//        rc = deInterlace_c ( ( char * ) newFname,&raw->image,&raw->timestamps,
//                             &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
//                             0,0,
//                             ImageCropping::chipSubRegion.col,ImageCropping::chipSubRegion.row,
//                             ImageCropping::chipSubRegion.col+ImageCropping::chipSubRegion.w,ImageCropping::chipSubRegion.row+ImageCropping::chipSubRegion.h,
//                             ignoreChecksumErrors, &raw->imageState );
//    }


    if ( ImageCropping::chipSubRegion.h != 0 )
      raw->rows = ImageCropping::chipSubRegion.h;
    if ( ImageCropping::chipSubRegion.w != 0 )
      raw->cols = ImageCropping::chipSubRegion.w;
    
    if (rc == 0 )
      {
	std::cout << "Invalid dat file: " << rawFileName << endl;
	exit (EXIT_FAILURE);
      }
      TimeStampCalculation();
      FileLoadTime=TinyTimer()-startT;
  }

  raw->frameStride = raw->rows * raw->cols;

    //  printf ( "Loading raw file: %s...done\n", rawFileName );

    return ( rc );
}

int Image::ActuallyLoadRaw_noSem ( const char *rawFileName, int frames,  bool headerOnly )
{
    int rc;
    raw->channels = 4;
    raw->interlaceType = 0;
    raw->image = NULL;
    if ( frames )
        raw->frames = frames;

    if ( headerOnly )
    {
        rc = deInterlace_c ( ( char * ) rawFileName,NULL,NULL,
                             &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                             0,0,
                             ImageCropping::chipSubRegion.col,ImageCropping::chipSubRegion.row,
                             ImageCropping::chipSubRegion.col+ImageCropping::chipSubRegion.w,ImageCropping::chipSubRegion.row+ImageCropping::chipSubRegion.h,
                             ignoreChecksumErrors, &raw->imageState );
    }
    else
    {
        double startT = TinyTimer();

        rc = deInterlace_c ( ( char * ) rawFileName,&raw->image,&raw->timestamps,
                             &raw->rows,&raw->cols,&raw->frames,&raw->uncompFrames,
                             0,0,
                             ImageCropping::chipSubRegion.col,ImageCropping::chipSubRegion.row,
                             ImageCropping::chipSubRegion.col+ImageCropping::chipSubRegion.w,ImageCropping::chipSubRegion.row+ImageCropping::chipSubRegion.h,
                             ignoreChecksumErrors, &raw->imageState );



        if ( ImageCropping::chipSubRegion.h != 0 )
            raw->rows = ImageCropping::chipSubRegion.h;
        if ( ImageCropping::chipSubRegion.w != 0 )
            raw->cols = ImageCropping::chipSubRegion.w;

        if (rc == 0 )
        {
            std::cout << "Invalid dat file: " << rawFileName << endl;
            exit (EXIT_FAILURE);
        }
        TimeStampCalculation();
        FileLoadTime=TinyTimer()-startT;
    }

    raw->frameStride = raw->rows * raw->cols;

    //  printf ( "Loading raw file: %s...done\n", rawFileName );

  return ( rc );
}

bool Image::WaitForMyFileToWakeMeFromSleep ( const char *rawFileName )
{
  FILE *fp = NULL;
  if ( recklessAbandon )
  {
    fopen_s ( &fp, rawFileName, "rb" );
  }
  else   // Try open and wait until open or timeout
  {

    uint32_t waitTime = retry_interval;
    int32_t timeOut = total_timeout;
    //--- Wait up to 3600 seconds for a file to be available
    while ( timeOut > 0 )
    {
      //--- Is the file we want available?
      // why is this a 'static' function?
      if ( ReadyToLoad ( rawFileName , acqPrefix,datPostfix) )
      {
        //--- Open the file we want
        fopen_s ( &fp, rawFileName, "rb" );
        break;  // any error will be reported below
      }
      //DEBUG
      fprintf ( stdout, "Waiting to load %s\n", rawFileName );
      uint32_t timeWaited = 0;
      uint32_t timeLeft = sleep ( waitTime );
      timeWaited = waitTime - timeLeft;
      //      fprintf ( stdout, "Waited to %u load %s\n", timeWaited, rawFileName );
      timeOut -= timeWaited;
    }
    if (timeOut <= 0) {
        fprintf ( stdout, "Waiting to load %s (timed out)\n", rawFileName );
    }
  }
  if ( fp == NULL )
  {
    perror ( rawFileName );
    return false;
  }

//  printf ( "\nLoading raw file: %s...\n", rawFileName );
  fflush ( stdout );
//  size_t rdSize;

  fclose ( fp );
  return ( true );
}

void Image::TimeStampCalculation()
{
  int i,j;
  raw->baseFrameRate=raw->timestamps[0];
  if ( raw->baseFrameRate == 0 )
    { // there were a couple of versions where the thumbnail had a zero first timestamp.
      // correct this by adding the second timestamp to all timestamps...
      raw->baseFrameRate=raw->timestamps[1];
      for (i=0;i<raw->frames;i++)
        raw->timestamps[i] += raw->baseFrameRate;
    }

  raw->interpolatedFrames = ( int * ) malloc ( sizeof ( int ) *raw->uncompFrames );
  raw->interpolatedMult = ( float * ) malloc ( sizeof ( float ) *raw->uncompFrames );
  raw->interpolatedDiv  = ( float * ) malloc ( sizeof ( float ) *raw->uncompFrames );
  raw->compToUncompFrames = (int *) malloc(sizeof(int) * raw->frames);
  fill(raw->compToUncompFrames, raw->compToUncompFrames + raw->frames, -1);
  //  if ( raw->uncompFrames != raw->frames )
  {
    int prevTime,nextTime;
    int numFrames,addedFrames;
    double numFramesF;
    j=0;
    // some dat files have bug with first timestamp being 0
    double baseline = raw->timestamps[0] > 0 ? raw->timestamps[0] : raw->timestamps[1];
    assert(baseline > 0);
    for (i=0;i<raw->frames;i++)
      {
        nextTime = raw->timestamps[i];
        if(i)
          prevTime = raw->timestamps[i-1];
        else
          prevTime = 0;
        
        numFramesF = ((nextTime - prevTime) + 2);
        numFramesF /= baseline; // gets rounded down.  because of the +2 above, this should be right
        numFrames = (uint32_t)numFramesF;
        int prevFrame = i ? raw->compToUncompFrames[i-1] : -1;
        raw->compToUncompFrames[i] = prevFrame + numFrames;
        // add this many entries
        for(addedFrames=0;addedFrames<numFrames;addedFrames++)
          {
            if ((i+1) < raw->frames)
              raw->interpolatedFrames[j] = (i+1);
            else
              raw->interpolatedFrames[j] = raw->frames-1; // don't go past the end..
            //			else
            //				raw->interpolatedFrames[j] = i | 0x8000;
            raw->interpolatedMult[j] = ((float)(numFrames - (float)addedFrames))/((float)numFrames);
            raw->interpolatedDiv[j++] = numFrames;
          }
      }
    if(j != raw->uncompFrames)
      {
	printf("Got the mult wrong %d %d....\n",j,raw->uncompFrames);
      }
  }
  //  else
  //  {
  //	for (i=0;i<raw->uncompFrames;i++)
  //	{
  //		raw->interpolatedFrames[i] = i;
  //		raw->interpolatedMult[i] = 1.0f;
  //		raw->interpolatedDiv[i] = 1.0f;
  //	}
  //  }
}


void Image::TimeStampReporting ( int rc )
{
  if ( rc && raw->timestamps )
  {
    uint32_t prev = 0;
    float avgTimestamp = 0;
    double fps;

    // read the raw data, and convert it into image data
    int i;

    for ( i=0;i<raw->frames;i++ )
    {
      avgTimestamp += ( raw->timestamps[i] - prev );
      prev = raw->timestamps[i];
    }
    avgTimestamp = avgTimestamp / ( raw->frames - 1 );  // milliseconds
    fps = ( 1000.0/avgTimestamp );  // convert to frames per second


    // Subtle hint to users of "old" cropped datasets that did not have real timestamps written
    /*if (rint(fps) == 10) {
      fprintf (stdout, "\n\nWARNING: if this is a cropped dataset, it may have incorrect frame timestamp!\n");
      fprintf (stdout, "Your results will not be valid\n\n");
    }*/

    //DEBUG
    fprintf ( stdout, "Avg Image Time = %f ", avgTimestamp );
    fprintf ( stdout, "Frames = %d ", raw->frames );
    fprintf ( stdout, "FPS = %f\n", fps );

  }
}



void Image::Cleanup()
{
  if ( results )
    delete [] results;
  results = NULL;
}

void Image::SetImage ( RawImage *img )
{
  cleanupRaw();
  delete raw;
  raw = img;
}


//
//  input time is in milliseconds
//  returns frame number corresponding to that time
int Image::GetFrame ( int time , int offset)
{
  int frame = 0;
  int prev=0;
  //flowOffset is time between image start and nuke flow.
  //all times provided are relative to the nuke flow.
  time = time + offset;
  for ( frame=0;frame < raw->frames;frame++ )
  {
    if ( raw->timestamps[frame] >= time )
      break;
    prev = frame;
  }

  return ( prev );
}

int Image::GetFrame ( int time )
{
  return GetFrame(time, flowOffset);
}

// Special:  we usually get >all< the values for a given trace and send them to the bkgmodel.
// more efficient to keep variables around than repeatedly calling GetInterpolatedVal
void Image::GetUncompressedTrace ( float *val, int last_frame, int x, int y )
{
  // set up:  index by well, make sure last_frame is within range
  int l_coord = y*raw->cols+x;

  if ( last_frame>raw->uncompFrames )
    last_frame = raw->uncompFrames;
// if compressed
  if ( raw->uncompFrames != raw->frames )
  {
    int my_frame = 0;
    val[my_frame] = raw->image[l_coord];

    float prev=raw->image[l_coord];
    float next=0.0f;

    for ( my_frame=1; my_frame<last_frame; my_frame++ )
    {
      // need to make this faster!!!
      int interf= raw->interpolatedFrames[my_frame];

      int f_coord = l_coord+raw->frameStride*interf;
      next = raw->image[f_coord];
      prev = raw->image[f_coord-raw->frameStride];

      // interpolate
      float mult = raw->interpolatedMult[my_frame];
      val[my_frame] = ( prev-next ) *mult + next;
    }
  }
  else
  {
    // the rare "uncompressed" case
    for ( int my_frame=0; my_frame<last_frame; my_frame++ )
    {
      val[my_frame] = raw->image[l_coord+my_frame*raw->frameStride];
    }
  }
}



float Image::GetInterpolatedValue ( int frame, int x, int y )
{
  float rc;
  if ( frame < 0 )
  {
    printf ( "asked for negative frame!!!  %d\n",frame );
    return 0.0f;
  }
  if ( raw->uncompFrames == raw->frames )
  {
    rc = raw->image[raw->frameStride*frame+y*raw->cols+x];
  }
  else
  {
    if ( frame==0 )
    {
      rc = raw->image[y*raw->cols+x];
    }
    else
    {
      if ( frame >= raw->uncompFrames )
        frame = raw->uncompFrames-1;

      // need to make this faster!!!
      int interf=raw->interpolatedFrames[frame];
      float mult = raw->interpolatedMult[frame];

      float prev=0.0f;
      float next=0.0f;

      next = raw->image[raw->frameStride*interf+y*raw->cols+x];
      if ( interf )
        prev = raw->image[raw->frameStride* ( interf-1 ) +y*raw->cols+x];

      // interpolate
      rc = ( prev-next ) *mult + next;
    }
  }

  return rc;
}



int Image::FilterForPinned ( Mask *mask, MaskType these, int markBead )
{

  int x, y, frame;
  int pinnedCount = 0;
  int i = 0;
  const short pinLow = GetPinLow();
  const short pinHigh = GetPinHigh();

  printf ( "Filtering for pinned pixels between %d & %d.\n", pinLow, pinHigh );

  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      if ( ( *mask ) [i] & these )
      {
        for ( frame=0;frame<raw->frames;frame++ )
        {
          if ( raw->image[frame*raw->frameStride + i] <= pinLow ||
               raw->image[frame*raw->frameStride + i] >= pinHigh )
          {
            ( *mask ) [i] = MaskPinned; // this pixel is pinned high or low
            if ( markBead )
              ( *mask ) [i] |= MaskBead;
            pinnedCount++;
            break;
          }
        }
      }
      i++;
    }
  }
  fprintf ( stdout, "FilterForPinned: found %d\n", pinnedCount );
  return pinnedCount;
}

typedef int16_t v8s16_t __attribute__ ((vector_size (16)));
typedef union{
	v8s16_t V;
	int16_t A[8];
}v8s16_e;

void Image::SetMeanOfFramesToZero ( int startPos, int endPos, int use_compressed_frame_nums )
{
  // normalize trace data to the input frame per well
//  double stopT,startT = TinyTimer();
//  printf ( "SetMeanOfFramesToZero from frame %d to %d\n",startPos,endPos );
  int frame, x, y,rStartPos=0,rEndPos=0;
  int32_t refLA[8];
  v8s16_e refA;
  v8s16_t * __restrict imgV;
  int ref;
  int i = 0;
  short * __restrict imagePtr;
  int nframes = raw->frames;
  int32_t idx;

  if(use_compressed_frame_nums)
  {
	  int frm;
	  for(frm=0;frm<raw->frames;frm++)
	  {
		  if(rStartPos == 0 && raw->timestamps[frm] > startPos)
			  rStartPos = frm;
		  if(rEndPos == 0 && raw->timestamps[frm] > endPos)
			  rEndPos = frm;

	  }
  }
  else
  {
	  rStartPos = startPos;
	  rEndPos = endPos;
  }

  if((raw->cols % 8) != 0)
  {
	  for ( y=0;y<raw->rows;y++ )
	  {
	    for ( x=0;x<raw->cols;x++ )
	    {
	      int pos;
	      ref = 0;
	      for ( pos=rStartPos;pos<=rEndPos;pos++ )
	        ref += raw->image[pos*raw->frameStride+i];
	      ref /= ( rEndPos-rStartPos+1 );
	      imagePtr = &raw->image[i];
	      for ( frame=0;frame<nframes;frame++ )
	      {
	        *imagePtr -= ref;
	        imagePtr += raw->frameStride;
	      }
	      i++;
	    }
	  }
  }
  else
  {
	  for ( y=0;y<raw->rows;y++ )
	  {
	    for ( x=0;x<raw->cols;x+=8 )
	    {
	      int pos;
	      refLA[0] = 0;
	      refLA[1] = 0;
	      refLA[2] = 0;
	      refLA[3] = 0;
	      refLA[4] = 0;
	      refLA[5] = 0;
	      refLA[6] = 0;
	      refLA[7] = 0;
	      for ( pos=rStartPos;pos<=rEndPos;pos++ )
	      {
	          imagePtr = &raw->image[pos*raw->frameStride+i];
	          refLA[0] += imagePtr[0];
	    	  refLA[1] += imagePtr[1];
	    	  refLA[2] += imagePtr[2];
	    	  refLA[3] += imagePtr[3];
	    	  refLA[4] += imagePtr[4];
	    	  refLA[5] += imagePtr[5];
	    	  refLA[6] += imagePtr[6];
	    	  refLA[7] += imagePtr[7];
	      }
	      idx = ( endPos-startPos+1 );
	      refA.A[0] = (int16_t)(refLA[0]/idx);
	      refA.A[1] = (int16_t)(refLA[1]/idx);
	      refA.A[2] = (int16_t)(refLA[2]/idx);
	      refA.A[3] = (int16_t)(refLA[3]/idx);
	      refA.A[4] = (int16_t)(refLA[4]/idx);
	      refA.A[5] = (int16_t)(refLA[5]/idx);
	      refA.A[6] = (int16_t)(refLA[6]/idx);
	      refA.A[7] = (int16_t)(refLA[7]/idx);

	      imagePtr = &raw->image[i];
	      for ( frame=0;frame<nframes;frame++ )
	      {
	    	imgV = (v8s16_t *)  imagePtr;
	        *imgV -= refA.V;
	        imagePtr += raw->frameStride;
	      }
	      i += 8;
	    }
	  }
  }
//  stopT = TinyTimer();
//  printf ( "SetMeanOfFramesToZero(%d-%d) in %0.2lf sec\n",startPos,endPos,stopT - startT );
}

void Image::IntegrateRaw ( Mask *mask, MaskType these, int start, int end )
{
  printf ( "Integrating Raw...\n" );
  if ( !results )
    results = new double[raw->rows * raw->cols];
  memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );

  int x, y, frame, k;
  double buf;
  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      if ( ( *mask ) [x+raw->cols*y] & these )
      {
        k = y*raw->cols + x + start*raw->frameStride;
        buf = 0;
        for ( frame=start;frame<=end;frame++ )
        {
          buf += raw->image[k];
          k += raw->frameStride;
        }
        results[x+y*raw->cols] = buf; // /(double)(end-start+1.0);
      }
    }
  }
}

void Image::IntegrateRawBaseline ( Mask *mask, MaskType these, int start, int end, int baselineStart, int baselineEnd, double *_minval, double *_maxval )
{
  printf ( "Integrating Raw Baseline...\n" );
  if ( !results )
    results = new double[raw->rows * raw->cols];
  memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );

  int x, y, frame, k;
  double buf;
  double minval = 99999999999.0, maxval = -99999999999.0;
  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      if ( ( *mask ) [x+raw->cols*y] & these )
      {
        k = y*raw->cols + x + start*raw->frameStride;
        buf = 0;
        int pos;
        int ref = 0;
        for ( pos=baselineStart;pos<=baselineEnd;pos++ )
          ref += raw->image[pos*raw->frameStride+x+y*raw->cols];
        ref /= ( baselineEnd-baselineStart+1 );
        for ( frame=start;frame<=end;frame++ )
        {
          buf += ( raw->image[k] - ref );
          k += raw->frameStride;
        }
        results[x+y*raw->cols] = buf; // /(double)(end-start+1.0);
        if ( buf > maxval ) maxval = buf;
        if ( buf < minval ) minval = buf;
      }
    }
  }
  if ( _minval ) *_minval = minval;
  if ( _maxval ) *_maxval = maxval;
}




void Image::FindPeak ( Mask *mask, MaskType these )
{
  if ( !results )
    results = new double[raw->rows * raw->cols];
  memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );

  int frame, x, y;
  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      if ( ( *mask ) [x+raw->cols*y] & these )
      {
        for ( frame=0;frame<raw->frames;frame++ )
        {
          if ( frame == 0 || ( raw->image[frame*raw->frameStride + x+y*raw->cols] > results[x+y*raw->cols] ) )
          {
            results[x+y*raw->cols] = frame; // only need to save what frame we found the peak
          }
        }
      }
    }
  }
}


///------------------------Additive scaling

void Image::SubtractLocalReferenceTrace ( Mask *mask, MaskType these, MaskType usingThese, int inner, int outer, bool saveBkg, bool onlyBkg, bool replaceWBkg )
{
  SubtractLocalReferenceTrace ( mask, these, usingThese, inner, inner, outer, outer, saveBkg, onlyBkg,replaceWBkg );
}

void Image::GenerateCumulativeSumMatrix ( int64_t *workTotal, unsigned int *workNum, uint16_t *MaskPtr, MaskType derive_from_these, int frame )
{
  unsigned int *lWorkNumPtr;
  int64_t *lWorkTotalPtr;
  short *fptr;
  short *Rfptr;
  uint16_t lmsk,*rMaskPtr;

  memset ( workNum  ,0,sizeof ( unsigned int ) *raw->rows*raw->cols );
  memset ( workTotal,0,sizeof ( int64_t ) *raw->rows*raw->cols );
  fptr = &raw->image[frame*raw->frameStride];
  lWorkTotalPtr = workTotal;
  lWorkNumPtr = workNum;
//    int skipped = 0;

  // sum empty wells on the whole image
  for ( int y=0;y<raw->rows;y++ )
  {
    rMaskPtr = &MaskPtr[raw->cols*y];
    Rfptr = &fptr[raw->cols*y];
    for ( int x=0;x<raw->cols;x++ )
    {
      lmsk = rMaskPtr[x];

      if ( ( lmsk & derive_from_these ) // look only at our beads...
           /*!(lmsk & MaskWashout)*/ )   // Skip any well marked ignore
      {
        *lWorkTotalPtr = Rfptr[x];
        *lWorkNumPtr   = 1;
      }
      else
      {
        *lWorkTotalPtr = 0;
        *lWorkNumPtr   = 0;
      }
      if ( x )
      {
        *lWorkNumPtr   += * ( lWorkNumPtr-1 );  // the one to the left
        *lWorkTotalPtr += * ( lWorkTotalPtr-1 ); // the one to the left
      }
      if ( y )
      {
        *lWorkNumPtr   += * ( lWorkNumPtr   - raw->cols ); // the one above
        *lWorkTotalPtr += * ( lWorkTotalPtr - raw->cols ); // the one above
      }
      if ( x && y )
      {
        *lWorkNumPtr   -= * ( lWorkNumPtr   - raw->cols - 1 ); // add the common area
        *lWorkTotalPtr -= * ( lWorkTotalPtr - raw->cols - 1 ); // add the common area
      }
      lWorkNumPtr++;
      lWorkTotalPtr++;
    }
  }
}

int Image::WholeFrameMean ( int64_t *workTotal, unsigned int *workNum )
{
  int sum, count;

  int tlo = 0;
  int blo = ( raw->rows-1 ) *raw->cols;
  int tro = raw->cols-1;
  int bro = ( raw->rows-1 ) *raw->cols + raw->cols-1;
  sum    = workTotal[bro] + workTotal[tlo];
  sum   -= workTotal[tro] + workTotal[blo];
  count  = workNum[bro] + workNum[tlo];
  count -= workNum[tro] + workNum[blo];
  int WholeBkg = 0;
  if ( sum != 0 && count != 0 )
  {
    WholeBkg = sum/count;
  }
  return ( WholeBkg );
}


void Image::ApplyLocalReferenceToWholeChip ( int64_t *workTotal, unsigned int *workNum,
    uint16_t *MaskPtr, MaskType apply_to_these,
    int innerx, int innery, int outerx, int outery,
    bool saveBkg, bool onlyBkg, bool replaceWBkg, int frame )
{
  short *fptr;
  fptr = &raw->image[frame*raw->frameStride];
  short *Rfptr;
  uint16_t lmsk,*rMaskPtr;
    int typical_value = WholeFrameMean ( workTotal,workNum );
    
  // now, compute background for each live bead
  for ( int y=0;y<raw->rows;y++ )
  {
    rMaskPtr = &MaskPtr[raw->cols*y];
    Rfptr = &fptr[raw->cols*y];
    for ( int x=0;x<raw->cols;x++ )
    {
      lmsk = rMaskPtr[x];
      if ( ( lmsk & apply_to_these ) )
      {
        // compute the whole chip subtraction coefficient
        int yo1,yo2,xo1,xo2,yi1,yi2,xi1,xi2,tli,bli,tri,bri;
        int tlo,blo,tro,bro;
        yo1 = ( y-outery ) < 0 ? 0 : ( y-outery );
        yo2 = ( y+outery ) >= raw->rows ? raw->rows-1 : ( y+outery );
        xo1 = ( x-outerx ) < 0 ? 0 : ( x-outerx );
        xo2 = ( x+outerx ) >= raw->cols ? raw->cols-1 : ( x+outerx );
        yi1 = ( y-innery ) < 0 ? 0 : ( y-innery );
        yi2 = ( y+innery ) >= raw->rows ? raw->rows-1 : ( y+innery );
        xi1 = ( x-innerx ) < 0 ? 0 : ( x-innerx );
        xi2 = ( x+innerx ) >= raw->cols ? raw->cols-1 : ( x+innerx );

        tli = ( yi1?yi1-1:0 ) *raw->cols + ( xi1?xi1-1:0 );
        bli = yi2*raw->cols + ( xi1?xi1-1:0 );
        tri = ( yi1?yi1-1:0 ) *raw->cols + xi2;
        bri = yi2*raw->cols + xi2;
        tlo = ( yo1?yo1-1:0 ) *raw->cols + ( xo1?xo1-1:0 );
        blo = yo2*raw->cols + ( xo1?xo1-1:0 );
        tro = ( yo1?yo1-1:0 ) *raw->cols + xo2;
        bro = yo2*raw->cols + xo2;

        int64_t sum,innersum;
        int count,innercount;
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
        if ( count > 0 )
        {
          sum /= count;
        }
        else
        {
          sum =typical_value;
        }
        // update the value (background subtract) in the work buffer
        if ( !onlyBkg )
        {
          if ( replaceWBkg )
            Rfptr[x] = sum;
          else
            Rfptr[x] -= sum;
        }

        if ( saveBkg )
        {
          bkg[frame*raw->frameStride+y*raw->cols+x] = sum;
        }
      }
//        else
//          skipped++;
    }
  }
}

void Image::SetUpBkgSave ( bool saveBkg )
{
  if ( saveBkg )
  {
    if ( bkg )
      free ( bkg );
    bkg = ( int16_t * ) malloc ( sizeof ( int16_t ) * raw->rows*raw->cols*raw->frames );
    memset ( bkg,0,sizeof ( int16_t ) *raw->rows*raw->cols*raw->frames );
  }
}



void Image::SubtractLocalReferenceTrace ( Mask *mask, MaskType apply_to_these, MaskType derive_from_these, int innerx, int innery, int outerx, int outery, bool saveBkg, bool onlyBkg, bool replaceWBkg )
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
  int64_t *workTotal = ( int64_t * ) malloc ( sizeof ( int64_t ) *raw->rows*raw->cols );
  unsigned int *workNum   = ( unsigned int * ) malloc ( sizeof ( unsigned int ) *raw->rows*raw->cols );
  uint16_t *MaskPtr = ( uint16_t * ) mask->GetMask();

  SetUpBkgSave ( saveBkg );

  for ( int frame=0;frame<raw->frames;frame++ )
  {
    GenerateCumulativeSumMatrix ( workTotal, workNum, MaskPtr, derive_from_these, frame );
    ApplyLocalReferenceToWholeChip ( workTotal,workNum,MaskPtr,apply_to_these, innerx, innery,outerx,outery,saveBkg,onlyBkg,replaceWBkg,frame );
  }

  free ( workNum );
  free ( workTotal );

}


void Image::GenerateCumulativeSumMatrixInRegion ( Region &reg, int64_t *workTotal, unsigned int *workNum, uint16_t *MaskPtr, MaskType derive_from_these, int frame )
{
  unsigned int *lWorkNumPtr;
  int64_t *lWorkTotalPtr;
  short *fptr;

  uint16_t lmsk;

  memset ( workNum  ,0,sizeof ( unsigned int ) *raw->rows*raw->cols );
  memset ( workTotal,0,sizeof ( int64_t ) *raw->rows*raw->cols );

  fptr = &raw->image[frame*raw->frameStride];
  lWorkTotalPtr = workTotal;
  lWorkNumPtr = workNum;
  //    int skipped = 0;
  int rowStart = reg.row;
  int rowEnd = reg.row+reg.h;
  int colStart = reg.col;
  int colEnd = reg.col+reg.w;

  // calculate cumulative sum once so fast to calculate sum  empty wells on the whole image
  for ( int y=rowStart;y<rowEnd;y++ )
  {
    for ( int x=colStart;x<colEnd;x++ )
    {
      int wellIx = y * raw->cols + x;
      lmsk = MaskPtr[wellIx];
      if ( ( lmsk & derive_from_these ) )   // look only at our beads...
      {
        lWorkTotalPtr[wellIx] = fptr[wellIx];
        lWorkNumPtr[wellIx]   = 1;
      }
      else
      {
        lWorkTotalPtr[wellIx] = 0;
        lWorkNumPtr[wellIx]   = 0;
      }
      if ( x != colStart )
      {
        lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx-1];   // the one to the left
        lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx-1]; // the one to the left
      }
      if ( y != rowStart )
      {
        lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx - raw->cols]; // the one below
        lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx - raw->cols]; // the one below
      }
      if ( x != colStart && y != rowStart )
      {
        lWorkNumPtr[wellIx]   -= lWorkNumPtr[wellIx - raw->cols - 1]; // add the common area
        lWorkTotalPtr[wellIx] -= lWorkTotalPtr[wellIx - raw->cols - 1]; // add the common area
      }
    }
  }
}

int Image::FindMeanValueInRegion ( Region &reg, int64_t *workTotal, unsigned int *workNum )
{
  int bro = ( reg.row+reg.h-1 ) *raw->cols + reg.col+reg.w-1;
  int sum    = workTotal[bro];
  int count = workNum[bro];
  int WholeBkg = 0;
  if ( sum != 0 && count != 0 )
  {
    WholeBkg = sum/count;
  }
  return ( WholeBkg );
}

void Image::ApplyLocalReferenceInRegion ( Region &reg, int64_t *workTotal, unsigned int *workNum,
    uint16_t *MaskPtr, MaskType apply_to_these,
    int innerx, int innery, int outerx, int outery,
    bool saveBkg, bool onlyBkg, bool replaceWBkg, int frame )
{

  int rowStart = reg.row;
  int rowEnd = reg.row+reg.h;
  int colStart = reg.col;
  int colEnd = reg.col+reg.w;

  int64_t sum,innersum;
  int count,innercount;

  int typical_value= FindMeanValueInRegion ( reg,workTotal,workNum );

  short *fptr;
  uint16_t lmsk;

  // now, compute background for each live bead
  fptr = &raw->image[frame*raw->frameStride];
  for ( int y=rowStart;y<rowEnd;y++ )
  {
    //      rMaskPtr = &MaskPtr[raw->cols*y];
    //      Rfptr = &fptr[raw->cols*y];
    for ( int x=colStart;x<colEnd;x++ )
    {
      int wellIx = y * raw->cols + x;
      lmsk = MaskPtr[wellIx];        //if ( (lmsk & these) != 0 )
      if ( ( lmsk & apply_to_these ) )
      {
        // Compute the whole chip subtraction coefficient
        int yo1,yo2,xo1,xo2, yi1,yi2,xi1,xi2,tli,bli,tri,bri;
        int tlo,blo,tro,bro;
        yo1 = ( y-outery ) < rowStart ? rowStart : ( y-outery );
        yo2 = ( y+outery ) >= rowEnd ? rowEnd-1 : ( y+outery );
        xo1 = ( x-outerx ) < colStart ? colStart : ( x-outerx );
        xo2 = ( x+outerx ) >= colEnd ? colEnd-1 : ( x+outerx );
        yi1 = ( y-innery ) < rowStart ? rowStart : ( y-innery );
        yi2 = ( y+innery ) >= rowEnd ? rowEnd-1 : ( y+innery );
        xi1 = ( x-innerx ) < colStart ? colStart : ( x-innerx );
        xi2 = ( x+innerx ) >= colEnd ? colEnd-1 : ( x+innerx );

        tli = ( yi1 != rowStart ? yi1-1: rowStart ) * raw->cols + ( xi1 != colStart ? xi1-1 : colStart );
        bli = yi2*raw->cols + ( xi1 != colStart ? xi1-1 : colStart );
        tri = ( yi1 != rowStart ? yi1-1 : rowStart ) *raw->cols + xi2;
        bri = yi2*raw->cols + xi2;
        tlo = ( yo1 != rowStart ? yo1-1 : rowStart ) *raw->cols + ( xo1 != colStart? xo1-1 : colStart );
        blo = yo2*raw->cols + ( xo1 != colStart ? xo1-1 : colStart );
        tro = ( yo1 != rowStart ? yo1-1:rowStart ) *raw->cols + xo2;
        bro = yo2*raw->cols + xo2;

        sum = workTotal[bro];
        count  = workNum[bro];
        if ( yo1 != rowStart )
        {
          sum -= workTotal[tro];
          count -= workNum[tro];
        }
        if ( xo1 != colStart )
        {
          sum -= workTotal[blo];
          count -= workNum[blo];
        }
        if ( xo1 != colStart && yo1 != rowStart )
        {
          sum += workTotal[tlo];
          count += workNum[tlo];
        }

        innersum = workTotal[bri];
        innercount  = workNum[bri];
        if ( yi1 != rowStart )
        {
          innersum -= workTotal[tri];
          innercount -= workNum[tri];
        }
        if ( xi1 != colStart )
        {
          innersum -= workTotal[bli];
          innercount -= workNum[bli];
        }
        if ( xi1 != colStart && yi1 != rowStart )
        {
          innersum += workTotal[tli];
          innercount += workNum[tli];
        }

        if ( count > 0 )
        {
          sum /= count;
        }
        else
        {
          sum = typical_value; // fill in with generic value for region
        }
        // update the value (background subtract) in the work buffer
        if ( !onlyBkg )
        {
          if ( replaceWBkg )
            fptr[wellIx] = sum;
          else
            fptr[wellIx] -= sum;
        }
        if ( saveBkg )
        {
          bkg[frame*raw->frameStride+y*raw->cols+x] = sum;
        }
      }
    }
  }
}

void Image::SubtractLocalReferenceTraceInRegion ( Region &reg, Mask *mask, MaskType apply_to_these, MaskType derive_from_these,
    int innerx, int innery, int outerx, int outery,
    bool saveBkg, bool onlyBkg, bool replaceWBkg )
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
  int64_t *workTotal = ( int64_t * ) malloc ( sizeof ( int64_t ) *raw->rows*raw->cols );
  unsigned int *workNum   = ( unsigned int * ) malloc ( sizeof ( unsigned int ) *raw->rows*raw->cols );

  uint16_t *MaskPtr = ( uint16_t * ) mask->GetMask();

  SetUpBkgSave ( saveBkg );

  for ( int frame=0;frame<raw->frames;frame++ )
  {

    GenerateCumulativeSumMatrixInRegion ( reg, workTotal, workNum, MaskPtr, derive_from_these, frame );
    ApplyLocalReferenceInRegion ( reg,workTotal,workNum, MaskPtr,apply_to_these, innerx, innery,outerx,outery,saveBkg,onlyBkg,replaceWBkg,frame );

  }

  free ( workNum );
  free ( workTotal );
}

// Metric 1 is, for each trace, the max - abs(min)
// Arguments are only used for generating the metrics histogram which is only
// used for reporting purposes.  Note that the actual clustering bead finding
// happens in the Separator - find beads.
void Image::CalcBeadfindMetric_1 ( Mask *mask, Region region, char *idStr, int frameStart, int frameEnd )
{
  printf ( "gathering metric 1...\n" );
  if ( !results )
  {
    results = new double[raw->rows * raw->cols];
    memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );
    //fprintf (stdout, "Image::CalcBeadfindMetric_1 allocated: %lu\n",raw->rows * raw->cols * sizeof(double) );
  }

  if ( frameStart == -1 )
  {
    frameStart = GetFrame ( 12 ); // Frame 15;
  }
  if ( frameEnd == -1 )
  {
    frameEnd = GetFrame ( 2374 ); // Frame 50;
  }
  fprintf ( stdout, "Image: CalcBeadFindMetric_1 %d %d\n", frameStart, frameEnd );
  int frame, x, y;
  int k;
  int min, max;
  //int printCnt = 0;
  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      k = x + y*raw->cols;
      min = 65536;
      max = -65536;
      //int minK = 0;
      //int maxK = 0;
      // for(frame=0;frame<raw->frames;frame++) {
      k += frameStart*raw->frameStride;
      for ( frame=frameStart;frame<frameEnd;frame++ )
      {
        if ( frame == frameStart || ( raw->image[k] > max ) )
        {
          max = raw->image[k];
          //maxK = frame;
        }
        if ( frame == frameStart || ( raw->image[k] < min ) )
        {
          min = raw->image[k];
          //minK = frame;
        }
        k += raw->frameStride;
      }
      results[x+y*raw->cols] = max - abs ( min );

    }
  }
}

void Image::CalcBeadfindMetricRegionMean ( Mask *mask, Region region, const char *idStr, int frameStart, int frameEnd )
{
  //  printf ( "gathering regional mean...\n" );
  if ( !results )
  {
    results = new double[raw->rows * raw->cols];
    memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );
    //fprintf (stdout, "Image::CalcBeadfindMetric_1 allocated: %lu\n",raw->rows * raw->cols * sizeof(double) );
  }

  if ( frameStart == -1 )
  {
    frameStart = GetFrame ( 12 ); // Frame 15;
  }
  if ( frameEnd == -1 )
  {
    frameEnd = GetFrame ( 2374 ); // Frame 50;
  }
  //  fprintf ( stdout, "Image: CalcBeadFindMetricRegionMean %d %d\n", frameStart, frameEnd );
  int frame, x, y;
  int k;
  int min, max;
  vector<float> mean(frameEnd - frameStart, 0);
  //int printCnt = 0;
  int count = 0;
  int rStart = region.row;
  int rEnd = region.row + region.h;
  int cStart = region.col;
  int cEnd = region.col + region.w;

  // Get the mean for the region
  for ( y=rStart;y<rEnd;y++ )
  {
    for ( x=cStart;x<cEnd;x++ )
    {
      if (!mask->Match(x,y,MaskPinned)) {
        k = x + y*raw->cols;
        k += frameStart*raw->frameStride;
        for ( frame=frameStart;frame<frameEnd;frame++ ) {
          mean[frame - frameStart] += raw->image[k];
          k += raw->frameStride;
        }
        count++;        
      }
    }
  }

  for (size_t i = 0; i < mean.size(); i++) {
    mean[i] = mean[i] / count;
  }

  // Get the beadfind signal from mean
  for ( y=rStart;y<rEnd;y++ )
  {
    for ( x=cStart;x<cEnd;x++ )
    {
      k = x + y*raw->cols;
      min = 65536;
      max = -65536;
      k += frameStart*raw->frameStride;
      for ( frame=frameStart;frame<frameEnd;frame++ )
      {
        int val = raw->image[k] - mean[frame - frameStart];
        if ( frame == frameStart || ( val > max ) ) {
          max = val;
        }
        if ( frame == frameStart || ( val < min ) ) {
          min = val;
        }
        k += raw->frameStride;
      }
      results[x+y*raw->cols] = max - abs ( min );
    }
  }
}

void Image::CalcBeadfindMetricIntegral ( Mask *mask, Region region, char *idStr, int frameStart, int frameEnd )
{
  printf ( "gathering metric integral...\n" );
  if ( !results )
  {
    results = new double[raw->rows * raw->cols];
    memset ( results, 0, raw->rows * raw->cols * sizeof ( double ) );
    //fprintf (stdout, "Image::CalcBeadfindMetric_1 allocated: %lu\n",raw->rows * raw->cols * sizeof(double) );
  }

  if ( frameStart == -1 )
  {
    frameStart = GetFrame ( 12 ); // Frame 15;
  }
  if ( frameEnd == -1 )
  {
    frameEnd = GetFrame ( 2374 ); // Frame 50;
  }
  fprintf ( stdout, "Image: CalcBeadFindMetricIntegral %d %d\n", frameStart, frameEnd );
  int frame, x, y;
  int k;
  //int printCnt = 0;
  for ( y=0;y<raw->rows;y++ )
  {
    for ( x=0;x<raw->cols;x++ )
    {
      k = x + y*raw->cols;
      k += frameStart*raw->frameStride;
      double sum = 0;
      for ( frame=frameStart;frame<frameEnd;frame++ )
      {
        sum += raw->image[k];
        k += raw->frameStride;
      }
      results[x+y*raw->cols] = sum;
    }
  }
}


//*-------------------------------multiplicative scaling



// really?
// should this be always positive, as max-min instead?
short Image::CalculateCharacteristicValue(short *prow, int ax)
{
  // should this, perhaps, be maxval-minval?
        short *ptrc = prow + ax;

        short maxval = 0;
        for ( int i=0;i < raw->frames;i++ )
        {
          if ( * ( ptrc+i*raw->frameStride ) > maxval )
            maxval = * ( ptrc+i*raw->frameStride );
        }
        return(maxval);
}

void Image::FindCharacteristicValuesInReferenceWells(short *tmp, short *pattern_flag,
                                                     MaskType reference_type, Mask *mask_ptr, PinnedInFlow &pinnedInFlow,
                                                     int flow)
{
  // find the peak values of the raw signal in all the empty wells
  for ( int ay=0;ay < raw->rows;ay++ )
  {
    short *prow;
    prow = raw->image+ay*raw->cols;

    for ( int ax=0;ax < raw->cols;ax++ )
    {
      int ix = mask_ptr->ToIndex ( ay, ax );
      bool isEmpty = mask_ptr->Match ( ax,ay,reference_type );
      bool isIgnoreOrAmbig = mask_ptr->Match ( ax,ay, ( MaskType ) ( MaskIgnore ) );
      bool isUnpinned = ! ( pinnedInFlow.IsPinned ( flow, ix ) );
      if ( isEmpty & isUnpinned & ~isIgnoreOrAmbig )   // valid empty well
      {
        tmp[ay*raw->cols + ax] = CalculateCharacteristicValue(prow,ax);
        pattern_flag[ay*raw->cols+ax] = 1;
      }
    }
  }
}

void Image::SlowSmoothPattern(short *tmp, short *pattern_flag, int smooth_span)
{
  // neighbor average the peak signal in empty wells
  for ( int ay=0;ay < raw->rows;ay++ )
  {
    for ( int ax=0;ax < raw->cols;ax++ )
    {
      float sum = 0.0f;
      int nsum = 0;

      for ( int y = ay-smooth_span;y <= ay+smooth_span;y++ )
      {
        if ( ( y < 0 ) || ( y >= raw->rows ) )
          continue;

        for ( int x = ax-smooth_span;x <= ax+smooth_span;x++ )
        {
          if ( ( x < 0 ) || ( x >= raw->cols ) )
            continue;

          if ( pattern_flag[y*raw->cols+x] )   // valid empty well
          {
            sum += ( float ) ( tmp[y*raw->cols + x] );
            nsum++;
          }
        }
      }

      if ( nsum > 0 )
        smooth_max_amplitude[ax+ay*raw->cols] = sum / nsum;
      else
        smooth_max_amplitude[ax+ay*raw->cols] = 0.0f;
    }
  }
}



void Image::FastSmoothPattern(short *tmp, short *pattern_flag, int smooth_span)
{
  // allocate a temporary one-frame buffer
  int64_t *workTotal = ( int64_t * ) malloc ( sizeof ( int64_t ) *raw->rows*raw->cols );
  unsigned int *workNum   = ( unsigned int * ) malloc ( sizeof ( unsigned int ) *raw->rows*raw->cols );

  GenerateCumulativeSumMatrixFromPattern(workTotal,workNum,tmp, pattern_flag);
  // now send smoothed values to smooth_max_amplitude
  SmoothMaxByPattern(workTotal, workNum, smooth_span);

  free( workTotal);
  free( workNum);
}


void Image::GenerateCumulativeSumMatrixFromPattern ( int64_t *workTotal, unsigned int *workNum, short *my_values, short *my_pattern )
{
  unsigned int *lWorkNumPtr;
  int64_t *lWorkTotalPtr;


  memset ( workNum  ,0,sizeof ( unsigned int ) *raw->rows*raw->cols );
  memset ( workTotal,0,sizeof ( int64_t ) *raw->rows*raw->cols );

  lWorkTotalPtr = workTotal;
  lWorkNumPtr = workNum;
  //    int skipped = 0;
  int rowStart = 0;
  int rowEnd = raw->rows;
  int colStart = 0;
  int colEnd = raw->cols;

  // calculate cumulative sum once so fast to calculate sum  empty wells on the whole image
  for ( int y=rowStart;y<rowEnd;y++ )
  {
    for ( int x=colStart;x<colEnd;x++ )
    {
      int wellIx = y * raw->cols + x;

      if ( ( my_pattern[wellIx] ) )   // look only at our beads...
      {
        lWorkTotalPtr[wellIx] = my_values[wellIx];
        lWorkNumPtr[wellIx]   = 1;
      }
      else
      {
        lWorkTotalPtr[wellIx] = 0;
        lWorkNumPtr[wellIx]   = 0;
      }
      
      if ( x != colStart )
      {
        lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx-1];   // the one to the left
        lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx-1]; // the one to the left
      }
      if ( y != rowStart )
      {
        lWorkNumPtr[wellIx]   += lWorkNumPtr[wellIx - raw->cols]; // the one above
        lWorkTotalPtr[wellIx] += lWorkTotalPtr[wellIx - raw->cols]; // the one above
      }
      if ( x != colStart && y != rowStart )
      {
        lWorkNumPtr[wellIx]   -= lWorkNumPtr[wellIx - raw->cols - 1]; // add the common area only once, not twice
        lWorkTotalPtr[wellIx] -= lWorkTotalPtr[wellIx - raw->cols - 1]; // add the common area
      }
    }
  }
}

void FindMyCorners(int &least_x, int &least_y, int &max_x, int &max_y,
                   int rowStart, int rowEnd, int colStart, int colEnd,
                   int x, int y,int smooth_span)
{
  least_x = x-smooth_span-1;
  if (least_x<colStart) least_x = colStart-1;
  least_y = y-smooth_span-1;
  if (least_y<rowStart) least_y = rowStart-1;
  max_x = x+smooth_span;
  if (max_x>(colEnd-1)) max_x = colEnd-1;
   max_y = y+smooth_span;
  if (max_y>(rowEnd-1)) max_y = rowEnd-1;
}

void Image::SmoothMaxByPattern (  int64_t *workTotal, unsigned int *workNum, int smooth_span)
{

  int rowStart = 0;
  int rowEnd = raw->rows;
  int colStart = 0;
  int colEnd = raw->cols;

  float typical_value= 0.0f; // skip me if zero count

  for ( int y=rowStart;y<rowEnd;y++ )
  {
    for ( int x=colStart;x<colEnd;x++ )
    {
      int wellIx = y * raw->cols + x;
      int least_x, least_y,max_x, max_y;
        // corners of region
        FindMyCorners(least_x,least_y,max_x,max_y,
                      rowStart, rowEnd, colStart, colEnd,
                      x,y,smooth_span);
        int64_t sum;
        unsigned int count;
        
        // inclusion exclusion
        sum = workTotal[max_y*raw->cols+max_x];
        count = workNum[max_y*raw->cols+max_x];
        if (least_x>=colStart) {
          sum -= workTotal[max_y*raw->cols+least_x];
          count -= workNum[max_y*raw->cols+least_x];
        }
        if (least_y>=rowStart){
          sum  -= workTotal[least_y*raw->cols+max_x];
          count -=  workNum[least_y*raw->cols+max_x];
        }
        if ((least_x>=colStart) & (least_y>=rowStart))
        {
          sum += workTotal[least_y*raw->cols+least_x];
          count += workNum[least_y*raw->cols+least_x];
        }

        float fsum = typical_value;
        if ( count > 0 )
        {
          fsum = (float) sum;
          fsum /= (float)count;
        }

        smooth_max_amplitude[wellIx] = fsum;
      
    }
  }
}


// make a smooth estimate of the amplitude of the pH step in the image across the region of interest...using
// only empty wells.  This is used to 'normalize' the pH step ampltidue across the region in all wells and
// can correct for bulk blow-by accumulation across a BkgModel analysis region
//@TODO:  Note that this is fictitious
//@TODO: blow-by is not multiplicative
//@TODO: estimating max value is not stable to outliers
//@TODO: values can go negative without warning
void Image::CalculateEmptyWellLocalScaleForFlow ( PinnedInFlow& pinnedInFlow,Mask *bfmask,int flow,MaskType referenceMask, int smooth_span )
{
  if ( smooth_max_amplitude == NULL )
    smooth_max_amplitude = new float[raw->rows*raw->cols];

  memset ( smooth_max_amplitude,0,sizeof ( float[raw->rows*raw->cols] ) );

  // need a 1-frame scratch-pad for doing NN calc
  short *tmp = new short [raw->rows*raw->cols];
  memset ( tmp,0,sizeof ( short [raw->rows*raw->cols] ) );
  short *pattern_flag = new short[raw->rows*raw->cols];
  memset ( pattern_flag,0,sizeof ( short [raw->rows*raw->cols] ) );
  
  FindCharacteristicValuesInReferenceWells(tmp,pattern_flag,referenceMask, bfmask,pinnedInFlow,flow);

  //SlowSmoothPattern(tmp,pattern_flag, smooth_span);

  FastSmoothPattern(tmp,pattern_flag, smooth_span);
  
  delete [] pattern_flag;
  delete [] tmp;
}

float Image::GetEmptyWellAmplitudeRegionAverage ( Region *region )
{
  float ew_avg = 0.0f;
  int ew_sum = 0;

  assert ( smooth_max_amplitude != NULL );

  // calculate the average within the region
  for ( int ay=region->row;ay < ( region->row+region->h );ay++ )
  {
    for ( int ax=region->col;ax < ( region->col+region->w );ax++ )
    {
      if ( smooth_max_amplitude[ax+ay*raw->cols] > 0.0f )
      {
        ew_avg += smooth_max_amplitude[ax+ay*raw->cols];
        ew_sum++;
      }
    }
  }
  ew_avg /= ew_sum;

  return ( ew_avg );
}

// Use the smooth estimate of the Empty Well peak signal to scale all wells within a region
// such that the empty wells will have the same amplitude across the region
void Image::LocalRescaleRegionByEmptyWells ( Region *region )
{
  // if the smoothed max amplitude hasn't been computed
  // then just return and do no correction
  if ( smooth_max_amplitude == NULL )
    return;

  float ew_avg = GetEmptyWellAmplitudeRegionAverage ( region );

  // now scale everything in the region to the region average
  for ( int ay=region->row;ay < ( region->row+region->h );ay++ )
  {
    for ( int ax=region->col;ax < ( region->col+region->w );ax++ )
    {
      if ( smooth_max_amplitude[ax+ay*raw->cols] > 0.0f )
      {
        float scale = ew_avg/smooth_max_amplitude[ax+ay*raw->cols];
        short *ptrc = raw->image+ay*raw->cols+ax;

        for ( int i=0;i < raw->frames;i++ )
          * ( ptrc+i*raw->frameStride ) = ( short ) ( ( float ) ( * ( ptrc+i*raw->frameStride ) ) *scale+0.5f );
      }
    }
  }
}






/////-----------------------------Dumping routines down here where no-one cares

void Image::DebugDumpResults ( char *fileName, Region region )
{
  if ( !results )
    return;

  int x,y,inx;
  FILE *fp = NULL;
  fopen_s ( &fp, fileName, "w" );
  if ( !fp )
  {
    fprintf ( fp, "%s: %s\n", fileName, strerror ( errno ) );
    return;
  }
  for ( y = region.row; y < ( region.row+region.h ); y++ )
    for ( x = region.col; x < ( region.col+region.w ); x++ )
    {
      inx = x + ( y * raw->cols );
      fprintf ( fp, "%d\n", ( int ) results[inx] );
    }
  fclose ( fp );
  return;
}

void Image::DebugDumpResults ( char *fileName )
{
  if ( !results )
    return;

  int x,y,inx;
  FILE *fp = NULL;
  fopen_s ( &fp, fileName, "w" );
  if ( !fp )
  {
    fprintf ( fp, "%s: %s\n", fileName, strerror ( errno ) );
    return;
  }
  for ( y = 0; y < raw->rows; y++ )
    for ( x = 0; x < raw->cols; x++ )
    {
      inx = x + ( y * raw->cols );
      fprintf ( fp, "%d\n", ( int ) results[inx] );
    }
  fclose ( fp );
  return;
}

//
void Image::DumpTrace ( int r, int c, char *fileName )
{
  FILE *fp = NULL;
  fopen_s ( &fp, fileName, "w" );
  if ( !fp )
  {
    printf ( "%s: %s\n", fileName, strerror ( errno ) );
    return;
  }
  int frameStart = 0;
  int frameEnd = raw->frames;
  int k = ( frameStart * raw->frameStride ) + ( c + ( raw->cols * r ) );

  for ( int frame=frameStart;frame<frameEnd;frame++ )
  {
    //Prints a column
    fprintf ( fp, "%d\n", raw->image[k] );
    //Prints comma delimited row
    //fprintf (fp, "%d", raw->image[k]);
    //if (frame < (frameEnd - 1))
    //  fprintf (fp, ",");
    k += raw->frameStride;
  }
  fprintf ( fp, "\n" );
  fclose ( fp );
}


int Image::DumpDcOffset ( int nSample, string dcOffsetDir, char nucChar, int flowNumber )
{

  // DC-offset using time 0 through noFlowTime milliseconds
  int dcStartFrame = GetFrame ( 0 - GetFlowOffset() );
  int dcEndFrame   = GetFrame ( GetNoFlowTime() - GetFlowOffset() );
  dcStartFrame = std::min ( raw->frames-1,std::max ( dcStartFrame,0 ) );
  dcEndFrame   = std::min ( raw->frames-1,std::max ( dcEndFrame,  0 ) );
  dcEndFrame   = std::max ( dcStartFrame,dcEndFrame );
  float nDcFrame = dcEndFrame-dcStartFrame+1;

  // Init random seed
  int random_seed=0;
  srand ( random_seed );

  // Set sample size
  int nWells = raw->cols * raw->rows;
  nSample = std::min ( nSample,nWells );

  // Get the random sample of wells and compute dc offsets
  vector<int> dcOffset ( nSample );
  for ( int iSample=0; iSample<nSample; iSample++ )
  {
    int maskIndex = rand() % nWells;
    int k = raw->frameStride*dcStartFrame + maskIndex;
    float sum = 0;
    for ( int frame = dcStartFrame; frame <= dcEndFrame; frame++, k += raw->frameStride )
      sum += raw->image[k];
    sum /= nDcFrame;
    dcOffset[iSample] = ( int ) sum;
  }

  // Sort results so we can return percentiles
  sort ( dcOffset.begin(),dcOffset.end() );

  // Open results file for append
  string dcOffsetFileName = dcOffsetDir + string ( "/dcOffset.txt" );
  FILE *dcOffsetFP  = NULL;
  fopen_s ( &dcOffsetFP, dcOffsetFileName.c_str(), "a" );
  if ( !dcOffsetFP )
  {
    printf ( "Could not open/append to %s, err %s\n", dcOffsetFileName.c_str(), strerror ( errno ) );
    return EXIT_FAILURE;
  }

  // Write percentiles and close file
  fprintf ( dcOffsetFP, "%d\t%c", flowNumber, nucChar );
  fprintf ( dcOffsetFP, "\t%d\t%d", dcOffset.front(),dcOffset.back() ); // min and max
  float nQuantiles=100;
  float jump = dcOffset.size() / nQuantiles;
  for ( int i=1; i<nQuantiles; i++ )
    fprintf ( dcOffsetFP, "\t%d", dcOffset[floor ( i*jump ) ] );
  fprintf ( dcOffsetFP, "\n" );
  fclose ( dcOffsetFP );

  return ( EXIT_SUCCESS );
}

double Image::DumpStep ( int c, int r, int w, int h, string regionName, char nucChar, string nucStepDir, Mask *mask, PinnedInFlow *pinnedInFlow, int flowNumber )
{
  // make sure user calls us with sane bounds
  if ( w < 1 ) w = 1;
  if ( h < 1 ) h = 1;
  if ( r < 0 ) r = 0;
  if ( c < 0 ) c = 0;
  if ( r >= raw->rows ) r = raw->rows-1;
  if ( c >= raw->cols ) c = raw->cols-1;
  if ( r+h > raw->rows ) h = raw->rows - r;
  if ( c+w > raw->cols ) w = raw->cols - c;
  if ( h < 1 || w < 1 ) // ok, nothing left to compute?
    return 0.0;

  // DC-offset using time 0 through noFlowTime milliseconds
  int dcStartFrame = GetFrame ( 0 - GetFlowOffset() );
  int dcEndFrame   = GetFrame ( GetNoFlowTime() - GetFlowOffset() );
  dcStartFrame = std::min ( raw->frames-1,std::max ( dcStartFrame,0 ) );
  dcEndFrame   = std::min ( raw->frames-1,std::max ( dcEndFrame,  0 ) );
  dcEndFrame   = std::max ( dcStartFrame,dcEndFrame );

  string baseName = nucStepDir + string ( "/NucStep_" ) + regionName;
  string nucStepSizeFileName    = baseName + string ( "_step.txt" );
  string nucStepBeadFileName    = baseName + string ( "_bead.txt" );
  string nucStepEmptyFileName   = baseName + string ( "_empty.txt" );
  string nucStepEmptySdFileName = baseName + string ( "_empty_sd.txt" );

  string frameTimeFileName = nucStepDir + string ( "/NucStep_frametime.txt" );
  FILE *fpFrameTime  = NULL;
  fopen_s ( &fpFrameTime, frameTimeFileName.c_str(), "w" );
  if ( !fpFrameTime )
  {
    printf ( "Could not open to %s, err %s\n", frameTimeFileName.c_str(), strerror ( errno ) );
    return 0.0;
  }
  // this function appends data per image asynchronously loaded from multiple
  // threads to output files.  Each append must be atomic, so lock the output
  // using: flockfile(FILE *); ... output steps ...; funlockfile(FILE *);

  if ( fpFrameTime )
  {
    flockfile ( fpFrameTime );
    fprintf ( fpFrameTime, "0" );
    for ( int iFrame=1; iFrame<raw->frames; iFrame++ )
    {
      fprintf ( fpFrameTime, "\t%.3f", raw->timestamps[iFrame-1] / FRAME_SCALE );
    }
    fprintf ( fpFrameTime, "\n" );
    funlockfile ( fpFrameTime );
    fclose ( fpFrameTime );
  }

  FILE *fpSize  = NULL;
  fopen_s ( &fpSize, nucStepSizeFileName.c_str(), "a" );
  if ( !fpSize )
  {
    printf ( "Could not open/append to %s, err %s\n", nucStepSizeFileName.c_str(), strerror ( errno ) );
    return 0.0;
  }

  FILE *fpBead  = NULL;
  fopen_s ( &fpBead, nucStepBeadFileName.c_str(), "a" );
  if ( !fpBead )
  {
    printf ( "Could not open/append to %s, err %s\n", nucStepBeadFileName.c_str(), strerror ( errno ) );
    return 0.0;
  }

  FILE *fpEmpty = NULL;
  fopen_s ( &fpEmpty, nucStepEmptyFileName.c_str(), "a" );
  if ( !fpEmpty )
  {
    printf ( "Could not open/append to %s, err %s\n", nucStepEmptyFileName.c_str(), strerror ( errno ) );
    return 0.0;
  }

  FILE *fpEmptySd = NULL;
  fopen_s ( &fpEmptySd, nucStepEmptySdFileName.c_str(), "a" );
  if ( !fpEmptySd )
  {
    printf ( "Could not open/append to %s, err %s\n", nucStepEmptySdFileName.c_str(), strerror ( errno ) );
    return 0.0;
  }

  vector<float> frameWeight ( raw->frames,0 );
  float weightSum=0;
  for ( int frame = dcStartFrame; frame <= dcEndFrame; frame++ )
  {
    float thisWeight = raw->timestamps[frame] - ( ( frame > 0 ) ? raw->timestamps[frame-1] : 0 );
    frameWeight[frame] = thisWeight;
    weightSum += thisWeight;
  }
  for ( int frame = dcStartFrame; frame <= dcEndFrame; frame++ )
  {
    frameWeight[frame] /= weightSum;
  }

  unsigned int nBead=0;
  unsigned int nEmpty=0;
  vector<float> valBead ( raw->frames,0 );
  vector<float> valEmpty ( raw->frames,0 );
  vector<float> valEmptySd ( raw->frames,0 );
  for ( int y=r;y< ( r+h );y++ )
  {
    int maskIndex = raw->cols * y + c;
    for ( int x=c;x< ( c+w );x++, maskIndex++ )
    {
      bool unPinned = ! ( pinnedInFlow->IsPinned ( flowNumber, maskIndex ) );
      bool notBad   = unPinned && ( !mask->Match ( maskIndex, ( MaskType ) ( MaskExclude | MaskIgnore ) ) );
      bool isBead   = notBad && mask->Match ( maskIndex, ( MaskType ) MaskBead );
      bool isEmpty  = notBad && mask->Match ( maskIndex, ( MaskType ) ( MaskEmpty | MaskReference ) );
      if ( isBead || isEmpty )
      {
        // First subtract dc offsets from the bead
        int k = raw->frameStride*dcStartFrame + maskIndex;
        float sum = 0;
        for ( int frame = dcStartFrame; frame <= dcEndFrame; frame++, k += raw->frameStride )
          sum += raw->image[k] * frameWeight[frame];

        if ( isBead )
          nBead++;
        else
          nEmpty++;
        k = maskIndex;
        for ( int frame = 0; frame < raw->frames; frame++, k += raw->frameStride )
        {
          float thisVal = ( float ) raw->image[k] - sum;
          if ( isBead )
            valBead[frame] += thisVal;
          else
          {
            valEmpty[frame] += thisVal;
            valEmptySd[frame] += thisVal*thisVal;
          }
        }
      }
    }
  }

  if ( nBead > 0 )
  {
    for ( int frame=0; frame < raw->frames; frame++ )
    {
      valBead[frame] /= ( float ) nBead;
    }
  }
  if ( nEmpty > 0 )
  {
    for ( int frame=0; frame < raw->frames; frame++ )
    {
      valEmpty[frame] /= ( float ) nEmpty;
    }
    if ( nEmpty > 1 )
    {
      for ( int frame=0; frame < raw->frames; frame++ )
      {
        valEmptySd[frame] /= ( float ) nEmpty;
        valEmptySd[frame] -= valEmpty[frame]*valEmpty[frame];
        valEmptySd[frame] *= ( ( float ) nEmpty ) / ( ( float ) ( nEmpty-1 ) );
        valEmptySd[frame] = sqrt ( valEmptySd[frame] );
      }
    }
    else
    {
      for ( int frame=0; frame < raw->frames; frame++ )
      {
        valEmpty[frame] = 0;
      }
    }
  }

  // Compute step size in empty wells
  double minVal = 0.0;
  double maxVal = 0.0;
  if ( nEmpty > 0 )
  {
    minVal = maxVal = valEmpty[0];
    for ( int frame=1; frame < raw->frames; frame++ )
    {
      double thisVal = valEmpty[frame];
      if ( thisVal < minVal )
      {
        minVal = thisVal;
      }
      if ( thisVal > maxVal )
      {
        maxVal = thisVal;
      }
    }
  }
  double stepSize = maxVal-minVal;

  if ( fpSize )
  {
    flockfile ( fpSize );
    fprintf ( fpSize, "%d\t%c\t%.2lf\n", flowNumber, nucChar, stepSize );
    funlockfile ( fpSize );
    fclose ( fpSize );
  }
  if ( fpBead )
  {
    flockfile ( fpBead );
    fprintf ( fpBead, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nBead );
    for ( int frame = 0; frame < raw->frames; frame++ )
    {
      fprintf ( fpBead, "\t%.3f", valBead[frame] );
    }
    fprintf ( fpBead, "\n" );
    funlockfile ( fpBead );
    fclose ( fpBead );
  }
  if ( fpEmpty )
  {
    flockfile ( fpEmpty );
    fprintf ( fpEmpty, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nEmpty );
    for ( int frame = 0; frame < raw->frames; frame++ )
    {
      fprintf ( fpEmpty, "\t%.3f", valEmpty[frame] );
    }
    fprintf ( fpEmpty, "\n" );
    funlockfile ( fpEmpty );
    fclose ( fpEmpty );
  }
  if ( fpEmptySd )
  {
    flockfile ( fpEmptySd );
    fprintf ( fpEmptySd, "%d\t%c\t%d\t%d\t%d\t%d\t%d", flowNumber, nucChar, c, c+w, r, r+h, nEmpty );
    for ( int frame = 0; frame < raw->frames; frame++ )
    {
      fprintf ( fpEmptySd, "\t%.3f", valEmptySd[frame] );
    }
    fprintf ( fpEmptySd, "\n" );
    funlockfile ( fpEmptySd );
    fclose ( fpEmptySd );
  }

  return stepSize;
}

////-----------------torrentR interface

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
  if ( col.size() > 0 || row.size() > 0 )
  {
    if ( row.size() != col.size() )
    {
      cerr << "number of requested rows and columns should be the same" << endl;
      return ( false );
    }
    nSavedWells = col.size();
    cherryPickWells=true;
    // Figure out min & max cols to load less data
    minCol = col[0];
    maxCol = minCol+1;
    minRow = row[0];
    maxRow = minRow+1;
    for ( unsigned int iWell=1; iWell < col.size(); iWell++ )
    {
      if ( col[iWell] < ( unsigned int ) minCol )
        minCol = col[iWell];
      if ( col[iWell] >= ( unsigned int ) maxCol )
        maxCol = 1+col[iWell];
      if ( row[iWell] < ( unsigned int ) minRow )
        minRow = row[iWell];
      if ( row[iWell] >= ( unsigned int ) maxRow )
        maxRow = 1+row[iWell];
    }
  }

  // Read header to determine full-chip rows & cols
  if ( !LoadRaw ( rawFileName[0].c_str(), 0, true, true ) )
  {
    cerr << "Problem loading dat file " << rawFileName[0] << endl;
    return ( false );
  }
  nColFull = raw->cols;
  nRowFull = raw->rows;

  // This next block of ugliness will not be needed when we encode the chip type in the dat file
  char *chipID = NULL;
  if ( chipType != "" )
  {
    chipID = strdup ( chipType.c_str() );
  }
  else
  {
    if ( nColFull == 1280 && nRowFull == 1152 )
    {
      chipID = strdup ( "314" );
    }
    else if ( nColFull == 2736 && nRowFull == 2640 )
    {
      chipID = strdup ( "316" );
    }
    else if (nColFull == 3392 && nRowFull ==2120 )
    {
        chipID =strdup ( "316v2");
    }
    else if ( nColFull == 3392 && nRowFull == 3792 )
    {
      chipID = strdup ( "318" );
    }
    else if ( nColFull == 7680 && nRowFull == 5328 )
    {
      chipID = strdup ( "p1.0.19" ); // this is for P1.0 but will not work for a thumb nail.
    }
    else if ( nColFull == 7728 && nRowFull == 5328 )
    {
      chipID = strdup ( "p1.0.20" ); // this is for P1.0 but will not work for a thumb nail.
    }
    else
    {
      ION_WARN ( "Unable to determine chip type from dimensions" );
    }
  }

  // Only allow for XTCorrection on 316, 318 and P1.0 chips
  if ( XTCorrect )
  {
    if ( ( chipID == NULL ) || ( strcmp ( chipID,"318" ) && strcmp ( chipID,"316" ) && strcmp ( chipID,"316v2" ) && strcmp(chipID,"p1.0.19")) )
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
  if ( minCol > -1 || minRow > -1 || maxCol > -1 || maxRow > -1 )
  {
    // First do a few boundary checks
    bool badBoundary=false;
    if ( minCol >= ( int ) nColFull )
    {
      cerr << "Error in Image::LoadSlice() - minCol is " << minCol << " which should be less than nColFull which is " << nColFull << endl;
      badBoundary=true;
    }
    if ( maxCol <= 0 )
    {
      cerr << "Error in Image::LoadSlice() - maxCol is " << maxCol << " which is less than 1" << endl;
      badBoundary=true;
    }
    if ( minCol >= maxCol )
    {
      cerr << "Error in Image::LoadSlice() - maxCol is " << maxCol << " which is not greater than minCol which is " << minCol << endl;
      badBoundary=true;
    }
    if ( minRow >= ( int ) nRowFull )
    {
      cerr << "Error in Image::LoadSlice() - minRow is " << minRow << " which should be less than nRowFull which is " << nRowFull << endl;
      badBoundary=true;
    }
    if ( maxRow <= 0 )
    {
      cerr << "Error in Image::LoadSlice() - maxRow is " << maxRow << " which is less than 1" << endl;
      badBoundary=true;
    }
    if ( minRow >= maxRow )
    {
      cerr << "Error in Image::LoadSlice() - maxRow is " << maxRow << " which is not greater than minRow which is " << minRow << endl;
      badBoundary=true;
    }
    if ( badBoundary )
      return ( false );
    // Expand boundaries as necessary for XTCorrection
    minColOuter = minCol;
    maxColOuter = maxCol;
    minRowOuter = minRow;
    maxRowOuter = maxRow;
    if ( XTCorrect )
    {
      minColOuter = std::max ( 0, minCol + ImageTransformer::chan_xt_column_offset[0] );
      maxColOuter = std::min ( ( int ) nColFull, maxCol + ImageTransformer::chan_xt_column_offset[DEFAULT_VECT_LEN-1] );
    }
  }
  else
  {
    if ( minCol < 0 )
      minCol = 0;
    if ( minRow < 0 )
      minRow = 0;
    if ( maxCol < 0 )
      maxCol = nColFull;
    if ( maxRow < 0 )
      maxRow = nRowFull;
    minColOuter = minCol;
    maxColOuter = maxCol;
    minRowOuter = minRow;
    maxRowOuter = maxRow;
  }
  // Set up Image class to read only the sub-range
  ImageCropping::chipSubRegion.col = minColOuter;
  ImageCropping::chipSubRegion.row = minRowOuter;
  ImageCropping::chipSubRegion.w   = maxColOuter-minColOuter;
  ImageCropping::chipSubRegion.h   = maxRowOuter-minRowOuter;
  // Set region origin for proper XTCorrection
  ImageCropping::SetCroppedRegionOrigin ( minColOuter,minRowOuter );

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
  for ( unsigned int iDat=0; iDat < nDat; iDat++ )
  {
    if ( !LoadRaw ( rawFileName[iDat].c_str(), 0, true, false ) )
    {
      cerr << "Problem loading dat file " << rawFileName[iDat] << endl;
      return ( false );
    }
    unsigned int nCol = raw->cols;
    unsigned int nRow = raw->rows;

    // We normalize if asked
    if ( doNormalize )
    {
      SetMeanOfFramesToZero ( normStart,normEnd );
    }

    // cross-channel correction
    if ( XTCorrect )
    {
      // Mask tempMask (raw->cols, raw->rows);
      if ( chipID != NULL )
      {
        ChipIdDecoder::SetGlobalChipId ( chipID );
        const char *rawDir = dirname ( ( char * ) rawFileName[iDat].c_str() );
        char lsrow[1024];
        sprintf(lsrow,"lsrowimage.%s",datPostfix);
        ImageTransformer::CalibrateChannelXTCorrection ( rawDir,lsrow );
        // XTChannelCorrect (&tempMask);
        ImageTransformer::XTChannelCorrect ( raw,results_folder );
      }
    }

    unsigned int nLoadedWells = nRow * nCol;
    unsigned int nFramesCompressed = raw->frames;
    unsigned int nFramesUncompressed = raw->uncompFrames;
    if ( iDat==0 )
    {
      nPrevCol=nCol;
      nPrevRow=nRow;
      nPrevFramesCompressed=nFramesCompressed;
      nPrevFramesUncompressed=nFramesUncompressed;
      // Size the return objects depending on whether or not we're returning compressed data
      if ( uncompress )
      {
        nFrame = nFramesUncompressed;
      }
      else
      {
        nFrame = nFramesCompressed;
      }
      frameStart.resize ( nDat );
      frameEnd.resize ( nDat );
      for ( unsigned int jDat=0; jDat<nDat; jDat++ )
      {
        frameStart[jDat].resize ( nFrame );
        frameEnd[jDat].resize ( nFrame );
      }
      if ( !cherryPickWells )
      {
        // If not using a specified set of row,col coordinates then return a rectangular region
        nSavedWells = ( maxCol-minCol ) * ( maxRow-minRow );
      }
      if ( returnSignal )
      {
        signal.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          signal[jDat].resize ( nSavedWells );
          // Will resize for number of frames to return later, when that has been determined
        }
      }
      if ( returnMean )
      {
        mean.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          mean[jDat].resize ( nSavedWells );
        }
      }
      if ( returnSD )
      {
        sd.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          sd[jDat].resize ( nSavedWells );
        }
      }
      if ( returnLag )
      {
        lag.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          lag[jDat].resize ( nSavedWells );
        }
      }
      if ( returnMean || returnSD )
      {
        signalStats.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          signalStats[jDat].resize ( nSavedWells );
        }
      }
      if ( returnLag )
      {
        lagStats.resize ( nDat );
        for ( unsigned int jDat=0; jDat<nDat; jDat++ )
        {
          lagStats[jDat].resize ( nSavedWells );
        }
      }
    }
    else
    {
      if ( nCol != nPrevCol )
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of cols to first dat file " << rawFileName[0] << endl;
        return ( false );
      }
      if ( nRow != nPrevRow )
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of rows to first dat file " << rawFileName[0] << endl;
        return ( false );
      }
      if ( nFramesCompressed != nPrevFramesCompressed )
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of compressed frames to first dat file " << rawFileName[0] << endl;
        return ( false );
      }
      if ( nFramesUncompressed != nPrevFramesUncompressed )
      {
        cerr << "Dat file " << rawFileName[iDat] << " has different number of uncompressed frames to first dat file " << rawFileName[0] << endl;
        return ( false );
      }
    }

    // Determine well offsets to collect
    if ( iDat==0 )
    {
      wellIndex.resize ( nSavedWells );
      colOut.resize ( nSavedWells );
      rowOut.resize ( nSavedWells );
      if ( cherryPickWells )
      {
        for ( unsigned int iWell=0; iWell < nSavedWells; iWell++ )
        {
          wellIndex[iWell] = ( row[iWell]-minRowOuter ) *nCol + ( col[iWell]-minColOuter );
          colOut[iWell] = col[iWell];
          rowOut[iWell] = row[iWell];
        }
      }
      else
      {
        for ( unsigned int iWell=0, iRow=minRow; iRow < ( unsigned int ) maxRow; iRow++ )
        {
          for ( unsigned int iCol=minCol; iCol < ( unsigned int ) maxCol; iCol++, iWell++ )
          {
            wellIndex[iWell] = ( iRow-minRowOuter ) *nCol + ( iCol-minColOuter );
            colOut[iWell] = iCol;
            rowOut[iWell] = iRow;
          }
        }
      }
    }

    // Check if we need to uncompress
    // Make rawTimestamps point to the timestamps we need to report
    int *rawTimestamps = NULL;
    if ( iDat==0 && uncompress && ( nFramesUncompressed != nFramesCompressed ) )
    {
      int baseTime = raw->timestamps[0];
      uncompressedTimestamps = new int[sizeof ( int ) * nFramesUncompressed];
      uncompressedTimestamps[0] = 0;
      for ( int iFrame=1; iFrame < raw->uncompFrames; iFrame++ )
        uncompressedTimestamps[iFrame] = baseTime+uncompressedTimestamps[iFrame-1];
      rawTimestamps = uncompressedTimestamps;
    }
    else
    {
      rawTimestamps = raw->timestamps;
    }

    // Determine timeframe starts & stops
    bool oldMode = ( rawTimestamps[0]==0 );
    for ( unsigned int iFrame=0; iFrame<nFrame; iFrame++ )
    {
      if ( oldMode )
      {
        frameStart[iDat][iFrame] = rawTimestamps[iFrame] / FRAME_SCALE;
        if ( iFrame < nFrame-1 )
          frameEnd[iDat][iFrame] = ( ( double ) rawTimestamps[iFrame+1] ) / FRAME_SCALE;
        else if ( iFrame > 0 )
          frameEnd[iDat][iFrame] = ( 2.0 * ( double ) rawTimestamps[iFrame] - ( double ) rawTimestamps[iFrame-1] ) / FRAME_SCALE;
        else
          frameEnd[iDat][iFrame] = 0;
      }
      else
      {
        frameEnd[iDat][iFrame] = ( ( double ) rawTimestamps[iFrame] ) / FRAME_SCALE;
        frameStart[iDat][iFrame] = ( ( double ) ( ( iFrame > 0 ) ? rawTimestamps[iFrame-1] : 0 ) ) / FRAME_SCALE;
      }
    }

    // Determing per-well baseline values to subtract
    bool doBaseline=false;
    int baselineMinFrame = -1;
    int baselineMaxFrame = -1;
    std::vector<double> baselineWeight;
    if ( baselineMinTime < baselineMaxTime )
    {
      double baselineWeightSum = 0;
      for ( unsigned int iFrame=0; iFrame < nFrame; iFrame++ )
      {
        if ( ( frameStart[iDat][iFrame] > ( baselineMinTime-numeric_limits<double>::epsilon() ) ) && frameEnd[iDat][iFrame] < ( baselineMaxTime+numeric_limits<double>::epsilon() ) )
        {
          // frame is in our baseline timeframe
          if ( baselineMinFrame < 0 )
            baselineMinFrame = iFrame;
          baselineMaxFrame = iFrame;
          baselineWeight.push_back ( frameEnd[iDat][iFrame]-frameStart[iDat][iFrame] );
          baselineWeightSum += frameEnd[iDat][iFrame]-frameStart[iDat][iFrame];
        }
      }
      if ( baselineWeightSum > 0 )
      {
        unsigned int nBaselineFrame = baselineWeight.size();
        for ( unsigned int iFrame=0; iFrame < nBaselineFrame; iFrame++ )
          baselineWeight[iFrame] /= baselineWeightSum;
        doBaseline=true;
      }
    }
    vector<double> baselineVal;
    short bVal=0;
    if ( doBaseline )
    {
      baselineVal.resize ( nSavedWells,0 );
      for ( unsigned int iWell=0; iWell < nSavedWells; iWell++ )
      {
        for ( int iFrame=baselineMinFrame; iFrame <= baselineMaxFrame; iFrame++ )
        {
          if ( uncompress )
            bVal = GetInterpolatedValue ( iFrame, colOut[iWell]-minColOuter, rowOut[iWell]-minRowOuter );
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
    if ( loadMinTime < loadMaxTime )
    {
      loadMinFrame = -1;
      for ( unsigned int iFrame=0; iFrame < nFrame; iFrame++ )
      {
        if ( ( frameStart[iDat][iFrame] > ( loadMinTime-numeric_limits<double>::epsilon() ) ) && frameEnd[iDat][iFrame] < ( loadMaxTime+numeric_limits<double>::epsilon() ) )
        {
          if ( loadMinFrame < 0 )
            loadMinFrame = iFrame;
          loadMaxFrame = iFrame+1;
        }
      }
      if ( loadMinFrame == -1 )
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
    if ( returnSignal )
    {
      for ( unsigned int iWell=0; iWell < nSavedWells; iWell++ )
        signal[iDat][iWell].resize ( nLoadFrame );
    }
    // subset frameStart/frameEnd for frame range requested
    if ( loadFrameSubset )
    {
      if ( loadMaxFrame < ( int ) nFrame )
      {
        unsigned int nToDrop = nFrame - loadMaxFrame;
        frameStart[iDat].erase ( frameStart[iDat].end()-nToDrop,frameStart[iDat].end() );
        frameEnd[iDat].erase ( frameEnd[iDat].end()-nToDrop,frameEnd[iDat].end() );
      }
      if ( loadMinFrame > 0 )
      {
        unsigned int nToDrop = loadMinFrame+1;
        frameStart[iDat].erase ( frameStart[iDat].begin(),frameStart[iDat].begin() +nToDrop );
        frameEnd[iDat].erase ( frameEnd[iDat].begin(),frameEnd[iDat].begin() +nToDrop );
      }
    }

    short val=0;
    for ( unsigned int iWell=0; iWell < nSavedWells; iWell++ )
    {
      // do frames
      short oldval =0;
      for ( int iFrame=loadMinFrame; iFrame < loadMaxFrame; iFrame++ )
      {
        if ( uncompress )
          val = GetInterpolatedValue ( iFrame, colOut[iWell]-minColOuter, rowOut[iWell]-minRowOuter );
        else
          val = raw->image[iFrame * nLoadedWells + wellIndex[iWell]];
        if ( doBaseline )
        {
          val = ( short ) ( ( double ) val - baselineVal[iWell] );
        }
        if ( returnSignal )
          signal[iDat][iWell][iFrame-loadMinFrame] = val;
        if ( returnMean || returnSD )
          signalStats[iDat][iWell].AddValue ( ( float ) val );
        if ( returnLag & ( iFrame>loadMinFrame ) )
          lagStats[iDat][iWell].AddValue ( ( float ) ( val-oldval ) );
        oldval = val;
      }
    }
    for ( unsigned int iWell=0; iWell < nSavedWells; iWell++ )
    {
      if ( returnMean )
      {
        mean[iDat][iWell] = ( short ) ( signalStats[iDat][iWell].GetMean() );
      }
      if ( returnSD )
      {
        sd[iDat][iWell] = ( short ) ( signalStats[iDat][iWell].GetSD() );
      }
      if ( returnLag )
      {
        lag[iDat][iWell] = ( short ) ( lagStats[iDat][iWell].GetSD() ); // sd on lagged signal
      }
    }

    cleanupRaw();
  }

  if ( chipID != NULL )
    free ( chipID );

  if ( uncompressedTimestamps != NULL )
    delete [] uncompressedTimestamps;

  if ( problem )
    return ( false );
  else
    return ( true );

};
