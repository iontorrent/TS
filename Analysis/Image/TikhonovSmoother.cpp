/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
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

#include "Utils.h"
#include "deInterlace.h"
#include "LinuxCompat.h"
#include "ChipIdDecoder.h"
#include "LSRowImageProcessor.h"

#include "dbgmem.h"
#include "IonErr.h"

#include "TikhonovSmoother.h"

using namespace std;


#define TIK_USE_INTS 1

// this routine takes the smoothing parameters and stretches them to fit the actual number of frames
// in the data to be smoothed.  This is done once per file, but doesn't take long.
// imatrix is stretched by adding copies of the middle row to fill out the difference in frames between the
// pre-computed matrix and the actual frame data.
// This stretching means we don't have to have pre-computed matricies sitting around for all possible frame
// lengths. (APB)

void TikhonovSmoother::StretchMatrix ( int frame_cnt, int i_low[], int i_high[], int *imatrix[] )
{
  // allocate space for imatrix
  imatrix[0] = ( int * ) malloc ( sizeof ( int ) * frame_cnt * frame_cnt );
  if ( imatrix[0] == NULL ) {
    fprintf ( stdout, "Malloc for imatrix failed in StretchMatrix in TikhonovSmoother.cpp!!!\n" );
    doSmoothing = false;  // stop any further smoothing activity;
    return;
  }
  {// init imatrix[]
    int i,j;
    for ( i=1; i<frame_cnt; i++ ) {
      imatrix[i] = imatrix[0] + ( i*frame_cnt );
    }
    for ( i=0; i<frame_cnt; i++ ) {
      for ( j=0; j<frame_cnt; j++ ) {
        imatrix[i][j] = 0;
      }
    }
  }
  // figure out the bandwidth
  if ( frame_cnt >= smoothParams.frame_cnt ) { // have to add rows, and/or just copy data
    int i, j, offset;
    // we copy everything from 0 to bottom and from top to frame_cnt-1 from smoothParams
    // the middle we copy from frame_cnt/2
    int bottom = ( smoothParams.frame_cnt/2 ) - 1;
   // int top = smoothParams.frame_cnt - ( smoothParams.frame_cnt/2 );
    int excess = frame_cnt - smoothParams.frame_cnt;

    for ( i=0; i<=bottom; i++ ) {
      i_low[i]  = smoothParams.i_low[i];
      i_high[i] = smoothParams.i_high[i];
      for ( j=smoothParams.i_low[i]; j<=smoothParams.i_high[i]; j++ ) {
        imatrix[i][j] = smoothParams.imatrix[i][j];
      }
    }
    // now the middle
    for ( i=bottom+1, offset=0; i<bottom+1+excess; i++, offset++ ) {
      i_low[i]  = smoothParams.i_low[bottom+1] + offset;
      i_high[i] = smoothParams.i_high[bottom+1] + offset;
      for ( j=i_low[i]; j<=i_high[i]; j++ ) {
        imatrix[i][j] = smoothParams.imatrix[bottom+1][j-offset];
      }
    }
    // now the top
    for ( i=bottom+1+excess; i<frame_cnt; i++ ) {
      i_low[i]  = smoothParams.i_low[i-excess] + excess;
      i_high[i] = smoothParams.i_high[i-excess] + excess;
      for ( j=i_low[i]; j<=i_high[i]; j++ ) {
        imatrix[i][j] = smoothParams.imatrix[i-excess][j-excess];
      }
    }
  }
}


// The actual smoothing code.  image is smoothed in-place so everything following sees
// a smoothed version of the data
// integer version
void TikhonovSmoother::SmoothTrace ( short *image, int rows, int cols, int frame_cnt, const char *rawFileName )
{
  int i,r,c,stride,frame,itmp,well_offset,rounder;
  float ftime;
  int iftmp[TIK_MAX_FRAME_CNT];
  struct timeval start, stop, diff;
  int i_low[TIK_MAX_FRAME_CNT], i_high[TIK_MAX_FRAME_CNT]; // local stretched copies
  int *imatrix[TIK_MAX_FRAME_CNT];                         // need local copies to be thread-safe


  gettimeofday ( &start, NULL );
  if ( !doSmoothing ) { //Abort
    fprintf ( stdout, "doSmoothing is false ! Aborting smoothing\n" );
    return;
  }
  StretchMatrix ( frame_cnt, i_low, i_high, imatrix );
  fprintf ( stdout, "stretch matrix from %d to %d frames. Done.\n",smoothParams.frame_cnt, frame_cnt );

  stride = rows * cols;
  rounder = smoothParams.denominator/2;
  for ( r=0; r<rows; r++ ) {
    for ( c=0; c<cols; c++ ) {
      well_offset = ( c+ ( r*cols ) );
      for ( frame=0; frame<frame_cnt; frame++ ) {
        iftmp[frame] = 0.0;
        itmp = well_offset + ( i_low[frame] * stride );
        for ( i= i_low[frame]; i<= i_high[frame]; i++, itmp += stride ) {
          iftmp[frame] += image[itmp] * imatrix[frame][i];
        }
      }
      itmp = well_offset;
      for ( frame=0; frame<frame_cnt; frame++, itmp += stride ) {
        // maybe add some value-bounding code?
        image[itmp] = ( short ) ( ( ( iftmp[frame] + rounder ) /smoothParams.denominator ) & 0x3FFF );
        // +rounder is to round properly. 0x3FFF is to keep only lower 14 bits
      }
    }
  }
  gettimeofday ( &stop, NULL );
  timersub ( &stop, &start, &diff );
  ftime = ( float ) ( diff.tv_sec ) + ( ( float ) ( diff.tv_usec ) /1000000.0 );
  fprintf ( stdout, "SMOOTHING %f sec rows=%d cols=%d fc=%d - %s\n", ftime,
            rows, cols,frame_cnt,rawFileName );
}

// links the internal smoothing parameters to the desired set of static constants
void ParseInternalOpt ( const char *smoothingInternal, int *int_frame_cnt, int *int_denominator, int **int_i_low,
                        int **int_i_high, int **int_imatrix )
{
  int i;
  if ( 0 == strcmp ( smoothingInternal, "05" ) ) {
    *int_frame_cnt = tik05_frame_cnt;
    *int_denominator = tik05_denominator;
    *int_i_low = &tik05_i_low[0];
    *int_i_high = &tik05_i_high[0];
    for ( i=0; i<*int_frame_cnt; i++ ) {
      int_imatrix[i] = &tik05_imatrix[0][0] + ( i* ( *int_frame_cnt ) );
    };
    return;
  }
  if ( 0 == strcmp ( smoothingInternal, "075" ) ) {
    *int_frame_cnt = tik075_frame_cnt;
    *int_denominator = tik075_denominator;
    *int_i_low = &tik075_i_low[0];
    *int_i_high = &tik075_i_high[0];
    for ( i=0; i<*int_frame_cnt; i++ ) {
      int_imatrix[i] = &tik075_imatrix[0][0] + ( i* ( *int_frame_cnt ) );
    };
    return;
  }
  if ( 0 == strcmp ( smoothingInternal, "10" ) ) {
    *int_frame_cnt = tik10_frame_cnt;
    *int_denominator = tik10_denominator;
    *int_i_low = &tik10_i_low[0];
    *int_i_high = &tik10_i_high[0];
    for ( i=0; i<*int_frame_cnt; i++ ) {
      int_imatrix[i] = &tik10_imatrix[0][0] + ( i* ( *int_frame_cnt ) );
    };
    return;
  }
  if ( 0 == strcmp ( smoothingInternal, "15" ) ) {
    *int_frame_cnt = tik15_frame_cnt;
    *int_denominator = tik15_denominator;
    *int_i_low = &tik15_i_low[0];
    *int_i_high = &tik15_i_high[0];
    for ( i=0; i<*int_frame_cnt; i++ ) {
      int_imatrix[i] = &tik15_imatrix[0][0] + ( i* ( *int_frame_cnt ) );
    };
    return;
  }
  if ( 0 == strcmp ( smoothingInternal, "20" ) ) {
    *int_frame_cnt = tik20_frame_cnt;
    *int_denominator = tik20_denominator;
    *int_i_low = &tik20_i_low[0];
    *int_i_high = &tik20_i_high[0];
    for ( i=0; i<*int_frame_cnt; i++ ) {
      int_imatrix[i] = &tik20_imatrix[0][0] + ( i* ( *int_frame_cnt ) );
    };
    return;
  }
  fprintf ( stdout, "INVALID SMOOTHING ARG.  GOING TO EXIT!" );
  exit ( 1 );
}

// This is the TikhonovSmoother constructor, called once in ImageLoader.cpp (APB)
// if smoothingFile is not null, read in the smoothing paramenters from a file
// if smoothingInternal is not null, smoothingInternal specifies a compiled-in set of parameters instead of the hassle
// of providing an external file
// .imatrix is ints scaled by .denominator.  We do it this way to avoid unnecessary floating-point arithmetic
// sets .doSmoothing as signal to actually do the smoothing when loading in files.
TikhonovSmoother::TikhonovSmoother ( const char *datDirectory, const char *smoothingFile, const char *smoothingInternal )
{
  int infd;
  ssize_t bytes_to_read, bytes_read;
  smoothHeader sh;
  int *itmp;
  char smooth_fn[512];
  doSmoothing = false;
  fprintf ( stdout, "smoothing file dir is %s\n", datDirectory );
  if ( smoothingFile[0] != '\000' ) {// read in parameters from file
    if ( smoothingFile[0] == '/' ) { // absolute path.   Don't prepend directory
      sprintf ( smooth_fn, "%s",smoothingFile );
    } else {
      sprintf ( smooth_fn, "%s/%s",datDirectory, smoothingFile );
    }
    // read in smoothing file
    fprintf ( stdout, "Loading smoothing file %s\n", smooth_fn );
    infd = open ( smooth_fn, O_RDONLY );
    if ( infd < 0 ) {// ERROR
      fprintf ( stdout, "Failed to open %s\n", smooth_fn );
      return;
    }
    bytes_to_read = sizeof ( sh );
    bytes_read = read ( infd, &sh, bytes_to_read );
    if ( bytes_read != bytes_to_read ) { //ERROR
      fprintf ( stdout, "bytes_read=%lu != bytes_to_read=%lu\n", bytes_read, bytes_to_read );
      return;
    }
    if ( sh.magic != 0xABCDBEEF ) {  // ERROR
      fprintf ( stdout, "Smoothing file read %0X expecting 0XABCD.  Byte order problem in %s ?\n",
                sh.magic, smooth_fn );
      return;
    }

    // malloc and init .imatrix
    itmp = ( int * ) malloc ( sizeof ( int ) * sh.frame_cnt * sh.frame_cnt );
    if ( itmp == NULL ) { // ERROR
      fprintf ( stdout, "smoothParams.imatrix malloc failed\n" );
      return;
    }
    for ( int i=0; i<sh.frame_cnt; i++ ) {
      smoothParams.imatrix[i] = &itmp[i*sh.frame_cnt];
    }
    // read in i_lo and i_high
    bytes_to_read = sh.frame_cnt * sizeof ( int );
    bytes_read = read ( infd, smoothParams.i_low, bytes_to_read );
    bytes_read = read ( infd, smoothParams.i_high, bytes_to_read );
    // read in .matrix
    bytes_to_read = sizeof ( int ) * sh.frame_cnt * sh.frame_cnt;
    bytes_read = read ( infd, itmp, bytes_to_read );
    if ( bytes_read != bytes_to_read ) {
      fprintf ( stdout, "Read from %s failed at matrix\n", smooth_fn );
      return;
    }
    fprintf ( stdout, "Read in matrix\n" );
    close ( infd );
    smoothParams.frame_cnt = sh.frame_cnt;
    smoothParams.denominator = sh.denominator;
    fprintf ( stdout, "Smoothing file %s Done! found %d frames\n", smooth_fn, sh.frame_cnt );
    doSmoothing = true;
    // end of reading in smoothing file
  } else {// check to see if we should use compiled-in parameters
    int int_frame_cnt, int_denominator, *int_i_low, *int_i_high;
    int *int_imatrix[TIK_MAX_FRAME_CNT];

    if ( smoothingInternal[0] != '\000' ) {  // use internal smoothing matrix
      int i,j;
      // this parses smoothingInternal and populates int_frame_cnt &friends for xfer to .smoothingParams
      ParseInternalOpt ( smoothingInternal, &int_frame_cnt, &int_denominator, &int_i_low, &int_i_high, int_imatrix );
      fprintf ( stdout, "populating smoothParams from %s ...\n", smoothingInternal ); // DEBUG

      //DEBUG
      fprintf ( stdout, "int_frame_cnt=%d int_denominator=%d\n",int_frame_cnt, int_denominator );
      for ( i=0; i<int_frame_cnt; i++ ) {
        fprintf ( stdout, "int_i_low[%d]=%d  int_i_high[%d]=%d\n",i,int_i_low[i],i,int_i_high[i] ); // DEBUG
      }

      // malloc and init .imatrix
      itmp = ( int * ) malloc ( sizeof ( int ) * int_frame_cnt * int_frame_cnt );
      if ( itmp == NULL ) { // ERROR
        fprintf ( stdout, "smoothParams.imatrix malloc failed\n" );
        return;
      }
      for ( i=0; i<int_frame_cnt; i++ ) {
        smoothParams.imatrix[i] = &itmp[i*int_frame_cnt];
      }
      // matrix inited.  Now xfer the parameters to .smoothParams
      smoothParams.frame_cnt = int_frame_cnt;
      smoothParams.denominator = int_denominator;
      for ( i=0; i<int_frame_cnt; i++ ) {
        smoothParams.i_low[i] = int_i_low[i];
      }
      for ( i=0; i<int_frame_cnt; i++ ) {
        smoothParams.i_high[i] = int_i_high[i];
      }
      for ( i=0; i<int_frame_cnt; i++ ) {
        fprintf ( stdout,"\ni=%d ",i ); //LOGGING
        for ( j=0; j<int_frame_cnt; j++ ) {
          smoothParams.imatrix[i][j] = int_imatrix[i][j];
          fprintf ( stdout, "%6d ",int_imatrix[i][j] ); // LOGGING
          if ( j%10==0 ) {
            fprintf ( stdout, "\n" );  // LOGGING
          }
        }
      }
      fprintf ( stdout, "\npopulating from %s Done\n", smoothingInternal ); // DEBUG
      doSmoothing = true;
    }
  }
  // looks like we got here with no obvious errors.
  return;
}
