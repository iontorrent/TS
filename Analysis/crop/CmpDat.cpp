/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/vfs.h>
#endif
#ifdef __APPLE__
#include <sys/uio.h>
#include <sys/mount.h>
#endif
#include <errno.h>
#include <assert.h>
#include "ByteSwapUtils.h"
#include "datahdr.h"
#include "LinuxCompat.h"
// #include "Raw2Wells.h"
#include "Image.h"
#include "crop/Acq.h"
#include "IonVersion.h"
#include "Utils.h"




void usage ( int cropx, int cropy, int cropw, int croph )
{
  fprintf ( stdout, "CmpDat - Utility to compare two dat files.\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   CmpDat /results/analysis/PGM/testRun1 /results/analysis/PGM/testRun2\n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}

int main ( int argc, char *argv[] )
{
  //int cropx = 624, cropy = 125, cropw = 100, croph = 100;
  int cropx = 0, cropy = 0, cropw = 0, croph = 0;
  int i,x,y,idx;
  char *srcFile,*dstFile;
  Image loader_src,loader_dst;
  bool allocate = true;
  RawImage *src_raw,*dst_raw;
	struct timeval tv;
	double startT;
	double stopT;

  if ( argc < 2 ) {
    usage ( cropx, cropy, cropw, croph );
  }

  srcFile = argv[1];
  dstFile = argv[2];

//  int argcc = 1;
//  while ( argcc < argc ) {
//    switch ( argv[argcc][1] ) {
//    default:
//      argcc++;
//      fprintf ( stdout, "\n" );
//
//    }
//    argcc++;
//  }

	gettimeofday ( &tv, NULL );
	startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );

	if(!loader_src.LoadRaw ( srcFile, 0, allocate, false ))
	{
		printf("failed to load file %s\n",srcFile);
		return -1;
	}
    gettimeofday ( &tv, NULL );
    stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
    printf ( "Converted: %s in %0.2lf sec\n", srcFile,stopT - startT );
    fflush ( stdout );
	src_raw = loader_src.raw;

    if(argc == 2)
    {
    	// only one file name given
        printf("rows=%d cols=%d frames_in_file=%d uncomp=%d\n",src_raw->rows,src_raw->cols,src_raw->frames,src_raw->uncompFrames);
        printf("%d frames - Timestamps:\n",src_raw->frames);
        for(int idx=0;idx<src_raw->frames;idx++)
        {
        	printf(" %d", src_raw->timestamps[idx]);
        }
        printf("\n");
        printf("%d frames - InterpolatedFrames:\n",src_raw->frames);
        for(int idx=0;idx<src_raw->frames;idx++)
        {
        	printf(" %d", src_raw->interpolatedFrames[idx]);
        }
        printf("\n");
        printf("%d frames - InterpolatedMult:\n",src_raw->frames);
        for(int idx=0;idx<src_raw->frames;idx++)
        {
        	printf(" %f", src_raw->interpolatedMult[idx]);
        }
        printf("\n");
        return 0;
    }

	gettimeofday ( &tv, NULL );
	startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
	if(!loader_dst.LoadRaw ( dstFile, 0, allocate, false ))
	{
		// spit out info on the loader_src

		printf("failed to load file %s\n",dstFile);
		return 0;
	}
    gettimeofday ( &tv, NULL );
    stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
    printf ( "Converted: %s in %0.2lf sec\n", dstFile,stopT - startT );
    fflush ( stdout );

	// we now have the two files in memory

	dst_raw = loader_dst.raw;

	if ((src_raw->rows != dst_raw->rows) ||
		(src_raw->cols != dst_raw->cols))
	{
		printf("Number of rows and cols is different %d/%d %d/%d\n",src_raw->rows,src_raw->cols,dst_raw->rows,dst_raw->cols);
		return -1;
	}

	if ((src_raw->frames != dst_raw->frames) ||
		(src_raw->compFrames != dst_raw->compFrames) ||
		(src_raw->uncompFrames != dst_raw->uncompFrames))
	{
		printf("Number of frames is different %d/%d/%d %d/%d/%d\n",src_raw->frames,src_raw->compFrames,src_raw->uncompFrames,dst_raw->frames,dst_raw->compFrames,dst_raw->uncompFrames);
		return -1;
	}

	// at this point, the amount of data should be the same
	for(i=0;i<src_raw->frames;i++)
	{
		if(src_raw->timestamps[i] != dst_raw->timestamps[i])
		{
			printf("timestamp on frame %d is not the same %d %d\n",i,src_raw->timestamps[i], dst_raw->timestamps[i]);
			return -1;
		}

		for(y=0;y<src_raw->rows;y++)
		{
			for(x=0;x<src_raw->cols;x++)
			{
				idx = i*src_raw->rows*src_raw->cols + y*src_raw->cols + x;
				if((src_raw->image[idx] & 0x3ffc) != (dst_raw->image[idx] & 0x3ffc))
				{
					printf("the images are different at %d/%d/%d  %d %d\n",i,x,y,src_raw->image[idx],dst_raw->image[idx]);
					return -1;
				}
			}
		}
	}

	printf("the images are the same %d %d %d\n",src_raw->frames,src_raw->rows,src_raw->cols);
	return 0;
}


