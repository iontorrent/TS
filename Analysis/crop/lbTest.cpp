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




void usage (char *name )
{
  fprintf ( stdout, "%s - Utility to investigate using less bits to transmit data.\n",name );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   %s <srcDir> <dstDir>\n",name );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}

void MaskBits(RawImage *img, int bits)
{
	short mask=(1<<bits)-1;
	short *imgPtr=img->image;
	int limit=img->frameStride*img->frames;
	for(int idx=0;idx<limit;idx++){
		imgPtr[idx] &= mask;
	}
}

void getAvgTrace(RawImage *img, unsigned short *avgTrace, const char *fname)
{
	for(int frm=0;frm<img->frames;frm++){
		uint64_t avg=0;
		short *imgPtr=&img->image[frm*img->frameStride];
		for(int idx=0;idx<img->frameStride;idx++){
			avg += imgPtr[idx];
		}
		avgTrace[frm]=avg/(uint64_t)img->frameStride;
	}
	for(int frm=1;frm<img->frames;frm++){
		avgTrace[frm]-=avgTrace[0];
	}
	avgTrace[0]=0;

	printf("Avg Trc %s:\n    ",fname);
	for(int frm=0;frm<img->frames;frm++){
		//avgTrace[frm] = (avgTrace[frm] * 15) / 10;
		printf(" %03d",avgTrace[frm]);
	}
	printf("\n");
}

void modAvgTrace(RawImage *img, unsigned short *avg, int sub)
{
	for(int frm=0;frm<img->frames;frm++){
		short *imgPtr=&img->image[frm*img->frameStride];
		short subVal=avg[frm];
		for(int idx=0;idx<img->frameStride;idx++){
			if(sub)
				imgPtr[idx] -= subVal;
			else
				imgPtr[idx] += subVal;
		}
	}
}

void PrintHistogram(Image *mimg, char *fname,const char *txt)
{
	RawImage *img = mimg->raw;
#define NUMBINS 16
	uint64_t bins[NUMBINS]={0};

	for(int frm=1;frm<img->frames;frm++){
		short *imgPtr=&img->image[frm*img->frameStride];
		short *pimgPtr=&img->image[(frm-1)*img->frameStride];
		for(int idx=0;idx<img->frameStride;idx++){
			short val = imgPtr[idx]-pimgPtr[idx];
			if(val <0)
				val=-val;
			for(int bit=15;bit >=0;bit--){
				if((1<<bit) & val){
					bins[bit]++;
					break;
				}
			}
		}
	}
	uint64_t total=0;
	printf("%s %s histogram: ",fname,txt);
	for(int bin=0;bin<NUMBINS;bin++){
		//printf(" %ld",bins[bin]);
		total+=bins[bin];
	}
	//printf("   histogram: ");
	for(int bin=0;bin<NUMBINS;bin++){
		printf(" %02d",(int)((100*bins[bin])/total));
	}

	printf("\n");

	// save the result
    Acq srcAcq;
    srcAcq.SetData(mimg);
    srcAcq.WriteVFC(fname,0,0,img->cols,img->rows);


}

void ZeroFirstFrame(RawImage *img)
{
	for(int idx=0;idx<img->frameStride;idx++){
		short *imgPtr=&img->image[idx];
		short subVal=imgPtr[0];
		for(int frm=1;frm<img->frames;frm++){
			imgPtr[frm*img->frameStride] -= subVal-32;
		}
		imgPtr[0]=32;
	}
}



int main ( int argc, char *argv[] )
{
  //int cropx = 624, cropy = 125, cropw = 100, croph = 100;
  int cropx = 0, cropy = 0, cropw = 0, croph = 0;
  int i,x,y,idx;
  char srcFile[1024],dstFile[1024];
  char *srcDir;
  unsigned short avgTrace[105];
  unsigned short avgTrace1[105];
  Image loader_extraG, loader_src, loader_dst;
  bool allocate = true;
  RawImage *src_extraG,*src_raw,*dst_raw;
	struct timeval tv;
	double startT;
	double stopT;

  if ( argc < 2 ) {
    usage ( argv[0]);
  }

  srcDir = argv[1];

  sprintf(srcFile,"%s/extraG_0000.dat",srcDir);
  sprintf(dstFile,"%s/op",srcDir);
  mkdir(dstFile,0644);

	gettimeofday ( &tv, NULL );
	startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );

	if(!loader_extraG.LoadRaw ( srcFile, 0, allocate, false, false ))
	{
		printf("failed to load file %s\n",srcFile);
		return -1;
	}
    gettimeofday ( &tv, NULL );
    stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
    printf ( "Converted: %s in %0.2lf sec\n", srcFile,stopT - startT );
    fflush ( stdout );
	src_extraG = loader_extraG.raw;

	// determine the average trace
	getAvgTrace(src_extraG,avgTrace,srcFile);


	for(int fnum=0;fnum<4;fnum++){
	  sprintf(srcFile,"%s/acq_000%d.dat",srcDir,fnum);

		gettimeofday ( &tv, NULL );
		startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );

		if(!loader_src.LoadRaw ( srcFile, 0, allocate, false, false ))
		{
			printf("failed to load file %s\n",srcFile);
			return -1;
		}
	    gettimeofday ( &tv, NULL );
	    stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
	    printf ( "Converted: %s in %0.2lf sec\n", srcFile,stopT - startT );
	    fflush ( stdout );
		src_raw = loader_src.raw;

		sprintf(dstFile,"%s/op/acq_000%d.dat_orig",srcDir,fnum);
        PrintHistogram(&loader_src,dstFile,"");
    	getAvgTrace(src_raw,avgTrace1,srcFile);
    	printf("Sub Avg Trc :\n    ");
    	for(int frm=0;frm<src_raw->frames;frm++){
    		//avgTrace[frm] = (avgTrace[frm] * 15) / 10;
    		printf(" %03d",(avgTrace[frm] * 15) / 10);
    	}
    	printf("\n");


        // subtract extraG trace
		modAvgTrace(src_raw, avgTrace, 1);

  	    sprintf(dstFile,"%s/op/acq_000%d.dat_subt",srcDir,fnum);
        PrintHistogram(&loader_src,dstFile,"");

		//ZeroFirstFrame(src_raw);

		// mask off upper bits..
		MaskBits(src_raw,8);

		// determine the histogram of differences
  	    sprintf(dstFile,"%s/op/acq_000%d.dat_mask",srcDir,fnum);
        PrintHistogram(&loader_src,dstFile,"");

		// try to re-construct the traces..
		// add extraG trace

		// save re-constructed traces
	}
	return 0;
}


