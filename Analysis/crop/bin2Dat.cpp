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
  fprintf ( stdout, "bin2Dat - Utility to create a dat file from a bin file.  eg, bfmask.bin\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   bin2Dat bfmask.bin acq_0000.dat bfmask.dat \n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}

int main ( int argc, char *argv[] )
{
  //int cropx = 624, cropy = 125, cropw = 100, croph = 100;
  int cropx = 0, cropy = 0, cropw = 0, croph = 0;
  int i,x,y,idx;
  char *srcFile,*dstFile,*donorFile;
  Image loader_src,loader_dst;
  bool allocate = true;
  RawImage *src_raw,*dst_raw;
	struct timeval tv;
	double startT;
	double stopT;
	uint32_t width=0,height=0;

  if ( argc < 3 ) {
    usage ( cropx, cropy, cropw, croph );
  }

  srcFile   = argv[1];
  donorFile = argv[2];
  dstFile   = argv[3];


  // read in the src file
  FILE *fp = fopen(srcFile,"r");
  if(fp){
	  if(fread(&width,sizeof(width),1,fp) < 1)
		  printf("failed to read width\n");
	  if(fread(&height,sizeof(height),1,fp) < 1)
		  printf("failed to read height\n");
	  printf("width=%d height=%d\n",width,height);

	  if(width < 20000 && height < 20000){

		  uint16_t *buffer = (uint16_t *)malloc(width*height*2);
		  if(fread(buffer,2,width*height,fp) < width*height)
			  printf("failed to read data\n");

		  // now, write this data into a .dat file
			if(!loader_src.LoadRaw ( donorFile, 0, allocate, false ))
			{
				printf("failed to load file %s\n",donorFile);
				free(buffer);
				return -1;
			}
		  RawImage *raw = loader_src.raw;
		  memcpy(raw->image,buffer,width*height*2);
		  raw->frames=1;
		  raw->uncompFrames=1;
	      Acq saver;
	      saver.SetData ( &loader_src );
	      saver.WriteVFC(dstFile, 0, 0, loader_src.raw->cols, loader_src.raw->rows);
	      printf("Wrote %s\n",dstFile);

		  free(buffer);
	  }
	  fclose(fp);

	  // write out the dat file
  }
  else{
	  printf("failed to open src file %s\n",srcFile);
  }

  return 0;
}


