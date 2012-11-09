/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

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


// thread function declaration
void *do_region_crop ( void *ptr );

struct foobar {
  struct crop_region *CropRegion;
  const RawImage *raw;
  char *destName;
  char *destPath;
  Acq *saver;
  Image *loader;
  int doAscii;
  int vfc;
  int i;
};

struct crop_region {
  int region_origin_x;
  int region_origin_y;
  int region_len_x;
  int region_len_y;
  char region_name[24];
};

void usage ( int cropx, int cropy )
{
  fprintf ( stdout, "Blocks - Utility to chunk a raw data set into image blocks.\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "   -a\tOutput flat files; ascii text\n" );
  fprintf ( stdout, "   -b\tUse alternate sampling rate\n" );
  fprintf ( stdout, "   -x\tNumber of blocks along x axis. Default is %d\n",cropx );
  fprintf ( stdout, "   -y\tNumber of blocks along y axis. Default is %d\n",cropy );
  fprintf ( stdout, "   -t\tchip type 314, 316, or 318\n" );
  fprintf ( stdout, "   -s\tSource directory containing raw data\n" );
  fprintf ( stdout, "   -f\tConverts only the one file named as an argument\n" );
  fprintf ( stdout, "   -z\tTells the image loader not to wait for a non-existent file\n" );
  fprintf ( stdout, "   -H\tPrints this message and exits.\n" );
  fprintf ( stdout, "   -v\tPrints version information and exits.\n" );
  fprintf ( stdout, "   -c\tOutput a variable rate frame compressed data set.  Default to whole chip\n" );
  fprintf ( stdout, "   -n\tOutput a non-variable rate frame compressed data set.\n" );
  fprintf ( stdout, "   -d\tOutput directory.\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   Blocks -s /results/analysis/PGM/testRun1 -t [314|316|318]\n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}

int main ( int argc, char *argv[] )
{
  int cropx = 2, cropy = 2;
  int region_len_x = 0;
  int region_len_y = 0;
  char *expPath  = const_cast<char*> ( "." );
  char *destPath = const_cast<char*> ( "./converted" );
  char *oneFile = NULL;
  int chipType = 0;
  //int alternate_sampling=0;
  int doAscii = 0;
  int vfc = 1;
  int dont_retry = 0;
  if ( argc == 1 ) {
    usage ( cropx, cropy );
  }
  int argcc = 1;
  while ( argcc < argc ) {
    switch ( argv[argcc][1] ) {
    case 'a':
      doAscii = 1;
      break;

    case 'x':
      argcc++;
      cropx = atoi ( argv[argcc] );
      break;

    case 'y':
      argcc++;
      cropy = atoi ( argv[argcc] );
      break;

    case 's':
      argcc++;
      expPath = argv[argcc];
      break;

    case 't':
      argcc++;
      chipType = atoi ( argv[argcc] );
      break;

    case 'f':
      argcc++;
      oneFile = argv[argcc];
      break;

    case 'z':
      dont_retry = 1;
      break;

    case 'c':
      vfc=1;
      cropx=0;
      cropy=0;
      //cropw=0;
      //croph=0;
      break;

    case 'n':
      vfc=0;
      break;

    case 'b':
      //alternate_sampling=1;
      break;

    case 'v':
      fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "Blocks" ).c_str() );
      exit ( 0 );
      break;
    case 'H':
      usage ( cropx, cropy );
      break;
    case 'd':
      argcc++;
      destPath = argv[argcc];
      break;

    default:
      argcc++;
      fprintf ( stdout, "\n" );

    }
    argcc++;
  }

  char name[MAX_PATH_LENGTH];
  char destName[MAX_PATH_LENGTH];
  int i;
  Image loader;
  Acq saver;
  int mode = 0;
  i = 0;
  bool allocate = true;
  char **nameList;
  char *defaultNameList[] = {"beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
                             "beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
                             "prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat", "prerun_0004.dat"
                            };
  int nameListLen;



  // if requested...do not bother waiting for the files to show up
  if ( dont_retry )
    loader.SetTimeout ( 1,1 );

  if ( oneFile != NULL ) {
    nameList = &oneFile;
    nameListLen = 1;
    mode = 1;
  } else {
    nameList = defaultNameList;
    nameListLen = sizeof ( defaultNameList ) /sizeof ( defaultNameList[0] );
  }

  // Create results folder
  umask ( 0 ); // make permissive permissions so its easy to delete.
  if ( mkdir ( destPath, 0777 ) ) {
    if ( errno == EEXIST ) {
      //already exists? well okay...
    } else {
      perror ( destPath );
      exit ( 1 );
    }
  }

  // Initialize array of crop regions
  int numRegions = cropx * cropy;
  struct crop_region CropRegions[numRegions];
  pthread_t threads[numRegions];

  // Calculate regions based on chip type and number of blocks requested per axis
  // cropx is number of regions to carve along x axis
  // cropy is number of regions to carve along the y axis
  if ( chipType == 314 ) {
    //[1280,1152]
    // x axis length is 1280 pixels
    // y axis length is 1152 pixels
    region_len_x = 1280 / cropx;
    region_len_y = 1152 / cropy;
  } else if ( chipType == 316 ) {
    //[2736,2640]
    region_len_x = 2736 / cropx;
    region_len_y = 2640 / cropy;
  } else if ( chipType == 318 ) {
    //[3392,3792]
    region_len_x = 3392 / cropx;
    region_len_y = 3792 / cropy;
  } else {
    fprintf ( stderr, "Unknown chip: %d\n", chipType );
    exit ( 1 );
  }


  //Open outputfile for the BlockStatus text
  FILE *blockline = NULL;
  blockline = fopen ( "blockStatus_output", "w" );

  for ( int y = 0; y < cropy;y++ ) {
    for ( int x = 0; x < cropx;x++ ) {
      int region = x + ( y*cropx );
      CropRegions[region].region_len_x = region_len_x;
      CropRegions[region].region_len_y = region_len_y;
      CropRegions[region].region_origin_x = x * region_len_x;
      CropRegions[region].region_origin_y = y * region_len_y;
      snprintf ( CropRegions[region].region_name, 24, "X%d_Y%d",
                 CropRegions[region].region_origin_x,
                 CropRegions[region].region_origin_y );
      //write out the BLockStatus line
      fprintf ( blockline, "BlockStatus: X%d, Y%d, W%d, H%d, AutoAnalyze:1, AnalyzeEarly:1, nfsCopy:/results-dnas1,  ftpCopy://\n",
                CropRegions[region].region_origin_x,
                CropRegions[region].region_origin_y,
                CropRegions[region].region_len_x,
                CropRegions[region].region_len_y );
    }
  }
  fclose ( blockline );

  // Copy explog.txt file: all .txt files
  char cmd[1024];
  sprintf ( cmd, "cp -v %s/*.txt %s", expPath, destPath );
  assert ( system ( cmd ) == 0 );

  while ( mode < 2 ) {
    if ( mode == 0 ) {
      sprintf ( name, "%s/acq_%04d.dat", expPath, i );
      sprintf ( destName, "acq_%04d.dat", i );
    } else if ( mode == 1 ) {
      if ( i >= nameListLen )
        break;
      sprintf ( name, "%s/%s", expPath, nameList[i] );
      sprintf ( destName, "%s", nameList[i] );
    } else
      break;
    if ( loader.LoadRaw ( name, 0, allocate, false ) ) {
      allocate = false;
      struct timeval tv;
      double startT;
      double stopT;
      gettimeofday ( &tv, NULL );
      startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
      const RawImage *raw = loader.GetImage();

      printf ( "Blocking raw data %d %d frames: %d UncompFrames: %d\n", raw->cols, raw->rows, raw->frames, raw->uncompFrames );
      fflush ( stdout );

      // Threaded
      //NOTE: Made attempt to use threads here, but am not sure now if Image class is thread safe.
      //Left threads in place.
      // create data structure for threads
      struct foobar data;
      data.CropRegion = NULL;
      data.raw = raw;
      data.destName = strdup ( destName );
      data.destPath = strdup ( destPath );
      data.saver = &saver;
      data.loader = &loader;
      data.doAscii = doAscii;
      data.vfc = vfc;

      for ( int region = 0; region < numRegions;region++ ) {
        data.CropRegion = &CropRegions[region];
        data.i = i;
        pthread_create ( &threads[region],NULL,do_region_crop, ( void * ) &data );
        pthread_join ( threads[region],NULL );
        //sequential execution of threads.  Is Image thread safe?

      }


      gettimeofday ( &tv, NULL );
      stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
      printf ( "Blocked: %s in %0.2lf sec\n", name,stopT - startT );
      fflush ( stdout );
      i++;
    } else {
      if ( ( mode == 1 && i >= 12 ) || ( mode == 0 ) ) {
        mode++;
        i = 0;
        allocate = true;
      } else
        i++;
    }
  }

  exit ( 0 );
}

void *do_region_crop ( void *ptr )
{
  struct foobar *data;
  data = ( struct foobar * ) ptr;

  char destSubPath[200];

  struct timeval tv;
  double startT;
  double stopT;
  gettimeofday ( &tv, NULL );
  startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );

  sprintf ( destSubPath,"%s/%s",data->destPath,data->CropRegion->region_name );
  // Create results folder
  umask ( 0 ); // make permissive permissions so its easy to delete.
  if ( mkdir ( destSubPath, 0777 ) ) {
    if ( errno == EEXIST ) {
      //already exists? well okay...
    } else {
      perror ( destSubPath );
      return NULL;
    }
  }

  int cropx = data->CropRegion->region_origin_x;
  int cropy = data->CropRegion->region_origin_y;
  int cropw = data->CropRegion->region_len_x;
  int croph = data->CropRegion->region_len_y;

  char destFile[200];
  sprintf ( destFile,"%s/%s",destSubPath, data->destName );
  fprintf ( stdout, "Writing %s\n", destFile );
  data->saver->SetData ( data->loader );

  if ( data->doAscii ) {
    data->saver->WriteAscii ( destFile, cropx, cropy, cropw, croph );
  } else {
    if ( data->vfc ) {
      data->saver->WriteVFC ( destFile, cropx, cropy, cropw, croph );
    } else {
      data->saver->Write ( destFile, cropx, cropy, cropw, croph );
    }
  }

  gettimeofday ( &tv, NULL );
  stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
  printf ( "Region: %s in %0.2lf sec\n", data->CropRegion->region_name,stopT - startT );
  fflush ( stdout );
  return NULL;
}
