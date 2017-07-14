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
#include "ImageTransformer.h"
#include "ComparatorNoiseCorrector.h"
#include "CorrNoiseCorrector.h"
#include "Image/Vecs.h"
#include "Image/AdvCompr.h"
#include "Image/RowSumCorrector.h"
#include "PairPixelXtalkCorrector.h"
#include "ChipIdDecoder.h"

static pthread_t thr[200];
static uint32_t numDirs=0;
static uint32_t numThreads=12;
static char *OrigExpPath = NULL;
static char *OrigDestPath = NULL;

void T0Transform(Image *img);


void DetermineCropWidthAndHeight ( int& cropx, int& cropy, int& cropw, int& croph, int w, int h )
{
  if ( cropw == 0 ) {
    cropx = 0;
    cropw = w;
  }
  if ( croph == 0 ) {
    cropy = 0;
    croph = h;
  }

  if ((cropw != w) &&  (cropw & 7)) {
    cropw += 7;
    cropw &= ~7;
  }
  if ((croph != h) && (croph & 7)) {
    croph += 7;
    croph &= ~7;
  }
}



void usage ( int cropx, int cropy, int cropw, int croph )
{
  fprintf ( stdout, "Crop - Utility to extract a subregion from a raw data set.  Output directory is created named './converted'.\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "   -a\tOutput flat files; ascii text\n" );
  fprintf ( stdout, "   -b\tConvert HighSampleRate run to multiple runs\n" );
  fprintf ( stdout, "   -j\tnumber of HighSampleRate threads to use.\n" );
  fprintf ( stdout, "   -x\tStarting x axis position (origin lower left) Default: %d\n",cropx );
  fprintf ( stdout, "   -y\tStarting y axis position (origin lower left) Default: %d\n",cropy );
  fprintf ( stdout, "   -w\tWidth of crop region Default: %d\n",cropw );
  fprintf ( stdout, "   -h\tHeight of crop region Default: %d\n",croph );
  fprintf ( stdout, "   -s\tSource directory containing raw data\n" );
  fprintf ( stdout, "   -f\tConverts only the one file named as an argument\n" );
  fprintf ( stdout, "   -z\tTells the image loader not to wait for a non-existent file\n" );
  fprintf ( stdout, "   -H\tPrints this message and exits.\n" );
  fprintf ( stdout, "   -v\tPrints version information and exits.\n" );
  fprintf ( stdout, "   -c\tOutput a variable rate frame compressed data set.  Default to whole chip\n" );
  fprintf ( stdout, "   -n\tOutput a non-variable rate frame compressed data set.\n" );
  fprintf ( stdout, "   -r\tOutput a regional t0 based dataset.\n" );
  fprintf ( stdout, "   -g\tProvide exclusion mask file for the chip type for which crop is being run. Should be supplied when -r is supplied.\n" );
  fprintf ( stdout, "   -e\tProvide t0 file for the chip type for which crop is being run. Should be supplied when -r is supplied.\n" );
  fprintf ( stdout, "   -l\tOutput a time compressed dataset pegged at max acquisition time of 5s.\n" );
  fprintf ( stdout, "   -p\tOutput a PCA compressed dataset\n" );
  fprintf ( stdout, "   -d\tOutput directory.\n" );
  fprintf ( stdout, "   -q\tPCA test type.\n" );
  fprintf ( stdout, "   -u\tPCA options.\n" );
  fprintf ( stdout, "   -i\tOverSample Values <combine> <skip>\n");
  fprintf ( stdout, "   -C <agressive>\tEnable Column noise correction.\n" );
  fprintf ( stdout, "   -D\tEnable thumbnail Column noise correction.\n" );
  fprintf ( stdout, "   -E t0 transform\n" );
  fprintf ( stdout, "   -F diff file\n" );
  fprintf ( stdout, "   -G Subtract average\n" );
  fprintf ( stdout, "   -I <chipType>  set chip type\n" );
  fprintf ( stdout, "   -J apply row correction\n" );
  fprintf ( stdout, "   -K apply col correction\n" );
  fprintf ( stdout, "   -L <corr> apply xtalk correction\n" );
  fprintf ( stdout, "   -M <corr> apply xtalk correction 2\n" );
  fprintf ( stdout, "   -N Gain correction\n" );
  fprintf ( stdout, "   -O Gain correction from beadfind\n" );
  fprintf ( stdout, "   -P row sum correction\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   Crop -s /results/analysis/PGM/testRun1\n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}


typedef struct{
	  int doAscii;
	  int vfc;
	  int pfc;
	  int regBasedAcq;
	  int TimeBasedAcq;
	  int excludeMask;
	  int OverSamplingSplit;
	  int useSeparatorT0File;
	  int dont_retry;
	  int skipCopy;
	  int cropx;
	  int cropy;
	  int cropw;
	  int croph;
	  int OverSample_skip;
	  int OverSample_combine;
	  char *separatorIn;
	  char *t0In;
	  char *excludeMaskFile;
	  char *PCAOpts;
	  char *expPath;
	  char *destPath;
	  char *oneFile;
	  int doColumnCorrection;
	  int doColumnCorrectionTn;
	  int cncType;
	  int T0Test;
	  char *cmpFile;
	  int subtractAverage;
	  char *chipType;
	  int applyRowCorrection;
	  int applyColCorrection;
	  float applyXtalkCorrection1;
	  float applyXtalkCorrection2;
	  int doGainCorrection;
	  int doGainCorrection_bf;
	  int doGainCorrection_bf_doneOnce;
	  int doFluidCorr;
	  char *applyThumbnailXtalkCorrection;
} OptionsCls;

void DoCrop(OptionsCls &options);
void DoOverSampleSplit(OptionsCls &options);
void SubtractOffMean(Image &loader);

void InitOptionsCls(OptionsCls &options)
{
	memset(&options,0,sizeof(options));
	options.vfc=1;
	options.croph=50;
	options.cropw=50;
	options.useSeparatorT0File=true;
	options.separatorIn = const_cast<char*> ( "./separator.summary.txt" );
	options.t0In = const_cast<char*> ( "./T0file.txt" );
	options.excludeMaskFile = const_cast<char*> ( "/opt/ion/config/exclusionMask_318.bin" );
	options.expPath  = const_cast<char*> ( "." );
	options.destPath = const_cast<char*> ( "./converted" );
	options.applyThumbnailXtalkCorrection = const_cast<char*> ( "" );
}

OptionsCls options;


int main ( int argc, char *argv[] )
{

	InitOptionsCls(options);

//  int dont_retry = 0;
  if ( argc == 1 ) {
    usage ( options.cropx, options.cropy, options.cropw, options.croph );
  }
  int argcc = 1;
  while ( argcc < argc ) {
    switch ( argv[argcc][1] ) {
    case 'a':
    	options.doAscii = 1;
      break;

    case 'x':
      argcc++;
      options.cropx = atoi ( argv[argcc] );
      break;

    case 'y':
      argcc++;
      options.cropy = atoi ( argv[argcc] );
      break;

    case 'w':
      argcc++;
      options.cropw = atoi ( argv[argcc] );
      break;

    case 'h':
      argcc++;
      // don't segfault if called with -h (user expects help)
      if ( argcc >= argc ) {
        usage ( options.cropx, options.cropy, options.cropw, options.croph );
      }
      options.croph = atoi ( argv[argcc] );
      break;

    case 's':
      argcc++;
      options.expPath = argv[argcc];
      break;

    case 'f':
      argcc++;
      options.oneFile = argv[argcc];
      break;

    case 'z':
    	options.dont_retry = 1;
      break;

    case 'p':
    	options.pfc=1;
    	options.cropx=0;
    	options.cropy=0;
    	options.cropw=0;
    	options.croph=0;
    	break;

    case 'c':
    	options.vfc=1;
    	options.cropx=0;
    	options.cropy=0;
    	options.cropw=0;
    	options.croph=0;
      break;


    case 'b':
    	options.OverSamplingSplit=1;
    	options.cropx=0;
    	options.cropy=0;
    	options.cropw=0;
    	options.croph=0;
      break;

    case 'n':
    	options.vfc=0;
      break;

    case 'v':
      fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "Crop" ).c_str() );
      exit ( 0 );
      break;
    case 'H':
      usage ( options.cropx, options.cropy, options.cropw, options.croph );
      break;
    case 'd':
      argcc++;
      options.destPath = argv[argcc];
      break;
    case 'r':
    	options.regBasedAcq=1;
      break;
    case 'l':
    	options.TimeBasedAcq=1;
      break;
    case 't':
      argcc++;
      options.separatorIn = argv[argcc];
      break;
    case 'e':
      argcc++;
      options.t0In = argv[argcc];
      options.useSeparatorT0File = false;
      break;
    case 'q':
      argcc++;
      strncpy(ImageTransformer::PCATest , argv[argcc], sizeof(ImageTransformer::PCATest)-1);
      break;
    case 'u':
      argcc++;
      options.PCAOpts=argv[argcc];
      break;
    case 'g':
      argcc++;
      options.excludeMaskFile = argv[argcc];
      options.excludeMask = 1;
      break;
    case 'i':
      argcc++;
      options.OverSample_combine = atoi(argv[argcc]);
      argcc++;
      options.OverSample_skip = atoi(argv[argcc]);
      break;

    case 'j':
    	argcc++;
    	numThreads = atoi(argv[argcc]);
    	break;

    case 'C':
    	options.doColumnCorrection=1;
    	if(argv[argcc+1][0] != '-'){
			argcc++;
			if(argc <= argcc  || (argv[argcc][0] != '0' && argv[argcc][0] != '1')){
				printf("Bad option on column noise correction\n");
				exit(-1);
			}

			options.cncType = atoi(argv[argcc]);
    	}
    	break;

    case 'D':
    	options.doColumnCorrectionTn=1;
    	break;

    case 'E':
    	options.T0Test=1;
    	break;

    case 'F':
        argcc++;
       options.cmpFile = argv[argcc];
    	break;

    case 'G':
    	options.subtractAverage=1;
    	break;

    case 'I':
    	if(argv[argcc+1][0] != '-'){
			argcc++;
			options.chipType = argv[argcc];
    	}
    	break;

    case 'J':
        options.applyRowCorrection = 1;
    	break;

    case 'K':
        options.applyColCorrection = 1;
    	break;

    case 'L':
    	if(argv[argcc+1][0] != '-'){
			argcc++;
			sscanf(argv[argcc],"%f",&options.applyXtalkCorrection1);
    	}
    	else
    		options.applyXtalkCorrection1 = 0.2;
    	break;

    case 'M':
    	if(argv[argcc+1][0] != '-'){
			argcc++;
			sscanf(argv[argcc],"%f",&options.applyXtalkCorrection2);
    	}
    	else
    		options.applyXtalkCorrection2 = 0.2;
    	break;
    case 'N':
    	options.doGainCorrection = 1;
    	break;
    case 'O':
    	options.doGainCorrection_bf = 1;
    	break;
    case 'P':
    	options.doFluidCorr = 1;
    	break;
    case 'k':

    	if(argcc+1 <argc && argv[argcc+1][0] != '-'){
			argcc++;
			options.applyThumbnailXtalkCorrection = argv[argcc];
    	}
    	else
    	{
			printf("Missing file name for apply thumbnail xtalk correction (-k)\n");
			exit(-1);
    	}
    	fprintf(stdout, "Apply thumbnail xtalk correction %s\n", options.applyThumbnailXtalkCorrection);
    	break;

    default:

      fprintf ( stdout, "unknown option %s\n",argv[argcc] );
      argcc++;
      break;
    }
    argcc++;
  }


  if(options.OverSamplingSplit)
  {
	  // crop an entire directory many times
	  DoOverSampleSplit(options);
  }
  else
  {
	  DoCrop(options);
  }
}


// averages frames together, or throws them away
void DoOverSample(OptionsCls &options, Image &img)
{
	int skip = options.OverSample_skip;
	int combine = options.OverSample_combine;
	int oframe=0;
	int cnt=0;

    RawImage *raw = (RawImage *)img.GetImage();
    uint32_t newFrames = raw->frames/(skip*combine);
    uint16_t *newImg=(uint16_t *)malloc(2*raw->rows*raw->cols*newFrames);
    uint16_t *newImgPtr=newImg;
    uint16_t *oldImgPtr=(uint16_t *)raw->image;
    int fs = raw->rows*raw->cols;
    int lfs = fs/8;
    v8su combineV;
    v8su maskV;
    v8su divV;

    combineV = LD_VEC8SU((short unsigned int)combine);
    maskV = LD_VEC8SU(0x3fff);
    divV = LD_VEC8SU(4);

    for(int frame=0;frame<(raw->frames/skip);frame++)
    {
    	v8su *lsrc = (v8su *)oldImgPtr;
		v8su *ldst = (v8su *)newImgPtr;
    	if(cnt == 0)
    	{// copy
			if(cnt == (combine-1))
			{ // 1x
				for(int pos=0;pos<lfs;pos++)
				{
					*ldst = *lsrc & maskV;
					ldst++;
					lsrc++;
				}
				newImgPtr += fs; // skip to the next frame
			}
			else
			{
				for(int pos=0;pos<lfs;pos++)
				{
					*ldst = *lsrc/divV;
					ldst++;
					lsrc++;
				}
			}
    	}
    	else if(cnt && combine && cnt < (combine-1))
    	{// add
			for(int pos=0;pos<lfs;pos++)
			{
				*ldst += *lsrc/divV;
				ldst++;
				lsrc++;
			}
    	}
    	else if(cnt == (combine-1))
    	{// add and divide
			for(int pos=0;pos<lfs;pos++)
			{
				*ldst = ((*ldst + (*lsrc/divV))/combineV)*divV;
				ldst++;
				lsrc++;
//				newImgPtr[pos] = (newImgPtr[pos] + oldImgPtr[pos])/combine;
			}
			newImgPtr += fs; // skip to the next frame
    	}
    	if(cnt == (combine-1))
    	    cnt=0;
    	else
    		cnt++;
    	oldImgPtr += fs*skip; // skip to the next frame
    }
    free(raw->image);
    raw->image = (short int *)newImg; // replace the image with the averaged one...
    // now, handle the timestamps
//    printf("timestamps before (%d/%d/%d): ",raw->frames,raw->uncompFrames,raw->baseFrameRate);
//    for(int frame=0;frame<raw->frames;frame++)
//    	printf("%d ",raw->timestamps[frame]);
//    printf("\n");
    raw->frames = newFrames;
    raw->baseFrameRate /= (combine*skip);
    raw->uncompFrames=raw->frames;
    for(int frame=0;frame<raw->frames;frame++)
    {
    	raw->timestamps[frame] = raw->timestamps[(frame+1)*(combine*skip) -1];
    }
//    printf("timestamps after (%d/%d/%d): ",raw->frames,raw->uncompFrames,raw->baseFrameRate);
//    for(int frame=0;frame<raw->frames;frame++)
//    	printf("%d ",raw->timestamps[frame]);
//    printf("\n");
}

void DoCrop(OptionsCls &options)
{
	  char name[MAX_PATH_LENGTH];
	  char BaseName[MAX_PATH_LENGTH];
	  char destName[MAX_PATH_LENGTH];
	  int i;
	  Image loader;
	  Acq saver;
	  int mode = 0;
	  i = 0;
	  bool allocate = true;
	  int nameListLen;
	  char **nameList;
	  int maxMode=2;
    //@WARNING: >must< copy beadfind_post_0000 if it exists, or Analysis will >fail to complete< on PGM run crops
    // This is an inobvious bug due to the fact that the Image routine "ReadyToLoad" checks for >the next< acq file
    // or explog_final.txt, >or< beadfind_post.  If none of those three exist, we wait forever.
    // not all runs appear to have explog_final.txt copied around, so can't depend on that.
	  const char* defaultNameList[] = {"beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
                                           "beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
	                                   "prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat"
	                                  };


	  // if requested...do not bother waiting for the files to show up
  if ( options.dont_retry )
    loader.SetTimeout ( 1,1 );

  if ( options.oneFile != NULL ) {
    nameList = &options.oneFile;
    nameListLen = 1;
    maxMode=1;
  } else {
    nameList = const_cast<char**> ( defaultNameList );
    nameListLen = sizeof ( defaultNameList ) /sizeof ( defaultNameList[0] );
  }

  // Create results folder
  umask ( 0 ); // make permissive permissions so its easy to delete.
  if ( mkdir ( options.destPath, 0777 ) ) {
    if ( errno == EEXIST ) {
      //already exists? well okay...
    } else {
      perror ( options.destPath );
      exit ( 1 );
    }
  }

  if(!options.skipCopy)
  {
  // Copy explog.txt file: all .txt files
  char cmd[1024];
  sprintf ( cmd, "cp -v %s/*.txt %s", options.expPath, options.destPath );
  if(system ( cmd ) != 0 )
	  printf("failed to copy txt files from src\n");
  // Copy lsrowimage.txt file
  const char *filesToMove[] = {
    "lsrowimage.dat",
    "gainimage.dat",
    "reimage.dat",
    "rmsimage.dat"
  };
  for(int iFile=0; iFile < 4; iFile++) {
    sprintf ( cmd, "cp -v %s/%s %s", options.expPath, filesToMove[iFile], options.destPath);
    if(system ( cmd ) == 1)
      fprintf (stdout, "No %s file found\n",filesToMove[iFile]);
  }
  }
  while ( mode < maxMode ) {
    if ( mode == 1 ) {
        sprintf ( BaseName, "acq_%04d.dat",i);
        sprintf ( name, "%s/%s", options.expPath, BaseName );
        sprintf ( destName, "%s/%s", options.destPath, BaseName );
    } else if ( mode == 0 ) {
      if ( i >= nameListLen ){
        mode++;
        i=0;
        continue;
      }
      sprintf ( BaseName, "%s", nameList[i]);
      sprintf ( name, "%s/%s", options.expPath, BaseName );
      sprintf ( destName, "%s/%s", options.destPath, BaseName );
    } else
      break;

    if ( loader.LoadRaw ( name, 0, allocate, false, false ) ) {
      allocate = false;
      const RawImage *raw = loader.GetImage();
      DetermineCropWidthAndHeight ( options.cropx, options.cropy, options.cropw, options.croph, raw->cols, raw->rows );
      struct timeval tv;
      double startT;
      double stopT;
      gettimeofday ( &tv, NULL );
      startT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );

      if(options.OverSample_skip || options.OverSample_combine)
    	  DoOverSample(options,loader);

      if(options.chipType)
    	  ChipIdDecoder::SetGlobalChipId(options.chipType);

      if(options.applyXtalkCorrection1){
			PairPixelXtalkCorrector xtalkCorrector;
			xtalkCorrector.Correct(loader.raw, options.applyXtalkCorrection1);
      }

      if (strlen(options.applyThumbnailXtalkCorrection)){
			PairPixelXtalkCorrector xtalkCorrector;
			xtalkCorrector.CorrectThumbnailFromFile(loader.raw, options.applyThumbnailXtalkCorrection);
      }


      if(options.applyRowCorrection){
    	  printf("Applying row noise correction\n");
     		CorrNoiseCorrector rnc;
     		rnc.CorrectCorrNoise(loader.raw,options.applyColCorrection?3:1,(loader.raw->cols==1200 && loader.raw->rows==800)?true:false,true,false );
      }
      else if(options.applyColCorrection){
    	    printf("Applying column noise correction\n");
     		CorrNoiseCorrector rnc;
      		rnc.CorrectCorrNoise(loader.raw,0,true,true,false );
      }

      if(options.applyXtalkCorrection2){
			PairPixelXtalkCorrector xtalkCorrector;
			xtalkCorrector.Correct(loader.raw, options.applyXtalkCorrection2);
      }
      if(options.doFluidCorr){
    	  char tn_name[2048];
    	  sprintf(tn_name,"%s/../thumbnail/",options.expPath);
    	  GenerateRowCorrFile(tn_name,BaseName);
    	  CorrectRowAverages(options.expPath,BaseName,&loader);
      }
		if (options.doGainCorrection_bf) {
			if (!options.doGainCorrection_bf_doneOnce) {
				options.doGainCorrection_bf_doneOnce = 1;
				// figure out gain
				ImageTransformer::CalculateGainCorrectionFromBeadfindFlow(options.expPath,false);
			}
			// apply gain
			ImageTransformer::GainCorrectImage(loader.raw);
		}
		if (options.doGainCorrection) {
			options.doGainCorrection = 0;
			static const char *gainName = "Gain.lsr";
			// compute the gain image first...
			char tstName3[1024];
			char *last_lptr = NULL;
			strcpy(tstName3, name);
			char *lptr = (char *) tstName3;
			while ((lptr = strstr(lptr, "/"))) {
				if (lptr)
					lptr++;
				last_lptr = lptr;
			}
			if (last_lptr)
				strcpy(last_lptr, gainName);
			else
				strcpy(tstName3, gainName);
			printf("Loading gain mask from %s\n", tstName3);
			float *gainPtr = AdvCompr::ReadGain(0, raw->cols, raw->rows, tstName3);
			if(!options.pfc){
				// apply it to the incoming file!
				int frameStride=raw->rows*raw->cols;
				for(int idx=0;idx<frameStride;idx++){
					for(int frame=raw->frames-1;frame>=0;frame--){
						raw->image[idx+frame*frameStride] = 12384/*raw->image[idx]*/ + ((raw->image[idx+frame*frameStride]-raw->image[idx])*(gainPtr[idx]));
					}
				}
#if 0
				// debug..  replace the last frame with the gain
				for(int idx=0;idx<frameStride;idx++){
					raw->image[idx+(raw->frames-1)*frameStride] = (8192.0f*gainPtr[idx]);
				}
#endif
			}
		}


      if(options.doColumnCorrectionTn)
	  {
		ComparatorNoiseCorrector cnc;
		cnc.CorrectComparatorNoiseThumbnail(loader.raw, NULL, 50,50, false);
	  } else if(options.doColumnCorrection){
		ComparatorNoiseCorrector cnc;
		bool beadfind = false;//((strstr(name,"beadfind_pre"))?true:false);
		cnc.CorrectComparatorNoise(loader.raw, NULL, false, options.cncType,beadfind );
	  }



      if(options.cmpFile){
    	  // subtract this file before continuing
    	  Image loader2;

    	  loader2.LoadRaw ( options.cmpFile, 0, 1, 0, 2 );

          if(options.OverSample_skip || options.OverSample_combine)
        	  DoOverSample(options,loader2);

          if(options.doColumnCorrectionTn)
    	  {
    		ComparatorNoiseCorrector cnc;
    		cnc.CorrectComparatorNoiseThumbnail(loader2.raw, NULL, 100,100, false);
    	  } else if(options.doColumnCorrection){
    		ComparatorNoiseCorrector cnc;
    		bool beadfind = ((strstr(options.cmpFile,"beadfind_pre"))?true:false);
    		cnc.CorrectComparatorNoise(loader2.raw, NULL, 0, options.cncType,beadfind );
    	  }



          {
            	short int *rawPtr = loader.raw->image;
              	short int *rawPtr2 = loader2.raw->image;
          	int frameStride=loader.raw->cols*loader.raw->rows;

          	for(int frame=0;frame<loader.raw->frames;frame++){
          		for(int idx=0;idx<frameStride;idx++){
          			*rawPtr -= *rawPtr2;
          			*rawPtr += 8192;
          			rawPtr++;
          			rawPtr2++;
          		}
          	}
          }
          printf("subtracted %s from the file\n",options.cmpFile);
      }

      if(options.subtractAverage)
      {
          // subtract off the mean trace
    	  SubtractOffMean(loader);
    	  printf("subtracted mean from trace\n");

      }

      // testing of lossy compression
      if(!options.pfc && ImageTransformer::PCATest[0]) {
        AdvComprTest(name,&loader,ImageTransformer::PCATest,true );
      }

      saver.SetData ( &loader );

      if(options.T0Test)
    	  saver.doT0Compression();

      if ( options.regBasedAcq && i == 0 ) {

        if ( options.excludeMask )
          saver.GenerateExcludeMaskRegions ( ( const char* ) options.excludeMaskFile );

        if ( options.useSeparatorT0File )
          saver.PopulateCroppedRegionalAcquisitionWindow ( ( const char* ) options.separatorIn, "t0Map.txt",
        		  options.cropx, options.cropy, options.cropw, options.croph, raw->timestamps[0] );
        else
          saver.ParseT0File ( ( const char* ) options.t0In, "t0Map.txt",
        		  options.cropx, options.cropy, options.cropw, options.croph, raw->timestamps[0] );

      }
      printf ( "Converting raw data %d %d frames: %d UncompFrames: %d\n", raw->cols, raw->rows, raw->frames, raw->uncompFrames );
      if ( options.doAscii ) {
        if ( !saver.WriteAscii ( destName, options.cropx, options.cropy, options.cropw, options.croph ) )
          break;
      } else {
        if ( options.vfc ) {
          if(options.pfc) {
            saver.WritePFV(destName, options.cropx, options.cropy, options.cropw, options.croph, options.PCAOpts);
          }
          else if ( options.regBasedAcq ) {
            if ( !saver.WriteFrameAveragedRegionBasedAcq ( destName, options.cropx, options.cropy, options.cropw, options.croph ) )
              break;
          } else if ( options.TimeBasedAcq ) {
            if ( !saver.WriteTimeBasedAcq ( destName, options.cropx, options.cropy, options.cropw, options.croph ) )
              break;
          } else {
            if ( !saver.WriteVFC ( destName, options.cropx, options.cropy, options.cropw, options.croph ) )
              break;
          }
        } else {
          if ( !saver.Write ( destName, options.cropx, options.cropy, options.cropw, options.croph ) )
            break;
        }
      }
      gettimeofday ( &tv, NULL );
      stopT = ( double ) tv.tv_sec + ( ( double ) tv.tv_usec/1000000 );
      printf ( "Converted: %s in %0.2lf sec\n", name,stopT - startT );
      fflush ( stdout );
      i++;
    } else {
      if ( ( mode == 0 && i >= 12 ) || ( mode ==1 ) ) {
        mode++;
        i = 0;
        allocate = true;
      } else
        i++;
    }
  }
}

void subWorker(int os, int skip, char *tmpExpPath, char *tmpDestPath)
{
	OptionsCls LocalOptions;

	memcpy(&LocalOptions,&options,sizeof(LocalOptions));

	LocalOptions.expPath=tmpExpPath;
	LocalOptions.destPath=tmpDestPath;
	LocalOptions.OverSample_combine = os;
	LocalOptions.OverSample_skip = skip;
	DoCrop(LocalOptions);
}

void ZeroPinned(Image &loader)
{
	short int *rawPtr = loader.raw->image;
	uint64_t avg;
	int cnt=0;
	short int avgs;
	int frameStride = loader.raw->cols * loader.raw->rows;
	char *pinned = (char *)malloc(frameStride);

	memset(pinned,0,frameStride);

	for (int frame = 0; frame < loader.raw->frames; frame++) {
		rawPtr = loader.raw->image + frame*frameStride;
		for (int idx = 0; idx < frameStride; idx++) {
			if(*rawPtr < 5 || *rawPtr > 16380)
				pinned[idx]=1;
			rawPtr++;
		}
	}
	for (int idx = 0; idx < frameStride; idx++) {
		if(pinned[idx]){
			cnt++;
			rawPtr = loader.raw->image + idx;
			for (int frame = 0; frame < loader.raw->frames; frame++) {
				rawPtr[frame*frameStride] = 0;
			}
		}
	}
	free(pinned);
	printf("pinned: %d out of %d\n",cnt,frameStride);
}

void SubtractOffMean(Image &loader)
{
	short int *rawPtr = loader.raw->image;
	uint64_t avg;
	short int avgs;
	int frameStride = loader.raw->cols * loader.raw->rows;

	printf("avg trace= ");
	for (int frame = 0; frame < loader.raw->frames; frame++) {
		avg=0;
		rawPtr = loader.raw->image + frame*frameStride;
		for (int idx = 0; idx < frameStride; idx++) {
			avg += *rawPtr++;
		}
		avgs = (avg / frameStride);
		printf(" %d",avgs);
		rawPtr = loader.raw->image + frame*frameStride;
		for (int idx = 0; idx < frameStride; idx++) {
			*rawPtr = *rawPtr - avgs + 8192;
			rawPtr++;
		}
	}
	printf("\n");
	ZeroPinned(loader);
}





void DoOverSampleSplitSubRegion(OptionsCls &options, char *dirName)
{
	// look through a directory, and crop each file into its four base experiments
	char destPath[4][2048];
	DIR *d = opendir(options.expPath);
	struct dirent *entry;
	int osC[4] = {8,4,2,1};
	int osS[4] = {1,2,4,8};


	for(int i=0;i<4;i++)
		sprintf(destPath[i],"%s_%d/%s",options.destPath,osC[i],dirName);

	if(d)
	{
	    while ((entry = readdir(d)) != NULL)
	    {
	    	if ((entry->d_type != DT_DIR) && ((entry->d_name[0] == 'a') || entry->d_name[0] == 'b' || entry->d_name[0] == 'p'))
	    	{
	    		for(int i=0;i<4;i++)
	    		{
					// crop this file into its four sub-sampling files
					options.oneFile = entry->d_name;
					options.destPath = destPath[i];
					options.OverSample_combine=osC[i];
					options.OverSample_skip=osS[i];
					options.skipCopy=1;
					DoCrop(options);
	    		}
	    	}
	    }
	    closedir(d);
	}
}


void *worker(void *arg)
{
	DIR *d = opendir(OrigExpPath);
	struct dirent *entry;
	uint localDirNum=0;
	uint64_t threadNum = (uint64_t)arg;
	uint32_t startIdx  = (numDirs*threadNum)/numThreads;
	uint32_t endIdx;
	if(numThreads > 1)
		endIdx = (numDirs*(threadNum+1))/numThreads - 1;
	else
		endIdx = numDirs;


	if(d)
	{
	    while ((entry = readdir(d)) != NULL)
	    {
	    	if ((entry->d_type == DT_DIR) && ((entry->d_name[0] == 'X') || !strcmp(entry->d_name,"thumbnail")))
	    	{
		    	if(localDirNum >= startIdx && localDirNum <= endIdx)
		    	{
		    		// crop this directory
		    		char tmpExpPath[2048];
		    		char tmpDestPath[2048];
		    		sprintf(tmpExpPath,"%s/%s",OrigExpPath,entry->d_name);
		    		sprintf(tmpDestPath,"%s_8/%s",OrigDestPath,entry->d_name);
		    		mkdir(tmpDestPath,0777);
		    		sprintf(tmpDestPath,"%s_4/%s",OrigDestPath,entry->d_name);
		    		mkdir(tmpDestPath,0777);
		    		sprintf(tmpDestPath,"%s_2/%s",OrigDestPath,entry->d_name);
		    		mkdir(tmpDestPath,0777);
		    		sprintf(tmpDestPath,"%s_1/%s",OrigDestPath,entry->d_name);
		    		mkdir(tmpDestPath,0777);
		    		OptionsCls LocalOptions;

		    		memcpy(&LocalOptions,&options,sizeof(LocalOptions));

		    		LocalOptions.expPath=tmpExpPath;
//		    		LocalOptions.destPath=tmpDestPath;

		    		printf("worker %" PRIu64 " croping %s to %s\n",threadNum,tmpExpPath,tmpDestPath);
		    		DoOverSampleSplitSubRegion(LocalOptions,entry->d_name);
		    	}
	    		localDirNum++;
	    	}
	    }
	    closedir(d);
	}
	return NULL;
}


// crop a run with all the oversample data saved into it's sub-sampled run counterparts.
//   ie, collect a run with the advanced option HighSampleRate on and 8x oversampling.
//       Then, run this command to split the run into 8x, 4x, 2x, and 1x runs.
void DoOverSampleSplit(OptionsCls &options)
{
	char DestPath[2048];
	int osC[4] = {8,4,2,1};

	if(options.destPath == NULL || strcmp(options.destPath,"./converted") == 0)
	{
		options.destPath=strdup(options.expPath);
		char *ptr = strstr(options.destPath,"R_"); //change the first character to non-R
		if(ptr)
		{
			ptr[0]='T';
		}
		else
		{
			char newPath[2048];
			sprintf(newPath,"%s/crop",options.destPath);
			free(options.destPath);
			options.destPath=strdup(newPath);
		}
	}

	OrigExpPath = options.expPath;
	OrigDestPath = options.destPath;

	//for(int os=8;os>=1;os/=2)
	{
		// create the top-level experiment directory
		for(int os=0;os<4;os++)
		{
			sprintf(DestPath,"%s_%d",OrigDestPath,osC[os]);
			mkdir(DestPath,0777);
		}
		numDirs=0;
//		memset(&DirList,0,sizeof(DirList));
		// populate dirlist
		DIR *d = opendir(OrigExpPath);
		struct dirent *entry;
		if(d)
		{
		    while ((entry = readdir(d)) != NULL)
		    {
		    	if ((entry->d_type == DT_DIR) && ((entry->d_name[0] == 'X') || !strcmp(entry->d_name,"thumbnail")))
		    	{
//		    		strcpy(DirList[numDirs++],entry->d_name);
//		    		char tmpExpPath[2048];
//		    		char tmpDestPath[2048];
//					sprintf(tmpExpPath,"%s/%s",OrigExpPath,entry->d_name);
//					sprintf(tmpDestPath,"%s/%s",DestPath,entry->d_name);
//			    	printf(" Cropping %s to %s\n",tmpExpPath,tmpDestPath);
//					subWorker(os,(base_os-os),tmpExpPath,tmpDestPath);
		    		numDirs++;
		    	}
		    }
		    closedir(d);
//		    if(numDirs==0)
//		    {
//		    	// didn't add anything..
//		    	strcpy(DirList[numDirs++],".");
//		    }
		}

//		// fill in the number of acquisitions to check
//		if(numAcq == 0)
//			numAcq = GetNumAcq();
//
		if(numThreads > numDirs)
			numThreads = numDirs;

		// spawn the worker threads
		for(uint64_t i=0;i<numThreads;i++)
		{
			pthread_create(&thr[i],NULL,worker,(void *)i);
			usleep(100000);
		}

		// wait for them to complete
		for(uint i=0;i<numThreads;i++)
		{
			pthread_join(thr[i],NULL);
		}

		// copy all the txt files.  wait till last so analysis doesn't launch
		for(int os=0;os<4;os++)
		{
			char cmd[4096];
			char DstExp[2048];
			sprintf(cmd,"cp %s/*.txt %s_%d/",OrigExpPath,OrigDestPath,osC[os]);
	    	printf("executing %s\n",cmd);
		    if(system ( cmd ) != 0)
			{
		    	printf("Failed to copy over .txt files\n");
			}
		    else
		    {
		    	sprintf(cmd,"RNAME=`grep runName %s_%d/explog.txt | sed 's/runName://g'`; sed -i \"s/${RNAME}/${RNAME}_%d/g\" %s_%d/explog*.txt",OrigDestPath,osC[os],osC[os],OrigDestPath,osC[os]);
		    	printf("executing %s\n",cmd);
		        if(system ( cmd ) == 1){
		          fprintf (stdout, "failed %s\n",cmd);
		        }
		    	char *ptr = strstr(OrigDestPath,"/results/");
		    	if(ptr)
		    	{
		    		ptr += strlen("/results/");
		    		strcpy(DstExp,ptr);
		    	}
		    	else
		    	{
		    		strcpy(DstExp,OrigDestPath);
		    	}
		    	ptr = strstr(DstExp,"/");
		    	if(ptr)
		    		*ptr = 0;
		    	sprintf(cmd,"/software/testing/ForceReanalysis.sh %s_%d",DstExp,osC[os]);
		    	printf("executing %s\n",cmd);
		        if(system ( cmd ) == 1){
		          fprintf (stdout, "failed %s\n",cmd);
		        }
		    }
		}
	}
}



