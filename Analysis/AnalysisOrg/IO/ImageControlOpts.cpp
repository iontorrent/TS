/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageControlOpts.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "Utils.h"

using namespace std;

void ImageControlOpts::DefaultImageOpts()
{
  maxFrames = 0;    // Set later from the first raw image header.
  totalFrames = 0;
  nn_subtract_empties = false; // do >not< subtract nearest-neighbor empties
  NNinnerx = 1;
  NNinnery = 1;
  NNouterx = 12;
  NNoutery = 8;
  ignoreChecksumErrors = 0;
  hilowPixFilter = false;   // default is disabled
  flowTimeOffset = 1000;
  col_flicker_correct = false; //default to turn on
  col_flicker_correct_verbose = false;
  col_pair_pixel_xtalk_correct = false;
  pair_xtalk_fraction = 0.;
  aggressive_cnc = false;
  gain_correct_images = false;
  gain_debug_output = false;
  has_wash_flow = 0;
  // image diagnostics
  outputPinnedWells = false;
  tikSmoothingFile[0] = '\000';   // (APB)
  tikSmoothingInternal[0] = '\000'; // (APB)
  doSdat = false; // Look for synchronized dat (sdat) files instead of usual dats.
  total_timeout = 0; // 0 means use whatever the image class has set as default
  sdatSuffix = "sdat";
  //if (acqPrefix != NULL) free (acqPrefix);
  acqPrefix = strdup("acq_");
  threaded_file_access = true;
  PCATest[0]=0;
  readaheadDat = 0;
}

ImageControlOpts::~ImageControlOpts()
{
  if (acqPrefix!=NULL) {
    free( acqPrefix);
    acqPrefix = NULL;
  }
}

void ImageControlOpts::ImageControlForProton(bool default_aggressive){
  aggressive_cnc = default_aggressive;
  fprintf ( stdout, "Option %s: %s\n", "--col-flicker-correct-aggressive",(aggressive_cnc)?"on":"off");
}

void ImageControlOpts::PrintHelp()
{
	printf ("     ImageControlOpts\n");
    printf ("  -f,--frames                INT               max frames []\n");
    printf ("     --do-sdat               BOOL              enable do sdat [false]\n");
    printf ("     --pca-test              STRING            setup PCA test []\n");
    printf ("     --PCA-test              STRING            same as --pca-test []\n");
	printf ("     --smoothing-file        FILE              tik smoothing file name []\n");
    printf ("     --smoothing             STRING            tik smoothing internal []\n");
    printf ("     --img-gain-correct      BOOL              enable image gain correction [true for Proton; false for PGM]\n");
    printf ("     --output-pinned-wells   BOOL              output pinned wells [false]\n");
    printf ("     --flowtimeoffset        INT               setup flow time offset [1000]\n");
    printf ("     --nn-subtract-empties   BOOL              enable nn subtract empties [false]\n");
    printf ("     --hilowfilter           BOOL              enable hi low pixel filter [false]\n");
    printf ("     --total-timeout         INT               total timeout [0]\n");
    printf ("     --readaheaddat          INT               setup readaheadDat [0]\n");
    printf ("     --readaheadDat          INT               same as readaheaddat [0]\n");
    printf ("     --pair-xtalk-coeff      FLOAT             setup pair xtalk fraction [0.0]\n");
    printf ("     --col-flicker-correct   BOOL              enable col flicker correction [true for Proton; false for PGM]\n");
    printf ("     --col-flicker-correct-aggressive    BOOL  enable col flicker correction aggressive [true for Proton; false for PGM]\n");
    printf ("     --col-flicker-correct-verbose       BOOL  enable col flicker correction verbose [false]\n");
    printf ("     --ignore-checksum-errors            BOOL  ignore checksum errors [false]\n");
    printf ("     --ignore-checksum-errors-1frame     BOOL  ignore checksum errors 1 frame [false]\n");
    printf ("     --no-threaded-file-access           BOOL  no threaded file access [false]\n");
    printf ("     --col-doubles-xtalk-correct         BOOL  enable col pair pixel xtalk correction [false]\n");
    printf ("     --nnmask                INT VECTOR OF 2   setup NN inner and outer [1,3]\n");
    printf ("     --nnMask                INT VECTOR OF 2   same as --nnmask [1,3]\n");
    printf ("     --nnmaskwh              INT VECTOR OF 4   setup NN inner and outer [1,1,12,8]\n");
    printf ("     --nnMaskWH              INT VECTOR OF 4   same as --nnmaskwh [1,1,12,8]\n");
    printf ("\n");
}

void ImageControlOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	doSdat = RetrieveParameterBool(opts, json_params, '-', "do-sdat", false);
	string s1 = RetrieveParameterString(opts, json_params, '-', "pca-test", "");
	if(s1.length() > 0)
	{
		sprintf(PCATest, "%s", s1.c_str());
	}

	//jz check the following opts, proton is true
	col_flicker_correct = RetrieveParameterBool(opts, json_params, '-', "col-flicker-correct", false);
	col_flicker_correct_verbose = RetrieveParameterBool(opts, json_params, '-', "col-flicker-correct-verbose", false);
	aggressive_cnc = RetrieveParameterBool(opts, json_params, '-', "col-flicker-correct-aggressive", false);
	gain_correct_images = RetrieveParameterBool(opts, json_params, '-', "img-gain-correct", false);
	string s2 = RetrieveParameterString(opts, json_params, '-', "smoothing-file", "");
	if(s2.length() > 0)
	{
		sprintf(tikSmoothingFile, "%s", s2.c_str());
	}
	string s3 = RetrieveParameterString(opts, json_params, '-', "smoothing", "");
	if(s3.length() > 0)
	{
		sprintf(tikSmoothingInternal, "%s", s3.c_str());
	}
	bool b1 = RetrieveParameterBool(opts, json_params, '-', "ignore-checksum-errors", false);
	if(b1)
	{
		ignoreChecksumErrors |= 0x01;
	}
	bool b2 = RetrieveParameterBool(opts, json_params, '-', "ignore-checksum-errors-1frame", false);
	if(b2)
	{
		ignoreChecksumErrors |= 0x02;
	}
	outputPinnedWells = RetrieveParameterBool(opts, json_params, '-', "output-pinned-wells", false);
	flowTimeOffset = RetrieveParameterInt(opts, json_params, '-', "flowtimeoffset", 1000);
	nn_subtract_empties = RetrieveParameterBool(opts, json_params, '-', "nn-subtract-empties", false);
	vector<int> vec1;
	RetrieveParameterVectorInt(opts, json_params, '-', "nnmask", "1,3", vec1);
	if(vec1.size() == 2)
	{
		NNinnerx = vec1[0];
		NNinnery = vec1[0];
		NNouterx = vec1[1];
		NNoutery = vec1[1];
	}
	else
	{
        fprintf ( stderr, "Option Error: nnmask format wrong, not size = 2\n" );
        exit ( EXIT_FAILURE );
	}
	vector<int> vec2;
	RetrieveParameterVectorInt(opts, json_params, '-', "nnmaskwh", "1,1,12,8", vec2);
	if(vec2.size() == 4)
	{
		NNinnerx = vec2[0];
		NNinnery = vec2[1];
		NNouterx = vec2[2];
		NNoutery = vec2[3];
	}
	else
	{
        fprintf ( stderr, "Option Error: nnmaskwh format wrong, not size = 4\n" );
        exit ( EXIT_FAILURE );
	}
	hilowPixFilter = RetrieveParameterBool(opts, json_params, '-', "hilowfilter", false);
	total_timeout = RetrieveParameterInt(opts, json_params, '-', "total-timeout", 0);
	readaheadDat = RetrieveParameterInt(opts, json_params, '-', "readaheaddat", 0);
	bool no_threaded_file_access = RetrieveParameterBool(opts, json_params, '-', "no-threaded-file-access", false);
	threaded_file_access = !no_threaded_file_access;
	//jz the following comes from CommandLineOpts::GetOpts
	int maxFramesInput = RetrieveParameterInt(opts, json_params, 'f', "frames", -1);
	if(maxFramesInput > 0)
	{
		maxFrames = maxFramesInput;
	}
	col_pair_pixel_xtalk_correct = RetrieveParameterBool(opts, json_params, '-', "col-doubles-xtalk-correct", false);
	pair_xtalk_fraction = RetrieveParameterFloat(opts, json_params, '-', "pair-xtalk-coeff", 0.0f);
    fluid_potential_correct = RetrieveParameterBool(opts, json_params, '-', "fluid-potential-correct", false);
    fluid_potential_threshold = RetrieveParameterFloat(opts, json_params, '-', "fluid-potential-threshold", 1.0f);

}
