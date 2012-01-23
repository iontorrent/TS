/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * Ranger.cpp
 *
 *  Created on: Nov 22, 2010
 *      Author: mbeauchemin
 */

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

int main(int argc, char *argv[])
{
	int cropx = 624, cropy = 125, cropw = 100, croph = 100;
	char *expPath  = const_cast<char*>(".");
	char *destPath = const_cast<char*>("./converted");
	char *oneFile = NULL;
	int doAscii = 0;
	int dont_retry = 0;

	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'a':
				doAscii = 1;
			break;

			case 'x':
				argcc++;
				cropx = atoi(argv[argcc]);
			break;

			case 'y':
				argcc++;
				cropy = atoi(argv[argcc]);
			break;

			case 'w':
				argcc++;
				cropw = atoi(argv[argcc]);
			break;

			case 'h':
				argcc++;
				croph = atoi(argv[argcc]);
			break;

			case 's':
				argcc++;
				expPath = argv[argcc];
			break;

			case 'f':
				argcc++;
				oneFile = argv[argcc];
			break;

			case 'z':
				dont_retry = 1;
			break;

			case 'v':
				fprintf (stdout, "%s", IonVersion::GetFullVersion("Crop").c_str());
				exit (0);
			break;

			case 'H':
			default:
				argcc++;
				fprintf (stdout, "\n");
				fprintf (stdout, "Ranger - Utility to get the range of pixel values durring an experiment.\n");
				fprintf (stdout, "options:\n");
				fprintf (stdout, "   -a\tOutput flat files; ascii text\n");
				fprintf (stdout, "   -x\tStarting x axis position (origin lower left) Default: %d\n",cropx);
				fprintf (stdout, "   -y\tStarting y axis position (origin lower left) Default: %d\n",cropy);
				fprintf (stdout, "   -w\tWidth of crop region Default: %d\n",cropw);
				fprintf (stdout, "   -h\tHeight of crop region Default: %d\n",croph);
				fprintf (stdout, "   -s\tSource directory containing raw data\n");
				fprintf (stdout, "   -f\tConverts only the one file named as an argument\n");
				fprintf (stdout, "   -z\tTells the image loader not to wait for a non-existent file\n");
				fprintf (stdout, "   -H\tPrints this message and exits.\n");
				fprintf (stdout, "   -v\tPrints version information and exits.\n");
				fprintf (stdout, "\n");
				fprintf (stdout, "usage:\n");
				fprintf (stdout, "   Crop -s /results/analysis/PGM/testRun1\n");
				fprintf (stdout, "\n");
				exit (1);
		}
		argcc++;
	}

	char name[256];
	char destName[256];
	int i,j,k;
	Image loader;
	Acq saver;
	int mode = 0;
	i = 0;
	bool allocate = true;
	char **nameList;
	char *defaultNameList[] = {/*"beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
				"beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
				"prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat", "prerun_0004.dat"*/};
	int nameListLen;
	uint16_t *minArray=NULL;
	uint16_t *maxArray=NULL;
	uint16_t *minptr,*maxptr,*fptr;
	int rows=0;
	int cols=0;
	// if requested...do not bother waiting for the files to show up
	if (dont_retry)
		loader.SetTimeout(1,1);

	if (oneFile != NULL)
	{
		nameList = &oneFile;
		nameListLen = 1;
		mode = 1;
	}
	else
	{
		nameList = defaultNameList;
		nameListLen = sizeof(defaultNameList)/sizeof(defaultNameList[0]);
	}

#if 0
	// Create results folder
	umask (0);	// make permissive permissions so its easy to delete.
    if (mkdir (destPath, 0777))
    {
        if (errno == EEXIST) {
            //already exists? well okay...
        }
        else {
            perror (destPath);
            exit (1);
        }
    }

	// Copy explog.txt file: all .txt files
	char cmd[1024];
	sprintf (cmd, "cp -v %s/*.txt %s", expPath, destPath);
	assert(system(cmd) == 0);
#endif

	while (mode < 2) {
		if (mode == 0) {
			sprintf(name, "%s/acq_%04d.dat", expPath, i);
			sprintf(destName, "%s/acq_%04d.dat", destPath, i);
		} else if (mode == 1) {
			if(i >= nameListLen)
				break;
			sprintf(name, "%s/%s", expPath, nameList[i]);
			sprintf(destName, "%s/%s", destPath, nameList[i]);
		} else
			break;
		if (loader.LoadRaw(name, 0, allocate, false)) {
			allocate = false;
			const RawImage *raw = loader.GetImage();


			fptr = (uint16_t *)raw->image;
			if(minArray == NULL)
			{
				rows = raw->rows;
				cols = raw->cols;
				minArray = (uint16_t *)malloc(cols*rows*sizeof(uint16_t));
				maxArray = (uint16_t *)malloc(cols*rows*sizeof(uint16_t));

				memset(maxArray,0,cols*rows*sizeof(uint16_t));
				memset(minArray,0xff,cols*rows*sizeof(uint16_t));
			}
			for(j=0;j<raw->frames;j++)
			{
				minptr = minArray;
				maxptr = maxArray;
				for(k=0;k<(rows*cols);k++)
				{
					// check this frame
					if(*fptr < *minptr)
						*minptr = *fptr;
					if(*fptr > *maxptr)
						*maxptr = *fptr;

					fptr++;
					minptr++;
					maxptr++;
				}
			}

#if 0
			printf("Converting raw data %d %d frames: %d\n", raw->cols, raw->rows, raw->frames);
			saver.SetData(raw->cols, raw->rows, raw->frames, (unsigned short *)raw->image, (int *)raw->timestamps);
			if (doAscii) {
				if (!saver.WriteAscii(destName, cropx, cropy, cropw, croph))
					break;
			}
			else {
				if (!saver.Write(destName, cropx, cropy, cropw, croph))
					break;
			}
#endif
			printf("Read: %s\n", name);
			i++;
		} else {
			if ((mode == 1 && i >= 12) || (mode == 0)) {
				mode++;
				i = 0;
				allocate = true;
			} else
				i++;
		}
	}
	// we now have the min and max arrays...  bin them.
#define NUMBINS 20

	minptr = minArray;
	maxptr = maxArray;
	int index;
	uint32_t val;
	uint32_t minBins[NUMBINS+2];
	uint32_t maxBins[NUMBINS+2];
	memset(minBins,0,sizeof(minBins));
	memset(maxBins,0,sizeof(maxBins));

	for(k=0;k<(rows*cols);k++)
	{
		val = *minptr;
		index = (val*NUMBINS)/16384;
		if(index > NUMBINS)
			index = NUMBINS;

		if(val == 0)
			minBins[0]++;
		else
			minBins[index+1]++;

		val = *maxptr;
		index = (val*NUMBINS)/16384;
		if(index > NUMBINS)
			index = NUMBINS;
		if(val == 0)
			maxBins[0]++;
		else
			maxBins[index+1]++;

		minptr++;
		maxptr++;
	}

	printf("Bins\n\n");
	for(k=0;k<=NUMBINS;k++)
	{
		printf("%05d\n",16384*k/NUMBINS/*,16384*(k+1)/NUMBINS*/);
	}
	printf("\n\n");

	printf("min histogram\n\n");
	for(k=0;k<=NUMBINS;k++)
	{
		printf("%d\n",minBins[k]);
	}
	printf("\n\n");

	printf("max histogram\n\n");
	for(k=0;k<=NUMBINS;k++)
	{
		printf("%d\n",maxBins[k]);
	}
	printf("\n\n");
}

