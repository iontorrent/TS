/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// GetRawTraces - simple util for extracting well traces
//

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "Image.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "Utils.h"

void TrimWhiteSpace(char *buf)
{
	int len = strlen(buf);
	while (len > 1 && (buf[len-1] == '\r' || buf[len-1] == '\n' || buf[len-1] == ' '))
		len--;
	buf[len] = 0;
}

int main(int argc, char *argv[])
{
	int flowNum = 0;
	char *reportDir = ".";
	char *acqPrefix = "acq_";
	char expDir[MAX_PATH_LENGTH];
	strcpy(expDir, ".");
	bool bsub = false;
	bool normalize = false;

	int rx=200, ry=200, rw=10, rh=10;

	MaskType useThese = MaskNone; // MaskLib;

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'r': // directory to pull report data from (bfmask.bin)
				argcc++;
				reportDir = argv[argcc];
			break;

			case 'b': // background subtract
				bsub = true;
			break;

			case 'n':
				normalize = true;
			break;

			case 'm': // set the mask type to dump stats for
				argcc++;
				if (strcmp(argv[argcc], "MaskTF")==0)
					useThese = (MaskType)(useThese | MaskTF);
				else if (strcmp(argv[argcc], "MaskLib")==0)
					useThese = (MaskType)(useThese | MaskLib);
				else if (strcmp(argv[argcc], "MaskEmpty")==0)
					useThese = (MaskType)(useThese | MaskEmpty);
				else if (strcmp(argv[argcc], "MaskKeypass")==0)
					useThese = (MaskType)(useThese | MaskKeypass);
			break;

			case 'l': // location on chip
				argcc++;
				sscanf(argv[argcc], "%dx%dx%dx%d", &rx, &ry, &rw, &rh);
			break;

			case 'f': // flow number to dump
				argcc++;
				sscanf(argv[argcc], "%d", &flowNum);
			break;
		}
		argcc++;
	}

        // see if we can determine expDir dir from report dir
        char processParamsFile[MAX_PATH_LENGTH];
        sprintf(processParamsFile, "%s/processParameters.txt", reportDir);
        FILE *fp = fopen(processParamsFile, "r");
        if (fp) {
                char buf[MAX_PATH_LENGTH];
                assert(fgets(buf, sizeof(buf), fp));
		if (strncmp(buf, "Command line =", 14) == 0) {
			TrimWhiteSpace(buf);
			char *ptr = strrchr(buf, ' ');
			if (ptr)
				strcpy(expDir, ptr+1);
			else
				strcpy(expDir, ".");
		} else {
                	assert(fgets(buf, sizeof(buf), fp));
                	strcpy(expDir, &buf[16]);
			TrimWhiteSpace(expDir);
		}
                fclose(fp);
        }
	printf("Raw data location: %s\n", expDir);

        // crazy, but only way to get rows/cols right now is from mask.
        Mask mask(1,1);
	char maskPath[MAX_PATH_LENGTH];
	sprintf(maskPath, "%s/bfmask.bin", reportDir);
        mask.SetMask(maskPath);

	Mask localMask(&mask);
	char acqFileName[MAX_PATH_LENGTH];
	sprintf(acqFileName, "%s/%s%04d.dat", expDir, acqPrefix, flowNum);
	Image img;
	img.LoadRaw(acqFileName);
	img.FilterForPinned(&localMask, MaskEmpty);
	if (normalize)
		img.Normalize(0, 5);
	if (bsub)
		img.BackgroundCorrect(&localMask, useThese, MaskEmpty, 2, 5, NULL);
	const RawImage *raw = img.GetImage();

	int numFrames = raw->frames;
	double avgTrace[numFrames];
	memset(avgTrace, 0, sizeof(avgTrace));

	// set up our regions
	int xinc = 100;
	int yinc = 100;
	int cols = mask.W();
	int rows = mask.H();
	int regionsX = cols/xinc;
	int regionsY = rows/yinc;
	// make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
	if (((double)cols/(double)xinc) != regionsX)
		regionsX++;
	if (((double)rows/(double)yinc) != regionsY)
		regionsY++;
	int numRegions = regionsX*regionsY;
	Region regions[numRegions];
	int i, x, y;
	for(i = 0, x=0;x<cols;x+=xinc) {
		for(y=0;y<rows;y+=yinc) {
			regions[i].col = x;
			regions[i].row = y;
			regions[i].w = xinc;
			regions[i].h = yinc;
			if (regions[i].col + regions[i].w > cols) // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
				regions[i].w = cols - regions[i].col; // but better to be safe!
			if (regions[i].row + regions[i].h > rows)
				regions[i].h = rows - regions[i].row;
			i++;
		}
	}

        // Average the traces for this region, this flow
	int r;
	for(r=0;r<numRegions;r++) {
		Region *region = &regions[r];
        	int cnt = 0;
        	memset(avgTrace, 0, sizeof(double)*raw->frames);
        	for (int y=region->row;y<(region->row+region->h);y++) {
                	for (int x=region->col;x<(region->col+region->w);x++) {
                        	if (localMask[x+(y*raw->cols)] & (MaskIgnore|MaskWashout)) {
                                	continue;
                        	}
                        	if (localMask[x+(y*raw->cols)] & useThese) {
                                	for (int frame = 0; frame < numFrames; frame++) {
                                        	avgTrace[frame] += raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
                                	}
                                	cnt++;
                        	}
			}
                }

		printf("region: %d\n", r);
        	if (cnt > 0) {
                	for (int frame = 0; frame < numFrames; frame++) {
                        	avgTrace[frame] /= cnt;
				printf("%.5lf ", avgTrace[frame]);
                	}
        	}
		printf("\n");
	}

}

