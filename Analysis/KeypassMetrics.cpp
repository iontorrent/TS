/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// KeypassMetrics - simple util for extracting well traces over multiple acquisitions and summarizing info on it
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "Image.h"
#include "Mask.h"
#include "Region.h"
#include "Zeromer.h"
#include "LinuxCompat.h"
#include "IonVersion.h"


void TrimWhiteSpace(char *buf)
{
	int len = strlen(buf);
	while (len > 1 && (buf[len-1] == '\r' || buf[len-1] == '\n' || buf[len-1] == ' '))
		len--;
	buf[len] = 0;
}

int printHelp ()
{
	fprintf (stdout,"\n");
	fprintf (stdout,"KeypassMetrics - Simple util for extracting well traces over multiple acquisitions and summarizing info on it.\n");
	fprintf (stdout,"options:\n");
	fprintf (stdout,"   -b\tOverride default (MaskEmpty) background type.  Only valid argument is 'd'.\n");
	fprintf (stdout,"   -e\tSpecify raw data location.  This is normally determined automatically from the analysis output files.\n");
	fprintf (stdout,"   -h\tPrint this help information and exit.\n");
	fprintf (stdout,"   -i\tDump info on reads to file individuals.txt\n");
	fprintf (stdout,"   -k\tOverride the default (TCAG) key.\n");
	fprintf (stdout,"   -m\tSet MaskType to dump stats for.  Valid arguments are 'MaskTF' or 'MaskLib'.\n");
	fprintf (stdout,"   -n\tTurn off neighbor subtract.\n");
	fprintf (stdout,"   -r\tDirectory to pull report data files from (Default is ./).\n");
	fprintf (stdout,"   -v\tPrint version information and exit.\n");
	fprintf (stdout,"\n");
	fprintf (stdout,"usage:\n   KeypassMetrics -i -m MaskLib -k TCAG -r ${ANALYSIS_DIR}/ -e ${RAW_DATA_DIR}\n");
	fprintf (stdout,"\n");
	return (0);
}

int main(int argc, char *argv[])
{
	char *flowOrder = "TACG";
	int numKeyFlows = 8;
	char *key = "TCAG";
	char *reportDir = ".";
	char *acqPrefix = "acq_";
	char expDir[MAX_PATH_LENGTH];
	int numFrames = 100; // MGD - will want to change this to be in seconds - so its file dependent
	strcpy(expDir, ".");
	bool nsub = true;
	bool individual = false;
	MaskType useThese = MaskLib;

	MaskType backgroundType = MaskEmpty;

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'r': // directory to pull report data from (bfmask.bin)
				argcc++;
				reportDir = argv[argcc];
			break;

			case 'b': // backgroud type
				argcc++;
				switch (argv[argcc][0]) {
					case 'd': // duds
						backgroundType = MaskDud;
						printf("Processing with dud background.\n");
					break;
				}
			break;

			case 'n': // no neighbor subtract
				nsub = false;
			break;

			case 'i': // also dump info for individual reads
				individual = true;
			break;

			case 'k': // set the key
				argcc++;
				key = argv[argcc];
			break;

			case 'm': // set the mask type to dump stats for
				argcc++;
				if (strcmp(argv[argcc], "MaskTF")==0)
					useThese = MaskTF;
				else if (strcmp(argv[argcc], "MaskLib")==0)
					useThese = MaskLib;
			break;

			case 'e': // experiment raw directory
				argcc++;
				strcpy(expDir, argv[argcc]);
			break;
			
			case 'h':
				printHelp();
				exit (0);
			break;
			case 'v':
				fprintf (stdout, "%s", IonVersion::GetFullVersion("KeypassMetrics").c_str());
				exit (0);
			break;
		}
		argcc++;
	}

	// we only care about keypassed reads
	//THIS DOES NOT WORK WITH CURRENT Mask::Match method
	//useThese = (MaskType)(useThese | MaskKeypass);

	// see if we can determine expDir dir from report dir
	if (expDir[0] == '.') {
		char processParamsFile[MAX_PATH_LENGTH];
		sprintf(processParamsFile, "%s/processParameters.txt", reportDir);
		FILE *fp = fopen(processParamsFile, "r");
		if (fp) {
			char buf[1024];
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
		} else {
			perror (processParamsFile);
			return (1);
		}
	}

	printf("Raw data location: %s\n", expDir);

	// crazy, but only way to get rows/cols right now is from mask.
	Mask mask(1,1);
	char maskPath[MAX_PATH_LENGTH];
	sprintf(maskPath, "%s/bfmask.bin", reportDir);
	if (mask.SetMask(maskPath)){
		perror (maskPath);
		return (1);
	}

	int numRegionsX = 13;
	int numRegionsY = 12;
	int regionW = ceil((double)mask.W()/(double)numRegionsX);
	int regionH = ceil((double)mask.H()/(double)numRegionsY);

/*
	int regionW = 100;
	int regionH = 100;
	int numRegionsX = ceil((double)mask.W()/(double)regionW);
	int numRegionsY = ceil((double)mask.H()/(double)regionH);
*/

	int numRegions = numRegionsX * numRegionsY;
	Region regions[numRegions];
	double *individualAvg = new double[mask.W()*mask.H()];
	memset(individualAvg, 0, sizeof(double) * mask.W()*mask.H());
	int i, x, y;
	for(i = 0, x=0;x<mask.W();x+=regionW) {
		for(y=0;y<mask.H();y+=regionH) {
			regions[i].col = x;
			regions[i].row = y;
			regions[i].w = regionW;
			regions[i].h = regionH;
			if (regions[i].col + regions[i].w > mask.W())
				regions[i].w = mask.W() - regions[i].col;
			if (regions[i].row + regions[i].h > mask.H())
				regions[i].h = mask.H() - regions[i].row;
			i++;
		}
	}


	// make the ionogram for the key
	int ionogram[numKeyFlows];
	seqToFlow(key,strlen(key),ionogram,numKeyFlows,flowOrder,strlen(flowOrder));


	// for each flow, make an avg trace from all keypassed library beads
	char acqFileName[MAX_PATH_LENGTH];
	Image   img[2];
	Zeromer avgTrace(numRegions, numKeyFlows, numFrames);
	Zeromer avgDudTrace(numRegions, numKeyFlows, numFrames);
	int r;

	printf("Loading raw acq files...\n");

	int nuc;
	for(nuc=0;nuc<4;nuc++) {
		int zeromer = -1;
		int onemer = -1;
		if (ionogram[nuc] == 1) {
			onemer = nuc;
			zeromer = nuc+4;
		} else {
			zeromer = nuc;
			onemer = nuc+4;
		}

		Mask localMask0(&mask);
		sprintf(acqFileName, "%s/%s%04d.dat", expDir, acqPrefix, zeromer);
		img[0].LoadRaw(acqFileName);
		img[0].FilterForPinned(&localMask0, MaskEmpty);
		img[0].Normalize(0, 5);
		if (nsub)
			img[0].BackgroundCorrect(&localMask0, MaskBead, MaskEmpty, 2, 5, NULL);
		Mask localMask1(&mask);
		sprintf(acqFileName, "%s/%s%04d.dat", expDir, acqPrefix, onemer);
		if (!img[1].LoadRaw(acqFileName)){
			perror (acqFileName);
            delete [] individualAvg;
			return (1);
		}
		img[1].FilterForPinned(&localMask1, MaskEmpty);
		img[1].Normalize(0, 5);
		if (nsub)
			img[1].BackgroundCorrect(&localMask1, MaskBead, MaskEmpty, 2, 5, NULL);

		for(r=0;r<numRegions;r++) {
			avgTrace.CalcAverageTrace(&img[0], &regions[r], r, &mask, useThese, zeromer);
			avgDudTrace.CalcAverageTrace(&img[0], &regions[r], r, &mask, MaskDud, zeromer);
			avgTrace.CalcAverageTrace(&img[1], &regions[r], r, &mask, useThese, onemer);
			avgDudTrace.CalcAverageTrace(&img[1], &regions[r], r, &mask, MaskDud, onemer);
		}

		// update avg peak 1-mers for individual reads
		if (individual && (onemer != 7)) { // only look at 1-mers first 7 flows
			Image *img0 = &img[0];
			Image *img1 = &img[1];
			int x, y;
			for(y=0;y<mask.H();y++) {
				for(x=0;x<mask.W();x++) {
					int k = x+y*mask.W();
					if (mask.Match(k, useThese) && mask.Match(k,MaskKeypass)) {
						double peak = 0.0;
						double curPeak = 0.0;
						int frame;
						for(frame=0;frame<numFrames;frame++) {
							curPeak = img1->GetInterpolatedValue(frame,x,y) - img0->GetInterpolatedValue(frame,x,y);
							if (curPeak > peak)
								peak = curPeak;
						}
						individualAvg[k] += 0.33333*peak;
					}
				}
			}
		}
	}

	// now calc some stats on the avg traces we gathered
	int frame;
	double trace[numFrames];
	for(nuc=0;nuc<4;nuc++) {
		printf("Nuc: %c\n", flowOrder[nuc]);
		// for the given nuc, find the 0-mer and 1-mer flow indexes
		int zeromer = -1;
		int onemer = -1;
		if (ionogram[nuc] == 1) {
			onemer = nuc;
			zeromer = nuc+4;
		} else {
			zeromer = nuc;
			onemer = nuc+4;
		}

		double *zeromerTrace;
		double *onemerTrace;
		double *dudTrace;
		for(r=0;r<numRegions;r++) {
			zeromerTrace = avgTrace.GetAverageZeromer(r, zeromer);
			onemerTrace = avgTrace.GetAverageZeromer(r, onemer);
			dudTrace = avgDudTrace.GetAverageZeromer(r, onemer);
			printf("Region: %d\n", r);
			for(frame=0;frame<numFrames;frame++) {
				if (backgroundType == MaskEmpty)
					trace[frame] = onemerTrace[frame] - zeromerTrace[frame];
				else if (backgroundType == MaskDud)
					trace[frame] = onemerTrace[frame] - dudTrace[frame];
				printf("%.3lf ", trace[frame]);
			}
			printf("\n");
		}
	}

	// dump info on individual wells
	if (individual) {
		FILE *individuals_fp = fopen("individuals.txt", "w");
		int x, y;
		for(y=0;y<mask.H();y++) {
			for(x=0;x<mask.W();x++) {
				int k = x+y*mask.W();
				if (mask.Match(k, useThese) && mask.Match(k,MaskKeypass)) {
					fprintf(individuals_fp, "fubar:%d:%d %.4lf\n", y, x, individualAvg[k]);
				}
			}
		}
		fclose(individuals_fp);
	}

    delete [] individualAvg;

	printf("Done.\n");
}

