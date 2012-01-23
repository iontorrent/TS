/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// GetTraces - simple util for extracting one or more single well traces over multiple acquisitions
//             mainly used for generation of data for golden beads
//             also a good unit-test of the Mask & Image classes
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "Image.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "Utils.h"

int LoadMask(Mask &mask, char *fileName, int version, MaskType withThese)
{
	int beadsFound = 0;
    int n = 0;

	int w, h;
	w = mask.W();
	h = mask.H();

	FILE *fp = NULL;
	fopen_s(&fp, fileName, "r");
	if (fp) {
		int x, y;
		// read the XY offset and the size of the input mask
		x = 0; y = 0;
		if (version == 1) { // version 1 masks are from TorrentExplorer, so they are not full-chip sized
			double _x, _y;
			n = fscanf_s(fp, "%lf,%lf\n", &_x, &_y);
            assert(n == 2);
			x = (int)_x; y = (int)_y;
			n = fscanf_s(fp, "%d,%d\n", &w, &h);
            assert(n==2);
		}
		int i,j;
		int val;
		for(j=0;j<h;j++) {
			for(i=0;i<w;i++) {
				if (i < (w-1)) {
					n = fscanf_s(fp, "%d,", &val); // comma scanned in, but seems to also work with just a space in a mask input file
                    assert(n==1);
				} else {
					n = fscanf_s(fp, "%d\n", &val); // last entry per row has no comma
                    assert(n==1);
                }
				if (val > 0) {
					// if (val == 64)
						// mask[(i+x)+(j+y)*mask.W()] |= MaskBead;
					// else
						mask[(i+x)+(j+y)*mask.W()] |= withThese;
					beadsFound++;
				}
			}
		}
		fclose(fp);
	}
	else
	{
		fprintf (stderr, "%s: %s\n", fileName, strerror(errno));
	}

	return beadsFound;
}

int main(int argc, char *argv[])
{
	int numCycles = 2;//int numCycles = 25;
	int numFlows = numCycles*4;
	int numFrames = 100;
	int hasWashFlow = 1;

    int n; // number of elements read

	// inputs:
	//    beadmask
	//    ignore mask
	//    target mask
	//    weka vector

	// for each flow:
	// X.  memmap acq file
	// 1.  save raw traces for each target bead
	// 2.  calc and save nsub trace for each target bead

	// for each target bead:
	// X.  write out all raw traces for all flows
	// 1.  write out (raw - nsub) for all flows
	// 2.  write out (raw - nsub) - (0-mer[nuc] - nsub) for all flows
	// 3.  write out weka*(raw - nsub) for all flows

	// char *beadMaskName = "Beads300x300";
	// char *targetMaskName = "Golden";
	// char *wekaVector = NULL;
	char *coordFile = "/home/ion/JD1114.blastn.coords";

	// set a few defaults
	char *dirExt = ".";

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'd': // directory to pull raw data from
				argcc++;
				dirExt = argv[argcc];
			break;

			case 'f': // coord file
				argcc++;
				coordFile = argv[argcc];
			break;
		}
		argcc++;
	}

        // crazy, but only way to get rows/cols right now is from mask.
        Mask mask(1,1);
	char maskPath[MAX_PATH_LENGTH];
	sprintf(maskPath, "%s/bfmask.bin", dirExt);
        // mask.SetMask(maskPath);
        mask.SetMask("/results/analysis/output/IonEast/Flux_v.029_VZ_069/bfmask.bin");

	// for each flow, make an avg trace from all row/col's requested

	int flow;
	char acqFileName[MAX_PATH_LENGTH];
	char *acqPrefix = "acq_";
	Image   img;
	img.SetMaxFrames(numFrames);
	int frame;
	double beadTrace[numFlows][numFrames];
	memset(beadTrace, 0, sizeof(beadTrace));
	for(flow=0;flow<numFlows;flow++) {
		Mask localMask(&mask);
		sprintf(acqFileName, "%s/%s%04d.dat", dirExt, acqPrefix, (flow + (flow/4)*hasWashFlow));
		img.LoadRaw(acqFileName);
		img.FilterForPinned(&localMask, MaskEmpty);
		img.Normalize(0, 5);
		img.BackgroundCorrect(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);
		//img.BackgroundCorrectMulti(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);

		const RawImage *raw = img.GetImage();
		int frameStride = raw->rows*raw->cols;

		// FILE *fp = fopen("/home/ion/aligReadsPlusCords", "r");
		FILE *fp = fopen(coordFile, "r");
		char line[512];
		int row, col;
		int count = 0;
		while(fgets(line, sizeof(line), fp) != NULL) {
			n = sscanf(line, "%d %d", &row, &col);
            assert(n==2);
			count++;
			for(frame=0;frame<numFrames;frame++) {
				beadTrace[flow][frame] += raw->image[col+row*raw->cols+frame*frameStride];
			}
		}
		fclose(fp);
		for(frame=0;frame<numFrames;frame++) {
			beadTrace[flow][frame] /= count;
		}
	}

	// now dump the avg traces
	for(flow=0;flow<numFlows;flow++) {
		for(frame=0;frame<numFrames;frame++) {
			printf("%.2lf\t", beadTrace[flow][frame]);
		}
		printf("\n");
	}
	printf("Done.\n");

#ifdef NOT_USED_RIGHT_NOW

	// char *ignoreMaskName = "Exclude300x300";
	char *ignoreMaskName = "all2430";

	// read in masks
	Mask targetMask(1348, 1152);
	int numBeads = 5;
	// LoadMask(targetMask, targetMaskName, 1, MaskBead);
	LoadMask(targetMask, ignoreMaskName, 1, MaskIgnore);
/*
	sprintf (fileName, "%s/%s", dirExt, targetMaskName);
	int numBeads = LoadMask(targetMask, fileName, 1, MaskBead);
	if (numBeads <= 0)
	{
		fprintf (stderr, "No beads loaded!\n\n");
		return (1);
	}
	sprintf (fileName, "%s/%s", dirExt, ignoreMaskName);
	LoadMask(targetMask, fileName, 1, MaskIgnore);
*/
	int ox = 1348/2-50;
	int oy = 1150/2 - 450;

	targetMask[(5+oy)*1348 + 19+ox] |= MaskBead;
	targetMask[(10+oy)*1348 + 17+ox] |= MaskBead;
	targetMask[(20+oy)*1348 + 66+ox] |= MaskBead;
	targetMask[(27+oy)*1348 + 18+ox] |= MaskBead;
	targetMask[(30+oy)*1348 + 69+ox] |= MaskBead;

	printf("Analyzing %d beads.\n", numBeads);

	// make array of bead XY positions
	int *beadx = new int[numBeads];
	int *beady = new int[numBeads];
	int i;
	int k = 0;
	int x, y;
	for(i=0;i<targetMask.W()*targetMask.H();i++) {
		x = i%targetMask.W();
		y = i/targetMask.W();
		if (targetMask[i] &  MaskBead) {
			beadx[k] = i%targetMask.W();
			beady[k] = i/targetMask.W();
printf("Bead %d as (%d,%d)\n", k, beadx[k], beady[k]);
			k++;
		}
	}

printf("Allocating...\n");
	// allocate trace storage
	short **beadTrace = new short *[numBeads*numFlows];
	for(i=0;i<numBeads*numFlows;i++)
		beadTrace[i] = new short[numFrames];

	// loop through all flows
	// load up acq file and extract nsub data for the target beads
	int flow;
	char acqFileName[MAX_PATH_LENGTH];
	char *acqPrefix = "acq_";
	Image   img;
	img.SetMaxFrames(numFrames);
	int bead;
	int frame;
	for(flow=0;flow<numFlows;flow++) {
		Mask localMask(&targetMask);
		sprintf(acqFileName, "%s/%s%04d.dat", dirExt, acqPrefix, (flow + (flow/4)*hasWashFlow));
		img.LoadRaw(acqFileName);
		img.FilterForPinned(&localMask, MaskEmpty);
		img.Normalize(0, 5);
		img.BackgroundCorrect(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);

		// save off our nsub traces
printf("Saving bead nsub traces...\n");
		const RawImage *raw = img.GetImage();
		int frameStride = raw->rows*raw->cols;
		for(bead=0;bead<numBeads;bead++) {
			for(frame=0;frame<numFrames;frame++) {
				beadTrace[bead+flow*numBeads][frame] = raw->image[beadx[bead]+beady[bead]*raw->cols+frame*frameStride];
			}
		}
	}

	// int zeromer[4] = {0, 5, 2, 3}; // TF key: ATCG
	 int zeromer[4] = {4, 1, 6, 3}; // TF key: TCAG
	short *beadzero = new short[numFrames];

	// write out bead raw 0-mer subtracted data
	for(bead=0;bead<numBeads;bead++) {
		printf("Bead: %d at (%d,%d)\n", bead, beadx[bead], beady[bead]);
		for(flow=0;flow<numFlows;flow++) {
			// first generate a smoothed 0-mer to subtract from... (grossly inefficient since I really only need to calc this 4 times)
			img.SGFilterApply(beadTrace[bead+zeromer[flow%4]*numBeads], beadzero);
			for(frame=0;frame<numFrames;frame++) {
				printf("%d", beadTrace[bead+flow*numBeads][frame] - beadzero[frame]);
				if (frame < (numFrames-1))
					printf(",");
			}
			printf("\n");
		}
	}

	// write out bead smoothed 0-mer subtracted data
	short *smoothedTraceIn = new short[numFrames];
	short *smoothedTraceOut = new short[numFrames];
	for(bead=0;bead<numBeads;bead++) {
		printf("Smoothed Bead: %d at (%d,%d)\n", bead, beadx[bead], beady[bead]);
		for(flow=0;flow<numFlows;flow++) {
			// first generate a smoothed 0-mer to subtract from... (grossly inefficient since I really only need to calc this 4 times)
			img.SGFilterSet(2, 1); // mild smoothing
			img.SGFilterApply(beadTrace[bead+zeromer[flow%4]*numBeads], beadzero);

			// now smooth the 0-mer subtracted trace
			for(frame=0;frame<numFrames;frame++) {
				smoothedTraceIn[frame] = beadTrace[bead+flow*numBeads][frame] - beadzero[frame];
			}
			img.SGFilterSet(4, 2); // moderate smoothing
			img.SGFilterApply(smoothedTraceIn, smoothedTraceOut);

			// now plot the trace
			for(frame=0;frame<numFrames;frame++) {
				printf("%d", smoothedTraceOut[frame]);
				if (frame < (numFrames-1))
					printf(",");
			}
			printf("\n");
		}
	}

	// load up our weka filter
	float *filterVec = new float[numFrames];
	FILE *fp = fopen_s(&fp, "Weights.csv", "r");
	if (fp) {
		char line[4096];
		// skip the first line
		fgets(line, sizeof(line), fp);
		// skip the first 3 values
		float dummy;
		n = fscanf(fp, "%f,", &dummy);
        assert(n==1);
		n = fscanf(fp, "%f,", &dummy);
        assert(n==1);
		n = fscanf(fp, "%f,", &dummy);
        assert(n==1);
		// read in the vector
		for(frame=0;frame<numFrames;frame++) {
			n = fscanf(fp, "%f,", &filterVec[frame]);
            assert(n==1);
		}
		fclose(fp);
		printf("Filter vector:\n");
		for(frame=0;frame<numFrames;frame++) {
			printf("%.3f ", filterVec[frame]);
		}
		printf("\n");
	}

	// write out weka-vector applied & smoothed traces
	float *filteredTraceIn = new float[numFrames];
	float *filteredTraceOut = new float[numFrames];
	for(bead=0;bead<numBeads;bead++) {
		printf("Filtered Bead: %d at (%d,%d)\n", bead, beadx[bead], beady[bead]);
		for(flow=0;flow<numFlows;flow++) {
			// first generate a smoothed 0-mer to subtract from... (grossly inefficient since I really only need to calc this 4 times)
			img.SGFilterSet(2, 1); // mild smoothing
			img.SGFilterApply(beadTrace[bead+zeromer[flow%4]*numBeads], beadzero);

			// now filter the 0-mer subtracted trace
			for(frame=0;frame<numFrames;frame++) {
				smoothedTraceIn[frame] = beadTrace[bead+flow*numBeads][frame] - beadzero[frame];
				filteredTraceIn[frame] = smoothedTraceIn[frame] * filterVec[frame];
			}

			// now smooth the filtered 0-mer subtracted trace
			img.SGFilterSet(4, 2); // moderate smoothing
			img.SGFilterApply(filteredTraceIn, filteredTraceOut);

			// now plot the trace
			for(frame=0;frame<numFrames;frame++) {
				printf("%.3f", filteredTraceOut[frame]);
				if (frame < (numFrames-1))
					printf(",");
			}
			printf("\n");
		}
	}

	// cleanups
	delete [] beadzero;
	delete [] smoothedTraceIn;
	delete [] smoothedTraceOut;
	delete [] filterVec;
	delete [] filteredTraceIn;
	delete [] filteredTraceOut;
    delete [] beadx;
    delete [] beady;
#endif /* NOT_USED_RIGHT_NOW */
}

