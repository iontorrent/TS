/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// HeatMaps - simple util for extracting some metric over the whole-chip and plotting
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

#include "Image.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "Region.h"
#include "Separator.h"

#include "dbgmem.h"

int LoadMask(Mask &mask, char *fileName, int version, MaskType withThese)
{
	int beadsFound = 0;

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
			assert(fscanf_s(fp, "%lf,%lf\n", &_x, &_y) == 2);
			x = (int)_x; y = (int)_y;
			assert(fscanf_s(fp, "%d,%d\n", &w, &h) == 2);
		}
		int i,j;
		int val;
		for(j=0;j<h;j++) {
			for(i=0;i<w;i++) {
				if (i < (w-1))
					assert(fscanf_s(fp, "%d,", &val) == 1); // comma scanned in, but seems to also work with just a space in a mask input file
				else
					assert(fscanf_s(fp, "%d\n", &val) == 1); // last entry per row has no comma
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

void ConvertMap(short *srcHeatMap, int w, int h, int dx, int dy, int ww, int wh, short *destHeatMap, int mode)
{
	int x, y;
	int wx, wy;
	int wxlocal, wylocal;
        for(y=0;y<dy;y++) {
                for(x=0;x<dx;x++) {
                        wx = x*w/dx;
                        wy = y*h/dy;
                        if (wx >= (w-ww))
                                wx = w-ww-1;
                        if (wy >= (h-wh))
                                wy = h-wh-1;
                        double density = 0;
                        int densityCount = 0;
                        for(wylocal=wy;wylocal<(wy+wh);wylocal++) {
                                for(wxlocal=wx;wxlocal<(wx+ww);wxlocal++) {
                                        if (wxlocal < w && wylocal < h) {
                                                if (srcHeatMap[wxlocal+wylocal*w] > 0) {
                                                        density = density + srcHeatMap[wxlocal+wylocal*w];
                                                        densityCount++;
                                                }
                                        }
                                }
                        }
                        if (densityCount > 0) {
				if (mode == 1)
                                	density /= densityCount;
				else if (mode == 2) {
					density /= (ww*wh);
					density *= 1000; // convert to integer percentage with 0.1% resolution
				} else
                                	density = 1;
                                destHeatMap[x+y*dx] = (short)(density+0.5);
                        } else
                                destHeatMap[x+y*dx] = 0;
                }
        }
}

#ifdef _DEBUG
void memstatus(void)
{
	memdump();
}

#endif /* _DEBUG */

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	atexit(memstatus);
	dbgmemInit();
#endif /* _DEBUG */

	int numFrames = 100;
	int hasWashFlow = 1;

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

	// set a few defaults
	char *dirExt = ".";
	char *flowOrder = "TACG";
	char *keySeq = "TCAG";
	char *baseToMap = "T";

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'd': // directory to pull raw data from
				argcc++;
				dirExt = argv[argcc];
			break;

			case 'k': // key
				argcc++;
				keySeq = argv[argcc];
			break;

			case 'b':
				argcc++;
				baseToMap = argv[argcc];
			break;
		}
		argcc++;
	}

        // a few globals
        Image   img;
	img.SetDir(dirExt);
	img.SetMaxFrames(numFrames);
	char beadfindFile[512];
	sprintf(beadfindFile, "%s/%s", dirExt, "beadfind_post_0003.dat");
        img.LoadRaw(beadfindFile, 0, false, true); // only load up the header
        // grab rows & cols here - as good a place as any
        int rows = img.GetImage()->rows;
        int cols = img.GetImage()->cols;

	// set up vectors to hold the expected incorporations
	int incorporations[1000];
	int bases = strlen(keySeq);
	int base = 0;
	int flow = 0;
	while (base < bases) {
		while (keySeq[base] != flowOrder[flow%4]) {
			incorporations[flow] = 0;
			flow++;
		}
		incorporations[flow] = 1;
		flow++;
		base++;
	}
	int numKeyFlows = flow;
	for(flow=0;flow<numKeyFlows;flow++)
		printf("%d", incorporations[flow]);
	printf("\n");

	// find the index for the 0-mer and the 1-mer for the base we want to heatmap
	int zeromer = 0, onemer = 0;
	int i;
	for(i=0;i<numKeyFlows;i++) {
		if (flowOrder[i%4] == baseToMap[0] && incorporations[i] == 0)
			zeromer = i;
		if (flowOrder[i%4] == baseToMap[0] && incorporations[i] == 1)
			onemer = i;
	}
	printf("0-mer: %d\n1-mer: %d\n", zeromer, onemer);

	// create a heatmap
	short *heatMap = new short[rows*cols];
	memset(heatMap, 0, sizeof(short) * rows * cols);

	// fixed region size
	int xinc = 100;
	int yinc = 100;
	int regionsX = cols/xinc;
	int regionsY = rows/yinc;
	// make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
	if (((double)cols/(double)xinc) != regionsX)
		regionsX++;
	if (((double)rows/(double)yinc) != regionsY)
		regionsY++;
	int numRegions = regionsX*regionsY;
    
	Region region[numRegions];
	int x, y;
	i = 0;
	for(x=0;x<cols;x+=xinc) {
		for(y=0;y<rows;y+=yinc) {
			region[i].col = x;
			region[i].row = y;
			region[i].w = xinc;
			region[i].h = yinc;
			if (region[i].col + region[i].w > cols) // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
				region[i].w = cols - region[i].col; // but better to be safe!
			if (region[i].row + region[i].h > rows)
				region[i].h = rows - region[i].row;
			i++;
		}
	}

	// MGD note - would want to do per-region beadfind here soon
	Mask mask(cols, rows);
	img.LoadRaw(beadfindFile);
	img.FilterForPinned(&mask, MaskEmpty);
//	img.FilterForLowHigh(&mask, MaskEmpty);
	img.Normalize(0, 5);
	//img.BackgroundCorrectMulti(&mask, (MaskType) ~(MaskPinned|MaskIgnore), (MaskType) ~(MaskPinned|MaskIgnore), 1, 2, NULL);
	img.BackgroundCorrect(&mask, (MaskType) ~(MaskPinned|MaskIgnore), (MaskType) ~(MaskPinned|MaskIgnore), 1, 2, NULL);
	for (int r=0;r<numRegions;r++) {
		img.CalcBeadfindMetric_1(&mask,region[r],"pre"); // results stored in internal results per well, get via Results()
	}

	// generate a 100x100 heatmap of pinned pixels
	memset(heatMap, 0, sizeof(short) * rows * cols);
	for(i=0;i<mask.W()*mask.H();i++) {
		if (mask[i] &  MaskPinned) {
			heatMap[i] = 1;
		}
	}

	short pinnedHeatMap[100*100];
	ConvertMap(heatMap, cols, rows, 100, 100, 20, 20, pinnedHeatMap, 2);

	// dump out the heatmap
	FILE *fp = NULL;
	fopen_s(&fp, "pinnedmap.txt", "w");
	for(y=0;y<100;y++) {
		for(x=0;x<100;x++) {
			fprintf(fp, "%d ", pinnedHeatMap[x+y*100]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//
	// separator section
	//
	Separator separator;
	separator.SetSize(cols, rows, numKeyFlows); // a bit ugly, but separator needs to know the size to build temp storage
	separator.SetFlowOrder(flowOrder);
	int r;
#ifdef _DEBUG
	dbgmemDisableCheck();
#endif /* _DEBUG */
	for(r=0;r<numRegions;r++) {
		// do per-region beadfind - each region is updated in the global mask - probably can be done in threads, but not much gain from this call
		// this envokes the standard k-means clustering on the beadfind metric to group them automatically
		separator.FindBeads(&img, &region[r], &mask, "pre");
        }
#ifdef _DEBUG
	dbgmemEnableCheck();
#endif /* _DEBUG */

	// at this point, the mask contains MaskEmpty, MaskBead, and possibly MaskPinned & MaskIgnore
	// make heatmap of beadfind
	memset(heatMap, 0, sizeof(short) * rows * cols);
	int numBeads = 0;
	for(i=0;i<mask.W()*mask.H();i++) {
		if (mask[i] &  MaskBead) {
			numBeads++;
		heatMap[i] = 1;
		}
	}
	printf("Found %d beads\n", numBeads);

	// convert and dump the bead heatmap
	short destHeatMap[100*100];
	ConvertMap(heatMap, cols, rows, 100, 100, 20, 20, destHeatMap, 2);
	// dump out the heatmap
	fopen_s(&fp, "beadfindmap.txt", "w");
	for(y=0;y<100;y++) {
		for(x=0;x<100;x++) {
			fprintf(fp, "%d ", destHeatMap[x+y*100]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	// clear the heatmap
	memset(heatMap, 0, sizeof(short) * rows * cols);

	// generate an avg 0-mer trace for each region
	double **avg0mer = new double *[numRegions];
	int *count = new int[numRegions];
	memset(count, 0, sizeof(int) * numRegions);
	for(i=0;i<numRegions;i++) {
		avg0mer[i] = new double[numFrames];
		memset(avg0mer[i], 0, sizeof(double) * numFrames);
	}

	// load up the 0-mer
	char acqFile[512];
	sprintf(acqFile, "%s/%s_%04d.dat", dirExt, "acq", zeromer + (zeromer/4)*hasWashFlow);
	img.LoadRaw(acqFile);
	Mask localMask(&mask);
	img.FilterForPinned(&localMask, MaskEmpty);
	img.Normalize(0, 5);
	//img.BackgroundCorrectMulti(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);
	img.BackgroundCorrect(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);

	// do per-region avg
	int f;
	const RawImage *raw = img.GetImage();
	int frameStride = raw->rows*raw->cols;
	for(r=0;r<numRegions;r++) {
		// first sum all beads in the region
		for(y=region[r].row;y<(region[r].row+region[r].h);y++) {
			for(x=region[r].col;x<(region[r].col+region[r].w);x++) {
				if (localMask[x+y*raw->cols] & MaskBead) {
					for(f=0;f<numFrames;f++) {
						avg0mer[r][f] += raw->image[x+y*raw->cols+f*frameStride];
					}
					count[r]++;
				}
			}
		}
		// then avg them
		if (count[r] > 0) {
			for(f=0;f<numFrames;f++) {
				avg0mer[r][f] /= count[r];
			}
		}
	}

	// load up the 1-mer
	sprintf(acqFile, "%s/%s_%04d.dat", dirExt, "acq", onemer + (onemer/4)*hasWashFlow);
	img.LoadRaw(acqFile);
	img.FilterForPinned(&localMask, MaskEmpty);
	img.Normalize(0, 5);
	//img.BackgroundCorrectMulti(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);
	img.BackgroundCorrect(&localMask, MaskBead, MaskEmpty, 2, 5, NULL);
	raw = img.GetImage();

	// now, for each region, look at each bead, and find the nsub 0-mer sub peak, thats our metric
	short peak, maxPeak;
	for(r=0;r<numRegions;r++) {
		for(y=region[r].row;y<(region[r].row+region[r].h);y++) {
			for(x=region[r].col;x<(region[r].col+region[r].w);x++) {
				if (localMask[x+y*raw->cols] & MaskBead) {
					maxPeak = 0;
					for(f=0;f<numFrames;f++) {
						peak = (short)(raw->image[x+y*raw->cols+f*frameStride] - avg0mer[r][f] + 0.5);
						if (peak > maxPeak)
							maxPeak = peak;
					}
					heatMap[x+y*raw->cols] = maxPeak;
				}
			}
		}
	}

	// convert heatmap into something smaller with some averaging applied
	ConvertMap(heatMap, cols, rows, 100, 100, 20, 20, destHeatMap, 1);

	// dump out the heatmap
	for(y=0;y<100;y++) {
		for(x=0;x<100;x++) {
			printf("%d ", destHeatMap[x+y*100]);
		}
		printf("\n");
	}

	// cleanups
	for(i=0;i<numRegions;i++)
		delete [] avg0mer[i];
	delete [] avg0mer;
	delete [] count;
	delete [] heatMap;
}

