/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include "LinuxCompat.h"
#include "Utils.h"
#include "Trainer.h"
#include "KMrand.h"
#include "dbgmem.h"
#include "sgfilter/SGFilter.h"


Trainer::Trainer(int _maxFrames, int _numRegions, int _numKeyFlows, char *_keySequence, int _numGroups)
{
	// Training Vectors
	numRegions = _numRegions;
	maxFrames = _maxFrames;
	numKeyFlows = _numKeyFlows;
	numGroups = _numGroups;
	if (numGroups < 1)
		numGroups = 1;
	wellCnt = new int [numRegions*numGroups];	// per region well count
	memset (wellCnt, 0, numRegions * numGroups * sizeof(int));

	int i = 0;
	vec = new double *[numRegions*numGroups];
	for(i=0;i<numRegions*numGroups;i++) {
		vec[i] = new double[maxFrames + 1];	// final element will contain 0-mer, 1-mer designation
		memset (vec[i], 0, sizeof(double) * (maxFrames + 1));
	}
	
	//Raw traces
	rawTraces = new double **[numRegions*numGroups];
	for (i = 0; i < numRegions*numGroups; i++)
		rawTraces[i] = NULL;
		
	maxTraces = new int [numRegions*numGroups];
	for (i = 0; i < numRegions*numGroups; i++)
			maxTraces[i] = 0;
	
	keySequence = (char *) malloc (strlen (_keySequence) + 1);
	strncpy (keySequence, _keySequence, strlen (_keySequence) + 1);
	
	experimentName = "./";
	SetDir (experimentName);
	
	results=NULL;
	
	use_flux_integration = false;

	maxTracesPerFlow = 1000;
	if (numGroups > 1)
		maxTracesPerFlow = 1000/numGroups;
	
	using_NN_data = 0;
	lastFrame = 0;
}

Trainer::~Trainer()
{
	for(int i=0;i<numRegions*numGroups;i++)
		delete [] vec[i];
	delete [] vec;
	
	CleanupRawTraces();
	
	if (results)
		delete [] results;
		
	delete [] wellCnt;
	delete [] maxTraces;
		
	free (keySequence);
}

void Trainer::CleanupRawTraces ()
{
	if (rawTraces) {
		volatile int i,j;
		for (i = 0; i < numRegions*numGroups; i++) {
			for (j = 0; j < maxTraces[i]; j++) {
				delete [] rawTraces[i][j];
			}
			delete	[] rawTraces[i];
			rawTraces[i] = NULL;
		}
		delete [] rawTraces;
		rawTraces = NULL;
	}
	return;
}

void Trainer::SetDir (char *directory)
{
	experimentName = directory;
}

// Compare function for qsort for Ascending order
static int doubleCompare(const void *v1, const void *v2)
{
	double val1 = *(double *)v1;
	double val2 = *(double *)v2;

	if (val1 < val2)
		return -1;
	else if (val2 < val1)
		return 1;
	else
		return 0;
}
void Trainer::AddTraces(Image *img, Region *region, int r, Mask *mask, MaskType these, Separator *sep, int trainingValue)
{	
	int traceCnt = (*mask).GetCount (these, *region);
	traceCnt = (traceCnt < maxTracesPerFlow ? traceCnt:maxTracesPerFlow);
	
	int group;
	int i;
	for(group=0;group<numGroups;group++) {
		i = r + group*numRegions;
		if (rawTraces[i] == NULL) {
			maxTraces[i] = traceCnt * numKeyFlows;
			int j;
			rawTraces[i] = new double *[maxTraces[i]];
			for (j = 0; j < maxTraces[i]; j++) {
				rawTraces[i][j] = new double [maxFrames+1];	// the plus one is to hold the final weka vector thingy
				memset (rawTraces[i][j], 0, sizeof(double) * (maxFrames+1));
			}
		}
	}
	
	const RawImage *raw = img->GetImage ();
	int y;
	int x;
	int endWell;
	int frame;
	endWell = wellCnt[r] + traceCnt;
	
	//Retrieve rqs scores: stored in first 'frame' of work matrix.
	double *work = sep->GetWork();
	double *rqsList = new double[(*mask).GetCount(these, *region)];
	int bead = 0;
	for(y=region->row;y<(region->row+region->h);y++) {
		for(x=region->col;x<(region->col+region->w);x++) {
			if ((*mask)[x+y*raw->cols] & (MaskIgnore|MaskWashout|MaskPinned)) {
				continue;
			}
			if ((*mask)[x+y*raw->cols] & these && (*mask)[x+y*raw->cols] & MaskLive) {
				rqsList[bead] = work[x+y*raw->cols];
				bead++;
			}
		}
	}
	

	// Not enough valid beads
	if (bead < (region->w*region->h)/500) {
		fprintf (stdout, "AddTraces: Region skipped r%04d c%04d\n", region->row, region->col);
		if (rqsList)
			delete [] rqsList;
		return;
	}
	
	//sort the beads in Ascending order
	qsort (rqsList, bead, sizeof(double), doubleCompare);
	
	//get the cutoff
	// we want at most the top maxTracesPerFlow beads in the region; when there are less, we take them all
	double cutoffValue = rqsList[(bead > traceCnt ? traceCnt:bead) - 1];
		
	//selects beads above the cutoff
	for (y=region->row;y<(region->row+region->h);y++) {
		 for (x=region->col;x<(region->col+region->w);x++) {
			if (wellCnt[r] < endWell) {
				if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout|MaskPinned)) {
					continue;
				}
				if (((*mask)[x+(y*raw->cols)] & these) && ((*mask)[x+y*raw->cols] & MaskLive) && (work[x+(y*raw->cols)] <= cutoffValue)) {
					group = sep->GetGroup(x, y);
					i = r+group*numRegions;
					for (frame = 0; frame < maxFrames; frame++) {
						rawTraces[i][wellCnt[i]][frame] = raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
					}
					rawTraces[i][wellCnt[i]][maxFrames] = trainingValue;
					wellCnt[i] += 1;
				}
			}
			else {
				break;
			}
		}
	}
	
	if (rqsList)
		delete [] rqsList;
}

//void Trainer::AddTraces(Image *img, Region *region, int r, Mask *mask, MaskType these, Separator *sep, int trainingVal, int zeroFlow)
void Trainer::AddTraces(Image *img, Region *region, int r, Mask *mask, MaskType these, Separator *sep, int trainingVal, const double *zero)
{	
	int traceCnt = (*mask).GetCount (these, *region);
	traceCnt = (traceCnt < maxTracesPerFlow ? traceCnt:maxTracesPerFlow);
	
	int group;
	int i;
	for(group=0;group<numGroups;group++) {
		i = r + group*numRegions;
		if (rawTraces[i] == NULL) {
			maxTraces[i] = traceCnt * numKeyFlows;
			int j;
			rawTraces[i] = new double *[maxTraces[i]];
			for (j = 0; j < maxTraces[i]; j++) {
				rawTraces[i][j] = new double [maxFrames+1];	// the plus one is to hold the final weka vector thingy
				memset (rawTraces[i][j], 0, sizeof(double) * (maxFrames+1));
			}
		}
	}
	
	const RawImage *raw = img->GetImage ();
	int y;
	int x;
	int endWell;
	int frame;
	endWell = wellCnt[r] + traceCnt;
	
	//Retrieve rqs scores: stored in first 'frame' of work matrix.
	double *work = sep->GetWork();
	double *rqsList = new double[(*mask).GetCount(these, *region)];
	int bead = 0;
	for(y=region->row;y<(region->row+region->h);y++) {
		for(x=region->col;x<(region->col+region->w);x++) {
			if ((*mask)[x+y*raw->cols] & (MaskIgnore|MaskWashout|MaskPinned)) {
				continue;
			}
			if ((*mask)[x+y*raw->cols] & these && (*mask)[x+y*raw->cols] & MaskLive) {
				rqsList[bead] = work[x+y*raw->cols];
				bead++;
			}
		}
	}
	

	// Not enough valid beads
	if (bead < (region->w*region->h)/500) {
		fprintf (stdout, "AddTraces: %s Region skipped r%04d c%04d\n", (these == MaskLib ? "Lib":"TF"),region->row, region->col);
		fprintf (stdout, "Not enough beads %d (need > %d)\n", bead, (region->w*region->h)/500);
		if (rqsList)
			delete [] rqsList;
		return;
	}
	if (!zero) {
		fprintf (stdout, "AddTraces: %s Region skipped r%04d c%04d\n", (these == MaskLib ? "Lib":"TF"),region->row, region->col);
		fprintf (stdout, "Did not get valid zeromer vector\n");
		if (rqsList)
			delete [] rqsList;
		return;
	}
	
	//sort the beads in Ascending order
	qsort (rqsList, bead, sizeof(double), doubleCompare);
	
	//get the cutoff
	// we want at most the top maxTracesPerFlow beads in the region; when there are less, we take them all
	double cutoffValue = rqsList[(bead > traceCnt ? traceCnt:bead) - 1];
	
	double *flux_signal = NULL;
	
	//selects beads above the cutoff
	for (y=region->row;y<(region->row+region->h);y++) {
		 for (x=region->col;x<(region->col+region->w);x++) {
			if (wellCnt[r] < endWell) {
				if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout|MaskPinned)) {
					continue;
				}
				if (((*mask)[x+(y*raw->cols)] & these) && ((*mask)[x+y*raw->cols] & MaskLive) && (work[x+(y*raw->cols)] <= cutoffValue)) {
					group = sep->GetGroup(x, y);
					i = r+group*numRegions;
					
					if (use_flux_integration) {
						flux_signal = FluxIntegrator (img, r, x, y, zero, these);
						
						for (frame = 0; frame < maxFrames; frame++) {
							rawTraces[i][wellCnt[i]][frame] = flux_signal[frame];
						}
						rawTraces[i][wellCnt[i]][maxFrames] = trainingVal;
						wellCnt[i] += 1;
						delete [] flux_signal;
					}
					//NON-FLUX INTEGRATION
					else {
						for (frame = 0; frame < maxFrames; frame++) {
							rawTraces[i][wellCnt[i]][frame] = raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
						}
						rawTraces[i][wellCnt[i]][maxFrames] = trainingVal;
						wellCnt[i] += 1;
					}
				}
			}
			else {
				break;
			}
		}
	}
	
	if (rqsList)
		delete [] rqsList;
}

//Latest test code from Jon S.	
#if 1
double *Trainer::FluxIntegrator (Image *img, int r, int x, int y, const double *zeroTrace, MaskType these)
{
	////	debug output files
	//FILE *kvalfp = NULL;
	//FILE *signalfp = NULL;
	//FILE *integralfp = NULL;
	//FILE *fluxfp = NULL;
	//char kvalFileName[MAX_PATH_LENGTH] = {"0"};
	//char signalFileName[MAX_PATH_LENGTH] = {"0"};
	//char integralFileName[MAX_PATH_LENGTH] = {"0"};
	//char fluxFileName[MAX_PATH_LENGTH] = {"0"};
	//snprintf (kvalFileName, 256, "%s/%s_%s", experimentName, "kval", (these == MaskLib ? "Lib":"TF"));
	//snprintf (signalFileName, 256, "%s/%s_%s", experimentName, "signal", (these == MaskLib ? "Lib":"TF"));
	//snprintf (integralFileName, 256, "%s/%s_%s", experimentName, "integral", (these == MaskLib ? "Lib":"TF"));
	//snprintf (fluxFileName, 256, "%s/%s_%s", experimentName, "flux", (these == MaskLib ? "Lib":"TF"));
	//fopen_s(&kvalfp, kvalFileName, "ab");
	//fopen_s(&signalfp, signalFileName, "ab");
	//fopen_s(&integralfp, integralFileName, "ab");
	//fopen_s(&fluxfp, fluxFileName, "ab");
	////	end debug output files
	
	const RawImage *raw = img->GetImage ();
	
	// Some parameters, empirically derived
	// Tau is 25 @ 15fps rate
	// window is 30 frames @15fps
	// starting frame is 15 (1 second @ 15fps)
	// ending frame is 50 (5 seconds @ 15fps)
	int sigStart = 30; //new constant, intended to be as close to the start of incorporation as possible, without being sooner;
	int Tau = img->GetFrame(687);	//25;
	int frameStart;
	if (using_NN_data) {
		frameStart = 0;
	}
	else {
		frameStart = img->GetFrame(12);	//15;
	}
	int frameEnd = img->GetFrame(2374);	//50;
	//fprintf (stdout, "Trainer: FluxIntegrator %d %d %d (expect 25 15 50)\n", Tau, frameStart, frameEnd);
	int frameWidth;
	if (using_NN_data) {
		frameWidth = img->GetFrame(341);	//20;
	}
	else {
		frameWidth = img->GetFrame(1024);	//30;
	}
	int frameSkip = 1; //has to be 1 for using_NN_data;
	int avgWidth = img->GetFrame(-325);	//10; //will reuse this constant, but divide by 2;
	int numWindows;
	if (using_NN_data) {
		numWindows = (maxFrames - frameWidth - avgWidth/2);
	}
	else{
		numWindows = (frameEnd - frameStart) / frameSkip;
	}
	int maxWindows = (int) floor ((double)(maxFrames - (frameStart + frameWidth + avgWidth/2)) / frameSkip); // upper limit
	numWindows = (numWindows > maxWindows ? maxWindows:numWindows);
	lastFrame = maxFrames - (using_NN_data * (frameWidth + avgWidth/2));
	
	double *k_val = new double [numWindows];
	memset(k_val,0,sizeof(double)*numWindows);
	
	double onemerSigFront;
	double onemerSigBack;
	double onemerIntegrated;
	double zeromerSigFront;
	double zeromerSigBack;
	double zeromerIntegrated;
	double *signalTrace = new double [maxFrames];
	memset(signalTrace,0,sizeof(double)*maxFrames);
	double *integralTrace = new double [maxFrames];
	memset(integralTrace,0,sizeof(double)*maxFrames);
	double *fluxTrace = new double [maxFrames];
	memset(fluxTrace,0,sizeof(double)*maxFrames);

	// need to subtract zeroTrace right off the bat without any scaling in every processing block below
	// signal after subtraction is almost fully corrected, just need to do flux correction with a redefined version of zeroTrace that is set by a function
	// not sure if it is ok to reuse 'zeroTrace' array, but it makes the code easier to do regular flux case
	double *zeroTraceFunc = NULL;
	if (using_NN_data) {
		//create a duplicate of zeroTrace that is subtracted off the raw signal in every case
		zeroTraceFunc = (double *) malloc (sizeof(double)*raw->frames);
		
		//redefine zeroTrace with explicit function
		for (int i=0;i<raw->frames;i++)
			zeroTraceFunc[i] = (i - sigStart < 0 ? 0:1) * (1 - exp((sigStart - i)/Tau)); // for i=0, i<=maxFrames;
	}
	else {
		//Copy zerotrace values here to zerotracefunc
		zeroTraceFunc = (double *) malloc (sizeof(double)*raw->frames);
		for (int i=0;i<raw->frames;i++)
			zeroTraceFunc[i] = zeroTrace[i];
	}
	
	// Calculate array of k_val for sliding window
	//fprintf (kvalfp, "%02d-%03d-%03d ",r,y,x);
	for (int i = 0, frame = frameStart; i < numWindows; i++, frame += frameSkip) {
		
		// Average +/- 5 frames from each front and back window location
		int cnt = 0;
		onemerSigFront = 0;
		onemerSigBack = 0;
		zeromerSigFront = 0;
		zeromerSigBack = 0;
		for (int j = frame - (1 - using_NN_data)*avgWidth/2; j <= frame + avgWidth/2; j++) { //only advance forward avgWidth/2 (5) for using_NN_data
			onemerSigFront += raw->image[x+(y*raw->cols) + (raw->frameStride * j)] - (using_NN_data * zeroTrace[j]);
			onemerSigBack += raw->image[x+(y*raw->cols) + (raw->frameStride * (j+frameWidth))] - (using_NN_data * zeroTrace[j+frameWidth]);
			
			//zeromerSigFront += avgTrace[r][zeroFlow][j];
			//zeromerSigBack += avgTrace[r][zeroFlow][j+frameWidth];
			zeromerSigFront += zeroTraceFunc[j];
			zeromerSigBack += zeroTraceFunc[j+frameWidth];
			
			cnt++;
		}
		onemerSigFront /= cnt;
		onemerSigBack /= cnt;
		zeromerSigFront /= cnt;
		zeromerSigBack /= cnt;
		
		// Integrated signal within the window width
		zeromerIntegrated = 0;
		onemerIntegrated = 0;
		for (int j = frame; j < frame+frameWidth; j++) {
			onemerIntegrated += raw->image[x+(y*raw->cols) + (raw->frameStride * j)] - (using_NN_data * zeroTrace[j]);
			//zeromerIntegrated += avgTrace[r][zeroFlow][j];
			zeromerIntegrated += zeroTraceFunc[j];
		}
		
		// K value calculation
		double val1 = onemerIntegrated + (Tau * (onemerSigBack-onemerSigFront));
		double val2 = zeromerIntegrated + (Tau * (zeromerSigBack-zeromerSigFront));
		if (val2 == 0 || val1 == 0) {
			k_val[i] = 1;
		}
		else {
			k_val[i] = val1 / val2; //need a small fudge factor to prevent div by zero;
		}
			
		//fprintf (kvalfp, "%0.2lf ", k_val[i]);
	}
	//fprintf (kvalfp, "\n");
	
	double k_val_min = 0.0;
	int k_min_frame = 0;
	if (using_NN_data) { // same as flux_2v == true, entire block below can be omitted because k_val_min and k_min_frame are not calculated
		// Find maximum K_val and associated front frame position
		for (int i = 0; i < numWindows; i++){
			if (i == 0 || k_val[i] > k_val_min) {
				k_val_min = k_val[i];
				k_min_frame = (i * frameSkip) + frameStart;
			}
		}
	}
	else {
		// Find minimum K_val and associated front frame position
		for (int i = 0; i < numWindows; i++){
			if (i == 0 || k_val[i] < k_val_min) {
				k_val_min = k_val[i];
				k_min_frame = (i * frameSkip) + frameStart;
			}
		}
	}
	// Signal = well trace - Kay(min) * zeromer trace
	// if using_NN_data then Signal = well trace - k_val[i]*zeroTrace[i], for i=0 to (maxFrames - frameWidth - avgWidth/2)
	//fprintf (signalfp, "%02d-%03d-%03d ",r,y,x);
	for (int j = 0; j < maxFrames - (using_NN_data * (frameWidth + avgWidth/2)); j++) { //have to trim the end of the time history for using_NN_data because window and avgWidth tail portion are unusable
		//signalTrace[j] = raw->image[x+(y*raw->cols) + (raw->frameStride * j)] -
		//				 (k_val_min * avgTrace[r][zeroFlow][j]);
		if (using_NN_data) {
			signalTrace[j] = raw->image[x+(y*raw->cols) + (raw->frameStride * j)] -
						 (k_val[j] * zeroTraceFunc[j]) - zeroTrace[j];
		}
		else
		{
		signalTrace[j] = raw->image[x+(y*raw->cols) + (raw->frameStride * j)] -
						 (k_val_min * zeroTraceFunc[j]);
		}
		integralTrace[j] = signalTrace[j];
		
		//fprintf (signalfp, "%0.2lf ", signalTrace[j]);
	}
	//fprintf (signalfp, "\n");
	
	// integral of Signal
	//fprintf (integralfp, "%0.2lf ", integralTrace[0]);
	for (int j = 1; j < maxFrames - (using_NN_data * (frameWidth + avgWidth/2)); j++) {
		integralTrace[j] = integralTrace[j] + integralTrace[j - 1];
		
		//fprintf (integralfp, "%0.2lf ", integralTrace[j]);
	}
	//fprintf (integralfp, "\n");
	
	//  Flux = Signal + Integral-Signal/Tau
	for (int j = 0; j < maxFrames - (using_NN_data * (frameWidth + avgWidth/2)); j++) {
		fluxTrace[j] = signalTrace[j] + ((integralTrace[j])/Tau);
	}
	// DEBUG print
	// Flux trace before trimming
	//fprintf (fluxfp, "%02d-%03d-%03d ",r,y,x);
	//for (int j = 0; j < maxFrames - (using_NN_data * (1 + frameWidth + avgWidth/2)); j++) {
	//	fprintf (fluxfp, "%0.2lf ", fluxTrace[j]);
	//}
	//fprintf (fluxfp, "\n");
	
	//  Remove step portion of flux from frameStart to k_min index
	//	Zero out the rest of the array
	//if (using_NN_data) {
	//	k_min_frame = 40; //just use some explicit number for now;
	//	}
	frameStart = img->GetFrame(12);    //15; //reset to original value based on 1sec of prewash;
    if (using_NN_data) {
        k_min_frame = 3 * frameStart; //just use some explicit number for now;
        for (int j = frameStart; j < 3 * frameStart; j++) {  //average all of the frames between 1sec and 3sec and store value at (1sec - 1frame);
			fluxTrace[frameStart-1] += fluxTrace[j];
        }
        fluxTrace[frameStart-1] /= (2 * frameStart)+1;
	}
	
	for (int j = frameStart, i = 0; j < maxFrames; j++, i++) {
		if (k_min_frame + i >= maxFrames - (using_NN_data * (frameWidth + avgWidth/2))) {
			fluxTrace[j] = fluxTrace[maxFrames - frameWidth - avgWidth/2 - 1];
		}
		else {
			fluxTrace[j] = fluxTrace[k_min_frame + i];
		}
	}
	

	//// Flux trace after trimming
	//fprintf (fluxfp, "%02d-%03d-%03d ",r,y,x);
	//for (int j = 0; j < maxFrames; j++) {
	//  fprintf (fluxfp, "%0.2lf ", fluxTrace[j]);
	//}
	//fprintf (fluxfp, "\n");
	//
	//fclose(kvalfp);
	//fclose(signalfp);
	//fclose(integralfp);
	//fclose(fluxfp);
	
	delete [] k_val;
	delete [] signalTrace;
	delete [] integralTrace;
	if (zeroTraceFunc)
		free (zeroTraceFunc);
	return (fluxTrace);
}
#endif

#if 0
//
//	Calculates the flux integrator method a la John Shultz/Jeff Branciforte
//
double *Trainer::FluxIntegrator (Image *img, int r, int x, int y, const double *zeroTrace, MaskType these)
{
	////	debug output files
	//FILE *kvalfp = NULL;
	//FILE *signalfp = NULL;
	//FILE *integralfp = NULL;
	//FILE *fluxfp = NULL;
	//char kvalFileName[MAX_PATH_LENGTH] = {"0"};
	//char signalFileName[MAX_PATH_LENGTH] = {"0"};
	//char integralFileName[MAX_PATH_LENGTH] = {"0"};
	//char fluxFileName[MAX_PATH_LENGTH] = {"0"};
	//snprintf (kvalFileName, 256, "%s/%s_%s", experimentName, "kval", (these == MaskLib ? "Lib":"TF"));
	//snprintf (signalFileName, 256, "%s/%s_%s", experimentName, "signal", (these == MaskLib ? "Lib":"TF"));
	//snprintf (integralFileName, 256, "%s/%s_%s", experimentName, "integral", (these == MaskLib ? "Lib":"TF"));
	//snprintf (fluxFileName, 256, "%s/%s_%s", experimentName, "flux", (these == MaskLib ? "Lib":"TF"));
	//fopen_s(&kvalfp, kvalFileName, "ab");
	//fopen_s(&signalfp, signalFileName, "ab");
	//fopen_s(&integralfp, integralFileName, "ab");
	//fopen_s(&fluxfp, fluxFileName, "ab");
	////	end debug output files
	
	const RawImage *raw = img->GetImage ();
	
	// Some parameters, empirically derived
	// Tau is 25 @ 15fps rate
	// window is 30 frames @15fps
	// starting frame is 15 (1 second @ 15fps)
	// ending frame is 50 (5 seconds @ 15fps)
	int Tau = img->GetFrame(687);	//25;
	int frameStart = img->GetFrame(12);	//15;
	int frameEnd = img->GetFrame(2374);	//50;
	//fprintf (stdout, "Trainer: FluxIntegrator %d %d %d (expect 25 15 50)\n", Tau, frameStart, frameEnd);
	int frameWidth = img->GetFrame(1024);	//30;
	int frameSkip = 1;
	int avgWidth = img->GetFrame(-325);	//10;
	
	int numWindows = (frameEnd - frameStart) / frameSkip;
	int maxWindows = (int) floor ((double)(maxFrames - (frameStart + frameWidth + avgWidth/2)) / frameSkip); // upper limit
	numWindows = (numWindows > maxWindows ? maxWindows:numWindows);
	
	double *k_val = new double [numWindows];
	memset(k_val,0,sizeof(double)*numWindows);
	
	double onemerSigFront;
	double onemerSigBack;
	double onemerIntegrated;
	double zeromerSigFront;
	double zeromerSigBack;
	double zeromerIntegrated;
	double *signalTrace = new double [maxFrames];
	memset(signalTrace,0,sizeof(double)*maxFrames);
	double *integralTrace = new double [maxFrames];
	memset(integralTrace,0,sizeof(double)*maxFrames);
	double *fluxTrace = new double [maxFrames];
	memset(fluxTrace,0,sizeof(double)*maxFrames);
	
	// Calculate array of k_val for sliding window
	//fprintf (kvalfp, "%02d-%03d-%03d ",r,y,x);
	for (int i = 0, frame = frameStart; i < numWindows; i++, frame += frameSkip) {
		
		// Average +/- 5 frames from each front and back window location
		int cnt = 0;
		onemerSigFront = 0;
		onemerSigBack = 0;
		zeromerSigFront = 0;
		zeromerSigBack = 0;
		for (int j = frame - avgWidth/2; j <= frame + avgWidth/2; j++) {
			onemerSigFront += raw->image[x+(y*raw->cols) + (raw->frameStride * j)];
			onemerSigBack += raw->image[x+(y*raw->cols) + (raw->frameStride * (j+frameWidth))];
			
			//zeromerSigFront += avgTrace[r][zeroFlow][j];
			//zeromerSigBack += avgTrace[r][zeroFlow][j+frameWidth];
			zeromerSigFront += zeroTrace[j];
			zeromerSigBack += zeroTrace[j+frameWidth];
			
			cnt++;
		}
		onemerSigFront /= cnt;
		onemerSigBack /= cnt;
		zeromerSigFront /= cnt;
		zeromerSigBack /= cnt;
		
		// Integrated signal within the window width
		zeromerIntegrated = 0;
		onemerIntegrated = 0;
		for (int j = frame; j < frame+frameWidth; j++) {
			onemerIntegrated += raw->image[x+(y*raw->cols) + (raw->frameStride * j)];
			//zeromerIntegrated += avgTrace[r][zeroFlow][j];
			zeromerIntegrated += zeroTrace[j];
		}
		
		// K value calculation
		double val1 = onemerIntegrated + (Tau * (onemerSigBack-onemerSigFront));
		double val2 = zeromerIntegrated + (Tau * (zeromerSigBack-zeromerSigFront));
		if (val2 == 0 || val1 == 0) {
			k_val[i] = 1;
		}
		else {
			k_val[i] = val1 / val2;
		}
			
		//fprintf (kvalfp, "%0.2lf ", k_val[i]);
	}
	//fprintf (kvalfp, "\n");
	
	double k_val_min = 0.0;
	int k_min_frame = 0;
	if (using_NN_data) {
		// Find maximum K_val and associated front frame position
		for (int i = 0; i < numWindows; i++){
			if (i == 0 || k_val[i] > k_val_min) {
				k_val_min = k_val[i];
				k_min_frame = (i * frameSkip) + frameStart;
			}
		}
	}
	else {
		// Find minimum K_val and associated front frame position
		for (int i = 0; i < numWindows; i++){
			if (i == 0 || k_val[i] < k_val_min) {
				k_val_min = k_val[i];
				k_min_frame = (i * frameSkip) + frameStart;
			}
		}
	}
	// Signal = well trace - Kay(min) * zeromer trace
	//fprintf (signalfp, "%02d-%03d-%03d ",r,y,x);
	for (int j = 0; j < maxFrames; j++) {
		//signalTrace[j] = raw->image[x+(y*raw->cols) + (raw->frameStride * j)] -
		//				 (k_val_min * avgTrace[r][zeroFlow][j]);
		signalTrace[j] = raw->image[x+(y*raw->cols) + (raw->frameStride * j)] -
						 (k_val_min * zeroTrace[j]);
			
		integralTrace[j] = signalTrace[j];
		
		//fprintf (signalfp, "%0.2lf ", signalTrace[j]);
	}
	//fprintf (signalfp, "\n");
	
	// integral of Signal
	//fprintf (integralfp, "%0.2lf ", integralTrace[0]);
	for (int j = 1; j < maxFrames; j++) {
		integralTrace[j] = integralTrace[j] + integralTrace[j - 1];
		
		//fprintf (integralfp, "%0.2lf ", integralTrace[j]);
	}
	//fprintf (integralfp, "\n");
	
	//  Flux = Signal + Integral-Signal/Tau
	for (int j = 0; j < maxFrames; j++) {
		fluxTrace[j] = signalTrace[j] + ((integralTrace[j])/Tau);
	}
	// Flux trace before trimming
	//for (int j = 0; j < maxFrames; j++) {
	//	fprintf (fluxfp, "%0.2lf ", fluxTrace[j]);
	//}
	//fprintf (fluxfp, "\n");
	
	//  Remove step portion of flux from frameStart to k_min index
	//	Zero out the rest of the array
	for (int j = frameStart, i = 0; j < maxFrames; j++, i++) {
		if (k_min_frame + i >= maxFrames) {
			fluxTrace[j] = 0.0;
		}
		else {
			fluxTrace[j] = fluxTrace[k_min_frame + i];
		}
	}
	//// Flux trace after trimming
	//fprintf (fluxfp, "%02d-%03d-%03d ",r,y,x);
	//for (int j = 0; j < maxFrames; j++) {
	//  fprintf (fluxfp, "%0.2lf ", fluxTrace[j]);
	//}
	//fprintf (fluxfp, "\n");
	//
	//fclose(kvalfp);
	//fclose(signalfp);
	//fclose(integralfp);
	//fclose(fluxfp);
	
	delete [] k_val;
	delete [] signalTrace;
	delete [] integralTrace;
	return (fluxTrace);
}
#endif
void Trainer::ApplyVector(Image *img, int r, Region *region, Mask *mask, MaskType these, const double *zeroTrace)
{
	if (!zeroTrace)
		return;
	
	double val;
	const RawImage *raw = img->GetImage ();
	if (!results) {
		results = new double[raw->rows * raw->cols];
		memset(results, 0, raw->rows * raw->cols * sizeof(double));
	}
		//DEBUG
		double *average = new double [maxFrames];
		memset(average,0,sizeof(double)*maxFrames);
		int count = 0;
	
	double *flux_signal;
	for(int y=region->row;y<(region->row+region->h);y++) {
		for(int x=region->col;x<(region->col+region->w);x++) {
			if ((*mask)[x+raw->cols*y] & (MaskIgnore))
				continue;
			
			if ((*mask)[x+raw->cols*y] & these) {
				if (use_flux_integration) {
					flux_signal = FluxIntegrator (img, r, x, y, zeroTrace, these);
					val = 0;
					int fCnt = 0;
					if (using_NN_data) {
						//average of 'raw' signal
						for(int frame=30;frame<lastFrame;frame++) {
							val += flux_signal[frame];
							fCnt++;
						}
						val /= fCnt;
					}
					else {
						//standard dot product of raw signal with weka vector
						for(int frame=0;frame<maxFrames;frame++) {
							val += flux_signal[frame] * vec[r][frame];
						}
						val += vec[r][maxFrames];
					}
					
					results[x+y*raw->cols] = val;
					delete [] flux_signal;
					// Bad values need to be excluded
					if (isnan(val) || isinf(val)) {
						(*mask)[x+raw->cols*y] |= MaskIgnore;
						continue;
					}
				}
				else {
					val = 0;
					for(int frame=0;frame<maxFrames;frame++) {
						val += (raw->image[x+y*raw->cols+(raw->frameStride*frame)] - zeroTrace[frame]) * vec[r][frame];
					}
					val += vec[r][maxFrames];
					results[x+y*raw->cols] = val;//results[x+y*raw->cols] = (isnan (val) ? 0.0:val);
					// Bad values need to be excluded
					if (isnan(val) || isinf(val)) {
						(*mask)[x+raw->cols*y] |= MaskIgnore;
						continue;
					}
				}
				//DEBUG - calculate an average raw trace for all wells for this flow
				for(int frame=0;frame<maxFrames;frame++) {
					average[frame] += raw->image[x+y*raw->cols+(raw->frameStride*frame)];
				}
				count++;
			}
		}
	}
	if (false){
		FILE *fp = NULL;
		char filename[PATH_MAX];
		sprintf (filename, "%s/trace_average_%s", experimentName, (these & MaskTF ? "Lib":"TF"));
		fopen_s(&fp,filename,"ab");
		for(int frame=0;frame<maxFrames;frame++) {
			average[frame] /= count;
			fprintf (fp, "%0.1lf ", average[frame]-zeroTrace[frame]);
		}
		fprintf (fp, ", ");
		fclose (fp);
	}
	delete [] average;
}


void Trainer::TrainAll()
{
	#if 1
	runWeka ();
	#else
	//TODO: Create squarewave for each region
	int min = 20;
	int max = 90;
	for (int r = 0; r < numRegions*numGroups; r++) {
		for (int f = 0; f <= maxFrames;f++) {
			vec[r][f] = ((f > min && f < max) ? 1:0);
		}
	}
	#endif
	
	//ASAP clean up
	CleanupRawTraces ();
}

const double *Trainer::GetVector(int region)
{
	return vec[region];
}

/*
** WEKA specific methods
*/

// Writes the data for training to ARFF format file.
int Trainer::list_to_arff (char *fileName, int r)
{
	FILE *fp = NULL;
	fopen_s (&fp, fileName, "wb");
	if (!fp)
	{
		fprintf (stderr, "%s: %s\n",fileName, strerror(errno));
		return (errno);
	}
	
	fprintf (fp, "%% ARFF file created by '%s'\n", __FILE__);
	fprintf (fp, "@RELATION Incorp\n");
	for (int i = 0; i < maxFrames; i++)
	{
		fprintf (fp, "@ATTRIBUTE frame_%d NUMERIC\n", i);
	}
	fprintf (fp, "@ATTRIBUTE Incorp NUMERIC\n");
	fprintf (fp, "@DATA\n");
	for (int well = 0; well < wellCnt[r]; well++) {
		for (int frame = 0; frame < maxFrames; frame++) {
			fprintf (fp, "%lf,", rawTraces[r][well][frame]);
		}
		// python implementation shows the last value as a float...
		fprintf (fp, "%0.1lf\n", (float) rawTraces[r][well][maxFrames]);	//last element contains 0-mer, 1-mer designation
	}
	fclose (fp);
		
	return (0);
}

//  Parses the weka's output and extracts the weighting vector parameters
void Trainer::extractWeights (char *buffer, int r)
{
	//capture weka metrics in a file
	char metricFileName[MAX_PATH_LENGTH];
	sprintf (metricFileName, "%s/%s", experimentName,"wekaMetrics.txt");
	FILE *fp = NULL;
	fopen_s (&fp, metricFileName, "a");	// append to existing: multiple regions
	fprintf (fp, "\nRegion: %04d %s:\n", r, keySequence);
	
    int cnt = 0;
	float val;
    char *endptr;
	char *saveptr;
    char *tok = strtok_r (buffer, "\n", &saveptr);
    while (tok != NULL)
    {	
		val = strtof (tok, &endptr);
		if (endptr == tok)
		{
			fprintf (fp, "%s\n", tok);
		}
		else
		{
			vec[r][cnt++] = val;
		}
        tok = strtok_r (NULL, "\n", &saveptr);
    }
	
	fclose (fp);
    return;
}

//  Calls weka command-line program and creates training vector
void Trainer::runWeka ()
{
	char fileName[48] = "";	// final string "/tmp/NNNNN_XXXXXXXXXXX.arff"
	
	for (int r = 0; r < numRegions*numGroups; r++) {
		
		//TODO: Hardcoded minimum threshold.  Needs to be set by size of region
		if (wellCnt[r] <= 10) {
			fprintf (stderr, "There are %d %s wells in region %d! Skipping...\n", wellCnt[r], keySequence, r);
			continue;
		}
		#if 1
		//random filename generator
		sprintf (fileName, "/tmp/%d_", getpid());
		kmIdum = -time(0);	// reinitializes seed
		char pool[] = "abcdefghijklmnopqrstuvwxyz";
		char *ptr = strrchr (fileName, '_');
		for (int j=0;j<12;j++)
			*(++ptr) = pool[kmRanInt (26)];
		*(++ptr) = '\0';
		strcat (fileName,".arff");
		
		if (list_to_arff (fileName, r)) {
			//Error writing temporary ARFF file
			fprintf (stderr, "Skipping this region %d (%s) due to error writing arff file\n", r, keySequence);
			continue;
		}
		#else
		sprintf (fileName, "/tmp/rcuipstciuiwiif.arff");
		fprintf (stdout, "Using special arff file %s\n", fileName);
		#endif
		
		char cmd[1024];
		char *tptr = (char *) malloc (sizeof(char) *(80*(maxFrames+50)));
		char *buf = tptr;
		memset (tptr, 0, (80*(maxFrames+50)) *sizeof (char));
		snprintf (cmd, 1024, "java -Xmx1024M -cp /usr/share/java/weka.jar weka.classifiers.functions.LinearRegression -S 1 -R 1.0E-8 -no-cv -t %s", fileName);
	
		FILE *fp = popen (cmd, "r");
		if (!fp)
		{
			fprintf (stderr, "ERROR opening pipe: %s\n", strerror(errno));
			free(buf);
			continue;
		}
		
		while (!feof(fp))
		{
		   int elements_read = fread (tptr++, sizeof(char), 1, fp);
           assert(elements_read == 1);
		}
		pclose (fp);
		
		extractWeights (buf, r);
		
		free (buf);
		
		//Delete temporary files
		if (false) {
			fprintf (stdout, "DELETE %s !!\n", fileName);
		}
		else {
			if (unlink (fileName) != 0) {
				char msg[MAX_PATH_LENGTH] = {0};
				sprintf (msg, "Trying to unlink %s", fileName);
				perror (msg);
			}
		}
		
		//debug - dump the raw vector to a file
		if (true) {
			const double *vec = GetVector(r);
			FILE *fp = NULL;
			char vecFile[MAX_PATH_LENGTH];
			sprintf (vecFile, "%s/trainVec_%s.txt", experimentName, keySequence);
			fopen_s (&fp, vecFile, "ab");
			fprintf (fp, "%d ", r);
			for (int i = 0; i < maxFrames+1; i++){
				fprintf (fp, "%lf ", vec[i]);
			}
			fprintf (fp, "\n");
			fclose(fp);
		}
	}
	
}
