/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftw3.h"
#include "Zeromer.h"
#include "LinuxCompat.h"

//
//  Constructor
//
Zeromer::Zeromer (int _numRegions, int _numKeyFlows, int _numFrames)
{
    numRegions = _numRegions;
    numKeyFlows = _numKeyFlows;
    numFrames = _numFrames;
	avgTrace = NULL;
    H_vectors = NULL;
    strncpy (experimentName, "./", 3);
}
//
//  Destructor
//
Zeromer::~Zeromer ()
{
	if (H_vectors) {
		for (int i = 0; i < numRegions; i++) {
			for (int j = 0; j < numKeyFlows; j++) {
				if (H_vectors[i][j])
					fftw_free (H_vectors[i][j]);
			}
			delete [] H_vectors[i];
		}
		delete [] H_vectors;
	}
	
	if (avgTrace) {
		for (int i = 0; i < numRegions; i++) {
			for (int j = 0; j < numKeyFlows; j++) {
				delete [] avgTrace[i][j];
			}
			delete [] avgTrace[i];
		}
		delete [] avgTrace;
		avgTrace = NULL;
	}
}
//
//	Define output directory - for dumping debug log files
//
void Zeromer::SetDir (char *_experimentName)
{
    strncpy (experimentName, _experimentName, 256);
    experimentName[255] = '\0';
}
//
//	Returns average zeromer
//
double *Zeromer::GetAverageZeromer (int region, int flow)
{
	return (avgTrace[region][flow]);
}
//
//	Calculate average trace of all beads in region, in this image
//	Used to generate a zeromer average trace for Flux Integrator
//
void Zeromer::CalcAverageTrace (Image *img, Region *region, int r, Mask *mask, MaskType these, int flow)
{
	// Allocate memory for average trace of key flows
	if (avgTrace == NULL) {
		avgTrace = new double **[numRegions];
		for (int i = 0; i < numRegions; i++) {
			avgTrace[i] = new double *[numKeyFlows];
			for (int j = 0; j < numKeyFlows; j++) {
				avgTrace[i][j] = new double [numFrames+1];
				memset (avgTrace[i][j], 0, sizeof(double)*(numFrames+1));
			}
		}
	}

	const RawImage *raw = img->GetImage ();
	
	memset(avgTrace[r][flow], 0, sizeof(double)*(numFrames+1));

	// Average the traces for this region, this flow
	int cnt = 0;
	for (int y=region->row;y<(region->row+region->h);y++) {
		for (int x=region->col;x<(region->col+region->w);x++) {
			if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout)) {
				continue;
			}
			if ((*mask)[x+(y*raw->cols)] & these) {
				for (int frame = 0; frame < numFrames; frame++) {
					avgTrace[r][flow][frame] += raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
				}
				cnt++;
			}
		}
	}
	
	if (cnt > 0) {
		for (int frame = 0; frame < numFrames; frame++) {
			avgTrace[r][flow][frame] /= cnt;
		}
	}
	
	//	debug
	FILE *fp = NULL;
	char *fileName = (char *) malloc (256);
	if (these == MaskEmpty) {
		snprintf (fileName, 256, "%s/%s_%s.txt", experimentName, "averagedKeyTraces", "Empty");
	} else if (these == MaskLib) {
		snprintf (fileName, 256, "%s/%s_%s.txt", experimentName, "averagedKeyTraces", "Lib");
	} else {
		snprintf (fileName, 256, "%s/%s_%s.txt", experimentName, "averagedKeyTraces", "TF");
	}
	fopen_s (&fp, fileName, "ab");
	fprintf (fp, "%d %d ", r, flow);
	for (int i = 0; i < numFrames; i++) {
	  fprintf (fp, "%0.2lf ", avgTrace[r][flow][i]);
	}
	fprintf (fp, "\n");
	free (fileName);
	
	fclose (fp);
	
}
//
//  Get virtual zeromer
//  Allocates memory which calling function must free
//
double *Zeromer::GetVirtualZeromer (Image *img, Region *region, Mask *mask, int r, int flow)
{
    double *virtualZ = NULL;
    const RawImage *raw = img->GetImage ();
    
    // Get average of all the empty wells
    int cnt = 0;
    double *averageBead = new double [numFrames];
	memset (averageBead,0,sizeof(double)*numFrames);
	for (int y=region->row;y<(region->row+region->h);y++) {
		 for (int x=region->col;x<(region->col+region->w);x++) {
            if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout))
                continue;
            
			if ((*mask)[x+(y*raw->cols)] & MaskEmpty) {
				for (int frame = 0; frame < numFrames; frame++) {
					averageBead[frame] += raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
				}
				cnt++;
			}
		}
	}
	
	// Need minimum threshold amount of empty wells as a cutoff...
	if (cnt > 0) {
        for (int frame = 0; frame < numFrames; frame++) {
			averageBead[frame] /= cnt;
		}
	}
    else {
        fprintf (stderr, "Zeromer: No empty wells in region %d\n", r);
		// Do we mask out this region with MaskIgnore? - NO!
		fprintf (stderr, "Applying MaskIgnore to Region %d\n", r);
		for (int y=region->row;y<(region->row+region->h);y++)
			for (int x=region->col;x<(region->col+region->w);x++)
				(*mask)[x+(y*raw->cols)] |= MaskIgnore;
        return (NULL);
    }
    
    // Get FFT of empty wells average
	
    fftw_complex *emptyOut = NULL;
	fftw_plan 		p;
    emptyOut = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * numFrames);
	if (!emptyOut)
		exit (1);
        
    p = fftw_plan_dft_r2c_1d (numFrames, averageBead, emptyOut, FFTW_ESTIMATE);
	
	fftw_execute(p);
	
	fftw_destroy_plan(p);
    
    // Dot product FFT with Hvector for this region
    for (int i = 0; i < numFrames; i++) {
        emptyOut[i][0] = (emptyOut[i][0] * H_vectors[r][flow][i][0]) / numFrames;
        emptyOut[i][1] = (emptyOut[i][1] * H_vectors[r][flow][i][1]) / numFrames;
    }
    
    // Inverse FFT of dot product C2r
	virtualZ = new double [numFrames];
    p = fftw_plan_dft_c2r_1d (numFrames, emptyOut, virtualZ, FFTW_ESTIMATE);
	
	fftw_execute(p);
	
	fftw_destroy_plan(p);
    free (emptyOut);
	
    // Return
    delete [] averageBead;
    return (virtualZ);
}
//
//	Calculate H Vectors
//	Need an H vector for every nuke in every region.  The H vector is used to generate the virtual zeromer for subsequent
//	flows.
//
void Zeromer::InitHVector (Image *img, Region *region, int r, Mask *mask, MaskType these, int flow)
{	
	const RawImage *raw = img->GetImage ();
	
	// Average the MaskEmpty traces for this region, this flow
	double *averageEmpty = new double [numFrames];
	memset (averageEmpty,0,sizeof(double)*numFrames);
	
	int cnt = 0;
	for (int y=region->row;y<(region->row+region->h);y++) {
		 for (int x=region->col;x<(region->col+region->w);x++) {
            if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout))
                continue;
            
			if ((*mask)[x+(y*raw->cols)] & MaskEmpty) {
				for (int frame = 0; frame < numFrames; frame++) {
					averageEmpty[frame] += raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
				}
				cnt++;
			}
		 }
	}
	
	if (cnt > 0) {
		for (int frame = 0; frame < numFrames; frame++) {
			averageEmpty[frame] /= cnt;
		}
	}
	
	
	// Average the Bead traces for this region, this flow
	double *averageBead = new double [numFrames];
	memset (averageBead,0,sizeof(double)*numFrames);
	
	cnt = 0;
	for (int y=region->row;y<(region->row+region->h);y++) {
		 for (int x=region->col;x<(region->col+region->w);x++) {
            if ((*mask)[x+(y*raw->cols)] & (MaskIgnore|MaskWashout))
                continue;
            
			if ((*mask)[x+(y*raw->cols)] & these) {
				for (int frame = 0; frame < numFrames; frame++) {
					averageBead[frame] += raw->image[x+(y*raw->cols) + (raw->frameStride * frame)];
				}
				cnt++;
			}
		 }
	}
	
	if (cnt > 0) {
		for (int frame = 0; frame < numFrames; frame++) {
			averageBead[frame] /= cnt;
		}
	}
	
	//
	// H = FFT (averageBead) / FFT (averageEmpty)
	// Store H for this region
	//
	if (H_vectors == NULL) {
		H_vectors = new fftw_complex **[numRegions];
		for (int i = 0; i < numRegions; i++) {
			H_vectors[i] = new fftw_complex *[numKeyFlows];
			for (int j = 0; j < numKeyFlows; j++)
				H_vectors[i][j] = NULL;
		}
	}
	//fprintf (stdout, "Assigned H_vector for region %d flow %d\n", r, flow);
	H_vectors[r][flow] = HVector(averageBead, averageEmpty, numFrames);
	
	delete [] averageEmpty;
	delete [] averageBead;
}
//
// Discreet Fourier Transform
//
fftw_complex *Zeromer::HVector (double *avgBead, double *avgEmpty, int cnt)
{
	fftw_complex	*beadOut = NULL;
	fftw_complex	*emptyOut = NULL;
	fftw_complex	*H = NULL;
	fftw_plan 		p;
	fftw_plan 		q;
	int N = 2 * ((cnt/2) + 1);
    
	beadOut = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * N);
    memset(beadOut,0,N*sizeof(fftw_complex));
	if (!beadOut)
		exit (1);
	
    emptyOut = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * N);
    memset(emptyOut,0,N*sizeof(fftw_complex));
	if (!emptyOut)
		exit (1);
	
    H = (fftw_complex *) fftw_malloc (sizeof(fftw_complex) * N);
    memset(H,0,N*sizeof(fftw_complex));
	if (!H)
		exit (1);
	
	// FFT (averageBead trace)
	p = fftw_plan_dft_r2c_1d (N, avgBead, beadOut, FFTW_ESTIMATE);
	
	fftw_execute(p);
	
	//fftw_print_plan(p);
	
	fftw_destroy_plan(p);
	
	// FFT (averageEmpty trace)
	q = fftw_plan_dft_r2c_1d (N, avgEmpty, emptyOut, FFTW_ESTIMATE);
	
	fftw_execute(q);
	
	//fftw_print_plan(p);
		
	// H  = FFT(B) / FFT(E)
    //fprintf (stdout, "HVector\n");
	for (int i=0;i<N/2;i++) {
		H[i][0] = beadOut[i][0] / emptyOut[i][0];
		H[i][1] = beadOut[i][1] / emptyOut[i][1];
        //fprintf (stdout, "[%04d] %0.3lf\n", i, creal(H[i]));
	}
    //fprintf (stdout, "\n");
	
	//Free beadOut and emptyOut
	fftw_free (beadOut);
	fftw_free (emptyOut);
	
	fftw_destroy_plan(q);
	fftw_cleanup();
	
	return (H);
}
