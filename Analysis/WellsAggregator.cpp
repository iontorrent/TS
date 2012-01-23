/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *	Purpose: combine one or more bfmask.bin files into a single file.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include "RawWells.h"

#define INFILE_MAX 256

int main (int argc, char *argv[])
{
	char *inFileName[INFILE_MAX] = {NULL};
	int numInFiles = 0;
	char *outFileName = {"1.wells"};
	bool debugflag = false;
	int n;
	RawWells outWell("./", outFileName);
	
	// Parse command line arguments
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				
				case 'd':	// print debug info
					debugflag = true;
				break;
				
				default:
					fprintf (stderr, "Unknown option %s\n", argv[argcc]);
					exit (1);
				break;
			}
		}
		else {
			inFileName[numInFiles++] = argv[argcc];
		}
		argcc++;
	}
	
	if (!inFileName[0]) {
		fprintf (stdout, "No input files specified\n");
		fprintf (stdout, "Usage: %s [-d] filename[ ...]\n", argv[0]);
		fprintf (stdout, "\t-d Prints debug information.\n");
		exit (1);
	}
	
	// Get the dimensions of the chip; needed to create output wells file
	char * wellPath = strdup (inFileName[0]);
	char * wellName = strdup (inFileName[0]);
	RawWells inWell(dirname(wellPath), basename(wellName));
	inWell.OpenForRead();
	size_t rows = inWell.NumRows();
	size_t cols = inWell.NumCols();
	
	for (int inCnt = 0; inCnt < numInFiles;inCnt++) {
		FILE *fp = NULL;
		fp = fopen (inFileName[inCnt], "rb");
		if (fp == NULL)
		{
			fprintf (stderr, "Cannot read file '%s': %s\n", inFileName[inCnt], strerror(errno));
			continue;
		}
		
		// Show me the Header information
		struct WellHeader hdr;
		n = fread(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
		assert(n==1);
		n = fread(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
		assert(n==1);
		hdr.flowOrder = (char *)malloc(sizeof(char) * (hdr.numFlows+1));
		n = fread(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);
		assert(n==hdr.numFlows);
	
		// null termiante floworder string
		char *flowOrder = (char *) malloc (hdr.numFlows + 1);
		strncpy (flowOrder, hdr.flowOrder, hdr.numFlows);
		
		if (inCnt == 0) {
			outWell.CreateEmpty (hdr.numFlows, hdr.flowOrder, rows, cols);
			outWell.OpenForWrite ();
		}
		// Print header information
		fprintf (stdout,"Number of Wells: %d\n", hdr.numWells);
		fprintf (stdout,"Number of Flows: %d\n", hdr.numFlows);
		fprintf (stdout, "Flow Order: %s\n", flowOrder);
		
		// Print all the flow data
		int rank;
		short x, y;
		float val[hdr.numWells];
		for (unsigned int well = 0; well < hdr.numWells; well++) {
			//For each well, read the data
			n = fread (&rank, sizeof(int),1,fp),
			assert(n==1);
			n = fread (&x, sizeof(short),1,fp);
			assert(n==1);
			n = fread (&y, sizeof(short),1,fp);
			assert(n==1);
			n = fread (val,sizeof(float),hdr.numFlows,fp);
			assert(n==hdr.numFlows);
			
			/*
			fprintf (stdout, "%u:%04d:%04d: ", rank,x,y);
			for (int f = 0; f < hdr.numFlows; f++) {
				fprintf (stdout, "%0.2f", val[f]);
				if (f+1 == hdr.numFlows)
					fprintf (stdout, "\n");
				else
					fprintf (stdout, " ");
			}
			*/
			for (int f = 0; f < hdr.numFlows; f++) {
				if (val[0]+val[1]+val[2]+val[3]+val[4]+val[5]+val[6] > 0.001) {
					outWell.WriteFlowgram (f, x, y, val[f]);
				}
			}
		}
		
		fclose (fp);
        free(flowOrder);
	}
	
	exit (EXIT_SUCCESS);
}
 
