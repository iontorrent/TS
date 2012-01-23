/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *	Purpose: combine one or more bfmask.bin files into a single file.
 *
 */
#include "stdio.h"
#include "stdlib.h"

#include "../Mask.h"

#define INFILE_MAX 256

int main (int argc, char *argv[])
{
	char *inFileName[INFILE_MAX] = {NULL};
	int numInFiles = 0;
	char *outFileName = {"./bfmask.bin"};
	bool debugflag = false;
	
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
		fprintf (stdout, "Usage: %s [-d] bfmask.bin-filename[ ...]\n", argv[0]);
		fprintf (stdout, "\t-d Prints debug information.\n");
		exit (1);
	}

	// Create aggregate mask object
	Mask	aggregateMask(1,1);
	
	for (int inCnt = 0; inCnt < numInFiles;inCnt++) {
		
		//Open the next input file
		Mask inMask (inFileName[inCnt]);
		
		//Initialize the aggregate Mask object, if its not yet initialized
		if (aggregateMask.W() == 1 && aggregateMask.H() == 1) {
			aggregateMask.Init (inMask.W(),inMask.H(),MaskExclude);
		}
		
		// Start out with the mask all set to MaskExclude
		//		For every mask
		//			For every position
		//				if aggregate == MaskExclude
		//					Copy input to aggregate
		//					
		for (int y=0; y<aggregateMask.H(); y++) {
			for (int x=0; x<aggregateMask.W(); x++) {

				if (aggregateMask.Match (x+y*aggregateMask.W(),MaskExclude)) {
					aggregateMask[x+y*aggregateMask.W()] = inMask[x+y*aggregateMask.W()];
				}

			}
		}
	}

	// Save new mask to file
	aggregateMask.WriteRaw (outFileName);
	
	// Create bfmask.stats file for the new mask
	Region wholeChip;
	wholeChip.row = 0;
	wholeChip.col = 0;
	wholeChip.w = aggregateMask.W();
	wholeChip.h = aggregateMask.H();
	aggregateMask.DumpStats (wholeChip, "./bfmask.stats");
	
	exit (EXIT_SUCCESS);
}
