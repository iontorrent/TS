/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/* utility to create a large wells file from a small wells file */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "RawWells.h"
#include "Mask.h"

int main (int argc, char *argv[])
{
	char srcFile[MAX_PATH_LENGTH] = {""};
	char dstFile[MAX_PATH_LENGTH] = {""};
	char *srcDir = NULL;
	char *dstDir = NULL;
	int factor = 0;
	int cycleMax = 0;
	int flowMax = 0;
	int c;
	int option_index = 0;
	static struct option long_options[] =
		{
			{NULL, 0, NULL, 0}
		};
		
	while ((c = getopt_long (argc, argv, "d:m:s:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
            case (0):
			break;
			case ('c'):
				// Limit number of cycles
				cycleMax = atoi (optarg);
			case ('d'):
				// Destination directory
				dstDir = strdup (optarg);
			break;
			case ('m'):
				// Multiplier
				factor = atoi (optarg);
			break;
			case ('s'):
				// Source directory
				srcDir = strdup (optarg);
			break;
			default:
			break;
		}
	}
	
	// Input test
	if (factor <= 0) {
		fprintf (stdout, "multiplicative factor needs to be greater than zero.  ie, -m 2\n");
		return 0;
	}
	if (srcDir == NULL) {
		fprintf (stdout, "Must specify source directory, ie, -s /results/analysis/output/Home/Auto_0007\n");
		return 0;
	}
	if (dstDir == NULL) {
		fprintf (stdout, "Must specify destination directory, ie, -d ./\n");
		return 0;
	}
	
	// Open an existing 'small' 1.wells file
	RawWells rawWells(srcDir, "1.wells");
	if (rawWells.OpenForRead ()){
		fprintf (stdout, "Error opening source 1.wells file\n");
		return 0;
	}
	
	// Open the bfmask.bin file
	sprintf (srcFile, "%s/%s", srcDir, "bfmask.bin");
	Mask srcMask(srcFile);
	
	// Parameters defining the monster 1.wells file
	// For each pixel in the source wells file, replicate that pixel based on the
	// factor.  ie, for a factor of 2, we are replicating the source 4x.
	int numFlows= rawWells.NumFlows();
	if (cycleMax == 0) {
		cycleMax = numFlows/4;
		flowMax = numFlows;
	}
	else {
		numFlows = cycleMax*4;
		flowMax = numFlows;
	}
	char *flowOrder = strdup (rawWells.FlowOrder ());
	int rows = 0;
	int cols = 0;
	int i = 0;
	rawWells.GetDims(&rows, &cols);
	// Resets file pointer to beginning of file - GetDims screws it all up
	rawWells.Close();
	rawWells.OpenForRead();
	
	//DEBUG OUTPUT
	fprintf (stdout, "Source chip dimensions (%d,%d)\n", cols, rows);
	fprintf (stdout, "Source run cycles %d\n", numFlows/4);
	
	// Open a new monster 1.wells file
	fprintf (stdout, "Initializing the new wells file: %s/1.wells\n", dstDir);
	sprintf (dstFile, "%s/%s", dstDir, "1.wells");
	RawWells monsterWells(NULL, dstFile);
	monsterWells.CreateEmpty (numFlows, flowOrder, rows*factor, cols*factor);
	monsterWells.OpenForWrite ();
	
	//DEBUG OUTPUT
	fprintf (stdout, "Destination chip dimensions (%d,%d)\n", cols*factor, rows*factor);
	fprintf (stdout, "Destination run cycles %d\n", cycleMax);
	
	// Create new monster bfmask.bin file
	Mask dstMask(cols*factor, rows*factor);
	
	// Fill in the monster wells file
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			// Read this pixel's data from source
			const WellData *wellData = NULL;
			//wellData = rawWells.ReadXY (x, y);
			wellData = rawWells.ReadNext ();
			
			for (int n = factor; n > 0; n--) {
				
				int yp = y + (n - 1) * rows;
				
				for (int m = factor; m > 0; m--) {
					
					int xp = x + (m - 1) * cols;
					
					// Copy flow values to new location
					for (int flow = 0; flow < flowMax; flow++) {
						monsterWells.WriteFlowgram (flow, xp, yp, wellData->flowValues[flow]);
						
					}
					
					// Copy mask value to new location
					dstMask[xp + (yp*(cols*factor))] = srcMask[i];
				}
			}
			// increment mask index
			i++;
		}
	}
	
	// Write out new bfmask.bin file
	sprintf (dstFile, "%s/%s", dstDir, "bfmask.bin");
	dstMask.WriteRaw (dstFile);
	
	// cleanup
	rawWells.Close ();
	monsterWells.Close ();
	
	return 0;
}
