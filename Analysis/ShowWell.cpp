/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *  Opens a wells file and displays the header and data
 */
#define VERSION "0.1"
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "RawWells.h"
#include "LinuxCompat.h"
#include "IonVersion.h"

int main (int argc, char *argv[])
{
	bool allWells = false;
    char *wellFile = NULL;
    int n = 0; // number elements read
	
  	// process command-line args
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				case 'a': // all the wells
					allWells = true;
					break;
				case 'h':
				case 'H':
					fprintf (stdout, "ShowWell - Create ASCII dump of 1.wells file.\n");
					fprintf (stdout, "options:\n");
					fprintf (stdout, "   -a\tList all wells (default is to list only non-empty wells)\n");
					fprintf (stdout, "\n");
					fprintf (stdout, "usage:\n   ShowWell [-a] wellfilename\n");
					fprintf (stdout, "\n");
					return (0);
					break;
				case 'v':
				case 'V':
					fprintf (stdout, "%s", IonVersion::GetFullVersion("ShowWell").c_str());
					return (0);
					break;
				default:
					fprintf (stdout, "Unrecognized option -%c\n", argv[argcc][1]);
					fprintf (stdout, "Try -h for help\n");
					return (0);
					break;
			}
		}
		else {
			wellFile = argv[argcc];
		}
        argcc++;
    }
	    
    if (wellFile == NULL)
		wellFile = "./1.wells";
		
    FILE *fp = NULL;
    fopen_s (&fp, wellFile, "rb");
    if (fp == NULL)
    {
        fprintf (stderr, "Cannot read file '%s': %s\n", wellFile, strerror(errno));
        return (1);
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

    // Print header information
    fprintf (stdout,"Number of Wells: %d\n", hdr.numWells);
    fprintf (stdout,"Number of Flows: %d\n", hdr.numFlows);
    fprintf (stdout, "Flow Order: %s\n", hdr.flowOrder);
    
    // Print all the flow data
    int rank;
	unsigned int goodWells = 0;
    short x, y;
    short minX = SHRT_MAX , maxX = 0;
    short minY = SHRT_MAX, maxY = 0;
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
        
        // Only print non-zero flow data
        //if ((val[0] + val[1] + val[2] + val[3]) != 0.0){
		//if (rank != 0) {
		if (allWells) {
            fprintf (stdout, "%u:%04d:%04d: ", rank,x,y);
            for (int f = 0; f < hdr.numFlows; f++) {
                fprintf (stdout, "%0.2f", val[f]);
                if (f+1 == hdr.numFlows)
                    fprintf (stdout, "\n");
                else
                    fprintf (stdout, " ");
            }
			
			if (x < minX || well == 0)
				minX = x;
			if (x > maxX || well == 0)
				maxX = x;
			if (y < minY || well == 0)
				minY = y;
			if (y > maxY || well == 0)
				maxY = y;
			goodWells++;
		}
		else {
			if ((val[0] + val[1] + val[2] + val[3]) != 0.0){
				fprintf (stdout, "%u:%04d:%04d: ", rank,x,y);
				for (int f = 0; f < hdr.numFlows; f++) {
					fprintf (stdout, "%0.2f", val[f]);
					if (f+1 == hdr.numFlows)
						fprintf (stdout, "\n");
					else
						fprintf (stdout, " ");
				}
				
				if (x < minX || well == 0)
					minX = x;
				if (x > maxX || well == 0)
					maxX = x;
				if (y < minY || well == 0)
					minY = y;
				if (y > maxY || well == 0)
					maxY = y;
				goodWells++;
			}
		}
    }
    
    fclose (fp);
    
	fprintf (stdout, "\n");
	fprintf (stdout, "Good Wells: %d\n", goodWells);
	fprintf (stdout, "Row: %d to %d\n", minY, maxY);
	fprintf (stdout, "Column: %d to %d\n", minX, maxX);
	fprintf (stdout, "\n");
	
    return (0);
}
