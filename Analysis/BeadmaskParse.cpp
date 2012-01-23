/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
//	Filename:	beadmaskParse.cpp
//	Description:	Reads binary mask file (see Mask::WriteRaw), outputs two column text file of r,c locations of specified masktype.
//	Author:	Bernard Puc
//
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>

#include "IonVersion.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "Utils.h"

int main (int argc, char *argv[])
{
 	// process command-line args
    char *beadfindFileName = NULL;
	char outputFileName[MAX_PATH_LENGTH] = {0};
    MaskType maskType = MaskNone;
	bool EXCLUDE = false;
    int c;
    while ( (c = getopt (argc, argv, "ehm:v")) != -1 )
    {
        switch (c)
        {
			case 'e':
				//Exclude from output any wells marked with:
				//	MaskIgnore, MaskExclude
				EXCLUDE=true;
				break;
			case 'h':
				fprintf (stdout, "%s -mMASK_TYPE FILENAME\n", argv[0]);
				fprintf (stdout, "MASK_TYPE can be MaskBead, MaskLive, MaskTF, MaskLib, MaskEmpty, MaskIgnore, MaskPinned, MaskDud, MaskAmbiguous\n ");
				exit (0);
				break;
            case 'm':   // type of mask to extract
                fprintf (stdout, "Request for %s\n", optarg);
                if (strncmp (optarg, "MaskBead", sizeof("MaskBead")) == 0)
                {
					sprintf (outputFileName, "MaskBead.mask");
                    fprintf (stdout, "Generating MaskBead maskfile\n");
                    maskType = MaskBead;
                }
                else if (strncmp (optarg, "MaskLive", sizeof("MaskLive")) == 0)
                {
					sprintf (outputFileName, "MaskLive.mask");
                    fprintf (stdout, "Generating MaskLive maskfile\n");
                    maskType = MaskLive;
                }
                else if (strncmp (optarg, "MaskTF", sizeof("MaskTF")) == 0)
                {
					sprintf (outputFileName, "MaskTF.mask");
                    fprintf (stdout, "Generating MaskTF maskfile\n");
                    maskType = MaskTF;
                }
                else if (strncmp (optarg, "MaskLib", sizeof("MaskLib")) == 0)
                {
					sprintf (outputFileName, "MaskLib.mask");
                    fprintf (stdout, "Generating MaskLib maskfile\n");
                    maskType = MaskLib;
                }
                else if (strncmp (optarg, "MaskEmpty", sizeof("MaskEmpty")) == 0)
                {
					sprintf (outputFileName, "MaskEmpty.mask");
                    fprintf (stdout, "Generating MaskEmpty maskfile\n");
                    maskType = MaskEmpty;
                }
                else if (strncmp (optarg, "MaskIgnore", sizeof("MaskIgnore")) == 0)
				{
					sprintf (outputFileName, "MaskIgnore.mask");
                    fprintf (stdout, "Generating MaskIgnore maskfile\n");
                    maskType = MaskIgnore;
                }
                else if (strncmp (optarg, "MaskPinned", sizeof("MaskPinned")) == 0)
				{
					sprintf (outputFileName, "MaskPinned.mask");
                    fprintf (stdout, "Generating MaskPinned maskfile\n");
                    maskType = MaskPinned;
                }
                else if (strncmp (optarg, "MaskDud", sizeof("MaskDud")) == 0)
                {
					sprintf (outputFileName, "MaskDud.mask");
                    fprintf (stdout, "Generating MaskDud maskfile\n");
                    maskType = MaskDud;
                }
                else if (strncmp (optarg, "MaskAmbiguous", sizeof("MaskAmbiguous")) == 0)
                {
					sprintf (outputFileName, "MaskAmbiguous.mask");
                    fprintf (stdout, "Generating MaskAmbiguous maskfile\n");
                    maskType = MaskAmbiguous;
                }
                else if (strncmp (optarg, "MaskExclude", sizeof("MaskExclude")) == 0)
                {
					sprintf (outputFileName, "MaskExclude.mask");
                    fprintf (stdout, "Generating MaskExclude maskfile\n");
                    maskType = MaskExclude;
                }
                else
                {
                    fprintf (stdout, "MaskType '%s' is not supported.\n", optarg);
                    exit (0);
                }
                break;
            case 'v':   //version
                fprintf (stdout, "%s", IonVersion::GetFullVersion("BeadmaskParse").c_str());
                return (0);
                break;
            
            default:
				fprintf (stdout, "whatever");
            break;
        }
    }
    
    // Pick up any non-option arguments (ie, source directory)
    for (c = optind; c < argc; c++)
    {
        beadfindFileName = argv[c];
        break; //cause we only expect one non-option argument
    }
    
	if (!beadfindFileName) {
		fprintf (stderr, "No input file specified\n");
		exit (1);
	}
	else {
		fprintf (stdout, "Parsing file: %s\n", beadfindFileName);
	}
	
    FILE *fp = NULL;
    fopen_s (&fp, beadfindFileName, "rb");
    if (!fp) {
        perror (beadfindFileName);
        exit (1);
    }
    
    FILE *fOut = NULL;
    fopen_s (&fOut, outputFileName, "wb");
    if (!fOut) {
        perror (outputFileName);
        exit (1);
    }
    
    int32_t w = 0;
    int32_t h = 0;
    
   	//  number of rows - height
    if ((fread (&h, sizeof(uint32_t), 1, fp )) != 1) {
        perror ("Reading width");
		exit (1);
    }
    //  number of columns - width
    if ((fread (&w, sizeof(uint32_t), 1, fp )) != 1) {
        perror ("Reading height");
		exit (1);
    }
	//	First line of output file contains, width and height of mask
	fprintf (fOut, "%d %d\n", w, h);
	
    //  mask value
    uint16_t mask = 0;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if ((fread (&mask, sizeof(uint16_t), 1, fp)) != 1) {	// Mask values , row-major
				perror ("Reading binary mask values");
				exit (1);
			}
            if (mask & maskType) {
				//if (EXCLUDE && ((mask & MaskIgnore) || (mask & MaskExclude)) {
				if (EXCLUDE && (mask & MaskIgnore)) {
						continue;
				}
                // Mask matches selector at this location
                fprintf (fOut, "%d %d\n", y, x);
                
            }
		}
    }
	fclose (fp);
    fclose (fOut);
    
    exit (0);
}
