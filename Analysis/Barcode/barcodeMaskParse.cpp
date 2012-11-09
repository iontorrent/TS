/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//	Filename:	barcodeMaskParse.cpp
//	Description:	Reads binary mask file (see Mask::WriteRaw), outputs two column text file of r,c locations of specified masktype.
//	Author:	Eric Tsung
//


#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "IonVersion.h"
#include "BarCode.h"

int showHelp ()
{
	fprintf (stdout, "barcodeMaskParse - Dumps contents of barcode mask binary file to ");
	fprintf (stdout, "a text file.\n");
	fprintf (stdout, "Usage:\n");
	fprintf (stdout, "   barcodeMaskParse barcodeMaskInput\n");
	fprintf (stdout, "Options:\n");
	fprintf (stdout, "    -o textOutput [barcodeMask.txt]\n");
	return 0;
}

int main (int argc, char *argv[]) {
	//Defaults
	char *barcodeMaskBinFilename = NULL;
	char *barcodeMaskTextOutput = strdup("barcodeMask.txt");

    int c;
    while ( (c = getopt (argc, argv, "o:e:b:hv")) != -1 ) {
        switch (c) {
        	case 'o':
        		barcodeMaskTextOutput = strdup(optarg);
        		break;
			case 'e':
				//barcode ids to excluded
				break;
			case 'b':
				//barcode ids to include
				break;
			case 'h':
				showHelp();
				return (EXIT_SUCCESS);
				break;
            case 'v':   //version
            	fprintf (stdout, "%s", IonVersion::GetFullVersion("barcodeMaskParse").c_str());
            	return (EXIT_SUCCESS);
            	break;
            default:
            	fprintf (stderr, "What have we here? (%c)\n", c);
            	exit (EXIT_FAILURE);
        }
    }
    //fprintf (stdout, "optind,argc: %d,%d\n", optind, argc);
    if(argc - optind < 1) {  //One non-option arguments are required
		showHelp();
		return (EXIT_SUCCESS);
    }
    barcodeMaskBinFilename = argv[optind];

    barcode::OutputTextMask(barcodeMaskBinFilename, barcodeMaskTextOutput);
}
