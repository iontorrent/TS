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

#include <iostream>
#include <sstream>
#include <fstream>

#include "IonVersion.h"
#include "Mask.h"

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


void OutputTextMask ( const char *bcmask_filename, const char *bcmask_text_out )
{

  std::cerr << "Loading binary mask file '" << bcmask_filename << "'\n";
  Mask recheck ( bcmask_filename ); //check if it read in
  std::ofstream outStrm;
  std::cerr << "Writing text mask file to '" << bcmask_text_out << "'\n";
  outStrm.open ( bcmask_text_out );
  bool isGood = outStrm.good();
  assert ( isGood );
  if ( !isGood )
  {
    std::cerr << "Failure to write file.  Exiting.\n";
    exit ( 1 );
  }
  int h = recheck.H(); // max. y value
  int w = recheck.W(); // max. x value

  //fprintf (fOut, "%d %d\n", w, h);
  outStrm << "#Barcode locations, first row is flowcell's width and height.\n";
  outStrm << "#col\trow\tvalue\n";
  outStrm << w << "\t" << h << "\t-1\n";
  for ( int row=0; row<h; row++ )  // y
  {
    for ( int col=0; col<w; col++ ) // x
    {
      uint16_t barcodeId = recheck.GetBarcodeId ( col,row );
      if ( barcodeId!=0xffff )   //Don't bother to output empties
      {
        outStrm << row << "\t" << col << "\t" << barcodeId << "\n";
        isGood = isGood & outStrm.good();
      }
    }
  }
  if ( !isGood )
  {
    std::cerr << "Failure to write file.  Exiting.\n";
    exit ( 1 );
  }
  else
  {
    std::cerr << "Completed successfully.\n";
  }
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

    OutputTextMask(barcodeMaskBinFilename, barcodeMaskTextOutput);
}
