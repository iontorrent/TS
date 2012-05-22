/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BaseCallerControlOpts.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h> //EXIT_FAILURE
#include <ctype.h>  //tolower
#include <libgen.h> //dirname, basename
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "IonErr.h"

using namespace std;

void CafieControlOpts::DefaultCAFIEControl()
{
  singleCoreCafie = false;
  libPhaseEstimator = "spatial-refiner";
  //libPhaseEstimator = "nel-mead-treephaser";
//    libPhaseEstimator = "nel-mead-adaptive-treephaser";
  cfiedrRegionsX = 13, cfiedrRegionsY = 12;
  cfiedrRegionSizeX = 0, cfiedrRegionSizeY = 0;
  blockSizeX = 0, blockSizeY = 0;
  LibcfOverride = 0.0;
  LibieOverride = 0.0;
  LibdrOverride = 0.0;
  basecaller = "treephaser-swan";
  doCafieResidual = 0;
  numCafieSolveFlows = 0;
  // Options related to doing basecalling on just a subset of wells
  basecallSubsetFile = NULL;
}


CafieControlOpts::~CafieControlOpts()
{

  if (basecallSubsetFile)
    free (basecallSubsetFile);
}

void CafieControlOpts::EchoDerivedChipParams (int chip_len_x, int chip_len_y)
{
  //@TODO: isolate to cfe_control
  //overwrite cafie region size (13x12)
  if ( (cfiedrRegionSizeX != 0) && (cfiedrRegionSizeY != 0) && (blockSizeX != 0) && (blockSizeY != 0))
  {
    std::cout << "INFO: blockSizeX: " << blockSizeX << " ,blockSizeY: " << blockSizeY << std::endl;
    cfiedrRegionsX = blockSizeX /cfiedrRegionSizeX;
    cfiedrRegionsY = blockSizeY / cfiedrRegionSizeY;
    std::cout << "INFO: cfiedrRegionsX: " << cfiedrRegionsX << " ,cfiedrRegionsY: " << cfiedrRegionsY << std::endl;
  }

  //print debug information
  if ( (blockSizeX == 0) && (blockSizeY == 0))
  {
    unsigned short cafieYinc =
      ceil (chip_len_y / (double) cfiedrRegionSizeY);
    unsigned short cafieXinc =
      ceil (chip_len_x / (double) cfiedrRegionSizeX);
    std::cout << "DEBUG: precalculated values: cafieXinc: " << cafieXinc << " ,cafieYinc: " << cafieYinc << std::endl;
  }
}


void readBasecallSubsetFile (char *basecallSubsetFile, set< pair <unsigned short,unsigned short> > &basecallSubset)
{
  ifstream inFile;
  inFile.open (basecallSubsetFile);
  if (inFile.fail())
    ION_ABORT ("Unable to open basecallSubset file for read: " + string (basecallSubsetFile));

  vector <unsigned short> data;
  if (inFile.good())
  {
    string line;
    getline (inFile,line);
    char delim = '\t';

    // Parse the line
    size_t current = 0;
    size_t next = 0;
    while (current < line.length())
    {
      next = line.find (delim, current);
      if (next == string::npos)
      {
        next = line.length();
      }
      string entry = line.substr (current, next-current);
      istringstream i (entry);
      unsigned short value;
      char c;
      if (! (i >> value) || (i.get (c)))
      {
        ION_ABORT ("Problem converting entry \"" + entry + "\" from file " + string (basecallSubsetFile) + " to unsigned short");
      }
      else
      {
        data.push_back (value);
        if (data.size() ==2)
        {
          pair< unsigned short, unsigned short> thisPair;
          thisPair.first  = data[0];
          thisPair.second = data[1];
          basecallSubset.insert (thisPair);
          data.erase (data.begin(),data.begin() +2);
        }
      }
      current = next + 1;
    }
  }
  if (data.size() > 0)
    ION_WARN ("expected an even number of entries in basecallSubset file " + string (basecallSubsetFile));
}
