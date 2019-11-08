/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include "IonVersion.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "Utils.h"
#include "Region.h"

bool ReadExclusionMask(int excludeMask[][2], const char *excludeFileName, int wt, int ht)
{
  // read exclusion mask, stored as [start end] values
  FILE *excludefp = NULL;
  char* path = GetIonConfigFile(excludeFileName);
  if ( path ){
    
    fopen_s ( &excludefp, path, "rt" );
    if ( excludefp ){
      fprintf( stdout, "Reading Exclusion mask from file: %s\n", path);
      int nChar = 64;
      char *line;
      line = new char[nChar];

      // first line is chip size
      int maskSize[2];
      int bytes_read = getline(&line,(size_t *)&nChar,excludefp);
      sscanf ( line,"%d\t%d",&maskSize[0], &maskSize[1] );
      if ( (maskSize[0] != wt) || (maskSize[1] != ht) ){
          fprintf( stdout, "Error: Incorrect exclusion mask size.\n" );
          return(false);
      }

      for ( int y = 0; y < ht; y++ )
      {
        bytes_read = getline(&line,(size_t *)&nChar,excludefp);
        if ( bytes_read > 0 ){
          sscanf ( line,"%d\t%d",&excludeMask[y][0], &excludeMask[y][1] );
        }
      }
      delete [] line;
      fclose(excludefp);
      return(true);
    } 
  } else {
    fprintf( stdout, "Unable to find exclusion mask file %s\n", excludeFileName);
  }
  return(false);
}

int main ( int argc, char *argv[] )
{
  // process command-line args
  char* beadfindFileName = NULL;
  char* outputFileName = NULL;
  std::vector<std::string> folders;
  int c;
  char* excludeFileName = NULL;
  bool initWithExclude = false;
  while ( ( c = getopt ( argc, argv, "i:o:e:hm:v:c" ) ) != -1 ) {
    switch ( c ) {
    case 'i':
      beadfindFileName = strdup ( optarg );
      break;
    case 'o':
      outputFileName = strdup ( optarg );
      break;
    case 'h':
      fprintf ( stdout, "%s -i in -o out folders \n", argv[0] );
      exit ( 0 );
      break;
    case 'v':   //version
      fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "BeadmaskMerge" ).c_str() );
      return ( 0 );
      break;
    case 'e':
      excludeFileName = strdup ( optarg );
      break;
    case 'c':
      initWithExclude = true;
      break;
    default:
      fprintf ( stdout, "whatever" );
      break;
    }
  }

  for ( c = optind; c < argc; c++ ) {
    folders.push_back ( argv[c] );
  }

  if ( !beadfindFileName ) {
    fprintf ( stderr, "No input file specified\n" );
    exit ( 1 );
  } else {
    fprintf ( stdout, "Reading from file: %s\n", beadfindFileName );
  }

  if ( folders.size() < 1 ) {
    fprintf ( stderr, "No input directories specified\n" );
    exit ( 1 );
  } else {
    for ( unsigned int f=0;f<folders.size();f++ ) {
      fprintf ( stdout, "Reading from folder: %s\n", folders[f].c_str() );
    }
  }

  if ( !outputFileName ) {
    fprintf ( stderr, "No output file specified\n" );
    exit ( 1 );
  } else {
    fprintf ( stdout, "Writing into file: %s\n", outputFileName );
  }


  const char* size = GetProcessParam ( folders[0].c_str(), "Chip" );
  char* size_copy = strdup( size );
  int wt=atoi ( strtok ( size_copy,"," ) );
  int ht=atoi ( strtok ( NULL, "," ) );
  free ( size_copy );
  if ( wt==0 || ht==0 ) {
    fprintf ( stderr, "Incorrect Chip Dimensions [%s]\n", size);
    exit ( 1 );
  }

  Mask fullmask (wt, ht);
  if(initWithExclude){
	  fullmask.SetAllExclude();
  }

  //  mask value
  uint16_t mask = 0;

  // read exclusion mask from file
  bool applyExclusionMask = false;
  int excludeMask[ht][2];
  if (excludeFileName){
    applyExclusionMask = ReadExclusionMask(excludeMask, excludeFileName, wt, ht);
  }

  for ( unsigned int f=0;f<folders.size();f++ ) {

    std::stringstream ss;
    ss << folders[f] << "/" << std::string ( beadfindFileName );
    FILE *fp = NULL;
    fopen_s ( &fp, ss.str().c_str(), "rb" );
    if ( !fp ) {
      perror ( ss.str().c_str() );
      exit ( 1 );
    }

    int32_t w = 0;
    int32_t h = 0;
    //  number of rows - height
    if ( ( fread ( &h, sizeof ( uint32_t ), 1, fp ) ) != 1 ) {
      perror ( "Reading width" );
      exit ( 1 );
    }
    //  number of columns - width
    if ( ( fread ( &w, sizeof ( uint32_t ), 1, fp ) ) != 1 ) {
      perror ( "Reading height" );
      exit ( 1 );
    }

    // extract offsets from folder name
    char* size = GetProcessParam ( folders[f].c_str(), "Block" );
    int xoffset=atoi ( strtok ( size,"," ) );
    int yoffset=atoi ( strtok ( NULL, "," ) );

    for ( int y = 0; y < h; y++ ) {
      for ( int x = 0; x < w; x++ ) {
        if ( ( fread ( &mask, sizeof ( uint16_t ), 1, fp ) ) != 1 ) { // Mask values , row-major
          perror ( "Reading binary mask values" );
          exit ( 1 );
        }

        //fullmask.SetBarcodeId(x+xoffset,y+yoffset,mask); // same as next line
        fullmask[wt* ( yoffset+y ) +xoffset+x] = mask;
      }
    }
    fclose ( fp );
  }
  
  // apply exclusion mask
  if ( applyExclusionMask ){
    for ( int y = 0; y < ht; y++ ) {
      for ( int x = 0; x < wt; x++ ) {
        if ( excludeMask[y][0] == excludeMask[y][1] ){
          fullmask[wt*y + x] = MaskExclude;
        } else {
          if ( (x < excludeMask[y][0]) || (x > excludeMask[y][1]) )
            fullmask[wt*y + x] = MaskExclude;
        }
      }
    }
  }

  // write mask and stats files
  Region wholeChip(0, 0, wt, ht);
  std::string outputStatsFile = outputFileName;
  outputStatsFile = outputStatsFile.substr(0,outputStatsFile.size()-3) + "stats";
  
  fullmask.UpdateBeadFindOutcomes ( wholeChip, outputFileName, 0, 0, outputStatsFile.c_str() );
  //fullmask.WriteRaw ( outputFileName );
    
  exit ( 0 );
}
