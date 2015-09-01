/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "tvcutils.h"

#include <string>
//#include <fstream>
#include <stdio.h>
#include <ReferenceReader.h>

#include "OptArgs.h"
#include "IonVersion.h"

using namespace std;


void TVCUtilsHelp()
{
  printf ("\n");
  printf ("tvcutils %s-%s (%s) - Miscellaneous tools used by Torrent Variant Caller plugin and workflow.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   tvcutils <command> [options]\n");
  printf ("\n");
  printf ("Commands:\n");
  printf ("         prepare_hotspots  Convert BED or VCF file into a valid hotspot file\n");
  printf ("         validate_bed      Validate targets or hotspots file\n");
  printf ("         unify_vcf         Unify variants and annotations from all sources (tvc,IndelAssembly,hotpots)\n");
  printf ("\n");
}


int main(int argc, const char *argv[])
{

  if(argc < 2) {
    TVCUtilsHelp();
    return 1;
  }

  string tvcutils_command = argv[1];

  if      (tvcutils_command == "prepare_hotspots") return PrepareHotspots(argc-1, argv+1);
  else if (tvcutils_command == "validate_bed") return ValidateBed(argc-1, argv+1);
  else if (tvcutils_command == "unify_vcf") return UnifyVcf(argc-1, argv+1);
  else {
      fprintf(stderr, "ERROR: unrecognized tvcutils command '%s'\n", tvcutils_command.c_str());
      return 1;
  }
  return 0;
}

