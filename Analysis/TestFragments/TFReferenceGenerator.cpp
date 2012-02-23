/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <getopt.h> // for getopt_long
#include <assert.h>

#include "IonVersion.h"
#include "TFs.h"
#include "dbgmem.h"

using namespace std;

// TODO: Replace me with a cute python script

void usage ()
{
  fprintf (stdout, "TFReferenceGenerator - Convert DefaultTFs.conf to fasta\n");
  fprintf (stdout, "options:\n");
  fprintf (stdout, "   --TF\t\tSpecify filename containing Test Fragments to map to.\n");
  fprintf (stdout, "   --key\tOnly convert TFs with this key. (default ATCG)\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "usage:\n");
  fprintf (stdout, "   TFReferenceGenerator output.fasta\n");
  fprintf (stdout, "\n");
  exit(1);
}

#ifdef _DEBUG
void memstatus(void)
{
  memdump();
  dbgmemClose();
}

#endif /* _DEBUG */

int main(int argc, char *argv[])
{
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */

  printf ("%s - %s-%s (%s)\n\n", argv[0],
       IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());
  fflush (stdout);

  string key = "ATCG";
  char  *TFoverride = NULL;
  int c;
  int option_index = 0;

  static struct option long_options[] = {
      {"TF",  required_argument,  NULL, 0},
      {"key", required_argument,  NULL, 0},
      {NULL, 0, NULL, 0}};

  while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (c) {
    case (0):
      if (long_options[option_index].flag != 0)
        break;

      if (strcmp(long_options[option_index].name, "TF") == 0)
        TFoverride = optarg;

      if (strcmp(long_options[option_index].name, "key") == 0)
        key = optarg;
      break;

    default:
      fprintf(stderr, "Unrecognized option (%c)\n", c);
      exit(1);
    }
  }


  // Pick up the sff filename
  if (optind >= argc)
    usage();

  // open up our TF config file
  TFs tfs("TACG");
  if (TFoverride != NULL) { // TF config file was specified on command line
    if (tfs.LoadConfig(TFoverride) == false) {
      fprintf (stderr, "Error: Specified TF config file, '%s', was not found.\n", TFoverride);
      exit(1);
    }
  }
  else {  // Search for TF config file and open it
    char *tfConfigFileName = tfs.GetTFConfigFile();
    if (tfConfigFileName == NULL) {
      fprintf (stderr, "Error: No TF config file was found.\n");
      exit(1);
    }
    else {
      tfs.LoadConfig(tfConfigFileName);
      free (tfConfigFileName);
    }
  }

  TFInfo *tfInfo = tfs.Info();
  int numTFs = tfs.Num();

  FILE *fasta = fopen(argv[optind],"w");
  if (!fasta) {
    printf("Failed to open %s\n", argv[optind]);
    exit(1);
  }

  for(int tf = 0; tf < numTFs; tf++) {
    if (key != tfInfo[tf].key)
      continue;

    fprintf(fasta,">%s\n", tfInfo[tf].name);
    fprintf(fasta,"%s\n", tfInfo[tf].seq);
  }

  fclose(fasta);
}



