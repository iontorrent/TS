/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <getopt.h> // for getopt_long

//#include "IonVersion.h"
#include "SFFWrapper.h"
#include "dbgmem.h"
//#include "json/json.h"

using namespace std;


class MetricGeneratorSNR {
public:
  MetricGeneratorSNR() {
    for (int idx = 0; idx < 8; idx++) {
      zeromerFirstMoment[idx] = 0;
      zeromerSecondMoment[idx] = 0;
      onemerFirstMoment[idx] = 0;
      onemerSecondMoment[idx] = 0;
    }
    count = 0;
  }

  void AddElement (uint16_t *corValues, const string& flowOrder)
  {
    int numFlowsPerCycle = flowOrder.length();
    count++;

    for (int iFlow = 0; iFlow < 8; iFlow++) {
      char nuc = flowOrder[iFlow%numFlowsPerCycle];
      if (corValues[iFlow] < 50) {        // Zeromer
        zeromerFirstMoment[nuc&7] += corValues[iFlow];
        zeromerSecondMoment[nuc&7] += corValues[iFlow] * corValues[iFlow];
      } else if (corValues[iFlow] < 150) { // Onemer
        onemerFirstMoment[nuc&7] += corValues[iFlow];
        onemerSecondMoment[nuc&7] += corValues[iFlow]* corValues[iFlow];
      }
    }
  }
  void PrintSNR() {
    double SNR = 0;
    if (count > 0) {
      double SNRx[8];
      for(int idx = 0; idx < 8; idx++) { // only care about the first 3, G maybe 2-mer etc
        double mean0 = zeromerFirstMoment[idx] / count;
        double mean1 = onemerFirstMoment[idx] / count;
        double var0 = zeromerSecondMoment[idx] / count - mean0*mean0;
        double var1 = onemerSecondMoment[idx] / count - mean1*mean1;
        double avgStdev = (sqrt(var0) + sqrt(var1)) / 2.0;
        if (avgStdev > 0.0)
          SNRx[idx] = (mean1-mean0) / avgStdev;
      }
      SNR = (SNRx['A'&7] + SNRx['C'&7] + SNRx['T'&7]) / 3.0;
    }
    printf("System SNR = %.2lf\n", SNR);
  }

private:
  int count;
  double zeromerFirstMoment[8];
  double zeromerSecondMoment[8];
  double onemerFirstMoment[8];
  double onemerSecondMoment[8];
};




int usage_keysnr ()
{
  fprintf(stderr, "Usage: %s keysnr [options] <in.sff>\n", "SFFUtils");
  fprintf(stderr, "\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "         -k,--key STRING   key sequence of reads included in snr calculation (default TCAG)\n");
  fprintf(stderr, "         -h,--help         print this message\n");
  fprintf(stderr, "\n");
  return 1;
}



int main_keysnr(int argc, char *argv[])
{
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */


  string key = "TCAG";

  int c;
  int option_index = 0;
  static struct option long_options[] =
    {
      {"key",         required_argument, NULL, 'k'},
      {"help",        no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}
    };

  while ((c = getopt_long(argc, argv, "k:", long_options, &option_index)) != -1) {
    switch (c) {
    case 'k':
      key = optarg;
      break;
    default:
      return usage_keysnr();
    }
  }

  if(argc != 1+optind)
    return usage_keysnr();


  MetricGeneratorSNR                metricGeneratorSNRLibrary;

  SFFWrapper sff;
  sff.OpenForRead(argv[optind]);
  string flowOrder = sff.GetHeader()->flow->s;

  while(true) {
    bool sffSuccess = true;
    const sff_t *readInfo = sff.LoadNextEntry(&sffSuccess);
    if (!readInfo || !sffSuccess)
      break;

    if (strncmp(sff_bases(readInfo), key.c_str(), key.length()) != 0)  // Key mismatch? Skip.
      continue;

    metricGeneratorSNRLibrary.AddElement(sff_flowgram(readInfo), flowOrder);
  }

  sff.Close();

  metricGeneratorSNRLibrary.PrintSNR();

  return 0;
}







/////////////////////////////////////////////////////////////////////////////////////
// The main code starts here




int usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Program: %s (Ion Analysis post-processing utilities)\n", "SFFUtils");
//  fprintf(stderr, "Version: %s\n\n", PACKAGE_VERSION);
  fprintf(stderr, "Usage:   %s <command> [options]\n\n", "SFFUtils");
  fprintf(stderr, "Command:\n");
  fprintf(stderr, "         keysnr        Compute key SNR from SFF ionogram signal\n");
  fprintf(stderr, "         summary       Produce summary of read lengths and predicted qualities\n");
  fprintf(stderr, "         read          Converts SFF formatted file to FASTQ formatted file\n");
  fprintf(stderr, "\n");
  return 1;
}

int main_summary(int argc,  const char *argv[]);
int main_read(int argc, char *argv[]);


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

  if(argc < 2) return usage();
  else if(0 == strcmp(argv[1], "keysnr")) return main_keysnr(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "summary")) return main_summary(argc-1, ((const char **)argv)+1);
  else if(0 == strcmp(argv[1], "read")) return main_read(argc-1, argv+1);
  else {
      fprintf(stderr, "unrecognized command '%s'\n", argv[1]);
      return 1;
  }
  return 0;
}




