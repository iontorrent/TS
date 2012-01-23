/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/* -*- Mode: C; tab-width: 4 -*- */
//
// Ion Torrent Systems, Inc.
// Test Fragment Mapper & Stats tool
// (c) 2009
// $Rev: $
//      $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h> // for getopt_long
#include <assert.h>

#include <vector>

#include "IonVersion.h"
#include "RawWells.h"
#include "SFFWrapper.h"
#include "fstrcmp.h"
#include "Histogram.h"
#include "file-io/ion_util.h"
#include "fstrcmp.h"
#include "TFs.h"

#include "dbgmem.h"

// #define EXACT_MATCH
#define MAX_READLEN 100
#define MAX_BUFF 2048
#define MAX_HP  8
#define strMax  1024


const int numHPs = MAX_HP; // 0 thru 7
char *flowOrder = strdup ("TACG");
int numFlowsPerCycle = strlen (flowOrder);


/////////////////////////////////////////////////////////////////////////////////////
// Below is a collection of small classes dedicated to calculating and reporting metrics or metric groups

class MetricGeneratorRawSignal {
public:
  MetricGeneratorRawSignal() {
    maxHP = MAX_HP;
    numSamples = 1000; // pick this many TF's per TF as representative samples for overlap plotting
    rawSignal.resize(maxHP);
    corSignal.resize(maxHP);
    numSignals.assign(maxHP,0);
    signalOverlapList.assign(maxHP,std::vector<float>(numSamples,0));

    for(int hp = 0; hp < maxHP; hp++) {
      rawSignal[hp] = new Histogram(1001, -2.0, maxHP+1.0);
      corSignal[hp] = new Histogram(1001, -2.0, maxHP+1.0);
    }
  }

  ~MetricGeneratorRawSignal() {
    for(int hp = 0; hp < maxHP; hp++) {
      delete rawSignal[hp];
      delete corSignal[hp];
    }
  }

  void AddElement(int nmer, float currentValueRaw, float currentValueSff) {
    rawSignal[nmer]->Add(currentValueRaw);
    corSignal[nmer]->Add(currentValueSff);

    int loc;
    if (numSignals[nmer] < numSamples) {
      loc = numSignals[nmer];
      numSignals[nmer]++;
    } else // once we get numSamples, just start replacing at random, should provide a representitive sampled population, but clearly favoring the later samples, if I knew in advance how many I had, I could select fairly uniformly throughout
      loc = rand() % numSamples;
    signalOverlapList[nmer][loc] = currentValueSff;
  }

  void PrintSNRMetrics() {
    printf("Raw HP SNR = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %.2lf", hp, rawSignal[hp]->SNR() / (double)std::max(hp,1));
    }
    printf("\n");

    printf("Corrected HP SNR = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %.2lf", hp, corSignal[hp]->SNR() / (double)std::max(hp,1));
    }
    printf("\n");
  }

  void PrintOverlapMetrics() {
    printf("Raw signal overlap = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      if (rawSignal[hp]->Count() > 0)
        rawSignal[hp]->Dump((char *)"stdout", 3);
      else
        printf("0");
    }
    printf("\n");

    printf("Corrected signal overlap = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      if (rawSignal[hp]->Count() > 0) {
        for(int samp = 0; samp < numSignals[hp]; samp++)
          printf("%.3lf ", signalOverlapList[hp][samp]);
      } else
        printf("0");
    }
    printf("\n");

    printf("TransferPlot = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      printf("%d %.4lf %.4lf", hp, corSignal[hp]->Mean(), corSignal[hp]->StdDev());
    }
    printf("\n");
  }

private:
  int maxHP;
  int numSamples;
  std::vector<Histogram *> rawSignal;
  std::vector<Histogram *> corSignal;
  std::vector<std::vector<float> > signalOverlapList;
  std::vector<int> numSignals;
};




class MetricGeneratorQualityHistograms {
public:
  MetricGeneratorQualityHistograms() {
    matchHist = new Histogram(MAX_READLEN + 1, 0, MAX_READLEN);
    q10Hist = new Histogram(MAX_READLEN + 1, 0, MAX_READLEN);
    q17Hist = new Histogram(MAX_READLEN + 1, 0, MAX_READLEN);
  }
  ~MetricGeneratorQualityHistograms() {
    delete matchHist;
    delete q10Hist;
    delete q17Hist;
  }

  void AddElement(int matchMinusMisMatch, int q10readlen, int q17readlen) {
    matchHist->Add(matchMinusMisMatch);
    q10Hist->Add(q10readlen);
    q17Hist->Add(q17readlen);
  }

  void PrintMeanMode() {
    printf("Match = %.1lf\n", matchHist->Mean());
    printf("Avg Q10 read length = %.1lf\n", q10Hist->Mean());
    printf("Avg Q17 read length = %.2lf\n", q17Hist->Mean());
    printf("MatchMode = %d\n", matchHist->ModeBin());
    printf("Q10Mode = %d\n", q10Hist->ModeBin());
    printf("Q17Mode = %d\n", q17Hist->ModeBin());
  }

  void PrintMetrics50() {
    int numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += matchHist->GetCount(readLen);
    printf("50Match = %d\n", numHQ);

    numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += q10Hist->GetCount(readLen);
    printf("50Q10 = %d\n", numHQ);
    // now calc the # reads such that the avg is exactly (or above) 50
    if (numHQ > 0) {
      numHQ = 0;
      double avg = 0.0;
      int total = 0;
      for(int readLen = MAX_READLEN; readLen >= 0; readLen--) {
        int num = q10Hist->GetCount(readLen);
        if ((num > 0) && (total == 0)) {
          avg = readLen;
          total = num;
          numHQ += num;
        } else if (num > 0) {
          total += num;
          double newAvg = avg*((double)total-(double)num)/(double)total + (double)num/(double)total * readLen;
          if (newAvg >= 50.0) {
            avg = newAvg;
            numHQ += num;
          } else {
            // need to take the subset of num that exactly makes this 50
            int j;
            for(j=1;j<num;j++) {
              newAvg = avg*((double)num-(double)j-1.0)/(double)num + (double)j/(double)(numHQ+j) * readLen;
              if (newAvg < 50.0) {
                break;
              }
            }
            numHQ += j;
            break;
          }
        }
      }
    }
    printf("50Q10A = %d\n", numHQ);

    numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += q17Hist->GetCount(readLen);
    printf("50Q17 = %d\n", numHQ);
    // now calc the # reads such that the avg is exactly (or above) 50
    if (numHQ > 0) {
      numHQ = 0;
      double avg = 0.0;
      int total = 0;
      for(int readLen = MAX_READLEN; readLen >= 0; readLen--) {
        int num = q17Hist->GetCount(readLen);
        if ((num > 0) && (total == 0)) {
          avg = readLen;
          total = num;
          numHQ += num;
        } else if (num > 0) {
          total += num;
          double newAvg = avg*((double)total-(double)num)/(double)total + (double)num/(double)total * readLen;
          if (newAvg >= 50.0) {
            avg = newAvg;
            numHQ += num;
          } else {
            // need to take the subset of num that exactly makes this 50
            int j;
            for(j=0;j<num;j++) {
              newAvg = avg*((double)num-(double)j-1.0)/(double)num + (double)j/(double)(numHQ+j) * readLen;
              if (newAvg < 50.0)
                break;
            }
            numHQ += j;
            break;
          }
        }
      }
    }
    printf("50Q17A = %d\n", numHQ);
  }

  void PrintHistogramDumps() {
    printf("Match-Mismatch = ");
    matchHist->Dump(stdout);
    printf("\n");

    printf("Q10 = ");
    q10Hist->Dump(stdout);
    printf("\n");

    printf("Q17 = ");
    q17Hist->Dump(stdout);
    printf("\n");
  }

private:
  Histogram *matchHist;
  Histogram *q10Hist;
  Histogram *q17Hist;
};



class MetricGeneratorTopIonograms {
public:
  MetricGeneratorTopIonograms() {numTopIonograms = 0;}
  ~MetricGeneratorTopIonograms() {}
  void SetNumTopIonograms(int _numTopIonograms) {
    numTopIonograms = _numTopIonograms;
    row.assign(numTopIonograms,0);
    col.assign(numTopIonograms,0);
    qualMetric.assign(numTopIonograms,0);
    raw.resize(numTopIonograms);
    cor.resize(numTopIonograms);
  }

  void AddElement(int x, int y, double metric, float *rawValues, uint16_t *corValues, int len) {
    // see if this is one of our top Ionorgams
    for(int topI = 0; topI < numTopIonograms; topI++) {
      if (metric > qualMetric[topI]) { // MGD - if the qualMetric were initialized to some value, would act as a nice filter to allow only reads above 50 for example, but for now set to zero
        qualMetric[topI] = metric;
        row[topI] = y;
        col[topI] = x;
        raw[topI].assign(rawValues,rawValues+len);
        cor[topI].assign(corValues,corValues+len);
        break;
      }
    }
  }


  void PrintTopIonograms(char *flowOrder, int numFlowsPerCycle, char *currentKey, char *currentSeq) {

    // dump the top N ionograms
    char topSeq[strMax];
    char tfSeq[strMax];
    for(int topI = 0; topI < numTopIonograms; topI++) {
      if (qualMetric[topI] == 0)
        continue;

      printf("Top %d = %d, %d, ", topI+1, row[topI], col[topI]);
      for(int flow = 0; flow < (int)raw[topI].size(); flow++) {
        printf("%.2lf ", raw[topI][flow]);
      }
      printf(", ");
      int topBases = 0;
      for(int flow = 0; flow < (int)raw[topI].size(); flow++) {
        printf("%.2lf ", 0.01f*cor[topI][flow]);
        int mer = (int)(0.01f*cor[topI][flow]+0.5f);
        while (mer > 0) {
          topSeq[topBases] = flowOrder[flow%numFlowsPerCycle];
          mer--;
          topBases++;
          if (topBases >= strMax) {
            topBases = strMax - 1;
            break;
          }
        }
      }
      topSeq[topBases] = 0;
      printf(",");
      strcpy(tfSeq, currentKey);
      strcat(tfSeq, currentSeq);
      PrintAlignment(tfSeq, topSeq, flowOrder, numFlowsPerCycle);
      printf("\n");
    }
  }

  void PrintAlignment(char *refSeq, char *testSeq, char *flowOrder, int numFlowsPerCycle)
  {
    int refLen = strlen(refSeq);
    int testLen = strlen(testSeq);

    // convert each into array of flow-space hits
    int refArray[1000];
    int testArray[1000];
    int i;
    int refBases = 0;
    int testBases = 0;
    int flowLenRef = 0; // this var used to stop us from penalizing in the match-mismatch calcs when we go beyond our ref readlength
    i = 0;
    while (refBases < refLen && testBases < testLen) {
      refArray[i] = 0;
      while (flowOrder[i%numFlowsPerCycle] == refSeq[refBases] && refBases < refLen) {
        refArray[i]++;
        refBases++;
        flowLenRef = i;
      }

      testArray[i] = 0;
      while (flowOrder[i%numFlowsPerCycle] == testSeq[testBases] && testBases < testLen) {
        testArray[i]++;
        testBases++;
      }

      i++;
    }
    int flowLen = i;

    // generate the alignment strings
    char refBuf[strMax];
    char testBuf[strMax];
    char bars[strMax];
    int k = 0;
    int j;
    int iref = 0;
    int itest = 0;
    for(i=0;i<flowLen;i++) {
      if (refArray[i] > 0 || testArray[i] > 0) {
        int max = (refArray[i] > testArray[i] ? refArray[i] : testArray[i]);
        int refCount = 0;
        int testCount = 0;
        for(j=0;j<max;j++) {
          if (refCount < refArray[i]) {
            refBuf[k] = refSeq[iref];
            iref++;
          } else {
            refBuf[k] = '-';
          }
          refCount++;

          if (testCount < testArray[i]) {
            testBuf[k] = testSeq[itest];
            itest++;
          } else {
            testBuf[k] = '-';
          }
          testCount++;

          k++;
          if (k >= (strMax - 1)) {
            k = strMax - 1;
            break;
          }
        }
      }
    }
    refBuf[k] = 0;
    testBuf[k] = 0;
    for(i=0;i<k;i++) {
      if (refBuf[i] == testBuf[i] && testBuf[i] != '-') {
        bars[i] = '|';
      } else {
        bars[i] = ' ';
      }
    }
    bars[i] = 0;

    printf("%s,%s,%s", refBuf, bars, testBuf);
  }

private:
  int numTopIonograms;
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double>  qualMetric;
  std::vector<std::vector<float> > raw;
  std::vector<std::vector<uint16_t> > cor;
};



class MetricGeneratorHPAccuracy {
public:
  MetricGeneratorHPAccuracy() {
    for(int hp = 0; hp < MAX_HP; hp++)
      hpAccuracy[hp] = hpCount[hp] = 0;
  }

  void PrintHPAccuracy() {
    printf("Per HP accuracy = ");
    for(int hp = 0; hp < MAX_HP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %d/%d", hp, hpAccuracy[hp], hpCount[hp]);
    }
    printf("\n");
  }

  int hpAccuracy[MAX_HP];
  int hpCount[MAX_HP];
};


class MetricGeneratorSNR {
public:
  MetricGeneratorSNR() {
    for (int nuc = 0; nuc < 4; nuc++) {
      zeromerFirstMoment[nuc] = 0;
      zeromerSecondMoment[nuc] = 0;
      onemerFirstMoment[nuc] = 0;
      onemerSecondMoment[nuc] = 0;
    }
    count = 0;
  }

  void AddElement (uint16_t *corValues, char *Key)
  {
    count++;
    for (int iFlow = 0; iFlow < 8; iFlow++) {
      if (*Key == flowOrder[iFlow%numFlowsPerCycle]) { // Onemer
        switch (flowOrder[iFlow%numFlowsPerCycle]) {
        case 'T':
          onemerFirstMoment[0] += corValues[iFlow];
          onemerSecondMoment[0] += corValues[iFlow]* corValues[iFlow];
          break;
        case 'A':
          onemerFirstMoment[1] += corValues[iFlow];
          onemerSecondMoment[1] += corValues[iFlow] * corValues[iFlow];
          break;
        case 'C':
          onemerFirstMoment[2] += corValues[iFlow];
          onemerSecondMoment[2] += corValues[iFlow] * corValues[iFlow];
          break;
        case 'G':
          onemerFirstMoment[3] += corValues[iFlow];
          onemerSecondMoment[3] += corValues[iFlow] * corValues[iFlow];
          break;
        }
        Key++;
      } else {  // Zeromer
        switch (flowOrder[iFlow%numFlowsPerCycle]) {
        case 'T':
          zeromerFirstMoment[0] += corValues[iFlow];
          zeromerSecondMoment[0] += corValues[iFlow] * corValues[iFlow];
          break;
        case 'A':
          zeromerFirstMoment[1] += corValues[iFlow];
          zeromerSecondMoment[1] += corValues[iFlow] * corValues[iFlow];
          break;
        case 'C':
          zeromerFirstMoment[2] += corValues[iFlow];
          zeromerSecondMoment[2] += corValues[iFlow] * corValues[iFlow];
          break;
        case 'G':
          zeromerFirstMoment[3] += corValues[iFlow];
          zeromerSecondMoment[3] += corValues[iFlow] * corValues[iFlow];
          break;
        }
      }
    }
  }
  void PrintSNR() {
    double SNR = 0;
    if (count > 0) {
      for(int nuc = 0; nuc < 3; nuc++) { // only care about the first 3, G maybe 2-mer etc
        double mean0 = zeromerFirstMoment[nuc] / count;
        double mean1 = onemerFirstMoment[nuc] / count;
        double var0 = zeromerSecondMoment[nuc] / count - mean0*mean0;
        double var1 = onemerSecondMoment[nuc] / count - mean1*mean1;
        double avgStdev = (sqrt(var0) + sqrt(var1)) / 2.0;
        if (avgStdev > 0.0)
          SNR += (mean1- mean0) / avgStdev;
      }
      SNR /= 3.0;
    }
    printf("System SNR = %.2lf\n", SNR);
  }

private:
  int count;
  double zeromerFirstMoment[4];
  double zeromerSecondMoment[4];
  double onemerFirstMoment[4];
  double onemerSecondMoment[4];
};


/////////////////////////////////////////////////////////////////////////////////////
// The main code starts here





#ifdef _DEBUG
void memstatus(void)
{
  memdump();
  dbgmemClose();
}

#endif /* _DEBUG */

int showHelp ()
{
  fprintf (stdout, "TFMapper - Basic mapping of reads to Test Fragments listed in DefaultTFs.conf file\n");
  fprintf (stdout, "options:\n");
  fprintf (stdout, "   --TF\t\tSpecify filename containing Test Fragments to map to.\n");
  fprintf (stdout, "   -e,--experiment-Directory\tSpecify directory containing experiment files\n");
  fprintf (stdout, "   -s,--tf-score\tSet confidence level for scoring.  (0.70 is default)\n");
  fprintf (stdout, "   -m,--mode\tSet processing mode.  0 = TF, 1 is Lib.(Required option)\n");
  fprintf (stdout, "   -v,--version\tPrint version information and exit.\n");
  fprintf (stdout, "   -h,--help\tPrint this help information and exit.\n");
  fprintf (stdout, "   -n\t\tNumber of top ionograms to calculate.\n");
  fprintf (stdout, "   --minTF\tOverride minimum (1000) number of reads required for mapping.\n");
  fprintf (stdout, "   -i,--individual\tDump indidivual TF reads to file tfsam.txt.\n");
  fprintf (stdout, "   -f,--flows\tNumber of flows to write to file tf_flow_values.txt.\n");
  fprintf (stdout, "   --tfkey\tOverride default TF (ATCG) key.\n");
  fprintf (stdout, "   --libkey\tOverride default Library (TCAG) key.\n");
  fprintf (stdout, "   --flow-order\tOverride default (TACG) flow order\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "usage:\n");
  fprintf (stdout, "   TFMapper --mode [0|1] rawtf.sff\n");
  fprintf (stdout, "\n");
  return (0);
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */

  // set up a few default variables
  char  *sffFileName = NULL;
  char  *expDir = (char *)".";
  std::string TFKEY = "ATCG";
  std::string LIBKEY = "TCAG";

  double  minTFScore = 0.7; // if we can't score this TF with 70% confidence, its unknown
  int minTFFlows = 12; // 8 flows for key, plus at least one more cycle, or we ignore this TF
  int   mode = -1; // mode 0 is TF, 1 is Lib - so we can look in the library sff file for TF's with a library key
  char  *TFoverride = NULL;
  int minTFCount = 1000;
  bool dumpToTFSam = false;
  FILE *tfsam_fp = NULL;
  FILE *tfFlowVals_fp = NULL;
  int numTopIonograms = 10;
  int dumpFlows = 0;
  int c;
  int option_index = 0;
  int numCafieFlows = 120;
  int alternateTFMode = 0;
  static struct option long_options[] =
    {
      {"TF",                    required_argument,  NULL,   0},
      {"experiment-Directory",  required_argument,  NULL, 'e'},
      {"tf-score",        required_argument,  NULL, 's'},
      {"mode",          required_argument,  NULL, 'm'},
      {"version",         no_argument,    NULL, 'v'},
      {"minTF",         required_argument,  NULL, 0},
      {"alternateTFMode",       required_argument,  NULL, 0},
      {"individual",        no_argument,    NULL, 'i'},
      {"flows",         required_argument,  NULL, 'f'},
      {"tfkey",         required_argument,  NULL, 0},
      {"libkey",          required_argument,  NULL, 0},
      {"flow-order",        required_argument,  NULL, 0},
      {"help",          no_argument,  NULL, 'h'},
      {NULL, 0, NULL, 0}
    };

  while ((c = getopt_long(argc, argv, "e:hm:s:f:vin:", long_options, &option_index)) != -1) {
    switch (c) {
    case (0):
      if (long_options[option_index].flag != 0)
        break;

      if (strcmp(long_options[option_index].name, "TF") == 0) {
        TFoverride = optarg;
      }
      // minimum number of TFs to report
      if (strcmp(long_options[option_index].name, "minTF") == 0) {
        minTFCount = atoi(optarg);
      }
      if (strcmp(long_options[option_index].name, "alternateTFMode") == 0) {
        alternateTFMode = atoi(optarg);
      }
      if (strcmp(long_options[option_index].name, "tfkey") == 0) {
        TFKEY = optarg;
      }
      if (strcmp(long_options[option_index].name, "libkey") == 0) {
        LIBKEY = optarg;
      }
      if (strcmp(long_options[option_index].name, "flow-order") == 0) {
        free(flowOrder);
        flowOrder = strdup(optarg);
        numFlowsPerCycle = strlen(flowOrder);
      }
      break;

    case 'h': // show help
      showHelp();
      exit(EXIT_SUCCESS);
      break;

    case 'e': // set experiment directory
      expDir = optarg;
      break;

    case 's': // TF Score
      sscanf(optarg, "%lf", &minTFScore);
      break;

    case 'm': // mode
      mode = atoi(optarg);
      break;

    case 'n': // top N ionograms to calcuate
      numTopIonograms = atoi(optarg);
      break;

    case 'v': //version
      fprintf(stdout, "%s", IonVersion::GetFullVersion("TFMapper").c_str());
      return (0);
      break;

    case 'i': // dump individual TF reads to TF sam file
      dumpToTFSam = true;
      tfsam_fp = fopen("tfsam.txt", "w");
      fprintf(tfsam_fp,
          "name\tstrand\ttStart\tLen\tqLen\tmatch\tpercent.id\tq10Errs\thomErrs\tmmErrs\tindelErrs\tq7Len\tq10Len\tq17Len\tq20Len\tqDNA.a\tmatch.a\ttDNA.a\n");
      break;

    case 'f': // first X flows to file for processing
      dumpFlows = atoi(optarg);
      tfFlowVals_fp = fopen("tf_flow_values.txt", "w");
      break;

    default:
      fprintf(stderr, "What have we here? (%c)\n", c);
      return (-1);
    }
  }

  // Pick up the sff filename
  for (c = optind; c < argc; c++)
  {
    sffFileName = argv[c];
    break; //cause we only expect one non-option argument
  }

  if (!sffFileName){
    showHelp();
    fprintf (stderr, "\nMissing sff_filename\n\n");
    exit (1);
  }

  if (mode == -1) {
    showHelp();
    fprintf (stderr, "\nMissing -m, --mode option.  The --mode option is required.\n\n");
    exit (1);
  }

  char *Key = (char*)TFKEY.c_str();
  if (mode == 1)
    Key = (char*)LIBKEY.c_str();

  if (tfFlowVals_fp) {
    fprintf(tfFlowVals_fp, "row\tcol\ttfName");
    for (int i = 0; i < dumpFlows; i++)
      fprintf(tfFlowVals_fp, "\tflow.%d.%c", i, flowOrder[i % numFlowsPerCycle]);
    fprintf(tfFlowVals_fp, "\n");
  }


  /*
   *  Write a header section
   */
  fprintf (stdout, "# %s - %s-%s (%s)\n#\n", argv[0],
       IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());
  fflush (stdout);

  // open up our TF config file
  TFs tfs(flowOrder);
  if (TFoverride != NULL) { // TF config file was specified on command line
    if (tfs.LoadConfig(TFoverride) == false) {
      fprintf (stderr, "# Error: Specified TF config file, '%s', was not found.\n", TFoverride);
    }
  }
  else {  // Search for TF config file and open it
    char *tfConfigFileName = tfs.GetTFConfigFile();
    if (tfConfigFileName == NULL) {
      fprintf (stderr, "# Error: No TF config file was found.\n");
    }
    else {
      tfs.LoadConfig(tfConfigFileName);
      free (tfConfigFileName);
    }
  }

  TFInfo *tfInfo = tfs.Info();
  int numTFs = tfs.Num();


  /*
  **  Allocate the memory
  */

  MetricGeneratorRawSignal          metricGeneratorRawSignal[numTFs];
  MetricGeneratorQualityHistograms  metricGeneratorQualityHistograms[numTFs];
  MetricGeneratorTopIonograms       metricGeneratorTopIonograms[numTFs];
  for (int tf = 0; tf < numTFs; tf++)
    metricGeneratorTopIonograms[tf].SetNumTopIonograms(numTopIonograms);
  MetricGeneratorHPAccuracy         metricGeneratorHPAccuracy[numTFs];
  MetricGeneratorSNR                metricGeneratorSNR[numTFs];
  MetricGeneratorSNR                metricGeneratorSNRLibrary;


  // open up the Raw Wells file
  RawWells wells(expDir, (char *)"1.wells");
  bool stat = wells.OpenForRead();
  if (stat) {
    fprintf (stdout, "# ERROR: Could not open %s/%s\n", expDir, "1.wells");
    exit (1);
  }

  // open up the TF SFF file
  SFFWrapper tfSFF;
  tfSFF.OpenForRead(expDir, sffFileName);

  while(true) {
    bool sffSuccess = true;
    const sff_t *readInfo = tfSFF.LoadNextEntry(&sffSuccess);
    if (!readInfo)
      break;
    if (!sffSuccess)
      break;

    int x, y;
    if (1 != ion_readname_to_xy(sff_name(readInfo), &x, &y)) {
      fprintf(stderr, "Error parsing read name: '%s'\n", sff_name(readInfo));
      continue;
    }

    if (strncmp(sff_bases(readInfo), Key, 4) != 0)  // Key mismatch? Skip.
      continue;


    metricGeneratorSNRLibrary.AddElement(sff_flowgram(readInfo),Key);

    // compare this TF to the sequence in flow space, calculating the percentage of correct hits
    int seqFlows = readInfo->gheader->flow_length;

    seqFlows = (seqFlows > 40 ? 40 : seqFlows); // don't want to compare more than 40 flows, too many errors
    if (alternateTFMode == 1)
      seqFlows = (int)(numCafieFlows*0.9+0.5); // just fit to 90% of the flows to get good confidence in the basecalls, the last few are lower qual due to CF

    int bestTF = -1;
    double bestScore = minTFScore;

    for(int tf = 0; tf < numTFs; tf++) {
      if (strncmp(tfInfo[tf].key, Key, 4) != 0) // Ignore TF types that do not have the key we currently evaluate
        continue;

      int numTestFlows = (seqFlows > tfInfo[tf].flows ? tfInfo[tf].flows : seqFlows); // don't compare more than this TF's flows
      if (numTestFlows <= minTFFlows)   // Too few flows to consider testing for this TF
        continue;

      int correct = 0;
      for(int iFlow = 0; iFlow < numTestFlows; iFlow++) {
        int tempIonogram = (int)(sff_flowgram(readInfo)[iFlow]/100.0 + 0.5);

        if (alternateTFMode == 1) {
          if (tempIonogram == tfInfo[tf].Ionogram[iFlow])
            correct++;
        } else {
          if ((tempIonogram>0) == (tfInfo[tf].Ionogram[iFlow]>0))
            correct++;
        }
      }
      double score = (double)correct / (double)numTestFlows;
      if (score > bestScore) {
        bestScore = score;
        bestTF = tf;
      }
    }

    if (bestTF == -1) // Failed to classify this TF. Skip.
      continue;

    tfInfo[bestTF].count++;

    // keep track of each TF's signals for per-tf SNR

    metricGeneratorSNR[bestTF].AddElement(sff_flowgram(readInfo),Key);

    int matchCount = 0;
    int misMatchCount = 0;

    int q7readlen = 0;
    int q10readlen = 0;
    int q17readlen = 0;

    int numErrors7 = (int)pow(10.0, 0.7);
    int numErrors10 = (int)pow(10.0, 1.0);
    int numErrors17 = (int)pow(10.0, 1.7);

    int testLen = std::min((int)strlen(sff_bases(readInfo))-4,tfInfo[bestTF].len);
    int errCount = 0;
    int refBases = 0;
    int testBases = 0;

    int flowLenRef = 0; // this var used to stop us from penalizing in the match-mismatch calcs when we go beyond our ref readlength
    int iFlow = 0;
    while (refBases < tfInfo[bestTF].len && testBases < testLen) {

      while (flowOrder[iFlow % numFlowsPerCycle] == tfInfo[bestTF].seq[refBases] && refBases < tfInfo[bestTF].len) {
        refBases++;
        flowLenRef = iFlow;
      }
      while (flowOrder[iFlow % numFlowsPerCycle] == sff_bases(readInfo)[testBases+4] && testBases < testLen)
        testBases++;

      iFlow++;
    }
    refBases = 0;
    testBases = 0;

    iFlow = 0;
    while (refBases < tfInfo[bestTF].len && testBases < testLen) {

      int refHP = 0;
      while (flowOrder[iFlow % numFlowsPerCycle] == tfInfo[bestTF].seq[refBases] && refBases < tfInfo[bestTF].len) {
        refHP++;
        refBases++;
      }
      refHP = std::min(refHP,MAX_HP-1);

      int testHP = 0;
      while (flowOrder[iFlow % numFlowsPerCycle] == sff_bases(readInfo)[testBases+4] && testBases < testLen) {
        testHP++;
        testBases++;
      }

      int numErrors = abs(refHP - testHP);
      errCount += numErrors;

      if (errCount * numErrors7 <= testBases)
        q7readlen = testBases;
      if (errCount * numErrors10 <= testBases)
        q10readlen = testBases;
      if (errCount * numErrors17 <= testBases)
        q17readlen = testBases;

      metricGeneratorHPAccuracy[bestTF].hpCount[refHP]++;
      if (!numErrors)
        metricGeneratorHPAccuracy[bestTF].hpAccuracy[refHP]++;

      if (iFlow <= flowLenRef) {
        if (numErrors == 0)
          matchCount += refHP;
        else
          misMatchCount += numErrors;
      }

      iFlow++;
    }

    int matchMinusMisMatch = std::max(matchCount-misMatchCount,0);

    metricGeneratorQualityHistograms[bestTF].AddElement(matchMinusMisMatch,q10readlen,q17readlen);

    // dump this TF to the TF_sam file if desired
    if (dumpToTFSam) {
      fprintf(tfsam_fp, "r%d|c%d\t%d\t0\t0\t0\t0\t0.0\t0\t0\t0\t0\t%d\t%d\t%d\t0\tGCAT\t||||\tGCAT\n", y, x, bestTF, q7readlen, q10readlen, q17readlen);
    }
    if (tfFlowVals_fp) {
      fprintf(tfFlowVals_fp, "%d\t%d\t%s", y, x, tfInfo[bestTF].name);
      for (int flowIx = 0; flowIx < dumpFlows && flowIx <  seqFlows; flowIx++)
        fprintf(tfFlowVals_fp, "\t%d", sff_flowgram(readInfo)[flowIx]);
      fprintf(tfFlowVals_fp, "\n");
    }

    const WellData *wellData = wells.ReadXY(x, y);

    metricGeneratorTopIonograms[bestTF].AddElement(x,y,(double)q17readlen, wellData->flowValues,
        sff_flowgram(readInfo), (int)readInfo->gheader->flow_length);

    // keep track of the signal (from the Ionogram 'corrected' signals) for each homopolymer
    int validFlows = (readInfo->gheader->flow_length < tfInfo[bestTF].flows ? readInfo->gheader->flow_length : tfInfo[bestTF].flows);
    for(int iFlow = 0; iFlow < validFlows; iFlow++) {

      float currentValueRaw = wellData->flowValues[iFlow];
      float currentValueSff = sff_flowgram(readInfo)[iFlow]*0.01;

      // see what base we called on this flow
      int nmer = (int)(currentValueSff + 0.5);
      if (iFlow < tfInfo[bestTF].flows)
        nmer = tfInfo[bestTF].Ionogram[iFlow];

      metricGeneratorRawSignal[bestTF].AddElement(std::min(nmer,numHPs-1),currentValueRaw,currentValueSff);
    }
  }

  tfSFF.Close();
  wells.Close();
  if (tfsam_fp)
    fclose(tfsam_fp);
  if (tfFlowVals_fp)
    fclose(tfFlowVals_fp);

  // now dump out the stats file
  for(int i = 0; i < numTFs; i++) {
    if (tfInfo[i].count < minTFCount) {
      fprintf (stdout, "# TF Name = %s; not enough beads %d (req'd %d)\n", tfInfo[i].name, tfInfo[i].count, minTFCount);
      continue;
    }

    printf("TF Name = %s\n", tfInfo[i].name);
    printf("TF Seq = %s\n", tfInfo[i].seq);
    printf("Num = %d\n", tfInfo[i].count);
    metricGeneratorQualityHistograms[i].PrintMeanMode();
    metricGeneratorSNR[i].PrintSNR();
    metricGeneratorQualityHistograms[i].PrintMetrics50();
    metricGeneratorRawSignal[i].PrintSNRMetrics();
    metricGeneratorHPAccuracy[i].PrintHPAccuracy();
    metricGeneratorRawSignal[i].PrintOverlapMetrics();
    metricGeneratorQualityHistograms[i].PrintHistogramDumps();
    metricGeneratorTopIonograms[i].PrintTopIonograms(flowOrder,numFlowsPerCycle,tfInfo[i].key,tfInfo[i].seq);
  }

  if (mode == 1) { // library SFF, so dump library SNR
    printf("TF Name = LIB\n");
    metricGeneratorSNRLibrary.PrintSNR();
  }
}



