/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/* -*- Mode: C; tab-width: 4 -*- */
//
// Ion Torrent Systems, Inc.
// Test Fragment Mapper & Stats tool
// (c) 2009
// $Rev: $
// $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h> // for getopt_long
#include <assert.h>
#include <vector>
#include <iostream>

#include "IonVersion.h"
#include "RawWells.h"
#include "SFFWrapper.h"
#include "Mask.h"
#include "fstrcmp.h"
#include "Histogram.h"
#include "file-io/ion_util.h"
#include "fstrcmp.h"
#include "TFs.h"

#include "dbgmem.h"

#include <iostream>
#include <fstream>

// #define EXACT_MATCH
#define MAX_READLEN 100

struct TopIonogram {
    int row;
    int col;
    double  qualMetric;
    float   *raw;
    float   *cor;
};

const int numHPs = 8; // 0 thru 7


Histogram **systemSignal;
Histogram **systemSignal2;
Histogram **q10Hist;
Histogram **q17Hist;
Histogram **matchHist;
Histogram ***rawSignal;
Histogram ***corSignal;
TopIonogram **topIonograms;

char *flowOrder = strdup ("TACG");
int numFlowsPerCycle = strlen (flowOrder);


int numTopIonograms = 10;
int strMax = 1024;


void PrintAlignment(char *refSeq, char *testSeq);

void dumpstats(int numTFs,
               TFInfo *tfInfo,
               int minTFCount,
               Histogram **matchHist,
               Histogram **q10Hist,
               Histogram **q17Hist,
               int hpAccuracy[][numHPs],
               int hpCount[][numHPs],
               int numSignals[][numHPs],
               float *signalOverlapList[][numHPs],
               int numFlowsPerRead,
               int mode,
               double* SNR,
               double SNRTF[][3]
               ) {

    // now dump out the stats file
    for(int i=0;i<numTFs;i++) {
        if (tfInfo[i].count < minTFCount) {
            fprintf (stdout, "# TF Name = %s; not enough beads %d (req'd %d)\n", tfInfo[i].name, tfInfo[i].count,minTFCount);
            continue;
        }

        printf("TF Name = %s\n", tfInfo[i].name);
        printf("TF Seq = %s\n", tfInfo[i].seq);
        printf("Num = %d\nMatch = %.1lf\nAvg Q10 read length = %.1lf\nAvg Q17 read length = %.2lf\nMatchMode = %d\nQ10Mode = %d\nQ17Mode = %d\nSystem SNR = %.2lf\n",
               tfInfo[i].count, matchHist[i]->Mean(), q10Hist[i]->Mean(), q17Hist[i]->Mean(),
               matchHist[i]->ModeBin(), q10Hist[i]->ModeBin(), q17Hist[i]->ModeBin(),
               /*systemSignal[i]->SNR()*/(SNRTF[i][0]+SNRTF[i][1]+SNRTF[i][2])/3.0);

        int numHQ = 0;
        int readLen;
        for(readLen=50;readLen<=MAX_READLEN;readLen++)
            numHQ += matchHist[i]->GetCount(readLen);
        printf("50Match = %d\n", numHQ);

        numHQ = 0;
        for(readLen=50;readLen<=MAX_READLEN;readLen++)
            numHQ += q10Hist[i]->GetCount(readLen);
        printf("50Q10 = %d\n", numHQ);
        // now calc the # reads such that the avg is exactly (or above) 50
        if (numHQ > 0) {
            numHQ = 0;
            double avg = 0.0;
            int total = 0;
            for(readLen=MAX_READLEN;readLen>=0;readLen--) {
                int num = q10Hist[i]->GetCount(readLen);
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
        for(readLen=50;readLen<=MAX_READLEN;readLen++)
            numHQ += q17Hist[i]->GetCount(readLen);
        printf("50Q17 = %d\n", numHQ);
        // now calc the # reads such that the avg is exactly (or above) 50
        if (numHQ > 0) {
            numHQ = 0;
            double avg = 0.0;
            int total = 0;
            for(readLen=MAX_READLEN;readLen>=0;readLen--) {
                int num = q17Hist[i]->GetCount(readLen);
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

        int hp;
        printf("Raw HP SNR = ");
        for(hp=0;hp<numHPs;hp++) {
            printf("%d : %.2lf", hp, rawSignal[i][hp]->SNR() / (hp < 1 ? 1.0 : (double)hp));
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("Corrected HP SNR = ");
        for(hp=0;hp<numHPs;hp++) {
            printf("%d : %.2lf", hp, corSignal[i][hp]->SNR() / (hp < 1 ? 1.0 : (double)hp));
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("Per HP accuracy = ");
        for(hp=0;hp<numHPs;hp++) {
            // printf("%d : %.1lf, ", hp, (hpCount[i][hp] > 0 ? 100.0 * (double)hpAccuracy[i][hp] / (double)hpCount[i][hp] : 0));
            printf("%d : %d/%d", hp, hpAccuracy[i][hp], hpCount[i][hp]);
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("Raw signal overlap = ");
        for(hp=0;hp<numHPs;hp++) {
            if (rawSignal[i][hp]->Count() > 0)
                rawSignal[i][hp]->Dump((char *)"stdout", 3);
            else
                printf("0");
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("Corrected signal overlap = ");
        for(hp=0;hp<numHPs;hp++) {
            /*
              corSignal[i][hp]->Dump(stdout);
            */
            if (rawSignal[i][hp]->Count() > 0)
                for(int samp=0;samp<numSignals[i][hp];samp++) {
                    printf("%.3lf ",  signalOverlapList[i][hp][samp]);
                }
            else
                printf("0");
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("TransferPlot = ");
        for(hp=0;hp<numHPs;hp++) {
            printf("%d %.4lf %.4lf", hp, corSignal[i][hp]->Mean(), corSignal[i][hp]->StdDev());
            if (hp < (numHPs-1))
                printf(", ");
        }
        printf("\n");

        printf("Match-Mismatch = ");
        matchHist[i]->Dump(stdout);
        printf("\n");

        printf("Q10 = ");
        q10Hist[i]->Dump(stdout);
        printf("\n");

        printf("Q17 = ");
        q17Hist[i]->Dump(stdout);
        printf("\n");

        // dump the top N ionograms

        char topSeq[strMax];
        char tfSeq[strMax];
        int topBases = 0;
        for(int topI=0;topI<numTopIonograms;topI++) {
            if (topIonograms[i][topI].raw != NULL) {
                printf("Top %d = %d, %d, ", topI+1, topIonograms[i][topI].row, topIonograms[i][topI].col);
                for(int flow=0;flow<numFlowsPerRead;flow++) {
                    printf("%.2lf ", topIonograms[i][topI].raw[flow]);
                }
                printf(", ");
                topBases = 0;
                for(int flow=0;flow<numFlowsPerRead;flow++) {
                    printf("%.2lf ", topIonograms[i][topI].cor[flow]);
                    int mer = (int)(topIonograms[i][topI].cor[flow]+0.5f);
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
                strcpy(tfSeq, tfInfo[i].key);
                strcat(tfSeq, tfInfo[i].seq);
                PrintAlignment(tfSeq, topSeq);
                printf("\n");
            }
        }

        // and dump the per-HP histograms
    }

    if (mode == 1) { // library SFF, so dump library SNR
        printf("TF Name = LIB\n");
        printf("System SNR = %.2lf\n", (SNR[0] + SNR[1] + SNR[2])/3.0);
    }

}


int GetReadLen(char *refSeq, int refLen, char *testSeq, int q, bool best, bool details, int *perFlowErrors, int *_matchCount, int *_misMatchCount, int *hpAccuracy, int *hpCount)
{
    // printf("Calculating Q%d score for %s aligned to %s\n", q, testSeq, refSeq);
    int numErrors = (int)(pow(10.0, q/10.0));
    double errRatio = 1.0/numErrors;
    int errCount = 0;
    int testLen = strlen(testSeq);
    if (testLen > refLen)
        testLen = refLen;

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
    int readLen = 0;

    if (details) {
        printf("Flow-space ref:\n");
        for(i=0;i<flowLen;i++)
            printf("%d", refArray[i]);
        printf("\n");
        printf("Flow-space tf:\n");
        for(i=0;i<flowLen;i++)
            printf("%d", testArray[i]);
        printf("\n");

        // generate the alignment strings
        char refBuf[256];
        char testBuf[256];
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
                }
            }
        }
        refBuf[k] = 0;
        testBuf[k] = 0;
        printf("%s\n%s\n", refBuf, testBuf);
    }

    // printf("Using %d flows to test with\n", flowLen);

    int matchCount = 0;
    int misMatchCount = 0;
    int bestReadlen = 0;
    int hp;
    for(i=0;i<flowLen;i++) {
        hp = refArray[i];
        if (hp >= numHPs)
            hp = numHPs-1;
        if (hpCount)
            hpCount[hp]++;
        if (refArray[i] == 0 && testArray[i] == 0) { // match but no read length boost
            if (hpAccuracy)
                hpAccuracy[0]++;
        } else if (refArray[i] == testArray[i]) { // match & read length boost
            readLen += testArray[i];
            double ratio = (double)errCount/(double)readLen;
            if (ratio <= errRatio)
                bestReadlen = readLen;
            if (i <= flowLenRef)
                matchCount += refArray[i];
            if (hpAccuracy)
                hpAccuracy[hp]++;
        } else { // miss-match
            readLen += testArray[i];
            int numErrors = abs(refArray[i] - testArray[i]);
            errCount += numErrors;
            double ratio = 1.0;
            if (readLen > 0)
                ratio = (double)errCount/(double)readLen;
            if (best) {
                if (ratio <= errRatio)
                    bestReadlen = readLen;
            } else {
                if (ratio > errRatio) {
                    bestReadlen = readLen;
                    break;
                }
            }
            perFlowErrors[i] += numErrors;
            if (i <= flowLenRef)
                misMatchCount += numErrors;
        }
    }

    // printf("Got readlength: %d\n", bestReadlen);
    if (_matchCount)
        *_matchCount = matchCount;
    if (_misMatchCount)
        *_misMatchCount = misMatchCount;

    return bestReadlen;
}

void PrintAlignment(char *refSeq, char *testSeq)
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
/*
  int GetNuc(char base)
  {
  if (base == 'T')
  return 0;
  if (base == 'A')
  return 1;
  if (base == 'C')
  return 2;
  if (base == 'G')
  return 3;
  return 0;
  }
*/
//
// BuildNucIndex - generates a lookup array that returns what nuc index corresponds to what flow - used by GetNuc
//
int *flowOrderIndex = NULL;
void BuildNucIndex()
{
    flowOrderIndex = new int[numFlowsPerCycle];

    int i;
    for(i=0;i<numFlowsPerCycle;i++) {
        if (flowOrder[i] == 'T')
            flowOrderIndex[i] = 0;
        if (flowOrder[i] == 'A')
            flowOrderIndex[i] = 1;
        if (flowOrder[i] == 'C')
            flowOrderIndex[i] = 2;
        if (flowOrder[i] == 'G')
            flowOrderIndex[i] = 3;
    }
}

//
// GetNuc - For a given flow & flow order, returns the nuc as an index 0 thru 3
//

inline int GetNuc(int flow)
{
    // nuc's by definition are 0=T, 1=A, 2=C, 3=G - this is the default flow order on the PGM, so we wanted to make sure it makes sense for our standard
    return flowOrderIndex[flow%numFlowsPerCycle];
}

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
    char     *sffFileName = NULL;
    std::vector<std::string> folders;
    char     *TFKEY = strdup ("ATCG");
    char     *LIBKEY = strdup ("TCAG");
    char     *Key[2] = {TFKEY, LIBKEY};
    double   minTFScore = 0.7; // if we can't score this TF with 70% confidence, its unknown
    int      minTFFlows = 12; // 8 flows for key, plus at least one more cycle, or we ignore this TF
    int      mode = -1; // mode 0 is TF, 1 is Lib - so we can look in the library sff file for TF's with a library key
    MaskType beadmask[2] = {MaskTF, MaskLib};
    char     *TFoverride = NULL;
    bool     verbose = false;
    int      minTFCount = 1000;
    bool     dumpToTFSam = false;
    FILE     *tfsam_fp = NULL;
    FILE     *tfFlowVals_fp = NULL;
    int      dumpFlows = 0;
    int      c;
    int      option_index = 0;
    int      numCafieFlows = 120;
    int      alternateTFMode = 0;
    static struct option long_options[] =
        {
            {"TF",                      required_argument,  NULL,   0},
            {"experiment-Directory",    required_argument,  NULL,   'e'},
            {"tf-score",                required_argument,  NULL,   's'},
            {"mode",                    required_argument,  NULL,   'm'},
            {"version",                 no_argument,        NULL,   'v'},
            {"minTF",                   required_argument,  NULL,   0},
            {"alternateTFMode",         required_argument,  NULL,   0},
            {"individual",              no_argument,        NULL,   'i'},
            {"flows",                   required_argument,  NULL,   'f'},
            {"tfkey",                   required_argument,  NULL,   0},
            {"libkey",                  required_argument,  NULL,   0},
            {"flow-order",              required_argument,  NULL,   0},
            {"help",                    no_argument,        NULL,   'h'},
            {NULL, 0, NULL, 0}
        };

    std::ofstream logfile;
    logfile.open ("TFMapperIterative.log", std::ios::out | std::ios::app );
    logfile << "TFMapperIterative started" << std::endl;

    while ((c = getopt_long (argc, argv, "e:hm:s:f:vin:", long_options, &option_index)) != -1)
        {
            switch (c)
                {
                case (0):
                    if (long_options[option_index].flag != 0)
                        break;

                    if (strcmp (long_options[option_index].name, "TF") == 0) {
                        TFoverride = optarg;
                    }
                    // minimum number of TFs to report
                    if (strcmp (long_options[option_index].name, "minTF") == 0) {
                        minTFCount = atoi (optarg);
                    }
                    if (strcmp (long_options[option_index].name, "alternateTFMode") == 0) {
                        alternateTFMode = atoi (optarg);
                    }
                    if (strcmp (long_options[option_index].name, "tfkey") == 0) {
                        free (TFKEY);
                        TFKEY = strdup (optarg);
                    }
                    if (strcmp (long_options[option_index].name, "libkey") == 0) {
                        free (LIBKEY);
                        LIBKEY = strdup (optarg);
                    }
                    if (strcmp (long_options[option_index].name, "flow-order") == 0) {
                        free (flowOrder);
                        flowOrder = strdup (optarg);
                        numFlowsPerCycle = strlen (flowOrder);
                    }
                    break;

                case 'h': // show help
                    showHelp ();
                    exit (EXIT_SUCCESS);
                    break;

                case 'e': // set experiment directory
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

                case 'v':   //version
                    fprintf (stdout, "%s", IonVersion::GetFullVersion("TFMapper").c_str());
                    return (0);
                    break;

                case 'i': // dump individual TF reads to TF sam file
                    dumpToTFSam = true;
                    tfsam_fp = fopen("tfsam.txt", "w");
                    fprintf(tfsam_fp, "name\tstrand\ttStart\tLen\tqLen\tmatch\tpercent.id\tq10Errs\thomErrs\tmmErrs\tindelErrs\tq7Len\tq10Len\tq17Len\tq20Len\tqDNA.a\tmatch.a\ttDNA.a\n");
                    break;

                case 'f': // first X flows to file for processing
                    dumpFlows = atoi(optarg);
                    tfFlowVals_fp = fopen("tf_flow_values.txt", "w");
                    break;

                default:
                    fprintf (stderr, "What have we here? (%c)\n", c);
                    return (-1);
                }
        }

    // Pick up the sff filename
    for (c = optind; c < argc; c++)
        {
            sffFileName = argv[c];
            break; //cause we only expect one non-option argument
        }

    // Pick up the dirnames
    for (c = optind+1; c < argc; c++)
        {
            folders.push_back(argv[c]);
            logfile << "block to process: " << argv[c] << std::endl;
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


    /*
     *  Write a header section
     */
    fprintf (stdout,
             "# %s - %s-%s (%s)\n#\n",
             argv[0],
             IonVersion::GetVersion().c_str(),
             IonVersion::GetRelease().c_str(),
             IonVersion::GetSvnRev().c_str());
    fflush (stdout);

    //    int nb_folders = 1;
    //    for (int i=0; i<nb_folders; i++) {


    int numFlowsPerRead = -1;

    // open one TF SFF file to get the flow order
    SFFWrapper tfSFF;
    logfile << "Open " << folders[0].c_str() << "/" << sffFileName << std::endl;
    tfSFF.OpenForRead(folders[0].c_str(), sffFileName);
    numFlowsPerRead = tfSFF.LoadEntry(0)->gheader->flow_length;
    logfile << "reads: " << numFlowsPerRead << std::endl;
    if (tfFlowVals_fp) {
        int flowLength = strlen(flowOrder);
        fprintf(tfFlowVals_fp, "row\tcol\ttfName");
        for (int i = 0; i < dumpFlows; i++) {
            fprintf(tfFlowVals_fp, "\tflow.%d.%c", i, flowOrder[i % flowLength]);
        }
        fprintf(tfFlowVals_fp, "\n");
    }
    tfSFF.Close();

    // build a nuc index array now that we know the flow order
    BuildNucIndex();

    // open up our TF config file (DefaultTFs.conf)
    TFs *tfs = new TFs(flowOrder);
#if 0
    if (TFoverride != NULL) {
        if (tfs->LoadConfig(TFoverride) == true) {
            //            fprintf (stderr, "# Loading TFs from '%s'\n", TFoverride);
        }
        else {
            //            fprintf (stderr, "# File not found: %s\nUsing default TFs\n", TFoverride);
        }
    }
    else {
        char *tfConfigFileName = tfs->GetTFConfigFile();
        if (tfConfigFileName == NULL) {
            fprintf (stderr, "# Error: No TF config file was found.\n");
        }
        else {
            tfs->LoadConfig(tfConfigFileName);
            //          fprintf (stderr, "# Loading TFs from '%s'\n", tfConfigFileName);
            free (tfConfigFileName);
        }
    }
#else
    if (TFoverride != NULL) {   // TF config file was specified on command line
        if (tfs->LoadConfig(TFoverride) == false) {
            fprintf (stderr, "# Error: Specified TF config file, '%s', was not found.\n", TFoverride);
        }
    }
    else {  // Search for TF config file and open it
        char *tfConfigFileName = tfs->GetTFConfigFile();
        if (tfConfigFileName == NULL) {
            fprintf (stderr, "# Error: No TF config file was found.\n");
        }
        else {
            tfs->LoadConfig(tfConfigFileName);
            free (tfConfigFileName);
        }
    }
#endif

    TFInfo *tfInfo = tfs->Info();
    int numTFs = tfs->Num();


    /*
    **  Allocate the memory
    */
    systemSignal = new Histogram *[numTFs];
    systemSignal2 = new Histogram *[numTFs];
    q10Hist = new Histogram *[numTFs];
    q17Hist = new Histogram *[numTFs];
    matchHist = new Histogram *[numTFs];
    rawSignal = new Histogram **[numTFs];
    for (int i=0;i<numTFs;i++)
        rawSignal[i] = new Histogram *[numHPs];
    corSignal = new Histogram **[numTFs];
    for (int i=0;i<numTFs;i++)
        corSignal[i] = new Histogram *[numHPs];
    int hpAccuracy[numTFs][numHPs];
    int hpCount[numTFs][numHPs];
    float *signalOverlapList[numTFs][numHPs];
    int numSignals[numTFs][numHPs];
    topIonograms = new TopIonogram *[numTFs];

    memset(numSignals, 0, sizeof(numSignals));
    int numSamples = 1000; // pick this many TF's per TF as representitive samples for overlap plotting

    memset(hpAccuracy, 0, sizeof(hpAccuracy));
    memset(hpCount, 0, sizeof(hpCount));

    // var init
    for(int i=0;i<numTFs;i++) {
        if (verbose) {
            int flow;
            printf("TF: %s - ", tfInfo[i].name);
            for(flow=0;flow<tfInfo[i].flows;flow++) {
                printf("%d ", tfInfo[i].Ionogram[flow]);
            }
            printf("\n");
        }
        systemSignal[i] = new Histogram(1001, -2.0, 4.0);
        systemSignal2[i] = new Histogram(1001, -2.0, 4.0);
        q10Hist[i] = new Histogram(MAX_READLEN+1, 0, MAX_READLEN);
        q17Hist[i] = new Histogram(MAX_READLEN+1, 0, MAX_READLEN);
        matchHist[i] = new Histogram(MAX_READLEN+1, 0, MAX_READLEN);

        for(int hp=0;hp<numHPs;hp++) {
            rawSignal[i][hp] = new Histogram(1001, -2.0, numHPs+1.0);
            corSignal[i][hp] = new Histogram(1001, -2.0, numHPs+1.0);

            signalOverlapList[i][hp] = new float[numSamples];
        }

        // allocate space for our top N list
        topIonograms[i] = (TopIonogram *)malloc(sizeof(TopIonogram) * numTopIonograms);
        memset(topIonograms[i], 0, sizeof(TopIonogram) * numTopIonograms);
    }
    memset(hpAccuracy, 0, sizeof(hpAccuracy));

    // convert each TF into array of flow-space hits
    // MGD note - this is now in the TFs.h code, so can pull from there if needed
    /*
      char keyAndSeq[512];
      int tf;
      for(tf=0;tf<numTFs;tf++) {
      sprintf(keyAndSeq, "%s%s", tfInfo[tf].key, tfInfo[tf].seq);
      int keyAndSeqLen = strlen(keyAndSeq);
      int tfBases = 0;
      tfInfo[tf].flows = 0;
      while (tfInfo[tf].flows < 800 && tfBases < keyAndSeqLen) {
      tfInfo[tf].Ionogram[tfInfo[tf].flows] = 0;
      while (flowOrder[tfInfo[tf].flows%4] == tfInfo[tf].seq[tfBases] && tfBases < keyAndSeqLen) {
      tfInfo[tf].Ionogram[tfInfo[tf].flows]++;
      tfBases++;
      }
      tfInfo[tf].flows++;
      }
      }
    */

    // a few vars used below
    double bestScore = 0.0;
    int bestTF;
    int x, y;
    int perFlowErrors[800] = {0};
    int matchCount, misMatchCount;
    double perNuc0mer[4];
    double perNuc1mer[4];
    memset(perNuc0mer, 0, sizeof(perNuc0mer));
    memset(perNuc1mer, 0, sizeof(perNuc1mer));
    int perNucCount = 0;

    // calculate the nuc to flow lookup - this is based on the key and flow order
    int nuc2flow[4];
    for (int n=0;n<4;n++)
        nuc2flow[n] = 666;

    int base = 0;
    for(int flow=0;flow<2*numFlowsPerCycle;flow++) {
        if (Key[mode][base] == flowOrder[flow%numFlowsPerCycle]) {
            nuc2flow[GetNuc(flow)] = flow;
            base++;
        }
    }

    for (int n = 0; n < 4;n++) {
        //fprintf (stderr, "nuc2flow[%d] = %d\n", n, nuc2flow[n]);
        assert (nuc2flow[n] != 666);
    }

    double perNuc0merTF[numTFs][3];
    double perNuc1merTF[numTFs][3];
    memset(perNuc0merTF, 0, sizeof(perNuc0merTF));
    memset(perNuc1merTF, 0, sizeof(perNuc1merTF));

    const sff_t *readInfo;
    int tempIonogram[2000];
    std::vector<unsigned char*> tfMapVector;
    unsigned char *tfMap = NULL;

    logfile << "number blocks: " << folders.size() << std::endl;

    // calculate offsets
    std::vector<int> xoffsets;
    std::vector<int> yoffsets;

    for (unsigned int f=0;f<folders.size();f++) {

        // extract chip offset
        char* size = GetProcessParam (folders[f].c_str(), "Block" );
        xoffsets.push_back(atoi(strtok (size,",")));
        yoffsets.push_back(atoi(strtok(NULL, ",")));
        logfile << "chip offset: " << xoffsets[f] << "," << yoffsets[f] << std::endl;
    }

    for (unsigned int i=0; i<folders.size(); i++) {

        tfSFF.OpenForRead(folders[i].c_str(), sffFileName);
        logfile << "Open " << folders[i].c_str() << "/" << sffFileName << std::endl;

        // open up the mask file so we know what wells contain TF's
        Mask mask(1, 1);
        char maskFileName[MAX_PATH_LENGTH];
        sprintf(maskFileName, "%s/bfmask.bin", folders[i].c_str());
        int rflag = mask.SetMask(maskFileName);
        if (rflag != 0) {
            fprintf (stdout, "# ERROR: Could not open %s\n", maskFileName);
            exit (1);
        }


        // open up the Raw Wells file
        RawWells wells(folders[i].c_str(), (char *)"1.wells", mask.H(), mask.W() );
        bool stat = wells.OpenForRead();
        if (stat) {
            fprintf (stdout, "# ERROR: Could not open %s/%s\n", folders[i].c_str(), "1.wells");
            exit (1);
        }

        // loop through the sff file (since its the only file that is not organized as a grid, so efficiency at play here)
        int numReads = tfSFF.GetReadCount();
        tfMap = new unsigned char[numReads];
        memset(tfMap, 0, numReads);

        // store tfMap for calculating SNR's
        tfMapVector.push_back(tfMap);


        // printf("Processing %d reads\n", numReads);

        for(int read=0;read<numReads;read++) {
            readInfo = tfSFF.LoadEntry(read);
            if(1 != ion_readname_to_xy(sff_name(readInfo), &x, &y)) {
                fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(readInfo));
                continue;
            }
            //            std::cerr << sff_name(readInfo) << " x: " << x << " y: " << y << std::endl;
            if (mask.Match(x, y, beadmask[mode])) {
                // get the TF sequence
                // nothing to do here,

                // see if it keypassed
                if (strncmp(sff_bases(readInfo), Key[mode], 4) == 0) {
                    // keep track of the per-nuc 1-mer signals in the key for accurate SNR calcs
                    // MGD warning - not TANGO safe!
                    int nuc;
                    int flow1mer, flow0mer;
                    for(nuc=0;nuc<3;nuc++) {
                        flow1mer = nuc2flow[nuc];
                        flow0mer = (flow1mer > 3 ? flow1mer - 4 : flow1mer + 4);
                        perNuc1mer[nuc] += sff_flowgram(readInfo)[flow1mer]/100.0;
                        perNuc0mer[nuc] += sff_flowgram(readInfo)[flow0mer]/100.0;
                    }
                    perNucCount++;

                    // compare this TF to the sequence in flow space, calculating the percentage of correct hits
                    int seqFlows = readInfo->gheader->flow_length;
                    for(int flow=0;flow<seqFlows;flow++) {
                        tempIonogram[flow] = (int)(sff_flowgram(readInfo)[flow]/100.0 + 0.5);
                    }
                    seqFlows = (seqFlows > 40 ? 40 : seqFlows); // don't want to compare more than 40 flows, too many errors
                    if (alternateTFMode == 1)
                        seqFlows = (int)(numCafieFlows*0.9+0.5); // just fit to 90% of the flows to get good confidence in the basecalls, the last few are lower qual due to CF
                    bestTF = -1;
                    bestScore = minTFScore;
                    for(int tf=0;tf<numTFs;tf++) {
                        if (strncmp(tfInfo[tf].key, Key[mode], 4) == 0) {
                            int numTestFlows = (seqFlows > tfInfo[tf].flows ? tfInfo[tf].flows : seqFlows); // don't compare more than this TF's flows
                            if (numTestFlows > minTFFlows) {
                                int correct = 0;
                                for(int flow=0;flow<numTestFlows;flow++) {
#ifdef EXACT_MATCH
                                    if (tempIonogram[flow] == tfInfo[tf].Ionogram[flow])
                                        correct++;
#else
                                    if (alternateTFMode == 1) {
                                        if (tempIonogram[flow] == tfInfo[tf].Ionogram[flow])
                                            correct++;
                                    } else {
                                        if ((tempIonogram[flow] == tfInfo[tf].Ionogram[flow] && tempIonogram[flow] == 0) ||
                                            (tempIonogram[flow] > 0 && tfInfo[tf].Ionogram[flow] > 0))
                                            correct++;
                                    }
#endif
                                }
                                double score = (double)correct/(double)numTestFlows;
                                if (score > bestScore) {
                                    bestScore = score;
                                    bestTF = tf;
                                }
                            }
                        }
                    }

                    if (bestTF > -1) {
                        // keep track of each TF's signals for per-tf SNR
                        for(nuc=0;nuc<3;nuc++) {
                            flow1mer = nuc2flow[nuc];
                            flow0mer = (flow1mer > 3 ? flow1mer - 4 : flow1mer + 4);
                            perNuc1merTF[bestTF][nuc] += sff_flowgram(readInfo)[flow1mer]/100.0;
                            perNuc0merTF[bestTF][nuc] += sff_flowgram(readInfo)[flow0mer]/100.0;
                        }
                        tfMap[read] = bestTF;

                        tfInfo[bestTF].count++;
                        matchCount = 0;
                        misMatchCount = 0;
                        // hey, we found the TF, so calc metrics for it
                        int q7readlen = GetReadLen(tfInfo[bestTF].seq, tfInfo[bestTF].len, &sff_bases(readInfo)[4], 7, true, false, perFlowErrors, NULL, NULL, NULL, NULL);
                        int q10readlen = GetReadLen(tfInfo[bestTF].seq, tfInfo[bestTF].len, &sff_bases(readInfo)[4], 10, true, false, perFlowErrors, &matchCount, &misMatchCount, hpAccuracy[bestTF], hpCount[bestTF]);
                        if (verbose)
                            printf("Found TF %s with score: %.2lf\nQ10: %d for %s\n", tfInfo[bestTF].name, bestScore, q10readlen, sff_bases(readInfo));

                        q10Hist[bestTF]->Add(q10readlen);
                        int matchMinusMisMatch = matchCount-misMatchCount;
                        if (matchMinusMisMatch < 0)
                            matchMinusMisMatch = 0;
                        matchHist[bestTF]->Add(matchMinusMisMatch);

                        int q17readlen = GetReadLen(tfInfo[bestTF].seq, tfInfo[bestTF].len, &sff_bases(readInfo)[4], 17, true, false, perFlowErrors, NULL, NULL, NULL, NULL);
                        q17Hist[bestTF]->Add(q17readlen);
                        const WellData *wellData = wells.ReadXY(x, y);

                        // dump this TF to the TF_sam file if desired
                        if (dumpToTFSam) {
                            fprintf(tfsam_fp, "r%d|c%d\t%d\t0\t0\t0\t0\t0.0\t0\t0\t0\t0\t%d\t%d\t%d\t0\tGCAT\t||||\tGCAT\n", y+yoffsets[i], x+xoffsets[i], bestTF, q7readlen, q10readlen, q17readlen);
                            //                      fprintf(tfsam_fp, "r%d|c%d\t%d\t0\t0\t0\t0\t0.0\t0\t0\t0\t0\t0\t%d\t%d\t0\tGCAT\t||||\tGCAT\n", y, x, bestTF, q10readlen, q17readlen);
                        }
                        if (tfFlowVals_fp) {
                            fprintf(tfFlowVals_fp, "%d\t%d\t%s", y+yoffsets[i], x+xoffsets[i], tfInfo[bestTF].name);
                            for (int flowIx = 0; flowIx < dumpFlows && flowIx <  seqFlows; flowIx++) {
                                fprintf(tfFlowVals_fp, "\t%d", sff_flowgram(readInfo)[flowIx]);
                            }
                            fprintf(tfFlowVals_fp, "\n");
                        }
                        // see if this is one of our top Ionorgams

                        for(int topI=0;topI<numTopIonograms;topI++) {
                            if (q17readlen > topIonograms[bestTF][topI].qualMetric) { // MGD - if the qualMetric were initialized to some value, would act as a nice filter to allow only reads above 50 for example, but for now set to zero
                                topIonograms[bestTF][topI].qualMetric = q17readlen;
                                topIonograms[bestTF][topI].row = y+yoffsets[i];
                                topIonograms[bestTF][topI].col = x+xoffsets[i];
                                // allocate space if necessary
                                if (topIonograms[bestTF][topI].raw == NULL) {
                                    topIonograms[bestTF][topI].raw = (float *)malloc(sizeof(float) * readInfo->gheader->flow_length);
                                    topIonograms[bestTF][topI].cor = (float *)malloc(sizeof(float) * readInfo->gheader->flow_length);
                                }
                                for(int flow=0;flow<readInfo->gheader->flow_length;flow++) {
                                    topIonograms[bestTF][topI].raw[flow] = wellData->flowValues[flow];
                                    topIonograms[bestTF][topI].cor[flow] = sff_flowgram(readInfo)[flow]*0.01f;
                                }
                                break;
                            }
                        }

                        // keep track of the system signal (and later extract SNR), and other HP signals
                        double signal = 0.0;
                        int sigCount = 0;
                        int nmer = 0;
                        for(int flow=0;flow<8;flow++) {
                            if (flow < tfInfo[bestTF].flows)
                                nmer = tfInfo[bestTF].Ionogram[flow];
                            else
                                nmer = (int)(sff_flowgram(readInfo)[flow]*0.01 + 0.5);
                            if (nmer == 1) {
                                signal += wellData->flowValues[flow];
                                sigCount++;
                            }
                        }
                        if (sigCount > 0) {
                            signal /= (double)sigCount;
                            systemSignal[bestTF]->Add(signal);
                        }

                        // alternate method for calculating system SNR is to look at the 1st positive incorporation after the key
                        for(int flow=8;flow<tfInfo[bestTF].flows;flow++) {
                            if (tfInfo[bestTF].Ionogram[flow] > 0) {
                                systemSignal2[bestTF]->Add(wellData->flowValues[flow]/tfInfo[bestTF].Ionogram[flow], true);
                                break;
                            }
                        }

                        // keep track of the signal (from the Ionogram 'corrected' signals) for each homopolymer
                        int validFlows = (readInfo->gheader->flow_length < tfInfo[bestTF].flows ? readInfo->gheader->flow_length : tfInfo[bestTF].flows);
                        for(int flow=0;flow<validFlows;flow++) {
                            // see what base we called on this flow
                            if (flow < tfInfo[bestTF].flows)
                                nmer = tfInfo[bestTF].Ionogram[flow];
                            else
                                nmer = (int)(sff_flowgram(readInfo)[flow]*0.01 + 0.5);
                            if (nmer >= numHPs)
                                nmer = numHPs-1;
                            rawSignal[bestTF][nmer]->Add(wellData->flowValues[flow]);

                            float val = sff_flowgram(readInfo)[flow]*0.01;
                            corSignal[bestTF][nmer]->Add(val);

                            int loc;
                            if (numSignals[bestTF][nmer] < numSamples) {
                                loc = numSignals[bestTF][nmer];
                                numSignals[bestTF][nmer]++;
                            } else // once we get numSamples, just start replacing at random, should provide a representitive sampled population, but clearly favoring the later samples, if I knew in advance how many I had, I could select fairly uniformly throughout
                                loc = rand() % numSamples;
                            signalOverlapList[bestTF][nmer][loc] = val;
                        }
                    } else {
                        // not a TF we know about, keep track in our top 10 list
                    }
                }
            }
        }

        wells.Close();


        tfSFF.Close();
        //        delete [] tfMap;
    }
    if (tfsam_fp)
        fclose(tfsam_fp);

    if (tfFlowVals_fp) {
        fclose(tfFlowVals_fp);
    }

    // calculate SNR's & SNRTF's
    // relies on previous calculated tfMap and tfInfo values


    double SNR[3];
    double SNRTF[numTFs][3];


    double mean0mer[3];
    double mean1mer[3];
    double perNuc0merStdev[3];
    double perNuc1merStdev[3];
    double delta;
    memset(perNuc0merStdev, 0, sizeof(perNuc0merStdev));
    memset(perNuc1merStdev, 0, sizeof(perNuc1merStdev));

    double mean0merTF[numTFs][3];
    double mean1merTF[numTFs][3];
    double perNuc0merStdevTF[numTFs][3];
    double perNuc1merStdevTF[numTFs][3];
    memset(perNuc0merStdevTF, 0, sizeof(perNuc0merStdevTF));
    memset(perNuc1merStdevTF, 0, sizeof(perNuc1merStdevTF));

    // calculate the means

    for(int nuc=0;nuc<3;nuc++) { // only care about the first 3, G maybe 2-mer etc
        mean1mer[nuc] = perNuc1mer[nuc] / perNucCount;
        mean0mer[nuc] = perNuc0mer[nuc] / perNucCount;
    }
    for(int tf=0;tf<numTFs;tf++) {
        for(int nuc=0;nuc<3;nuc++) { // only care about the first 3, G maybe 2-mer etc
            if (tfInfo[tf].count > 0) {
                mean0merTF[tf][nuc] = perNuc0merTF[tf][nuc] / tfInfo[tf].count;
                mean1merTF[tf][nuc] = perNuc1merTF[tf][nuc] / tfInfo[tf].count;
            } else {
                mean0merTF[tf][nuc] = 0.0;
                mean1merTF[tf][nuc] = 0.0;
            }
        }
    }

    for (unsigned int i=0; i<folders.size(); i++) {

        tfSFF.OpenForRead(folders[i].c_str(), sffFileName);
        logfile << "Open " << folders[i].c_str() << "/" << sffFileName << " for SNR calculation." << std::endl;
        int numReads = tfSFF.GetReadCount();
        tfMap = tfMapVector[i];

        // open up the mask file so we know what wells contain TF's
        Mask mask(1, 1);
        char maskFileName[MAX_PATH_LENGTH];
        sprintf(maskFileName, "%s/bfmask.bin", folders[i].c_str());
        int rflag = mask.SetMask(maskFileName);
        if (rflag != 0) {
            fprintf (stdout, "# ERROR: Could not open %s\n", maskFileName);
            exit (1);
        }

        // calculate the standard deviations - yes we need to loop through the whole read file again! (now that we know the mean)
        for(int read=0;read<numReads;read++) {
            readInfo = tfSFF.LoadEntry(read);
            if(1 != ion_readname_to_xy(sff_name(readInfo), &x, &y)) {
                fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(readInfo));
                continue;
            }

            if (mask.Match(x, y, beadmask[mode])) {
                // see if it keypassed
                if (strncmp(sff_bases(readInfo), Key[mode], 4) == 0) {

                    int flow1mer, flow0mer;
                    for(int nuc=0;nuc<3;nuc++) {
                        flow1mer = nuc2flow[nuc];
                        flow0mer = (flow1mer > 3 ? flow1mer - 4 : flow1mer + 4);
                        delta = mean1mer[nuc] - sff_flowgram(readInfo)[flow1mer]/100.0;
                        perNuc1merStdev[nuc] += delta * delta;
                        delta = mean0mer[nuc] - sff_flowgram(readInfo)[flow0mer]/100.0;
                        perNuc0merStdev[nuc] += delta * delta;
                    }

                    for(int nuc=0;nuc<3;nuc++) {
                        flow1mer = nuc2flow[nuc];
                        flow0mer = (flow1mer > 3 ? flow1mer - 4 : flow1mer + 4);
                        delta = mean1merTF[tfMap[read]][nuc] - sff_flowgram(readInfo)[flow1mer]/100.0;
                        perNuc1merStdevTF[tfMap[read]][nuc] += delta * delta;
                        delta = mean0merTF[tfMap[read]][nuc] - sff_flowgram(readInfo)[flow0mer]/100.0;
                        perNuc0merStdevTF[tfMap[read]][nuc] += delta * delta;
                    }
                }
            }
        }
        tfSFF.Close();
    }

    // now calc the SNR's
    for(int nuc=0;nuc<3;nuc++) {
        if (perNucCount > 0) {
            perNuc0merStdev[nuc] /= perNucCount;
            perNuc0merStdev[nuc] = sqrt(perNuc0merStdev[nuc]);
            perNuc1merStdev[nuc] /= perNucCount;
            perNuc1merStdev[nuc] = sqrt(perNuc1merStdev[nuc]);

            double avgStdev = (perNuc0merStdev[nuc] + perNuc1merStdev[nuc])*0.5;
            if (avgStdev > 0.0)
                SNR[nuc] = (mean1mer[nuc] - mean0mer[nuc]) / avgStdev;
            else
                SNR[nuc] = 0.0;
        } else
            SNR[nuc] = 0.0;
    }

    // calc SNRTF's
    for(int tf=0;tf<numTFs;tf++) {
        for(int nuc=0;nuc<3;nuc++) {
            if (tfInfo[tf].count > 0) {
                perNuc0merStdevTF[tf][nuc] /= tfInfo[tf].count;
                perNuc1merStdevTF[tf][nuc] /= tfInfo[tf].count;
                perNuc0merStdevTF[tf][nuc] = sqrt(perNuc0merStdevTF[tf][nuc]);
                perNuc1merStdevTF[tf][nuc] = sqrt(perNuc1merStdevTF[tf][nuc]);

                double avgStdev = (perNuc0merStdevTF[tf][nuc] + perNuc1merStdevTF[tf][nuc])*0.5;
                if (avgStdev > 0.0)
                    SNRTF[tf][nuc] = (mean1merTF[tf][nuc] - mean0merTF[tf][nuc]) / avgStdev;
                else
                    SNRTF[tf][nuc] = 0.0;
            } else {
                SNRTF[tf][nuc] = 0.0;
            }
        }
    }


    dumpstats(numTFs, tfInfo, minTFCount, matchHist, q10Hist, q17Hist, hpAccuracy, hpCount, numSignals, signalOverlapList, numFlowsPerRead, mode, SNR, SNRTF);


    // cleanup

    // clean tfMapVector
    // delete [] tfMap;

    for(int i=0;i<numTFs;i++) {
        delete systemSignal[i];
        delete systemSignal2[i];
        delete q10Hist[i];
        delete q17Hist[i];
        delete matchHist[i];
        for(int hp=0;hp<numHPs;hp++) {
            delete rawSignal[i][hp];
            delete corSignal[i][hp];
            delete [] signalOverlapList[i][hp];
        }
    }

    for(int i=0;i<numTFs;i++) {
        for(int topI=0;topI<numTopIonograms;topI++) {
            if (topIonograms[i][topI].raw != NULL) {
                free(topIonograms[i][topI].raw);
                free(topIonograms[i][topI].cor);
            }
        }
        free(topIonograms[i]);
    }

    delete tfs;

    free (TFKEY);
    free (LIBKEY);

}
