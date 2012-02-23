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
#include <iostream>
#include <fstream>

#include "IonVersion.h"
#include "RawWells.h"
#include "SFFWrapper.h"
#include "fstrcmp.h"
#include "Histogram.h"
#include "file-io/ion_util.h"
#include "fstrcmp.h"
#include "TFs.h"
#include "DPTreephaser.h"

#include "dbgmem.h"

#include "json/json.h"

using namespace std;


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

  void PrintSNRMetrics(Json::Value& currentTFJson) {
    printf("Raw HP SNR = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %.2lf", hp, rawSignal[hp]->SNR() / (double)std::max(hp,1));
    }
    printf("\n");

    for(int hp = 0; hp < maxHP; hp++)
      currentTFJson["Raw HP SNR"][hp] = rawSignal[hp]->SNR() / (double)std::max(hp,1);


    printf("Corrected HP SNR = ");
    for(int hp = 0; hp < maxHP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %.2lf", hp, corSignal[hp]->SNR() / (double)std::max(hp,1));
    }
    printf("\n");

    for(int hp = 0; hp < maxHP; hp++)
      currentTFJson["Corrected HP SNR"][hp] = corSignal[hp]->SNR() / (double)std::max(hp,1);
  }

  void PrintOverlapMetrics(Json::Value& currentTFJson) {
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

  void PrintMeanMode(Json::Value& currentTFJson) {
    printf("Match = %.1lf\n", matchHist->Mean());
    printf("Avg Q10 read length = %.1lf\n", q10Hist->Mean());
    printf("Avg Q17 read length = %.2lf\n", q17Hist->Mean());
    printf("MatchMode = %d\n", matchHist->ModeBin());
    printf("Q10Mode = %d\n", q10Hist->ModeBin());
    printf("Q17Mode = %d\n", q17Hist->ModeBin());

    //currentTFJson["Match"] = matchHist->Mean();
    //currentTFJson["Avg Q10 read length"] = q10Hist->Mean();
    //currentTFJson["Avg Q17 read length"] = q17Hist->Mean();
    //currentTFJson["MatchMode"] = matchHist->ModeBin();
    //currentTFJson["Q10Mode"] = q10Hist->ModeBin();
    //currentTFJson["Q17Mode"] = q17Hist->ModeBin();
  }

  void PrintMetrics50(Json::Value& currentTFJson) {
    int numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += matchHist->GetCount(readLen);
    printf("50Match = %d\n", numHQ);
    //currentTFJson["50Match"] = numHQ;

    numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += q10Hist->GetCount(readLen);
    printf("50Q10 = %d\n", numHQ);
    //currentTFJson["50Q10"] = numHQ;
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
    //currentTFJson["50Q10A"] = numHQ;

    numHQ = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++)
      numHQ += q17Hist->GetCount(readLen);
    printf("50Q17 = %d\n", numHQ);
    //currentTFJson["50Q17"] = numHQ;
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
    //currentTFJson["50Q17A"] = numHQ;
  }

  void PrintHistogramDumps(Json::Value& currentTFJson) {
    printf("Match-Mismatch = ");
    matchHist->Dump(stdout);
    printf("\n");
    //for (int idx = 0; idx <= MAX_READLEN; idx++)
    //  currentTFJson["Match-Mismatch"][idx] = matchHist->GetCount(idx);

    printf("Q10 = ");
    q10Hist->Dump(stdout);
    printf("\n");
    for (int idx = 0; idx <= MAX_READLEN; idx++)
      currentTFJson["Q10"][idx] = q10Hist->GetCount(idx);

    printf("Q17 = ");
    q17Hist->Dump(stdout);
    printf("\n");
    for (int idx = 0; idx <= MAX_READLEN; idx++)
      currentTFJson["Q17"][idx] = q17Hist->GetCount(idx);
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


  void PrintTopIonograms(char *flowOrder, int numFlowsPerCycle, char *currentKey, char *currentSeq, Json::Value& currentTFJson) {

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
      strcpy(tfSeq, currentKey);
      strcat(tfSeq, currentSeq);

      string alignedReadSeq;
      string alignedRefSeq;
      string alignedBars;

      PrintAlignment(tfSeq, topSeq, flowOrder, numFlowsPerCycle, alignedReadSeq, alignedRefSeq, alignedBars);
      printf(",%s,%s,%s\n", alignedRefSeq.c_str(), alignedBars.c_str(), alignedReadSeq.c_str());


      currentTFJson["Top Reads"][topI]["row"] = row[topI];
      currentTFJson["Top Reads"][topI]["col"] = col[topI];
      currentTFJson["Top Reads"][topI]["metric"] = qualMetric[topI];

      for(int flow = 0; flow < (int)raw[topI].size(); flow++) {
        currentTFJson["Top Reads"][topI]["Raw Signal"][flow] = raw[topI][flow];
        currentTFJson["Top Reads"][topI]["Corrected Signal"][flow] = 0.01f * cor[topI][flow];
      }

      currentTFJson["Top Reads"][topI]["Read Seq"] = alignedReadSeq;
      currentTFJson["Top Reads"][topI]["Ref Seq"] = alignedRefSeq;
      currentTFJson["Top Reads"][topI]["Read Bars"] = alignedBars;

    }
  }

  void PrintAlignment(char *refSeq, char *testSeq, char *flowOrder, int numFlowsPerCycle,
        string& alignedReadSeq, string& alignedRefSeq, string& alignedBars)
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

    alignedReadSeq = testBuf;
    alignedRefSeq = refBuf;
    alignedBars = bars;

    //printf("%s,%s,%s", refBuf, bars, testBuf);
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

  void PrintHPAccuracy(Json::Value& currentTFJson) {
    printf("Per HP accuracy = ");
    for(int hp = 0; hp < MAX_HP; hp++) {
      if (hp)
        printf(", ");
      printf("%d : %d/%d", hp, hpAccuracy[hp], hpCount[hp]);
      currentTFJson["Per HP accuracy NUM"][hp] = hpAccuracy[hp];
      currentTFJson["Per HP accuracy DEN"][hp] = hpCount[hp];
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
  void PrintSNR(Json::Value& currentTFJson) {
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
    currentTFJson["System SNR"] = SNR;
  }

private:
  int count;
  double zeromerFirstMoment[4];
  double zeromerSecondMoment[4];
  double onemerFirstMoment[4];
  double onemerSecondMoment[4];
};



class CafieMetricsGenerator {
public:
  CafieMetricsGenerator (const char* OutputDirectory, int _numTFs, int numFlows) {

    numTFs = _numTFs;
    numFlowsTFClassify = numFlows;
    avgTFSignal.assign(numTFs, std::vector<double>(numFlowsTFClassify, 0.0));
    avgTFCorrected.assign(numTFs, std::vector<double>(numFlowsTFClassify, 0.0));

    //cafieMetricsFilename = (std::string(OutputDirectory) + "/cafieMetrics.txt").c_str();
    //CF = 0;
    //IE = 0;
    //DR = 0;
    //blockCount = 0;
    numRows = -1;
    numCols = -1;
    numRegionRows = -1;
    numRegionCols = -1;
  }

  ~CafieMetricsGenerator() {}

  void AddBlock(const char *AnalysisDirectory, int _numRows, int _numCols) {

    std::ifstream in((std::string(AnalysisDirectory) + "/BaseCaller.json").c_str(), std::ifstream::in);
    numRows = -1;
    numCols = -1;
    numRegionRows = -1;
    numRegionCols = -1;
    if (!in.good())
      return;
    Json::Value BaseCallerJson;
    in >> BaseCallerJson;

    //CF += BaseCallerJson["Phasing"].get("CF",0).asFloat();
    //IE += BaseCallerJson["Phasing"].get("IE",0).asFloat();
    //DR += BaseCallerJson["Phasing"].get("DR",0).asFloat();
    //blockCount++;

    numRows = _numRows;
    numCols = _numCols;

    numRegionRows = BaseCallerJson["Phasing"].get("RegionRows",1).asInt();
    numRegionCols = BaseCallerJson["Phasing"].get("RegionCols",1).asInt();

    CFbyRegion.resize(numRegionRows*numRegionCols);
    IEbyRegion.resize(numRegionRows*numRegionCols);
    for (int iRegion = 0; iRegion < numRegionRows*numRegionCols; iRegion++) {
      CFbyRegion[iRegion] = BaseCallerJson["Phasing"]["CFbyRegion"].get(iRegion,0.0).asFloat();
      IEbyRegion[iRegion] = BaseCallerJson["Phasing"]["IEbyRegion"].get(iRegion,0.0).asFloat();
    }

  }

  void AddElement(int tf, TFInfo *tfInfo, float *rawValues, uint16_t *corValues, int len, int x, int y, char *flowOrder) {

    if (numRows <= 0)
      return;

    BasecallerRead well;
    well.SetDataAndKeyNormalize(rawValues, len, tfInfo[tf].Ionogram, 7);
    for (int iFlow = 0; (iFlow < len); iFlow++)
      well.solution[iFlow] = (char)((corValues[iFlow]+50)/100);

    int cafieYinc = ceil(numRows / (double)numRegionRows);
    int cafieXinc = ceil(numCols / (double)numRegionCols);
    int iRegion = (y / cafieYinc) + (x / cafieXinc) * numRegionRows;

    DPTreephaser dpTreephaser(flowOrder, len, 8);
    dpTreephaser.SetModelParameters(CFbyRegion[iRegion], IEbyRegion[iRegion], 0);
    dpTreephaser.Simulate3(well, len);
    well.FitBaselineVector((len+49)/50,50);
    well.FitNormalizerVector((len+49)/50,50);

    for (int iFlow = 0; (iFlow < len) && (iFlow < numFlowsTFClassify); iFlow++) {
      avgTFSignal[tf][iFlow] += well.normalizedMeasurements[iFlow];
      avgTFCorrected[tf][iFlow] += corValues[iFlow] / 100.0;
    }
  }

/*
  void GenerateCafieMetrics(TFInfo *tfInfo, int len, char *flowOrder) {

    if (blockCount > 0) {
      CF /= blockCount;
      IE /= blockCount;
      DR /= blockCount;
    }

    //  CAFIE Metrics file
    FILE *cmfp = NULL;
    fopen_s(&cmfp, cafieMetricsFilename.c_str(), "w");
    if (!cmfp)
      return;

    for (int tf = 0; tf < numTFs; tf++) {
      if (tfInfo[tf].count <= 1000)
        continue;

      // show raw Ionogram
      fprintf(cmfp, "TF = %s\n", tfInfo[tf].name);
      fprintf(cmfp, "Avg Ionogram = ");
      for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
        fprintf(cmfp, "%.2lf ", avgTFSignal[tf][iFlow] / tfInfo[tf].count);
      fprintf(cmfp, "\n");

      // show a bunch 'O stats
      fprintf(cmfp, "Estimated TF = %s\n", tfInfo[tf].name);
//      fprintf(cmfp, "CF = %.5f\n", CF);
//      fprintf(cmfp, "IE = %.5f\n", IE);
//      fprintf(cmfp, "Signal Droop = %.5f\n", DR);
//      fprintf(cmfp, "Error = %.4f\n", 0.0f);
      fprintf(cmfp, "Count = %d\n", tfInfo[tf].count);

      fprintf(cmfp, "Corrected Avg Ionogram = ");
      for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
        fprintf(cmfp, "%.2lf ", avgTFCorrected[tf][iFlow] / tfInfo[tf].count);
      fprintf(cmfp, "\n");
    }

    // calculate & print to cafieMetrics file the "system cf/ie/dr"
    // its just an avg of the regional estimates
    fprintf(cmfp, "Estimated System CF = %.5f\n", 100.0 * CF);
    fprintf(cmfp, "Estimated System IE = %.5f\n", 100.0 * IE);
    fprintf(cmfp, "Estimated System Signal Droop = %.5f\n", 100.0 * DR);

    fclose(cmfp);

  }*/

  void PrintIonograms(Json::Value& currentTFJson, TFInfo *tfInfo, int tf) {
    printf("Avg Ionogram = ");
    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++) {
      printf("%.2lf ", avgTFSignal[tf][iFlow] / tfInfo[tf].count);
      currentTFJson["Avg Ionogram NUM"][iFlow] = avgTFSignal[tf][iFlow];
      currentTFJson["Avg Ionogram DEN"][iFlow] = tfInfo[tf].count;
    }
    printf("\n");

    printf("Corrected Avg Ionogram = ");
    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++) {
      printf("%.2lf ", avgTFCorrected[tf][iFlow] / tfInfo[tf].count);
      currentTFJson["Corrected Avg Ionogram NUM"][iFlow] = avgTFCorrected[tf][iFlow];
      currentTFJson["Corrected Avg Ionogram DEN"][iFlow] = tfInfo[tf].count;
    }
    printf("\n");
  }


private:
  //bool active;
  //float CF, IE, DR;
  //int blockCount;
  //std::string cafieMetricsFilename;
  int numTFs;

  int numRows, numCols, numRegionRows, numRegionCols;

  int numFlowsTFClassify;
  std::vector<std::vector<double> > avgTFSignal;
  std::vector<std::vector<double> > avgTFCorrected;

  std::vector<float> CFbyRegion;
  std::vector<float> IEbyRegion;

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
  fprintf (stdout, "   -s,--tf-score\tSet confidence level for scoring.  (0.70 is default)\n");
  fprintf (stdout, "   -v,--version\tPrint version information and exit.\n");
  fprintf (stdout, "   -h,--help\tPrint this help information and exit.\n");
  fprintf (stdout, "   -n\t\tNumber of top ionograms to calculate.\n");
  fprintf (stdout, "   --minTF\tOverride minimum (1000) number of reads required for mapping.\n");
  fprintf (stdout, "   -i,--individual\tDump indidivual TF reads to file tfsam.txt.\n");
  fprintf (stdout, "   --wells-dir\tSpecify directory containing the 1.wells file\n");
  fprintf (stdout, "   --sff-dir\tSpecify directory containing *sff files\n");
  fprintf (stdout, "   --output-dir\tSpecify directory containing experiment files\n");
  fprintf (stdout, "   --tfkey\tOverride default TF (ATCG) key.\n");
  fprintf (stdout, "   --logfile\tSpecify a logfile for debug output.\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "usage:\n");
  fprintf (stdout, "   TFMapper [options] rawtf.sff\n");
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
  char* sffFileName = NULL;
  char* wellsdir = (char *)".";
  char* sffdir = (char *)".";
  char* outputdir = (char *)".";
  std::string TFKEY = "ATCG";

  string      outputJsonFilename = "";
  bool        outputJsonRequested = false;
  Json::Value outputJson;

  double  minTFScore = 0.87; // if we can't score this TF with 70% confidence, its unknown
  int minTFFlows = 12; // 8 flows for key, plus at least one more cycle, or we ignore this TF
  int   mode = -1; // mode 0 is TF, 1 is Lib - so we can look in the library sff file for TF's with a library key
  char  *TFoverride = NULL;
  char* logfilepath = NULL;
  bool logging = false;
  std::ofstream logfile;
  int minTFCount = 1000;
  bool dumpToTFSam = false;
  FILE *tfsam_fp = NULL;
  int numTopIonograms = 10;
  int c;
  int option_index = 0;
  int numCafieFlows = 120;
  int alternateTFMode = 0;
  static struct option long_options[] =
    {
      {"TF",                    required_argument,  NULL,   0},
      {"tf-score",        required_argument,  NULL, 's'},
      {"mode",          required_argument,  NULL, 'm'},
      {"version",         no_argument,    NULL, 'v'},
      {"minTF",         required_argument,  NULL, 0},
      {"alternateTFMode",       required_argument,  NULL, 0},
      {"individual",        no_argument,    NULL, 'i'},
      {"tfkey",         required_argument,  NULL, 0},
      {"libkey",          required_argument,  NULL, 0},
      {"wells-dir",          required_argument,  NULL, 0},
      {"sff-dir",          required_argument,  NULL, 0},
      {"output-dir",          required_argument,  NULL, 0},
      {"logfile",          required_argument,  NULL, 0},
      {"flow-order",        required_argument,  NULL, 0},
      {"output-json",        required_argument,  NULL, 0},
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
      if (strcmp(long_options[option_index].name, "wells-dir") == 0) {
        wellsdir = optarg;
      }
      if (strcmp(long_options[option_index].name, "sff-dir") == 0) {
        sffdir = optarg;
      }
      if (strcmp(long_options[option_index].name, "output-dir") == 0) {
        outputdir = optarg;
      }
      if (strcmp(long_options[option_index].name, "output-json") == 0) {
        outputJsonRequested = true;
        outputJsonFilename = optarg;
      }
      if (strcmp(long_options[option_index].name, "logfile") == 0) {
        logfilepath = optarg;
        logging = true;
      }
      break;

    case 'h': // show help
      showHelp();
      exit(EXIT_SUCCESS);
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

    default:
      fprintf(stderr, "What have we here? (%c)\n", c);
      return (-1);
    }
  }

  if (logging) {
      logfile.open (logfilepath, std::ios::out | std::ios::app );
      logfile << "TFMapperIterative started" << std::endl;
  }

  // Pick up the sff filename
  if (optind >= argc) {
    showHelp();
    fprintf (stderr, "\nMissing sff_filename\n\n");
    exit (1);
  }

  sffFileName = argv[optind++];


  // Pick up the dirnames
  std::vector<std::string> block_folders; // Folder list for per-block operation. If empty, work in single-block mode
  std::vector<std::string> wells_folders;
  std::vector<std::string> sff_files;
  while (optind < argc) {
    block_folders.push_back(argv[optind]);
    wells_folders.push_back(std::string(wellsdir) + "/" + std::string(argv[optind]));
    sff_files.push_back(std::string(sffdir) + "/" + std::string(argv[optind]) + "/" + sffFileName);
    if (logging) {
        logfile << "block to process: " << argv[optind] << std::endl;
    }
    optind++;
  }
  if (block_folders.empty()) {
    block_folders.push_back("./");
    wells_folders.push_back(std::string(wellsdir) + "/" + std::string("./"));
    sff_files.push_back(std::string(sffdir) + "/" + std::string("./") + "/" + sffFileName);
  }





  // open up the TF SFF file
  SFFWrapper tfSFFEarly;
  if (logging) {
      logfile << "Open " << sff_files[0].c_str() << std::endl;
  }
  tfSFFEarly.OpenForRead(sff_files[0].c_str());
  int seqFlows = tfSFFEarly.GetHeader()->flow_length;
  if (logging ) {
      logfile << "reads: " << seqFlows << std::endl;
  }
  free(flowOrder);
  flowOrder = strdup(tfSFFEarly.GetHeader()->flow->s);
  numFlowsPerCycle = strlen(flowOrder);

  char *Key = (char*)TFKEY.c_str();

  tfSFFEarly.Close();


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

  CafieMetricsGenerator cafieMetricsGenerator(outputdir, numTFs, std::min(80, seqFlows));


  TFTracker *tracker = new TFTracker(outputdir);

  if (logging) {
      logfile << "number blocks: " << block_folders.size() << std::endl;
  }
  // calculate offsets
  std::vector<int> xoffsets;
  std::vector<int> yoffsets;
  std::vector<int> nRow;
  std::vector<int> nCol;



  for (unsigned int f = 0;f < block_folders.size(); f++) {
    // extract chip offset
    xoffsets.push_back(0);
    yoffsets.push_back(0);
    nRow.push_back(0);
    nCol.push_back(0);
    int x = -1, y = -1;
    sscanf(GetProcessParam (wells_folders[f].c_str(), "Block"), "%d,%d,%d,%d", &x, &y, &(nCol.back()), &(nRow.back()));
    if ((x >= 0) && (y >= 0)) {
      xoffsets.back() = x;
      yoffsets.back() = y;
    }
    if (logging) {
        logfile << "chip offset: " << xoffsets[f] << "," << yoffsets[f] << std::endl;
    }
  }

  for (unsigned int i = 0; i < sff_files.size(); i++) {

    vector<int32_t>  xSubset;
    vector<int32_t>  ySubset;
    {
      SFFWrapper tfSFF;
      tfSFF.OpenForRead(sff_files[i].c_str());

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

        xSubset.push_back(x);
        ySubset.push_back(y);
      }
      tfSFF.Close();
    }


    SFFWrapper tfSFF;
    tfSFF.OpenForRead(sff_files[i].c_str());
    if (logging) {
        logfile << "Open " << sff_files[i].c_str() << std::endl;
    }
    // open up the Raw Wells file
    RawWells wells(wells_folders[i].c_str(), (char *)"1.wells", nRow[i], nCol[i]);
    wells.SetSubsetToLoad(xSubset, ySubset);
    if (wells.OpenForRead()) {
      fprintf (stdout, "# ERROR: Could not open %s/%s\n", wells_folders[i].c_str(), "1.wells");
      exit (1);
    }

    cafieMetricsGenerator.AddBlock(wells_folders[i].c_str(), (int)wells.NumRows(), (int)wells.NumCols());


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

      seqFlows = std::min(seqFlows, 40); // don't want to compare more than 40 flows, too many errors
      if (alternateTFMode == 1)
        seqFlows = (int)(numCafieFlows*0.9+0.5); // just fit to 90% of the flows to get good confidence in the basecalls, the last few are lower qual due to CF


      int bestTF = -1;
      double bestScore = minTFScore;

      for(int tf = 0; tf < numTFs; tf++) {
        if (strncmp(tfInfo[tf].key, Key, 4) != 0) // Ignore TF types that do not have the key we currently evaluate
          continue;

        int numTestFlows = std::min(seqFlows,tfInfo[tf].flows); // don't compare more than this TF's flows
        if (numTestFlows <= minTFFlows)   // Too few flows to consider testing for this TF
          continue;

        int correct = 0;
        for(int iFlow = 0; iFlow < numTestFlows; iFlow++) {
          int tempIonogram = (int)rint(sff_flowgram(readInfo)[iFlow]/100.0);

          if (alternateTFMode == 1) {
            if (tempIonogram == tfInfo[tf].Ionogram[iFlow])
              correct++;
          } else {
            if ((tempIonogram > 0) == (tfInfo[tf].Ionogram[iFlow] > 0))
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
      if (tracker)
        tracker->Add(y+yoffsets[i], x+xoffsets[i], tfInfo[bestTF].name);

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
      if (dumpToTFSam)
        fprintf(tfsam_fp, "r%d|c%d\t%d\t0\t0\t0\t0\t0.0\t0\t0\t0\t0\t%d\t%d\t%d\t0\tGCAT\t||||\tGCAT\n",
            y+yoffsets[i], x+xoffsets[i], bestTF, q7readlen, q10readlen, q17readlen);

      const WellData *wellData = wells.ReadXY(x, y);

      metricGeneratorTopIonograms[bestTF].AddElement(x+xoffsets[i],y+yoffsets[i],(double)q17readlen, wellData->flowValues,
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

      cafieMetricsGenerator.AddElement(bestTF,tfInfo,wellData->flowValues,
          sff_flowgram(readInfo), validFlows, x, y, flowOrder);

    }

    tfSFF.Close();
    wells.Close();

  }

  if (tfsam_fp)
    fclose(tfsam_fp);

  if (tracker) {
    tracker->Close();
    delete tracker;
  }

  // now dump out the stats file
  for(int i = 0; i < numTFs; i++) {
    if (tfInfo[i].count < minTFCount) {
      fprintf (stdout, "# TF Name = %s; not enough beads %d (req'd %d)\n", tfInfo[i].name, tfInfo[i].count, minTFCount);
      continue;
    }


    printf("TF Name = %s\n", tfInfo[i].name);
    printf("TF Seq = %s\n", tfInfo[i].seq);
    printf("Num = %d\n", tfInfo[i].count);

    Json::Value currentTFJson;
    currentTFJson["TF Name"] = tfInfo[i].name;
    currentTFJson["TF Seq"] = tfInfo[i].seq;
    currentTFJson["Num"] = tfInfo[i].count;

    metricGeneratorQualityHistograms[i].PrintMeanMode(currentTFJson);
    metricGeneratorSNR[i].PrintSNR(currentTFJson);
    metricGeneratorQualityHistograms[i].PrintMetrics50(currentTFJson);
    metricGeneratorRawSignal[i].PrintSNRMetrics(currentTFJson);
    metricGeneratorHPAccuracy[i].PrintHPAccuracy(currentTFJson);
    metricGeneratorRawSignal[i].PrintOverlapMetrics(currentTFJson);
    metricGeneratorQualityHistograms[i].PrintHistogramDumps(currentTFJson);
    metricGeneratorTopIonograms[i].PrintTopIonograms(flowOrder,numFlowsPerCycle,tfInfo[i].key,tfInfo[i].seq,currentTFJson);

    cafieMetricsGenerator.PrintIonograms(currentTFJson, tfInfo, i);


    outputJson[tfInfo[i].name] = currentTFJson;

  }

  //if (generateCafieMetrics)
  //  cafieMetricsGenerator.GenerateCafieMetrics(tfInfo, seqFlows, flowOrder);


  if (outputJsonRequested) {
    ofstream out(outputJsonFilename.c_str(), ios::out);
    if (out.good())
      out << outputJson.toStyledString();
  }

}



