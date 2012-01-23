/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cstdio>
#include <cstring>

#include <string>
#include <vector>

#include <pthread.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "Mask.h"
#include "RawWells.h"
#include "PhaseSolve.h"
#include "DPTreephaser.h"

#include "SamUtils/BAMReader.h"
#include "SamUtils/BAMUtils.h"
#include "SamUtils/types/BAMRead.h"

#include "json/json.h"

using namespace std;

void usage(char *pluginName)
{
  printf("BruteForcePhase - Assess 100q17 performance of a run for different cf/ie/dr assumptions.\n\n");
  printf("Usage:\n\n");
  printf("  %s analysis_dir plugin_output_dir reference_name\n\n", pluginName);
  exit(0);
}

class PhaseParameterGrid {
public:

  vector<float> uniqueCF;
  vector<float> uniqueIE;
  vector<float> uniqueDR;

  vector<float> CF;
  vector<float> IE;
  vector<float> DR;

  void clear()
  {
    uniqueCF.clear();
    uniqueIE.clear();
    uniqueDR.clear();
    CF.clear();
    IE.clear();
    DR.clear();
  }

  int size()
  {
    return CF.size();
  }

  int gridToIndex(int idxCF, int idxIE, int idxDR)
  {
    return idxCF + idxIE * uniqueCF.size() + idxDR * uniqueCF.size() * uniqueIE.size();
  }

  void buildGridFromUnique()
  {
    CF.resize(uniqueCF.size() * uniqueIE.size() * uniqueDR.size());
    IE.resize(uniqueCF.size() * uniqueIE.size() * uniqueDR.size());
    DR.resize(uniqueCF.size() * uniqueIE.size() * uniqueDR.size());
    for (unsigned int idxCF = 0; idxCF < uniqueCF.size(); idxCF++) {
      for (unsigned int idxIE = 0; idxIE < uniqueIE.size(); idxIE++) {
        for (unsigned int idxDR = 0; idxDR < uniqueDR.size(); idxDR++) {
          CF[gridToIndex(idxCF, idxIE, idxDR)] = uniqueCF[idxCF];
          IE[gridToIndex(idxCF, idxIE, idxDR)] = uniqueIE[idxIE];
          DR[gridToIndex(idxCF, idxIE, idxDR)] = uniqueDR[idxDR];
        }
      }
    }
  }
};

class BruteForcePhase {
public:

  BruteForcePhase(const char *_pathAnalysis, const char *_pathOutput, const char *_referenceName);

  void openReport();
  void closeReport();

  void addRegionToReport(int regionX, int regionY);

private:

  void parseCafieRegions();

  void initializeGrids(int regionX, int regionY);
  void sampleReadsFromRegion(int regionX, int regionY);
  void decodeToFastq(string &filenameFastq);
  void determineConsensusReference(string &filenameBam);
  void analyzeResultsGrid();

  void reportRegionResults(int regionX, int regionY);

  string pathAnalysis;
  string pathOutput;
  string referenceName;

  FILE *f_report;
  FILE *f_block;

  int numRegionsX;
  int numRegionsY;

  vector<float> estimatedCF;
  vector<float> estimatedIE;
  vector<float> estimatedDR;

  int numFlows;
  int keyFlow[7];
  int keyFlowLen;

  PhaseParameterGrid alignmentGrid;
  PhaseParameterGrid resultsGrid;

  deque<vector<float> > regionMeasurements;
  deque<vector<int> > regionReferences;
  string flowOrder;

  int numReadsInRegion;
  int numReadsSampled;
  vector<int> numReads100q17;
  vector<bool> isRead100q17;
  int numReads100q17Anywhere;

  friend void *BruteForceWorker(void *);
  void worker_Treephaser();
  pthread_mutex_t *mutex;
  int nextParameters;

  Json::Value  phaseJson;

};

void *BruteForceWorker(void *);

BruteForcePhase::BruteForcePhase(const char *_pathAnalysis, const char *_pathOutput, const char *_referenceName)
{
  pathAnalysis = _pathAnalysis;
  pathOutput = _pathOutput;
  referenceName = _referenceName;

  parseCafieRegions();
}

void BruteForcePhase::parseCafieRegions()
{
  string filenameCafieRegions = pathAnalysis + "/cafieRegions.txt";

  estimatedCF.clear();
  estimatedIE.clear();
  estimatedDR.clear();
  numRegionsY = 0;
  numRegionsX = 0;

  // Open cafieRegions.txt and skip the header

  FILE *f_cafie = fopen(filenameCafieRegions.c_str(), "r");

  if (!f_cafie) {
    fprintf(stderr, "BruteForcePhase: %s cannot be opened (%s)\n", filenameCafieRegions.c_str(), strerror(errno));
    exit(EXIT_FAILURE);
  }

  char line[2048];

  while (fgets(line, 2048, f_cafie)) {
    if (0 == strncmp(line, "Region CF = ", 12))
      break;
  }

  // Read CF and in the process establish numRegionsX and numRegionsY

  while (true) {
    char *c = line;
    if (0 == strncmp(line, "Region CF = ", 12))
      c += 12;

    while (true) {
      char *c_next;
      float val = strtof(c, &c_next);
      if (c == c_next)
        break;
      c = c_next;
      estimatedCF.push_back(val);
      if (numRegionsY == 0)
        numRegionsX++;
    }
    numRegionsY++;

    if (!fgets(line, 2048, f_cafie))
      break;
    if (0 == strncmp(line, "Region IE = ", 12))
      break;
  }

  // Read IE

  for (int iRegionY = 0; iRegionY < numRegionsY; iRegionY++) {
    char *c = line;
    if (0 == strncmp(line, "Region IE = ", 12))
      c += 12;

    for (int iRegionX = 0; iRegionX < numRegionsX; iRegionX++)
      estimatedIE.push_back(strtof(c, &c));

    if (!fgets(line, 2048, f_cafie))
      break;
  }

  // Read DR

  for (int iRegionY = 0; iRegionY < numRegionsY; iRegionY++) {
    char *c = line;
    if (0 == strncmp(line, "Region DR = ", 12))
      c += 12;

    for (int iRegionX = 0; iRegionX < numRegionsX; iRegionX++)
      estimatedDR.push_back(strtof(c, &c));

    if (!fgets(line, 2048, f_cafie))
      break;
  }

  fclose(f_cafie);

  if ((numRegionsX <= 0) || (numRegionsY <= 0)) {
    fprintf(stderr, "BruteForcePhase: parsing of cafieRegions.txt unsuccessful (%d x %d, %d %d %d)\n", numRegionsX, numRegionsY, (int) estimatedCF.size(),
        (int) estimatedIE.size(), (int) estimatedDR.size());
    exit(EXIT_FAILURE);
  }

  phaseJson["PhaseEstimates"]["numRegionsX"] = numRegionsX;
  phaseJson["PhaseEstimates"]["numRegionsY"] = numRegionsY;
  for (int idx = 0; idx < (int)estimatedCF.size(); idx++) {
    phaseJson["PhaseEstimates"]["estimatedCF"][idx] = estimatedCF[idx];
    phaseJson["PhaseEstimates"]["estimatedIE"][idx] = estimatedIE[idx];
    phaseJson["PhaseEstimates"]["estimatedDR"][idx] = estimatedDR[idx];
  }

  string filename = pathOutput + "/PhaseReport.json";
  ofstream out(filename.c_str(), ios::out);
  if (out.good())
    out << phaseJson.toStyledString();
}

void BruteForcePhase::initializeGrids(int regionX, int regionY)
{

  int idxRegion = regionX + regionY * numRegionsX;

  // Initialize "consensus alignment" grid

  alignmentGrid.clear();

  alignmentGrid.uniqueCF.push_back(0.003);
  alignmentGrid.uniqueCF.push_back(0.006);
  alignmentGrid.uniqueCF.push_back(0.009);

  alignmentGrid.uniqueIE.push_back(0.003);
  alignmentGrid.uniqueIE.push_back(0.006);
  alignmentGrid.uniqueIE.push_back(0.009);
  alignmentGrid.uniqueIE.push_back(0.012);

  alignmentGrid.uniqueDR.push_back(0.0);

  alignmentGrid.buildGridFromUnique();

  alignmentGrid.CF.push_back(estimatedCF[idxRegion]);
  alignmentGrid.IE.push_back(estimatedIE[idxRegion]);
  alignmentGrid.DR.push_back(estimatedDR[idxRegion]);

  // Initialize results generation grid

  resultsGrid.clear();


  float cf_mult = 0.002;
  int cf_offset = std::max((int) rint(estimatedCF[idxRegion] / cf_mult) - 4, 0);

  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 0));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 1));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 2));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 3));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 4));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 5));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 6));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 7));
  resultsGrid.uniqueCF.push_back(cf_mult * (cf_offset + 8));

  float ie_mult = 0.002;
  int ie_offset = std::max((int) rint(estimatedIE[idxRegion] / ie_mult) - 4, 0);

  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 0));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 1));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 2));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 3));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 4));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 5));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 6));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 7));
  resultsGrid.uniqueIE.push_back(ie_mult * (ie_offset + 8));

  resultsGrid.uniqueDR.push_back(0);

/*
  int cf_offset = 3; //std::max((int) rint(estimatedCF[idxRegion] / 0.001) - 4, 0);

  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 0));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 1));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 2));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 3));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 4));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 5));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 6));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 7));
  resultsGrid.uniqueCF.push_back(0.001 * (cf_offset + 8));

  int ie_offset = 3; //std::max((int) rint(estimatedIE[idxRegion] / 0.001) - 4, 0);

  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 0));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 1));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 2));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 3));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 4));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 5));
  resultsGrid.uniqueIE.push_back(0.001 * (ie_offset + 6));

  int dr_offset = 1; //std::max((int) rint(estimatedDR[idxRegion] / 0.0005) - 2, 0);

  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 0));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 1));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 2));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 3));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 4));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 5));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 6));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 7));
  resultsGrid.uniqueDR.push_back(0.003 * (dr_offset + 8));
//  resultsGrid.uniqueDR.push_back(1);
*/

  resultsGrid.buildGridFromUnique();

  resultsGrid.CF.push_back(estimatedCF[idxRegion]);
  resultsGrid.IE.push_back(estimatedIE[idxRegion]);
  resultsGrid.DR.push_back(estimatedDR[idxRegion]);

}

// preprocessRegion - takes care of:
//  - sampling the right # of reads from the file
//  - decoding them using the small grid
//  - getting consensus alignment
// and returns;
//  - measurements for the sampled reads
//  - alignment results for the sampled reads
//  - flow order

void BruteForcePhase::sampleReadsFromRegion(int regionX, int regionY)
{

  string filenameWells = "1.wells";
  string filenameMask = pathAnalysis + "/bfmask.bin";

  int numReadsToSample = 1000;

  //
  // STEP 1. Retrieve all reads from the requested region and select requested number of library reads
  //

  // Open mask & wells

  Mask mask(1, 1);
  if (mask.SetMask(filenameMask.c_str())) {
    fprintf(stderr, "Cannot open mask\n");
    exit(EXIT_FAILURE);
  }
  int rows = mask.H();
  int cols = mask.W();

  RawWells rawWells(pathAnalysis.c_str(), filenameWells.c_str(), rows, cols);
  rawWells.OpenForRead();

  numFlows = rawWells.NumFlows();

  // Artificially cap length at 260 since we only care about first 100 bases anyways
  // and decoding longer reads can be computationally taxing
  numFlows = std::min(numFlows, 260);

  string flowOrderUnit = string(rawWells.FlowOrder());
//  string flowOrderUnit = "TACG";
//  printf("Flow order unit :'%s'\n", flowOrderUnit.c_str());


  flowOrder = "";
  while ((int) flowOrder.length() < numFlows)
    flowOrder += flowOrderUnit;

  keyFlow[0] = 1;
  keyFlow[1] = 0;
  keyFlow[2] = 1;
  keyFlow[3] = 0;
  keyFlow[4] = 0;
  keyFlow[5] = 1;
  keyFlow[6] = 0;
  keyFlowLen = 7;

  int cafieXinc = ceil(cols / (double) numRegionsX);
  int cafieYinc = ceil(rows / (double) numRegionsY);

  // Count library reads in the current region

  numReadsInRegion = 0;

  for (int x = regionX * cafieXinc; (x < (regionX + 1) * cafieXinc) && (x < cols); x++) {
    for (int y = regionY * cafieYinc; (y < (regionY + 1) * cafieYinc) && (y < rows); y++) {
      if (mask.Match(x + y * cols, MaskLib))
        numReadsInRegion++;
    }
  }

  int wellCounter = 0;
  int wellInterval = numReadsInRegion / (numReadsToSample + 1);
  regionMeasurements.clear();
  regionReferences.clear();

  numReadsSampled = 0;
  for (int x = regionX * cafieXinc; (x < (regionX + 1) * cafieXinc) && (x < cols); x++) {
    for (int y = regionY * cafieYinc; (y < (regionY + 1) * cafieYinc) && (y < rows); y++) {
      if (!mask.Match(x + y * cols, MaskLib))
        continue;
      wellCounter++;
      if (wellCounter < wellInterval)
        continue;
      wellCounter = 0;

      const WellData *data = rawWells.ReadXY(x, y);

      // Store the read

      regionMeasurements.push_back(vector<float> (numFlows));
      regionReferences.push_back(vector<int> (numFlows));
      for (int iFlow = 0; iFlow < numFlows; iFlow++) {
        regionMeasurements.back()[iFlow] = data->flowValues[iFlow];
        regionReferences.back()[iFlow] = 0;
      }

      numReadsSampled++;
      if (numReadsSampled == numReadsToSample)
        break;
    }
    if (numReadsSampled == numReadsToSample)
      break;
  }
  rawWells.Close();
}

void BruteForcePhase::decodeToFastq(string &filenameFastq)
{

  //
  // STEP 2. Decode the reads using provided grid and save to a fastq
  //

  DPTreephaser dpTreephaser(flowOrder.c_str(), numFlows, 8);

  FILE *f_fastq = fopen(filenameFastq.c_str(), "w");

  for (int iRead = 0; iRead < numReadsSampled; iRead++) {

    for (int iParameter = 0; iParameter < alignmentGrid.size(); iParameter++) {

      dpTreephaser.SetModelParameters(alignmentGrid.CF[iParameter], alignmentGrid.IE[iParameter], alignmentGrid.DR[iParameter]);

      BasecallerRead well;
      well.SetDataAndKeyNormalize(&(regionMeasurements[iRead][0]), numFlows, keyFlow, keyFlowLen);

      dpTreephaser.NormalizeAndSolve3(well, numFlows); // Adaptive normalization


      int numBases = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++)
        numBases += (int) well.solution[iFlow];

      if (numBases < 100)
        continue;


      fprintf(f_fastq, "@READ%08d_%08d\n", iRead, iParameter);

      numBases = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++) {
        int nHP = (int) well.solution[iFlow];
        for (int iHP = 0; iHP < nHP; iHP++) {
          numBases++;
          if (numBases > 4)
            putc(flowOrder[iFlow], f_fastq);
        }
      }
      fprintf(f_fastq, "\n+\n");
      for (int iBase = 4; iBase < numBases; iBase++)
        putc(')', f_fastq);
      fprintf(f_fastq, "\n");
    }
  }

  fflush(f_fastq);
  fclose(f_fastq);
}

void BruteForcePhase::determineConsensusReference(string &filenameBam)
{

  //
  // STEP 4. Parse alignment results. The "consensus" is really the longest alignment
  //

  vector<int> alignmentLength(numReadsSampled, 0);

  BAMReader reader(filenameBam);
  reader.open(); //don't like doing this, should just be able to get an iterator

  for (BAMReader::iterator i = reader.get_iterator(); i.good(); i.next()) {
    BAMRead read = i.get();
    BAMUtils util(read);
    int iRead = -1;
    int iParameter = -1;
    sscanf(util.get_name().c_str(), "READ%d_%d", &iRead, &iParameter);

    // Old

    if (iParameter < 0) {
      printf("iParameter = %d\n", iParameter);
      continue;
    }
    if (iRead < 0) {
      printf("iRead = %d\n", iRead);
      continue;
    }

    if (iParameter >= alignmentGrid.size()) {
      printf("iParameter = %d\n", iParameter);
      continue;
    }
    if (iRead >= numReadsSampled) {
      printf("iRead = %d\n", iRead);
      continue;
    }

    // New
    string reference = "TCAG" + util.get_tdna();
    int localLength = 0;

    for (const char *c = reference.c_str(); *c; c++)
      if ((*c == 'A') || (*c == 'C') || (*c == 'T') || (*c == 'G'))
        localLength++;

    if (localLength <= alignmentLength[iRead])
      continue;

    alignmentLength[iRead] = localLength;

    for (int iFlow = 0; iFlow < numFlows; iFlow++)
      regionReferences[iRead][iFlow] = 0;

    int iPos = 0;
    for (const char *c = reference.c_str(); *c; c++) {
      if ((*c != 'A') && (*c != 'C') && (*c != 'T') && (*c != 'G'))
        continue;
      while ((*c != flowOrder[iPos]) && (iPos < numFlows))
        iPos++;
      if (iPos == numFlows)
        break;
      regionReferences[iRead][iPos]++;
    }
  }

  reader.close();

}
void BruteForcePhase::analyzeResultsGrid()
{
  int numWorkers = 12;

  numReads100q17.assign(resultsGrid.size(), 0);
  isRead100q17.assign(numReadsSampled, false);
  nextParameters = 0;

  pthread_mutex_t tmpMutex = PTHREAD_MUTEX_INITIALIZER;
  mutex = &tmpMutex;

  pthread_t workerId[numWorkers];

  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    if (pthread_create(&workerId[iWorker], NULL, BruteForceWorker, this)) {
      fprintf(stderr, "*Error* - problem starting thread\n");
      exit(EXIT_FAILURE);
    }

  for (int iWorker = 0; iWorker < numWorkers; iWorker++)
    pthread_join(workerId[iWorker], NULL);

  numReads100q17Anywhere = 0;
  for (int iRead = 0; iRead < numReadsSampled; iRead++)
    if (isRead100q17[iRead])
      numReads100q17Anywhere++;

}

void *BruteForceWorker(void *arg)
{
  reinterpret_cast<BruteForcePhase *> (arg)->worker_Treephaser();
  return NULL;
}


void BruteForcePhase::worker_Treephaser()
{

  while (true) {

    pthread_mutex_lock(mutex);
    int iParameters = nextParameters;
    nextParameters++;
    pthread_mutex_unlock(mutex);

    if (iParameters >= resultsGrid.size())
      return;

    DPTreephaser dpTreephaser(flowOrder.c_str(), numFlows, 8);
    dpTreephaser.SetModelParameters(resultsGrid.CF[iParameters], resultsGrid.IE[iParameters], resultsGrid.DR[iParameters]);

    for (int iRead = 0; iRead < numReadsSampled; iRead++) {

      int alignmentLength = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++)
        alignmentLength += regionReferences[iRead][iFlow];
      if (alignmentLength < 104)
        continue;

      BasecallerRead well;
      well.SetDataAndKeyNormalize(&(regionMeasurements[iRead][0]), numFlows, keyFlow, keyFlowLen);
      dpTreephaser.NormalizeAndSolve3(well, numFlows);

      int numBases = 0;
      int numErrors = 0;
      for (int iFlow = 0; iFlow < numFlows; iFlow++) {

        int delta = regionReferences[iRead][iFlow] - (int) well.solution[iFlow];
        if (delta > 0)
          numErrors += delta;
        else
          numErrors += -delta;

        numBases += (int) well.solution[iFlow];
        if (numBases >= 104)
          break;
        if (numErrors > 2)
          break;
      }
      if (numErrors <= 2) {
        numReads100q17[iParameters]++;
        pthread_mutex_lock(mutex);
        isRead100q17[iRead] = true;
        pthread_mutex_unlock(mutex);
      }
    }
  }
}


void BruteForcePhase::openReport()
{
  /*
  string filenameReport = pathOutput + "/BruteForcePhase.html";
  f_report = fopen(filenameReport.c_str(), "w");

  fprintf(f_report, "<html><head><link rel=\"stylesheet\" type=\"text/css\" href=\"BruteForcePhase.css\" /></head>\n");
  fprintf(f_report, "<body>\n");
  fprintf(f_report, "<h1>BruteForcePhase report</h1>\n");
  fprintf(f_report, "<div class=\"description\">\n");
  fprintf(f_report, "    BruteForcePhase evaluates the quality of phase parameter estimation\n");
  fprintf(f_report, "    by basecalling a subset of library reads using a grid of CF/IE/DR values\n");
  fprintf(f_report, "    and reporting the number of 100Q17 alignments.\n");
  fprintf(f_report, "</div><br/>\n\n<table>\n\n");

  fflush(f_report);

  string filenameBlock = pathOutput + "/BruteForcePhase_block.html";
  f_block = fopen(filenameBlock.c_str(), "w");
  fprintf(f_block, "<html><body>\n");
  fprintf(f_block, "<div style=\"margin:0px;padding:0px;\">\n\n");
  fprintf(f_block, "<div style=\"margin:2px;padding:2px;padding-left:0px;padding-right:40px;float:left;\">");
  //    fprintf(f_block, "<a href=\"BruteForcePhase.htm\">Phase estimation performance</a>:</div>\n\n");
  fprintf(f_block, "Phase estimation performance:</div>\n\n");

  fflush(f_block);
  */
}

void BruteForcePhase::reportRegionResults(int regionX, int regionY)
{

  int idxRegion = regionX + regionY * numRegionsX;

  double red[numReadsSampled + 1];
  double green[numReadsSampled + 1];
  double blue[numReadsSampled + 1];

  int minReads100q17 = numReadsSampled;
  int maxReads100q17 = 0;

  double systemMatch[resultsGrid.size()];
  double systemMatchMin = 1000;
  int match100q17 = 0;
  int bestParameter = 0;

  for (int iParameter = 0; iParameter < resultsGrid.size(); iParameter++) {

    maxReads100q17 = std::max(maxReads100q17, numReads100q17[iParameter]);
    minReads100q17 = std::min(minReads100q17, numReads100q17[iParameter]);

    if (iParameter < (resultsGrid.size() - 1)) {
      if (numReads100q17[iParameter] > numReads100q17[bestParameter])
        bestParameter = iParameter;

      systemMatch[iParameter] = fabs(estimatedCF[idxRegion] - resultsGrid.CF[iParameter]) + fabs(estimatedIE[idxRegion] - resultsGrid.IE[iParameter]) + fabs(
          estimatedDR[idxRegion] - resultsGrid.DR[iParameter]);

      if (systemMatchMin > systemMatch[iParameter])
        systemMatchMin = systemMatch[iParameter];
    }
  }
  match100q17 = numReads100q17.back();

  for (int iCount = minReads100q17; iCount <= maxReads100q17; iCount++) {

    double fraction = ((double) (iCount - minReads100q17)) / (double) (1 + maxReads100q17 - minReads100q17);

    if (fraction < 0.125) {
      red[iCount] = 0;
      green[iCount] = 0;
      blue[iCount] = 0.5 + 4 * fraction;

    } else if (fraction < 0.375) {
      red[iCount] = 0;
      green[iCount] = 4 * (fraction - 0.125);
      blue[iCount] = 1;

    } else if (fraction < 0.625) {
      red[iCount] = 4 * (fraction - 0.375);
      green[iCount] = 1;
      blue[iCount] = 1 - 4 * (fraction - 0.375);

    } else if (fraction < 0.875) {
      red[iCount] = 1;
      green[iCount] = 1 - 4 * (fraction - 0.625);
      blue[iCount] = 0;

    } else {
      red[iCount] = 1 - 4 * (fraction - 0.875);
      green[iCount] = 0;
      blue[iCount] = 0;
    }
  }
/*
  fprintf(f_report, "<tr><th><table class=\"zone1\">\n");
  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th rowspan=\"5\" class=\"zone1A\"/>\n");
  fprintf(f_report, "        <th rowspan=\"5\" class=\"zone1B\"><div>region %d</div> <div style=\"font-size:20px\">(%d,%d)</div></th>\n", idxRegion, regionX,
      regionY);
  fprintf(f_report, "        <th rowspan=\"5\" class=\"zone1C\"/>\n");
  fprintf(f_report, "        <th rowspan=\"5\" class=\"zone1separator\"/>\n");
  fprintf(f_report, "        <th rowspan=\"5\"><table class=\"regionPhase\" style=\"margin:0px\">\n");

  for (int y = numRegionsY - 1; y >= 0; y--) {
    fprintf(f_report, "            <tr>");
    for (int x = 0; x < numRegionsX; x++) {
      if ((x == regionX) && (y == regionY))
        fprintf(f_report, "<th class=\"qdark\" />");
      else
        fprintf(f_report, "<th class=\"qlight\" style=\"background-color:#%02X%02X%02X\"/>", (int) (255 - 2000.0 * estimatedIE[x + y * numRegionsX]),
            (int) (255 - 2000.0 * estimatedIE[x + y * numRegionsX]), (int) (255 - 2000.0 * estimatedIE[x + y * numRegionsX]));
    }
    fprintf(f_report, "</tr>\n");
  }

  fprintf(f_report, "        </table></th>\n");
  fprintf(f_report, "        <th rowspan=\"5\" class=\"zone1separator\"/>\n");
  fprintf(f_report, "        <th class=\"zone1D\">estimation</th>\n");
  fprintf(f_report, "        <th class=\"zone1E\" style=\"background-color:#%02X%02X%02X;color:white\">%1.0f%%</th>\n",
      (unsigned int) (210 * red[match100q17]), (unsigned int) (210 * green[match100q17]), (unsigned int) (210 * blue[match100q17]),
      (100.0 * match100q17) / std::max(numReads100q17[bestParameter], 1));

  fprintf(f_report, "        <th class=\"zone1F\"/>\n");
  fprintf(f_report, "    </tr>\n");
  fprintf(f_report, "    <tr height=3 />\n");
  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th class=\"zone1D\">total reads</th>\n");
  fprintf(f_report, "        <th class=\"zone1E\">%d</th>\n", numReadsInRegion);
  fprintf(f_report, "        <th class=\"zone1F\"/>\n");
  fprintf(f_report, "    </tr>\n");
  fprintf(f_report, "    <tr height=3 />\n");
  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th class=\"zone1D\">sampled reads</th>\n");
  fprintf(f_report, "        <th class=\"zone1E\">%d</th>\n", numReadsSampled);
  fprintf(f_report, "        <th class=\"zone1F\"/>\n");
  fprintf(f_report, "    </tr>\n");
  fprintf(f_report, "</table>\n");

  fprintf(f_report, "<table class=\"zone2\">\n");
  fprintf(f_report, "    <tr><th/><th/><th>cf</th><th>ie</th><th>dr</th><th>100q17</th></tr>\n");
  fprintf(f_report, "    <tr height=5 />\n");

  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th class=\"zone2A\"></th>\n");
  fprintf(f_report, "        <th class=\"zone2B\">estimated</th>\n");
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * estimatedCF[idxRegion]);
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * estimatedIE[idxRegion]);
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * estimatedDR[idxRegion]);
  fprintf(f_report, "        <th class=\"zone2D\">%d</th>\n", match100q17);
  fprintf(f_report, "        <th class=\"zone2E\"></th>\n");
  fprintf(f_report, "    </tr>\n");

  fprintf(f_report, "    <tr height=3 />\n");

  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th class=\"zone2A\"/>\n");
  fprintf(f_report, "        <th class=\"zone2B\">brute force</th>\n");
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * resultsGrid.CF[bestParameter]);
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * resultsGrid.IE[bestParameter]);
  fprintf(f_report, "        <th class=\"zone2C\">%1.2f %%</th>\n", 100.0 * resultsGrid.DR[bestParameter]);
  fprintf(f_report, "        <th class=\"zone2D\">%d</th>\n", numReads100q17[bestParameter]);
  fprintf(f_report, "        <th class=\"zone2E\"></th>\n");
  fprintf(f_report, "    </tr>\n");

  fprintf(f_report, "    <tr height=3 />\n");

  fprintf(f_report, "    <tr>\n");
  fprintf(f_report, "        <th class=\"zone2A\"></th>\n");
  fprintf(f_report, "        <th class=\"zone2B\">read-optimal</th>\n");
  fprintf(f_report, "        <th class=\"zone2C\">*</th>\n");
  fprintf(f_report, "        <th class=\"zone2C\">*</th>\n");
  fprintf(f_report, "        <th class=\"zone2C\">*</th>\n");
  fprintf(f_report, "        <th class=\"zone2D\">%d</th>\n", numReads100q17Anywhere);
  fprintf(f_report, "        <th class=\"zone2E\"></th>\n");
  fprintf(f_report, "    </tr>\n");

  fprintf(f_report, "</table>\n");
  fprintf(f_report, "</th>\n");

  for (unsigned int dr_idx = 0; dr_idx < resultsGrid.uniqueDR.size(); dr_idx++) {

    fprintf(f_report, "<th style=\"font-family:arial;color:gray\">\n");
    fprintf(f_report, "DR = %1.2f%%\n", resultsGrid.uniqueDR[dr_idx] * 100.0);
    fprintf(f_report, "<table class=\"regionInner2\">\n");
    fprintf(f_report, "<tr><th>IE</th><th>CF</th>");
    for (unsigned int cf_idx = 0; cf_idx < resultsGrid.uniqueCF.size(); cf_idx++)
      fprintf(f_report, "<th><font size=1>%1.1f%%</font></th>", resultsGrid.uniqueCF[cf_idx] * 100.0);
    fprintf(f_report, "</tr>\n");

    for (unsigned int ie_idx = 0; ie_idx < resultsGrid.uniqueIE.size(); ie_idx++) {

      fprintf(f_report, "<tr><th><font size=1>%1.1f%%</font></th><th></th>", resultsGrid.uniqueIE[ie_idx] * 100.0);

      for (unsigned int cf_idx = 0; cf_idx < resultsGrid.uniqueCF.size(); cf_idx++) {

        int iParameters = resultsGrid.gridToIndex(cf_idx, ie_idx, dr_idx);

        fprintf(f_report, "<th height=35 width=30 style=\"background-color:#%02X%02X%02X\">", (unsigned int) (210 * red[numReads100q17[iParameters]]),
            (unsigned int) (210 * green[numReads100q17[iParameters]]), (unsigned int) (210 * blue[numReads100q17[iParameters]]));

        if (systemMatchMin == systemMatch[iParameters])
          fprintf(f_report, "<font color=white size=1>estim</font> ");

        fprintf(f_report, "<font color=white size=1>%d</font>", numReads100q17[iParameters]);

        if (numReads100q17[iParameters] == maxReads100q17)
          fprintf(f_report, " <font color=white size=1>best</font>");

        fprintf(f_report, "</th>\n");

      }
      fprintf(f_report, "</tr>\n");
    }

    fprintf(f_report, "</table>\n");
    fprintf(f_report, "</th>\n");

  }
  fprintf(f_report, "</tr>\n\n");

  fflush(f_report);

  fprintf(f_block,
      "<div style=\"background-color:#%02X%02X%02X;color:white;padding:2px;padding-left:5px;padding-right:10px;margin:2px;float:left;\">%1.0f%%</div>",
      (unsigned int) (210 * red[match100q17]), (unsigned int) (210 * green[match100q17]), (unsigned int) (210 * blue[match100q17]),
      (100.0 * match100q17) / std::max(numReads100q17[bestParameter], 1));

  fflush(f_block);
*/


  char jsonRegionName[256];
  sprintf(jsonRegionName,"%d,%d",regionX,regionY);

  for (unsigned int cf_idx = 0; cf_idx < resultsGrid.uniqueCF.size(); cf_idx++)
    phaseJson["BruteForce"][jsonRegionName]["gridCF"][cf_idx] = resultsGrid.uniqueCF[cf_idx];

  for (unsigned int ie_idx = 0; ie_idx < resultsGrid.uniqueIE.size(); ie_idx++)
    phaseJson["BruteForce"][jsonRegionName]["gridIE"][ie_idx] = resultsGrid.uniqueIE[ie_idx];

  for (unsigned int ie_idx = 0; ie_idx < resultsGrid.uniqueIE.size(); ie_idx++) {
    for (unsigned int cf_idx = 0; cf_idx < resultsGrid.uniqueCF.size(); cf_idx++) {

      int entryIdx = ie_idx + cf_idx * resultsGrid.uniqueIE.size();
      int iParameters = resultsGrid.gridToIndex(cf_idx, ie_idx, 0);

      phaseJson["BruteForce"][jsonRegionName]["grid100q17"][entryIdx] = numReads100q17[iParameters];

    }
    phaseJson["BruteForce"][jsonRegionName]["estimCF"] = estimatedCF[idxRegion];
    phaseJson["BruteForce"][jsonRegionName]["estimIE"] = estimatedIE[idxRegion];
    phaseJson["BruteForce"][jsonRegionName]["estim100q17"] = match100q17;
    phaseJson["BruteForce"][jsonRegionName]["perread100q17"] = numReads100q17Anywhere;
    phaseJson["BruteForce"][jsonRegionName]["numReadsInRegion"] = numReadsInRegion;
    phaseJson["BruteForce"][jsonRegionName]["numReadsSampled"] = numReadsSampled;
  }
}

void BruteForcePhase::closeReport()
{
  /*
  fprintf(f_report, "</table></body></html>\n");
  fclose(f_report);

  fprintf(f_block, "</div></table></body></html>\n");
  fclose(f_block);
*/

  string filename = pathOutput + "/PhaseReport.json";
  ofstream out(filename.c_str(), ios::out);
  if (out.good())
    out << phaseJson.toStyledString();

}

void BruteForcePhase::addRegionToReport(int regionX, int regionY)
{

  string filenameFastq = pathOutput + "/BruteForcePhase.fastq";
  string filenameBam = pathOutput + "/BruteForcePhase.bam";
  string filenameReference = "/results/referenceLibrary/tmap-f2/" + referenceName + "/" + referenceName + ".fasta";

  // STEP 1. Initialize small and large cf/ie/dr grids, centering them around Analysis estimates

  printf("Region (%d,%d): Initializing grids\n", regionX, regionY);
  fflush(stdout);

  initializeGrids(regionX, regionY);

  // STEP 2. Attempt to sample 1000 library reads from the region

  printf("Region (%d,%d): Sampling library reads\n", regionX, regionY);
  fflush(stdout);

  sampleReadsFromRegion(regionX, regionY);

  printf("Region (%d,%d): Sampled %d/%d\n", regionX, regionY, numReadsSampled, numReadsInRegion);

  if (numReadsSampled > 0) {

    // STEP 3. Decode sampled reads using small grid and save calls to a temporary fastq file

    printf("Region (%d,%d): Decoding for alignment\n", regionX, regionY);
    fflush(stdout);

    decodeToFastq(filenameFastq);

    // STEP 4. Invoke the aligner

    printf("Region (%d,%d): Aligning\n", regionX, regionY);
    fflush(stdout);

    char commandTmap[2048];
    snprintf(commandTmap, 2048, "tmap mapall -n 4 -f %s -r %s -v stage1 map1 map2 map3  | samtools view -Sb -o %s -", filenameReference.c_str(),
        filenameFastq.c_str(), filenameBam.c_str());
    if (system(commandTmap)) {
      fprintf(stderr, "Failed to execute tmap or samtools\n");
      exit(EXIT_FAILURE);
    }

    // STEP 5. Parse alignment results and determine a consensus reference for each read

    printf("Region (%d,%d): Parsing alignment results\n", regionX, regionY);
    fflush(stdout);

    determineConsensusReference(filenameBam);

    // STEP 6. Clean up and return

    printf("Region (%d,%d): Cleaning up temp files\n", regionX, regionY);
    fflush(stdout);

    unlink(filenameFastq.c_str());
    unlink(filenameBam.c_str());
  }

  // STEP 7: Decode sampled reads using large grid and count how many of them are 100q17

  printf("Region (%d,%d): Decoding for analysis\n", regionX, regionY);
  fflush(stdout);

  analyzeResultsGrid();

  // STEP 8: Write out the results as a chunk of html table

  printf("Region (%d,%d): Appending analysis results to report\n", regionX, regionY);
  fflush(stdout);

  reportRegionResults(regionX, regionY);
}

int main(int argc, char *argv[])
{
  if (argc <= 3)
    usage(argv[0]);

  BruteForcePhase bruteForcePhase(argv[1], argv[2], argv[3]);

  bruteForcePhase.openReport();

  bruteForcePhase.addRegionToReport(1, 1);
  bruteForcePhase.addRegionToReport(6, 6);
  bruteForcePhase.addRegionToReport(10, 2);
  bruteForcePhase.addRegionToReport(11, 10);

  bruteForcePhase.closeReport();



  return 0;
}

