/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */


#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h> // for getopt_long
#include <assert.h>
#include <time.h>

#include <vector>
#include <iostream>
#include <fstream>

#include "IonVersion.h"
#include "RawWells.h"
//#include "SFFWrapper.h"
#include "fstrcmp.h"
#include "Histogram.h"
#include "file-io/ion_util.h"
#include "fstrcmp.h"
#include "TFs.h"
#include "DPTreephaser.h"
#include "Utils.h"

#include "SamUtils/BAMReader.h"
#include "SamUtils/BAMUtils.h"
#include "SamUtils/types/BAMRead.h"

#include "file-io/sff_definitions.h"
#include "file-io/sff.h"
#include "file-io/sff_file.h"


#include "dbgmem.h"

#include "json/json.h"

using namespace std;





class SemiOrderedSFFReader {
public:
  SemiOrderedSFFReader() : nRows(0), nCols(0), sff_file(NULL) { }
  ~SemiOrderedSFFReader() { Close(); }

  void Open(const char *filename, int _nRows, int _nCols) {
    nRows = _nRows; nCols = _nCols;
    sff_file = sff_fopen(filename, "rb", NULL, NULL);
//    fseek(sff_file->fp, sff_file->header->gheader_length + sff_file->header->index_length, SEEK_SET);
    sff.assign(nRows*nCols, NULL);
  }
  void Close() {
    for (vector<sff_t *>::iterator I = sff.begin(); I != sff.end(); I++) {
      sff_destroy(*I);
      *I = NULL;
    }
    if (sff_file) {
        sff_fclose(sff_file);
        sff_file = NULL;
    }
  }
  sff_t *LoadXY(int x, int y) {
    int myRead = x + y*nCols;
    while (sff[myRead] == NULL) {
      sff_t *next_sff = sff_read(sff_file);
      if (next_sff == NULL)
        break;
      int next_x = -1, next_y = -1;
      const char *ptr = next_sff->rheader->name->s;
      while (*ptr && *ptr != ':')
        ptr++;
      sscanf(ptr, ":%d:%d", &next_y, &next_x);
      if ((next_x < 0) || (next_y < 0)) {
        sff_destroy(next_sff);
        continue;
      }
      int nextRead = next_x + next_y*nCols;
      sff_destroy(sff[nextRead]);
      sff[nextRead] = next_sff;
    }
    return sff[myRead];
  }
  void FreeXY(int x, int y) {
    int myRead = x + y*nCols;
    sff_destroy(sff[myRead]);
    sff[myRead] = NULL;
  }

private:
  int nRows;
  int nCols;
  sff_file_t *sff_file;
  vector<sff_t *> sff;
};




class CafieMetricsGenerator {
public:
  CafieMetricsGenerator (bool isActive, int _numTFs, int numFlows) {
    if (!isActive) {
      active = false;
      return;
    }
    active = true;

    numTFs = _numTFs;
    numFlowsTFClassify = numFlows;
    avgTFSignal.assign(numTFs, std::vector<double>(numFlowsTFClassify, 0.0));
    avgTFCorrected.assign(numTFs, std::vector<double>(numFlowsTFClassify, 0.0));
    avgTFSignalSquared.assign(numTFs, vector<double>(numFlowsTFClassify, 0.0));
    tfSignalHist.assign(numTFs, vector<int>(numFlowsTFClassify*maxTFSignalHist,0));

    tfCallCorrect2.assign(numTFs, vector<int>(4*maxTFHPHist,0));
    tfCallUnder2.assign(numTFs, vector<int>(4*maxTFHPHist,0));
    tfCallOver2.assign(numTFs, vector<int>(4*maxTFHPHist,0));

    tfCallCorrect3.assign(numTFs, vector<int>(maxTFSparklineFlows,0));
    tfCallTotal3.assign(numTFs, vector<int>(maxTFSparklineFlows,0));

    tfCount.assign(numTFs,0);

    cafieMetricsFilename = "cafieMetrics.txt";
    CF = 0;
    IE = 0;
    DR = 0;
    blockCount = 0;
    numRows = -1;
    numCols = -1;
    numRegionRows = -1;
    numRegionCols = -1;

//    allIonograms.resize(numTFs);
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
    in >> BaseCallerJson;


    numRows = _numRows;
    numCols = _numCols;

    numRegionRows = BaseCallerJson["Phasing"].get("RegionRows",-1).asInt();
    numRegionCols = BaseCallerJson["Phasing"].get("RegionCols",-1).asInt();

    if ((numRegionRows <=0) || (numRegionCols <= 0)) {
      parseCafieRegions(AnalysisDirectory);
      return;

//      printf("Parsing of %s unsuccessful. Aborting\n", (string(AnalysisDirectory) + "/BaseCaller.json").c_str());
//      exit(1);
    }

    CFbyRegion.resize(numRegionRows*numRegionCols);
    IEbyRegion.resize(numRegionRows*numRegionCols);
    for (int iRegion = 0; iRegion < numRegionRows*numRegionCols; iRegion++) {
      CFbyRegion[iRegion] = BaseCallerJson["Phasing"]["CFbyRegion"][iRegion].asFloat();
      IEbyRegion[iRegion] = BaseCallerJson["Phasing"]["IEbyRegion"][iRegion].asFloat();
    }
    CF += BaseCallerJson["Phasing"].get("CF",0).asFloat();
    IE += BaseCallerJson["Phasing"].get("IE",0).asFloat();
    DR += BaseCallerJson["Phasing"].get("DR",0).asFloat();
    blockCount++;

  }




  void parseCafieRegions(const char *AnalysisDirectory)
  {


//    if ((numRegionRows <=0) || (numRegionCols <= 0)) {
//      printf("Parsing of %s unsuccessful. Aborting\n", (string(AnalysisDirectory) + "/BaseCaller.json").c_str());
//      exit(1);
//    }

    string filenameCafieRegions = string(AnalysisDirectory) + "/cafieRegions.txt";

    CFbyRegion.clear();
    IEbyRegion.clear();
//    estimatedDR.clear();
    numRegionRows = 0;
    numRegionCols = 0;

    // Open cafieRegions.txt and skip the header

    FILE *f_cafie = fopen(filenameCafieRegions.c_str(), "r");

    if (!f_cafie) {
      fprintf(stderr, "parseCafieRegions: %s cannot be opened (%s)\n", filenameCafieRegions.c_str(), strerror(errno));
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
        CFbyRegion.push_back(val);
        if (numRegionRows == 0)
          numRegionCols++;
      }
      numRegionRows++;

      if (!fgets(line, 2048, f_cafie))
        break;
      if (0 == strncmp(line, "Region IE = ", 12))
        break;
    }

    // Read IE

    for (int iRegionY = 0; iRegionY < numRegionRows; iRegionY++) {
      char *c = line;
      if (0 == strncmp(line, "Region IE = ", 12))
        c += 12;

      for (int iRegionX = 0; iRegionX < numRegionCols; iRegionX++)
        IEbyRegion.push_back(strtof(c, &c));

      if (!fgets(line, 2048, f_cafie))
        break;
    }

    // Read DR

    for (int iRegionY = 0; iRegionY < numRegionRows; iRegionY++) {
      char *c = line;
      if (0 == strncmp(line, "Region DR = ", 12))
        c += 12;

//      for (int iRegionX = 0; iRegionX < numRegionCols; iRegionX++)
//        estimatedDR.push_back(strtof(c, &c));

      if (!fgets(line, 2048, f_cafie))
        break;
    }

    fclose(f_cafie);

    if ((numRegionCols <= 0) || (numRegionRows <= 0)) {
      fprintf(stderr, "Parsing of cafieRegions.txt unsuccessful (%d x %d, %d %d)\n", numRegionCols, numRegionRows, (int) CFbyRegion.size(),
          (int) IEbyRegion.size());
      exit(EXIT_FAILURE);
    }



    vector<float> newCF(numRegionRows*numRegionCols);
    vector<float> newIE(numRegionRows*numRegionCols);

    int idx = 0;
    for (int iRegionX = 0; iRegionX < numRegionCols; iRegionX++) {
      for (int iRegionY = 0; iRegionY < numRegionRows; iRegionY++) {
        newCF[idx] = CFbyRegion[iRegionX + iRegionY*numRegionCols];
        newIE[idx] = IEbyRegion[iRegionX + iRegionY*numRegionCols];
        idx++;
      }
    }
    CFbyRegion.swap(newCF);
    IEbyRegion.swap(newIE);

    float blockavgCF = 0;
    float blockavgIE = 0;
    for (int i = 0; i < (int)CFbyRegion.size(); i++) {
      blockavgCF += CFbyRegion[i];
      blockavgIE += IEbyRegion[i];
    }

    CF += (blockavgCF / (float)CFbyRegion.size());
    IE += (blockavgIE / (float)IEbyRegion.size());
    blockCount++;

  }







  void AddElement(int tf, float *rawValues, uint16_t *corValues, int len, int x, int y, char *flowOrder,
      const string & genome, const string & calls) {
    if ((!active) || (numRows <= 0))
      return;

    int numFlows = len;
    int flowOrderLength = strlen(flowOrder);

    //
    // Use alignments to generate "synchronized" reference and read ionograms
    //

    int numBases = min(genome.length(),calls.length());
    vector<int> refIonogram(numFlows, 0);
    vector<int> readIonogram(numFlows, 0);

    int numFlowsRead = 0;
    int numFlowsRef = 0;
    char gC = flowOrder[0];
    int gBC = 0;

    for (int iBase = 0; (iBase < numBases) && (numFlowsRead < numFlows) && (numFlowsRef < numFlows); iBase++) {

      // Conversion for reads (independent of reference)
      if (calls[iBase] != '-') {
        while ((calls[iBase] != flowOrder[numFlowsRead % flowOrderLength]) && (numFlowsRead < numFlows))
          numFlowsRead++;
        if (numFlowsRead < numFlows)
          readIonogram[numFlowsRead]++;
      }

      if (genome[iBase] != '-') {

        if (genome[iBase] != gC) {
          // Since a new homopolymer begins, need to drop off the old one
          while ((gC != flowOrder[numFlowsRef % flowOrderLength]) && (numFlowsRef < numFlows)) {
            numFlowsRef++;
            if (numFlowsRef < numFlows)
              refIonogram[numFlowsRef] = 0;
          }
          if (numFlowsRef < numFlows)
            refIonogram[numFlowsRef] = gBC;

          gC = genome[iBase];
          gBC = 0;
        }
        gBC++;

        if (genome[iBase] == calls[iBase])
          numFlowsRef = numFlowsRead;
      }
    }


    for (int iFlow = 8; (iFlow < numFlowsRef-20) && (iFlow < numFlowsRead-20); iFlow++) {

      int baseIdx = 0;
      if      (flowOrder[iFlow % flowOrderLength] == 'C') baseIdx = 1;
      else if (flowOrder[iFlow % flowOrderLength] == 'G') baseIdx = 2;
      else if (flowOrder[iFlow % flowOrderLength] == 'T') baseIdx = 3;

      if (readIonogram[iFlow] == refIonogram[iFlow])
        tfCallCorrect2[tf][refIonogram[iFlow] + baseIdx*maxTFHPHist]++;
      else if (readIonogram[iFlow] > refIonogram[iFlow])
        tfCallOver2[tf][refIonogram[iFlow] + baseIdx*maxTFHPHist]++;
      else
        tfCallUnder2[tf][refIonogram[iFlow] + baseIdx*maxTFHPHist]++;
    }

    // Sparkline data
    for (int iFlow = 0; (iFlow < numFlowsRef) && (iFlow < numFlowsRead) && (iFlow < maxTFSparklineFlows); iFlow++) {
      tfCallTotal3[tf][iFlow]++;
      if (readIonogram[iFlow] == refIonogram[iFlow])
        tfCallCorrect3[tf][iFlow]++;
    }



    tfCount[tf]++;

    BasecallerRead well;
    well.SetDataAndKeyNormalize(rawValues, len, &(refIonogram[0]), 7);
    int lastCalledFlow = 0;
    for (int iFlow = 0; (iFlow < len); iFlow++) {
      well.solution[iFlow] = (char)((corValues[iFlow]+50)/100);
      if (well.solution[iFlow] > 0)
        lastCalledFlow = iFlow;
    }


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
      avgTFSignalSquared[tf][iFlow] += well.normalizedMeasurements[iFlow] * well.normalizedMeasurements[iFlow];

      int quantizedTFSignal = (int) rint(40.0 * well.normalizedMeasurements[iFlow]);
      quantizedTFSignal = min(max(quantizedTFSignal,0),maxTFSignalHist-1);
      tfSignalHist[tf][iFlow * maxTFSignalHist + quantizedTFSignal]++;
    }



  }


  void GenerateCafieMetrics(const bam_header_t *header, int len, char *flowOrder, const string &jsonFilename, const vector<string> &referenceSequences) {
    if (!active)
      return;

    if (blockCount > 0) {
      CF /= blockCount;
      IE /= blockCount;
      DR /= blockCount;
    }



    // New stuff: Output TF-related data into the json structure

    int flowOrderLength = strlen(flowOrder);

    int iTFPresent = 0;
    for (int tf = 0; tf < numTFs; tf++) {
      if (tfCount[tf] <= 200)
        continue;

      Json::Value tfJson;
      tfJson["name"] = header->target_name[tf];
      tfJson["count"] = tfCount[tf];

      DPTreephaser dpTreephaser(flowOrder, len, 8);

      dpTreephaser.SetModelParameters(CF, IE, 0); // Adaptive normalization

      BasecallerRead read;
      read.numFlows = len;
      read.solution.assign(len, 0);
      read.prediction.assign(len, 0);

      for (int iFlow = 0, iBase = 0; (iFlow < len) && (iBase < (int)referenceSequences[tf].length());) {
        char base = referenceSequences[tf][iBase];
        if ((base != 'A') && (base != 'C') && (base != 'G') && (base != 'T')) {
          iBase++;
          continue;
        }

        if (base != flowOrder[iFlow % flowOrderLength]) {
          iFlow++;
          continue;
        }

        read.solution[iFlow]++;
        iBase++;
      }

      dpTreephaser.Simulate3(read, len);

      for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
        tfJson["signalExpected"][iFlow] = read.prediction[iFlow];




      string tfFlowOrder;
      tfFlowOrder.resize(numFlowsTFClassify);

      for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++) {
        tfJson["signalCorrected"][iFlow] = avgTFCorrected[tf][iFlow] / tfCount[tf];

        double signal = avgTFSignal[tf][iFlow] / tfCount[tf];
        tfJson["signalMean"][iFlow] = signal;
        tfJson["signalStd"][iFlow] = avgTFSignalSquared[tf][iFlow] / tfCount[tf] - signal * signal;

        tfFlowOrder[iFlow] = flowOrder[iFlow % flowOrderLength];

      }
      tfJson["flowOrder"] = tfFlowOrder;

      for (int idx = 0; idx < numFlowsTFClassify * maxTFSignalHist; idx++)
        tfJson["signalHistogram"][idx] = tfSignalHist[tf][idx];

      for (int hp = 0; hp < 4*maxTFHPHist; hp++) {
        tfJson["tfCallCorrect2"][hp] = tfCallCorrect2[tf][hp];
        tfJson["tfCallOver2"][hp] = tfCallOver2[tf][hp];
        tfJson["tfCallUnder2"][hp] = tfCallUnder2[tf][hp];
      }

      for (int iFlow = 0; iFlow < maxTFSparklineFlows; iFlow++) {
        tfJson["tfCallCorrect3"][iFlow] = tfCallCorrect3[tf][iFlow];
        tfJson["tfCallTotal3"][iFlow] = tfCallTotal3[tf][iFlow];
        if (iFlow < len)
          tfJson["tfCallHP3"][iFlow] = (int)read.solution[iFlow];
        else
          tfJson["tfCallHP3"][iFlow] = 0;
      }


      basecallerJson["TestFragments"][iTFPresent] = tfJson;
      iTFPresent++;
    }

    ofstream out(jsonFilename.c_str(), ios::out);
    if (out.good())
      out << basecallerJson.toStyledString();
  }





private:
  bool active;
  Json::Value BaseCallerJson;
  float CF, IE, DR;
  int blockCount;
  std::string cafieMetricsFilename;
  int numTFs;

  int numRows, numCols, numRegionRows, numRegionCols;

  int numFlowsTFClassify;
  std::vector<std::vector<double> > avgTFSignal;
  std::vector<std::vector<double> > avgTFCorrected;
  vector<vector<double> > avgTFSignalSquared;
  const static int        maxTFSignalHist = 200;
  vector<vector<int> >    tfSignalHist;
  const static int        maxTFHPHist = 20;
  vector<vector<int> >    tfCallCorrect2;
  vector<vector<int> >    tfCallOver2;
  vector<vector<int> >    tfCallUnder2;

  const static int        maxTFSparklineFlows = 250;
  vector<vector<int> >    tfCallCorrect3;
  vector<vector<int> >    tfCallTotal3;

  vector<int>       tfCount;

  std::vector<float> CFbyRegion;
  std::vector<float> IEbyRegion;

  Json::Value             basecallerJson;

};


// Matches "chromosome" names from BAM to sequences in the reference.fasta file
void PopulateReferenceSequences(vector<string> &referenceSequences, const string &refFastaFilename, int nTargets, char **targetNames, const string &key)
{

  referenceSequences.resize(nTargets);

  // Iterate through the fasta file. Check each sequence name for a match to BAM.

  ifstream fasta;
  fasta.open(refFastaFilename.c_str());
  if (!fasta.is_open()) {
    printf ("Failed to open reference %s\n", refFastaFilename.c_str());
    exit(1);
  }

  char line[4096] = "";

  while (!fasta.eof()) {

    string entryName;
    string entrySequence = key;

    if ((strlen(line) <= 1) || (line[0] != '>')) {
      fasta.getline(line,4096);
      continue;
    }

    entryName = (line+1);

    while (!fasta.eof()) {
      fasta.getline(line,4096);
      if ((strlen(line) > 1) && (line[0] == '>'))
        break;

      entrySequence += line;
    }

    for (int tf = 0; tf < nTargets; tf++) {
      if (entryName == targetNames[tf]) {
        referenceSequences[tf] = entrySequence;
        break;
      }
    }
  }

  fasta.close();
}




#ifdef _DEBUG
void memstatus(void)
{
  memdump();
  dbgmemClose();
}

#endif /* _DEBUG */

void usage ()
{
  fprintf (stdout, "TFTorturer analysisDir sff bam out.json key ref.fasta\n");
  fprintf (stdout, "options:\n");
  fprintf (stdout, "   analysisDir\tDirectory containing 1.wells, processParameters.txt, and BaseCaller.json\n");
  fprintf (stdout, "   sff\t\tSFF file name containing test fragments\n");
  fprintf (stdout, "   bam\t\tBAM aligment results for the above SFF\n");
  fprintf (stdout, "   out.json\tGenerated metrics will be saved to this JSON file\n");
  fprintf (stdout, "   key\t\tTest fragment key sequence\n");
  fprintf (stdout, "   ref.fasta\tFASTA file containing TF names and sequences\n");
  fprintf (stdout, "\n");
  exit(1);
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */


  if (argc <= 6) {
    usage();
  }

  string analysisDir = argv[1];   // Needed for processParameters.txt, BaseCaller.json, 1.wells
  string sffFilename = argv[2];
  string bamFilename = argv[3];
  string jsonFilename = argv[4];  // Results go here
  string key = argv[5];

  string refFastaFilename = argv[6];


  int x = -1, y = -1, nRow = -1, nCol = -1;
  sscanf(GetProcessParam (analysisDir.c_str(), "Block"), "%d,%d,%d,%d", &x, &y, &nCol, &nRow);
  if ((nRow <= 0) || (nCol <= 0)) {
    printf ("Failed to determine chip size from processParameters.txt\n");
    exit(1);
  }
  printf ("Chip size: nRow = %d, nCol = %d\n", nRow, nCol);

  printf ("Using key %s\n", key.c_str());


  // Before doing any actual work, scan through the bam file to build a list of reads. This will help RawWells
  vector<int32_t>  xSubset;
  vector<int32_t>  ySubset;
  {
    BAMReader bamReader1(bamFilename);
    bamReader1.open();
    for (BAMReader::iterator i = bamReader1.get_iterator(); i.good(); i.next()) {
      BAMRead bamRead = i.get();

      int x = -1, y = -1;
      const char *ptr = bamRead.get_qname();
      while (*ptr && *ptr != ':')
        ptr++;
      sscanf(ptr, ":%d:%d", &y, &x);
      if ((x>=0) && (y>=0)) {
        xSubset.push_back(x);
        ySubset.push_back(y);
      }
    }
    bamReader1.close();
  }



  BAMReader bamReader(bamFilename);

  bamReader.open();
  const bam_header_t *header = bamReader.get_header();

  vector<string>  referenceSequences;
  PopulateReferenceSequences(referenceSequences, refFastaFilename, header->n_targets, header->target_name, key);

  int numTFs = header->n_targets;
  for (int tf = 0; tf < numTFs; tf++)
    printf("Target %d : %s, len=% 3d \t%s\n", tf, header->target_name[tf], header->target_len[tf], referenceSequences[tf].c_str());


  SemiOrderedSFFReader sffReader;
  sffReader.Open(sffFilename.c_str(), nRow, nCol);


  RawWells wells(analysisDir.c_str(), "1.wells", nRow, nCol);


  time_t startBasecall;
  time(&startBasecall);

  wells.SetSubsetToLoad(xSubset, ySubset);


  if (wells.OpenForRead()) {
    fprintf (stdout, "ERROR: Could not open %s/%s\n", analysisDir.c_str(), "1.wells");
    exit (1);
  }
  time_t endBasecall;
  time(&endBasecall);

  printf ("Opening wells took %1.2f seconds\n",  difftime(endBasecall,startBasecall));


  int numFlows = wells.NumFlows();
  char *flowOrder = (char *)wells.FlowOrder();
//  int flowOrderLen = strlen(flowOrder);


  // Initialize TF-specific structures

  printf("numTFs = %d, numFlows = %d\n", numTFs, numFlows);
  CafieMetricsGenerator cafieMetricsGenerator(true, numTFs, std::min(80, numFlows));

  cafieMetricsGenerator.AddBlock(analysisDir.c_str(), nRow, nCol);

  // All readers are open, now to iterate over the reads

  for (BAMReader::iterator i = bamReader.get_iterator(); i.good(); i.next()) {
    BAMRead bamRead = i.get();

    int x = -1, y = -1;
    const char *ptr = bamRead.get_qname();
    while (*ptr && *ptr != ':')
      ptr++;
    sscanf(ptr, ":%d:%d", &y, &x);


    sff_t *sffRead = sffReader.LoadXY(x,y);

    if (bamRead.get_tid() < 0) {
      sffReader.FreeXY(x,y);
      continue;
    }
    int bestTF = bamRead.get_tid();

    const WellData *wellData = wells.ReadXY(x, y);

    BAMUtils bamUtil(bamRead);
    string reference = key + bamUtil.get_tdna();
    string calls = key + bamUtil.get_qdna();

    cafieMetricsGenerator.AddElement(bestTF,wellData->flowValues,
        sff_flowgram(sffRead), numFlows, x, y, flowOrder, reference, calls);

    sffReader.FreeXY(x,y);

  }

  cafieMetricsGenerator.GenerateCafieMetrics(header, numFlows, flowOrder, jsonFilename, referenceSequences);

  sffReader.Close();
  bamReader.close();
  wells.Close();

}



