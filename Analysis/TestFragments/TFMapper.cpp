/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <math.h>
#include <vector>
#include <fstream>

#include "dbgmem.h"
#include "IonVersion.h"
#include "OptArgs.h"
#include "json/json.h"
#include "SamUtils/BAMReader.h"
#include "SamUtils/BAMUtils.h"
#include "SamUtils/types/BAMRead.h"
#include "sam_header.h"
using namespace std;


#define MAX_READLEN       100
#define MAX_HP            8
#define MAX_IONOGRAM_LEN  80
#define minTFCount        1000

/////////////////////////////////////////////////////////////////////////////////////
// Below is a collection of small classes dedicated to calculating and reporting metrics or metric groups

class MetricGeneratorHPAccuracy {
public:
  MetricGeneratorHPAccuracy() {
    for(int hp = 0; hp < MAX_HP; hp++) {
      hpCount[hp] = 0;
      hpAccuracy[hp] = 0;
    }
  }

  void AddElement(int refNmer, int calledNmer) {
    if (refNmer >= MAX_HP)
      return;
    hpCount[refNmer]++;
    if (refNmer == calledNmer)
      hpAccuracy[refNmer]++;
  }

  void PrintHPAccuracy(Json::Value& currentTFJson) {
    for(int hp = 0; hp < MAX_HP; hp++) {
      currentTFJson["Per HP accuracy NUM"][hp] = hpAccuracy[hp];
      currentTFJson["Per HP accuracy DEN"][hp] = hpCount[hp];
      currentTFJson["Raw HP SNR"][hp] = 0;
      currentTFJson["Corrected HP SNR"][hp] = 0;
    }
  }

  int hpAccuracy[MAX_HP];
  int hpCount[MAX_HP];
};



class MetricGeneratorQualityHistograms {
public:
  MetricGeneratorQualityHistograms() {
    Q10.assign(MAX_READLEN+1,0);
    Q17.assign(MAX_READLEN+1,0);
    accumulatedQ10 = 0;
    accumulatedQ17 = 0;
    count = 0;
  }

  void AddElement(int q10readlen, int q17readlen) {
    Q10[max(0,min(MAX_READLEN,q10readlen))]++;
    Q17[max(0,min(MAX_READLEN,q17readlen))]++;
    accumulatedQ10 += q10readlen;
    accumulatedQ17 += q17readlen;
    count++;
  }

  void PrintMetrics(Json::Value& currentTFJson) {

    int Q10_50 = 0;
    int Q17_50 = 0;
    for(int readLen = 50; readLen <= MAX_READLEN; readLen++) {
      Q10_50 += Q10[readLen];
      Q17_50 += Q17[readLen];
    }

    currentTFJson["Q10 Mean"] = accumulatedQ10 / count;
    currentTFJson["Q17 Mean"] = accumulatedQ17 / count;
    currentTFJson["50Q10"] = Q10_50;
    currentTFJson["50Q17"] = Q17_50;

    for (int idx = 0; idx <= MAX_READLEN; idx++) {
      currentTFJson["Q10"][idx] = Q10[idx];
      currentTFJson["Q17"][idx] = Q17[idx];
    }
  }

private:
  vector<int> Q10;
  vector<int> Q17;
  double accumulatedQ10;
  double accumulatedQ17;
  int count;
};



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

  void AddElement (uint16_t *corValues, const char *Key, const string& flowOrder)
  {
    int numFlowsPerCycle = flowOrder.length();
    count++;
    for (int iFlow = 0; iFlow < 8; iFlow++) {
      char nuc = flowOrder[iFlow%numFlowsPerCycle];
      if (*Key == nuc) { // Onemer
        onemerFirstMoment[nuc&7] += corValues[iFlow];
        onemerSecondMoment[nuc&7] += corValues[iFlow]* corValues[iFlow];
        Key++;
      } else {  // Zeromer
        zeromerFirstMoment[nuc&7] += corValues[iFlow];
        zeromerSecondMoment[nuc&7] += corValues[iFlow] * corValues[iFlow];
      }
    }
  }
  void PrintSNR(Json::Value& currentTFJson) {
    double SNRx[8];
    for(int idx = 0; idx < 8; idx++) { // only care about the first 3, G maybe 2-mer etc
      double mean0 = zeromerFirstMoment[idx] / count;
      double mean1 = onemerFirstMoment[idx] / count;
      double var0 = zeromerSecondMoment[idx] / count - mean0*mean0;
      double var1 = onemerSecondMoment[idx] / count - mean1*mean1;
      double avgStdev = (sqrt(var0) + sqrt(var1)) / 2.0;
      SNRx[idx] = 0;
      if (avgStdev > 0.0)
        SNRx[idx] = (mean1-mean0) / avgStdev;
    }
    double SNR = (SNRx['A'&7] + SNRx['C'&7] + SNRx['T'&7]) / 3.0;

    currentTFJson["System SNR"] = SNR;
  }

private:
  int count;
  double zeromerFirstMoment[8];
  double zeromerSecondMoment[8];
  double onemerFirstMoment[8];
  double onemerSecondMoment[8];
};



class MetricGeneratorAvgIonogram {
public:
  MetricGeneratorAvgIonogram () {
    count = 0;
    avgTFCorrected.assign(MAX_IONOGRAM_LEN, 0.0);
  }

  void AddElement(uint16_t *corValues, int numFlows) {
    for (int iFlow = 0; (iFlow < MAX_IONOGRAM_LEN) && (iFlow < numFlows); iFlow++)
      avgTFCorrected[iFlow] += corValues[iFlow] / 100.0;
    count++;
  }

  void PrintIonograms(Json::Value& currentTFJson) {
    for (int iFlow = 0; iFlow < MAX_IONOGRAM_LEN; iFlow++) {
      currentTFJson["Avg Ionogram NUM"][iFlow] = 0;
      currentTFJson["Avg Ionogram DEN"][iFlow] = count;
      currentTFJson["Corrected Avg Ionogram NUM"][iFlow] = avgTFCorrected[iFlow];
      currentTFJson["Corrected Avg Ionogram DEN"][iFlow] = count;
    }
  }

private:
  int count;
  vector<double> avgTFCorrected;
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
  fprintf (stdout, "TFMapper - Basic metrics calculation for classified test fragment reads\n");
  fprintf (stdout, "options:\n");
  fprintf (stdout, "   -h,--help\tPrint this help information and exit.\n");
  fprintf (stdout, "   --bam\tBAM file containing TF classification results (required).\n");
  fprintf (stdout, "   --ref\tFASTA file containing a reference list of TF sequences (required).\n");
  fprintf (stdout, "   --output-json\tWrite JSON file containing the results (default: TFStats.json).\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "usage:\n");
  fprintf (stdout, "   TFMapper [options] --bam BAM_FILENAME --ref FASTA_FILENAME\n");
  fprintf (stdout, "\n");
  return 0;
}

int main(int argc, const char *argv[])
{
#ifdef _DEBUG
  atexit(memstatus);
  dbgmemInit();
#endif /* _DEBUG */

  printf ("%s - %s-%s (%s)\n", argv[0], IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());

  string bamInputFilename;
  string fastaInputFilename;
  string jsonOutputFilename;
  bool help;

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(bamInputFilename,    "",             '-',  "bam");
  opts.GetOption(fastaInputFilename,  "",             '-',  "ref");
  opts.GetOption(jsonOutputFilename,  "TFStats.json", '-',  "output-json");
  opts.GetOption(help,                "false",        'h',  "help");
  opts.CheckNoLeftovers();

  if (help || bamInputFilename.empty() || fastaInputFilename.empty())
    return showHelp();


  // Parse BAM header

  BAMReader bamReader(bamInputFilename);
  bamReader.open();
  bam_header_t *header = (bam_header_t *)bamReader.get_header_ptr();

  int numFlows = 0;
  string flowOrder;
  string key;

  if (header->l_text >= 3) {
    if (header->dict == 0)
      header->dict = sam_header_parse2(header->text);
    int nEntries = 0;
    char **tmp = sam_header2list(header->dict, "RG", "FO", &nEntries);
    if (nEntries) {
      flowOrder = tmp[0];
      numFlows = flowOrder.length();
    }
    if (tmp)
      free(tmp);
    nEntries = 0;
    tmp = sam_header2list(header->dict, "RG", "KS", &nEntries);
    if (nEntries) {
      key = tmp[0];
    }
    if (tmp)
      free(tmp);
  }

  if (numFlows <= 0) {
    fprintf(stderr, "[TFMapper] Could not retrieve flow order from FO BAM tag. SFF-specific tags absent?\n");
    exit(1);
  }
  if (key.empty()) {
    fprintf(stderr, "[TFMapper] Could not retrieve key sequence from KS BAM tag. SFF-specific tags absent?\n");
    exit(1);
  }
  //printf("Retrieved flow order from bam: %s (%d)\n", flowOrder.c_str(), numFlows);
  //printf("Retrieved key from bam: %s\n", key.c_str());


  // Retrieve test fragment sequences

  vector<string>  referenceSequences;
  PopulateReferenceSequences(referenceSequences, fastaInputFilename, header->n_targets, header->target_name, string(""));


  //  Process the BAM reads and generate metrics

  int numTFs = header->n_targets;
  vector<int>     TFCount(numTFs,0);
  MetricGeneratorQualityHistograms  metricGeneratorQualityHistograms[numTFs];
  MetricGeneratorHPAccuracy         metricGeneratorHPAccuracy[numTFs];
  MetricGeneratorSNR                metricGeneratorSNR[numTFs];
  MetricGeneratorAvgIonogram        metricGeneratorAvgIonogram[numTFs];

  for (BAMReader::iterator i = bamReader.get_iterator(); i.good(); i.next()) {

    BAMRead bamRead = i.get();
    int bestTF = bamRead.get_tid();
    if (bestTF < 0)
      continue;
    BAMUtils bamUtil(bamRead);
    TFCount[bestTF]++;

    // Extract flowspace signal from FZ BAM tag

    uint16_t *bam_flowgram = NULL;
    uint8_t *fz = bam_aux_get(bamRead.get_bam_ptr(), "FZ");
    if (fz != NULL) {
      if (fz[0] == (uint8_t)'B' && fz[1] == (uint8_t)'S' && *((uint32_t *)(fz+2)) == (uint32_t)numFlows)
        bam_flowgram = (uint16_t *)(fz+6);
    }
    if (bam_flowgram == NULL) {
      fprintf(stderr, "[TFMapper] Could not retrieve flow signal from FZ BAM tag. SFF-specific tags absent?\n");
      exit(1);
    }


    // Use alignments to generate "synchronized" flowspace reference and read ionograms
    // TODO: Do proper flowspace alignment

    string genome = key + bamUtil.get_tdna();
    string calls = key + bamUtil.get_qdna();

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
        while ((calls[iBase] != flowOrder[numFlowsRead]) && (numFlowsRead < numFlows))
          numFlowsRead++;
        if (numFlowsRead < numFlows)
          readIonogram[numFlowsRead]++;
      }

      if (genome[iBase] != '-') {

        if (genome[iBase] != gC) {
          // Since a new homopolymer begins, need to drop off the old one
          while ((gC != flowOrder[numFlowsRef]) && (numFlowsRef < numFlows)) {
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

    int validFlows = min(numFlowsRef, numFlowsRead);


    metricGeneratorSNR[bestTF].AddElement(bam_flowgram ,key.c_str(), flowOrder);
    metricGeneratorAvgIonogram[bestTF].AddElement(bam_flowgram, numFlows);
    metricGeneratorQualityHistograms[bestTF].AddElement(bamUtil.get_phred_len(10),bamUtil.get_phred_len(17));
    for (int iFlow = 0; iFlow < validFlows-20; iFlow++)
      metricGeneratorHPAccuracy[bestTF].AddElement(refIonogram[iFlow],readIonogram[iFlow]);
  }


  // Save stats to a json file

  Json::Value outputJson(Json::objectValue);

  for(int i = 0; i < numTFs; i++) {
    if (TFCount[i] < minTFCount)
      continue;

    Json::Value currentTFJson(Json::objectValue);
    currentTFJson["TF Name"] = header->target_name[i];
    currentTFJson["TF Seq"] = referenceSequences[i];
    currentTFJson["Num"] = TFCount[i];
    currentTFJson["Top Reads"] = Json::Value(Json::arrayValue); // Obsolete

    metricGeneratorSNR[i].PrintSNR(currentTFJson);
    metricGeneratorHPAccuracy[i].PrintHPAccuracy(currentTFJson);
    metricGeneratorQualityHistograms[i].PrintMetrics(currentTFJson);
    metricGeneratorAvgIonogram[i].PrintIonograms(currentTFJson);

    outputJson[header->target_name[i]] = currentTFJson;
  }

  bamReader.close();  // Closing invalidates the header pointers

  if (!jsonOutputFilename.empty()) {
    ofstream out(jsonOutputFilename.c_str(), ios::out);
    if (out.good())
      out << outputJson.toStyledString();
  }

  return 0;
}



