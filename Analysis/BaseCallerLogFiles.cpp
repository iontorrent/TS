/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BaseCaller.h"

using namespace std;



void BaseCaller::saveBaseCallerJson(const char *experimentName)
{
  string filename = string(experimentName) + "/BaseCaller.json";
  ofstream out(filename.c_str(), ios::out);
  if (out.good())
    out << basecallerJson.toStyledString();
  else
    ION_WARN( "Unable to write JSON file " + filename )
}


void BaseCaller::generateTFTrackingFile(char *experimentName)
{

  //  Create tracking file to record row,col of discovered TFs
  TFTracker tracker(experimentName);

  for (map<int,int>::iterator I = readClass.begin(); I != readClass.end(); I++) {
    if (I->second >= (numTFs+1))
      continue;
    int x = I->first % cols;
    int y = (I->first - x) / cols;
    tracker.Add(y, x, tfInfo[I->second - 1].name);
  }

  tracker.Close();
}


// generateCafieRegionsFile - generate the cafieRegions.txt file containing all cf/ie/dr estimates

void BaseCaller::generateCafieRegionsFile(const char *experimentName)
{

  // Open CAFIE region metrics file for writing
  FILE *crfp = NULL;
  char fileName[512];
  snprintf(fileName, 512, "%s/%s", experimentName, "cafieRegions.txt");
  fopen_s(&crfp, fileName, "wb");
  if (!crfp)
    return;

//  int cfiedrRegions = clo->cfiedrRegionsX * clo->cfiedrRegionsY;

  fprintf(crfp, "Width = %d\n", cols);
  fprintf(crfp, "Height = %d\n", rows);

  // Save cf/ie/dr values at each region for library fragments

  fprintf(crfp, "TF = LIB\n");
  fprintf(crfp, "Region CF = ");
  for (int y = 0; y < clo->cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", cf[y + x * clo->cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }
  fprintf(crfp, "Region IE = ");
  for (int y = 0; y < clo->cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", ie[y + x * clo->cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }
  fprintf(crfp, "Region DR = ");
  for (int y = 0; y < clo->cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", droop[y + x * clo->cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }

  fclose(crfp);

}

//


// generateCafieMetricsFile - generate the cafieMetrics.txt file containing useful stats about test fragments

void BaseCaller::generateCafieMetricsFile(const char *experimentName)
{

  //  CAFIE Metrics file
  FILE *cmfp = NULL;
  char fileName[512];
  snprintf(fileName, 512, "%s/%s", experimentName, "cafieMetrics.txt");
  fopen_s(&cmfp, fileName, "wb");
  if (!cmfp)
    return;

  double system_cf = 0.0, system_ie = 0.0, system_dr = 0.0;
  for (int r = 0; r < numRegions; r++) {
    system_cf += cf[r] / numRegions;
    system_ie += ie[r] / numRegions;
    system_dr += droop[r] / numRegions;
  }

  for (int tf = 0; tf < numTFs; tf++) {
    if (TFCount[tf] <= clo->minTFCount)
      continue;

    // show raw Ionogram
    fprintf(cmfp, "TF = %s\n", tfInfo[tf].name);
    fprintf(cmfp, "Avg Ionogram = ");
    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
      fprintf(cmfp, "%.2lf ", avgTFSignal[tf][iFlow] / TFCount[tf]);
    fprintf(cmfp, "\n");

    // show a bunch 'O stats
    fprintf(cmfp, "Estimated TF = %s\n", tfInfo[tf].name);
    fprintf(cmfp, "CF = %.5lf\n", system_cf);
    fprintf(cmfp, "IE = %.5lf\n", system_ie);
    fprintf(cmfp, "Signal Droop = %.5lf\n", system_dr);
    fprintf(cmfp, "Error = %.4lf\n", (double)0.0);
    fprintf(cmfp, "Count = %d\n", TFCount[tf]);

    fprintf(cmfp, "Corrected Avg Ionogram = ");
    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
      fprintf(cmfp, "%.2lf ", avgTFCorrected[tf][iFlow] / TFCount[tf]);
    fprintf(cmfp, "\n");
  }

  // calculate & print to cafieMetrics file the "system cf/ie/dr"
  // its just an avg of the regional estimates
  fprintf(cmfp, "Estimated System CF = %.5lf\n", system_cf);
  fprintf(cmfp, "Estimated System IE = %.5lf\n", system_ie);
  fprintf(cmfp, "Estimated System Signal Droop = %.5lf\n", system_dr);

  fclose(cmfp);


  // New stuff: Output TF-related data into the json structure

  int iTFPresent = 0;
  for (int tf = 0; tf < numTFs; tf++) {
    if (TFCount[tf] <= clo->minTFCount)
//    if (TFCount[tf] <= 1)
      continue;

    Json::Value tfJson;
    tfJson["name"] = tfInfo[tf].name;
    tfJson["count"] = TFCount[tf];

    string tfFlowOrder;
    tfFlowOrder.resize(numFlowsTFClassify);

    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++) {
      tfJson["signalCorrected"][iFlow] = avgTFCorrected[tf][iFlow] / TFCount[tf];

      double signal = avgTFSignal[tf][iFlow] / TFCount[tf];
      tfJson["signalMean"][iFlow] = signal;
      tfJson["signalStd"][iFlow] = avgTFSignalSquared[tf][iFlow] / TFCount[tf] - signal * signal;

      tfFlowOrder[iFlow] = flowOrder[iFlow % flowOrder.length()];

    }
    tfJson["flowOrder"] = tfFlowOrder;

    for (int idx = 0; idx < numFlowsTFClassify * maxTFSignalHist; idx++)
      tfJson["signalHistogram"][idx] = tfSignalHist[tf][idx];

    for (int hp = 0; hp < 4*maxTFHPHist; hp++) {
      tfJson["tfCallCorrect"][hp] = tfCallCorrect[tf][hp];
      tfJson["tfCallOver"][hp] = tfCallOver[tf][hp];
      tfJson["tfCallUnder"][hp] = tfCallUnder[tf][hp];
      tfJson["tfCallCorrect2"][hp] = tfCallCorrect2[tf][hp];
      tfJson["tfCallOver2"][hp] = tfCallOver2[tf][hp];
      tfJson["tfCallUnder2"][hp] = tfCallUnder2[tf][hp];
    }

    for (int iFlow = 0; iFlow < maxTFSparklineFlows; iFlow++) {
      tfJson["tfCallCorrect3"][iFlow] = tfCallCorrect3[tf][iFlow];
      tfJson["tfCallTotal3"][iFlow] = tfCallTotal3[tf][iFlow];
      if (iFlow < tfInfo[tf].flows)
        tfJson["tfCallHP3"][iFlow] = tfInfo[tf].Ionogram[iFlow];
      else
        tfJson["tfCallHP3"][iFlow] = 0;
    }

    DPTreephaser dpTreephaser(flowOrder.c_str(), numFlows, 8);

    if (clo->basecaller == "dp-treephaser")
      dpTreephaser.SetModelParameters(system_cf, system_ie, system_dr);
    else
      dpTreephaser.SetModelParameters(system_cf, system_ie, 0); // Adaptive normalization

    BasecallerRead read;
    read.numFlows = numFlows;
    read.solution.assign(numFlows, 0);
    read.prediction.assign(numFlows, 0);
    for (int iFlow = 0; (iFlow < tfInfo[tf].flows) && (iFlow < numFlows); iFlow++)
      read.solution[iFlow] = (char)tfInfo[tf].Ionogram[iFlow];

    dpTreephaser.Simulate3(read, numFlows);

    for (int iFlow = 0; iFlow < numFlowsTFClassify; iFlow++)
      tfJson["signalExpected"][iFlow] = read.prediction[iFlow];

    basecallerJson["TestFragments"][iTFPresent] = tfJson;
    iTFPresent++;

  }

}




void BaseCaller::OpenWellStatFile()
{
  wellStatFileFP = NULL;
  if (clo->wellStatFile == NULL)
    return;

  fopen_s(&wellStatFileFP, clo->wellStatFile, "wb");
  if (!wellStatFileFP) {
    perror(clo->wellStatFile);
    return;
  }
  fprintf(wellStatFileFP,
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
      "col", "row", "isTF", "isLib", "isDud", "isAmbg", "nCall",
      "cf", "ie", "dr", "keySNR", "keySD", "keySig", "oneSig",
      "zeroSig", "ppf", "isClonal", "medAbsRes", "multiplier");
}

void BaseCaller::WriteWellStatFileEntry(MaskType bfType, int *keyFlow, int keyFlowLen, int x, int y, vector<weight_t> &keyNormSig, int numCalls, double cf, double ie, double dr, double multiplier, double ppf, bool clonal, double medianAbsCafieResidual)
{
  if (!wellStatFileFP)
    return;

  int isTF        = (bfType == MaskTF)        ? 1 : 0;
  int isLib       = (bfType == MaskLib)       ? 1 : 0;
  int isDud       = (bfType == MaskDud)       ? 1 : 0;
  int isAmbiguous = (bfType == MaskAmbiguous) ? 1 : 0;
  int isClonal    = clonal                    ? 1 : 0;

  // Key statistics
  double zeroMerSig, zeroMerSD, oneMerSig, oneMerSD, keySig, keySD;
  double keySNR = KeySNR(keyNormSig, keyFlow, keyFlowLen-1, &zeroMerSig, &zeroMerSD, &oneMerSig, &oneMerSD, &keySig, &keySD);

  // Other things to consider adding:
  //   metric of signal in flows after key

  // Write a line of results - make sure this stays in sync with the header line written by writeWellStatFileHeader()
  fprintf(
      wellStatFileFP,
      "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%1.4f\t%1.4f\t%1.5f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%d\t%1.3f\t%1.3f\n",
      x, y, isTF, isLib, isDud, isAmbiguous, numCalls,
      cf, ie, dr, keySNR, keySD, keySig, oneMerSig,
      zeroMerSig, ppf, isClonal, medianAbsCafieResidual, multiplier);
}



void BaseCaller::writePrettyText(std::ostream &out)
{
  // Header
  out << setw( 8) << "class";
  out << setw( 8) << "key";
  out << setw(12) << "polyclonal";
  out << setw(12) << "highPPF";
  out << setw( 9) << "zero";
  out << setw( 9) << "short";
  out << setw( 9) << "badKey";
  out << setw( 9) << "highRes";
  out << setw( 9) << "valid";
  out << endl;

  for (int iClass = 0; iClass < numClasses; iClass++) {
    if (classCountTotal[iClass] == 0)
      continue;
    out << setw( 8) << className[iClass];
    out << setw( 8) << classKey[iClass];
    out << setw(12) << classCountPolyclonal[iClass];
    out << setw(12) << classCountHighPPF[iClass];
    out << setw( 9) << classCountZeroBases[iClass];
    out << setw( 9) << classCountTooShort[iClass];
    out << setw( 9) << classCountFailKeypass[iClass];
    out << setw( 9) << classCountHighResidual[iClass];
    out << setw( 9) << classCountValid[iClass];
    out << endl;
  }
}

void BaseCaller::writeTSV(char *filename)
{
  if (!filename)
    return;

  ofstream out;
  out.open(filename);

  string separator("\t");

  // Header
  out << "class";
  out << separator << "key";
  out << separator << "polyclonal";
  out << separator << "highPPF";
  out << separator << "zero";
  out << separator << "short";
  out << separator << "badKey";
  out << separator << "highRes";
  out << separator << "valid";
  out << endl;

  for (int iClass = 0; iClass < numClasses; iClass++) {
    if (classCountTotal[iClass] == 0)
      continue;
    out << className[iClass];
    out << separator << classKey[iClass];
    out << separator << classCountPolyclonal[iClass];
    out << separator << classCountHighPPF[iClass];
    out << separator << classCountZeroBases[iClass];
    out << separator << classCountTooShort[iClass];
    out << separator << classCountFailKeypass[iClass];
    out << separator << classCountHighResidual[iClass];
    out << separator << classCountValid[iClass];
    out << endl;
  }
  out.close();
}



