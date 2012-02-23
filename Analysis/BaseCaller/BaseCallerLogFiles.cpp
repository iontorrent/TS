/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BaseCaller.h"

using namespace std;



void BaseCaller::saveBaseCallerJson(const char *basecaller_output_directory)
{
  string filename = string(basecaller_output_directory) + "/BaseCaller.json";
  ofstream out(filename.c_str(), ios::out);
  if (out.good())
    out << basecallerJson.toStyledString();
  else
    ION_WARN( "Unable to write JSON file " + filename )
}




// generateCafieRegionsFile - generate the cafieRegions.txt file containing all cf/ie/dr estimates

void BaseCaller::generateCafieRegionsFile(const char *basecaller_output_directory)
{

  // Open CAFIE region metrics file for writing
  FILE *crfp = NULL;
  char fileName[512];
  snprintf(fileName, 512, "%s/%s", basecaller_output_directory, "cafieRegions.txt");
  fopen_s(&crfp, fileName, "wb");
  if (!crfp)
    return;

//  int cfiedrRegions = clo->cfiedrRegionsX * clo->cfiedrRegionsY;

  fprintf(crfp, "Width = %d\n", cols);
  fprintf(crfp, "Height = %d\n", rows);

  // Save cf/ie/dr values at each region for library fragments

  fprintf(crfp, "TF = LIB\n");
  fprintf(crfp, "Region CF = ");
  for (int y = 0; y < clo->cfe_control.cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfe_control.cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", cf[y + x * clo->cfe_control.cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }
  fprintf(crfp, "Region IE = ");
  for (int y = 0; y < clo->cfe_control.cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfe_control.cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", ie[y + x * clo->cfe_control.cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }
  fprintf(crfp, "Region DR = ");
  for (int y = 0; y < clo->cfe_control.cfiedrRegionsY; y++) {
    for (int x = 0; x < clo->cfe_control.cfiedrRegionsX; x++)
      fprintf(crfp, "%.5lf ", droop[y + x * clo->cfe_control.cfiedrRegionsY]);
    fprintf(crfp, "\n");
  }

  fclose(crfp);

}

//


void BaseCaller::OpenWellStatFile()
{
  wellStatFileFP = NULL;
  if (clo->sys_context.wellStatFile == NULL)
    return;

  fopen_s(&wellStatFileFP, clo->sys_context.wellStatFile, "wb");
  if (!wellStatFileFP) {
    perror(clo->sys_context.wellStatFile);
    return;
  }
  fprintf(wellStatFileFP,
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
      "col", "row", "isTF", "isLib", "isDud", "isAmbg", "nCall",
      "cf", "ie", "dr", "keySNR", "keySD", "keySig", "oneSig",
      "zeroSig", "ppf", "isClonal", "medAbsRes", "multiplier");
}

void BaseCaller::WriteWellStatFileEntry(MaskType bfType, int *keyFlow, int keyFlowLen, int x, int y, vector<weight_t> &keyNormSig,
    int numCalls, double cf, double ie, double dr, double multiplier, double ppf, bool clonal, double medianAbsCafieResidual)
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
    out << setw( 8) << className[iClass];
    out << setw( 8) << classKeyBases[iClass];
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
    out << className[iClass];
    out << separator << classKeyBases[iClass];
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



