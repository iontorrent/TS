/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellFileManipulation.h"
#include "Mask.h"

using namespace std;

void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr)
{
  // Get subset of wells we want to track, live only...
  vector<int> subset;
  size_t maskWells = maskPtr->H() * maskPtr->W();
  subset.reserve(maskWells);
  for (size_t i = 0; i < maskWells; i++) {
    if (maskPtr->Match(i, MaskLive)) {
      subset.push_back(i);
    }
  }
  rawWells.SetSubsetToWrite(subset);
}

void SetChipTypeFromWells(RawWells &rawWells)
{
  if (rawWells.OpenMetaData())   //Use chip type stored in the wells file
  {
    if (rawWells.KeyExists("ChipType"))
    {
      string chipType;
      rawWells.GetValue("ChipType", chipType);
      ChipIdDecoder::SetGlobalChipId(chipType.c_str());
    }
  }
}


void GetMetaDataForWells(char *dirExt, RawWells &rawWells, const char *chipType)
{
  const char * paramsToKeep[] = {"Project","Sample","Library","Start Time","End Time","Experiment Name","Pending Run Short ID","User Name","Serial Number","Gain","Noise","Datacollect version","LiveView version","Firmware version","FPGA version","Driver version","Script version","Board version","Board serial","Kernel Build","PreRun","Frequency","Oversample","Frame Time","Num Frames","Ref Electrode","DAC","VREFS","Cal Chip Hist","Cal Chip High/Low/InRange","Image Map","Cycles","Flows","ChipType","RunType","ChipBarCode","SeqBarCode","LibBarCode","AnalyzeEarly","AutoAnalyze","AutoAnalysisName","PreBeadFind","PostBeadFind","LibraryKeySequence","AutoSettleCalibrate","CalibratePassed","ChipTemperature","PGMTemperature","PGMPressure","WetLoad","AdvScriptFeatures","Advanced User","barcodeId","R1pH","R2pH","R3pH","R4pH","W1pH","W2pH","RawTracesOK","HardDisk","HardDisk","User Notes"};
  std::string logFile = getExpLogPath(dirExt);
  char* paramVal = NULL;
  for (size_t pIx = 0; pIx < sizeof(paramsToKeep)/sizeof(char *); pIx++)
  {
    if ((paramVal = GetExpLogParameter(logFile.c_str(), paramsToKeep[pIx])) != NULL)
    {
      string value = paramVal;
      size_t pos = value.find_last_not_of("\n\r \t");
      if (pos != string::npos)
      {
        value = value.substr(0,pos+1);
      }
      rawWells.SetValue(paramsToKeep[pIx], value);
    }
  }
  rawWells.SetValue("ChipType", chipType);
}

void CopyTmpWellFileToPermanent(CommandLineOpts &clo, char *experimentName)
{
  // defaults moved here because never changed

  static char *wellfileIndex = "1";
  static char *wellfileExt = "wells";

  if (clo.LOCAL_WELLS_FILE && !clo.USE_RAWWELLS)
  {
    char wellFileName[MAX_PATH_LENGTH];
    sprintf(wellFileName, "%s/%s.%s", experimentName, wellfileIndex, wellfileExt);
    CopyFile(clo.tmpWellsFile, wellFileName);
  }
}


void MakeNewTmpWellsFile(CommandLineOpts &clo, char *experimentName)
{
  if (clo.wellsFilePath[0] == '\0')
  {
    if (clo.LOCAL_WELLS_FILE)
    {
      char fTemplate[256] = { 0 };
      //Utils:ClearStaleWellsFile() is sensitive to temp well filename format
      sprintf(fTemplate, "/tmp/well_%d_XXXXXX", getpid());
      int tmpFH = mkstemp(fTemplate);
      if (tmpFH > 0)
        close(tmpFH);
      else
        exit(EXIT_FAILURE);

      strcpy(clo.tmpWellsFile, fTemplate);
      strcpy(clo.wellsFilePath, "/tmp");
      strcpy(clo.wellsFileName, basename(fTemplate));
    }
    else
    {
      strcpy(clo.wellsFilePath, experimentName);
    }
  }
}

void CleanupTmpWellsFile(CommandLineOpts &clo)
{
  //Cleanup
  //Copy wells file from temporary, local file to permanent; remove temp file
  //Copy temp wells file moved to pre-cafie code.
  if (clo.LOCAL_WELLS_FILE && !clo.USE_RAWWELLS)
  {
    unlink(clo.tmpWellsFile);
  }
}
