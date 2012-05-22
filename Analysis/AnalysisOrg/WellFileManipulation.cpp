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
  const char * paramsToKeep[] = {"Project","Sample","Start Time","Experiment Name","User Name","Serial Number","Oversample","Frame Time", "Num Frames", "Cycles", "Flows", "LibraryKeySequence", "ChipTemperature", "PGMTemperature", "PGMPressure","W2pH","W1pH","Cal Chip High/Low/InRange"};
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



void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &clo,
                                int num_fb,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType)
{
  // set up wells data structure
  MemUsage ("BeforeWells");
  int flowChunk = min (clo.bkg_control.saveWellsFrequency*num_fb, numFlows);
  //rawWells.SetFlowChunkSize(flowChunk);
  rawWells.SetCompression (3);
  rawWells.SetRows (numRows);
  rawWells.SetCols (numCols);
  rawWells.SetFlows (numFlows);
  rawWells.SetFlowOrder (clo.flow_context.flowOrder); // 6th duplicated code
  SetWellsToLiveBeadsOnly (rawWells,maskPtr);
  // any model outputs a wells file of this nature
  GetMetaDataForWells (clo.sys_context.dat_source_directory,rawWells,chipType);
  rawWells.SetChunk (0, rawWells.NumRows(), 0, rawWells.NumCols(), 0, flowChunk);
  rawWells.OpenForWrite();
  MemUsage ("AfterWells");
}


void IncrementalWriteWells (RawWells &rawWells,int flow, bool last_flow,int saveWellsFrequency,int num_fb, int numFlows)
{
  int testWellFrequency = saveWellsFrequency*num_fb; // block size
  if ( ( (flow+1) % (testWellFrequency) == 0 && (flow != 0))  || (flow+1) >= numFlows || last_flow) //@TODO: this extends logic in CheckFlowForWrite, perhaps...
  {
    fprintf (stdout, "Writing incremental wells at flow: %d\n", flow);
    MemUsage ("BeforeWrite");
    rawWells.WriteWells();
    rawWells.SetChunk (0, rawWells.NumRows(), 0, rawWells.NumCols(), flow+1, min (testWellFrequency,numFlows- (flow+1)));
    MemUsage ("AfterWrite");
  }
}


