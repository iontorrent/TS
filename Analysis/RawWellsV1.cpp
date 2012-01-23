/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RawWellsV1.h"
#include "Mask.h"
#include "Separator.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include "Utils.h"
#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#endif /* not WIN32 */

#include "LinuxCompat.h"
#include "IonErr.h"
#include "dbgmem.h"

RawWellsV1::RawWellsV1(const char *_experimentPath, const char *_rawWellsName, int _rows, int _cols)
{
  rows = _rows;
  cols = _cols;
  experimentPath = _experimentPath;
  rawWellsName = _rawWellsName;

  fileMemPtr = NULL;
  fp = NULL;
  rawWellsSize = 0;

  flowValuesSize = 0;
  dataBlockSize = 0;
  hMFile = NULL;
  hFile = NULL;
  memset (&hdr, 0, sizeof(hdr));
  memset (&data, 0, sizeof(data));

}

RawWellsV1::RawWellsV1(const char *_experimentPath, const char *_rawWellsName)
{
  rows = 0;
  cols = 0;
  experimentPath = _experimentPath;
  rawWellsName = _rawWellsName;

  fileMemPtr = NULL;
  fp = NULL;
  rawWellsSize = 0;

  flowValuesSize = 0;
  dataBlockSize = 0;
  hMFile = NULL;
  hFile = NULL;
  memset (&hdr, 0, sizeof(hdr));
  memset (&data, 0, sizeof(data));
}

RawWellsV1::~RawWellsV1()
{
  if (hdr.flowOrder)
    free (hdr.flowOrder);
  if (data.flowValues)
    free (data.flowValues);
}

void RawWellsV1::CreateEmpty(int numFlows, const char *flowOrder)
{ 
  char wellFileName[MAX_PATH_LENGTH];
  hdr.flowOrder = NULL;
  int flowCnt = strlen (flowOrder);
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  fopen_s(&fp, wellFileName, "wb");
  if (fp) {
    // generate and write the header
    hdr.numFlows = numFlows; // MGD - Do I need to byte-swap???
    hdr.numWells = rows*cols;
    hdr.flowOrder = (char *)malloc(hdr.numFlows);
    int i;
    for(i=0;i<hdr.numFlows;i++)
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    fwrite(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    fwrite(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    fwrite(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);
    flowValuesSize = sizeof(float) * hdr.numFlows;
    data.flowValues = (float *)malloc(flowValuesSize);
    memset(data.flowValues, 0, flowValuesSize);
    //for(i=0;i<(int)hdr.numWells;i++) {
    for (int y=0;y<rows;y++)
      for (int x=0;x<cols;x++) {
        data.rank = 0;//data.rank = i;
        data.x = x;
        data.y = y;
        fwrite(&data, dataBlockSize, 1, fp);
        fwrite(data.flowValues, flowValuesSize, 1, fp);
      }
    rawWellsSize = ftell(fp);

    fclose(fp);
    fp = NULL;
  }

}
void RawWellsV1::CreateEmpty(int numFlows, const char *flowOrder, int _rows, int _cols)
{ 
  rows = _rows;
  cols = _cols;
  char wellFileName[MAX_PATH_LENGTH];
  if (hdr.flowOrder) {
    free (hdr.flowOrder);
  }
  hdr.flowOrder = NULL;
  int flowCnt = strlen (flowOrder);
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  fopen_s(&fp, wellFileName, "wb");
  if (fp) {
    // generate and write the header
    hdr.numFlows = numFlows; // MGD - Do I need to byte-swap???
    hdr.numWells = rows*cols;

    hdr.flowOrder = (char *)malloc(hdr.numFlows);
    int i;
    for(i=0;i<hdr.numFlows;i++)
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    fwrite(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    fwrite(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    fwrite(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);
    flowValuesSize = sizeof(float) * hdr.numFlows;
    if (data.flowValues) {
      free (data.flowValues);
    }
    data.flowValues = (float *)malloc(flowValuesSize);
    memset(data.flowValues, 0, flowValuesSize);
    //for(i=0;i<(int)hdr.numWells;i++) {
    for (int y=0;y<rows;y++)
      for (int x=0;x<cols;x++) {
        data.rank = 0;//data.rank = i;
        data.x = x;
        data.y = y;
        fwrite(&data, dataBlockSize, 1, fp);
        fwrite(data.flowValues, flowValuesSize, 1, fp);
      }
    rawWellsSize = ftell(fp);

    fclose(fp);
    fp = NULL;
  }

}
#if 0
// Compare function for qsort for Descending order
static int WellRankDescend(const void *v1, const void *v2)
{
  struct WellRank val1 = *(struct WellRank *)v1;
  struct WellRank val2 = *(struct WellRank *)v2;

  if (val1.rqs < val2.rqs)
    return 1;
  else if (val2.rqs < val1.rqs)
    return -1;
  else
    return 0;
}
#endif
// Compare function for qsort for Ascending order
static int WellRankAscend(const void *v1, const void *v2)
{
  struct WellRank val1 = *(struct WellRank *)v1;
  struct WellRank val2 = *(struct WellRank *)v2;

  if (val1.rqs > val2.rqs)
    return 1;
  else if (val2.rqs > val1.rqs)
    return -1;
  else
    return 0;
}

//
// Modifies the rank field for TF and Lib beads based on Mask and RQS score
void RawWellsV1::AddRank (Mask *mask, Separator *sep)
{
  int x;
  int y;
  int beadTF = 0;
  int beadLib = 0;
 
  //Retrieve rqs scores: stored in first 'frame' of work matrix.
  double *work = sep->GetWork();
 
  struct WellRank *rqsListTF = new struct WellRank[(*mask).GetCount(MaskTF)];
  struct WellRank *rqsListLib = new struct WellRank[(*mask).GetCount(MaskLib)];
 
  for(y=0;y<rows;y++) {
    for(x=0;x<cols;x++) {
      if ((*mask)[x+y*cols] & MaskTF && ((*mask)[x+y*cols] & MaskLive)) {
        rqsListTF[beadTF].x = x;
        rqsListTF[beadTF].y = y;
        rqsListTF[beadTF].rqs = work[x+y*cols];
        beadTF++;
      }
      if ((*mask)[x+y*cols] & MaskLib && ((*mask)[x+y*cols] & MaskLive)) {
        rqsListLib[beadLib].x = x;
        rqsListLib[beadLib].y = y;
        rqsListLib[beadLib].rqs = work[x+y*cols];
        beadLib++;
      }
    }
  }

  //sort the beads: Best to worst. Bigger is better
  qsort (rqsListTF, beadTF, sizeof(struct WellRank), WellRankAscend);
  qsort (rqsListLib, beadLib, sizeof(struct WellRank), WellRankAscend);

  // update the wells file ranking field
  for (int i=0;i<beadLib;i++) {
    WriteRank (i+1, rqsListLib[i].x, rqsListLib[i].y);
  }
  for (int i=0;i<beadTF;i++) {
    WriteRank (i+1, rqsListTF[i].x, rqsListTF[i].y);
  }
 
  delete [] rqsListTF;
  delete [] rqsListLib;
  return;
}

void RawWellsV1::WriteRank (int rank, int x, int y)
{
  // skip to proper spot in memory - this is defined as the rank at location x,y

  int64_t headerSkipBytes = sizeof(hdr.numWells) + sizeof(hdr.numFlows) + sizeof(char) * hdr.numFlows;
  int64_t dataHdrSkipBytes = sizeof(unsigned int) + 2 * sizeof(unsigned short);

  int64_t offset = headerSkipBytes;
  offset += (dataHdrSkipBytes + sizeof(float) * hdr.numFlows) * (y*cols+x);

  unsigned int *valPtr = (unsigned int *)((unsigned char *)fileMemPtr + offset);
  *valPtr = (unsigned int)rank;
  return;
}

#ifdef WIN32
void RawWellsV1::OpenForWrite()
{
  // create a memory-mapped file, much better (faster, etc) for random write access on really big files!
  char wellFileName[MAX_PATH_LENGTH];
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  hFile = CreateFile(wellFileName, OPEN_EXISTING, /*FILE_SHARE_READ | FILE_SHARE_WRITE*/ 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
  hMFile = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, 0, "__RawWellsV1File");
  fileMemPtr = MapViewOfFile(hMFile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
}
#else
void RawWellsV1::OpenForWrite()
{
  char wellFileName[MAX_PATH_LENGTH];
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  hFile = open(wellFileName, O_RDWR);
  fileMemPtr = mmap(0, rawWellsSize, PROT_READ | PROT_WRITE, MAP_SHARED, hFile, 0);
  if (fileMemPtr == MAP_FAILED) {
    perror ("mmap");
    if (errno == ENODEV)
      fprintf (stderr, "Darn!  Your filesystem does not support mmap files\n");
    fprintf (stderr, "requested allocation size = %lu\n", rawWellsSize);
    exit (errno);
  }
}
#endif /* WIN32 */

bool RawWellsV1::OpenForRead(bool wantMemMap)
{
  int n = 0; //number elements read
  char wellFileName[MAX_PATH_LENGTH];
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  fp = NULL;
  hFile = 0;
  fileMemPtr = NULL;
  hdr.flowOrder = NULL;
  if (wantMemMap)
    hFile = open(wellFileName, O_RDONLY);
  else
    fopen_s(&fp, wellFileName, "rb");
  if (fp) {
    n = fread(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    assert(n==1);
    n = fread(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    assert(n==1);
    hdr.flowOrder = (char *)malloc(sizeof(char) * hdr.numFlows);
    n = fread(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);
    assert(n==hdr.numFlows);
  } else if (hFile) {
    struct stat result;
    fstat(hFile, &result);
    rawWellsSize = result.st_size;
    fileMemPtr = mmap(0, rawWellsSize, PROT_READ, MAP_SHARED, hFile, 0);
    if (fileMemPtr) {
      unsigned char *ptr = (unsigned char *)fileMemPtr;
      memcpy(&hdr.numWells, ptr, sizeof(hdr.numWells)); ptr += sizeof(hdr.numWells);
      memcpy(&hdr.numFlows, ptr, sizeof(hdr.numFlows)); ptr += sizeof(hdr.numFlows);
      hdr.flowOrder = (char *)malloc(sizeof(char) * hdr.numFlows);
      memcpy(hdr.flowOrder, ptr, hdr.numFlows); ptr += hdr.numFlows;
    }
  } else {
    ION_ABORT("Couldn't open RawWellsV1 file: " + std::string(wellFileName));
    return (true); //indicate there was an error
  }

  flowValuesSize = sizeof(float) * hdr.numFlows;
  data.flowValues = (float *)malloc(flowValuesSize);
  dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);

  return (false); // no error
}

const WellData *RawWellsV1::ReadNext()
{
  // extract the next flowgram
  if (fread(&data, dataBlockSize, 1, fp) == 1) {
    if (fread(data.flowValues, flowValuesSize, 1, fp) == 1)
      return &data;
  }
  return NULL;
}
bool RawWellsV1::ReadNext(WellData *_data)
{
  // extract the next flowgram
  if (fread(_data, dataBlockSize, 1, fp) == 1) {
    if (_data->flowValues)
      free (_data->flowValues);
    _data->flowValues = (float *) malloc (flowValuesSize);
    if (fread(_data->flowValues, flowValuesSize, 1, fp) != 1) {
      free (_data->flowValues);
      return true;
    }
  }
  else {
    return true;
  }
  return false;
}

bool RawWellsV1::ReadNextData(WellData *_data)
{
  // extract the next flowgram
  if (fread(_data, dataBlockSize, 1, fp) == 1) {
    if (!_data->flowValues)
      _data->flowValues = (float *) malloc (flowValuesSize);
    if (fread(_data->flowValues, flowValuesSize, 1, fp) != 1) {
      if (_data->flowValues)
        free (_data->flowValues);
      return true;
    }
  } else {
    if (_data->flowValues)
      free (_data->flowValues);
    return true;
  }
  return false;
}

const WellData *RawWellsV1::ReadXY(int x, int y)
{
  int64_t headerSkipBytes = sizeof(hdr.numWells) + sizeof(hdr.numFlows) + sizeof(char) * hdr.numFlows;
  int64_t dataHdrSkipBytes = sizeof(unsigned int) + 2 * sizeof(unsigned short);

  int64_t offset = headerSkipBytes;
  offset += (dataHdrSkipBytes + sizeof(float) * hdr.numFlows) * (y*cols+x);

  if (fp) {
    fseek(fp, offset, SEEK_SET);
    if (fread(&data, dataBlockSize, 1, fp) == 1) {
      if (fread(data.flowValues, flowValuesSize, 1, fp) == 1)
        return &data;
    }
  } else if (fileMemPtr) {
    unsigned char *ptr = (unsigned char *)fileMemPtr;
    ptr += offset;
    memcpy(&data, ptr, dataBlockSize); ptr += dataBlockSize;
    memcpy(data.flowValues, ptr, flowValuesSize);
    return &data;
  }

  return NULL;

  //offset += dataHdrSkipBytes;
  //offset += sizeof(float) * i;
}

void RawWellsV1::WriteFlowgram(int i, int x, int y, float val)
{
  // skip to proper spot in memory - this is defined as the i'th flowgram value of the
  // well at location x,y

  int64_t headerSkipBytes = sizeof(hdr.numWells) + sizeof(hdr.numFlows) + sizeof(char) * hdr.numFlows;
  int64_t dataHdrSkipBytes = sizeof(unsigned int) + 2 * sizeof(unsigned short);

  int64_t offset = headerSkipBytes;
  offset += (dataHdrSkipBytes + sizeof(float) * hdr.numFlows) * (y*cols+x);
  offset += dataHdrSkipBytes;
  offset += sizeof(float) * i;
  float *valPtr = (float *)((unsigned char *)fileMemPtr + offset);
  *valPtr = (float)val;
}

void RawWellsV1::Close()
{
  if (fileMemPtr) {
#ifdef WIN32
    UnmapViewOfFile(fileMemPtr);
    CloseHandle(hMFile);
    CloseHandle(hFile);
#else
    munmap(fileMemPtr, rawWellsSize);
    close(hFile);
#endif /* WIN32 */
    fileMemPtr = NULL;
  }

  if (fp) {
    fclose(fp);
    fp = NULL;

    if (hdr.flowOrder)
      free(hdr.flowOrder);
    hdr.flowOrder = NULL;

    if (data.flowValues)
      free(data.flowValues);
    data.flowValues = NULL;
  }
}

/*
 * This is a dangerous function cause it will wind up the file pointer to the end of file.
 *  Be careful.
 */
void RawWellsV1::GetDims(int *rows, int *cols)
{
  const WellData *Wdata;
 
  //while (!ReadNext (Wdata)) {
  while ((Wdata = this->ReadNext())) {
    *rows = 1 + (int) Wdata->y;
    *cols = 1 + (int) Wdata->x;
  }
}

void RawWellsV1::AllocateMemoryForRawFlowgrams(int numFlows, int _rows, int _cols) {
  rows = _rows;
  cols = _cols;
  hdr.numFlows = numFlows;
 
  rawflowgrams = (float*)malloc(sizeof(float)*rows*cols*numFlows);

  if (!rawflowgrams) {
    printf("Can't allocate memory for wells file\n");
    exit(1);
  }

  memset(rawflowgrams, 0, sizeof(float)*rows*cols*numFlows);
}

void RawWellsV1::WriteFlowgramToMemory(int flowIdx, int x, int y, float val) {
  uint64_t offset = (uint64_t)(y*cols + x) * hdr.numFlows;
  rawflowgrams[offset + flowIdx] = val;
}

void RawWellsV1::WriteWellsFile(const char *flowOrder) {

  char wellFileName[MAX_PATH_LENGTH];
  hdr.flowOrder = NULL;
  int flowCnt = strlen (flowOrder);
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  fopen_s(&fp, wellFileName, "wb");
  if (fp) {
    // generate and write the header
    hdr.numWells = rows*cols;
    hdr.flowOrder = (char *)malloc(hdr.numFlows);
    int i;
    for(i=0;i<hdr.numFlows;i++)
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    fwrite(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    fwrite(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    fwrite(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);
    flowValuesSize = sizeof(float) * hdr.numFlows;
    uint64_t offset;
    for (int y=0;y<rows;y++)
      for (int x=0;x<cols;x++) {
        data.rank = 0;//data.rank = i;
        data.x = x;
        data.y = y;
        fwrite(&data, dataBlockSize, 1, fp);
        offset = (uint64_t)(y*cols + x)*hdr.numFlows;
        fwrite(&rawflowgrams[offset], flowValuesSize, 1, fp);
      }
    rawWellsSize = ftell(fp);

    fclose(fp);
    fp = NULL;
  }

  free(rawflowgrams);
}

void RawWellsV1::AllocateMemoryForWellMap(int numFlows, int _rows, int _cols) {
  rows = _rows;
  cols = _cols;
  hdr.numFlows = numFlows;
 
  uint64_t totalWells = rows*cols;    

  // allocate memory for well memory map
  wellMemMap.resize(rows*cols);

  for (uint64_t i = 0; i < totalWells; ++i) {
    wellMemMap[i] = NULL;
  }
}

void RawWellsV1::WriteLiveWellToMemory(int flowIdx, int x, int y, float val) {

  uint64_t wellId = y*cols + x;

  float* wellPtr = wellMemMap[wellId];
  if (wellPtr == NULL) {
    wellPtr = (float*)malloc(sizeof(float)*hdr.numFlows);
    wellMemMap[wellId] = wellPtr;
  }
        
  wellPtr[flowIdx] = val;    
}

void RawWellsV1::WriteSparseWellsFile(const char *flowOrder) {

  char wellFileName[MAX_PATH_LENGTH];
  hdr.flowOrder = NULL;
  int flowCnt = strlen (flowOrder);
  sprintf_s(wellFileName, sizeof(wellFileName), "%s/%s", experimentPath, rawWellsName);
  fopen_s(&fp, wellFileName, "wb");
  if (fp) {
    // generate and write the header
    hdr.numWells = rows*cols;
    hdr.flowOrder = (char *)malloc(hdr.numFlows);
    int i;
    for(i=0;i<hdr.numFlows;i++)
      hdr.flowOrder[i] = flowOrder[i%flowCnt];
    fwrite(&hdr.numWells, sizeof(hdr.numWells), 1, fp);
    fwrite(&hdr.numFlows, sizeof(hdr.numFlows), 1, fp);
    fwrite(hdr.flowOrder, sizeof(char), hdr.numFlows, fp);

    // write the well data, initially all zeros, to create the file
    int dataBlockSize = sizeof(unsigned int) + 2 * sizeof(unsigned short);
    flowValuesSize = sizeof(float) * hdr.numFlows;
    uint64_t offset;

    float* zeroVec = (float*)malloc(sizeof(float) * hdr.numFlows);
    memset(zeroVec, 0, sizeof(float) * hdr.numFlows);
 
    float* wellPtr = NULL;
    for (int y=0;y<rows;y++) {
      for (int x=0;x<cols;x++) {
        data.rank = 0;//data.rank = i;
        data.x = x;
        data.y = y;
        fwrite(&data, dataBlockSize, 1, fp);
        offset = (uint64_t)(y*cols + x);
        wellPtr = wellMemMap[offset];
        if(wellPtr != NULL) {
          fwrite(wellPtr, flowValuesSize, 1, fp);
          free(wellPtr);
        }
        else {
          fwrite(zeroVec, flowValuesSize, 1, fp);
        }
      }
    }
    rawWellsSize = ftell(fp);

    fclose(fp);
    fp = NULL;
    free(zeroVec);
  }

  wellMemMap.clear();
}
