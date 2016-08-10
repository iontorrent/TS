/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RAWWELLSV1_H
#define RAWWELLSV1_H

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif /* WIN32 */
#include "Mask.h"
#include <stdio.h>
#include "RawWells.h"
	
class RawWellsV1 {
 public:
  RawWellsV1(const char *experimentPath, const char *rawWellsName, int rows, int cols);
  RawWellsV1(const char *experimentPath, const char *rawWellsName);
  virtual ~RawWellsV1();
  void CreateEmpty(int numFlows, const char *flowOrder);
  void CreateEmpty(int numFlows, const char *flowOrder, int rows, int cols);
  void OpenForWrite();
  void WriteFlowgram(int i, int x, int y, float val);
  bool OpenForRead(bool wantMemMap = false);
  void Close();
  const WellData *ReadNext();
  bool ReadNext(WellData *_data);
  bool ReadNextData(WellData *_data); // similar to above, but with way less memory thrashing
  const WellData *ReadXY(int x, int y);
  unsigned int NumWells() {return hdr.numWells;}
  unsigned int NumFlows() {return hdr.numFlows;}
  void WriteRank (int rank, int x, int y);
  unsigned int NumRows() { return rows; }
  unsigned int NumCols() { return cols; }
  char * FlowOrder() { return hdr.flowOrder; }
  void GetDims(int *rows, int *cols);
  void AllocateMemoryForRawFlowgrams(int numFlows, int rows, int cols); 
  void AllocateMemoryForWellMap(int numFlows, int _rows, int _cols);
  void WriteFlowgramToMemory(int flowIdx, int x, int y, float val);
  void WriteLiveWellToMemory(int flowIdx, int x, int y, float val);
  void WriteWellsFile(const char *flowOrder);
  void WriteSparseWellsFile(const char *flowOrder);
 protected:
  WellHeader hdr;
  WellData data;
#ifdef WIN32
  HANDLE hFile;
  HANDLE hMFile;
#else
  int hFile;
  void * hMFile;
#endif
  void *fileMemPtr;
  const char *rawWellsName;
  const char *experimentPath;
  FILE *fp;
  int dataBlockSize;
  int rows;
  int cols;
  int flowValuesSize;
  uint64_t rawWellsSize;
  float* rawflowgrams;
  std::vector<float*> wellMemMap;

 private:
  RawWellsV1(); // not implemented, don't call!
};

#endif // RAWWELLSV1_H
