/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SFFWRAPPER_H
#define SFFWRAPPER_H

#include <inttypes.h>
#include <string>
#include <cstring>
#include "file-io/sff_definitions.h" 
#include "file-io/sff.h" 
#include "file-io/sff_file.h" 
#include "SpecialDataTypes.h"

class SFFWrapper {
public:
  SFFWrapper();
  virtual ~SFFWrapper();

  void OpenForWrite(const char *experimentName, const char *sffFileName, int numReads, int numFlows, const char *flowChars, const char *keySequence);
  sff_t *DataToSFF(int *basesCalled, int numBasesCalled, int *baseFlowIndices, float *flowValues,
                    char *wellName, uint8_t *qualityScores, 
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right);
  void WriteData(int *basesCalled, int numBasesCalled, int *baseFlowIndices, float *flowValues,
                    char *wellName, uint8_t *qualityScores, 
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right);
  int WriteData(const hpLen_vec_t &calledHpLen, const weight_vec_t &flowValue, const std::vector<uint8_t> &quality,
                    std::string &flowOrder, std::string &readName,
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right);
  void WriteData(const sff_t *_sff);
  void UpdateReadCount(int numReads);

  void OpenForRead(const char *experimentName, const char *sffFileName);
  void OpenForRead(const char *fileName);
  int GetReadCount () {return (sff_file->header->n_reads);}

  const sff_header_t *GetHeader() {return (sff_file->header);}
  const sff_t *LoadEntry(uint32_t index);
  const sff_t *LoadNextEntry(bool *success);
  void Close();

protected:
  sff_file_t *sff_file;
  sff_t *sff;

  uint32_t localIndex;
};

#endif // SFFWRAPPER_H
