/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// basic sff reader/writer
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <assert.h>

#include "SFFWrapper.h"
#include "ByteSwapUtils.h"
#include "file-io/sff_definitions.h"
#include "file-io/sff.h"
#include "file-io/sff_file.h"
#include "file-io/sff_header.h"
#include "file-io/sff_read_header.h"
#include "file-io/sff_read.h"

#include "LinuxCompat.h"

#include "dbgmem.h"
#include "Utils.h"

using namespace std;

#define SAFE_FREE(ptr) if (ptr) free(ptr); ptr = NULL;

#define MAX_BASES 16384


SFFWrapper::SFFWrapper()
{
  localIndex = 0;
  sff_file = NULL;
  sff = NULL;
}

SFFWrapper::~SFFWrapper()
{
}

void SFFWrapper::OpenForWrite(const char *experimentName, const char *sffName,
                       int numReads, int numFlows, const char *flowChars, const char *keySequence)
{
  char *fileName = (char *)malloc(sizeof(char) * (strlen(experimentName) + strlen(sffName) + 2));
  sprintf(fileName, "%s/%s", experimentName, sffName);
  sff_file = sff_fopen(fileName, "wb", sff_header_init1(numReads, numFlows, flowChars, keySequence), NULL);
  free(fileName);
}

void SFFWrapper::OpenForRead(const char *experimentName, const char *sffName)
{
  char *fileName = (char *)malloc(sizeof(char) * (strlen(experimentName) + strlen(sffName) + 2));
  sprintf(fileName, "%s/%s", experimentName, sffName);
  OpenForRead(fileName);
  free(fileName);
}

void SFFWrapper::OpenForRead(const char *fileName)
{
  sff_file = sff_fopen(fileName, "rb", NULL, NULL);
}

const sff_t *SFFWrapper::LoadNextEntry(bool *success) {
    *success = true;

    if (!sff_file) {
        fprintf(stderr,"Warning: SFFWrapper::LoadNextEntry: attempt to load read when SFFWrapper file is not open\n");
        *success = false;
        return NULL;
    }

    // Check if we already reached the last read
    if((localIndex) == sff_file->header->n_reads)
      return(NULL);

    // If this is the first call, point to start of data section
    if (localIndex == 0) {
      fseek(sff_file->fp, sff_file->header->gheader_length + sff_file->header->index_length, SEEK_SET);
    }
    localIndex++;

    sff_destroy(sff);
    sff = sff_read(sff_file);
    return sff;
}

const sff_t *SFFWrapper::LoadEntry(uint32_t index)
{
  if (!sff_file)
    return NULL;

  if (localIndex == 0 || (localIndex != index)) {
      fseek(sff_file->fp, sff_file->header->gheader_length + sff_file->header->index_length, SEEK_SET);
      localIndex = index+1;
  } else {
      localIndex = index+1;
      index = 0; // force next read
  }

  uint32_t i;
  for(i=0;i<=index;i++) {
      sff_destroy(sff);
      sff = sff_read(sff_file);
  }

  return sff;
}

void SFFWrapper::Close()
{
  sff_destroy(sff);
  sff = NULL;
  if (sff_file) {
      sff_fclose(sff_file);
      sff_file = NULL;
  }
}

sff_t *SFFWrapper::DataToSFF(int *basesCalled, int numBasesCalled, int *baseFlowIndices, float *flowValues,
                    char *wellName, uint8_t *qualityScores, 
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right)
{
  int i;
  sff_t *_sff;
  sff_read_header_t *rh;
  sff_read_t *read;

  _sff = sff_init1();
  rh = _sff->rheader;
  read = _sff->read;
  
  // soft-copy the global header
  _sff->gheader = sff_file->header;

  // initialize the header
  rh->name_length = (int)strlen(wellName);
  rh->name = ion_string_init(0);
  ion_string_copy1(rh->name, wellName);
  rh->n_bases = numBasesCalled;
  rh->clip_qual_left = clip_qual_left;
  rh->clip_qual_right = clip_qual_right;
  rh->clip_adapter_left = clip_adapter_left;
  rh->clip_adapter_right = clip_adapter_right;
  // size will be re-calculated when written

  // initialize the read
  read->flowgram = (uint16_t*)malloc(sizeof(uint16_t) * sff_file->header->flow_length); 
  for(i=0;i<sff_file->header->flow_length;i++) {
      int flowVal = (int)(flowValues[i]*100.0+0.5); 
      read->flowgram[i] = (flowVal < 0) ? 0 : flowVal;
  }
  read->flow_index = (uint8_t *)malloc(sizeof(uint8_t) * numBasesCalled);
  for(i=0;i<numBasesCalled;i++) {
      read->flow_index[i] = baseFlowIndices[i];
  }
  read->bases = ion_string_init(rh->n_bases+1);
  read->quality = ion_string_init(rh->n_bases+1);
  for(i=0;i<numBasesCalled;i++) {
      read->bases->s[i] = sff_file->header->flow->s[basesCalled[i]];
      read->quality->s[i] = qualityScores[i];
  }
  read->bases->s[i] = read->quality->s[i] = '\0';
  read->bases->l = read->quality->l = numBasesCalled;

  return _sff;
}

// this should instead initialize a sff_t structure
void SFFWrapper::WriteData(int *basesCalled, int numBasesCalled, int *baseFlowIndices, float *flowValues,
                    char *wellName, uint8_t *qualityScores, 
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right)
{
  if(sff) sff_destroy(sff);

  sff = SFFWrapper::DataToSFF(basesCalled, numBasesCalled, baseFlowIndices, flowValues,
                    wellName, qualityScores, 
                    clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right);

  // write
  sff_write(sff_file, sff);

  // destroy
  sff_destroy(sff);
  sff = NULL;
}

int SFFWrapper::WriteData(const hpLen_vec_t &calledHpLen, const weight_vec_t &flowValue, const vector<uint8_t> &quality,
                    string &flowOrder, string &readName,
                    int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right)
{
  // Determine number of bases and allocate memory
  unsigned int numFlows = calledHpLen.size();
  int numBasesCalled = 0;
  for(unsigned int iFlow=0; iFlow < numFlows; iFlow++)
    numBasesCalled += calledHpLen[iFlow];
  if(numBasesCalled != (int) quality.size()) {
    return(1);
  } else {
    int *basesCalled = new int[numBasesCalled];
    int *baseFlowIndices = new int[numBasesCalled];
    uint8_t *qualityScores = new uint8_t[numBasesCalled];
    float *flowValues = new float[numFlows];

    // Copy data to arrays to pass along
    unsigned int iBase = 0;
    unsigned int prevPosFlow = 0;
    for(unsigned int iFlow=0; iFlow < numFlows; iFlow++) {
      for(hpLen_t iNuc=0; iNuc < calledHpLen[iFlow]; iNuc++, iBase++) {
        basesCalled[iBase] = iFlow % flowOrder.size();
        baseFlowIndices[iBase] = iFlow + 1 - prevPosFlow;
        prevPosFlow = iFlow + 1;
        qualityScores[iBase] = quality[iBase];
      }
      flowValues[iFlow] = flowValue[iFlow];
    }

    WriteData(basesCalled, numBasesCalled, baseFlowIndices, flowValues, (char *) readName.c_str(), qualityScores, 
      clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right);

    // cleanup
    delete [] basesCalled;
    delete [] baseFlowIndices;
    delete [] qualityScores;
    delete [] flowValues;

    return(0);
  }
}

void SFFWrapper::WriteData(const sff_t *_sff)
{
  if(sff) sff_destroy(sff); // destroy local sff copy

  // write
  sff_write(sff_file, _sff);
}

void SFFWrapper::UpdateReadCount(int numReads)
{
  if(!sff_file) {
      fprintf (stderr, "%s\n", strerror(errno));
      assert(0);
  }
  fseek(sff_file->fp, 0, SEEK_SET);
  sff_file->header->n_reads = numReads;
  sff_header_write(sff_file->fp, sff_file->header);
}
