/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <assert.h>

#include "OrderedRegionSFFWriter.h"

#include "file-io/sff.h"
#include "file-io/sff_file.h"
#include "file-io/sff_header.h"
#include "file-io/sff_read_header.h"
#include "file-io/sff_read.h"

using namespace std;



OrderedRegionSFFWriter::OrderedRegionSFFWriter()
{
  sff_file = NULL;
  sff = NULL;
  numReads = 0;
  numFlows = 0;
  numRegions = 0;
  pthread_mutex_t tmpMutex = PTHREAD_MUTEX_INITIALIZER;
  dropboxWriteMutex = new pthread_mutex_t(tmpMutex);
  sffWriteMutex = new pthread_mutex_t(tmpMutex);
  numRegionsWritten = 0;
}

OrderedRegionSFFWriter::~OrderedRegionSFFWriter()
{
  delete sffWriteMutex;
  delete dropboxWriteMutex;
}


void OrderedRegionSFFWriter::OpenForWrite(const char *experimentName, const char *sffFileName,
    int _numRegions, int _numFlows, const char *flowChars, const char *keySequence)
{
  numReads = 0;
  numRegionsWritten = 0;
  numFlows = _numFlows;
  numRegions = _numRegions;
  isRegionReady.assign(numRegions+1,false);
  regionDropbox.clear();
  regionDropbox.resize(numRegions);

  char fileName[strlen(experimentName) + strlen(sffFileName) + 2];
  sprintf(fileName, "%s/%s", experimentName, sffFileName);

  sff_header_t *sff_header = sff_header_init1(numReads, numFlows, flowChars, keySequence);
  sff_file = sff_fopen(fileName, "wb", sff_header, NULL);
  sff_header_destroy(sff_header);

  sff = sff_init1();
  sff->gheader = sff_file->header;
  sff->rheader->name = ion_string_init(0);
  sff->read->bases = ion_string_init(0);
  sff->read->quality = ion_string_init(0);
}


void OrderedRegionSFFWriter::Close()
{
  for (;numRegionsWritten < numRegions; numRegionsWritten++)
    PhysicalWriteRegion(numRegionsWritten);

  fseek(sff_file->fp, 0, SEEK_SET);
  sff_file->header->n_reads = numReads;
  sff_header_write(sff_file->fp, sff_file->header);

  sff_fclose(sff_file);

  free(sff->read->bases);
  free(sff->read->quality);
  free(sff->read);
  sff->read = NULL;
  free(sff->rheader->name);
  sff->rheader->name = NULL;
  sff_destroy(sff);
}


void OrderedRegionSFFWriter::WriteRegion(int iRegion, deque<SFFWriterWell> &regionWells)
{
  // Deposit results in the dropbox
  pthread_mutex_lock(dropboxWriteMutex);
  regionDropbox[iRegion].clear();
  regionDropbox[iRegion].swap(regionWells);
  isRegionReady[iRegion] = true;
  pthread_mutex_unlock(dropboxWriteMutex);

  // Attempt writing duty
  if (pthread_mutex_trylock(sffWriteMutex))
    return;
  while (true) {
    pthread_mutex_lock(dropboxWriteMutex);
    bool cannotWrite = !isRegionReady[numRegionsWritten];
    pthread_mutex_unlock(dropboxWriteMutex);
    if (cannotWrite)
      break;
    PhysicalWriteRegion(numRegionsWritten);
    numRegionsWritten++;
  }
  pthread_mutex_unlock(sffWriteMutex);
}


void OrderedRegionSFFWriter::PhysicalWriteRegion(int iRegion)
{
  for (deque<SFFWriterWell>::iterator well = regionDropbox[iRegion].begin(); well != regionDropbox[iRegion].end(); well++) {

    // initialize the header
    sff->rheader->name_length = well->name.length();
    sff->rheader->name->s = (char *)well->name.c_str();
    sff->rheader->n_bases = well->numBases;
    sff->rheader->clip_qual_left = well->clipQualLeft;
    sff->rheader->clip_qual_right = well->clipQualRight;
    sff->rheader->clip_adapter_left = well->clipAdapterLeft;
    sff->rheader->clip_adapter_right = well->clipAdapterRight;

    // initialize the read
    uint16_t  flowgram[numFlows];
    for(int iFlow = 0; iFlow < numFlows; iFlow++) {
        int flowVal = (int)(well->flowIonogram[iFlow]*100.0+0.5);
        flowgram[iFlow] = (flowVal < 0) ? 0 : flowVal;
    }
    sff->read->flowgram = flowgram;
    sff->read->flow_index = &(well->baseFlowIndex[0]);
    sff->read->bases->s = &(well->baseCalls[0]);
    sff->read->quality->s = (char *)&(well->baseQVs[0]);

    // write
    sff_write(sff_file, sff);

    numReads++;
  }
  regionDropbox[iRegion].clear();
}


void  SFFWriterWell::moveTo(SFFWriterWell &w)
{
  name.swap(w.name);
  name.clear();
  flowIonogram.swap(w.flowIonogram);
  flowIonogram.clear();
  baseFlowIndex.swap(w.baseFlowIndex);
  baseFlowIndex.clear();
  baseCalls.swap(w.baseCalls);
  baseCalls.clear();
  baseQVs.swap(w.baseQVs);
  baseQVs.clear();
  w.numBases = numBases;
  w.clipQualLeft = clipQualLeft;
  w.clipQualRight = clipQualRight;
  w.clipAdapterLeft = clipAdapterLeft;
  w.clipAdapterRight = clipAdapterRight;
}

void  SFFWriterWell::copyTo(SFFWriterWell &w)
{
  w.name = name;
  w.flowIonogram = flowIonogram;
  w.baseFlowIndex = baseFlowIndex;
  w.baseCalls = baseCalls;
  w.baseQVs = baseQVs;
  w.numBases = numBases;
  w.clipQualLeft = clipQualLeft;
  w.clipQualRight = clipQualRight;
  w.clipAdapterLeft = clipAdapterLeft;
  w.clipAdapterRight = clipAdapterRight;
}



