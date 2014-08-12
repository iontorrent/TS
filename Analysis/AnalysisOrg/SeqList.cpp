/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SeqList.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>


SeqListClass::SeqListClass()
{
  seqList = NULL;
  numSeqListItems = 0;
}

void SeqListClass::Allocate(int numKeys)
{
  seqList = new SequenceItem[numKeys];
  numSeqListItems = numKeys;
}

void SeqListClass::Delete()
{
  if (seqList!=NULL) delete[] seqList;
  numSeqListItems = 0;
}

SeqListClass::~SeqListClass()
{
  if (seqList!=NULL) delete[] seqList;
  numSeqListItems = 0;
}

void SeqListClass::StdInitialize(char *flowOrder, char *libKey, char *tfKey)
{
  Delete();
  Allocate(2);
  InitializeSeqList(seqList, numSeqListItems, flowOrder,libKey,tfKey);
}

void NullSeqItem(SequenceItem &my_item)
{
  my_item.numKeyFlows = 0;
  my_item.usableKeyFlows =0;
  memset(my_item.Ionogram, 0, sizeof(int[64]));
  memset(my_item.zeromers, 0, sizeof(int[64]));
  memset(my_item.onemers, 0, sizeof(int[64]));
}

void InitializeSeqList(SequenceItem *seqList, int numSeqListItems, char *letter_flowOrder, char *libKey, char *tfKey)
{
  int flowOrderLength = strlen(letter_flowOrder);
  
  seqList[0].type = MaskTF;
  //seqList[0].seq = strdup(tfKey);
  //strcpy((char *)seqList[0].seq,tfKey);
  seqList[0].seq.assign(tfKey);
  
  seqList[1].type = MaskLib;
  //seqList[1].seq = strdup(libKey);  // may release these objects sometime
  //strcpy((char *)seqList[1].seq,libKey);
  seqList[1].seq.assign(libKey);

  NullSeqItem(seqList[0]);
  NullSeqItem(seqList[1]);
  //const int numSeqListItems = sizeof(seqList) / sizeof(SequenceItem);
  // Calculate number of key flows & Ionogram
  //  TFs tempTF(flw->GetFlowOrder()); // MGD note - would like GenerateIonogram to be in the utils lib soon
  for (int i = 0; i < numSeqListItems; i++)
  {
    int zeroMerCount = 0;
    int oneMerCount = 0;

    seqList[i].numKeyFlows = seqToFlow(seqList[i].seq.c_str(), seqList[i].seq.length(),
                                       seqList[i].Ionogram, 64, letter_flowOrder, flowOrderLength);

    seqList[i].usableKeyFlows = seqList[i].numKeyFlows - 1;
    // and calculate for the given flow order, what nucs are 1-mers and which are 0-mers
    // requirement is that we get at least one flow for each nuc that has a 0 and a 1
    // it just might take lots of flows for a given key
    int flow;
    for (flow = 0; flow < seqList[i].numKeyFlows; flow++)
    {
      seqList[i].onemers[flow] = -1;
      seqList[i].zeromers[flow] = -1;
    }
    for (flow = 0; flow < seqList[i].numKeyFlows; flow++)
    {
      int nuc = 0;
      switch (letter_flowOrder[flow%flowOrderLength]) {
        case 'T': nuc = 0; break;
        case 'A': nuc = 1; break;
        case 'C': nuc = 2; break;
        case 'G': nuc = 3; break;
      }
      if (seqList[i].Ionogram[flow] == 1)
      {
        // for now just mark the first occurance of any nuc hit
        if (seqList[i].onemers[nuc] == -1)
        {
          oneMerCount++;
          seqList[i].onemers[nuc] = flow;
        }
      }
      else
      {
        // for now just mark the first occurance of any nuc hit
        if (seqList[i].zeromers[nuc] == -1)
        {
          zeroMerCount++;
          seqList[i].zeromers[nuc] = flow;
        }
      }
    }
    if (oneMerCount <= 1 || zeroMerCount <= 1)
    {
      fprintf(
        stderr,
        "Key: '%s' with flow order: '%s' does not have at least 2 0mers and 2 1mers.\n",
        seqList[i].seq.c_str(), letter_flowOrder);
      exit(EXIT_FAILURE);
    }
  }
}

void SeqListClass::UpdateMaxFlows(int &maxNumKeyFlows)
{
  for (int i=0; i<numSeqListItems; i++)
    if (seqList[i].numKeyFlows > maxNumKeyFlows)
      maxNumKeyFlows = seqList[i].numKeyFlows;
}

void SeqListClass::UpdateMinFlows(int &minNumKeyFlows)
{
  for (int i=0; i<numSeqListItems; i++)
    if ((seqList[i].numKeyFlows - 1) < minNumKeyFlows)
      minNumKeyFlows = seqList[i].numKeyFlows - 1;
}
