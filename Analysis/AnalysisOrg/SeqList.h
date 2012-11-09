/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SEQLIST_H
#define SEQLIST_H

#include <stdio.h>
#include <stdlib.h>
#include "SpecialDataTypes.h"
#include "Utils.h"


//@TODO:  Please make this a fully featured class so we can handle 3 or more keys
class SeqListClass{
  public:
  SequenceItem *seqList;
  int numSeqListItems;

  SeqListClass();
  ~SeqListClass();
  void Delete();
  void Allocate(int numItems);
  void StdInitialize(char *flowOrder, char *libKey, char *tfKey);
  void UpdateMinFlows(int &minNumKeyFlows);
  void UpdateMaxFlows(int &maxNumKeyFlows);
};

void InitializeSeqList(SequenceItem *seqList, int numSeqListItems, char *letter_flowOrder, char *libKey, char *tfKey);

#endif // SEQLIST_H
