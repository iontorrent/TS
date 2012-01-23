/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Flow.h"


Flow::Flow (char *_flowOrder)
{
    flowOrder = strdup (_flowOrder);
    numFlowsPerCycle = strlen (_flowOrder);
    BuildNucIndex();
}
Flow::~Flow()
{
    if (flowOrder)
        free (flowOrder);
    if (flowOrderIndex)
        delete [] flowOrderIndex;
}

void Flow::SetFlowOrder (char *_flowOrder)
{
    if (flowOrder)
        free (flowOrder);
    if (flowOrderIndex)
        delete [] flowOrderIndex;
    
    flowOrder = strdup (_flowOrder);
    numFlowsPerCycle = strlen (_flowOrder);
    BuildNucIndex();
}

//
// BuildNucIndex - generates a lookup array that returns what nuc index corresponds to what flow - used by GetNuc
//
void Flow::BuildNucIndex()
{
  flowOrderIndex = new int[numFlowsPerCycle];

  int i;
  for(i=0;i<numFlowsPerCycle;i++) {
    if (flowOrder[i] == 'T')
      flowOrderIndex[i] = 0;
    if (flowOrder[i] == 'A')
      flowOrderIndex[i] = 1;
    if (flowOrder[i] == 'C')
      flowOrderIndex[i] = 2;
    if (flowOrder[i] == 'G')
      flowOrderIndex[i] = 3;
  }
}

//
// GetNuc - For a given flow & flow order, returns the nuc as an index 0 thru 3
//
int Flow::GetNuc(int flow)
{
  // nuc's by definition are 0=T, 1=A, 2=C, 3=G - this is the default flow order on the PGM, so we wanted to make sure it makes sense for our standard
  return flowOrderIndex[flow%numFlowsPerCycle];
}