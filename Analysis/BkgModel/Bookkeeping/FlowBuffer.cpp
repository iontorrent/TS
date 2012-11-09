/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowBuffer.h"

// is this a flow where we are writing out information
// i.e. flow buffers full or last flow
// but might be different later, so isolate the code
bool CheckFlowForWrite(int flow, bool last_flow)
{
  return((flow+1) % NUMFB ==0 || last_flow);
}

bool CheckFlowForStartBlock(int flow)
{
  return(flow%NUMFB==0);
}

int CurComputeBlock(int flow){
  return(ceil ( float ( flow+1 ) /NUMFB ) - 1);
};

void flow_buffer_info::GenerateNucMap(int *prev_same_nuc_tbl, int *next_same_nuc_tbl)
{
    // look at background-tracking per flow
    for (int i=0; i < NUMFB;i++)
    {
      int NucID = flow_ndx_map[i];
      int prev = i;
      int next = i;

      for (int j=i-1;j > 0;j--)
      {
        if (flow_ndx_map[j] == NucID)
        {
          prev = j;
          break;
        }
      }

      for (int j=i+1;j < NUMFB;j++)
      {
        if (flow_ndx_map[j] == NucID)
        {
          next = j;
          break;
        }
      }

      prev_same_nuc_tbl[i] = prev;
      next_same_nuc_tbl[i] = next;
    }
}