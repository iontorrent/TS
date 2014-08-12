/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowBuffer.h"

void FlowBufferInfo::GenerateNucMap(int *prev_same_nuc_tbl, int *next_same_nuc_tbl)
{
    // look at background-tracking per flow
    for (int i=0; i < maxFlowCount ;i++)
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

      for (int j=i+1;j < maxFlowCount;j++)
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
