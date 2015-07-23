/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SLICEDCHIPEXTRAS_H
#define SLICEDCHIPEXTRAS_H

#include "FlowBuffer.h"
#include "DiffEqModel.h"

class SlicedChipExtras
{
  public:
  incorporation_params_block_flows  *cur_bead_block;
  buffer_params_block_flows         *cur_buffer_block;
  FlowBufferInfo                    *my_flow;
  int                               global_flow_max;

  SlicedChipExtras() : cur_bead_block(0), cur_buffer_block(0), my_flow(0), global_flow_max(0) {}
  ~SlicedChipExtras() {}
  void allocate( int _global_flow_max )
  {
    global_flow_max = _global_flow_max;
    cur_bead_block = new incorporation_params_block_flows( global_flow_max );
    cur_buffer_block = new buffer_params_block_flows( global_flow_max );
    my_flow = new FlowBufferInfo();
    my_flow->SetMaxFlowCount( global_flow_max );
  }

  void free()
  {
    delete cur_bead_block;
    delete cur_buffer_block;
    delete my_flow;
    cur_bead_block = 0;
    cur_buffer_block = 0;
    my_flow = 0;
    global_flow_max = 0;
  }
};

#endif // SLICEDCHIPEXTRAS_H
