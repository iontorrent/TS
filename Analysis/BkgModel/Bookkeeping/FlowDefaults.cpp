/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowDefaults.h"


///-----------------yet another flow object

FlowMyTears::FlowMyTears()
{
  flow_order_len = 4;
}

void FlowMyTears::SetFlowOrder ( char *_flowOrder )
{
  flowOrder      = _flowOrder;
  flow_order_len = flowOrder.length();
  glob_flow_ndx_map.resize(flow_order_len);

  for (int i=0; i<flow_order_len; ++i)
  {
    switch ( toupper ( flowOrder[i] ) )
    {
      case 'T':
        glob_flow_ndx_map[i]=TNUCINDEX;
        break;
      case 'A':
        glob_flow_ndx_map[i]=ANUCINDEX;
        break;
      case 'C':
        glob_flow_ndx_map[i]=CNUCINDEX;
        break;
      case 'G':
        glob_flow_ndx_map[i]=GNUCINDEX;
        break;
      default:
        glob_flow_ndx_map[i]=DEFAULTNUCINDEX;
        break;
    }
  }
}

void FlowMyTears::GetFlowOrderBlock ( int *my_flow, int i_start, int i_stop ) const
{
  for ( int i=i_start; i<i_stop; i++ )
    my_flow[i-i_start] = GetNucNdx ( i );
}
