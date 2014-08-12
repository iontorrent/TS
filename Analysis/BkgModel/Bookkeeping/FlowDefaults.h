/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef FLOWDEFAULTS_H
#define FLOWDEFAULTS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "BkgMagicDefines.h"
#include "Serialization.h"


// I'm crying because this object isn't unified across our codebase
// we have 4 essentially identical objects handling flow orders
// that one day I dream of someone unifying
// but today is apparently not that day
class FlowMyTears{
  // plausibly a shared object
  int              flow_order_len;     // length of entire flow order sequence (might be > num flows in a block)
  std::vector<int> glob_flow_ndx_map;  // maps flow number within a cycle to nucleotide (flow_order_len values)
  std::string      flowOrder;          // entire flow order as a char array (flow_order_len values)
public:
  FlowMyTears();

  void SetFlowOrder(char *_flowOrder);
  int  GetNucNdx(int flow_ndx) const
  {
    return glob_flow_ndx_map[flow_ndx%flow_order_len];
  }

  // special functions for double-tap flows
  int IsDoubleTap(int flow)
  {
    // may need to refer to earlier flows
    if (flow==0)
      return(1); // can't be double tap

    if (glob_flow_ndx_map[flow%flow_order_len]==glob_flow_ndx_map[(flow-1+flow_order_len)%flow_order_len])
      return(0);
    return(1);
  }

  void GetFlowOrderBlock(int *my_flow, int i_start, int i_stop) const;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & flow_order_len
        & glob_flow_ndx_map
        & flowOrder;
  }
};


#endif //FLOWDEFAULTS_H
