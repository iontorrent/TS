/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef COMPLEXMASK_H
#define COMPLEXMASK_H

#include "Mask.h"
#include "PinnedInFlow.h"
#include "Serialization.h"

// Our "mask" class is somewhat creaky and old
// and has no facility for extra data fields
// such as pinnedInFlow
// I therefore make this temporary class until we can upgrade
class ComplexMask{
  public:
  Mask *my_mask;
  PinnedInFlow *pinnedInFlow;
  ComplexMask();
  ~ComplexMask();
  void InitMask();
  void InitPinnedInFlow(int numFlows);

 private:	

  friend class boost::serialization::access;
  template<typename Archive>
    void serialize(Archive& ar, const unsigned version)
    {
      // fprintf(stdout, "Serialize ComplexMask ... ");
      ar & 
	my_mask &
	pinnedInFlow;
      // fprintf(stdout, "done ComplexMask\n");
    }

};

#endif // COMPLEXMASK_H
