/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PINNEDINFLOW_H
#define PINNEDINFLOW_H

#include "Image.h"

// keeps track of well pinning across all flows
// Every well is unpinned until it becomes pinned in a flow,
// after which it is permanently marked as pinned in later flows
class PinnedInFlow
{
 public:
  PinnedInFlow(Mask *maskPtr, int numFlows);
  virtual ~PinnedInFlow();

  virtual void Initialize (Mask *maskPtr);
  virtual int Update(int flow, Image *img);

  void UpdateMaskWithPinned (Mask *maskPtr);
  short *Pins() { return (mPinnedInFlow); };
  bool IsPinned(int flow, int index);
 
  int NumFlows() {return (mNumFlows); };
  void SetPinnedCount (int flow, int value) { mPinsPerFlow[flow] = value; };
  int GetPinnedCount (int flow) { return (mPinsPerFlow[flow]); };

  void DumpSummaryPinsPerFlow (char *experimentName);

 protected:
   // array that tracks first occurrence of pinning per well
  short *mPinnedInFlow; // indices match indices into Mask object

  int mNumWells;      // number of wells
  int mNumFlows;      // number of flows
  int *mPinsPerFlow;  // per flow count of pinned wells

 private:
  PinnedInFlow(); // don't use
  
};

inline bool PinnedInFlow::IsPinned (int flow, int maskIndex)
{
  return ( (mPinnedInFlow[maskIndex]>=0) & (mPinnedInFlow[maskIndex]<=flow) );
}

#endif // PINNEDINFLOW_H
