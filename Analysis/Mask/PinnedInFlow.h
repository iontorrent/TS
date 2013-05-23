/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PINNEDINFLOW_H
#define PINNEDINFLOW_H

#include "Image.h"
#include <vector>
#include "Serialization.h"
#include "SynchDat.h"

// keeps track of well pinning across all flows
// Every well is unpinned until it becomes pinned in a flow,
// after which it is permanently marked as pinned in later flows
class PinnedInFlow
{
 public:
  PinnedInFlow(Mask *maskPtr, int numFlows);
  virtual ~PinnedInFlow();

  virtual void Initialize (Mask *maskPtr);
  virtual int Update(int flow, Image *img, float *gainPtr);
  virtual int Update (int flow, class SynchDat *img, float *gainPtr);

  void UpdateMaskWithPinned (Mask *maskPtr);
  short *Pins() { return (&mPinnedInFlow[0]); };
  bool IsPinned(int flow, int index);
 
  int NumFlows() {return (mNumFlows); };
  void SetPinnedCount (int flow, int value) { mPinsPerFlow[flow] = value; };
  int GetPinnedCount (int flow) { return (mPinsPerFlow[flow]); };

  void DumpSummaryPinsPerFlow (char *experimentName);
  inline void SetPinned(int idx, int flow);

 protected:
   // array that tracks first occurrence of pinning per well
  std::vector<short> mPinnedInFlow; // indices match indices into Mask object
  int mNumWells;      // number of wells
  int mNumFlows;      // number of flows
  std::vector<int> mPinsPerFlow;  // per flow count of pinned wells


 private:
  PinnedInFlow(){ mNumWells=0; mNumFlows=0;} // don't use
  
  friend class boost::serialization::access;
  template<typename Archive>
    void serialize(Archive& ar, const unsigned version) {
    // fprintf(stdout, "Serialize PinnedInFlow ... ");
    ar & 
      mPinnedInFlow &
      mNumWells &
      mNumFlows &
      mPinsPerFlow;
    // fprintf(stdout, "done PinnedInFlow\n");
  }
};

inline bool PinnedInFlow::IsPinned (int flow, int maskIndex)
{
  return ( (mPinnedInFlow[maskIndex]>=0) & (mPinnedInFlow[maskIndex]<=flow) );
}

#endif // PINNEDINFLOW_H
