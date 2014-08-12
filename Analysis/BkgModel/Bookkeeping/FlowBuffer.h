/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWBUFFER_H
#define FLOWBUFFER_H

#include "BkgMagicDefines.h"
#include "Serialization.h"

class FlowBufferInfo {
    // information about the circular buffer of flows we can process at one time
    int   maxFlowCount;

    // No copies, please.
    FlowBufferInfo( const FlowBufferInfo & );
    FlowBufferInfo & operator=( const FlowBufferInfo & );

public:
    // TODO: In a sane universe, these int *'s wouldn't be public. This is not a sane universe.
    
    // information that maps between flow order and nucleotide
    int   *flow_ndx_map;    // maps buffer number to nucleotide.
    int   *dbl_tap_map;     // am I a double-tap and should have amplitude 0?

public:
    int GetMaxFlowCount() const { return maxFlowCount; }
    void SetMaxFlowCount( int n ) { 
      // Clean out what we have.
      delete [] flow_ndx_map;
      delete [] dbl_tap_map;

      // Save the new size.
      maxFlowCount = n;

      // Allocate new buffers.
      flow_ndx_map = new int[n];
      dbl_tap_map = new int[n];

      // Zero out these new buffers.
      for (int i=0; i<n; i++)
      {
        flow_ndx_map[i] = 0;
        dbl_tap_map[i] = 0;
      }
    }

    int     flowBufferCount;  // how many delayed flows we are at
    int     flowBufferReadPos;
    int     flowBufferWritePos;  // current place we are filling
    
    FlowBufferInfo() {
      maxFlowCount = 0;
      flow_ndx_map = 0;
      dbl_tap_map = 0;
      flowBufferCount = 0;
      flowBufferReadPos = 0;
      flowBufferWritePos = 0;
    };

    ~FlowBufferInfo(){
      delete [] flow_ndx_map;
      delete [] dbl_tap_map;
    }
    //bool FirstBlock(){
    //return(buff_flow[0] ==0);
    //}

    void Increment(){
          flowBufferCount++;
      flowBufferWritePos = (flowBufferWritePos + 1) % maxFlowCount;
    };
    void Decrement(){
          flowBufferCount--;
        flowBufferReadPos = (flowBufferReadPos + 1) % maxFlowCount;
    };
    void ResetBuffersForWriting(){
      flowBufferCount = 0;
      flowBufferReadPos = 0;
      flowBufferWritePos = 0;
    };

    bool BuffersNonZero(){
      return(flowBufferCount>0);
    };
    bool StopDropAndCalculate(bool last){
        return (flowBufferCount>=maxFlowCount or last);
    };
    void GenerateNucMap(int *prev_tbl, int *next_tbl);
    void SetFlowNdxMap(int xx){
       flow_ndx_map[flowBufferWritePos] = xx;
    };
    void SetDblTapMap(int xx){
       dbl_tap_map[flowBufferWritePos] = xx;
    };
    // Serialization section
    template<typename Archive>
    void serialize (Archive& ar, const unsigned version) {
      boost::serialization::split_member(ar, *this, version);
    }
    template<typename Archive>
    void save (Archive& ar, const unsigned version) const {
      ar << flowBufferCount;
      ar << flowBufferReadPos;
      ar << flowBufferWritePos;
      ar << maxFlowCount;
      for( int i = 0 ; i < maxFlowCount ; ++i )
      {
        ar << flow_ndx_map[i];
        ar << dbl_tap_map[i];
      }
    }
    template<typename Archive>
    void load (Archive& ar, const unsigned version) {
      ar >> flowBufferCount;
      ar >> flowBufferReadPos;
      ar >> flowBufferWritePos;
      int newFlowCount;
      ar >> newFlowCount;
      SetMaxFlowCount( newFlowCount );
      for( int i = 0 ; i < maxFlowCount ; ++i )
      {
        ar >> flow_ndx_map[i];
        ar >> dbl_tap_map[i];
      }
    }
};




#endif // FLOWBUFFER_H
