/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWBUFFER_H
#define FLOWBUFFER_H

#include "BkgMagicDefines.h"
#include "Serialization.h"

struct flow_buffer_info{
    // information about the circular buffer of flows we can process at one time
    int     numfb;
    int     buff_flow[NUMFB];  // what flow is this really
    int     flowBufferCount;  // how many delayed flows we are at
    int     flowBufferReadPos;
    int     flowBufferWritePos;  // current place we are filling
    // information that maps between flow order and nucleotide
    int flow_ndx_map[NUMFB];        // maps buffer number to nucleotide (NUMFB values)
    int dbl_tap_map[NUMFB]; // am I a double-tap and should have amplitude 0
    
    flow_buffer_info(){
      numfb=0;
      for (int i=0; i<NUMFB; i++)
      {
        buff_flow[i] = 0;
        flow_ndx_map[i] = 0;
        dbl_tap_map[i] = 0;
      }
      flowBufferCount = 0;
      flowBufferReadPos = 0;
      flowBufferWritePos = 0;
    };
    void Increment(){
          flowBufferCount++;
      flowBufferWritePos = (flowBufferWritePos + 1) % numfb;
    };
    void Decrement(){
          flowBufferCount--;
        flowBufferReadPos = (flowBufferReadPos + 1) % numfb;
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
        return (flowBufferCount>=numfb or last);
    };
    void GenerateNucMap(int *prev_tbl, int *next_tbl);
    void SetBuffFlow(int flow){
       buff_flow[flowBufferWritePos] = flow;
    };
    void SetFlowNdxMap(int xx){
       flow_ndx_map[flowBufferWritePos] = xx;
    };
    void SetDblTapMap(int xx){
       dbl_tap_map[flowBufferWritePos] = xx;
    };
};

bool CheckFlowForStartBlock(int flow);
bool CheckFlowForWrite(int flow, bool last_flow);

int CurComputeBlock(int flow);


// Serialization section
template<typename Archive>
void serialize (Archive& ar, flow_buffer_info &o, const unsigned version) {
  ar & 
    o.numfb &
    o.buff_flow &
    o.flowBufferCount &
    o.flowBufferReadPos &
    o.flowBufferWritePos &
    o.flow_ndx_map &
    o.dbl_tap_map;
}



#endif // FLOWBUFFER_H
