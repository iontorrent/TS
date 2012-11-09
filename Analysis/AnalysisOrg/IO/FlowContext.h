/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWCONTEXT_H
#define FLOWCONTEXT_H

// track the flow formula which gets translated at least 4 separate times in the code into the actual flows done by the PGM
// obvious candidate for centralized code
// Marcin also has a class like this in basecaller

class FlowContext{
  public:
      char *flowOrder;  // cyclic string of effect
    bool flowOrderOverride;
     int *flowOrderIndex;  // obviously this contains the nuc type per flow for all flows
     
   unsigned int numFlowsPerCycle;  // wash flow happens interspersed at this frequency
   
    unsigned int flowLimitSet;
    unsigned int numTotalFlows; // how many flows we're going to handle over the whole run

    bool flow_range_set;
    unsigned int startingFlow; // start of the chunk of flows we're going to process - 0 based notation
    unsigned int endingFlow; // end of the chunk of flows: the usual C++ convention of <N, 0 based
    
    int GetNumFlows() {
        return (numTotalFlows);
    }
    int getFlowSpan(){
      return(endingFlow-startingFlow);
    }
   char  ReturnNucForNthFlow(int flow);
   void DefaultFlowFormula();
   void DetectFlowFormula(char *explog_path); // if not specified, go find it out
   ~FlowContext();
   void SetFlowRange(int _startingFlow, int _flow_interval);
   void SetFlowLimit( long flowlimit);

 private:
   void CheckLimits();
};


#endif // FLOWCONTEXT_H
