/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOWCONTEXT_H
#define FLOWCONTEXT_H

#include "OptBase.h"

// track the flow formula which gets translated at least 4 separate times in the code into the actual flows done by the PGM
// obvious candidate for centralized code
// Marcin also has a class like this in basecaller

class FlowContext{
  public:
    char *flowOrder;  // cyclic string of effect
    bool flowOrderOverride;
    int *flowOrderIndex;  // obviously this contains the nuc type per flow for all flows
     
    int numFlowsPerCycle;  // wash flow happens interspersed at this frequency
   
    int flowLimitSet;
    int numTotalFlows; // how many flows we're going to handle over the whole run

    bool flow_range_set;
    int startingFlow; // start of the chunk of flows we're going to process - 0 based notation
    int endingFlow; // end of the chunk of flows: the usual C++ convention of <N, 0 based
    
    int GetNumFlows() const {
        return (numTotalFlows);
    }
    int getFlowSpan(){
      return(endingFlow-startingFlow);
    }
   char  ReturnNucForNthFlow(int flow) const;
   void DefaultFlowFormula();
   void DetectFlowFormula(char *explog_path); // if not specified, go find it out
   ~FlowContext();
   void SetFlowRange(int _startingFlow, int _flow_interval);
   void SetFlowLimit( long flowlimit);

   // read from command-line sub-string
   void PrintHelp();
   void SetOpts(OptArgs &opts, Json::Value& json_params);

 private:
   void CheckLimits();
};


#endif // FLOWCONTEXT_H
