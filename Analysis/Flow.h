/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FLOW_H
#define FLOW_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

class Flow
{
    public:
        Flow (char *_flowOrder);
        ~Flow ();
        void    BuildNucIndex();
        int     GetNuc(int flow);
        char*   GetFlowOrder() {return (flowOrder);}
		void	SetFlowOrder(char *_flowOrder);
        
    private:
        int *flowOrderIndex;
        int numFlowsPerCycle;
        char *flowOrder;
    
};

#endif // FLOW_H
