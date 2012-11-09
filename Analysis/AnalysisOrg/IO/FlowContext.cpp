/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowContext.h"
#include "HandleExpLog.h"
#include <assert.h>

#define  PROCESSCHUNKSIZE 20

void FlowContext::DefaultFlowFormula()
{
  flowOrder = strdup ( "TACG" );
  numFlowsPerCycle = strlen ( flowOrder );
  flowOrderOverride = false;
  flowOrderIndex = NULL;
  numTotalFlows = 0;
  flowLimitSet = 0;
  flow_range_set = false; // just process everything
  startingFlow =0;
  endingFlow = 0;
}

FlowContext::~FlowContext()
{
  if ( flowOrder )
    free ( flowOrder );
}

char FlowContext::ReturnNucForNthFlow(int flow)
{
  return(flowOrder[flow % strlen ( flowOrder ) ]);
}


void FlowContext::DetectFlowFormula ( char *explog_path )
{
// @TODO: obviously needs to be refactored into flow routine
// expand flow formula = flowOrder into appropriate number of flows
  //Determine total number of flows in experiment or previous analysis
  if ( true ) {
    numTotalFlows = GetTotalFlows (explog_path );
    assert ( numTotalFlows > 0 );
  }

  //If flow order was not specified on command line,
  //set it here from info from explog.txt or processParams.txt
  if ( !flowOrderOverride ) {
    if ( flowOrder )
      free ( flowOrder );
    // Get flow order from the explog.txt file
    if ( true ) {
      flowOrder = GetPGMFlowOrder ( explog_path );
      assert ( flowOrder != NULL );
      numFlowsPerCycle = strlen ( flowOrder );
      assert ( numFlowsPerCycle > 0 );
    }
  }

  // Adjust number of flows according to any command line options which may have been used
  // to limit these values
  if ( flowLimitSet ) {
    //support user specified number of flows
    numTotalFlows = ( flowLimitSet < numTotalFlows ? flowLimitSet: numTotalFlows );
    assert ( numTotalFlows > 0 );
  }

  //if flow range is not set, set the full range of flows as the flow range
  if (!flow_range_set){
    startingFlow = 0;
    endingFlow=numTotalFlows;
  }
}

void FlowContext::CheckLimits()
{
  // check for consistency between --flowlimit and --start-flow-plus-interval
  // this check happens before the other flow info is known
  bool flow_limit_set = (flowLimitSet == 0);

  if (flow_range_set)
  {
    // sanity check option 

    int flow_interval = (endingFlow - startingFlow ) + 1;
    if ( flow_interval < PROCESSCHUNKSIZE )
    {
      fprintf ( stderr, "Option Error in --start-flow-plus-interval: Minimum number of flows is %d, saw %d\n",PROCESSCHUNKSIZE, flow_interval );
      exit ( EXIT_FAILURE );
    }
  }

  if (flow_limit_set)
  {
    if ( flowLimitSet < PROCESSCHUNKSIZE )
    {
      fprintf ( stderr, "Option Error: Minimum number of flows is %d.\n", PROCESSCHUNKSIZE );
      exit ( EXIT_FAILURE );
    }
  }

  if (flow_limit_set && flow_range_set) 
  {
    if ( endingFlow > flowLimitSet )
    {
      fprintf ( stderr, "Error --flow-limit %d must be no greater than max specified using --start-flow-plus-interval: %d\n", flowLimitSet, endingFlow );
      exit ( EXIT_FAILURE );
    }
  }
}

void FlowContext::SetFlowRange(int _startingFlow, int _flow_interval)
{
  assert ( _startingFlow >= 0 );
  assert ( _flow_interval > 0 );

  flow_range_set = true;
  startingFlow = ( unsigned int ) _startingFlow;
  endingFlow = startingFlow+( unsigned int ) _flow_interval;

  CheckLimits();

  fprintf ( stdout, "DEBUG: Flows to be analyzed: %d thru %d\n", startingFlow,endingFlow-1 );
}

void FlowContext::SetFlowLimit( long flowlimit )
{
  flowLimitSet = ( unsigned int ) flowlimit;

  CheckLimits();
}
