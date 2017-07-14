/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "FlowContext.h"
#include "HandleExpLog.h"
#include <assert.h>
#include "Utils.h"

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

char FlowContext::ReturnNucForNthFlow(int flow) const
{
  return(flowOrder[flow % strlen ( flowOrder ) ]);
}


void FlowContext::DetectFlowFormula ( char *explog_path )
{
// @TODO: obviously needs to be refactored into flow routine
// expand flow formula = flowOrder into appropriate number of flows
  //Determine total number of flows in experiment or previous analysis
  numTotalFlows = GetTotalFlows (explog_path );
  //assert ( numTotalFlows > 0 ); //TS-14040: changed assert to error log and exit
  if ( numTotalFlows <= 0 ) {
    fprintf ( stderr, "DetectFlowFormula Error: numTotalFlows=%d<=0\n",numTotalFlows);
    exit ( EXIT_FAILURE );
  }

  //If flow order was not specified on command line,
  //set it here from info from explog.txt or processParams.txt
  if ( !flowOrderOverride ) {
    if ( flowOrder )
      free ( flowOrder );
    // Get flow order from the explog.txt file
      flowOrder = GetPGMFlowOrder ( explog_path );
      assert ( flowOrder != NULL );
      numFlowsPerCycle = strlen ( flowOrder );
      assert ( numFlowsPerCycle > 0 );
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
      fprintf ( stderr, "Option Error: Minimum number of flows is %d. Perhaps you need a --flowlimit option *before* --start-flow-plus-interval.\n", PROCESSCHUNKSIZE );
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

void FlowContext::PrintHelp()
{
	printf ("     FlowContext\n");
    printf ("     --flow-order            STRING            setup flow order []\n");
    printf ("     --flowlimit             INT               setup flow limit [-1]\n");
    printf ("     --start-flow-plus-interval          INT VECTOR OF 2  setup flow range [0,0]\n");
    printf ("\n");
}

void FlowContext::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	string fo = RetrieveParameterString(opts, json_params, '-', "flow-order", "");
	if(fo.length() > 0)
	{
		if ( flowOrder )
			free ( flowOrder );
    // upgrade floworder to all-caps(!)
    // otherwise problems as the code does direct char compares
    for (unsigned int i=0; i<fo.size(); i++){
      fo.at(i) = toupper(fo.at(i));
    }
		
		flowOrder = strdup ( fo.c_str() );
		numFlowsPerCycle = strlen ( flowOrder );
		flowOrderOverride = true;
	}

	int tmp_flowlimit = RetrieveParameterInt(opts, json_params, '-', "flowlimit", -1);
	if(tmp_flowlimit >= 0)
	{
		SetFlowLimit( tmp_flowlimit );
	}

	vector<int> vec;
	RetrieveParameterVectorInt(opts, json_params, '-', "start-flow-plus-interval", "0,0", vec);
	if(vec.size() == 2)
	{
		if( vec[1] > 0)
		{
			SetFlowRange(vec[0], vec[1]);
		}
	}
	else
	{
        fprintf ( stderr, "Option Error: start-flow-plus-interval format wrong, not size = 2\n" );
        exit ( EXIT_FAILURE );
	}
}
