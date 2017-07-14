/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "BkgControlOpts.h"
#include "IonErr.h"
#include "Utils.h"
#include "BkgMagicDefines.h"

using namespace std;

SignalProcessingBlockControl::SignalProcessingBlockControl(){
  save_wells_flow = 60;
  wellsCompression = 0;
  restart = false;
  restart_from = "";
  restart_next = "";
  restart_check = true;
  updateMaskAfterBkgModel = true;
  numCpuThreads = 0;
  flow_block_sequence.Defaults();
}

void SignalProcessingBlockControl::PrintHelp()
{
	printf ("     SignalProcessingBlockControl\n");
    printf ("     --numcputhreads         INT               number of CPU threads [0]\n");
    printf ("     --wells-compression     INT               set wells compression level [0]\n");
    printf ("     --wells-save-freq       INT               set saveWellsFrequency []\n");
    printf ("     --wells-save-flow       INT               set save_wells_flow (=saveWellsFrequency*20) [60]\n");
    printf ("     --sigproc-compute-flow  STRING            set flow block sequence []\n");
    printf ("     --restart-from          STRING            restart from []\n");
    printf ("     --restart-next          STRING            restart next []\n");
    printf ("     --restart-check         BOOL              restart check [true]\n");
	printf ("     --bkg-bfmask-update     BOOL              update mask after background modeling [true]\n");
    printf ("\n");
}

void SignalProcessingBlockControl::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	wellsCompression = RetrieveParameterInt(opts, json_params, '-', "wells-compression", 0);
    ION_ASSERT(wellsCompression >= 0 && wellsCompression <= 10, "--wells-compression must be between (0,10) inclusive.");
	fprintf(stdout, "wells compression: %d\n", wellsCompression);
	int saveWellsFrequency = RetrieveParameterInt(opts, json_params, '-', "wells-save-freq", -1);
	if(saveWellsFrequency > 0)
	{
		save_wells_flow = 20 * saveWellsFrequency;
        fprintf ( stdout, "Warning: --wells-save-freq is obsolete. Please use --wells-save-flow\n"
                        "         instead, with a value 20 times the old wells-save-freq value.\n");
	}
	else
	{
		save_wells_flow = RetrieveParameterInt(opts, json_params, '-', "wells-save-flow", 60);
	}
    fprintf ( stdout, "Saving wells every %d flows.\n", save_wells_flow );


    if ( save_wells_flow < 20 || save_wells_flow > 2000 )
    {
	  fprintf ( stderr, "Option Error, must be between 20 and 2000: wells-save-flow %d\n", save_wells_flow );
      exit ( EXIT_FAILURE );
    }
	restart_from = RetrieveParameterString(opts, json_params, '-', "restart-from", "");
	restart_next = RetrieveParameterString(opts, json_params, '-', "restart-next", "");
	restart_check = RetrieveParameterBool(opts, json_params, '-', "restart-check", true);
	numCpuThreads = RetrieveParameterInt(opts, json_params, '-', "numcputhreads", 0);
	updateMaskAfterBkgModel = RetrieveParameterBool(opts, json_params, '-', "bkg-bfmask-update", true);

	string s = RetrieveParameterString(opts, json_params, '-', "sigproc-compute-flow", "");
	if (s.length() > 0) 
	{
    // See if the FlowSequence can deal with it.
		if ( ! flow_block_sequence.Set( s.c_str() ) )
		{
		  fprintf ( stderr, "Option Error: --sigproc-compute-flow=%s\n", s.c_str() );
		  exit( EXIT_FAILURE );
		}
	}
}

void BkgModelControlOpts::DefaultBkgModelControl()
{
  polyclonal_filter.enable = true;
  emphasize_by_compression=1; // by default turned to the old method
  // cross-talk all together
  enable_trace_xtalk_correction = true;
  
  gpuControl.DefaultGpuControl();

  unfiltered_library_random_sample = 100000;
  nokey = false;

  washout_threshold = WASHOUT_THRESHOLD;
  washout_flow_detection = WASHOUT_FLOW_DETECTION;

  // Regional smoothing defaults.
  regional_smoothing.alpha = 1.f;
  regional_smoothing.gamma = 1.f;
}

TraceControl::TraceControl(){
  // emptyTrace outlier (wild trace) removal
  do_ref_trace_trim = false;
  span_inflator_min = 10;
  span_inflator_mult = 10;
  cutoff_quantile = .2;

  use_dud_and_empty_wells_as_reference = false;
  empty_well_normalization = false;
}

void TraceControl::PrintHelp()
{
	printf ("     TraceControl\n");
    printf ("     --trim-ref-trace        STRING            on or off to set do_ref_trace_trim; float vector of 3 to set span_inflator_min, span_inflator_mult and cutoff_quantile [10,10,0.2]\n");
    printf ("     --bkg-use-duds          BOOL              use dud and empty wells as reference [false]\n");
	printf ("     --bkg-empty-well-normalization      BOOL  enable empty well normalization [false]\n");
    printf ("\n");
}

void TraceControl::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	use_dud_and_empty_wells_as_reference = RetrieveParameterBool(opts, json_params, '-', "bkg-use-duds", false);
	empty_well_normalization = RetrieveParameterBool(opts, json_params, '-', "bkg-empty-well-normalization", false);
	string s = RetrieveParameterString(opts, json_params, '-', "trim-ref-trace", "");
	if(s.length() > 0)
	{
		if(s == "off")
		{
			do_ref_trace_trim = false;
		}
		else if(s == "on")
		{
			do_ref_trace_trim = true;
		}
		else
		{
			vector<float> vec;
			RetrieveParameterVectorFloat(opts, json_params, '-', "trim-ref-trace", "10,10,0.2", vec);
			if(vec.size() == 3)
			{
				do_ref_trace_trim = true;
				span_inflator_min = vec[0];
				span_inflator_mult = vec[1];
				cutoff_quantile = vec[2];
			}
		}
	}
	if (do_ref_trace_trim)
		fprintf(stdout, "Reference trimming enabled with options: span_inflator_min = %f, span_inflator_mult = %f, cutoff_quantile = %f\n", span_inflator_min, span_inflator_mult, cutoff_quantile);
}

void BkgModelControlOpts::PrintHelp()
{
    printf ("     BkgModelControlOpts\n");
    printf ("     --nokey                 BOOL              nokey [false]\n");
    printf ("     --xtalk-correction      BOOL              enable trace xtalk correction [false for Proton; true for P-zero and PGM]\n");
    printf ("     --n-unfiltered-lib      INT               number of unfiltered library random samples [100000]\n");
    printf ("     --bkg-dont-emphasize-by-compression INT   emphasize by compression [1]\n");
    printf ("     --sigproc-regional-smoothing-alpha  FLOAT sigproc regional smoothing alpha [1.0]\n");
    printf ("     --sigproc-regional-smoothing-gamma  FLOAT sigproc regional smoothing gamma [1.0]\n");
    printf ("     --restart-reg-params-file STRING json file to input/output regional parameters\n");
    printf ("\n");

    polyclonal_filter.PrintHelp(true);
    signal_chunks.PrintHelp();
    trace_control.PrintHelp();
    pest_control.PrintHelp();
    gpuControl.PrintHelp();
}

void BkgModelControlOpts::SetOpts(OptArgs &opts, Json::Value& json_params, int num_flows)
{
	gpuControl.SetOpts(opts, json_params);
	signal_chunks.SetOpts(opts, json_params);
	pest_control.SetOpts(opts, json_params);
	trace_control.SetOpts(opts, json_params);
	polyclonal_filter.SetOpts(true, opts,json_params, num_flows);

	//jz the following comes from CommandLineOpts::GetOpts
	unfiltered_library_random_sample = RetrieveParameterInt(opts, json_params, '-', "n-unfiltered-lib", 100000);
	enable_trace_xtalk_correction = RetrieveParameterBool(opts, json_params, '-', "xtalk-correction", true);
	emphasize_by_compression = RetrieveParameterInt(opts, json_params, '-', "bkg-dont-emphasize-by-compression", 1);
	nokey = RetrieveParameterBool(opts, json_params, '-', "nokey", false);


    washout_threshold = RetrieveParameterFloat(opts, json_params, '-', "bkg-washout-threshold", WASHOUT_THRESHOLD);
    washout_flow_detection = RetrieveParameterInt(opts, json_params, '-', "bkg-washout-flow-detection", WASHOUT_FLOW_DETECTION);

	regional_smoothing.alpha = RetrieveParameterFloat(opts, json_params, '-', "sigproc-regional-smoothing-alpha", 1.0f);
	regional_smoothing.gamma = RetrieveParameterFloat(opts, json_params, '-', "sigproc-regional-smoothing-gamma", 1.0f);

	restartRegParamsFile = RetrieveParameterString(opts, json_params, '-', "restart-region-params-file", "");
}
