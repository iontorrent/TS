/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdio.h>
#include <stdlib.h> //EXIT_FAILURE
#include <ctype.h>  //tolower
#include <libgen.h> //dirname, basename
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "CommandLineOpts.h"
#include "ChipIdDecoder.h"
#include "IonErr.h"
#include "ClonalFilter/mixed.h"

using namespace std;

void ModuleControlOpts::PrintHelp()
{
	printf ("     ModuleControlOpts\n");
    printf ("     --bfonly                BOOL              do bead finding only [false]\n");
    printf ("     --from-beadfind         BOOL              do analysis from bead finding result [false]\n");
    printf ("     --pass-tau              BOOL              pass tau value [true]\n");
    printf ("\n");
}

void ModuleControlOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	BEADFIND_ONLY = RetrieveParameterBool(opts, json_params, '-', "bfonly", false);
	reusePriorBeadfind = RetrieveParameterBool(opts, json_params, '-', "from-beadfind", false);
	passTau = RetrieveParameterBool(opts, json_params, '-', "pass-tau", true);
}

void ObsoleteOpts::PrintHelp()
{
	printf ("     ObsoleteOpts\n");
    printf ("     --nuc-correct           INT               do nuc trace correction [0]\n");
    printf ("     --forcenn               INT               use neighbor subtraction [0]\n");
    printf ("     --forceNN               INT               same as forcenn [0]\n");
    printf ("     --use-pinned            BOOL              use pinned [false]\n");
    printf ("\n");
}

void ObsoleteOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
{
	NUC_TRACE_CORRECT = RetrieveParameterInt(opts, json_params, '-', "nuc-correct", 0);
	USE_PINNED = RetrieveParameterBool(opts, json_params, '-', "use-pinned", false);
	neighborSubtract = RetrieveParameterInt(opts, json_params, '-', "forcenn", 0);
}

void CommandLineOpts::SetUpProcessing()
{
  SetSysContextLocations();
  SetFlowContext(sys_context.explog_path);

  loc_context.FindDimensionsByType ( (char*)(sys_context.explog_path.c_str()) );
  img_control.SetWashFlow ( (char*)(sys_context.explog_path.c_str()) );
}

void CommandLineOpts::SetSysContextLocations ()
{
  sys_context.GenerateContext (); // find our directories
  fprintf ( stdout, "dat source = %s\n",sys_context.dat_source_directory );

  // now use our source directory to find everything else
  sys_context.FindExpLogPath();
  sys_context.SetUpAnalysisLocation();

  // create the results folder if it doesn't already exist
  CreateResultsFolder (sys_context.GetResultsFolder());
}

void CommandLineOpts::SetFlowContext ( string explog_path )
{
  flow_context.DetectFlowFormula ( (char*)(explog_path.c_str()) ); // Set up flow order expansion
}


void CommandLineOpts::PrintHelp()
{
	mod_control.PrintHelp();
	no_control.PrintHelp();
	flow_context.PrintHelp();
	key_context.PrintHelp();
	loc_context.PrintHelp();
	sys_context.PrintHelp();
	img_control.PrintHelp();
	bkg_control.PrintHelp();
	bfd_control.PrintHelp();
}

void CommandLineOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
{
  flow_context.DefaultFlowFormula();
  flow_context.SetOpts(opts, json_params);
  bkg_control.DefaultBkgModelControl();
  // Apparently flow_context does not have a member that gives me the total number of flows in a run, so I'm hard coding this baby!
  bkg_control.SetOpts(opts, json_params, 100000);
  bfd_control.DefaultBeadfindControl();
  bfd_control.SetOpts(opts, json_params);
  img_control.DefaultImageOpts();
  img_control.SetOpts(opts, json_params);
  mod_control.SetOpts(opts, json_params);
  loc_context.DefaultSpatialContext();
  loc_context.SetOpts(opts, json_params);
  key_context.DefaultKeys();
  key_context.SetOpts(opts, json_params);
  no_control.SetOpts(opts, json_params);
  sys_context.DefaultSystemContext();
  sys_context.SetOpts(opts, json_params);

  // We can only do save and restore on an even flow block boundary.
  // Now that all the parameters have been set, we can check their sanity.

  // If we're goign to write out stuff just before a flow, the flow must exist.
  if ( ! bkg_control.signal_chunks.restart_next.empty() &&
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.endingFlow )->begin() !=
                                                    flow_context.endingFlow               )
  {
    fprintf( stderr, "Option Error: You're using --restart-next to write out a save/restore file\n"
                     "at a flow (%d) which isn't at the start of a flow block. Perhaps %d or %d\n"
                     "would be better choices.\n",
       flow_context.endingFlow,
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.endingFlow )->begin(),
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.endingFlow )->end() );

    exit( EXIT_FAILURE );
  }


  // If we're goign to read stuff in at a flow, the flow must exist.
  if ( ! bkg_control.signal_chunks.restart_from.empty() &&
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.startingFlow )->begin() !=
                                                    flow_context.startingFlow               )
  {
    fprintf( stderr, "Option Error: You're using --restart-from to read in a save/restore file\n"
                     "at a flow (%d) which isn't at the start of a flow block. Perhaps %d or %d\n"
                     "would be better choices.\n",
       flow_context.startingFlow,
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.startingFlow )->begin(),
       bkg_control.signal_chunks.flow_block_sequence.BlockAtFlow( flow_context.startingFlow )->end() );

    exit( EXIT_FAILURE );
  }

}

ValidateOpts::ValidateOpts()
{
	// BkgModelControlOpts
	m_opts["n-unfiltered-lib"] = VT_INT;
	m_opts["xtalk-correction"] = VT_BOOL;
	m_opts["bkg-dont-emphasize-by-compression"] = VT_INT;
	m_opts["nokey"] = VT_BOOL;
	m_opts["clonal-filter-bkgmodel"] = VT_BOOL;
	m_opts["clonal-filter-debug"] = VT_BOOL;
	m_opts["clonal-filter-use-last-iter-params"] = VT_BOOL;
        m_opts["filter-extreme-ppf-only"] = VT_BOOL;
	m_opts["mixed-first-flow"] = VT_INT;
	m_opts["mixed-last-flow"] = VT_INT;
	m_opts["max-iterations"] = VT_INT;
	m_opts["mixed-model-option"] = VT_INT;
	m_opts["mixed-stringency"] = VT_DOUBLE;
	m_opts["sigproc-regional-smoothing-alpha"] = VT_FLOAT;
	m_opts["sigproc-regional-smoothing-gamma"] = VT_FLOAT;
        m_opts["restart-region-params-file"] = VT_STRING;
        m_opts["skip-first-flow-block-regional-fitting"] = VT_BOOL;

	// GpuControlOpts
	m_opts["gpuworkload"] = VT_FLOAT;
	m_opts["gpuWorkLoad"] = VT_FLOAT;
	m_opts["gpu-num-streams"] = VT_INT;
	m_opts["gpu-memory-per-proc"] = VT_INT;
	m_opts["gpu-amp-guess"] = VT_INT;
	m_opts["gpu-single-flow-fit"] = VT_INT;
	m_opts["gpu-single-flow-fit-blocksize"] = VT_INT;
	m_opts["gpu-single-flow-fit-l1config"] = VT_INT;
	m_opts["gpu-multi-flow-fit"] = VT_INT;
	m_opts["gpu-force-multi-flow-fit"] = VT_BOOL;
	m_opts["gpu-multi-flow-fit-blocksize"] = VT_INT;
	m_opts["gpu-multi-flow-fit-l1config"] = VT_INT;
	m_opts["gpu-single-flow-fit-type"] = VT_INT;
	m_opts["gpu-hybrid-fit-iter"] = VT_INT;
	m_opts["gpu-partial-deriv-blocksize"] = VT_INT;
	m_opts["gpu-partial-deriv-l1config"] = VT_INT;
	m_opts["gpu-use-all-devices"] = VT_BOOL;
	m_opts["gpu-verbose"] = VT_BOOL;
	m_opts["gpu-device-ids"] = VT_INT;
	m_opts["gpu-fitting-only"] = VT_BOOL;
	m_opts["gpu-tmidnuc-shift-per-flow"] = VT_BOOL;
	m_opts["gpu-flow-by-flow"] = VT_BOOL;
	m_opts["post-fit-handshake-worker"] = VT_BOOL;
	m_opts["gpu-switch-to-flow-by-flow-at"] = VT_INT;
	m_opts["gpu-num-history-flows"] = VT_INT;


	// SignalProcessingBlockControl
	m_opts["wells-compression"] = VT_INT;
	m_opts["wells-save-freq"] = VT_INT;
	m_opts["wells-save-flow"] = VT_INT;
	m_opts["restart-from"] = VT_STRING;
	m_opts["restart-next"] = VT_STRING;
	m_opts["restart-check"] = VT_BOOL;
	m_opts["numcputhreads"] = VT_INT;
	m_opts["bkg-bfmask-update"] = VT_BOOL;
	m_opts["sigproc-compute-flow"] = VT_STRING;

	// TraceControl
	m_opts["bkg-use-duds"] = VT_BOOL;
	m_opts["bkg-empty-well-normalization"] = VT_BOOL;
	m_opts["trim-ref-trace"] = VT_STRING;

	// DebugMe
	m_opts["bkg-debug-param"] = VT_INT;
	m_opts["bkg-debug-nsamples"] = VT_INT;
	m_opts["bkg-debug-region"] = VT_VECTOR_INT;
	m_opts["bkg-debug-trace-sse"] = VT_STRING;
	m_opts["bkg-debug-trace-rcflow"] = VT_STRING;
	m_opts["bkg-debug-trace-xyflow"] = VT_STRING;
	m_opts["bkg-dbg-trace"] = VT_VECTOR_INT;
	m_opts["debug-bead-only"] = VT_BOOL;
	m_opts["region-vfrc-debug"] = VT_BOOL;
	m_opts["bkg-debug-files"] = VT_BOOL;

	// BeadfindControlOpts
	m_opts["beadfind-type"] = VT_STRING;
	m_opts["use-beadmask"] = VT_STRING;
    m_opts["exclusion-mask"] = VT_STRING;
	m_opts["beadmask-categorized"] = VT_BOOL;
	m_opts["beadfind-basis"] = VT_STRING;
	m_opts["beadfind-dat"] = VT_STRING;
	m_opts["beadfind-bgdat"] = VT_STRING;
	m_opts["beadfind-sdasbf"] = VT_BOOL;
	m_opts["beadfind-bfmult"] = VT_FLOAT;
	m_opts["beadfind-minlive"] = VT_DOUBLE;
	m_opts["beadfind-filt-noisy-col"] = VT_STRING;
	m_opts["beadfind-minlivesnr"] = VT_DOUBLE;
	m_opts["beadfind-min-tf-snr"] = VT_DOUBLE;
	m_opts["beadfind-tf-min-peak"] = VT_FLOAT;
	m_opts["beadfind-lib-min-peak"] = VT_FLOAT;
	m_opts["beadfind-lib-filt"] = VT_DOUBLE;
	m_opts["beadfind-tf-filt"] = VT_DOUBLE;
	m_opts["beadfind-skip-sd-recover"] = VT_INT;
	m_opts["beadfind-sep-ref"] = VT_BOOL;
	m_opts["beadfind-smooth-trace"] = VT_BOOL;
	m_opts["beadfind-diagnostics"] = VT_INT;
	m_opts["beadfind-gain-correction"] = VT_BOOL;
	m_opts["datacollect-gain-correction"] = VT_BOOL;
	m_opts["beadfind-blob-filter"] = VT_BOOL;
	m_opts["beadfind-predict-start"] = VT_INT;
	m_opts["beadfind-predict-end"] = VT_INT;
	m_opts["beadfind-sig-ref-type"] = VT_INT;
	m_opts["beadfind-zero-flows"] = VT_STRING;
	m_opts["beadfind-num-threads"] = VT_INT;
	m_opts["bfold"] = VT_BOOL;
	m_opts["noduds"] = VT_BOOL;
	m_opts["b"] = VT_STRING;
	m_opts["beadfindfile"] = VT_STRING;
	m_opts["beadfindFile"] = VT_STRING;
	m_opts["beadfind-mesh-step"] = VT_VECTOR_INT;

	//ImageControlOpts
	m_opts["pca-test"] = VT_STRING;
	m_opts["PCA-test"] = VT_STRING;
	m_opts["acq-prefix"] = VT_STRING;
	m_opts["dat-postfix"] = VT_STRING;
	m_opts["col-flicker-correct"] = VT_BOOL;
	m_opts["col-flicker-correct-verbose"] = VT_BOOL;
	m_opts["col-flicker-correct-aggressive"] = VT_BOOL;
	m_opts["img-gain-correct"] = VT_BOOL;
	m_opts["smoothing-file"] = VT_STRING;
	m_opts["smoothing"] = VT_STRING;
	m_opts["ignore-checksum-errors"] = VT_BOOL;
	m_opts["ignore-checksum-errors-1frame"] = VT_BOOL;
	m_opts["output-pinned-wells"] = VT_BOOL;
	m_opts["flowtimeoffset"] = VT_INT;
	m_opts["nn-subtract-empties"] = VT_BOOL;
	m_opts["nnmask"] = VT_VECTOR_INT;
	m_opts["nnMask"] = VT_VECTOR_INT;
	m_opts["nnmaskwh"] = VT_VECTOR_INT;
	m_opts["nnMaskWH"] = VT_VECTOR_INT;
	m_opts["hilowfilter"] = VT_INT;
	m_opts["total-timeout"] = VT_BOOL;
	m_opts["readaheaddat"] = VT_INT;
	m_opts["readaheadDat"] = VT_INT;
	m_opts["no-threaded-file-access"] = VT_BOOL;
	m_opts["f"] = VT_INT;
	m_opts["frames"] = VT_INT;
	m_opts["col-doubles-xtalk-correct"] = VT_BOOL;
	m_opts["pair-xtalk-coeff"] = VT_FLOAT;
	m_opts["fluid-potential-correct"] = VT_BOOL;
	m_opts["fluid-potential-threshold"] = VT_FLOAT;
	m_opts["corr-noise-correct"] = VT_BOOL;
	m_opts["mask-datacollect-exclude-regions"] = VT_BOOL;


	// ModuleControlOpts
	m_opts["bfonly"] = VT_BOOL;
	m_opts["from-beadfind"] = VT_BOOL;
	m_opts["pass-tau"] = VT_BOOL;

	// SpatialContext
	m_opts["region-size"] = VT_VECTOR_INT;
	m_opts["cropped"] = VT_VECTOR_INT;
	m_opts["analysis-region"] = VT_VECTOR_INT;
	m_opts["cropped-region-origin"] = VT_VECTOR_INT;

	// FlowContext
	m_opts["flow-order"] = VT_STRING;
	m_opts["flowlimit"] = VT_INT;
	m_opts["start-flow-plus-interval"] = VT_INT;

	// KeyContext
	m_opts["librarykey"] = VT_STRING;
	m_opts["libraryKey"] = VT_STRING;
	m_opts["tfkey"] = VT_STRING;
	m_opts["tfKey"] = VT_STRING;

	// ObsoleteOpts
	m_opts["nuc-correct"] = VT_INT;
	m_opts["use-pinned"] = VT_BOOL;
	m_opts["forcenn"] = VT_INT;
	m_opts["forceNN"] = VT_INT;

	// SystemContext
	m_opts["local-wells-file"] = VT_BOOL;
	m_opts["well-stat-file"] = VT_STRING;
	m_opts["stack-dump-file"] = VT_STRING;
	m_opts["wells-format"] = VT_STRING;
	m_opts["output-dir"] = VT_STRING;
	m_opts["explog-path"] = VT_STRING;
	m_opts["no-subdir"] = VT_BOOL;
	m_opts["dat-source-directory"] = VT_STRING;

	// GlobalDefaultsForBkgModel
	m_opts["gopt"] = VT_STRING;
	m_opts["bkg-dont-emphasize-by-compression"] = VT_BOOL;
	m_opts["xtalk"] = VT_STRING;
	m_opts["bkg-well-xtalk-name"] = VT_STRING;

	// LocalSigProcControl
	m_opts["bkg-kmult-adj-low-hi"] = VT_FLOAT;
	m_opts["kmult-low-limit"] = VT_FLOAT;
	m_opts["kmult-hi-limit"] = VT_FLOAT;

  // control bead selection
  m_opts["bkg-copy-stringency"] = VT_FLOAT;
  m_opts["bkg-min-sampled-beads"] = VT_INT;
  m_opts["bkg-max-rank-beads"] = VT_INT;
  m_opts["bkg-post-key-train"] = VT_INT;
  m_opts["bkg-post-key-step"] = VT_INT;

	m_opts["clonal-filter-bkgmodel"] = VT_BOOL;
	m_opts["bkg-use-proton-well-correction"] = VT_BOOL;
	m_opts["bkg-per-flow-time-tracking"] = VT_BOOL;

	m_opts["bkg-exp-tail-fit"] = VT_BOOL;
	m_opts["bkg-exp-tail-bkg-adj"] = VT_BOOL;
	m_opts["bkg-exp-tail-tau-adj"] = VT_BOOL;
	m_opts["bkg-exp-tail-bkg-limit"] = VT_FLOAT;
	m_opts["bkg-exp-tail-bkg-lower"] = VT_FLOAT;


	m_opts["bkg-pca-dark-matter"] = VT_BOOL;
	m_opts["regional-sampling"] = VT_BOOL;
	m_opts["regional-sampling-type"] = VT_INT;
	m_opts["num-regional-samples"] = VT_INT;
	m_opts["dark-matter-correction"] = VT_BOOL;
	m_opts["bkg-prefilter-beads"] = VT_BOOL;
	m_opts["vectorize"] = VT_BOOL;
	m_opts["bkg-ampl-lower-limit"] = VT_FLOAT;

	m_opts["limit-rdr-fit"] = VT_BOOL;
	m_opts["use-alternative-etbr-equation"] = VT_BOOL;
	m_opts["use-alternative-etbR-equation"] = VT_BOOL;

	m_opts["suppress-copydrift"]  = VT_BOOL;
	m_opts["use-safe-buffer-model"] = VT_BOOL;

  m_opts["stop-beads"] = VT_BOOL;

	m_opts["fitting-taue"] = VT_BOOL;
	m_opts["incorporation-type"] = VT_INT;

	m_opts["bkg-single-gauss-newton"] = VT_BOOL;

  m_opts["fit-region-kmult"] = VT_BOOL;
	m_opts["bkg-recompress-tail-raw-trace"] = VT_BOOL;

	// barcode experiment
	m_opts["barcode-flag"] = VT_BOOL;
	m_opts["barcode-radius"] = VT_FLOAT;
	m_opts["barcode-tie"] = VT_FLOAT;
	m_opts["barcode-penalty"] = VT_FLOAT;
	m_opts["barcode-debug"] = VT_BOOL;
	m_opts["barcode-spec-file"] = VT_STRING;
	m_opts["kmult-penalty"] = VT_FLOAT;

  m_opts["revert-regional-sampling"] = VT_BOOL;
  m_opts["always-start-slow"] = VT_BOOL;

	// double-tap control
	m_opts["double-tap-means-zero"] = VT_BOOL;

	// ProcessImageToWell
	m_opts["region-list"] = VT_VECTOR_INT;
    m_opts["wells-save-queue-size"] = VT_INT;
	m_opts["wells-save-as-ushort"] = VT_BOOL;
	m_opts["wells-convert-low"] = VT_FLOAT;
	m_opts["wells-convert-high"] = VT_FLOAT;
	m_opts["wells-save-number-copies"] = VT_BOOL;
	m_opts["wells-convert-with-copies"] = VT_BOOL;
	m_opts["bkg-washout-threshold"] = VT_FLOAT;
	m_opts["bkg-washout-flow-detection"] = VT_INT;

    m_opts["args-json"] = VT_STRING;
    m_opts["args-beadfind-json"] = VT_STRING;
    m_opts["thumbnail"] = VT_BOOL;
}

void ValidateOpts::Validate(const int argc, char *argv[])
{
	for(int i = 1; i < argc; ++i)
	{
		string s = argv[i];
		if(s == "-" || s == "--")
		{
			cerr << "ERROR: command line input \"-\" must be followed by a short option name (a letter) and \"--\" must be followed by a long option name." << endl;
			exit ( EXIT_FAILURE );
		}
		else if(s == "-v" || s == "--version")
		{
			fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "Analysis" ).c_str() );
			exit ( EXIT_SUCCESS );
		}
		else if(argv[i][0] == '-') // option name
		{
			if((!isdigit(argv[i][1])) && (argv[i][1] != '.'))
			{
                                if(s.length() > 2 && argv[i][1] != '-' && argv[i][2] != ' ' && argv[i][2] != '=') // handle mis-typing long option to short option
                                {
                                        fprintf ( stdout, "WARNING: %s may miss a leading - . Please check if it is a long option.\n", s.c_str());
                                }

				s = s.substr(1, s.length() - 1);
				if(argv[i][1] == '-') // long option
				{
					s = s.substr(1, s.length() - 1);
				}

				string value("");
				int index = s.find("=");
				int len = s.length();
				if(index > 0) // with value
				{
					value = s.substr(index + 1, len - index - 1);
					s = s.substr(0, index);
				}
				else if(i + 1 < argc)
				{
					if(argv[i + 1][0] != '-')
					{
						value = argv[i + 1];
					}
				}

				map<string, ValidTypes>::iterator iter = m_opts.find(s);
				if(iter == m_opts.end())
				{
					cerr << "ERROR: option " << argv[i] << " is unexpected/unconsumed." << endl;
					exit ( EXIT_FAILURE );
				}
			}
		}
	}
}
