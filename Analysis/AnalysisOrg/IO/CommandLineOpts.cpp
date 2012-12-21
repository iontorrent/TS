/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string.h>
#include <stdio.h>
#include <getopt.h> // for getopt_long
#include <stdlib.h> //EXIT_FAILURE
#include <ctype.h>  //tolower
#include <libgen.h> //dirname, basename
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "CommandLineOpts.h"
#include "IonErr.h"

using namespace std;


void CommandLineOpts::PrintHelp()
{
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "Usage:\n" );
  fprintf ( stdout, "\tAnalysis [options][data_directory]\n" );
  fprintf ( stdout, "\tOptions:\n" );
  fprintf ( stdout, "\t\tSee man page for Analysis for complete list of options\n" );
  fprintf ( stdout, "\n" );
  exit ( EXIT_FAILURE );
}

void ModuleControlOpts::DefaultControl()
{

  BEADFIND_ONLY = false;
  USE_BKGMODEL = false;
  passTau = true;
  reusePriorBeadfind = false; // when true, skip new beadfind
}


void ObsoleteOpts::Defaults()
{
  NUC_TRACE_CORRECT = 0;
  USE_PINNED = false;

  neighborSubtract = 0;
}


CommandLineOpts::CommandLineOpts ( int argc, char *argv[] )
{
  // backup command line arguments
  numArgs = argc;
  argvCopy = ( char ** ) malloc ( sizeof ( char * ) * argc );
  for ( int i=0;i<argc;i++ )
    argvCopy[i] = strdup ( argv[i] );

  //Constructor
  if ( argc == 1 )
  {
    PrintHelp();
  }
  // helper pointer for loading options
  sPtr = NULL;
  /*---   options variables       ---*/

  // controls for individual modules - how we are to analyze

  bkg_control.DefaultBkgModelControl();
  bfd_control.DefaultBeadfindControl();

  img_control.DefaultImageOpts();

  // overall program flow control what to do?
  mod_control.DefaultControl();
  // contexts for program operation - what the state of the world is
  sys_context.DefaultSystemContext();
  loc_context.DefaultSpatialContext();
  flow_context.DefaultFlowFormula();
  key_context.DefaultKeys();

  // obsolete
  no_control.Defaults();

  /*---   end options variables   ---*/


  /*---   Parse command line options  ---*/
  GetOpts ( argc, argv );

}

CommandLineOpts::~CommandLineOpts()
{
  //Destructor

}

/*
 *  Use getopts to parse command line
 */
void CommandLineOpts::GetOpts ( int argc, char *argv[] )
{
  //DEBUG
  //fprintf (stdout, "Number of arguments: %d\n", argc);
  //for (int i=0;i<argc;i++)
  //    fprintf (stdout, "%s\n", argv[i]);
  //


  //@TODO this structure needs sorting or replacing badly!!!
  //@TODO: Note the bad, bad use of "optarg" a global variable to return values
  static struct option long_options[] =
  {

    //-------what do I need to know if I'm seeking help
    {"help",                    no_argument,    NULL,               'h'},
    {"version",                 no_argument,        NULL,               'v'},

    //---- where is everything?
    {"no-subdir",        no_argument,       &sys_context.NO_SUBDIR, 1},
    {"output-dir",       required_argument, NULL, 0},
    {"local-wells-file", required_argument, NULL, 0},
    {"well-stat-file",   required_argument, NULL, 0},
    {"stack-dump-file",  required_argument, NULL, 0},
    {"wells-format",     required_argument, NULL, 0},
    {"explog-path",      required_argument, NULL, 0},

//-------module execution control
    {"analysis-mode",     required_argument,  NULL,       0},
    {"from-wells",              required_argument,  NULL,               0}, // kept only to signal that it no longer works
    {"bfonly",          no_argument,    &mod_control.BEADFIND_ONLY,   1},
    {"from-beadfind",   no_argument,    NULL, 0},
    {"wellsfileonly", no_argument, NULL, 0},


    //---------key options
    {"libraryKey",        required_argument,  NULL,       0},
    {"librarykey",        required_argument,  NULL,       0},
    {"tfKey",         required_argument,  NULL,       0},
    {"tfkey",         required_argument,  NULL,       0},

    //----------flow order control
    {"flow-order",        required_argument,  NULL,           0},
    {"flowlimit",       required_argument,  NULL,       0},
    {"flowrange",      required_argument,  NULL,       0},
    {"start-flow-plus-interval",      required_argument,  NULL,       0},

//------------Beadfind options
    {"beadfindFile",            required_argument,  NULL,               'b'},
    {"beadfindfile",            required_argument,  NULL,               'b'},
    {"bfold",         no_argument,    &bfd_control.BF_ADVANCED,   0},
    {"noduds",          no_argument,    &bfd_control.noduds,   1},
    {"beadmask-categorized",          no_argument,    &bfd_control.maskFileCategorized,   1},
    {"beadfind-type",           required_argument,  NULL,               0},
    {"beadfind-basis",          required_argument,  NULL,               0},
    {"beadfind-dat",            required_argument,  NULL,               0},
    {"beadfind-bgdat",          required_argument,  NULL,               0},
    {"beadfind-sdasbf",          required_argument,  NULL,               0},
    {"beadfind-bfmult",          required_argument,  NULL,               0},
    {"beadfind-minlive",        required_argument,  NULL,               0},
    {"beadfind-minlivesnr",     required_argument,  NULL,               0},
    {"beadfind-min-lib-snr",    required_argument,  NULL,               0},
    {"beadfind-min-tf-snr",     required_argument,  NULL,               0},
    {"beadfind-lib-filt",    required_argument,  NULL,               0},
    {"beadfind-tf-filt",     required_argument,  NULL,               0},
    {"beadfind-lib-min-peak",    required_argument,  NULL,               0},
    {"beadfind-tf-min-peak",     required_argument,  NULL,               0},
    {"beadfind-skip-sd-recover",required_argument,  NULL,               0},
    {"beadfind-thumbnail",      required_argument,  NULL,               0},
    {"beadfind-lagone-filt",    required_argument,  NULL,               0},
    {"beadfind-diagnostics",    required_argument,  NULL,               0},
    {"beadfind-num-threads",    required_argument,  NULL,               0},
    {"beadfind-sep-ref",    required_argument,  NULL,               0},
    {"beadfind-gain-correction",    required_argument,  NULL,               0},
    {"bead-washout",            no_argument,    NULL,       0},
    {"use-beadmask",      required_argument,  NULL,       0},
    {"bkg-use-duds",  required_argument, NULL, 0},
    {"cfiedr-regions-size", required_argument, NULL, 0},
    {"block-size", required_argument, NULL, 0},


//---------------signal processing options
    {"readaheadDat",            required_argument,  NULL,             0},
    {"save-wells-freq",         required_argument,  NULL,             0},
    {"gpuWorkLoad",             required_argument,  NULL,             0},
    {"numcputhreads",           required_argument,  NULL,       0},
    {"gpu-single-flow-fit",           required_argument,  NULL,       0},
    {"gpu-multi-flow-fit",           required_argument,  NULL,       0},

    {"bkg-record",     no_argument,    &bkg_control.recordBkgModelData,  1},
    {"bkg-replay",     no_argument,    &bkg_control.replayBkgModelData,   1},
    {"restart-from",        required_argument,    NULL,   0},
    {"restart-next",        required_argument,    NULL,   0},
    {"no-restart-check",    no_argument,    NULL,   0},
    {"trim-ref-trace",    required_argument,    NULL,   0},
    {"region-list",    required_argument,    NULL,   0},
    {"bkg-debug-param",     required_argument,    NULL,   0},
    {"debug-all-beads",         no_argument,    &bkg_control.debug_bead_only,              0}, // turn off what is turned on
    {"region-vfrc-debug",         no_argument,    &bkg_control.region_vfrc_debug,              1}, 
    {"bkg-h5-debug",           required_argument,  NULL,               0},
    {"n-unfiltered-lib",        required_argument,        &bkg_control.unfiltered_library_random_sample,    1},
    {"bkg-dbg-trace",           required_argument,  NULL,               0},

    {"bkg-effort-level",        required_argument,  NULL,               0},
    {"xtalk-correction",required_argument,     NULL,  0},
    {"dark-matter-correction",required_argument,     NULL,  0},
    {"clonal-filter-bkgmodel",required_argument,     NULL,  0},
    {"bkg-use-duds",required_argument,     NULL,  0},
    {"bkg-use-proton-well-correction",required_argument,     NULL,  0},
    {"bkg-empty-well-normalization",required_argument,     NULL,  0},
    {"bkg-single-flow-retry-limit", required_argument,  NULL,       0},
    {"bkg-per-flow-time-tracking", required_argument,  NULL,       0},
    {"regional-sampling", required_argument,  NULL,       0},
    {"bkg-prefilter-beads", required_argument, NULL, 0},
    {"bkg-empty-well-normalization",required_argument,     NULL,  0},
    {"bkg-bfmask-update",required_argument,     NULL,  0},
    {"bkg-damp-kmult",required_argument,     NULL,  0},
    {"bkg-ssq-filter-region",required_argument,     NULL,  0},
    {"bkg-kmult-adj-low-hi",required_argument,     NULL,  0},
    {"bkg-emphasis",            required_argument,  NULL,               0},
    {"dntp-uM",                 required_argument,  NULL,               0},
    {"bkg-ampl-lower-limit",    required_argument,  NULL,               0},
    {"gopt",                    required_argument,  NULL,               0},
    {"xtalk",                   required_argument,  NULL,               0},
    {"krate",                   required_argument,  NULL,               0},
    {"kmax",                    required_argument,  NULL,               0},
    {"diffusion-rate",          required_argument,  NULL,               0},
    {"limit-rdr-fit",           no_argument,        &bkg_control.no_rdr_fit_first_20_flows,     1},
    {"fitting-taue",            required_argument,        NULL,                0},
    {"var-kmult-only",          no_argument,        &bkg_control.var_kmult_only, 1},
    {"generic-test-flag",          no_argument,        &bkg_control.generic_test_flag, 1},
    {"bkg-single-alternate",          no_argument,        &bkg_control.fit_alternate, 1},
    {"bkg-dont-emphasize-by-compression",          no_argument,        &bkg_control.emphasize_by_compression,0},
    {"time-half-speed",     no_argument,    NULL,   0},
    {"pass-tau",            required_argument,  NULL,             0},
    {"single-flow-projection-search",required_argument,     NULL,  0},


//-----------------spatial control
    {"analysis-region",         required_argument,  NULL,       0},
    {"cropped",                 required_argument,  NULL,       0},
    {"cropped-region-origin",   required_argument,  NULL,               0},
    {"region-size",             required_argument,  NULL,             0},


//------------------------image control
    {"do-sdat",                 required_argument,  NULL,               0},
    {"output-pinned-wells",     no_argument,    &img_control.outputPinnedWells,   0},
    {"img-gain-correct",required_argument,     NULL,  0},
    {"col-flicker-correct",required_argument,     NULL,  0},
    {"col-flicker-correct-verbose",required_argument,     NULL,  0},
    {"nnMask",          required_argument,  NULL,       0},
    {"nnmask",          required_argument,  NULL,       0},
    {"nnMaskWH",        required_argument,  NULL,       0},
    {"nnmaskwh",        required_argument,  NULL,       0},
    {"smoothing-file",       required_argument,  NULL,             0}, // (APB)
    {"smoothing",       optional_argument,  NULL,             0}, // (APB)
    {"ignore-checksum-errors",      no_argument,  NULL,           0},
    {"ignore-checksum-errors-1frame",   no_argument,  NULL,           0},
    {"nn-subtract-empties",     no_argument,    NULL,   0},
    {"frames",                  required_argument,  NULL,               'f'},
    {"flowtimeoffset",          required_argument,  NULL,               0},
    {"hilowfilter",             required_argument,  NULL,       0},
    {"total-timeout",           required_argument,  NULL,       0},
    {"threaded-file-access",    no_argument,  NULL,       0},



    //----obsolete options
    {"cycles",                  required_argument,  NULL,               'c'}, //Deprecated; use flowlimit
    {"nuc-correct",             no_argument,        &no_control.NUC_TRACE_CORRECT, 1},
    {"forceNN",         no_argument,    &no_control.neighborSubtract,  1},
    {"forcenn",         no_argument,    &no_control.neighborSubtract,  1},
    {"use-pinned",        no_argument,    &no_control.USE_PINNED,    1},
    // soak up annoying extra arguments from blackbbird obsolete parameters
    {"clonal-filter-solve", required_argument, NULL, 0},
    {"cfiedr-regions-size", required_argument, NULL, 0},
    {"block-size", required_argument, NULL,0},
    {"ppf-filter", required_argument, NULL, 0},
    {"cr-filter", required_argument, NULL, 0},
    

    //-----table termination
    {NULL,                      0,                  NULL,               0}
  };

  int c;
  int option_index = 0;

  while ( ( c = getopt_long ( argc, argv, "b:c:f:hi:k:m:p:R:v", long_options, &option_index ) ) != -1 )
  {
    switch ( c )
    {
      case ( 0 ) :
        {
          char *lOption = NULL;
	  lOption = strdup ( long_options[option_index].name );
          ToLower ( lOption );

          if ( long_options[option_index].flag != 0 ) {
	    free (lOption);
            break;
	  }

          // module control:  what are we doing overall?
          SetModuleControlOption ( lOption, long_options[option_index].name );

          // Image control options ---------------------------------------------------

          SetAnyLongImageProcessingOption ( lOption, long_options[option_index].name );

          // flow entry and manipulation --------------------------------------------

          SetFlowContextOption ( lOption, long_options[option_index].name );

          // keys - only two types for now --------------------------------------------
          SetLongKeyOption ( lOption, long_options[option_index].name );
          // end keys ------------------------------------------------------------------

          // Spatial reasoning about the chip, cropped area, etc -----------------------------------
          SetAnyLongSpatialContextOption ( lOption, long_options[option_index].name );

          // System context: file manipulation, directories and names -----------------------------------

          SetSystemContextOption ( lOption, long_options[option_index].name );

          // All beadfind options in this section, please ---------------------------------
          SetAnyLongBeadFindOption ( lOption, long_options[option_index].name );

          // All bkg_control options in this section, please ------------------------
          SetAnyLongSignalProcessingOption ( lOption, long_options[option_index].name );

          free ( lOption );

          break;
        }
        /*  End processing long options */

      case 'b':   //beadfind file name
        /*
        **  When this is set, we override the find-washouts default by
        **  setting the preRun filename to NULL.
        */
        snprintf ( bfd_control.preRunbfFileBase, 256, "%s", optarg );
        //sprintf (preRunbfFileBase, "");
        bfd_control.bfFileBase[0] = '\0';
        bfd_control.SINGLEBF = true;
        break;
      case 'c':
        fprintf ( stderr,"\n* * * * * * * * * * * * * * * * * * * * * * * * * *\n" );
        fprintf ( stderr, "The --cycles, -c keyword has been deprecated.\n"
                  "Use the --flowlimit keyword instead.\n" );
        fprintf ( stderr,"* * * * * * * * * * * * * * * * * * * * * * * * * *\n\n" );
        exit ( EXIT_FAILURE );
        break;
      case 'f':   // maximum frames
        long tmp_frame;
        if ( validIn ( optarg, &tmp_frame ) )
        {
          fprintf ( stderr, "Option Error: %c %s\n", c,optarg );
          exit ( EXIT_FAILURE );
        }
        else
        {
          img_control.maxFrames = ( int ) tmp_frame;
        }
        break;
      case 'h': // command help
        PrintHelp();
        break;
      case 'k':
        printf("Obsolete option: -k\n");
        break;

      case 'v':   //version
        fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "Analysis" ).c_str() );
        exit ( EXIT_SUCCESS );
        break;
      case '?':
        /* getopt_long already printed an error message.*/
        exit ( EXIT_FAILURE );
        break;
      default:
        fprintf ( stderr, "What have we here? (%c)\n", c );
        exit ( EXIT_FAILURE );
    }
  }

  PickUpSourceDirectory ( argc,argv );
}

// set up processing variables after cmd-line options are processed
void CommandLineOpts::SetUpProcessing()
{
  SetSysContextLocations();
  SetFlowContext(sys_context.explog_path);

  SetGlobalChipID ( sys_context.explog_path );

  loc_context.FindDimensionsByType ( sys_context.explog_path );
  img_control.SetWashFlow ( sys_context.explog_path );

  // now that we know chip type, can set if needed
  SetProtonDefault();

  printf ( "Use dud and empty wells as reference: %s\n",bkg_control.use_dud_and_empty_wells_as_reference ? "yes" : "no" );
  printf ( "Proton 1.wells correction enabled   : %s\n",bkg_control.proton_dot_wells_post_correction ? "yes" : "no" );
  printf ( "Empty well normalization enabled    : %s\n",bkg_control.empty_well_normalization ? "yes" : "no" );
  printf ( "Per flow t-mid-nuc tracking enabled : %s\n",bkg_control.per_flow_t_mid_nuc_tracking ? "yes" : "no" );
  printf ( "Regional Sampling : %s\n",bkg_control.regional_sampling ? "yes" : "no" );
  printf ( "Image gain correction enabled       : %s\n",img_control.gain_correct_images ? "yes" : "no" );
  printf ( "Col flicker correction enabled      : %s\n",img_control.col_flicker_correct ? "yes" : "no" );
  printf ( "timeout                             : %d\n",img_control.total_timeout);
  printf ( "Threaded file access for signal processsing : %s\n",img_control.threaded_file_access ? "yes" : "no" );
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

void CommandLineOpts::SetFlowContext ( char *explog_path )
{
  flow_context.DetectFlowFormula ( explog_path ); // Set up flow order expansion
}


// explicitly set global variable
// if we're going to do this
// do this >once< only at the beginning
void CommandLineOpts::SetGlobalChipID ( char *explog_path )
{
  char *chipType = GetChipId ( explog_path );
  ChipIdDecoder::SetGlobalChipId ( chipType ); // @TODO: bad coding style, function side effect setting global variable
  if (chipType) free (chipType);
}

void CommandLineOpts::PickUpSourceDirectory ( int argc, char *argv[] )
{
  // Pick up any non-option arguments (ie, source directory)
  //@TODO: note bad use of global variable optind
  for ( int c_index = optind; c_index < argc; c_index++ )
  {
    sys_context.dat_source_directory = argv[c_index];
    break; //cause we only expect one non-option argument
  }
}

void CommandLineOpts::SetProtonDefault()
{  // based on chip Id, set a few things.  I really wanted to do this in the constructors for the individual
  // option objects, but alas the chip Id is unknown until AFTER the command line is parsed

  //@TODO: global variable abuse here
  if ( ChipIdDecoder::GetGlobalChipId() == ChipId900 )
  {
    if ( !radio_buttons.use_dud_reference_set )
      bkg_control.use_dud_and_empty_wells_as_reference = false;

    if ( !radio_buttons.empty_well_normalization_set )
      bkg_control.empty_well_normalization = false;

    if ( !radio_buttons.single_flow_fit_max_retry_set )
      bkg_control.single_flow_fit_max_retry = 4;

    if ( !radio_buttons.gain_correct_images_set )
      img_control.gain_correct_images = true;

    if ( !radio_buttons.col_flicker_correct_set )
      img_control.col_flicker_correct = true;

    if ( !radio_buttons.per_flow_t_mid_nuc_tracking_set )
      bkg_control.per_flow_t_mid_nuc_tracking = true;

    if ( !radio_buttons.regional_sampling_set )
      bkg_control.regional_sampling = true;

    if ( !radio_buttons.use_proton_correction_set )
    {
      bkg_control.proton_dot_wells_post_correction = true;
      bkg_control.enableXtalkCorrection = false;
    }
    if (!radio_buttons.clonal_solve_bkg_set)
    {
      bkg_control.enableBkgModelClonalFilter = false;
    }

    // maybe we should actually do this via the gopt file?
    if ( !radio_buttons.amplitude_lower_limit_set )
    {
      bkg_control.AmplLowerLimit = -0.5;
    }
  }

}

std::string CommandLineOpts::GetCmdLine()
{
  std::string cmdLine = "";
  for ( int i = 0; i < numArgs; i++ )
  {
    cmdLine += argvCopy[i];
    cmdLine += " ";
  }
  return cmdLine;
}

void CommandLineOpts::SetModuleControlOption ( char *lOption, const char *original_name )
{

  if ( strcmp ( lOption, "analysis-mode" ) == 0 )
  {
    ToLower ( optarg );
    if ( strcmp ( optarg,"bfonly" ) == 0 )
    {
      mod_control.BEADFIND_ONLY = 1;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s=%s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "from-beadfind" ) == 0 )
  {
    mod_control.reusePriorBeadfind = true;
  }

  if ( strcmp ( lOption, "from-wells" ) == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n" );
    fprintf ( stderr, "Analysis executable no longer supports analysis from wells. Use BaseCaller executable instead.\n" );
    fprintf ( stderr, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n" );
    fprintf ( stderr, "\n" );
    exit ( EXIT_FAILURE );
  }
  if ( strcmp ( lOption, "wellsfileonly" ) == 0 )
  {
    printf ( "NOTE: wellsfileonly now redundant - this version of analysis does not do basecalling.\n" );
  }

  if ( strcmp ( lOption, "fitting-taue" ) == 0 )
    {
      if ( !strcmp ( optarg,"off" ) )
	{
	  bkg_control.fitting_taue = 0;
	}
      else if ( !strcmp ( optarg,"on" ) )
	{
	  bkg_control.fitting_taue = 1;
	}
      else
	{
	  fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
	  exit ( EXIT_FAILURE );
	}
    }

  if ( strcmp ( lOption, "pass-tau" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      mod_control.passTau = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      mod_control.passTau = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
}

void CommandLineOpts::SetFlowContextOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( lOption, "flow-order" ) == 0 )
  {
    if ( flow_context.flowOrder )
      free ( flow_context.flowOrder );
    flow_context.flowOrder = strdup ( optarg );
    flow_context.numFlowsPerCycle = strlen ( flow_context.flowOrder );
    flow_context.flowOrderOverride = true;
  }

  if ( strcmp ( original_name, "flowlimit" ) == 0 )
  {
    long tmp_flowlimit;
    if ( validIn ( optarg, &tmp_flowlimit ) )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    flow_context.SetFlowLimit( tmp_flowlimit );
  }

  if (strcmp (original_name, "start-flow-plus-interval")==0)
  {
    //Format: --start-flow-plus-interval=20,40
    // Process 40 flows starting at flow 20, ie flows 20 thru 59, inclusive
    // This is a 0 based index.  Without this argument, all flows are processed.
    // Flowlimit controls the >total number< of flows to process, across all chunks
    // this controls a particular "chunk" of processing
    int startingFlow = 0;
    int flow_interval = 0;
    sPtr = strchr ( optarg,',' );
    if ( sPtr )
    {
      int stat = sscanf ( optarg, "%d,%d", &startingFlow, &flow_interval );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: --%s=%s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
     }
    else
    {
      fprintf ( stderr, "Option Error: --%s=%s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }

    flow_context.SetFlowRange(startingFlow, flow_interval);

  }
}

void CommandLineOpts::SetSystemContextOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( lOption,"local-wells-file" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      sys_context.LOCAL_WELLS_FILE = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      sys_context.LOCAL_WELLS_FILE = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }



  if ( strcmp ( lOption, "well-stat-file" ) == 0 )
  {
    if ( sys_context.wellStatFile )
      free ( sys_context.wellStatFile );
    sys_context.wellStatFile = strdup ( optarg );
  }

  if ( strcmp ( lOption, "stack-dump-file" ) == 0 )
  {
    if ( sys_context.stackDumpFile )
      free ( sys_context.stackDumpFile );
    sys_context.stackDumpFile = strdup ( optarg );
  }

  if ( strcmp ( original_name, "wells-format" ) == 0 )
  {
    sys_context.wellsFormat = optarg;
    if ( sys_context.wellsFormat != "legacy" && sys_context.wellsFormat != "hdf5" )
    {
      fprintf ( stderr, "*Error* - Illegal option to --wells-format: %s, valid options are 'legacy' or 'hdf5'\n",
                sys_context.wellsFormat.c_str() );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( lOption, "output-dir" ) == 0 )
  {
    sys_context.wells_output_directory = strdup ( optarg );
  }

  if ( strcmp ( lOption, "explog-path" ) == 0 )
  {
    FILE *explog = fopen ( optarg,"r" );
    if ( explog != NULL )
      fclose ( explog );
    else
    {
      fprintf ( stderr, "Option Error: %s cannot open file %s\n", original_name, optarg );
      exit ( EXIT_FAILURE );
    }
    sys_context.explog_path = strdup ( optarg );
  }
}

void CommandLineOpts::SetLongKeyOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( lOption, "librarykey" ) == 0 )
  {
    key_context.libKey = ( char * ) malloc ( strlen ( optarg ) +1 );
    strcpy ( key_context.libKey, optarg );
    ToUpper ( key_context.libKey );
  }
  if ( strcmp ( lOption, "tfkey" ) == 0 )
  {
    key_context.tfKey = ( char * ) malloc ( strlen ( optarg ) +1 );
    strcpy ( key_context.tfKey, optarg );
    ToUpper ( key_context.tfKey );
  }
}

void CommandLineOpts::SetAnyLongSpatialContextOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( lOption, "region-size" ) == 0 )
  {
    sPtr = strchr ( optarg,'x' );
    if ( sPtr )
    {
      int stat = sscanf ( optarg, "%dx%d", &loc_context.regionXSize, &loc_context.regionYSize );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
      if ( (&loc_context.regionXSize <= 0) || (&loc_context.regionYSize <= 0) )
      {
        fprintf ( stderr, "Option Error: %s %s must be positive\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "cropped" ) == 0 )
  {
    if ( optarg )
    {
      loc_context.numCropRegions++;
      loc_context.cropRegions = ( Region * ) realloc ( loc_context.cropRegions, sizeof ( Region ) * loc_context.numCropRegions );
      int stat = sscanf ( optarg, "%d,%d,%d,%d",
                          &loc_context.cropRegions[loc_context.numCropRegions-1].col,
                          &loc_context.cropRegions[loc_context.numCropRegions-1].row,
                          &loc_context.cropRegions[loc_context.numCropRegions-1].w,
                          &loc_context.cropRegions[loc_context.numCropRegions-1].h );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "analysis-region" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%d,%d,%d,%d",
                          &loc_context.chipRegion.col,
                          &loc_context.chipRegion.row,
                          &loc_context.chipRegion.w,
                          &loc_context.chipRegion.h );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "cropped-region-origin" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%d,%d",
                          &loc_context.cropped_region_x_offset,
                          &loc_context.cropped_region_y_offset );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
}

void CommandLineOpts::SetAnyLongBeadFindOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( original_name, "beadfind-type" ) == 0 )
  {
    bfd_control.beadfindType = optarg;
    if ( bfd_control.beadfindType != "differential" )
    {
      fprintf ( stderr, "*Error* - Illegal option to --beadfind-type: %s, valid options are 'differential'\n",
                bfd_control.beadfindType.c_str() );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "use-beadmask" ) == 0 )
  {
    bfd_control.beadMaskFile = strdup ( optarg );
  }
  if ( strcmp ( lOption, "beadmask-categorized" ) == 0 )
  {
    bfd_control.maskFileCategorized = 1;
  }
  if ( strcmp ( original_name, "beadfind-basis" ) == 0 )
  {
    bfd_control.bfType = optarg;
    if ( bfd_control.bfType != "signal" && bfd_control.bfType != "buffer" )
    {
      fprintf ( stderr, "*Error* - Illegal option to --beadfind-basis: %s, valid options are 'signal' or 'buffer'\n",
                bfd_control.bfType.c_str() );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "beadfind-dat" ) == 0 )
  {
    bfd_control.bfDat = optarg;
  }
  if ( strcmp ( original_name, "beadfind-bgdat" ) == 0 )
  {
    bfd_control.bfBgDat = optarg;
  }
  if ( strcmp ( original_name, "beadfind-sdasbf" ) == 0 )
  {
    bfd_control.sdAsBf = atoi(optarg);
  }
  if ( strcmp ( original_name, "beadfind-bfmult" ) == 0 )
  {
    bfd_control.bfMult = atof(optarg);
  }
  if ( strcmp ( original_name, "beadfind-minlive" ) == 0 )
  {
    bfd_control.bfMinLiveRatio = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-minlivesnr" ) == 0 ||
       strcmp ( original_name, "beadfind-min-lib-snr" ) == 0 )
  {
    bfd_control.bfMinLiveLibSnr = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-min-tf-snr" ) == 0 )
  {
    bfd_control.bfMinLiveTfSnr = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-tf-min-peak" ) == 0 )
  {
    bfd_control.minTfPeakMax = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-lib-min-peak" ) == 0 )
  {
    bfd_control.minLibPeakMax = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-lib-filt" ) == 0 )
  {
    bfd_control.bfLibFilterQuantile = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-tf-filt" ) == 0 )
  {
    bfd_control.bfTfFilterQuantile = atof ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-skip-sd-recover" ) == 0 )
  {
    bfd_control.skipBeadfindSdRecover = atoi ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-thumbnail" ) == 0 )
  {
    bfd_control.beadfindThumbnail = atoi ( optarg );
  }
  if ( strcmp ( original_name, "beadfind-sep-ref" ) == 0 )
  {
    if (strcmp(optarg, "on") == 0) {
      bfd_control.beadfindUseSepRef = 1;
    }
    else if (strcmp(optarg, "off") == 0) {
      bfd_control.beadfindUseSepRef = 0;
    }
    else {
      bfd_control.beadfindUseSepRef = atoi ( optarg );
    }
  }
  if ( strcmp ( original_name, "beadfind-lagone-filt" ) == 0 )
  {
    bfd_control.beadfindLagOneFilt = atoi ( optarg );
  }
  if ( strcmp ( original_name, "do-sdat" ) == 0 )
  {
    img_control.doSdat = atoi ( optarg ) > 0;
  }
  if ( strcmp ( original_name, "beadfind-diagnostics" ) == 0 )
  {
    bfd_control.bfOutputDebug = atoi ( optarg );
  }
  if ( strcmp ( original_name, "bead-washout" ) == 0 )
  {
    bfd_control.SINGLEBF = false;
  }
  if ( strcmp ( original_name, "beadfind-gain-correction" ) == 0 )
  {
    bfd_control.gainCorrection = atoi( optarg );
  }
  if ( strcmp ( original_name, "beadfind-num-threads" ) == 0 )
  {
    bfd_control.numThreads = atoi ( optarg );
  }
}


void CommandLineOpts::SetAnyLongSignalProcessingOption ( char *lOption, const char *original_name )
{
  if ( strcmp ( lOption, "save-wells-freq" ) == 0 )
  {
    bkg_control.saveWellsFrequency = atoi ( optarg );
    fprintf ( stdout, "Saving wells every %d blocks.\n", bkg_control.saveWellsFrequency );
    if ( bkg_control.saveWellsFrequency < 1 || bkg_control.saveWellsFrequency > 100 )
    {
      fprintf ( stderr, "Option Error, must be between 1 and 100: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }

  }
  if ( strcmp ( lOption, "clonal-filter-bkgmodel" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.enableBkgModelClonalFilter = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.enableBkgModelClonalFilter = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.clonal_solve_bkg_set = true;
  }
  if ( strcmp ( lOption, "bkg-use-duds" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.use_dud_and_empty_wells_as_reference = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.use_dud_and_empty_wells_as_reference = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.use_dud_reference_set=true;
  }
  if ( strcmp ( lOption, "bkg-use-proton-well-correction" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.proton_dot_wells_post_correction = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.proton_dot_wells_post_correction = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.use_proton_correction_set=true;
  }
  if ( strcmp ( lOption, "bkg-per-flow-time-tracking" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.per_flow_t_mid_nuc_tracking = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.per_flow_t_mid_nuc_tracking = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.per_flow_t_mid_nuc_tracking_set = true;
  }

  if ( strcmp ( lOption, "regional-sampling" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.regional_sampling = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.regional_sampling = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.regional_sampling_set = true;
  }
  
  if ( strcmp ( lOption, "bkg-prefilter-beads" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.prefilter_beads = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.prefilter_beads = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    
  }
  if ( strcmp ( lOption, "bkg-empty-well-normalization" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.empty_well_normalization = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.empty_well_normalization = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.empty_well_normalization_set = true;
  }
  if ( strcmp ( lOption, "bkg-bfmask-update" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.updateMaskAfterBkgModel = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.updateMaskAfterBkgModel = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( lOption, "trim-ref-trace" ) == 0 )
  {
    if ( !strcmp ( optarg, "off") )      // usage: --trim-ref-trace off
    {
      bkg_control.do_ref_trace_trim = false;
    }
    else if ( !strcmp (optarg, "on") )   // usage: --trim-ref-trace on 
    {
      bkg_control.do_ref_trace_trim = true;    // default values will be used
    }
    else                                 // usage --trim-ref-trace 10,50,.2
    {
      bkg_control.do_ref_trace_trim = true;
      int stat = sscanf ( optarg, "%f,%f,%f", &bkg_control.span_inflator_min,
			  &bkg_control.span_inflator_mult,
			  &bkg_control.cutoff_quantile);
      if ( stat != 3 ) {
	fprintf ( stderr, "Option error: If not 'on' or 'off', numeric args to --trim-ref-trace must be 3 comma-delimited floats: span_inflator_min, span_inflator_mult, cutff_quantile");
	exit( EXIT_FAILURE );
      }
    }
    if (bkg_control.do_ref_trace_trim)
      fprintf(stdout, "Reference trimming enabled with options: span_inflator_min = %f, span_inflator_mult = %f, cutoff_quantile = %f\n", bkg_control.span_inflator_min, bkg_control.span_inflator_mult, bkg_control.cutoff_quantile);
  }

  if ( strcmp ( lOption, "restart-from" ) == 0 )
  {
    bkg_control.restart_from = optarg;  // path to read restart info from
  }
  if ( strcmp ( lOption, "restart-next" ) == 0 )
  {
    bkg_control.restart_next = optarg;  // path to write restart info to
  }
  if ( strcmp ( lOption, "no-restart-check" ) == 0 )
  {
    bkg_control.restart_check = false;
  }

  if ( strcmp ( lOption, "region-list" ) == 0 )
  {
    bkg_control.region_list = optarg;  // path to read regions from
  }

  if ( strcmp ( lOption, "xtalk-correction" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.enableXtalkCorrection = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.enableXtalkCorrection = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
    if ( strcmp ( lOption, "dark-matter-correction" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.enable_dark_matter = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.enable_dark_matter = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "bkg-debug-param" ) == 0 )
  {
    bkg_control.bkgModelHdf5Debug = atoi ( optarg );
  }

  if ( strcmp ( original_name, "bkg-record" ) == 0 )
  {
    if ( bkg_control.recordBkgModelData &  bkg_control.replayBkgModelData )
    {
      fprintf ( stderr, "Option Error, bkg-replay and bkg-record cannot both be on\n" );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( original_name, "bkg-replay" ) == 0 )
  {
    if ( bkg_control.recordBkgModelData &  bkg_control.replayBkgModelData )
    {
      fprintf ( stderr, "Option Error, bkg-replay and bkg-record cannot both be on\n" );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( lOption, "bkg-damp-kmult" ) == 0 )
  {
    int stat = sscanf ( optarg, "%f", &bkg_control.damp_kmult );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.damp_kmult < 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "bkg-ssq-filter-region" ) == 0 )
  {
    int stat = sscanf ( optarg, "%f", &bkg_control.ssq_filter );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.damp_kmult < 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "bkg-kmult-adj-low-hi" ) == 0 )
  {
    int stat = sscanf ( optarg, "%f,%f,%f", &bkg_control.krate_adj_threshold, &bkg_control.kmult_low_limit,&bkg_control.kmult_hi_limit );
    if ( stat != 3 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.damp_kmult < 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a non-negative value (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "bkg-emphasis" ) == 0 )
  {
    sPtr = strchr ( optarg,',' );
    if ( sPtr )
    {
      int stat = sscanf ( optarg, "%f,%f", &bkg_control.bkg_model_emphasis_width, &bkg_control.bkg_model_emphasis_amplitude );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "dntp-uM" ) == 0 ) // this is now a vector of 4, one per nuc TACG
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%f,%f,%f,%f", &bkg_control.dntp_uM[0],&bkg_control.dntp_uM[1],&bkg_control.dntp_uM[2],&bkg_control.dntp_uM[3] );
      if ( stat != 4)
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "bkg-ampl-lower-limit" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%f", &bkg_control.AmplLowerLimit );
      if ( stat != 1 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
      radio_buttons.amplitude_lower_limit_set = true;
  }
  if ( strcmp ( original_name, "bkg-effort-level" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%d", &bkg_control.bkgModelMaxIter );
      if ( stat != 1 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }

      if ( bkg_control.bkgModelMaxIter < 5 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "gopt" ) == 0 )
  {
    bkg_control.gopt = optarg;
    if ( strcmp ( bkg_control.gopt, "disable" ) == 0 || strcmp ( bkg_control.gopt, "opt" ) == 0 );
    else
    {
      FILE *gopt_file = fopen ( bkg_control.gopt,"r" );
      if ( gopt_file != NULL )
        fclose ( gopt_file );
      else
      {
        fprintf ( stderr, "Option Error: %s cannot open file %s\n", original_name,optarg );
        exit ( 1 );
      }
    }
  }
  if ( strcmp ( original_name, "xtalk" ) == 0 )
  {
    bkg_control.xtalk = optarg;
    if ( strcmp ( bkg_control.xtalk, "disable" ) == 0 || strcmp ( bkg_control.xtalk, "opt" ) == 0 );
    else
    {
      bkg_control.enableXtalkCorrection=true;
      FILE *tmp_file = fopen ( bkg_control.xtalk,"r" );
      if ( tmp_file != NULL )
        fclose ( tmp_file );
      else
      {
        fprintf ( stderr, "Option Error: %s cannot open file %s\n", original_name,optarg );
        exit ( 1 );
      }
    }
  }

  if ( strcmp ( original_name, "krate" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%f,%f,%f,%f", &bkg_control.krate[0],&bkg_control.krate[1],&bkg_control.krate[2],&bkg_control.krate[3] );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( 1 );
      }

      for ( int i=0;i < 3;i++ )
      {
        if ( ( bkg_control.krate[i] < 0.01 ) || ( bkg_control.krate[i] > 100.0 ) )
        {
          fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
          exit ( 1 );
        }
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( 1 );
    }
  }
  if ( strcmp ( original_name, "kmax" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%f,%f,%f,%f", &bkg_control.kmax[0],&bkg_control.kmax[1],&bkg_control.kmax[2],&bkg_control.kmax[3] );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( 1 );
      }

      for ( int i=0;i < 3;i++ )
      {
        if ( ( bkg_control.kmax[i] < 0.01 ) || ( bkg_control.kmax[i] > 100.0 ) )
        {
          fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
          exit ( 1 );
        }
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( 1 );
    }
  }
  if ( strcmp ( original_name, "diffusion-rate" ) == 0 )
  {
    if ( optarg )
    {
      int stat = sscanf ( optarg, "%f,%f,%f,%f", &bkg_control.diff_rate[0],&bkg_control.diff_rate[1],&bkg_control.diff_rate[2],&bkg_control.diff_rate[3] );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( 1 );
      }

      for ( int i=0;i < 3;i++ )
      {
        if ( ( bkg_control.diff_rate[i] < 0.01 ) || ( bkg_control.diff_rate[i] > 1000.0 ) )
        {
          fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
          exit ( 1 );
        }
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( 1 );
    }
  }

  if ( strcmp ( lOption, "gpuworkload" ) == 0 )
  {
    int stat = sscanf ( optarg, "%f", &bkg_control.gpuWorkLoad );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( ( bkg_control.gpuWorkLoad > 1 ) || ( bkg_control.gpuWorkLoad < 0 ) )
    {
      fprintf ( stderr, "Option Error: %s must specify a value between 0 and 1 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "gpu-single-flow-fit" ) == 0 )
  {
    int stat = sscanf ( optarg, "%d", &bkg_control.gpuSingleFlowFit );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.gpuSingleFlowFit != 0 && bkg_control.gpuSingleFlowFit != 1 )
    {
      fprintf ( stderr, "Option Error: %s must be either 0 or 1 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "gpu-multi-flow-fit" ) == 0 )
  {
    int stat = sscanf ( optarg, "%d", &bkg_control.gpuMultiFlowFit);
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.gpuMultiFlowFit != 0 && bkg_control.gpuMultiFlowFit != 1 )
    {
      fprintf ( stderr, "Option Error: %s must be either 0 or 1 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( lOption, "numcputhreads" ) == 0 )
  {
    int stat = sscanf ( optarg, "%d", &bkg_control.numCpuThreads );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.numCpuThreads <= 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "bkg-single-flow-retry-limit" ) == 0 )
  {
    int stat = sscanf ( optarg, "%d", &bkg_control.single_flow_fit_max_retry );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.single_flow_fit_max_retry < 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a value greater than or equal to 0 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.single_flow_fit_max_retry_set = true;
  }
  if ( strcmp ( lOption, "readaheaddat" ) == 0 )
  {
    int stat = sscanf ( optarg, "%d", &bkg_control.readaheadDat );
    if ( stat != 1 )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else if ( bkg_control.readaheadDat <= 0 )
    {
      fprintf ( stderr, "Option Error: %s must specify a value greater than 0 (%s invalid).\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "bkg-dbg-trace" ) == 0 )
  {
    sPtr = strchr ( optarg,'x' );
    if ( sPtr )
    {
      Region dbg_reg;

      int stat = sscanf ( optarg, "%dx%d", &dbg_reg.col, &dbg_reg.row );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }

      bkg_control.BkgTraceDebugRegions.push_back ( dbg_reg );
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "single-flow-projection-search" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      bkg_control.useProjectionSearchForSingleFlowFit = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      bkg_control.useProjectionSearchForSingleFlowFit = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }

  if ( strcmp ( lOption, "time-half-speed" ) == 0 )
  {
    bkg_control.choose_time=1;
  }
}

void CommandLineOpts::SetAnyLongImageProcessingOption ( char *lOption, const char *original_name )
{
    if ( strcmp ( lOption, "col-flicker-correct" ) == 0 )
    {
      if ( !strcmp ( optarg,"off" ) )
      {
        img_control.col_flicker_correct = false;
      }
      else if ( !strcmp ( optarg,"on" ) )
      {
        img_control.col_flicker_correct = true;
      }
      else
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
      radio_buttons.col_flicker_correct_set = true;
    }

    if ( strcmp ( lOption, "col-flicker-correct-verbose" ) == 0 )
    {
      if ( !strcmp ( optarg,"off" ) )
      {
        img_control.col_flicker_correct_verbose = false;
      }
      else if ( !strcmp ( optarg,"on" ) )
      {
        img_control.col_flicker_correct_verbose = true;
      }
      else
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }

  if ( strcmp ( lOption, "img-gain-correct" ) == 0 )
  {
    if ( !strcmp ( optarg,"off" ) )
    {
      img_control.gain_correct_images = false;
    }
    else if ( !strcmp ( optarg,"on" ) )
    {
      img_control.gain_correct_images = true;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    radio_buttons.gain_correct_images_set = true;
  }
  if ( strcmp ( lOption, "smoothing-file" ) == 0 )   // Tikhonov Smoothing (APB)
  {
    strncpy ( img_control.tikSmoothingFile, optarg, 256 );
  }
  if ( strcmp ( lOption, "smoothing" ) == 0 )   // Tikhonov Smoothing (APB)
  {
    if ( optarg == NULL )   // use default
    {
      strncpy ( img_control.tikSmoothingInternal, "10", 32 );
    }
    else
    {
      strncpy ( img_control.tikSmoothingInternal, optarg, 32 );
    }
  }
  if ( strcmp ( lOption, "ignore-checksum-errors" ) == 0 )
  {
    img_control.ignoreChecksumErrors |= 0x01;
  }
  if ( strcmp ( lOption, "ignore-checksum-errors-1frame" ) == 0 )
  {
    img_control.ignoreChecksumErrors |= 0x02;
  }
  if ( strcmp ( lOption, "output-pinned-wells" ) == 0 )
  {
    img_control.outputPinnedWells = 1;
  }
  if ( strcmp ( lOption, "flowtimeoffset" ) == 0 )
  {
    long tmp_val;
    if ( validIn ( optarg, &tmp_val ) )
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
    else
    {
      img_control.flowTimeOffset = ( int ) tmp_val;
    }
  }
  if ( strcmp ( lOption, "nn-subtract-empties" ) == 0 )
  {
    img_control.nn_subtract_empties = 1;
  }
  if ( strcmp ( lOption, "nnmask" ) == 0 )
  {
    sPtr = strchr ( optarg,',' );
    if ( sPtr )
    {
      int inner = 1, outer = 3;
      int stat = sscanf ( optarg, "%d,%d", &inner, &outer );
      if ( stat != 2 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
      img_control.NNinnerx = inner;
      img_control.NNinnery = inner;
      img_control.NNouterx = outer;
      img_control.NNoutery = outer;
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( lOption, "nnmaskwh" ) == 0 )
  {
    sPtr = strchr ( optarg,',' );
    if ( sPtr )
    {
      int stat = sscanf ( optarg, "%d,%d,%d,%d", &img_control.NNinnerx, &img_control.NNinnery, &img_control.NNouterx, &img_control.NNoutery );
      if ( stat != 4 )
      {
        fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
        exit ( EXIT_FAILURE );
      }
    }
    else
    {
      fprintf ( stderr, "Option Error: %s %s\n", original_name,optarg );
      exit ( EXIT_FAILURE );
    }
  }
  if ( strcmp ( original_name, "hilowfilter" ) == 0 )
  {
    ToLower ( optarg );
    if ( strcmp ( optarg, "true" ) == 0 ||
         strcmp ( optarg, "on" ) == 0 ||
         atoi ( optarg ) == 1 )
    {
      img_control.hilowPixFilter = 1;
    }
    else
    {
      img_control.hilowPixFilter = 0;
    }
  }
  if ( strcmp ( lOption, "total-timeout" ) == 0 )
  {
    img_control.total_timeout = atoi ( optarg );
  }
  if ( strcmp ( lOption, "threaded-file-access" ) == 0 )
  {
    img_control.threaded_file_access = 1;
  }
}
