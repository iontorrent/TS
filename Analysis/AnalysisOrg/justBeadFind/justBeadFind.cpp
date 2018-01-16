/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <armadillo>
#include <fenv.h> // Floating point exceptions

#include "CommandLineOpts.h"
#include "OptBase.h"
#include "Mask.h"
#include "Region.h"
#include "SeqList.h"
#include "TrackProgress.h"
#include "SlicedPrequel.h"
#include "SeparatorInterface.h"
#include "SetUpForProcessing.h"

#include "IonErr.h"
#include "ImageSpecClass.h"
#include "MaskFunctions.h"

void PrintHelp()
{
    printf ("\n");
    printf ("Usage: justBeadFind [options] [dat-source-directory]\n");
    printf ("\n");
	printf ("[dat-source-directory]: if there is no option --dat-source-directory, this directory string must be the last one in the commandline\n");
    printf ("General options:(VECTOR input must be comma separated string)\n");
    printf ("  -h,--help                                    print this help message and exit\n");
    printf ("  -v,--version                                 print version and exit\n");
    printf ("     --region-list           INT VECTOR OF 4   region list for SlicedPrequel setup []\n");
	printf ("\n");

	CommandLineOpts cmdOpts;
	cmdOpts.PrintHelp();

    printf ("\n");
    exit (EXIT_SUCCESS);
}

void DumpStartingStateOfProgram (int argc, char *argv[], TrackProgress &my_progress)
{
  char myHostName[128] = { 0 };
  gethostname (myHostName, 128);
  fprintf (stdout, "\n");
  fprintf (stdout, "Hostname = %s\n", myHostName);
  fprintf (stdout, "Start Time = %s", ctime (&my_progress.analysis_start_time));
  fprintf (stdout, "Version = %s.%s (%s) (%s)\n",
           IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
           IonVersion::GetGitHash().c_str(), IonVersion::GetBuildNum().c_str());
  fprintf (stdout, "Command line = ");
  for (int i = 0; i < argc; i++)
    fprintf (stdout, "%s ", argv[i]);
  fprintf (stdout, "\n");
  fflush (NULL);
}

void TheSilenceOfTheArmadillos(ofstream &null_ostream)
{
    // Disable armadillo warning messages.
  arma::set_stream_err1(null_ostream);
  arma::set_stream_err2(null_ostream);
}

int TrapAndDeprecateOldArgs(int argc, char *argv[], char *argv2[])
{
  int datind = -1;
  for(int i = 1; i < argc; ++i)
  {
	  string s = argv[i];
	  int index = s.find("--dat-source-directory");
	  if(index == 0)
	  {
		  datind = i;
	  }
  }

  //jz check to see if the last argument is dat_source_directory
  if(datind < 0) // there is no "--dat-source-directory"
  {
	  if(argc > 1) // must have arg
	  {
		  if(argv[argc - 1][0] == '-')
		  {
			  string s = argv[argc - 1];
			  if(s != "-h" && s != "-v" && s != "--help" && s != "--version")
			  {
				delete [] argv2;

				cerr << "ERROR: dat_source_directory must be provided at the end of the command line." << endl;
				exit ( EXIT_FAILURE );
			  }
		  }

		  if(argc != 2) 
		  {

			  if(argc == 3) // 2 args
			  {			  
				  if(argv[argc - 2][0] != '-') // not option + dat_source_directory
				  {
					  delete [] argv2;

					  cerr << "ERROR: both arguments of the command line are not option name. The last argument must be dat_source_directory and the other one must be an option." << endl;
					  exit ( EXIT_FAILURE );
				  }
				  else
				  {
					  string s = argv[argc - 2];
					  int index = s.find("=");
					  if(index < 0)
					  {
						  delete [] argv2;

						  cerr << "ERROR: the last argument must be dat_source_directory and the other one must be an option with assignment value." << endl;
						  exit ( EXIT_FAILURE );
					  }
				  }
			  }
			  else
			  {
				  if(argv[argc - 2][0] == '-')
				  {
					  string s = argv[argc - 2];
					  int index = s.find("=");
					  if(index < 0)
					  {
						  delete [] argv2;

						  cerr << "ERROR: the last argument must be dat_source_directory and the second last one must be an option with assignment value." << endl;
						  exit ( EXIT_FAILURE );
					  }
				  }
			  }
		  }
	  }
  }

  for(int i = 0; i < argc; ++i)
  {
	  size_t slen0 = strlen(argv[i]) + 1;	

	  string s0 = argv[i];

	  int index0 = s0.find("--region-size");
	  if(index0 == 0)
	  {
		  if(s0 == "--region-size")
		  {
			  argv2[i] = new char[slen0];
			  memcpy(argv2[i], argv[i], slen0);

			  ++i;
			  if(i >= argc)
			  {
				  for(int k = 0; k <= i ; ++k)
				  {
				     delete [] argv2[k];
				  }
				  delete [] argv2;

				  cerr << "*ERROR* - Region-size option must be followed by a value." << endl;
			      exit ( EXIT_FAILURE );
			  }

			  string s1 = argv[i];
			  int index1 = s1.find("x");
			  if(index1 > 0)
			  {
				  cout << "*WARNING* - You are using a deprecated format of region-size value. Please change " << s1 << " to ";
				  s1.replace(index1, 1, ",");
				  cout << s1 << endl;
			  }

			  int slen1 = strlen(argv[i]) + 1;
			  argv2[i] = new char[slen1];
			  sprintf(argv2[i], "%s", s1.c_str());
		  }
		  else
		  {
			  int index2 = s0.find("x");
			  if(index2 > 0)
			  {
				  cout << "*WARNING* - You are using a deprecated format of region-size value. Please change " << s0 << " to ";
				  s0.replace(index2, 1, ",");
				  cout << s0 << endl;
			  }
			  argv2[i] = new char[slen0];
			  sprintf(argv2[i], "%s", s0.c_str());
		  }
	  }
	  else if(s0 == "on")
	  {
		  cout << "*WARNING* - You are using a deprecated format of " << argv[i - 1] << " value. Please change on to true." << endl;
		  argv2[i] = new char[5];
		  sprintf(argv2[i], "true");
	  }
	  else if(s0 == "off")
	  {
		  cout << "*WARNING* - You are using a deprecated format of " << argv[i - 1] << " value. Please change off to false." << endl;
		  argv2[i] = new char[6];
		  sprintf(argv2[i], "false");
	  }
	  else
	  {
		  int indexon = s0.find("=on");
		  int indexoff = s0.find("=off");
		  if(indexon > 0 && indexon == ((int)(s0.length()) - 3))
		  {
			  cout << "*WARNING* - You are using a deprecated format of " << argv[i] << ". Please change on to true as ";

			  string s3 = s0.substr(0, s0.length() - 2);
			  s3 += "true";
			  		
			  int slen3 = s3.length() + 1;
			  argv2[i] = new char[slen3];
			  sprintf(argv2[i], "%s", s3.c_str());

			  cout << argv2[i] << endl;
		  }
		  else if(indexoff > 0 && indexoff == ((int)(s0.length()) - 4))
		  {
  			  cout << "*WARNING* - You are using a deprecated format of " << argv[i] << ". Please change off to false as ";

			  string s4 = s0.substr(0, s0.length() - 3);
			  s4 += "false";
			  		
			  int slen4 = s4.length() + 1;
			  argv2[i] = new char[slen4];
			  sprintf(argv2[i], "%s", s4.c_str());

			  cout << argv2[i] << endl;
		  }
		  else
		  {
			  argv2[i] = new char[slen0];
			  memcpy( argv2[i], argv[i], slen0);
		  }
	  }

	  // convert long option to lower case
	  string s5 = argv2[i];
	  int index5 = s5.find("--");
	  int index6 = s5.find("=");
	  if(0 == index5)
	  {
		  if(index6 < 2)
		  {
			  index6 = strlen(argv2[i]) + 1;
		  }

		  for(int j = 2; j < index6; ++j)
		  {
		     argv2[i][j] = tolower (argv2[i][j]);
		  }
	  }
  }
  return datind;
}

void LoadArgsJson(const string& argsJsonFile, Json::Value& json_params, bool thumbnail)
{
    Json::Value json_params0;

    ifstream ifs(argsJsonFile.c_str());
    Json::Reader reader;
    reader.parse(ifs, json_params0, false);
    ifs.close();

    Json::Value json_tn;

     Json::Value::Members groups = json_params0.getMemberNames();
     for(Json::Value::Members::iterator it1 = groups.begin(); it1 != groups.end(); ++it1)
     {
         if(*it1 == "chipType")
         {
             json_params["chipType"] = json_params0["chipType"];
         }
         else if(*it1 == "ThumbnailControl")
         {
             if(thumbnail)
             {
                 Json::Value::Members items = json_params0[*it1].getMemberNames();
                 for(Json::Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
                 {
                     string sname(*it2);
                     for(size_t i = 0; i < sname.size(); ++i)
                     {
                         if(sname[i] == '-')
                         {
                             sname[i] = '_';
                         }
                     }
                     json_tn[sname] = json_params0[*it1][*it2];
                 }
             }
         }
         else
         {
             Json::Value::Members items = json_params0[*it1].getMemberNames();
             for(Json::Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
             {
                 string sname(*it2);
                 for(size_t i = 0; i < sname.size(); ++i)
                 {
                     if(sname[i] == '-')
                     {
                         sname[i] = '_';
                     }
                 }
                 json_params[sname] = json_params0[*it1][*it2];
             }
         }
     }

     if(thumbnail)
     {
         Json::Value::Members items = json_tn.getMemberNames();
         for(Json::Value::Members::iterator it2 = items.begin(); it2 != items.end(); ++it2)
         {
             string sname(*it2);

             if(json_params.isMember(sname))
             {
                json_params[sname] = json_tn[*it2];
             }
         }
     }
}

/*************************************************************************************************
 *************************************************************************************************
 *
 *  Start of Main Function
 *
 *************************************************************************************************
 ************************************************************************************************/
int main (int argc, char *argv[])
{
  init_salute();

  ofstream null_ostream("/dev/null"); // must stay live for entire scope, or crash when writing
  TheSilenceOfTheArmadillos(null_ostream);

  TrackProgress my_progress;  
  DumpStartingStateOfProgram (argc,argv,my_progress);
   
  if(argc < 2)
  {
      PrintHelp();
  }

  for(int i = 1; i < argc; ++i)
  {
	  string s = argv[i];
	  if(s == "-" || s == "--")
	  {
	      cerr << "ERROR: command line input \"-\" must be followed by a short option name (a letter) and \"--\" must be followed by a long option name." << endl; 
		  exit ( EXIT_FAILURE );
	  }
	  else if(s == "-?" || s == "-h" || s == "--help")
	  {
	      PrintHelp();
	  }
  }

  ValidateOpts validater;
  validater.Validate(argc, argv);

  char** argv2 = new char*[argc];
  int datind = TrapAndDeprecateOldArgs(argc, argv, argv2);

  OptArgs opts;
  opts.ParseCmdLine(argc, (const char**)argv2);

  // enable floating point exceptions during program execution
  if (opts.GetFirstBoolean('-', "float-exceptions", false)) {
      cout << "JustBeadFind: Floating point exceptions enabled." << endl;
      feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  } //*/


  for(int k = 0; k < argc ; ++k)
  {
	  delete [] argv2[k];
  }
  delete [] argv2;
   
  Json::Value json_params;
  bool thumbnail = opts.GetFirstBoolean('-', "thumbnail", false);
  string argsJsonFile = opts.GetFirstString('-', "args-json", "");
  if(argsJsonFile.length() > 0)
  {
      struct stat sb0;
      if(!(stat(argsJsonFile.c_str(), &sb0) == 0 && S_ISREG(sb0.st_mode)))
      {
          cerr << "ERROR: " << argsJsonFile << " does not exist or it is not a regular file." << endl;
          exit ( EXIT_FAILURE );
      }

      LoadArgsJson(argsJsonFile, json_params, thumbnail);
  }
  else
  {
      cerr << "ERROR: --args-json must be provided." << endl;
      exit ( EXIT_FAILURE );
  }
  CommandLineOpts inception_state;
  inception_state.SetOpts(opts, json_params);
  inception_state.bfd_control.SetThumbnail(thumbnail);

  if(datind < 0) // there is no "--dat-source-directory"
  {
	  inception_state.sys_context.dat_source_directory = argv[argc - 1];
	  cout << "dat_source_directory = " << inception_state.sys_context.dat_source_directory << endl;
  }

  struct stat sb;
  if(!(stat(inception_state.sys_context.dat_source_directory, &sb) == 0 && S_ISDIR(sb.st_mode)))
  {
      cerr << "ERROR: " << inception_state.sys_context.dat_source_directory << " does not exist or it is not a directory." << endl; 
	  exit ( EXIT_FAILURE );
  }

  string chipType = GetParamsString(json_params, "chipType", "");
  ChipIdDecoder::SetGlobalChipId ( chipType.c_str() );

  SeqListClass my_keys;
  ImageSpecClass my_image_spec;
  SlicedPrequel my_prequel_setup;  

  SetUpOrLoadInitialState(inception_state, my_keys, my_progress, my_image_spec, my_prequel_setup);

  // Start logging process parameters & timing now that we have somewhere to log
  my_progress.InitFPLog(inception_state);

  // Write processParameters.parse file now that processing is about to begin
  my_progress.WriteProcessParameters(inception_state);

  // Do separator
  Region wholeChip(0, 0, my_image_spec.cols, my_image_spec.rows);
  IsolatedBeadFind( my_prequel_setup, my_image_spec, wholeChip, inception_state,
        inception_state.sys_context.GetResultsFolder(), inception_state.sys_context.analysisLocation,  my_keys, my_progress, chipType);

  exit (EXIT_SUCCESS);
}


