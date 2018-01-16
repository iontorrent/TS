/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ionstats.h"

#include <string>
#include <fstream>
#include <stdio.h>
#include <fenv.h> // Floating point exceptions

#include "IonVersion.h"

using namespace std;


void IonstatsHelp()
{
  printf ("\n");
  printf ("ionstats %s-%s (%s) - Generate performance metrics and statistics for Ion sequences.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   ionstats <command> [options]\n");
  printf ("\n");
  printf ("Command: basecaller  Get statistics based on basecaller-generated quality values\n");
  printf ("         alignment   Get statistics based on alignment results\n");
  printf ("         tf          Get statistics for test fragments\n");
  printf ("         reduce      Merge multiple statistics files\n");
  printf ("         reduce-h5   Merge multiple hdf5 statistics files\n");
  printf ("\n");
}


void IonstatsReduceHelp()
{
  printf ("\n");
  printf ("ionstats %s-%s (%s) - Generate performance metrics and statistics for Ion sequences.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   ionstats reduce [options] <in1.json> [...]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -o,--output                FILE       output json file [required]\n");
  printf ("\n");
}


void IonstatsReduceH5Help()
{
  printf ("\n");
  printf ("ionstats %s-%s (%s) - Generate performance metrics and statistics for Ion sequences.\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());
  printf ("\n");
  printf ("Usage:   ionstats reduce-h5 [options] <in1.h5> [...]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -o,--output                FILE       output h5 file [required]\n");
  printf ("  -b,--merge-proton-blocks   BOOL       if true, Proton per-block read groups will be merged [true]\n");
  printf ("\n");
}



int IonstatsReduce(OptArgs &opts)
{
  string output_json_filename = opts.GetFirstString('o', "output", "");
  vector<string>  input_jsons;
  opts.GetLeftoverArguments(input_jsons);

  if(input_jsons.empty() or output_json_filename.empty()) {
    IonstatsReduceHelp();
    return 1;
  }

  ifstream in(input_jsons[0].c_str(), ifstream::in);
  if (!in.good()) {
    fprintf(stderr, "[ionstats] ERROR: cannot open %s\n", input_jsons[0].c_str());
    return 1;
  }
  Json::Value first_input_json;
  in >> first_input_json;
  in.close();

  if (!first_input_json.isMember("meta")) {
    fprintf(stderr, "[ionstats] ERROR: %s is not a valid input file for ionstats reduce\n", input_jsons[0].c_str());
    return 1;
  }
  string format_name = first_input_json["meta"].get("format_name","").asString();

  if (format_name == "ionstats_basecaller")
    return IonstatsBasecallerReduce(output_json_filename, input_jsons);
  if (format_name == "ionstats_tf")
    return IonstatsTestFragmentsReduce(output_json_filename, input_jsons);
  if (format_name == "ionstats_alignment")
    return IonstatsAlignmentReduce(output_json_filename, input_jsons);

  fprintf(stderr, "[ionstats] ERROR: %s is not a valid input file for ionstats reduce\n", input_jsons[0].c_str());
  return 1;
}



int IonstatsReduceH5(OptArgs &opts, const string &program_str)
{
  string output_h5_filename = opts.GetFirstString  ('o', "output", "");
  bool merge_proton_blocks  = opts.GetFirstBoolean ('b', "merge-proton-blocks", "true");
  vector<string>  input_h5_filename;
  opts.GetLeftoverArguments(input_h5_filename);

  if(input_h5_filename.empty() or output_h5_filename.empty()) {
    IonstatsReduceH5Help();
    return 1;
  }

  return IonstatsAlignmentReduceH5(output_h5_filename, input_h5_filename, merge_proton_blocks);
}



int main(int argc, const char *argv[])
{
  if(argc < 2) {
    IonstatsHelp();
    return 1;
  }

  string ionstats_command = argv[1];
  string program_str = string(argv[0]) + " " + string(argv[1]);
  OptArgs opts;
  opts.ParseCmdLine(argc-1, argv+1);
  // enable floating point exceptions during program execution
  if (opts.GetFirstBoolean('-', "float-exceptions", true)) {
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  } //*/

  if      (ionstats_command == "basecaller") return IonstatsBasecaller(opts);
  else if (ionstats_command == "alignment")  return IonstatsAlignment(opts, program_str);
  else if (ionstats_command == "tf")         return IonstatsTestFragments(opts);
  else if (ionstats_command == "reduce")     return IonstatsReduce(opts);
  else if (ionstats_command == "reduce-h5")  return IonstatsReduceH5(opts, program_str);
  else {
      fprintf(stderr, "ERROR: unrecognized ionstats command '%s'\n", ionstats_command.c_str());
      return 1;
  }
  return 0;
}



