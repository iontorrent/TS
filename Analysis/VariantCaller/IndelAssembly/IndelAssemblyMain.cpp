/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#include "IndelAssembly.h"

int main(int argc, char* argv[])
{

  printf("tvcassembly %s-%s (%s) - Torrent Variant Caller - Long Indel Assembly\n\n",
         IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());

  IndelAssemblyArgs parsed_opts(argc, argv);

  ReferenceReader reference_reader;
  reference_reader.Initialize(parsed_opts.reference);

  TargetsManager targets_manager;
  targets_manager.Initialize(reference_reader, parsed_opts.target_file);
  vector<MergedTarget>::iterator indel_target = targets_manager.merged.begin();


  BamMultiReader bam_reader;
  if (!bam_reader.SetExplicitMergeOrder(BamMultiReader::MergeByCoordinate)) {
    cerr << "ERROR: Could not set merge order to BamMultiReader::MergeByCoordinate" << endl;
    exit(1);
  }
  if (!bam_reader.Open(parsed_opts.bams)) {
    cerr << "ERROR: Could not open input BAM file(s) : " << bam_reader.GetErrorString() << endl;
    exit(1);
  }

  SampleManager sample_manager;
  sample_manager.Initialize(bam_reader.GetHeader(), parsed_opts.sample_name, parsed_opts.force_sample_name, parsed_opts.multisample);

  IndelAssembly indel_assembly(&parsed_opts, &reference_reader, &sample_manager, &targets_manager);

  BamAlignment alignment;
  while (bam_reader.GetNextAlignment(alignment)) {  
    if (!indel_assembly.processRead(alignment, indel_target)) {break;}
  }
  indel_assembly.onTraversalDone(true);

  bam_reader.Close();


  return 0;
}




