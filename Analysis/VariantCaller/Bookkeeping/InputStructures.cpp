/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "InputStructures.h"
#include "ExtendedReadInfo.h"

InputStructures::InputStructures() {
  flowKey = "TCAG";
  flowOrder = "";
  flowSigPresent=false;
  DEBUG = 0;
  nFlows = 0;
  min_map_qv = 4;
}


void InputStructures::bam_initialize(vector<string> bams ) {
  if (!bamMultiReader.Open(bams)) {
    cerr << " ERROR: fail to open input bam files " << bams.size() << endl;
    exit(-1);
  }

  if (!bamMultiReader.LocateIndexes()) {
    cerr << "ERROR: Unable to locate BAM Index (bai) files for input BAM files specified " << endl;
    exit(-1);
  }


  samHeader = bamMultiReader.GetHeader();
  if (!samHeader.HasReadGroups()) {
    bamMultiReader.Close();
    cerr << "ERROR: there is no read group in BAM files specified" << bams.size() << endl;
    exit(1);
  }

//TODO Need to handle multiple BAM files with different flowOrders, at least throw an error for now.
  for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {
    if (itr->HasFlowOrder()) {
      flowOrder = itr->FlowOrder;
      if (bamFlowOrderVector.empty())
              bamFlowOrderVector.push_back(flowOrder);
      else { //check if the flowOrder is the same if not throw an error, for now we dont support bams with different flow orders
              vector<string>::iterator it = std::find(bamFlowOrderVector.begin(), bamFlowOrderVector.end(), flowOrder);
              if (it == bamFlowOrderVector.end()) {
                cerr << "FATAL ERROR: BAM files specified as input have different flow orders. Currently tvc supports only BAM files with same flow order. " << endl;
                exit(-1);

              }
      }
      flowKey = itr->KeySequence;

    }

  }


  nFlows = flowOrder.length();

  if (nFlows > 0) {
    flowSigPresent = true;
    treePhaserFlowOrder.SetFlowOrder(flowOrder, nFlows);
    key.Set(treePhaserFlowOrder, flowKey, "key");
  }


  string bamseq;
  int ref_id = 0;

  for (BamTools::SamSequenceIterator itr = samHeader.Sequences.Begin(); itr != samHeader.Sequences.End(); ++itr) {
    bamseq = itr->Name;
    if (DEBUG)
      cout << "Contig Name in BAM file : " << bamseq << endl;
    sequences.push_back(bamseq);
    sequence_to_bam_ref_id[bamseq] = ref_id++;

    bool reffound = false;
    //check if reference fasta file provoided has the sequence name , if not exit
    for (unsigned int i = 0; i < refSequences.size(); i++) {
      if (refSequences[i].compare(bamseq) == 0) {
        reffound = true;
        break;

      }
    }
    if (!reffound) {
      cerr << "FATAL ERROR: Sequence Name: " << bamseq << " in BAM file is not found in Reference fasta file provoided " << endl;
      //exit(-1);
    }


  }



  if (DEBUG)
    cout << "Finished Initialization - flowPresent = " << flowSigPresent << " flowOrder  = " << flowOrder << endl;


}





void InputStructures::read_fasta(string chr_file, map<string, string> & chr_seq) {
  //chr_seq.clear();
  map<string,string>::iterator it;

  string temp_chr_seq = "";
  string line = "";
  ifstream inFile(chr_file.c_str());
  string chr_num = "";
  int chrNumber = 0;
  string chrName;
  bool firstContigFound = false;

  if (!inFile.is_open()) {
    cout << "Error opening file: " << chr_file << endl;
    return;
  }
  getline(inFile,line);

  it = chr_seq.begin();

  // Read and process records
  while (!inFile.eof() && line.length() > 0) {
    if (line.at(0) == '#' && !firstContigFound) {
      getline(inFile,line);
    } else
      if (line.at(0) == '>' || line.at(0) == '<') {
        firstContigFound = true;

        if (chrNumber > 0) {
          //chr_seq.push_back(temp_chr_seq);
          chr_seq.insert(it, pair<string,string>(chrName,temp_chr_seq));
          if (DEBUG)
            cout << "Chromosome Name = " << chrName << endl;

          if (temp_chr_seq.length() > 0) {
            if (DEBUG)
              cout << temp_chr_seq.length() << endl;
            temp_chr_seq.clear();
          }

          //chrNumber++;
        }

        chrNumber++;
        //find if there are more than contig name in the line
        int firstspace = line.find(" ");
        chrName = line.substr(1, firstspace-1);
        if (chrName.length() == 0) {
          cerr << " Reference csfasta file provided has no contig name : " << line << endl;
          exit(-1);
        }
        if (DEBUG)
          cout << " Chr Name found in Reference csfasta : " << chrName << endl;
        refSequences.push_back(chrName);

        getline(inFile, line);

      } else {
        temp_chr_seq.append(line);
        getline(inFile,line);
      }

  }

  if (temp_chr_seq.length() > 0) {
    //cout << temp_chr_seq.length() << endl;cout << "Dima's seq : " << temp_chr_seq.substr(279750,100) << endl;
    chr_seq.insert(it, pair<string,string>(chrName,temp_chr_seq));
  }

  inFile.close();
}

void InputStructures::BringUpReferenceData(ExtendParameters &parameters){
   
  DEBUG = parameters.program_flow.DEBUG;
  min_map_qv = parameters.MQL0;

  cout << "Loading reference." << endl;
  read_fasta(parameters.fasta, reference_contigs);
  cout << "Loaded reference. Ref length: " << reference_contigs.size() << endl;

  bam_initialize(parameters.bams);

  if (parameters.sseMotifsProvided) {
    cout << "Loading systematic error contexts." << endl;
    read_error_motifs(parameters.sseMotifsFileName);
    cout << "Loaded." << endl;
  }
}

// this is officially bad, as the scope of the reference is uncertain
// but since it is signed in blood that our reference_contigs will persist for the whole program
// we can pretend to be happy
string & InputStructures::ReturnReferenceContigSequence(vcf::Variant ** current_variant){
  map<string,string>::iterator itr = reference_contigs.find((*current_variant)->sequenceName);
  if (itr == reference_contigs.end()) {
    cerr << "FATAL: Reference sequence for Contig " << (*current_variant)->sequenceName << " , not found in reference fasta file " << endl;
    exit(-1);
  }
  
  return(itr->second);
}

void InputStructures::ShiftLocalBamReaderToCorrectBamPosition(BamTools::BamMultiReader &local_bamReader, vcf::Variant **current_variant) {
  //
  // Step 1: Use (*current_variant)->sequenceName and (*current_variant)->position to jump to the right place in BAM reader
  //

  if (sequence_to_bam_ref_id.find((*current_variant)->sequenceName) == sequence_to_bam_ref_id.end()) {
    cerr << "FATAL: Reference sequence for Contig " << (*current_variant)->sequenceName << " not found in BAM file " << endl;
    exit(-1);
  }
  int bam_ref_id = sequence_to_bam_ref_id[(*current_variant)->sequenceName];
  if (!local_bamReader.Jump(bam_ref_id, (*current_variant)->position)) {
    cerr << "Fatal ERROR: Unable to access ChrName " << bam_ref_id << " and position = " << (*current_variant)->position << " within the BAM file provoided " << endl;
    exit(-1);
  }

  if (DEBUG)
    cout << "VCF = " << (*current_variant)->sequenceName << ":" << (*current_variant)->position  << endl;
}


// ----------------------------------------------------------

LiveFiles::LiveFiles() {
  start_time = time(NULL);
}

void LiveFiles::ActivateFiles(ExtendParameters &parameters){
   SetBaseName(parameters);
   ActiveOutputDir(parameters);
   ActiveOutputVCF(parameters);
   ActiveFilterVCF(parameters);
   ActiveConsensus(parameters);
   ActiveDiagnostic(parameters);
}

void LiveFiles::ActiveOutputDir(ExtendParameters &parameters){
  // try to create output directory
  // because I hate having to make this manually when I run
  //@TODO: please put in real error checks here
    if (true){
    // make output directory "side effect bad"
    mkdir(parameters.outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}

void LiveFiles::ActiveDiagnostic(ExtendParameters &parameters){
  if (parameters.program_flow.rich_json_diagnostic){
    // make output directory "side effect bad"
    parameters.program_flow.json_plot_dir = parameters.outputDir + "/json_diagnostic/";
    mkdir(parameters.program_flow.json_plot_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}

void LiveFiles::ActiveOutputVCF(ExtendParameters &parameters){
     string full_output_vcf_filename = parameters.outputDir + "/" + parameters.outputFile;
  
  outVCFFile.open(full_output_vcf_filename.c_str());
  if (!outVCFFile.is_open()) {
    fprintf(stderr, "[tvc] FATAL: Cannot open %s: %s\n", full_output_vcf_filename.c_str(), strerror(errno));
    exit(-1);
  }


}

void LiveFiles::SetBaseName(ExtendParameters &parameters){
      char basename[256] = {'\0'};
  getBaseName(parameters.outputFile.c_str(), basename);
  file_base_name = basename;
}

void LiveFiles::ActiveFilterVCF(ExtendParameters &parameters){
  stringstream filterVCFFileNameStream;
  filterVCFFileNameStream <<  parameters.outputDir;
  filterVCFFileNameStream << "/";
  filterVCFFileNameStream << file_base_name;
  filterVCFFileNameStream << "_filtered.vcf";
  string filterVCFFileName = filterVCFFileNameStream.str();
  
  filterVCFFile.open(filterVCFFileName.c_str());
  if (!filterVCFFile.is_open()) {
    cerr << "[tvc] FATAL: Cannot open filter vcf file : " << filterVCFFileName << endl;
    exit(-1);
  }
}

void LiveFiles::ActiveConsensus(ExtendParameters &parameters){
    
  if (parameters.consensusCalls) { //output Consensus calls file used by Tumor-Normal module and others that are used to DiBayes
    stringstream consensusCallsStream;
    consensusCallsStream << parameters.outputDir;
    consensusCallsStream << "/";
    consensusCallsStream << file_base_name;
    consensusCallsStream << "_Consensus_calls.txt";
    string consensusCallsFileName = consensusCallsStream.str();
    consensusFile.open(consensusCallsFileName.c_str());
    if (!consensusFile.is_open()) {
      cerr << "[tvc] ERROR: Cannot open Consensus Calls file : " << consensusCallsFileName << endl;
      exit(-1);
    }
  }
}

void LiveFiles::ShutDown(){
  
  outVCFFile.close();
  cout << "[tvc] Normal termination. Processing time: " << (time(NULL)-start_time) << " seconds." << endl;
};
