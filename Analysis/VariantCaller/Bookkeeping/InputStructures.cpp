/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "InputStructures.h"
#include "ExtendedReadInfo.h"
#include "json/json.h"

InputStructures::InputStructures() {
    flowKey = "TCAG";
    flowOrder = "";
    flowSigPresent = false;
    DEBUG = 0;
    nFlows = 0;
    min_map_qv = 4;
    use_SSE_basecaller  = true;
    do_snp_realignment  = true;
    apply_normalization = false;
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

    do_recal.ReadRecalibrationFromComments(samHeader);

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

// this function is necessary because we have processed reads in sub-bams for areas of the chip
// first we look up the run (bams may be combined)
// then we look up the block for this area
// different size/shape chips mean can't use same lookup table
// inefficient for now: get the plane off the ground, then make the engines work
string RecalibrationHandler::FindKey(string &runid, int x, int y) {
    std::pair <std::multimap<string,pair<int,int> >:: iterator, std::multimap<string,pair<int,int> >:: iterator> blocks;
    blocks = block_hash.equal_range(runid);
    int tx,ty;
    tx = ty=0;
    for (std::multimap<string,pair<int,int> >:: iterator it = blocks.first; it!=blocks.second; ++it) {
        int ax = it->second.first;
        int ay = it->second.second;
        if ((ax<=x) & (ay<=y)) {
            // potential block including this point because it is less than the point coordinates
            // take the coordinates largest & closest
            if (ax >tx)
                tx = ax;
            if (ay>ty)
                ty = ay;
        }
    }
    // why in 2013 am I still using sprintf?
    char tmpstr[1024];
    sprintf(tmpstr,"%s.block_X%d_Y%d", runid.c_str(), tx, ty); // runid.block_X0_Y0 for example
    string retval = tmpstr;

    return(retval);
};


void RecalibrationHandler::ReadRecalibrationFromComments(SamHeader &samHeader) {
    // Read comment lines from Sam header
    // this will grab json files
    if (samHeader.HasComments()) {
        // parse the comment lines
        for (unsigned int i_co=0; i_co<samHeader.Comments.size(); i_co++) {
            // try printing for now
            //cout << samHeader.Comments[i_co] << endl;
            // might be all sorts of comments in the file
            // therefore must find the unlikely magic code in the line before trying to parse
            string magic_code = "6d5b9d29ede5f176a4711d415d769108"; // md5hash "This uniquely identifies json comments for recalibration."
            bool valid_line = false;
            std::size_t found = samHeader.Comments[i_co].find(magic_code);
            if (found !=std::string::npos)
                valid_line = true;

            if (valid_line) {
                // very likely to be a properly formatted json object coming from basecaller
                Json::Value recal_params(Json::objectValue);
                Json::Reader recal_reader;
                bool parse_success = recal_reader.parse(samHeader.Comments[i_co], recal_params);
                if (!parse_success) {
                    cout << "failed to parse comment line" << recal_reader.getFormattedErrorMessages() << endl;
                } else {
                    // you are a recalibration object waiting to happen
                    // let us parse you
                    // basic ID

                    //cout << my_members[0] << endl;
                    string my_block_key = recal_params["MasterKey"].asCString();
                    //cout << my_block_key << "\t" << recal_params[my_block_key]["modelParameters"].size() << endl;
                    recalModel.InitializeFromJSON(recal_params, my_block_key, false);  // don't spam here
                    // add a map to this entry
                    bam_header_recalibration.insert(pair<string,RecalibrationModel>(my_block_key, recalModel));
                    // parse out important information from the block key
                    // must look like <runid>.block_X%d_Y%d
                    int end_runid = my_block_key.find(".");
                    int bloc_loc = my_block_key.find("block_X")+7;
                    int y_loc = my_block_key.find("_Y");
                    // glorified assembly language
                    string runid = my_block_key.substr(0,end_runid);
                    int x_coord = atoi(my_block_key.substr(bloc_loc,y_loc-bloc_loc).c_str());
                    int y_coord = atoi(my_block_key.substr(y_loc+2, my_block_key.size()-y_loc+2).c_str());
                    //cout << runid << "\t" << x_coord << "\t" << y_coord << endl;
                    block_hash.insert(pair<string, pair<int,int > >(runid,pair<int,int>(x_coord,y_coord)));
                    is_live = true; // found at least one recalibration entry
                }
            }
        }
    }

    // okay, now, avoid spamming with possibly large number of lines
    if (is_live){
      cout << "Recalibration was detected from comment lines in bam file(s)" << endl;
      cout << bam_header_recalibration.size() << " unique blocks of recalibration info detected." << endl;
    }
}

void RecalibrationHandler::getAB(MultiAB &multi_ab, string &found_key, int x, int y) {
    if (use_recal_model_only)
        recalModel.getAB(multi_ab,x,y);
    else {
      // found_key in map to get iterator
      map<string, RecalibrationModel>::iterator it;
      it = bam_header_recalibration.find(found_key);
      if (it!=bam_header_recalibration.end()){ 
        it->second.getAB(multi_ab, x, y);
      } else
        multi_ab.Null();
    }
};

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
                // Convert reference sequence to upper case characters
                stringToUpper(line);
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

void InputStructures::BringUpReferenceData(ExtendParameters &parameters) {

    DEBUG = parameters.program_flow.DEBUG;
    min_map_qv = parameters.MQL0;
    use_SSE_basecaller = parameters.program_flow.use_SSE_basecaller;
    do_snp_realignment = parameters.program_flow.do_snp_realignment;

    cout << "Loading reference." << endl;
    read_fasta(parameters.fasta, reference_contigs);
    cout << "Loaded reference. Ref length: " << reference_contigs.size() << endl;

    // some recalibration information may be read from bam file header
    bam_initialize(parameters.bams);

    if (parameters.sseMotifsProvided) {
        cout << "Loading systematic error contexts." << endl;
        read_error_motifs(parameters.sseMotifsFileName);
        cout << "Loaded." << endl;
    }

    // Load homopolymer recalibration model
    // why is recal model using the command line directly? <-- Because the basecaller module is programmed that way.
    // initialize only if there's a model file
    if (parameters.recal_model_file_name.length()>0){
        do_recal.recalModel.Initialize(parameters.opts);
        do_recal.use_recal_model_only = true;
        do_recal.is_live = true;
    }
    
    // finally turn off recalibration if not wanted
    // even although we have a nice set of recalibration read-in.
    if (parameters.program_flow.suppress_recalibration) {
        printf("Recalibration model: suppressed\n");
        do_recal.recalModel.suppressEnabled();
        do_recal.is_live = false;
    }
}

// this is officially bad, as the scope of the reference is uncertain
// but since it is signed in blood that our reference_contigs will persist for the whole program
// we can pretend to be happy
string & InputStructures::ReturnReferenceContigSequence(vcf::Variant ** current_variant) {
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

void LiveFiles::ActivateFiles(ExtendParameters &parameters) {
    SetBaseName(parameters);
    ActiveOutputDir(parameters);
    ActiveOutputVCF(parameters);
    ActiveFilterVCF(parameters);
    ActiveConsensus(parameters);
    ActiveDiagnostic(parameters);
}

void LiveFiles::ActiveOutputDir(ExtendParameters &parameters) {
    // try to create output directory
    // because I hate having to make this manually when I run
    //@TODO: please put in real error checks here
    if (true) {
        // make output directory "side effect bad"
        mkdir(parameters.outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void LiveFiles::ActiveDiagnostic(ExtendParameters &parameters) {
    if (parameters.program_flow.rich_json_diagnostic) {
        // make output directory "side effect bad"
        parameters.program_flow.json_plot_dir = parameters.outputDir + "/json_diagnostic/";
        mkdir(parameters.program_flow.json_plot_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void LiveFiles::ActiveOutputVCF(ExtendParameters &parameters) {
    string full_output_vcf_filename = parameters.outputDir + "/" + parameters.outputFile;

    outVCFFile.open(full_output_vcf_filename.c_str());
    if (!outVCFFile.is_open()) {
        fprintf(stderr, "[tvc] FATAL: Cannot open %s: %s\n", full_output_vcf_filename.c_str(), strerror(errno));
        exit(-1);
    }


}

void LiveFiles::SetBaseName(ExtendParameters &parameters) {
    char basename[256] = {'\0'};
    getBaseName(parameters.outputFile.c_str(), basename);
    file_base_name = basename;
}

void LiveFiles::ActiveFilterVCF(ExtendParameters &parameters) {
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

void LiveFiles::ActiveConsensus(ExtendParameters &parameters) {

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

void LiveFiles::ShutDown() {

    outVCFFile.close();
    cout << "[tvc] Normal termination. Processing time: " << (time(NULL)-start_time) << " seconds." << endl;
};
