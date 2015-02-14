/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "InputStructures.h"
#include "ExtendedReadInfo.h"
#include "json/json.h"

InputStructures::InputStructures()
{
  flowKey = "TCAG";
  flowOrder = "";
  flowSigPresent = false;
  DEBUG = 0;
  nFlows = 0;
  use_SSE_basecaller  = true;
  apply_normalization = false;
  resolve_clipped_bases = false;
}


void InputStructures::Initialize(ExtendParameters &parameters, const ReferenceReader& ref_reader, const SamHeader &bam_header)
{
  DEBUG                 = parameters.program_flow.DEBUG;

  use_SSE_basecaller    = parameters.program_flow.use_SSE_basecaller;
  resolve_clipped_bases = parameters.program_flow.resolve_clipped_bases;

  // must do this first to detect nFlows
  DetectFlowOrderzAndKeyFromBam(bam_header);
  // now get recalibration information, padded to account if nFlows for some bam is large
  do_recal.ReadRecalibrationFromComments(bam_header,nFlows-1); // protect against over-flowing nFlows, 0-based

  if (parameters.sseMotifsProvided) {
    cout << "Loading systematic error contexts." << endl;
    read_error_motifs(parameters.sseMotifsFileName);
    cout << "Loaded." << endl;
  }

  // Load homopolymer recalibration model from file if the option was specified
  // why is recal model using the command line directly? <-- Because the basecaller module is programmed that way.
  // initialize only if there's a model file
  if (parameters.recal_model_file_name.length()>0){
    do_recal.recalModel.InitializeModel(parameters.recal_model_file_name, parameters.recalModelHPThres);
    do_recal.use_recal_model_only = true;
    do_recal.is_live = do_recal.recalModel.is_enabled();
  }

  // finally turn off recalibration if not wanted
  // even though we have a nice set of recalibration read-in.
  if (parameters.program_flow.suppress_recalibration) {
    printf("Recalibration model: suppressed\n");
    do_recal.recalModel.suppressEnabled();
    do_recal.is_live = false;
  }
}




void InputStructures::DetectFlowOrderzAndKeyFromBam(const SamHeader &samHeader){
  
//TODO Need to handle multiple BAM files with different flowOrders, at least throw an error for now.
    for (BamTools::SamReadGroupConstIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {
        if (itr->HasFlowOrder()) {
            string tmpflowOrder = itr->FlowOrder;
            if (bamFlowOrderVector.empty()){
                bamFlowOrderVector.push_back(tmpflowOrder);
                flowOrder = tmpflowOrder; // first one free
            }else { //check if the flowOrder is the same if not throw an error, for now we dont support bams with different flow orders
                vector<string>::iterator it = std::find(bamFlowOrderVector.begin(), bamFlowOrderVector.end(), tmpflowOrder);
                if (it == bamFlowOrderVector.end()) {
                   // check to see if flowOrder is a substring/superstring first
                  std::size_t found_me = std::string::npos; // assume bad
                  if (tmpflowOrder.length()>flowOrder.length()){
                    found_me = tmpflowOrder.find(flowOrder);
                    if (found_me==0){ // must find at first position
                      flowOrder = tmpflowOrder; // longer superstring
                      bamFlowOrderVector.push_back(tmpflowOrder);
                      //cout<< "Super: " << tmpflowOrder.length() << " " << tmpflowOrder << endl;
                    } else
                      found_me = std::string::npos;

                  }else{
                    found_me = flowOrder.find(tmpflowOrder);
                    if (found_me==0){ // must find at first position
                      // substring, so no need to update flowOrder
                      bamFlowOrderVector.push_back(tmpflowOrder);
                      //cout << "Sub: " << tmpflowOrder.length() << " "<< tmpflowOrder << endl;
                    } else
                      found_me = std::string::npos;
                  }

                  if (found_me==std::string::npos){
                    cerr << "FATAL ERROR: BAM files specified as input have different flow orders. Currently tvc supports only BAM files with same flow order. " << endl;
                    exit(-1);
                  }
                }
            }
            flowKey = itr->KeySequence;

        }

    }
    if (bamFlowOrderVector.size()>1)
      cout << "Compatibly nested flow orders found: " << bamFlowOrderVector.size() << " using longest, nFlows=  " << flowOrder.length() << endl;
    //cout << "Final: " << flowOrder.length() << " " << flowOrder << endl;
    nFlows = flowOrder.length();

    if (nFlows > 0) {
        flowSigPresent = true;
        treePhaserFlowOrder.SetFlowOrder(flowOrder, nFlows);
        key.Set(treePhaserFlowOrder, flowKey, "key");
    }

}


// this function is necessary because we have processed reads in sub-bams for areas of the chip
// first we look up the run (bams may be combined)
// then we look up the block for this area
// different size/shape chips mean can't use same lookup table
// inefficient for now: get the plane off the ground, then make the engines work
string RecalibrationHandler::FindKey(const string &runid, int x, int y) const {
    std::pair <std::multimap<string,pair<int,int> >:: const_iterator, std::multimap<string,pair<int,int> >:: const_iterator> blocks;
    blocks = block_hash.equal_range(runid);
    int tx,ty;
    tx = ty=0;
    for (std::multimap<string,pair<int,int> >:: const_iterator it = blocks.first; it!=blocks.second; ++it) {
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


void RecalibrationHandler::ReadRecalibrationFromComments(const SamHeader &samHeader, int max_flows_protect) {
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
                    recalModel.InitializeFromJSON(recal_params, my_block_key, false,max_flows_protect);  // don't spam here
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

void RecalibrationHandler::getAB(MultiAB &multi_ab, const string &found_key, int x, int y) const {
    if (use_recal_model_only)
        recalModel.getAB(multi_ab,x,y);
    else {
      // found_key in map to get iterator
      map<string, RecalibrationModel>::const_iterator it;
      it = bam_header_recalibration.find(found_key);
      if (it!=bam_header_recalibration.end()){ 
        it->second.getAB(multi_ab, x, y);
      } else
        multi_ab.Null();
    }
};



