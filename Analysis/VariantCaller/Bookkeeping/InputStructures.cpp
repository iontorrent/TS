/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "InputStructures.h"
#include "ExtendedReadInfo.h"
#include "json/json.h"
#include "MolecularTag.h"

InputStructures::InputStructures()
{
  DEBUG = 0;
#ifdef __SSE3__
  use_SSE_basecaller  = true;
#else
  use_SSE_basecaller  = false;
#endif
  resolve_clipped_bases = false;
}

// ------------------------------------------------------------------------------------

void InputStructures::Initialize(ExtendParameters &parameters, const ReferenceReader& ref_reader, const SamHeader &bam_header)
{
  DEBUG                 = parameters.program_flow.DEBUG;

  use_SSE_basecaller    = parameters.program_flow.use_SSE_basecaller;
  resolve_clipped_bases = parameters.program_flow.resolve_clipped_bases;

  // must do this first to detect nFlows
  DetectFlowOrderzAndKeyFromBam(bam_header);

  // now get calibration information, padded to account if nFlows for some bam is large
  do_recal.ReadRecalibrationFromComments(bam_header,num_flows_by_run_id); // protect against over-flowing nFlows, 0-based

  if ((parameters.sseMotifsProvided) && (parameters.my_controls.filter_variant.sseProbThreshold < 1.0)) {
    cout << "Loading systematic error contexts." << endl;
    read_error_motifs(parameters.sseMotifsFileName);
    cout << "Loaded." << endl;
  }

  // Load homopolymer recalibration model from file if the command line option was specified
  // why is recal model using the command line directly? <-- Because the basecaller module is programmed that way.
  // initialize only if there's a model file
  if (parameters.recal_model_file_name.length()>0){

    // We only allow the use of a command line txt file calibration model if there is a single run id
    if (num_flows_by_run_id.size() == 1) {

      do_recal.recalModel.InitializeModelFromTxtFile(parameters.recal_model_file_name, parameters.recalModelHPThres, num_flows_by_run_id.begin()->second);
      do_recal.use_recal_model_only = true;
      do_recal.is_live = do_recal.recalModel.is_enabled();
    }
    else{
      cerr << "TVC WARNING: Cannot initialize calibration model from text file for multiple run ids. " << endl;
      do_recal.is_live = false;
      do_recal.recalModel.disable();
    }
  }

  // finally turn off recalibration if not wanted
  // even though we have a nice set of recalibration read-in.
  if (parameters.program_flow.suppress_recalibration) {
    printf("Recalibration model: suppressed\n");
    do_recal.recalModel.disable();
    do_recal.is_live = false;
  }
}


// ------------------------------------------------------------------------------------

void InputStructures::DetectFlowOrderzAndKeyFromBam(const SamHeader &samHeader){

    // We only store flow orders that are different from each other in the flow order vector.
    // The same flow order but different total numbers of flow map to the  same flow order object
    // So multiple runs, even with different max flows, point to the same flow order object
    // We assume that the read group name is written in the form <run_id>.<Barcode Name>

    flow_order_vector.clear();
    vector<string> temp_flow_order_vector;
    int num_read_groups = 0;

    for (BamTools::SamReadGroupConstIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {

      num_read_groups++;
      if (itr->ID.empty()){
        cerr << "TVC ERROR: BAM file has a read group without ID." << endl;
        exit(EXIT_FAILURE);
      }
      // We need a flow order to do variant calling so throw an error if there is none.
      if (not itr->HasFlowOrder()) {
        cerr << "TVC ERROR: read group " << itr->ID << " does not have a flow order." << endl;
        exit(EXIT_FAILURE);
      }

      // Check for duplicate read group ID entries and throw an error if one is found
      std::map<string,string>::const_iterator key_it = key_by_read_group.find(itr->ID);
      if (key_it != key_by_read_group.end()) {
        cerr << "TVC ERROR: Multiple read group entries with ID " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Store Key Sequence for each read group
      // The read group key in the BAM file contains the full prefix: key sequence + barcode + barcode adapter
      key_by_read_group[itr->ID] = itr->KeySequence;

      // Get run id from read group name: convention <read group name> = <run_id>.<Barcode Name>
      string run_id = itr->ID.substr(0,itr->ID.find('.'));
      if (run_id.empty()) {
        cerr << "TVC ERROR: Unable to extract run id from read group name " << itr->ID << endl;
        exit(EXIT_FAILURE);
      }

      // Check whether an entry already exists for this run id and whether it is compatible
      std::map<string,int>::const_iterator fo_it = flow_order_index_by_run_id.find(run_id);
      if (fo_it != flow_order_index_by_run_id.end()) {
    	// Flow order for this run id may be equal or a subset of the stored one
        if (temp_flow_order_vector.at(fo_it->second).length() < itr->FlowOrder.length()
            or temp_flow_order_vector.at(fo_it->second).substr(0, itr->FlowOrder.length()) != itr->FlowOrder
            or num_flows_by_run_id.at(run_id) != (int)(itr->FlowOrder).length())
        {
          cerr << "TVC ERROR: Flow order information extracted from read group name " << itr->ID
               << " does not match previous entry for run id " << run_id << ": " << endl;
          cerr << "Exiting entry  : " << temp_flow_order_vector.at(fo_it->second) << endl;
          cerr << "Newly extracted: " << itr->FlowOrder << endl;
          cerr << temp_flow_order_vector.at(fo_it->second) << endl;
          exit(EXIT_FAILURE);
        }
        // Found matching entry and everything is OK.
        continue;
      }

      // New run id: Check whether this flow order is the same or a sub/ superset of an existing flow order
      unsigned int iFO = 0;
      for (; iFO< temp_flow_order_vector.size(); iFO++){

        // Is the new flow order a subset of an existing flow order?
        if ( temp_flow_order_vector.at(iFO).length() >= itr->FlowOrder.length() ) {
          if (temp_flow_order_vector.at(iFO).substr(0, itr->FlowOrder.length()) == itr->FlowOrder ) {
            flow_order_index_by_run_id[run_id] = iFO;
            num_flows_by_run_id[run_id] = itr->FlowOrder.length();
            break;
          }
          else
            continue;
        }

        // Is the new flow order a superset of an existing flow order?
        if ( temp_flow_order_vector.at(iFO).length() < itr->FlowOrder.length() ) {
          if ( itr->FlowOrder.substr(0, temp_flow_order_vector.at(iFO).length()) == temp_flow_order_vector.at(iFO) ) {
            temp_flow_order_vector.at(iFO) = itr->FlowOrder;
            flow_order_index_by_run_id[run_id] = iFO;
            num_flows_by_run_id[run_id] = itr->FlowOrder.length();
            break;
          }
        }
      }

      // Do we have a new flow order?
      if (iFO == temp_flow_order_vector.size()) {
    	temp_flow_order_vector.push_back(itr->FlowOrder);
    	flow_order_index_by_run_id[run_id] = iFO;
    	num_flows_by_run_id[run_id] = itr->FlowOrder.length();
      }

    } // --- end loop over read groups

    // Now we have amassed all the unique flow orders and can construct the FlowOrder objects
    for (unsigned int iFO=0; iFO < temp_flow_order_vector.size(); iFO++){
      ion::FlowOrder tempIonFlowOrder(temp_flow_order_vector.at(iFO), temp_flow_order_vector.at(iFO).length());
      flow_order_vector.push_back(tempIonFlowOrder);
    }

    // Verbose output
    cout << "Found a total of " << flow_order_vector.size() << " different flow orders of max flow lengths: ";
    int iFO=0;
    for (; iFO<(int)flow_order_vector.size()-1; iFO++)
      cout << flow_order_vector.at(iFO).num_flows() << ',';
    cout << flow_order_vector.at(iFO).num_flows() << endl;

}

// ===================================================================================

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

// ------------------------------------------------------------------------------------

void RecalibrationHandler::ReadRecalibrationFromComments(const SamHeader &samHeader, const map<string, int> &max_flows_by_run_id) {


  if (not samHeader.HasComments())
    return;

  unsigned int num_parsing_errors = 0;
  // Read comment lines from Sam header
  for (unsigned int i_co=0; i_co<samHeader.Comments.size(); i_co++) {

    // There might be all sorts of comments in the file
    // therefore must find the unlikely magic code in the line before trying to parse
    string magic_code = "6d5b9d29ede5f176a4711d415d769108"; // md5hash "This uniquely identifies json comments for recalibration."

    if (samHeader.Comments[i_co].find(magic_code) == std::string::npos) {
      //cout << endl << "No magic code found in comment line "<< i_co <<endl;
      //cout << samHeader.Comments.at(i_co) << endl;
      continue;
    }

    // Parse recalibration Json object
    Json::Value recal_params(Json::objectValue);
    Json::Reader recal_reader;
    if (not recal_reader.parse(samHeader.Comments[i_co], recal_params)) {
      cerr << "Failed to parse recalibration comment line " << recal_reader.getFormattedErrorMessages() << endl;
      num_parsing_errors++;
      continue;
    }

    string my_block_key = recal_params["MasterKey"].asString();

    // Assumes that the MasterKey is written in the format <run_id>.block_X<x_offset>_Y<y_offset>
    int end_runid = my_block_key.find(".");
    int x_loc     = my_block_key.find("block_X")+7;
    int y_loc     = my_block_key.find("_Y");

    // glorified assembly language
    string runid = my_block_key.substr(0,end_runid);
    int x_coord = atoi(my_block_key.substr(x_loc,y_loc-x_loc).c_str());
    int y_coord = atoi(my_block_key.substr(y_loc+2, my_block_key.size()-y_loc+2).c_str());

    // Protection against not having a flow order for a specified recalibration run id
    std::map<string, int>::const_iterator n_flows = max_flows_by_run_id.find(runid);
    if (n_flows == max_flows_by_run_id.end()) {
      cerr << "TVC ERROR: Recalibration information found for run id " << runid
    	   << " but there is no matching read group with this run id in the bam header." << endl;
      exit(EXIT_FAILURE);
    }

    //recalModel.InitializeFromJSON(recal_params, my_block_key, false, max_flows_by_run_id.at(runid));
    // void RecalibrationModel::InitializeFromJSON(Json::Value &recal_params, string &my_block_key, bool spam_enabled, int over_flow_protect) {
    // The calibration comment line contains  info about the hp threshold used during base calling, so set to zero here
    // XXX FIXME: The number of flows in the TVC group can be larger than the one specified in the calibration block.
    recalModel.InitializeModelFromJson(recal_params, n_flows->second);
    bam_header_recalibration.insert(pair<string,LinearCalibrationModel>(my_block_key, recalModel));
    block_hash.insert(pair<string, pair<int,int > >(runid,pair<int,int>(x_coord,y_coord)));
    is_live = true; // found at least one recalibration entry
  }

  // Verbose output
  if (is_live){
    cout << "Recalibration was detected from comment lines in bam file(s):" << endl;
    cout << bam_header_recalibration.size() << " unique blocks of recalibration info detected." << endl;
  }
  if (num_parsing_errors > 0) {
    cout << "Failed to parse " << num_parsing_errors << " recalibration comment lines." << endl;
  }
}

// ------------------------------------------------------------------------------------

void RecalibrationHandler::getAB(MultiAB &multi_ab, const string &found_key, int x, int y) const {
    if (use_recal_model_only)
        recalModel.getAB(multi_ab,x,y);
    else {
      // found_key in map to get iterator
      map<string, LinearCalibrationModel>::const_iterator it;
      it = bam_header_recalibration.find(found_key);
      if (it!=bam_header_recalibration.end()){ 
        it->second.getAB(multi_ab, x, y);
      } else {
        cerr << "Warning in RecalibrationHandler::getAB - block key " << found_key << " not found!" << endl;
        multi_ab.Null();
      }
    }
};

// =====================================================================================
// We create one basecaller object per unique flow order
// This prevents us from having to rebuild and initialize the whole object for every read

PersistingThreadObjects::PersistingThreadObjects(const InputStructures &global_context)
    : use_SSE_basecaller(global_context.use_SSE_basecaller), realigner(50, 1)
{
#ifdef __SSE3__
    if (use_SSE_basecaller) {
	  for (unsigned int iFO=0; iFO < global_context.flow_order_vector.size(); iFO++){
        TreephaserSSE treephaser_sse(global_context.flow_order_vector.at(iFO), DPTreephaser::kWindowSizeDefault_);
        treephaserSSE_vector.push_back(treephaser_sse);
      }
    }
    else {
#endif
      for (unsigned int iFO=0; iFO < global_context.flow_order_vector.size(); iFO++){
        DPTreephaser      dpTreephaser(global_context.flow_order_vector.at(iFO));
        dpTreephaser_vector.push_back(dpTreephaser);
      }
#ifdef __SSE3__
    }
#endif
};

// ------------------------------------------------------------------------------------



