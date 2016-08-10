/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ExtendedReadInfo.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ExtendedReadInfo.h"
#include "ion_util.h"
#include "ReferenceReader.h"
#include "BAMWalkerEngine.h"
#include "RandSchrange.h"
#include "MiscUtil.h"


// -------------------------------------------------------
//! @brief  Populates object members flowIndex

void CreateFlowIndex(Alignment *rai, const ion::FlowOrder & flow_order)
{
  rai->flow_index.assign(rai->read_bases.length(), flow_order.num_flows());
  int flow = rai->start_flow;
  unsigned int base_idx = 0;
  while (base_idx < rai->read_bases.length() and flow < flow_order.num_flows()){
    while (flow < flow_order.num_flows() and flow_order.nuc_at(flow) != rai->read_bases[base_idx])
      flow++;
    rai->flow_index[base_idx] = flow;
    base_idx++;
  }
  if (base_idx != rai->read_bases.length()) {
    cerr << "WARNING in ExtendedReadInfo::CreateFlowIndex: There are more bases in the read than fit into the flow order.";
    exit(1);
  }
}

// -------------------------------------------------------
//! @brief  Populates object members prefix_flow

void GetPrefixFlow(Alignment *rai, const string & prefix_bases, const ion::FlowOrder & flow_order)
{
  rai->prefix_flow = 0;
  unsigned int base_idx = 0;
  while (base_idx < prefix_bases.length() and rai->prefix_flow < flow_order.num_flows()) {
	while (rai->prefix_flow < flow_order.num_flows() and  flow_order.nuc_at(rai->prefix_flow) != prefix_bases.at(base_idx))
      rai->prefix_flow++;
	base_idx++;
  }
}


// -------------------------------------------------------

// Sets the member variables ref_aln, seq_aln, pretty_aln, startSC, endSC
void UnpackAlignmentInfo(Alignment *rai)
{
  rai->left_sc = 0;
  rai->right_sc = 0;

  unsigned int num_query_bases = 0;
  bool match_found = false;

  for (vector<CigarOp>::const_iterator cigar = rai->alignment.CigarData.begin(); cigar != rai->alignment.CigarData.end(); ++cigar) {
    switch (cigar->Type) {
      case 'M':
      case '=':
      case 'X':
        match_found = true;
        rai->pretty_aln.append(cigar->Length, '|');
        num_query_bases += cigar->Length;
        break;

      case 'I':
        rai->pretty_aln.append(cigar->Length, '+');
        num_query_bases += cigar->Length;
        break;

      case 'S':
		    num_query_bases += cigar->Length;
        if (match_found)
          rai->right_sc = cigar->Length;
        else
          rai->left_sc = cigar->Length;
        break;

      case 'D':
      case 'P':
      case 'N':
        rai->pretty_aln.append(cigar->Length, '-');
        break;
    }
  }
  // after possible trimming
  // rai->align_start = rai->alignment.Position;
  // rai->align_end = rai->alignment.GetEndPosition(false, true);

  // Basic alignment sanity check
  if (num_query_bases != rai->alignment.QueryBases.length()) {
    cerr << "WARNING in ExtendedReadInfo::UnpackAlignmentInfo: Invalid Cigar String in Read " << rai->alignment.Name << " Cigar: ";
    for (vector<CigarOp>::const_iterator cigar = rai->alignment.CigarData.begin(); cigar != rai->alignment.CigarData.end(); ++cigar)
      cerr << cigar->Length << cigar->Type;
    cerr << " Length of query string: " << rai->alignment.QueryBases.length() << endl;
    assert(num_query_bases == rai->alignment.QueryBases.length());
  }

}



// -------------------------------------------------------
// Unpacking read meta data and filtering read if this is not possible

void UnpackOnLoad(Alignment *rai, const InputStructures &global_context)
{
  // No need to waste time if the read is filtered
  if (rai->filtered)
    return;

  rai->is_reverse_strand = rai->alignment.IsReverseStrand();

  // Parse read name, run id & flow order index

  rai->runid.clear();
  if (not rai->alignment.Name.empty()) {
    rai->well_rowcol.resize(2);
    ion_readname_to_rowcol(rai->alignment.Name.c_str(), &rai->well_rowcol[0], &rai->well_rowcol[1]);
    // extract runid while we are at it
    rai->runid  = rai->alignment.Name.substr(0,rai->alignment.Name.find(":"));
  }
  
  if (rai->runid.empty()){
    cerr << "WARNING: Unable to determine run id of read " << rai->alignment.Name << endl;
    rai->filtered = true;
    return;
  }

  std::map<string,int>::const_iterator fo_it = global_context.flow_order_index_by_run_id.find(rai->runid);
  if (fo_it == global_context.flow_order_index_by_run_id.end()){
    cerr << "WARNING: No matching flow oder found for read " << rai->alignment.Name << endl;
    rai->filtered = true;
    return;
  }
  rai->flow_order_index = fo_it->second;
  const ion::FlowOrder & flow_order = global_context.flow_order_vector.at(rai->flow_order_index);

  // Retrieve measurements from ZM tag

  vector<int16_t> quantized_measurements;
  if (not rai->alignment.GetTag("ZM", quantized_measurements)) {
    cerr << "ERROR: Normalized measurements ZM:tag is not present in read " << rai->alignment.Name << endl;
    exit(1);
  }
  if ((int)quantized_measurements.size() > global_context.num_flows_by_run_id.at(rai->runid)) {
    cerr << "ERROR: Normalized measurements ZM:tag length " << quantized_measurements.size()
         << " exceeds flow order length " << global_context.num_flows_by_run_id.at(rai->runid)
         <<" in read " << rai->alignment.Name << endl;
    exit(1);
  }
  rai->measurements.assign(global_context.num_flows_by_run_id.at(rai->runid), 0.0);
  for (size_t counter = 0; counter < quantized_measurements.size(); ++counter)
    rai->measurements[counter] = (float)quantized_measurements[counter]/256;
  rai->measurements_length = quantized_measurements.size();

  // Retrieve phasing parameters from ZP tag

  if (not rai->alignment.GetTag("ZP", rai->phase_params)) {
    cerr << "ERROR: Phasing Parameters ZP:tag is not present in read " << rai->alignment.Name << endl;
    exit(1);
  }
  if (rai->phase_params.size() != 3) {
    cerr << "ERROR: Phasing Parameters ZP:tag does not have 3 phase parameters in read " << rai->alignment.Name << endl;
    exit(1);
  }
  if (rai->phase_params[0] < 0 or rai->phase_params[0] > 1 or rai->phase_params[1] < 0 or rai->phase_params[1] > 1
      or rai->phase_params[2] < 0 or rai->phase_params[2] > 1) {
    cerr << "ERROR: Phasing Parameters ZP:tag outside of [0,1] range in read " << rai->alignment.Name << endl;
    exit(1);
  }
  rai->phase_params[2] = 0.0f;   // ad-hoc corrector: zero droop

  // Populate read_bases (bases without rev-comp on reverse-mapped reads) and flow_index

  rai->read_bases = rai->alignment.QueryBases;
  if (rai->is_reverse_strand)
    RevComplementInPlace(rai->read_bases);
  if (rai->read_bases.empty()){
    cerr << "WARNING: Ignoring length zero read " << rai->alignment.Name << endl;
    rai->filtered = true;
    return;
  }

  // Unpack alignment

  rai->pretty_aln.reserve(global_context.num_flows_by_run_id.at(rai->runid));
  UnpackAlignmentInfo(rai);
  if (rai->is_reverse_strand)
    rai->start_sc = rai->right_sc;
  else
    rai->start_sc = rai->left_sc;

  // Generate flow index

  rai->start_flow = 0;
  if (not rai->alignment.GetTag("ZF", rai->start_flow)) {
    uint8_t start_flow_byte = 0;
    if (not rai->alignment.GetTag("ZF", start_flow_byte)) {
      cerr << "ERROR: Start Flow ZF:tag not found in read " << rai->alignment.Name << endl;
      exit(1);
    }
    rai->start_flow = (int)start_flow_byte;
  }
  if (rai->start_flow == 0) {
    cerr << "WARNING: Start Flow ZF:tag has zero value in read " << rai->alignment.Name << endl;
    rai->filtered = true;
    return;
  }
  CreateFlowIndex(rai, flow_order);

  if (global_context.resolve_clipped_bases) {
    // Increment start flow to first aligned base
    rai->start_flow = rai->flow_index[rai->start_sc];
  }

  // Check validity of input arguments
  if (rai->start_flow < 0 or rai->start_flow >= global_context.num_flows_by_run_id.at(rai->runid)) {
    cerr << "ERROR: Start flow outside of [0,num_flows) range in read " << rai->alignment.Name << endl;
    cerr << "Start flow: " << rai->start_flow << " Number of flows: " << global_context.flow_order_vector.at(rai->flow_order_index).num_flows();
    exit(1);
  }

  // Retrieve read group name & generate prefix flow

  if (not rai->alignment.GetTag("RG",rai->read_group)) {
    cerr << "WARNING: No read group found in read " << rai->alignment.Name << endl;
    // No big problem, we'll just have to solve the prefix like it's 2013!
    rai->read_group.clear();
  }

  // Get read prefix - hard clipped start of the read: [KS][ZT][ZE]
  rai->prefix_flow = -1;
  std::map<string,string>::const_iterator key_it = global_context.key_by_read_group.find(rai->read_group);
  if (key_it != global_context.key_by_read_group.end()) {
    rai->prefix_bases = key_it->second;

    string temp_zt, temp_ze;
    if (rai->alignment.GetTag("ZT", temp_zt))
      rai->prefix_bases += temp_zt;
    if (rai->alignment.GetTag("ZE", temp_ze))
      rai->prefix_bases += temp_ze;

    if (not rai->prefix_bases.empty())
	  GetPrefixFlow(rai, rai->prefix_bases, flow_order);
  }

  // Check consistency of prefix_flow and start_flow - maybe we don't have all info about hard clipped bases
  if (rai->prefix_flow >= 0) {
    int check_start_flow = rai->prefix_flow;
    while (check_start_flow < flow_order.num_flows() and  flow_order.nuc_at(check_start_flow) != rai->read_bases.at(0))
	  check_start_flow++;
    if (check_start_flow != rai->start_flow) {
      rai->prefix_flow = -1;
      rai->prefix_bases.clear();
    }
  }

}





