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

/*
//! @brief  Sets member variables containing alignment information
//! @brief [in]  local_contig_sequence    reference sequence
//! @brief [in]  aln_start_position       start position of the alignment
static void UnpackAlignmentInfo(Alignment *rai);

//! @brief  Populates object members flowIndex and read_seq
static void CreateFlowIndex(Alignment *rai, const string &flow_order);
*/

void CreateFlowIndex(Alignment *rai, const string &flowOrder)
{
  rai->flow_index.assign(rai->read_bases.length(), flowOrder.length());
  unsigned int flow = rai->start_flow;
  unsigned int base_idx = 0;
  while (base_idx < rai->read_bases.length() and flow < flowOrder.length()){
    while (flow < flowOrder.length() and flowOrder[flow] != rai->read_bases[base_idx])
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


void UnpackOnLoad(Alignment *rai, const InputStructures &global_context, const ExtendParameters& parameters)
{
  if (not rai->alignment.IsMapped()) {
    rai->evaluator_filtered = true;
    return;
  }

  // Mapping quality filter
  if (rai->alignment.MapQuality < parameters.min_mapping_qv) {
    rai->evaluator_filtered = true;
    return;
  }


  // Skip reads from samples other than the primary sample
  if (not rai->primary_sample) {
    rai->evaluator_filtered = true;
    return;
  }

  // Absolute number of mismatches filter
  if (rai->snp_count > parameters.read_snp_limit) {
    rai->evaluator_filtered = true;
    rai->worth_saving = false;
    return;
  }

  rai->is_reverse_strand = rai->alignment.IsReverseStrand();

  // Retrieve measurements from ZM tag

  vector<int16_t> quantized_measurements;
  if (not rai->alignment.GetTag("ZM", quantized_measurements)) {
    cerr << "ERROR: Normalized measurements ZM:tag is not present in read " << rai->alignment.Name << endl;
    exit(1);
  }
  if (quantized_measurements.size() > global_context.flowOrder.length()) {
    cerr << "ERROR: Normalized measurements ZM:tag length exceeds flow order length in read " << rai->alignment.Name << endl;
    exit(1);
  }
  rai->measurements.assign(global_context.flowOrder.length(), 0.0);
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

  // Parse read name

  if (not rai->alignment.Name.empty()) {
    rai->well_rowcol.resize(2);
    ion_readname_to_rowcol(rai->alignment.Name.c_str(), &rai->well_rowcol[0], &rai->well_rowcol[1]);
    // extract runid while we are at it
    int end_runid = rai->alignment.Name.find(":");
    rai->runid  = rai->alignment.Name.substr(0,end_runid);
  }

  // Populate read_bases (bases without rev-comp on reverse-mapped reads) and flow_index

  rai->read_bases = rai->alignment.QueryBases;
  if (rai->is_reverse_strand)
    RevComplementInPlace(rai->read_bases);

  // Unpack alignment

  rai->pretty_aln.reserve(global_context.flowOrder.length());
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
    rai->evaluator_filtered = true;
    rai->worth_saving = false;
    return;
  }
  CreateFlowIndex(rai, global_context.flowOrder);

  if (global_context.resolve_clipped_bases) {
    // Increment start flow to first aligned base
    rai->start_flow = rai->flow_index[rai->start_sc];
  }

  // Check validity of input arguments
  if (rai->start_flow < 0 or rai->start_flow >= global_context.treePhaserFlowOrder.num_flows()) {
    cerr << "ERROR: Start flow outsize of [0,num_flows) range in read " << rai->alignment.Name << endl;
    cerr << "Start flow: " << rai->start_flow << " Number of flows: " << global_context.treePhaserFlowOrder.num_flows();
    exit(1);
  }

}





