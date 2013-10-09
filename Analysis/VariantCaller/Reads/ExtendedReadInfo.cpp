/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ExtendedReadInfo.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "ExtendedReadInfo.h"
#include "ion_util.h"

void ExtendedReadInfo::GetUsefulTags(int DEBUG) {

  vector<int16_t> quantizedMeasures;
  if (!alignment.GetTag("ZM", quantizedMeasures)) {
    cerr << "ERROR in ExtendedReadInfo::GetUsefulTags: Normalized measurements ZM:tag is not present in the BAM file provided" << endl;
    is_happy_read = false;
    exit(-1);
  }
  if (!alignment.GetTag("ZP", phase_params)) {
    cerr << "ERROR in ExtendedReadInfo::GetUsefulTags: Phasing Parameters ZP:tag is not present in the BAM file provided" << endl;
    is_happy_read = false;
    exit(-1);
  }
  if (!alignment.GetTag("ZF", start_flow) || start_flow == 0) {
    cerr << "ERROR  in ExtendedReadInfo::GetUsefulTags: Start Flow ZF:tag is not present in the BAM file provided or Invalid Value returned : " << start_flow << endl;
    is_happy_read = false;
    exit(-1);
  }
  if (!alignment.Name.empty()) {
    well_rowcol.resize(2);
    ion_readname_to_rowcol(&alignment.Name[0], &well_rowcol[0], &well_rowcol[1]);
    // extract runid while we are at it
       
    int end_runid = alignment.Name.find(":");
    runid =alignment.Name.substr(0,end_runid);
  }

  map_quality = alignment.MapQuality;

  if (is_happy_read) {
    for (size_t counter = 0; counter < quantizedMeasures.size(); counter++) {
      measurementValue.at(counter) = ((float)quantizedMeasures.at(counter)/256);
    }
  }
  // ad-hoc corrector
  phase_params[2] = 0.0f; // zero droop
}

// -------------------------------------------------------

bool ExtendedReadInfo::CreateFlowIndex(const string &flowOrder) {

  bool is_happy = true;
  read_bases = alignment.QueryBases;
  if (!is_forward_strand)
    RevComplementInPlace(read_bases);

  flowIndex.assign(read_bases.length(), flowOrder.length());
  unsigned int flow = start_flow;
  unsigned int base_idx = 0;
  while (base_idx < read_bases.length() and flow < flowOrder.length()){
    while (flow < flowOrder.length() and flowOrder[flow] != read_bases[base_idx])
      flow++;
    flowIndex[base_idx] = flow;
    base_idx++;
  }
  if (base_idx != read_bases.length()) {
    cerr << "WARNING in ExtendedReadInfo::CreateFlowIndex: There are more bases in the read than fit into the flow order.";
    is_happy = false;
  }
  return (is_happy);
}

// -------------------------------------------------------

// Sets the member variables ref_aln, seq_aln, pretty_aln, startSC, endSC
bool ExtendedReadInfo::UnpackAlignmentInfo(const string &local_contig_sequence, unsigned int aln_start_position) {

  ref_aln.clear();
  seq_aln.clear();
  pretty_aln.clear();

  leftSC = 0;
  rightSC = 0;

  bool is_happy = true;
  unsigned int num_query_bases = 0;
  unsigned int read_pos = 0;
  unsigned int ref_pos = aln_start_position;
  bool match_found = false;

  for (vector<CigarOp>::const_iterator cigar = alignment.CigarData.begin(); cigar != alignment.CigarData.end(); ++cigar) {
    switch (cigar->Type) {
      case ('M') :
      case ('=') :
      case ('X') :
        match_found = true;
        ref_aln.append(local_contig_sequence, ref_pos, cigar->Length);
        seq_aln.append(alignment.QueryBases, read_pos, cigar->Length);
        pretty_aln.append(cigar->Length, '|');
        num_query_bases += cigar->Length;
        read_pos += cigar->Length;
        ref_pos  += cigar->Length;
        break;

      case ('I') :
        ref_aln.append(cigar->Length,'-');
        seq_aln.append(alignment.QueryBases, read_pos, cigar->Length);
        pretty_aln.append(cigar->Length, '+');
        num_query_bases += cigar->Length;
        read_pos += cigar->Length;
        break;

      case ('S') :
		num_query_bases += cigar->Length;
        read_pos += cigar->Length;
        if (match_found)
          rightSC = cigar->Length;
        else
          leftSC = cigar->Length;
        break;

      case ('D') :
      case ('P') :
      case ('N') :
        ref_aln.append(local_contig_sequence, ref_pos, cigar->Length);
        seq_aln.append(cigar->Length,'-');
        pretty_aln.append(cigar->Length, '-');
        ref_pos += cigar->Length;
        break;
    }
  }
  // Basic alignment sanity check
  if (num_query_bases != alignment.QueryBases.length()) {
    cerr << "WARNING in ExtendedReadInfo::UnpackAlignmentInfo: Invalid Cigar String in Read " << alignment.Name << " Cigar: ";
    for (vector<CigarOp>::const_iterator cigar = alignment.CigarData.begin(); cigar != alignment.CigarData.end(); ++cigar)
      cerr << cigar->Length << cigar->Type;
    cerr << " Length of query string: " << alignment.QueryBases.length() << endl;
    //is_happy = false;
  }

  return (is_happy);
}

// -------------------------------------------------------
// Increase start flow to main flow of first aligned base

void ExtendedReadInfo::IncreaseStartFlow() {
  if (is_forward_strand)
    start_flow = flowIndex.at(leftSC);
  else
    start_flow = flowIndex.at(rightSC);
}


// -------------------------------------------------------
unsigned int ExtendedReadInfo::GetStartSC() {
  if (is_forward_strand)
	return (leftSC);
  else
	return (rightSC);
}

unsigned int ExtendedReadInfo::GetEndSC() {
  if (is_forward_strand)
	return (rightSC);
  else
	return (leftSC);
}

// -------------------------------------------------------

bool ExtendedReadInfo::UnpackThisRead(const InputStructures &global_context, const string &local_contig_sequence, int DEBUG) {

  //is_happy_read = CheckHappyRead(global_context, variant_start_pos, int DEBUG);
  is_happy_read = true;

  // start working to unpack the read data we need
  start_flow = 0;
  start_pos = alignment.Position;
  is_forward_strand = !alignment.IsReverseStrand();

  GetUsefulTags(DEBUG);
  if (is_happy_read) {
    measurementValue.resize(global_context.flowOrder.length(), 0.0);
    CreateFlowIndex(global_context.flowOrder);
    UnpackAlignmentInfo(local_contig_sequence, alignment.Position);
  }
  return(is_happy_read); // happy to have unpacked this read
}
