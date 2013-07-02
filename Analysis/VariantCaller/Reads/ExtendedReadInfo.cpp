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
    cerr << "ERROR in ExtendedReadInfo::GetUsefulTags: Treephaser Params ZP:tag is not present in the BAM file provided" << endl;
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

void ExtendedReadInfo::CreateFlowIndex(string &flowOrder) {

  string read_bases = alignment.QueryBases;
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
  if (base_idx != read_bases.length())
    cerr << "WARNING in ExtendedReadInfo::CreateFlowIndex: There are more bases in the read than fit into the flow order.";
}

// -------------------------------------------------------

// Sets the member variables ref_aln, seq_aln, pretty_aln, startSC, endSC
void ExtendedReadInfo::UnpackAlignmentInfo(const string &local_contig_sequence, unsigned int aln_start_position) {

  ref_aln.clear();
  seq_aln.clear();
  pretty_aln.clear();

  startSC = 0;
  endSC = 0;

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
        read_pos += cigar->Length;
        ref_pos  += cigar->Length;
        break;

      case ('I') :
        ref_aln.append(cigar->Length,'-');
        seq_aln.append(alignment.QueryBases, read_pos, cigar->Length);
        pretty_aln.append(cigar->Length, '+');
        read_pos += cigar->Length;
        break;

      case ('S') :
        read_pos += cigar->Length;
        if (match_found)
          endSC = cigar->Length;
        else
          startSC = cigar->Length;
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
  // Update start flow to flow of first non-soft-clipped base <- splicing only takes mapped bases
  if (is_forward_strand)
    start_flow = flowIndex.at(startSC);
  else
    start_flow = flowIndex.at(endSC);
}

// -------------------------------------------------------

bool ExtendedReadInfo::UnpackThisRead(InputStructures &global_context, const string &local_contig_sequence, int DEBUG) {

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
