/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */



#include "ExtendedReadData.h"

using namespace std;


bool ExtendedReadInfo::CheckHappyRead(BamHeaderHelper &my_helper, string &variant_contig, int variant_start_pos, int DEBUG)
{
     if (DEBUG)
    {
      string seq_name;
      seq_name = alignment.Name;
      cout << "Seq Name = " << seq_name << endl;
    }   
    
    // check if we're done by some condition
      string chr_i = "";
    chr_i = my_helper.bam_sequence_names[alignment.RefID];
    if (chr_i.length() == 0 || chr_i.compare("*") == 0 || chr_i.compare(variant_contig)) {
      return(false); // stop once you have reached unmapped beads or a new contig
    }
    
     if (alignment.Position >= variant_start_pos)
      return(false);

     return(true);
}

void ExtendedReadInfo::UnpackAlignment()
{
  // read_stack contains our set of alignments (shouldn't be unnaturally large)
				string qSeq = alignment.QueryBases;
				qDNA = alignment.AlignedBases;
				string md;
				alignment.GetTag("MD", md);

        string tmpDNA;

				dna( qSeq, alignment.CigarData, md, tmpDNA );
				string  pad_query, pad_match, pad_target;
				padded_alignment( alignment.CigarData, qDNA, tmpDNA, pad_query, pad_target, pad_match, alignment.IsReverseStrand());
				//std::vector<int> qlen = score_alignments(pad_query, pad_target, pad_match );
				
				if( alignment.IsReverseStrand() ){
					reverse_comp(pad_target);
					reverse_comp(pad_query);
					std::reverse( pad_match.begin(), pad_match.end() );
				}
        tDNA = pad_target;
        qDNA = pad_query;
}

void ExtendedReadInfo::GetTags()
{
			std::vector<int16_t> quantized_measured; // round(256*val), signed
      if (!alignment.GetTag("ZM", quantized_measured)) {

      cerr << "ERROR: Normalized measurements ZM:tag is not present in the BAM file provided" << endl;
      //exit(-1);
    }
    measurementValue.resize(quantized_measured.size());
    for (size_t counter = 0; counter < quantized_measured.size(); counter++)
    	measurementValue[counter] = ((float)quantized_measured.at(counter)/256);

    if (!alignment.GetTag("ZP", phase_params)) {
      cerr << "ERROR: Phase Params ZP:tag is not present in the BAM file provided" << endl;
      //exit(-1);
    }

    if (!alignment.GetTag("ZF", start_flow) || start_flow == 0) {
      cerr << "ERROR: Start Flow ZF:tag is not present in the BAM file provided or Invalid Value returned : " << start_flow << endl;
      //exit(-1);
    }
// column & row from read name
			if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &row, &col))
				std::cerr << "Error parsing read name: " << alignment.Name << "\n";
}

bool ExtendedReadInfo::UnpackThisRead(BamHeaderHelper &my_helper, string &variant_contig, int variant_start_pos, int DEBUG)
{
  // debugging
     if (!CheckHappyRead(my_helper, variant_contig, variant_start_pos, DEBUG))
       return(false);

    UnpackAlignment();
    GetTags();
     // start working to unpack the read data we need
 /*   startFlow = 0;
    start_pos = alignment.Position;
    strand = !alignment.IsReverseStrand(); // true for positive, false for negative
    cigar = alignment.CigarData;;
    base_seq = alignment.QueryBases;

    GetUsefulTags(DEBUG);
    UnpackIntoFlowIndex(global_context.flowOrder);
    get_alignments(base_seq,chrseq, start_pos, cigar,ref_aln,seq_aln);

    startHC = endHC = startSC = endSC = 0;
    getHardClipPos(cigar, startHC, startSC, endHC, endSC);*/
    return(true); // happy to have unpacked this read
}

