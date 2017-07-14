/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "api/BamReader.h"
#include "ReservoirSample.h"
#include "../Analysis/file-io/ion_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include "BamHelper.h"

using namespace std;


RcppExport SEXP readBamReadGroup(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

// isolate c++ from Rcpp
  MyBamGroup my_bam_group;

  my_bam_group.ReadGroup(bamFile);

// Rcpp
  if (my_bam_group.errMsg.size()>0)
		Rf_error(strdup(my_bam_group.errMsg.c_str()));


    SEXP ret = R_NilValue;
	if(my_bam_group.ID.size() > 0) {
       ret = Rcpp::List::create(Rcpp::Named("ID")                   = my_bam_group.ID,
                                Rcpp::Named("FlowOrder")            = my_bam_group.FlowOrder,
                                Rcpp::Named("KeySequence")          = my_bam_group.KeySequence,
                                Rcpp::Named("Description")          = my_bam_group.Description,
                                Rcpp::Named("Library")              = my_bam_group.Library,
                                Rcpp::Named("PlatformUnit")         = my_bam_group.PlatformUnit,
                                Rcpp::Named("PredictedInsertSize")  = my_bam_group.PredictedInsertSize,
                                Rcpp::Named("ProductionDate")       = my_bam_group.ProductionDate,
                                Rcpp::Named("Program")              = my_bam_group.Program,
                                Rcpp::Named("Sample")               = my_bam_group.Sample,
                                Rcpp::Named("SequencingCenter")     = my_bam_group.SequencingCenter,
                                Rcpp::Named("SequencingTechnology") = my_bam_group.SequencingTechnology);
	}
    return ret;
}

class MyBamSequence{
public:
  	std::vector< std::string > AssemblyID;
	std::vector< std::string > Checksum;
	std::vector< std::string > Length;
	std::vector< std::string > Name;
	std::vector< std::string > Species;
	std::vector< std::string > URI;

  std::string errMsg;

  void ReadBamSequence(char *bamFile);
};

void MyBamSequence::ReadBamSequence(char *bamFile){
	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		 errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
	} else {
		BamTools::SamHeader samHeader = bamReader.GetHeader();
		for (BamTools::SamSequenceIterator itr = samHeader.Sequences.Begin(); itr != samHeader.Sequences.End(); ++itr ) {
			if(itr->HasAssemblyID()) {
				AssemblyID.push_back(itr->AssemblyID);
			} else {
				AssemblyID.push_back("");
			}
			if(itr->HasChecksum()) {
				Checksum.push_back(itr->Checksum);
			} else {
				Checksum.push_back("");
			}
			if(itr->HasLength()) {
				Length.push_back(itr->Length);
			} else {
				Length.push_back("");
			}
			if(itr->HasName()) {
				Name.push_back(itr->Name);
			} else {
				Name.push_back("");
			}
			if(itr->HasSpecies()) {
				Species.push_back(itr->Species);
			} else {
				Species.push_back("");
			}
			if(itr->HasURI()) {
				URI.push_back(itr->URI);
			} else {
				URI.push_back("");
			}
		}
		bamReader.Close();
	}

};

RcppExport SEXP readBamSequence(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

  MyBamSequence my_bam_sequence;
  my_bam_sequence.ReadBamSequence(bamFile);
  if (my_bam_sequence.errMsg.size()>0)
		Rf_error(strdup(my_bam_sequence.errMsg.c_str()));

    SEXP ret = R_NilValue;
	if(my_bam_sequence.AssemblyID.size() > 0) {
      ret = Rcpp::List::create(Rcpp::Named("AssemblyID") = my_bam_sequence.AssemblyID,
                               Rcpp::Named("Checksum")   = my_bam_sequence.Checksum,
                               Rcpp::Named("Length")     = my_bam_sequence.Length,
                               Rcpp::Named("Name")       = my_bam_sequence.Name,
                               Rcpp::Named("Species")    = my_bam_sequence.Species,
                               Rcpp::Named("URI")        = my_bam_sequence.URI);
    }
    return ret;
}

RcppExport SEXP readBamHeader(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

	std::string Version = "";
	std::string SortOrder = "";
	std::string GroupOrder = "";

	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		std::string errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
		Rf_error(strdup(errMsg.c_str()));
	} else {
		BamTools::SamHeader samHeader = bamReader.GetHeader();
		if(samHeader.HasVersion())
			Version = samHeader.Version;
		else
			Version = "";
		if(samHeader.HasSortOrder())
			SortOrder = samHeader.SortOrder;
		else
			SortOrder = "";
		if(samHeader.HasGroupOrder())
			GroupOrder = samHeader.GroupOrder;
		else
			GroupOrder = "";
		bamReader.Close();
	}

    return Rcpp::List::create(Rcpp::Named("Version")    = Version,
                              Rcpp::Named("SortOrder")  = SortOrder,
                              Rcpp::Named("GroupOrder") = GroupOrder);
}

// Reading flowgram information like ZM from BAM
bool ReadFlowgram(string tag, const BamTools::BamAlignment & alignment, Rcpp::NumericMatrix & out_data, unsigned int num_flows, unsigned int read_idx){

  std::vector<int16_t> flowMeasured; // round(256*val), signed
  bool success = alignment.GetTag(tag, flowMeasured);

  if(success){
    unsigned int i=0;
    for(; i<std::min(num_flows,(unsigned int)flowMeasured.size()); i++)
      out_data(read_idx,i) = flowMeasured[i]/256.0;
      while(i<num_flows)
        out_data(read_idx,i++) = 0; // which is bad because will lead to biases in extrapolation
    }

  return success;
}


class UsefulBamData{
public:
		std::map< std::string, int > keyLen;
		unsigned int nFlowFZ;
		unsigned int nFlowZM;
		unsigned int nPhase;
    std::map<std::string, int> groupID;
    UsefulBamData(){
      nFlowFZ = 0;
      nFlowZM = 0;
      nPhase  = 3;
    }
};

RcppExport SEXP readIonBam(SEXP RbamFile, SEXP Rcol, SEXP Rrow, SEXP RmaxBases, SEXP RnSample, SEXP RrandomSeed, SEXP RwantedGroupID, SEXP RhaveWantedGroups, SEXP RwantMappingData, SEXP RmaxCigarLength) {

	SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
	char *exceptionMesg = NULL;

	try {

		char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);
		Rcpp::IntegerVector              col(Rcol);
		Rcpp::IntegerVector              row(Rrow);
		unsigned int maxBases          = Rcpp::as<int>(RmaxBases);
		unsigned int nSample           = Rcpp::as<int>(RnSample);
		int randomSeed                 = Rcpp::as<int>(RrandomSeed);
		bool haveWantedGroups          = Rcpp::as<bool>(RhaveWantedGroups);
		bool wantMappingData           = Rcpp::as<bool>(RwantMappingData);
		unsigned int maxCigarLength    = Rcpp::as<int>(RmaxCigarLength);

		// Quick first pass through bam file to determine read group ID and nFlow
    UsefulBamData my_cache;

		std::string errMsg = getQuickStats(std::string(bamFile), my_cache.keyLen, my_cache.nFlowFZ, my_cache.nFlowZM);
		if(errMsg != "") {
			exceptionMesg = strdup(errMsg.c_str());
		}

		if(haveWantedGroups) {
			Rcpp::StringVector wantedGroupID(RwantedGroupID);
			if(wantedGroupID.size() > 0)
				for(int i=0; i<wantedGroupID.size(); i++)
					if(my_cache.keyLen.find(Rcpp::as<std::string>(wantedGroupID(i))) != my_cache.keyLen.end())
						my_cache.groupID[Rcpp::as<std::string>(wantedGroupID(i))] = 1;
			if(my_cache.groupID.size() == 0) {
				std::string errMsg = std::string("None of the wanted read group IDs found in ") + bamFile + std::string("\n");
				exceptionMesg = strdup(errMsg.c_str());
			}
		}


		// Some variables related to sampling
		std::map<std::string, int> wellIndex;
		std::vector< BamTools::BamAlignment > alignmentSample;
		bool filterByCoord = false;
		unsigned int nRead=0;
		if(col.size() > 0 || row.size() > 0) {
			// A subset of rows and columns is specified
			filterByCoord = true;
			if(col.size() != row.size()) {
				exceptionMesg = strdup("col and row should have the same number of entries, ignoring\n");
			} else {
				for(int i=0; i<col.size(); i++) {
					std::stringstream wellIdStream;
					wellIdStream << col(i) << ":" << row(i);
					wellIndex[wellIdStream.str()] = i;
				}
			}
		} else {
			// Set up for sampling, if requested
			ReservoirSample< BamTools::BamAlignment > sample;
			if(nSample > 0) {
				sample.Init(nSample,randomSeed);
			}

			BamTools::BamReader bamReader;
			bamReader.Open(std::string(bamFile));
			BamTools::BamAlignment alignment;
			while(bamReader.GetNextAlignmentCore(alignment)) {
				if(haveWantedGroups) {
					std::string thisReadGroupID = "";
					alignment.BuildCharData();
					alignment.GetTag("RG", thisReadGroupID);
					if( !alignment.GetTag("RG", thisReadGroupID) || (my_cache.groupID.find(thisReadGroupID)==my_cache.groupID.end()) )
						continue;
				}
				nRead++;
				if(nSample > 0)
					sample.Add(alignment);
			}
			bamReader.Close();
			if(nSample > 0){
				sample.Finished();
                                alignmentSample = sample.GetData();
                        }
		}

		unsigned int nReadOut = nRead;
		if(nSample > 0)
			nReadOut = nSample;
		else if(filterByCoord)
			nReadOut = wellIndex.size();

		// Initialize things to return
		Rcpp::StringVector     out_id(nReadOut);
		Rcpp::StringVector     out_readGroup(nReadOut);
		Rcpp::IntegerVector    out_col(nReadOut);
		Rcpp::IntegerVector    out_row(nReadOut);
		Rcpp::IntegerVector    out_length(nReadOut);
		Rcpp::IntegerVector    out_fullLength(nReadOut);
		Rcpp::IntegerVector    out_clipQualLeft(nReadOut);
		Rcpp::IntegerVector    out_clipQualRight(nReadOut);
		Rcpp::IntegerVector    out_clipAdapterLeft(nReadOut);
		Rcpp::IntegerVector    out_clipAdapterRight(nReadOut);
		Rcpp::IntegerVector    out_adapterOverlap(nReadOut);
		Rcpp::IntegerVector    out_flowClipLeft(nReadOut);
		Rcpp::IntegerVector    out_flowClipRight(nReadOut);
		Rcpp::IntegerVector    out_lastInsertFlow(nReadOut);
		Rcpp::IntegerVector    out_adapterType(nReadOut);
		Rcpp::NumericMatrix    out_flow(nReadOut,my_cache.nFlowFZ);
		Rcpp::NumericMatrix    out_meas(nReadOut,my_cache.nFlowZM);
		Rcpp::NumericMatrix    out_phase(nReadOut, my_cache.nPhase);
		Rcpp::StringVector     out_base(nReadOut);
		Rcpp::IntegerMatrix    out_qual(nReadOut,maxBases);
		Rcpp::IntegerMatrix    out_flowIndex(nReadOut,maxBases);

		// Debug normalization values
		Rcpp::NumericMatrix    out_additive(nReadOut,my_cache.nFlowZM);
		Rcpp::NumericMatrix    out_multiplicative(nReadOut,my_cache.nFlowZM);
		Rcpp::NumericMatrix    out_key_norm(nReadOut,my_cache.nFlowZM);
		Rcpp::NumericMatrix    out_uncalibrated(nReadOut,my_cache.nFlowZM);

		// Alignment-related data
		Rcpp::IntegerVector    out_aligned_flag(nReadOut);
		Rcpp::StringVector     out_aligned_base(nReadOut);
		Rcpp::IntegerVector    out_aligned_refid(nReadOut);
		Rcpp::IntegerVector    out_aligned_pos(nReadOut);
		Rcpp::IntegerVector    out_aligned_mapq(nReadOut);
		Rcpp::IntegerVector    out_aligned_bin(nReadOut);
		Rcpp::StringVector     out_aligned_cigar_type(nReadOut);
		Rcpp::NumericMatrix    out_aligned_cigar_len(nReadOut,maxCigarLength);

		Rcpp::StringVector     out_qDNA(nReadOut);
		Rcpp::StringVector     out_match(nReadOut);
		Rcpp::StringVector     out_tDNA(nReadOut);

		Rcpp::IntegerVector    out_q7Len(nReadOut);
		Rcpp::IntegerVector    out_q10Len(nReadOut);
		Rcpp::IntegerVector    out_q17Len(nReadOut);
		Rcpp::IntegerVector    out_q20Len(nReadOut);
		Rcpp::IntegerVector    out_q47Len(nReadOut);

		// Structures Reads: Read hard clipped sequence bits from tags
		Rcpp::StringVector     out_startUMI(nReadOut);       // ZT tag
		Rcpp::StringVector     out_endUMI(nReadOut);         // YT tag
		Rcpp::StringVector     out_ExtraClipLeft(nReadOut);  // ZE tag
		Rcpp::StringVector     out_ExtraClipRight(nReadOut); // YE tag

		bool                   have_startUMI       = false;
		bool                   have_endUMI         = false;
		bool                   have_ExtraClipLeft  = false;
		bool                   have_ExtraClipRight = false;

		// Extra tags for reading the spades file into R XXX
		//Rcpp::NumericVector    out_SpadesDelta(nReadOut);
		//Rcpp::NumericVector    out_SpadesFit(nReadOut);
		//Rcpp::StringVector     out_SpadesAlt(nReadOut);


		// Reopen the BAM, unless we already sampled the reads
		BamTools::BamReader bamReader;
		if(nSample==0)
			bamReader.Open(std::string(bamFile));
		unsigned int nReadsFromBam=0;
		BamTools::BamAlignment alignment;
		bool haveMappingData=false;
		bool have_debug_bam= false;
		bool have_uncalibrated_flows = false;

		while(getNextAlignment(alignment,bamReader,my_cache.groupID,alignmentSample,wellIndex,nSample)) {
			int thisCol = 0;
			int thisRow = 0;
			if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
				std::cerr << "Error parsing read name: " << alignment.Name << "\n";
			std::string readGroup = "";
			alignment.GetTag("RG", readGroup);

			// Store values that will be returned
			out_id(nReadsFromBam)               = alignment.Name;
			out_readGroup(nReadsFromBam)        = readGroup;
			out_col(nReadsFromBam)              = thisCol;
			out_row(nReadsFromBam)              = thisRow;
			out_clipQualLeft(nReadsFromBam)     = 0;
			out_clipQualRight(nReadsFromBam)    = 0;

			std::map<std::string, int>::iterator keyLenIter;
			keyLenIter = my_cache.keyLen.find(readGroup);
			out_clipAdapterLeft(nReadsFromBam) = (keyLenIter != my_cache.keyLen.end()) ? keyLenIter->second : 0;

			int64_t clipAdapterRight = 0;
			getTagParanoid(alignment,"ZA",clipAdapterRight);
			out_clipAdapterRight(nReadsFromBam) = clipAdapterRight;

            int64_t adapterOverlap = 0;
            getTagParanoid(alignment,"ZB",adapterOverlap);
            out_adapterOverlap(nReadsFromBam) = adapterOverlap;

			int64_t flowClipLeft = 0;
			getTagParanoid(alignment,"ZF",flowClipLeft);
			out_flowClipLeft(nReadsFromBam) = flowClipLeft;

			int64_t flowClipRight = 0;
			getTagParanoid(alignment,"ZG",flowClipRight);
			out_flowClipRight(nReadsFromBam) = flowClipRight;

			std::vector<int32_t> zm_tag_vec;
			if(alignment.GetTag("ZC", zm_tag_vec)){
				out_lastInsertFlow(nReadsFromBam) = zm_tag_vec.at(1);
				out_adapterType(nReadsFromBam)    = zm_tag_vec.at(3);
			}

			// Not every read is guaranteed to have structures detected
			string temp_tag;

			if (alignment.GetTag("ZT", temp_tag)){
			  have_startUMI = true;
			  out_startUMI(nReadsFromBam) = temp_tag;
			}

			if (alignment.GetTag("YT", temp_tag)){
			  have_endUMI = true;
			  out_endUMI(nReadsFromBam) = temp_tag;
			}

			if (alignment.GetTag("ZE", temp_tag)){
			  have_ExtraClipLeft = true;
			  out_ExtraClipLeft(nReadsFromBam) = temp_tag;
			}

			if (alignment.GetTag("YE", temp_tag)){
			  have_ExtraClipRight = true;
			  out_ExtraClipRight(nReadsFromBam) = temp_tag;
			}

			// Read extra spades tags XXX
            //float spades_delta = 0;
            //if (alignment.GetTag("YD", spades_delta))
            //  out_SpadesDelta(nReadsFromBam) = spades_delta;
            //float spades_fit = 0;
            //if (alignment.GetTag("YF", spades_fit))
            //  out_SpadesFit(nReadsFromBam) = spades_fit;
            //string spades_alt;
            //if (alignment.GetTag("YR", spades_alt))
            //  out_SpadesAlt(nReadsFromBam) = spades_alt;

			std::vector<uint16_t> flowInt;
			if(alignment.GetTag("FZ", flowInt)){
				unsigned int i=0;
				for(; i<std::min(my_cache.nFlowFZ,(unsigned int)flowInt.size()); i++)
					out_flow(nReadsFromBam,i) = flowInt[i] / 100.0;
				while(i<my_cache.nFlowFZ)
					out_flow(nReadsFromBam,i++) = 0;
			}

			// Read normalized measurements in ZM tag
			ReadFlowgram("ZM", alignment, out_meas, my_cache.nFlowZM, nReadsFromBam);

			// Read debug quantities if they are available
            have_debug_bam = ReadFlowgram("Ya", alignment, out_additive, my_cache.nFlowZM, nReadsFromBam);
            if (have_debug_bam){
              ReadFlowgram("Yb", alignment, out_multiplicative, my_cache.nFlowZM, nReadsFromBam);
              ReadFlowgram("Yw", alignment, out_key_norm, my_cache.nFlowZM, nReadsFromBam);
              have_uncalibrated_flows = ReadFlowgram("Yx", alignment, out_uncalibrated, my_cache.nFlowZM, nReadsFromBam);
            }

			// experimental tag for Project Razor: "phase" values
			std::vector<float> flowPhase;
			if(alignment.GetTag("ZP", flowPhase)){
				unsigned int i=0;
				for(; i<std::min(my_cache.nPhase,(unsigned int)flowPhase.size()); i++)
					out_phase(nReadsFromBam,i) = flowPhase[i];
				while(i<my_cache.nPhase)
					out_phase(nReadsFromBam,i++) = 0;
			}


			// limit scope of loop as we have too many "loop" variables named i running around
			if (true){
				unsigned int nBases = alignment.QueryBases.length();
				unsigned int trimLength = std::min(maxBases, nBases);
				out_fullLength(nReadsFromBam) = nBases;
				out_length(nReadsFromBam)     = trimLength;
				out_base(nReadsFromBam)       = alignment.QueryBases;
				unsigned int i=0;
				for(; i<trimLength; i++) {
					out_qual(nReadsFromBam,i) = ((int) alignment.Qualities[i]) - 33;
					//TODO - fill in proper flowindex info
					out_flowIndex(nReadsFromBam,i)  = 0;
				}
				// Pad out remaining bases with null entries
				for(; i<maxBases; i++) {
					out_qual(nReadsFromBam,i)         = 0;
					out_flowIndex(nReadsFromBam,i)    = 0;
				}
			}

			if(wantMappingData && alignment.IsMapped()) {
				if(!haveMappingData)
					haveMappingData=true;
				out_aligned_flag(nReadsFromBam)  = alignment.AlignmentFlag;
				out_aligned_base(nReadsFromBam)  = alignment.AlignedBases;
				out_aligned_refid(nReadsFromBam) = alignment.RefID;
				out_aligned_pos(nReadsFromBam)   = alignment.Position;
				out_aligned_mapq(nReadsFromBam)  = alignment.MapQuality;
				out_aligned_bin(nReadsFromBam)   = alignment.Bin;
				unsigned int cigarLength = std::min(maxCigarLength, (unsigned int) alignment.CigarData.size());
				unsigned int iCig=0;
				std::string temp;
				temp.clear();
				temp.reserve(cigarLength);
				for(iCig=0; iCig < cigarLength; iCig++) {
					temp.push_back(alignment.CigarData[iCig].Type);
					out_aligned_cigar_len(nReadsFromBam,iCig)   = alignment.CigarData[iCig].Length;
				}
				out_aligned_cigar_type(nReadsFromBam) = temp;
				for(; iCig < maxCigarLength; iCig++)
					out_aligned_cigar_len(nReadsFromBam,iCig) = 0;

				string tDNA, pad_query, pad_match, pad_target;
				string qSeq = alignment.QueryBases;
				string qDNA = alignment.AlignedBases;
				string md;
				alignment.GetTag("MD", md);

				dna( qSeq, alignment.CigarData, md, tDNA );
				padded_alignment( alignment.CigarData, qDNA, tDNA, pad_query, pad_target, pad_match, alignment.IsReverseStrand());
				std::vector<int> qlen = score_alignments(pad_query, pad_target, pad_match );

				if( alignment.IsReverseStrand() ){
					reverse_comp(pad_target);
					reverse_comp(pad_query);
					std::reverse( pad_match.begin(), pad_match.end() );
				}
				out_qDNA(nReadsFromBam)  = pad_query;
				out_tDNA(nReadsFromBam)  = pad_target;
				out_match(nReadsFromBam) = pad_match;

				out_q7Len(nReadsFromBam)  = qlen[0];
				out_q10Len(nReadsFromBam) = qlen[1];
				out_q17Len(nReadsFromBam) = qlen[2];
				out_q20Len(nReadsFromBam) = qlen[3];
				out_q47Len(nReadsFromBam) = qlen[4];
			}
			nReadsFromBam++;
		}
		if(nSample==0)
			bamReader.Close();

		// Organize and return results
        std::map<std::string,SEXP> map;
		if(nReadsFromBam == 0) {
			std::cerr << "WARNING: No matching reads found in " << bamFile << "\n";
		} else if(nReadsFromBam != nReadOut) {
			// If we find fewer reads than expected then issue warning and trim back data structures
			std::cerr << "WARNING: Expected to find " << nReadOut << " reads but got " << nReadsFromBam << " in " << bamFile << "\n";
			if(filterByCoord)
				std::cerr << "Some of the requested reads are missing from the SFF.\n";
			Rcpp::StringVector     out2_id(nReadsFromBam);
			Rcpp::StringVector     out2_groupID(nReadsFromBam);
			Rcpp::IntegerVector    out2_col(nReadsFromBam);
			Rcpp::IntegerVector    out2_row(nReadsFromBam);
			Rcpp::IntegerVector    out2_length(nReadsFromBam);
			Rcpp::IntegerVector    out2_fullLength(nReadsFromBam);
			Rcpp::IntegerVector    out2_clipQualLeft(nReadsFromBam);
			Rcpp::IntegerVector    out2_clipQualRight(nReadsFromBam);
			Rcpp::IntegerVector    out2_clipAdapterLeft(nReadsFromBam);
			Rcpp::IntegerVector    out2_clipAdapterRight(nReadsFromBam);
            Rcpp::IntegerVector    out2_adapterOverlap(nReadsFromBam);
			Rcpp::IntegerVector    out2_flowClipLeft(nReadsFromBam);
			Rcpp::IntegerVector    out2_flowClipRight(nReadsFromBam);
			Rcpp::IntegerVector    out2_lastInsertFlow(nReadsFromBam);
			Rcpp::IntegerVector    out2_adapterType(nReadsFromBam);
			Rcpp::NumericMatrix    out2_flow(nReadsFromBam,my_cache.nFlowFZ);
			//razor
			Rcpp::NumericMatrix    out2_meas(nReadsFromBam,my_cache.nFlowZM);
			Rcpp::NumericMatrix    out2_phase(nReadsFromBam,my_cache.nPhase);
			Rcpp::NumericMatrix    out2_additive(nReadsFromBam,my_cache.nFlowZM);
			Rcpp::NumericMatrix    out2_multiplicative(nReadsFromBam,my_cache.nFlowZM);
			Rcpp::NumericMatrix    out2_key_norm(nReadsFromBam,my_cache.nFlowZM);
			Rcpp::NumericMatrix    out2_uncalibrated(nReadsFromBam,my_cache.nFlowZM);
			// end
			Rcpp::StringVector     out2_base(nReadsFromBam);
			Rcpp::IntegerMatrix    out2_qual(nReadsFromBam,maxBases);
			Rcpp::IntegerMatrix    out2_flowIndex(nReadsFromBam,maxBases);
			Rcpp::IntegerVector    out2_aligned_flag(nReadsFromBam);
			Rcpp::StringVector     out2_aligned_base(nReadsFromBam);
			Rcpp::IntegerVector    out2_aligned_refid(nReadsFromBam);
			Rcpp::IntegerVector    out2_aligned_pos(nReadsFromBam);
			Rcpp::IntegerVector    out2_aligned_mapq(nReadsFromBam);
			Rcpp::IntegerVector    out2_aligned_bin(nReadsFromBam);
			Rcpp::StringVector     out2_aligned_cigar_type(nReadsFromBam);
			Rcpp::NumericMatrix    out2_aligned_cigar_len(nReadsFromBam,maxCigarLength);
			Rcpp::StringVector     out2_qDNA(nReadsFromBam);
			Rcpp::StringVector     out2_match(nReadsFromBam);
			Rcpp::StringVector     out2_tDNA(nReadsFromBam);

			Rcpp::IntegerVector    out2_q7Len(nReadsFromBam);
			Rcpp::IntegerVector    out2_q10Len(nReadsFromBam);
			Rcpp::IntegerVector    out2_q17Len(nReadsFromBam);
			Rcpp::IntegerVector    out2_q20Len(nReadsFromBam);
			Rcpp::IntegerVector    out2_q47Len(nReadsFromBam);

			Rcpp::StringVector     out2_startUMI(nReadOut);
			Rcpp::StringVector     out2_endUMI(nReadOut);
			Rcpp::StringVector     out2_ExtraClipLeft(nReadOut);
			Rcpp::StringVector     out2_ExtraClipRight(nReadOut);

			// Spades XXX
			//Rcpp::NumericVector    out2_SpadesDelta(nReadsFromBam);
			//Rcpp::NumericVector    out2_SpadesFit(nReadsFromBam);
			//Rcpp::StringVector     out2_SpadesAlt(nReadsFromBam);


			for(unsigned int i=0; i<nReadsFromBam; i++) {
				out2_id(i)               = out_id(i);
				out2_groupID(i)          = out_readGroup(i);
				out2_col(i)              = out_col(i);
				out2_row(i)              = out_row(i);
				out2_clipQualLeft(i)     = out_clipQualLeft(i);
				out2_clipQualRight(i)    = out_clipQualRight(i);
				out2_clipAdapterLeft(i)  = out_clipAdapterLeft(i);
				out2_clipAdapterRight(i) = out_clipAdapterRight(i);
                out2_adapterOverlap(i)   = out_adapterOverlap(i);
				out2_flowClipLeft(i)     = out_flowClipLeft(i);
				out2_flowClipRight(i)    = out_flowClipRight(i);

				out2_lastInsertFlow(i)   = out_lastInsertFlow(i);
				out2_adapterType(i)      = out_adapterType(i);


				out2_length(i)           = out_length(i);
				out2_fullLength(i)       = out_fullLength(i);
				out2_base(i)             = out_base(i);
				for(unsigned int j=0; j<my_cache.nFlowFZ; j++)
					out2_flow(i,j) = out_flow(i,j);
				for(unsigned int j=0; j<my_cache.nFlowZM; j++)
					out2_meas(i,j) = out_meas(i,j);
				for(unsigned int j=0; j<my_cache.nPhase; j++)
					out2_phase(i,j) = out_phase(i,j);
				for(int j=0; j<out2_length(i); j++) {
					out2_qual(i,j)         = out_qual(i,j);
					out2_flowIndex(i,j)    = out_flowIndex(i,j);
				}

				out2_startUMI(i) = out_startUMI(i);
				out2_endUMI(i) = out_endUMI(i);
				out2_ExtraClipLeft(i) = out_ExtraClipLeft(i);
				out2_ExtraClipRight(i) = out_ExtraClipRight(i);

				if(haveMappingData) {
					out2_aligned_flag(i)       = out_aligned_flag(i);
					out2_aligned_base(i)       = out_aligned_base(i);
					out2_aligned_refid(i)      = out_aligned_refid(i);
					out2_aligned_pos(i)        = out_aligned_pos(i);
					out2_aligned_mapq(i)       = out_aligned_mapq(i);
					out2_aligned_bin(i)        = out_aligned_bin(i);
					out2_aligned_cigar_type(i) = out_aligned_cigar_type(i);
					for(unsigned int j=0; j<maxCigarLength; j++)
						out2_aligned_cigar_len(i,j) = out_aligned_cigar_len(i,j);

					out2_qDNA(i) = out_qDNA(i);
					out2_tDNA(i) = out_tDNA(i);
					out2_match(i) = out_match(i);
					out2_q7Len(i) = out_q7Len(i);
					out2_q10Len(i) = out_q10Len(i);
					out2_q17Len(i) = out_q17Len(i);
					out2_q20Len(i) = out_q20Len(i);
					out2_q47Len(i) = out_q47Len(i);

					// Spades XXX
					//out2_SpadesDelta(i) = out_SpadesDelta(i);
					//out2_SpadesFit(i) = out_SpadesFit(i);
					//out2_SpadesAlt(i) = out_SpadesAlt(i);
				}

				if (have_debug_bam){
					for(unsigned int j=0; j<my_cache.nFlowZM; j++){
						out2_additive(i,j) = out_additive(i,j);
						out2_multiplicative(i,j) = out_multiplicative(i,j);
						out2_key_norm(i,j) = out_key_norm(i,j);
					}
					if (have_uncalibrated_flows){
						for(unsigned int j=0; j<my_cache.nFlowZM; j++)
							out2_uncalibrated(i,j) = out_uncalibrated(i,j);
					}
				}
			}

            /// map data
            map["nFlow"]            = Rcpp::wrap( (int) std::max(my_cache.nFlowFZ,my_cache.nFlowZM));
            map["id"]               = Rcpp::wrap( out2_id );
            map["readGroup"]        = Rcpp::wrap( out2_groupID );
	        map["col"]              = Rcpp::wrap( out2_col );
	        map["row"]              = Rcpp::wrap( out2_row );
		    map["length"]           = Rcpp::wrap( out2_length );
            map["fullLength"]       = Rcpp::wrap( out2_fullLength );
	        map["clipQualLeft"]     = Rcpp::wrap( out2_clipQualLeft );
	        map["clipQualRight"]    = Rcpp::wrap( out2_clipQualRight );
	        map["clipAdapterLeft"]  = Rcpp::wrap( out2_clipAdapterLeft );
	        map["clipAdapterRight"] = Rcpp::wrap( out2_clipAdapterRight );
	        map["flowClipLeft"]     = Rcpp::wrap( out2_flowClipLeft );
	        map["flowClipRight"]    = Rcpp::wrap( out2_flowClipRight );
	        map["flow"]             = Rcpp::wrap( out2_flow );
	        map["measured"]         = Rcpp::wrap( out2_meas );
	        map["phase"]            = Rcpp::wrap( out2_phase );
	        map["base"]             = Rcpp::wrap( out2_base );
	        map["qual"]             = Rcpp::wrap( out2_qual );

	        // Structured reads, report tags only if present in BAM
	        if (have_startUMI)
	          map["startTag"]       = Rcpp::wrap( out2_startUMI );
	        if (have_endUMI)
	          map["endTag"]         = Rcpp::wrap( out2_endUMI );
	        if (have_ExtraClipLeft)
	          map["extraClipLeft"]  = Rcpp::wrap( out2_ExtraClipLeft );
	        if (have_ExtraClipRight)
	          map["extraClipRight"] = Rcpp::wrap( out2_ExtraClipRight );

			if(haveMappingData) {
              map["alignFlag"]       = Rcpp::wrap( out2_aligned_flag );
              map["alignBase"]       = Rcpp::wrap( out2_aligned_base );
              map["alignRefID"]      = Rcpp::wrap( out2_aligned_refid );
              map["alignPos"]        = Rcpp::wrap( out2_aligned_pos );
              map["alignMapq"]       = Rcpp::wrap( out2_aligned_mapq );
              map["alignBin"]        = Rcpp::wrap( out2_aligned_bin );
              map["alignCigarType"]  = Rcpp::wrap( out2_aligned_cigar_type );
              map["alignCigarLen"]   = Rcpp::wrap( out2_aligned_cigar_len );
              map["qDNA"]            = Rcpp::wrap( out2_qDNA );
              map["tDNA"]            = Rcpp::wrap( out2_tDNA );
              map["match"]           = Rcpp::wrap( out2_match );
              map["q7Len"]           = Rcpp::wrap( out2_q7Len );
              map["q10Len"]          = Rcpp::wrap( out2_q10Len );
              map["q17Len"]          = Rcpp::wrap( out2_q17Len );
              map["q20Len"]          = Rcpp::wrap( out2_q20Len );
              map["q47Len"]          = Rcpp::wrap( out2_q47Len );
              // Spades XXX
              //map["SpadesDelta"]     = Rcpp::wrap( out2_SpadesDelta );
              //map["SpadesFit"]      = Rcpp::wrap( out2_SpadesFit );
              //map["SpadesAlt"]      = Rcpp::wrap( out2_SpadesAlt );
			}
			if (have_debug_bam){
				map["normAdditive"]       = Rcpp::wrap( out2_additive );
				map["normMultiplicative"] = Rcpp::wrap( out2_multiplicative );
				map["keyNorm"]            = Rcpp::wrap( out2_key_norm );
				if (have_uncalibrated_flows)
					map["uncalibrated"]       = Rcpp::wrap( out2_uncalibrated );
			}
		} else {
             map["nFlow"]            = Rcpp::wrap( (int) std::max(my_cache.nFlowFZ,my_cache.nFlowZM) );
             map["id"]               = Rcpp::wrap( out_id );
             map["readGroup"]        = Rcpp::wrap( out_readGroup );
             map["col"]              = Rcpp::wrap( out_col );
             map["row"]              = Rcpp::wrap( out_row );
             map["length"]           = Rcpp::wrap( out_length );
             map["fullLength"]       = Rcpp::wrap( out_fullLength );
             map["clipQualLeft"]     = Rcpp::wrap( out_clipQualLeft );
             map["clipQualRight"]    = Rcpp::wrap( out_clipQualRight );
             map["adapterOverlap"]   = Rcpp::wrap( out_adapterOverlap );
             map["clipAdapterLeft"]  = Rcpp::wrap( out_clipAdapterLeft );
             map["clipAdapterRight"] = Rcpp::wrap( out_clipAdapterRight );
             map["flowClipLeft"]     = Rcpp::wrap( out_flowClipLeft );
             map["flowClipRight"]    = Rcpp::wrap( out_flowClipRight );
             map["lastInsertFlow"]   = Rcpp::wrap( out_lastInsertFlow );
             map["adapterType"]      = Rcpp::wrap( out_adapterType );
             map["flow"]             = Rcpp::wrap( out_flow );
             map["measured"]         = Rcpp::wrap( out_meas );
             map["phase"]            = Rcpp::wrap( out_phase );
             map["base"]             = Rcpp::wrap( out_base );
             map["qual"]             = Rcpp::wrap( out_qual );
             map["flowIndex"]        = Rcpp::wrap( out_flowIndex );

 	         // Structured reads, report tags only if present in BAM
 	         if (have_startUMI)
 	           map["startTag"]       = Rcpp::wrap( out_startUMI );
 	         if (have_endUMI)
 	           map["endTag"]         = Rcpp::wrap( out_endUMI );
 	         if (have_ExtraClipLeft)
 	           map["extraClipLeft"]  = Rcpp::wrap( out_ExtraClipLeft );
 	         if (have_ExtraClipRight)
 	           map["extraClipRight"] = Rcpp::wrap( out_ExtraClipRight );

			if(haveMappingData) {
              map["alignFlag"]         = Rcpp::wrap( out_aligned_flag );
              map["alignBase"]         = Rcpp::wrap( out_aligned_base );
              map["alignRefID"]        = Rcpp::wrap( out_aligned_refid );
              map["alignPos"]          = Rcpp::wrap( out_aligned_pos );
              map["alignMapq"]         = Rcpp::wrap( out_aligned_mapq );
              map["alignBin"]          = Rcpp::wrap( out_aligned_bin );
              map["alignCigarType"]    = Rcpp::wrap( out_aligned_cigar_type );
              map["alignCigarLen"]     = Rcpp::wrap( out_aligned_cigar_len );
              map["qDNA"]              = Rcpp::wrap( out_qDNA );
              map["tDNA"]              = Rcpp::wrap( out_tDNA );
              map["match"]             = Rcpp::wrap( out_match );
              map["q7Len"]             = Rcpp::wrap( out_q7Len );
              map["q10Len"]            = Rcpp::wrap( out_q10Len );
              map["q17Len"]            = Rcpp::wrap( out_q17Len );
              map["q20Len"]            = Rcpp::wrap( out_q20Len );
              map["q47Len"]            = Rcpp::wrap( out_q47Len );
              // Spades XXX
              //map["SpadesDelta"]     = Rcpp::wrap( out_SpadesDelta );
              //map["SpadesFit"]      = Rcpp::wrap( out_SpadesFit );
              //map["SpadesAlt"]      = Rcpp::wrap( out_SpadesAlt );
			}
			if (have_debug_bam){
				map["normAdditive"]       = Rcpp::wrap( out_additive );
				map["normMultiplicative"] = Rcpp::wrap( out_multiplicative );
				map["keyNorm"]            = Rcpp::wrap( out_key_norm );
				if (have_uncalibrated_flows)
					map["uncalibrated"]       = Rcpp::wrap( out_uncalibrated );
			}
		}
        ret = Rcpp::wrap( map );

	} catch(std::exception& ex) {
		forward_exception_to_r(ex);
	} catch(...) {
		::Rf_error("c++ exception (unknown reason)");
	}

	if(exceptionMesg != NULL)
		Rf_error(exceptionMesg);

	return ret;
}

