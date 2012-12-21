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

using namespace std;

std::string getQuickStats(const std::string &bamFile, std::map< std::string, int > &keyLen, unsigned int &nFlowFZ, unsigned int &nFlowZM);
bool getNextAlignment(BamTools::BamAlignment &alignment, BamTools::BamReader &bamReader, const std::map<std::string, int> &groupID, std::vector< BamTools::BamAlignment > &alignmentSample, std::map<std::string, int> &wellIndex, unsigned int nSample);
bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value);


void dna(string &qDNA, const vector<BamTools::CigarOp>& cig, const string& md, string& tDNA);
void padded_alignment(const vector<BamTools::CigarOp>& cig, string& qDNA, string& tDNA, string& pad_query, string& pad_target, string& pad_match, bool isReversed);
void reverse_comp(std::string& c_dna);
std::vector<int> score_alignments(string& pad_source, string& pad_target, string& pad_match );

RcppExport SEXP readBamReadGroup(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

	std::vector< std::string > ID;
	std::vector< std::string > FlowOrder;
	std::vector< std::string > KeySequence;
	std::vector< std::string > Description;
	std::vector< std::string > Library;
	std::vector< std::string > PlatformUnit;
	std::vector< std::string > PredictedInsertSize;
	std::vector< std::string > ProductionDate;
	std::vector< std::string > Program;
	std::vector< std::string > Sample;
	std::vector< std::string > SequencingCenter;
	std::vector< std::string > SequencingTechnology;

	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		std::string errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
		Rf_error(copyMessageToR(errMsg.c_str()));
	} else {
		BamTools::SamHeader samHeader = bamReader.GetHeader();
		for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
			if(itr->HasID()) {
				ID.push_back(itr->ID);
			} else {
				ID.push_back("");
			}
			if(itr->HasFlowOrder()) {
				FlowOrder.push_back(itr->FlowOrder);
			} else {
				FlowOrder.push_back("");
			}
			if(itr->HasKeySequence()) {
				KeySequence.push_back(itr->KeySequence);
			} else {
				KeySequence.push_back("");
			}
			if(itr->HasDescription()) {
				Description.push_back(itr->Description);
			} else {
				Description.push_back("");
			}
			if(itr->HasLibrary()) {
				Library.push_back(itr->Library);
			} else {
				Library.push_back("");
			}
			if(itr->HasPlatformUnit()) {
				PlatformUnit.push_back(itr->PlatformUnit);
			} else {
				PlatformUnit.push_back("");
			}
			if(itr->HasPredictedInsertSize()) {
				PredictedInsertSize.push_back(itr->PredictedInsertSize);
			} else {
				PredictedInsertSize.push_back("");
			}
			if(itr->HasProductionDate()) {
				ProductionDate.push_back(itr->ProductionDate);
			} else {
				ProductionDate.push_back("");
			}
			if(itr->HasProgram()) {
				Program.push_back(itr->Program);
			} else {
				Program.push_back("");
			}
			if(itr->HasSample()) {
				Sample.push_back(itr->Sample);
			} else {
				Sample.push_back("");
			}
			if(itr->HasSequencingCenter()) {
				SequencingCenter.push_back(itr->SequencingCenter);
			} else {
				SequencingCenter.push_back("");
			}
			if(itr->HasSequencingTechnology()) {
				SequencingTechnology.push_back(itr->SequencingTechnology);
			} else {
				SequencingTechnology.push_back("");
			}
		}
		bamReader.Close();
	}

	RcppResultSet rs;
	if(ID.size() > 0) {
		rs.add("ID",                   ID);
		rs.add("FlowOrder",            FlowOrder);
		rs.add("KeySequence",          KeySequence);
		rs.add("Description",          Description);
		rs.add("Library",              Library);
		rs.add("PlatformUnit",         PlatformUnit);
		rs.add("PredictedInsertSize",  PredictedInsertSize);
		rs.add("ProductionDate",       ProductionDate);
		rs.add("Program",              Program);
		rs.add("Sample",               Sample);
		rs.add("SequencingCenter",     SequencingCenter);
		rs.add("SequencingTechnology", SequencingTechnology);
	}
	return(rs.getReturnList());
}

RcppExport SEXP readBamSequence(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

	std::vector< std::string > AssemblyID;
	std::vector< std::string > Checksum;
	std::vector< std::string > Length;
	std::vector< std::string > Name;
	std::vector< std::string > Species;
	std::vector< std::string > URI;

	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		std::string errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
		Rf_error(copyMessageToR(errMsg.c_str()));
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

	RcppResultSet rs;
	if(AssemblyID.size() > 0) {
		rs.add("AssemblyID",  AssemblyID);
		rs.add("Checksum",    Checksum);
		rs.add("Length",      Length);
		rs.add("Name",        Name);
		rs.add("Species",     Species);
		rs.add("URI",         URI);
	}
	return(rs.getReturnList());
}

RcppExport SEXP readBamHeader(SEXP RbamFile) {
	char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);

	std::string Version = "";
	std::string SortOrder = "";
	std::string GroupOrder = "";

	BamTools::BamReader bamReader;
	if(!bamReader.Open(std::string(bamFile))) {
		std::string errMsg = "Failed to open bam " + std::string(bamFile) + "\n";
		Rf_error(copyMessageToR(errMsg.c_str()));
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

	RcppResultSet rs;
	rs.add("Version",    Version);
	rs.add("SortOrder",  SortOrder);
	rs.add("GroupOrder", GroupOrder);
	return(rs.getReturnList());
}

RcppExport SEXP readIonBam(SEXP RbamFile, SEXP Rcol, SEXP Rrow, SEXP RmaxBases, SEXP RnSample, SEXP RrandomSeed, SEXP RwantedGroupID, SEXP RhaveWantedGroups, SEXP RwantMappingData, SEXP RmaxCigarLength) {

	SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
	char *exceptionMesg = NULL;

	try {

		char* bamFile         = (char *)Rcpp::as<const char*>(RbamFile);
		RcppVector<int> col(Rcol);
		RcppVector<int> row(Rrow);
		unsigned int maxBases          = Rcpp::as<int>(RmaxBases);
		unsigned int nSample           = Rcpp::as<int>(RnSample);
		int randomSeed                 = Rcpp::as<int>(RrandomSeed);
		bool haveWantedGroups          = Rcpp::as<bool>(RhaveWantedGroups);
		bool wantMappingData           = Rcpp::as<bool>(RwantMappingData);
		unsigned int maxCigarLength    = Rcpp::as<int>(RmaxCigarLength);

		// Quick first pass through bam file to determine read group ID and nFlow
		std::map< std::string, int > keyLen;
		unsigned int nFlowFZ=0;
		unsigned int nFlowZM=0;
		unsigned int nPhase = 3;
		std::string errMsg = getQuickStats(std::string(bamFile), keyLen, nFlowFZ, nFlowZM);
		if(errMsg != "") {
			exceptionMesg = copyMessageToR(errMsg.c_str());
		}
		std::map<std::string, int> groupID;

		if(haveWantedGroups) {
			RcppStringVector wantedGroupID(RwantedGroupID);
			if(wantedGroupID.size() > 0)
				for(int i=0; i<wantedGroupID.size(); i++)
					if(keyLen.find(wantedGroupID(i)) != keyLen.end())
						groupID[wantedGroupID(i)] = 1;
			if(groupID.size() == 0) {
				std::string errMsg = std::string("None of the wanted read group IDs found in ") + bamFile + std::string("\n");
				exceptionMesg = copyMessageToR(errMsg.c_str());
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
				exceptionMesg = copyMessageToR("col and row should have the same number of entries, ignoring\n");
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
					if( !alignment.GetTag("RG", thisReadGroupID) || (groupID.find(thisReadGroupID)==groupID.end()) )
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
		std::vector< std::string > out_id(nReadOut);
		std::vector< std::string > out_readGroup(nReadOut);
		RcppVector<int>    out_col(nReadOut);
		RcppVector<int>    out_row(nReadOut);
		RcppVector<int>    out_length(nReadOut);
		RcppVector<int>    out_fullLength(nReadOut);
		RcppVector<int>    out_clipQualLeft(nReadOut);
		RcppVector<int>    out_clipQualRight(nReadOut);
		RcppVector<int>    out_clipAdapterLeft(nReadOut);
		RcppVector<int>    out_clipAdapterRight(nReadOut);
		RcppVector<int>    out_flowClipLeft(nReadOut);
		RcppVector<int>    out_flowClipRight(nReadOut);
		RcppMatrix<double> out_flow(nReadOut,nFlowFZ);
		RcppMatrix<double> out_meas(nReadOut,nFlowZM);
		RcppMatrix<double> out_phase(nReadOut, nPhase);
		std::vector< std::string > out_base(nReadOut);
		RcppMatrix<int>    out_qual(nReadOut,maxBases);
		RcppMatrix<int>    out_flowIndex(nReadOut,maxBases);
		// Alignment-related data
		RcppVector<int>    out_aligned_flag(nReadOut);
		std::vector< std::string > out_aligned_base(nReadOut);
		RcppVector<int>    out_aligned_refid(nReadOut);
		RcppVector<int>    out_aligned_pos(nReadOut);
		RcppVector<int>    out_aligned_mapq(nReadOut);
		RcppVector<int>    out_aligned_bin(nReadOut);
		std::vector< std::string > out_aligned_cigar_type(nReadOut);
		RcppMatrix<double> out_aligned_cigar_len(nReadOut,maxCigarLength);

		std::vector< std::string > out_qDNA(nReadOut);
		std::vector< std::string > out_match(nReadOut);
		std::vector< std::string > out_tDNA(nReadOut);

		RcppVector<int>    out_q7Len(nReadOut);
		RcppVector<int>    out_q10Len(nReadOut);
		RcppVector<int>    out_q17Len(nReadOut);
		RcppVector<int>    out_q20Len(nReadOut);
		RcppVector<int>    out_q47Len(nReadOut);

		// Reopen the BAM, unless we already sampled the reads
		BamTools::BamReader bamReader;
		if(nSample==0)
			bamReader.Open(std::string(bamFile));
		unsigned int nReadsFromBam=0;
		BamTools::BamAlignment alignment;
		bool haveMappingData=false;
		while(getNextAlignment(alignment,bamReader,groupID,alignmentSample,wellIndex,nSample)) {
			int thisCol = 0;
			int thisRow = 0;
			if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
				std::cerr << "Error parsing read name: " << alignment.Name << "\n";
			std::string readGroup = "";
			alignment.GetTag("RG", readGroup);

			// Store values that will be returned
			out_id[nReadsFromBam]               = alignment.Name;
			out_readGroup[nReadsFromBam]        = readGroup;
			out_col(nReadsFromBam)              = thisCol;
			out_row(nReadsFromBam)              = thisRow;
			out_clipQualLeft(nReadsFromBam)     = 0;
			out_clipQualRight(nReadsFromBam)    = 0;

			std::map<std::string, int>::iterator keyLenIter;
			keyLenIter = keyLen.find(readGroup);
			out_clipAdapterLeft(nReadsFromBam) = (keyLenIter != keyLen.end()) ? keyLenIter->second : 0;

			int64_t clipAdapterRight = 0;
			getTagParanoid(alignment,"ZA",clipAdapterRight);
			out_clipAdapterRight(nReadsFromBam) = clipAdapterRight;

			int64_t flowClipLeft = 0;
			getTagParanoid(alignment,"ZF",flowClipLeft);
			out_flowClipLeft(nReadsFromBam) = flowClipLeft;

			int64_t flowClipRight = 0;
			getTagParanoid(alignment,"ZG",flowClipRight);
			out_flowClipRight(nReadsFromBam) = flowClipRight;

			std::vector<uint16_t> flowInt;
			if(alignment.GetTag("FZ", flowInt)){
				unsigned int i=0;
				for(; i<std::min(nFlowFZ,(unsigned int)flowInt.size()); i++)
					out_flow(nReadsFromBam,i) = flowInt[i] / 100.0;
				while(i<nFlowFZ)
					out_flow(nReadsFromBam,i++) = 0;
			}

			// experimental tag for Project Razor: "measured" values
			std::vector<int16_t> flowMeasured; // round(256*val), signed
			if(alignment.GetTag("ZM", flowMeasured)){
				unsigned int i=0;
				for(; i<std::min(nFlowZM,(unsigned int)flowMeasured.size()); i++)
					out_meas(nReadsFromBam,i) = flowMeasured[i]/256.0;
				while(i<nFlowZM)
					out_meas(nReadsFromBam,i++) = 0; // which is bad because will lead to biases in extrapolation
			} 

			// experimental tag for Project Razor: "phase" values
			std::vector<float> flowPhase;
			if(alignment.GetTag("ZP", flowPhase)){
				unsigned int i=0;
				for(; i<std::min(nPhase,(unsigned int)flowPhase.size()); i++)
					out_phase(nReadsFromBam,i) = flowPhase[i];
				while(i<nPhase)
					out_phase(nReadsFromBam,i++) = 0;
			}


			// limit scope of loop as we have too many "loop" variables named i running around
			if (true){
				unsigned int nBases = alignment.QueryBases.length();
				unsigned int trimLength = std::min(maxBases, nBases);
				out_fullLength(nReadsFromBam) = nBases;
				out_length(nReadsFromBam)     = trimLength;
				out_base[nReadsFromBam]       = alignment.QueryBases;
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
				out_aligned_base[nReadsFromBam]  = alignment.AlignedBases;
				out_aligned_refid(nReadsFromBam) = alignment.RefID;
				out_aligned_pos(nReadsFromBam)   = alignment.Position;
				out_aligned_mapq(nReadsFromBam)  = alignment.MapQuality;
				out_aligned_bin(nReadsFromBam)   = alignment.Bin;
				unsigned int cigarLength = std::min(maxCigarLength, (unsigned int) alignment.CigarData.size());
				out_aligned_cigar_type[nReadsFromBam].resize(cigarLength);
				unsigned int iCig=0;
				for(; iCig < cigarLength; iCig++) {
					out_aligned_cigar_type[nReadsFromBam][iCig] = alignment.CigarData[iCig].Type;
					out_aligned_cigar_len(nReadsFromBam,iCig)   = alignment.CigarData[iCig].Length;
				}
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
				out_qDNA[nReadsFromBam]  = pad_query;
				out_tDNA[nReadsFromBam]  = pad_target;
				out_match[nReadsFromBam] = pad_match;

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
		RcppResultSet rs;
		if(nReadsFromBam == 0) {
			std::cerr << "WARNING: No matching reads found in " << bamFile << "\n";
		} else if(nReadsFromBam != nReadOut) {
			// If we find fewer reads than expected then issue warning and trim back data structures
			std::cerr << "WARNING: Expected to find " << nReadOut << " reads but got " << nReadsFromBam << " in " << bamFile << "\n";
			if(filterByCoord)
				std::cerr << "Some of the requested reads are missing from the SFF.\n";
			std::vector< std::string > out2_id(nReadsFromBam);
			std::vector< std::string > out2_groupID(nReadsFromBam);
			RcppVector<int>    out2_col(nReadsFromBam);
			RcppVector<int>    out2_row(nReadsFromBam);
			RcppVector<int>    out2_length(nReadsFromBam);
			RcppVector<int>    out2_fullLength(nReadsFromBam);
			RcppVector<int>    out2_clipQualLeft(nReadsFromBam);
			RcppVector<int>    out2_clipQualRight(nReadsFromBam);
			RcppVector<int>    out2_clipAdapterLeft(nReadsFromBam);
			RcppVector<int>    out2_clipAdapterRight(nReadsFromBam);
			RcppVector<int>    out2_flowClipLeft(nReadsFromBam);
			RcppVector<int>    out2_flowClipRight(nReadsFromBam);
			RcppMatrix<double> out2_flow(nReadsFromBam,nFlowFZ);
			//razor
			RcppMatrix<double> out2_meas(nReadsFromBam,nFlowZM);
			RcppMatrix<double> out2_phase(nReadsFromBam,nPhase);
			// end
			std::vector< std::string >   out2_base(nReadsFromBam);
			RcppMatrix<int>    out2_qual(nReadsFromBam,maxBases);
			RcppMatrix<int>    out2_flowIndex(nReadsFromBam,maxBases);
			RcppVector<int>    out2_aligned_flag(nReadsFromBam);
			std::vector< std::string > out2_aligned_base(nReadsFromBam);
			RcppVector<int>    out2_aligned_refid(nReadsFromBam);
			RcppVector<int>    out2_aligned_pos(nReadsFromBam);
			RcppVector<int>    out2_aligned_mapq(nReadsFromBam);
			RcppVector<int>    out2_aligned_bin(nReadsFromBam);
			std::vector< std::string > out2_aligned_cigar_type(nReadsFromBam);
			RcppMatrix<double> out2_aligned_cigar_len(nReadsFromBam,maxCigarLength);
			std::vector< std::string > out2_qDNA(nReadsFromBam);
			std::vector< std::string > out2_match(nReadsFromBam);
			std::vector< std::string > out2_tDNA(nReadsFromBam);

			RcppVector<int>    out2_q7Len(nReadsFromBam);
			RcppVector<int>    out2_q10Len(nReadsFromBam);
			RcppVector<int>    out2_q17Len(nReadsFromBam);
			RcppVector<int>    out2_q20Len(nReadsFromBam);
			RcppVector<int>    out2_q47Len(nReadsFromBam);

			for(unsigned int i=0; i<nReadsFromBam; i++) {
				out2_id[i]               = out_id[i];
				out2_groupID[i]          = out_readGroup[i];
				out2_col(i)              = out_col(i);
				out2_row(i)              = out_row(i);
				out2_clipQualLeft(i)     = out_clipQualLeft(i);
				out2_clipQualRight(i)    = out_clipQualRight(i);
				out2_clipAdapterLeft(i)  = out_clipAdapterLeft(i);
				out2_clipAdapterRight(i) = out_clipAdapterRight(i);
				out2_flowClipLeft(i)     = out_flowClipLeft(i);
				out2_flowClipRight(i)    = out_flowClipRight(i);
				out2_length(i)           = out_length(i);
				out2_fullLength(i)       = out_fullLength(i);
				out2_base[i]             = out_base[i];
				for(unsigned int j=0; j<nFlowFZ; j++)
					out2_flow(i,j) = out_flow(i,j);
				for(unsigned int j=0; j<nFlowZM; j++)
					out2_meas(i,j) = out_meas(i,j);
				for(unsigned int j=0; j<3; j++)
					out2_phase(i,j) = out_phase(i,j);
				for(int j=0; j<out2_length(i); j++) {
					out2_qual(i,j)         = out_qual(i,j);
					out2_flowIndex(i,j)    = out_flowIndex(i,j);
				}
				if(haveMappingData) {
					out2_aligned_flag(i)       = out_aligned_flag(i);
					out2_aligned_base[i]       = out_aligned_base[i];
					out2_aligned_refid(i)      = out_aligned_refid(i);
					out2_aligned_pos(i)        = out_aligned_pos(i);
					out2_aligned_mapq(i)       = out_aligned_mapq(i);
					out2_aligned_bin(i)        = out_aligned_bin(i);
					out2_aligned_cigar_type[i] = out_aligned_cigar_type[i];
					for(unsigned int j=0; j<maxCigarLength; j++)
						out2_aligned_cigar_len(i,j) = out_aligned_cigar_len(i,j);

					out2_qDNA[i] = out_qDNA[i];
					out2_tDNA[i] = out_tDNA[i];
					out2_match[i] = out_match[i];
					out2_q7Len(i) = out_q7Len(i);
					out2_q10Len(i) = out_q10Len(i);
					out2_q17Len(i) = out_q17Len(i);
					out2_q20Len(i) = out_q20Len(i);
					out2_q47Len(i) = out_q47Len(i);
				}
			}
			rs.add("nFlow",            (int) std::max(nFlowFZ,nFlowZM));
			rs.add("id",               out2_id);
			rs.add("col",              out2_col);
			rs.add("row",              out2_row);
			rs.add("length",           out2_length);
			rs.add("fullLength",       out2_fullLength);
			rs.add("clipQualLeft",     out2_clipQualLeft);
			rs.add("clipQualRight",    out2_clipQualRight);
			rs.add("clipAdapterLeft",  out2_clipAdapterLeft);
			rs.add("clipAdapterRight", out2_clipAdapterRight);
			rs.add("flowClipLeft",     out2_flowClipLeft);
			rs.add("flowClipRight",    out2_flowClipRight);
			rs.add("flow",             out2_flow);
			rs.add("measured",         out2_meas);
			rs.add("phase",            out2_phase);
			rs.add("base",             out2_base);
			rs.add("qual",             out2_qual);
			if(haveMappingData) {
				rs.add("alignFlag",       out2_aligned_flag);
				rs.add("alignBase",       out2_aligned_base);
				rs.add("alignRefID",      out2_aligned_refid);
				rs.add("alignPos",        out2_aligned_pos);
				rs.add("alignMapq",       out2_aligned_mapq);
				rs.add("alignBin",        out2_aligned_bin);
				rs.add("alignCigarType",  out2_aligned_cigar_type);
				rs.add("alignCigarLen",   out2_aligned_cigar_len);
				rs.add("qDNA", out2_qDNA);
				rs.add("tDNA", out2_tDNA);
				rs.add("match", out2_match);
				rs.add("q7Len",out2_q7Len);
				rs.add("q10Len",out2_q10Len);
				rs.add("q17Len",out2_q17Len);
				rs.add("q20Len",out2_q20Len);
				rs.add("q47Len",out2_q47Len);
			}
		} else {
			rs.add("nFlow",            (int) std::max(nFlowFZ,nFlowZM));
			rs.add("id",               out_id);
			rs.add("readGroup",        out_readGroup);
			rs.add("col",              out_col);
			rs.add("row",              out_row);
			rs.add("length",           out_length);
			rs.add("fullLength",       out_fullLength);
			rs.add("clipQualLeft",     out_clipQualLeft);
			rs.add("clipQualRight",    out_clipQualRight);
			rs.add("clipAdapterLeft",  out_clipAdapterLeft);
			rs.add("clipAdapterRight", out_clipAdapterRight);
			rs.add("flowClipLeft",     out_flowClipLeft);
			rs.add("flowClipRight",    out_flowClipRight);
			rs.add("flow",             out_flow);
			rs.add("measured",         out_meas);
			rs.add("phase",            out_phase);
			rs.add("base",             out_base);
			rs.add("qual",             out_qual);
			rs.add("flowIndex",        out_flowIndex);
			if(haveMappingData) {
				rs.add("alignFlag",       out_aligned_flag);
				rs.add("alignBase",       out_aligned_base);
				rs.add("alignRefID",      out_aligned_refid);
				rs.add("alignPos",        out_aligned_pos);
				rs.add("alignMapq",       out_aligned_mapq);
				rs.add("alignBin",        out_aligned_bin);
				rs.add("alignCigarType",  out_aligned_cigar_type);
				rs.add("alignCigarLen",   out_aligned_cigar_len);
				rs.add("qDNA", out_qDNA);
				rs.add("tDNA", out_tDNA);
				rs.add("match", out_match);
				rs.add("q7Len",out_q7Len);
				rs.add("q10Len",out_q10Len);
				rs.add("q17Len",out_q17Len);
				rs.add("q20Len",out_q20Len);
				rs.add("q47Len",out_q47Len);
			}
		}
		ret = rs.getReturnList();

	} catch(std::exception& ex) {
		exceptionMesg = copyMessageToR(ex.what());
	} catch(...) {
		exceptionMesg = copyMessageToR("unknown reason");
	}

	if(exceptionMesg != NULL)
		Rf_error(exceptionMesg);

	return ret;
}

std::string getQuickStats(const std::string &bamFile, std::map< std::string, int > &keyLen, unsigned int &nFlowFZ, unsigned int &nFlowZM) {
	std::string errMsg = "";
	BamTools::BamReader bamReader;
	if(!bamReader.Open(bamFile)) {
		errMsg += "Failed to open bam " + bamFile + "\n";
		return(errMsg);
	}
	BamTools::SamHeader samHeader = bamReader.GetHeader();
	for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
		if(itr->HasID())
			keyLen[itr->ID] = itr->HasKeySequence() ? itr->KeySequence.length() : 0;
		if(itr->HasFlowOrder())
			nFlowZM = std::max(nFlowZM,(unsigned int) itr->FlowOrder.length());
	}
	BamTools::BamAlignment alignment;
	std::vector<uint16_t> flowIntFZ;
	while(bamReader.GetNextAlignment(alignment)) {
		if(alignment.GetTag("FZ", flowIntFZ))
			nFlowFZ = flowIntFZ.size();
		break;
	}
	bamReader.Close();
	if(nFlowFZ==0)
		std::cout << "NOTE: bam file has no flow signals in FZ tag: " + bamFile + "\n";
	if(nFlowZM==0)
		std::cout << "NOTE: bam file has no flow signals in ZM tag: " + bamFile + "\n";
	return(errMsg);
}

bool getNextAlignment(BamTools::BamAlignment &alignment, BamTools::BamReader &bamReader, const std::map<std::string, int> &groupID, std::vector< BamTools::BamAlignment > &alignmentSample, std::map<std::string, int> &wellIndex, unsigned int nSample) {
	if(nSample > 0) {
		// We are randomly sampling, so next read should come from the sample that was already taken from the bam file
		if(alignmentSample.size() > 0) {
			alignment = alignmentSample.back();
			alignmentSample.pop_back();
			alignment.BuildCharData();
			return(true);
		} else {
			return(false);
		}
	} else {
		// No random sampling, so we're either returning everything or we're looking for specific read names
		bool storeRead = false;
		while(bamReader.GetNextAlignment(alignment)) {
			if(groupID.size() > 0) {
				std::string thisReadGroupID = "";
				if( !alignment.GetTag("RG", thisReadGroupID) || (groupID.find(thisReadGroupID)==groupID.end()) );
					continue;
			}
			storeRead=true;
			if(wellIndex.size() > 0) {
				// We are filtering by position, so check if we should skip or keep the read
				int thisCol,thisRow;
				if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
					std::cerr << "Error parsing read name: " << alignment.Name << "\n";
				std::stringstream wellIdStream;
				wellIdStream << thisCol << ":" << thisRow;
				std::map<std::string, int>::iterator wellIndexIter;
				wellIndexIter = wellIndex.find(wellIdStream.str());
				if(wellIndexIter != wellIndex.end()) {
					// If the read ID matches we should keep, unless its a duplicate
					if(wellIndexIter->second >= 0) {
						storeRead=true;
						wellIndexIter->second=-1;
					} else {
						storeRead=false;
						std::cerr << "WARNING: found extra instance of readID " << wellIdStream.str() << ", keeping only first\n";
					}
				} else {
					// read ID is not one we should keep
					storeRead=false;
				}
			}
			if(storeRead)
				break;
		}
		return(storeRead);
	}
}

bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value) {
	char tagType = ' ';
	if(alignment.GetTagType(tag, tagType)) {
		switch(tagType) {
			case BamTools::Constants::BAM_TAG_TYPE_INT8: {
				int8_t value_int8 = 0;
				alignment.GetTag(tag, value_int8);
				value = value_int8;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT8: {
				uint8_t value_uint8 = 0;
				alignment.GetTag(tag, value_uint8);
				value = value_uint8;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_INT16: {
				int16_t value_int16 = 0;
				alignment.GetTag(tag, value_int16);
				value = value_int16;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT16: {
				uint16_t value_uint16 = 0;
				alignment.GetTag(tag, value_uint16);
				value = value_uint16;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_INT32: {
				int32_t value_int32 = 0;
				alignment.GetTag(tag, value_int32);
				value = value_int32;
			} break;
			case BamTools::Constants::BAM_TAG_TYPE_UINT32: {
				uint32_t value_uint32 = 0;
				alignment.GetTag(tag, value_uint32);
				value = value_uint32;
			} break;
			default: {
				alignment.GetTag(tag, value);
			} break;
		}
		return(true);
	} else {
		return(false);
	}
}



//Ported from BamUtils

//this could probably be faster -- maybe with an std::transform
void reverse_comp(std::string& c_dna) {
    for (unsigned int i = 0; i<c_dna.length(); i++) {
        switch (c_dna[i]) {
            case 'A':
                c_dna[i] = 'T';
                break;
            case 'T':
                c_dna[i] = 'A';
                break;
            case 'C':
                c_dna[i] = 'G';
                break;
            case 'G':
                c_dna[i] = 'C';
                break;
            case '-':
                c_dna[i] = '-';
                break;

            default:
                break;
        }
    }
    std::reverse(c_dna.begin(), c_dna.end());

}

void dna( string& qDNA, const vector<BamTools::CigarOp>& cig, const string& md, string& tDNA) {

    int position = 0;
    string seq;
    string::const_iterator qDNA_itr = qDNA.begin();

    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i != cig.end(); ++i) {
        if ( i->Type == 'M') {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {

                if (count >= i->Length) {
                    break;
                } else {
                    seq += *qDNA_itr;
                    ++qDNA_itr;
                    ++count;
                }
            }
        } else if ((i->Type == 'I') || (i->Type == 'S')) {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {
                if (count >= i->Length) {
                    break;
                }
                ++qDNA_itr;
                ++count;
            }
            //bool is_error = false;

//            if (i->Type == 'S') {
//                soft_clipped_bases += i->Length;
//                //is_error = true;
//            }
        }
        position++;
    }

    tDNA.reserve(seq.length());
    int start = 0;
    string::const_iterator md_itr = md.begin();
    std::string num;
    int md_len = 0;
    char cur;

    while (md_itr != md.end()) {

        cur = *md_itr;

        if (std::isdigit(cur)) {
            num+=cur;
            //md_itr.next();
        }
        else {
            if (num.length() > 0) {
                md_len = strtol(num.c_str(),NULL, 10);
                num.clear();

                tDNA += seq.substr(start, md_len);
                start += md_len;
            }
        }

        if (cur == '^') {
            //get nuc
            ++md_itr;
            char nuc = *md_itr;
            while (std::isalpha(nuc)) {
                tDNA += nuc;
                ++md_itr;
                nuc = *md_itr;
            }
            num += nuc; //it's a number now will
                        //lose this value if i don't do it here
            //cur = nuc;

        } else if (std::isalpha(cur)) {
            tDNA += cur;
            start++;

        }
        ++md_itr;
    }

    //clean up residual num if there is any
    if (num.length() > 0) {
        md_len = strtol(num.c_str(),NULL, 10);
        num.clear();
        tDNA += seq.substr(start, md_len);
        start += md_len;
    }
}


void padded_alignment(const vector<BamTools::CigarOp>& cig, string& qDNA, string& tDNA,  string& pad_query, string& pad_target, string& pad_match, bool isReversed) {

    int sdna_pos = 0;
    unsigned int tdna_pos = 0;
    pad_target.reserve(tDNA.length());
    pad_query.reserve(tDNA.length());
    pad_match.reserve(tDNA.length());
    string::iterator tdna_itr = tDNA.begin();
    unsigned int tot = 0;
    //find out if the first cigar op could be soft clipped or not
    bool is_three_prime_soft_clipped = false;


    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i!=cig.end(); ++i) {
        //i.op();		i.len();
        if (isReversed) {
            if (tot > ( cig.size() - 3) ){
                if (i->Type == 'S')
                    is_three_prime_soft_clipped = true;
                else
                    is_three_prime_soft_clipped = false;

            }
        } else {
            if (tot < 2) {
                if (i->Type == 'S')
                    is_three_prime_soft_clipped = true;
                else
                    is_three_prime_soft_clipped = false;

            }
        }

        if (i->Type == 'I' ) {
            pad_target.append(i->Length, '-');

            unsigned int count = 0;

            tdna_itr = qDNA.begin();
            advance(tdna_itr, sdna_pos);

            while (tdna_itr != tDNA.end() ) {
                if (count >= i->Length) {
                    break;
                } else {
                    pad_query += *tdna_itr;
                    ++tdna_itr;
                    //++tdna_pos;
                    ++sdna_pos;
                    ++count;
                }
            }
            pad_match.append(i->Length, '+');
        }
        else if(i->Type == 'D' || i->Type == 'N') {
            pad_target.append( tDNA.substr(tdna_pos, i->Length));
            sdna_pos += i->Length;
            tdna_pos += i->Length;
            pad_query.append(i->Length, '-');
            pad_match.append(i->Length, '-');
        }
        else if(i->Type == 'P') {
            pad_target.append(i->Length, '*');
            pad_query.append(i->Length, '*');
            pad_match.append(i->Length, ' ');
        } else if (i->Type == 'S') {

//            if (!truncate_soft_clipped) {

//                    pad_source.append(i->Length, '-');
//                    pad_match.append(i->Length, '+');
//                    pad_target.append(i->Length, '+');

//            }
//            int count = 0;
//            while (tdna_itr != tDNA.end()) {
//                if (count >= i->Length) {
//                    break;
//                }
//                ++tdna_pos;
//                ++tdna_itr;
//                ++count;
//            }
        }

        else if (i->Type == 'H') {
            //nothing for clipped bases
        }else {
            std::string ps, pt, pm;
            ps.reserve(i->Length);
            pm.reserve(i->Length);

            ps = qDNA.substr(sdna_pos,i->Length); //tdna is really qdna

            tdna_itr = tDNA.begin();
            advance(tdna_itr, tdna_pos);

            unsigned int count = 0;

            while (tdna_itr != tDNA.end()) {
                if (count < i->Length) {
                    pt += *tdna_itr;
                } else {
                    break;
                }

                ++tdna_itr;
                ++count;

            }
            for (unsigned int z = 0; z < ps.length(); z++) {
                if (ps[z] == pt[z]) {
                    pad_match += '|';
                } else {
                    pad_match += ' ';
                }
            }//end for loop
            pad_target += pt;
            pad_query += ps;

            sdna_pos += i->Length;
            tdna_pos += i->Length;
            if( tdna_pos >= tDNA.size() )
                break;
        }
        tot++;
    }
    /*
    std::cerr << "pad_source: " << pad_source << std::endl;
    std::cerr << "pad_target: " << pad_target << std::endl;
    std::cerr << "pad_match : " << pad_match << std::endl;
    */
}

std::vector<int> score_alignments(string& pad_source, string& pad_target, string& pad_match ){

    int n_qlen = 0;
    int t_len = 0;
    int t_diff = 0;
    int match_base = 0;
    int num_slop = 0;

    int consecutive_error = 0;

    //using namespace std;
    for (int i = 0; (unsigned int)i < pad_source.length(); i++) {
        //std::cerr << " i: " << i << " n_qlen: " << n_qlen << " t_len: " << t_len << " t_diff: " << t_diff << std::endl;
        if (pad_source[i] != '-') {
            t_len = t_len + 1;
        }

        if (pad_match[i] != '|') {
            t_diff = t_diff + 1;

            if (i > 0 && pad_match[i-1] != '|' && ( ( pad_target[i] == pad_target[i - 1] ) || pad_match[i] == '-' ) ) {
                consecutive_error = consecutive_error + 1;
            } else {
                consecutive_error = 1;
            }
        } else {
            consecutive_error = 0;
            match_base = match_base + 1;
        }
        if (pad_target[i] != '-') {
            n_qlen = n_qlen + 1;
        }
    }


    //get qual vals from  bam_record
    std::vector<double> Q;

    //setting acceptable error rates for each q score, defaults are
    //7,10,17,20,47
    //phred_val == 7
    Q.push_back(0.2);
    //phred_val == 10
    Q.push_back(0.1);
    //phred_val == 17
    Q.push_back(0.02);
    //phred_val == 20
    Q.push_back(0.01);
    //phred_val == 47
    Q.push_back(0.00002);

    std::vector<int> q_len_vec(Q.size(), 0);

    int prev_t_diff = 0;
    int prev_loc_len = 0;
    int i = pad_source.length() - 1;

    for (std::vector<std::string>::size_type k =0; k < Q.size(); k++) {
        int loc_len = n_qlen;
        int loc_err = t_diff;
        if (k > 0) {
            loc_len = prev_loc_len;
            loc_err = prev_t_diff;
        }

        while ((loc_len > 0) && (static_cast<int>(i) >= num_slop) && i > 0) {

            if (q_len_vec[k] == 0 && (((loc_err / static_cast<double>(loc_len))) <= Q[k]) /*&& (equivalent_length(loc_len) != 0)*/) {

                q_len_vec[k] = loc_len;

                prev_t_diff = loc_err;
                prev_loc_len = loc_len;
                break;
            }
            if (pad_match[i] != '|') {
                loc_err--;
            }
            if (pad_target[i] != '-') {

                loc_len--;
            }
            i--;
        }
    }
    return q_len_vec;
}
