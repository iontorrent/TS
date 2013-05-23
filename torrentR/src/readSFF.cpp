/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "../Analysis/file-io/sff_file.h"
#include "../Analysis/file-io/sff_definitions.h"
#include "../Analysis/file-io/sff.h"
#include "../Analysis/file-io/ion_util.h"
#include "ReservoirSample.h"
#include <iostream>
#include <sstream>

RcppExport SEXP readSFF(SEXP RsffFile, SEXP Rcol, SEXP Rrow, SEXP RmaxBases, SEXP RnSample, SEXP RrandomSeed) {

	SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
	char *exceptionMesg = NULL;

	try {

		char* sffFile         = (char *)Rcpp::as<const char*>(RsffFile);
		Rcpp::IntegerVector col(Rcol);
		Rcpp::IntegerVector row(Rrow);
		int maxBases          = Rcpp::as<int>(RmaxBases);
		int nSample           = Rcpp::as<int>(RnSample);
		int randomSeed        = Rcpp::as<int>(RrandomSeed);

		sff_file_t *sff_file = NULL;
		int thisCol,thisRow;

		// Check if we're sampling
		std::map<std::string, int> wellIndex;
		bool filterByCoord = false;
		bool filterById    = false;
		if(col.size() > 0 || row.size() > 0) {
			filterByCoord = true;
			// A subset of rows and columns is specified
			if(col.size() != row.size()) {
				exceptionMesg = strdup("col and row should have the same number of entries, ignoring\n");
			} else {
				for(int i=0; i<col.size(); i++) {
					std::stringstream wellIdStream;
					wellIdStream << col(i) << ":" << row(i);
					wellIndex[wellIdStream.str()] = i;
				}
			}
		} else if(nSample > 0) {
			filterById = true;
			// A random set of a given size is requested
			sff_file_t *sff_file = NULL;
			sff_file = sff_fopen(sffFile, "rb", NULL, NULL);
			sff_t *sff = NULL;
			ReservoirSample< std::string > sample;
			sample.Init(nSample,randomSeed);
			while(NULL != (sff =sff_read(sff_file))) {
				// Extract the row and column position
				if(1 != ion_readname_to_rowcol(sff_name(sff), &thisRow, &thisCol)) {
					fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(sff));
				}
				sample.Add(std::string(sff_name(sff)));
				sff_destroy(sff);
			}
			sff_fclose(sff_file);
			sample.Finished();
			// Then make a map from the sampled reads
			std::vector<std::string > sampleId = sample.GetData();
			for(unsigned int iSample=0; iSample < sampleId.size(); iSample++) {
				wellIndex[sampleId[iSample]] = iSample;
			}
		}

		sff_file = sff_fopen(sffFile, "rb", NULL, NULL);
		bool filter = (wellIndex.size() > 0);
		int nReadOut= (filter) ? (wellIndex.size()) : sff_file->header->n_reads;
		int nFlow = sff_file->header->flow_length;

		// Initialize things to return
		std::vector< std::string > out_id(nReadOut);
		Rcpp::IntegerVector    out_col(nReadOut);
		Rcpp::IntegerVector    out_row(nReadOut);
		Rcpp::IntegerVector    out_length(nReadOut);
		Rcpp::IntegerVector    out_fullLength(nReadOut);
		Rcpp::IntegerVector    out_clipQualLeft(nReadOut);
		Rcpp::IntegerVector    out_clipQualRight(nReadOut);
		Rcpp::IntegerVector    out_clipAdapterLeft(nReadOut);
		Rcpp::IntegerVector    out_clipAdapterRight(nReadOut);
		Rcpp::NumericMatrix out_flow(nReadOut,nFlow);
		std::vector< std::string > out_base(nReadOut);
		Rcpp::IntegerMatrix    out_qual(nReadOut,maxBases);
		Rcpp::IntegerMatrix    out_flowIndex(nReadOut,maxBases);

		int nReadsFromSFF=0;
		sff_t *sff = NULL;
		while(NULL != (sff =sff_read(sff_file))) {

			// Extract the row and column position
			if(1 != ion_readname_to_rowcol(sff_name(sff), &thisRow, &thisCol)) {
				fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(sff));
			}

			bool storeRead = true;
			if(filter) {
				storeRead=false;
				std::map<std::string, int>::iterator wellIndexIter;
				std::string id;
				if(filterByCoord) {
					std::stringstream wellIdStream;
					wellIdStream << thisCol << ":" << thisRow;
					id = wellIdStream.str();
				} else {
					id = std::string(sff_name(sff));
				}
				wellIndexIter = wellIndex.find(id);
				if(wellIndexIter != wellIndex.end()) {
					// If the read ID matches one we should keep, keep it unless its a duplicate
					if(wellIndexIter->second >= 0) {
						storeRead=true;
						wellIndexIter->second=-1;
					} else {
						std::cerr << "WARNING: found extra instance of readID " << id << ", keeping only first\n";
					}
				}
			}

			if(storeRead) {
                        	int nBases = sff_n_bases(sff);
				int trimLength = std::min(maxBases, nBases);

				// Store values that will be returned
				out_id[nReadsFromSFF]               = std::string(sff_name(sff));
				out_col(nReadsFromSFF)              = thisCol;
				out_row(nReadsFromSFF)              = thisRow;
				out_clipQualLeft(nReadsFromSFF)     = sff_clip_qual_left(sff);
				out_clipQualRight(nReadsFromSFF)    = sff_clip_qual_right(sff);
				out_clipAdapterLeft(nReadsFromSFF)  = sff_clip_adapter_left(sff);
				out_clipAdapterRight(nReadsFromSFF) = sff_clip_adapter_right(sff);
				int i;
				for(i=0; i<nFlow; i++)
					out_flow(nReadsFromSFF,i)         = sff_flowgram(sff)[i] / 100.0;
				out_fullLength(nReadsFromSFF)       = sff_n_bases(sff);
				out_length(nReadsFromSFF)           = trimLength;
				out_base[nReadsFromSFF].resize(maxBases);
				for(i=0; i<trimLength; i++) {
					out_base[nReadsFromSFF][i]        = sff_bases(sff)[i];
					out_qual(nReadsFromSFF,i)         = sff_quality(sff)[i];
					out_flowIndex(nReadsFromSFF,i)    = sff_flow_index(sff)[i];
				}
				// Pad out remaining bases with null entries
				for(; i<maxBases; i++) {
					out_base[nReadsFromSFF][i]        = 'N';
					out_qual(nReadsFromSFF,i)         = 0;
					out_flowIndex(nReadsFromSFF,i)    = 0;
				}
				nReadsFromSFF++;
			}
			sff_destroy(sff);
		}
		sff_fclose(sff_file);

		if(nReadsFromSFF != nReadOut) {
			// If we find fewer reads than expected then issue warning and trim back data structures
			fprintf(stderr,"Expected to find %d reads but got %d in %s\n",nReadOut,nReadsFromSFF,sffFile);
			if(filter)
				fprintf(stderr,"Some of the requested reads are missing from the SFF.\n");
			std::vector< std::string > out2_id(nReadsFromSFF);
			Rcpp::IntegerVector    out2_col(nReadsFromSFF);
			Rcpp::IntegerVector    out2_row(nReadsFromSFF);
			Rcpp::IntegerVector    out2_length(nReadsFromSFF);
			Rcpp::IntegerVector    out2_fullLength(nReadsFromSFF);
			Rcpp::IntegerVector    out2_clipQualLeft(nReadsFromSFF);
			Rcpp::IntegerVector    out2_clipQualRight(nReadsFromSFF);
			Rcpp::IntegerVector    out2_clipAdapterLeft(nReadsFromSFF);
			Rcpp::IntegerVector    out2_clipAdapterRight(nReadsFromSFF);
			Rcpp::NumericMatrix    out2_flow(nReadsFromSFF,nFlow);
			std::vector< std::string >   out2_base(nReadsFromSFF);
			Rcpp::IntegerMatrix    out2_qual(nReadsFromSFF,maxBases);
			Rcpp::IntegerMatrix    out2_flowIndex(nReadsFromSFF,maxBases);
			for(int i=0; i<nReadsFromSFF; i++) {
				out2_id[i]               = out_id[i];
				out2_col(i)              = out_col(i);
				out2_row(i)              = out_row(i);
				out2_clipQualLeft(i)     = out_clipQualLeft(i);
				out2_clipQualRight(i)    = out_clipQualRight(i);
				out2_clipAdapterLeft(i)  = out_clipAdapterLeft(i);
				out2_clipAdapterRight(i) = out_clipAdapterRight(i);
				out2_length(i)           = out_length(i);
				out2_fullLength(i)       = out_fullLength(i);
				out2_base[i]             = out_base[i];
				for(int j=0; j<nFlow; j++)
					out2_flow(i,j)         = out_flow(i,j);
				for(int j=0; j<out2_length(i); j++) {
					out2_qual(i,j)         = out_qual(i,j);
					out2_flowIndex(i,j)    = out_flowIndex(i,j);
				}
			}
            ret = Rcpp::List::create(Rcpp::Named("nFlow")           = nFlow,
                                     Rcpp::Named("id")              = out2_id,
                                     Rcpp::Named("col")             = out2_col,
                                     Rcpp::Named("row")             = out2_row,
                                     Rcpp::Named("length")           = out2_length,
                                     Rcpp::Named("fullLength")       = out2_fullLength,
                                     Rcpp::Named("clipQualLeft")     = out2_clipQualLeft,
                                     Rcpp::Named("clipQualRight")    = out2_clipQualRight,
                                     Rcpp::Named("clipAdapterLeft")  = out2_clipAdapterLeft,
                                     Rcpp::Named("clipAdapterRight") = out2_clipAdapterRight,
                                     Rcpp::Named("flow")             = out2_flow,
                                     Rcpp::Named("base")             = out2_base,
                                     Rcpp::Named("qual")             = out2_qual);
		} else {
            ret = Rcpp::List::create(Rcpp::Named("nFlow")            = nFlow,
                                     Rcpp::Named("id")               = out_id,
                                     Rcpp::Named("col")              = out_col,
                                     Rcpp::Named("row")              = out_row,
                                     Rcpp::Named("length")           = out_length,
                                     Rcpp::Named("fullLength")       = out_fullLength,
                                     Rcpp::Named("clipQualLeft")     = out_clipQualLeft,
                                     Rcpp::Named("clipQualRight")    = out_clipQualRight,
                                     Rcpp::Named("clipAdapterLeft")  = out_clipAdapterLeft,
                                     Rcpp::Named("clipAdapterRight") = out_clipAdapterRight,
                                     Rcpp::Named("flow")             = out_flow,
                                     Rcpp::Named("base")             = out_base,
                                     Rcpp::Named("qual")             = out_qual,
                                     Rcpp::Named("flowIndex")        = out_flowIndex);
		}
	} catch(std::exception& ex) {
		forward_exception_to_r(ex);
	} catch(...) {
		::Rf_error("c++ exception (unknown reason)");
	}

	if(exceptionMesg != NULL)
		Rf_error(exceptionMesg);

	return ret;
}
