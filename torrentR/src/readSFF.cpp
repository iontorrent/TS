/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <Rcpp.h>
#include "../Analysis/file-io/sff_file.h"
#include "../Analysis/file-io/sff_definitions.h"
#include "../Analysis/file-io/sff.h"
#include "../Analysis/file-io/ion_util.h"
#include "ReservoirSample.h"
#include <sstream>

RcppExport SEXP readSFF(SEXP RsffFile, SEXP Rcol, SEXP Rrow, SEXP RmaxBases, SEXP RnSample, SEXP RrandomSeed) {

	SEXP ret = R_NilValue; 		// Use this when there is nothing to be returned.
	char *exceptionMesg = NULL;

	try {

		char* sffFile         = (char *)Rcpp::as<const char*>(RsffFile);
		RcppVector<int> col(Rcol);
		RcppVector<int> row(Rrow);
		int maxBases          = Rcpp::as<int>(RmaxBases);
		int nSample           = Rcpp::as<int>(RnSample);
		int randomSeed        = Rcpp::as<int>(RrandomSeed);

		sff_file_t *sff_file = NULL;
		int thisCol,thisRow;

		// Check if we're sampling
		std::map<std::string, unsigned int> wellIndex;
		if(col.size() > 0 || row.size() > 0) {
			// A subset of rows and columns is specified
			if(col.size() != row.size()) {
				exceptionMesg = copyMessageToR("col and row should have the same number of entries, ignoring\n");
			} else {
				for(int i=0; i<col.size(); i++) {
					std::stringstream wellIdStream;
					wellIdStream << col(i) << ":" << row(i);
					wellIndex[wellIdStream.str()] = i;
				}
			}
		} else if(nSample > 0) {
			// A random set of a given size is requested
			sff_file_t *sff_file = NULL;
			sff_file = sff_fopen(sffFile, "rb", NULL, NULL);
			sff_t *sff = NULL;
			ReservoirSample<std::pair<int,int> > sample;
			sample.Init(nSample,randomSeed);
			while(NULL != (sff =sff_read(sff_file))) {
				// Extract the row and column position
				if(1 != ion_readname_to_rowcol(sff_name(sff), &thisRow, &thisCol)) {
					fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(sff));
				}
				sample.Add(std::pair<int,int>(thisCol,thisRow));
				sff_destroy(sff);
			}
			sff_fclose(sff_file);
			sample.Finished();
			// Then make a map from the sampled reads
			std::vector<std::pair<int,int> > sampleCoords = sample.GetData();
			for(unsigned int iSample=0; iSample < sampleCoords.size(); iSample++) {
				std::stringstream wellIdStream;
				wellIdStream << sampleCoords[iSample].first << ":" << sampleCoords[iSample].second;
				wellIndex[wellIdStream.str()] = iSample;
			}
		}

		sff_file = sff_fopen(sffFile, "rb", NULL, NULL);
		bool filter = (wellIndex.size() > 0);
		int nReadOut= (filter) ? (wellIndex.size()) : sff_file->header->n_reads;
		int nFlow = sff_file->header->flow_length;

		// Initialize things to return
		RcppVector<int>    out_col(nReadOut);
		RcppVector<int>    out_row(nReadOut);
		RcppVector<int>    out_length(nReadOut);
		RcppVector<int>    out_fullLength(nReadOut);
		RcppVector<int>    out_clipQualLeft(nReadOut);
		RcppVector<int>    out_clipQualRight(nReadOut);
		RcppVector<int>    out_clipAdapterLeft(nReadOut);
		RcppVector<int>    out_clipAdapterRight(nReadOut);
		RcppMatrix<double> out_flow(nReadOut,nFlow);
		std::vector< std::string > out_base(nReadOut);
		RcppMatrix<int>    out_qual(nReadOut,maxBases);
		RcppMatrix<int>    out_flowIndex(nReadOut,maxBases);

		int nReadsFromSFF=0;
		sff_t *sff = NULL;
		while(NULL != (sff =sff_read(sff_file))) {

			// Extract the row and column position
			if(1 != ion_readname_to_rowcol(sff_name(sff), &thisRow, &thisCol)) {
				fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(sff));
			}

			bool storeRead = true;
			if(filter) {
				std::map<std::string, unsigned int>::iterator wellIndexIter;
				std::stringstream wellIdStream;
				wellIdStream << thisCol << ":" << thisRow;
				wellIndexIter = wellIndex.find(wellIdStream.str());
				if(wellIndexIter == wellIndex.end())
					storeRead=false;
			}

			if(storeRead) {
                        	int nBases = sff_n_bases(sff);
				int trimLength = std::min(maxBases, nBases);

				// Store values that will be returned
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

		RcppResultSet rs;
		if(nReadsFromSFF != nReadOut) {
			// If we find fewer reads than expected then issue warning and trim back data structures
			fprintf(stderr,"Expected to find %d reads but got %d in %s\n",nReadOut,nReadsFromSFF,sffFile);
			if(filter)
				fprintf(stderr,"Some of the requested reads are missing from the SFF.\n");
			RcppVector<int>    out2_col(nReadsFromSFF);
			RcppVector<int>    out2_row(nReadsFromSFF);
			RcppVector<int>    out2_length(nReadsFromSFF);
			RcppVector<int>    out2_fullLength(nReadsFromSFF);
			RcppVector<int>    out2_clipQualLeft(nReadsFromSFF);
			RcppVector<int>    out2_clipQualRight(nReadsFromSFF);
			RcppVector<int>    out2_clipAdapterLeft(nReadsFromSFF);
			RcppVector<int>    out2_clipAdapterRight(nReadsFromSFF);
			RcppMatrix<double> out2_flow(nReadsFromSFF,nFlow);
			std::vector< std::string >   out2_base(nReadsFromSFF);
			RcppMatrix<int>    out2_qual(nReadsFromSFF,maxBases);
			RcppMatrix<int>    out2_flowIndex(nReadsFromSFF,maxBases);
			for(int i=0; i<nReadsFromSFF; i++) {
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
			rs.add("nFlow",            nFlow);
			rs.add("col",              out2_col);
			rs.add("row",              out2_row);
			rs.add("length",           out2_length);
			rs.add("fullLength",       out2_fullLength);
			rs.add("clipQualLeft",     out2_clipQualLeft);
			rs.add("clipQualRight",    out2_clipQualRight);
			rs.add("clipAdapterLeft",  out2_clipAdapterLeft);
			rs.add("clipAdapterRight", out2_clipAdapterRight);
			rs.add("flow",             out2_flow);
			rs.add("base",             out2_base);
			rs.add("qual",             out2_qual);
		} else {
			rs.add("nFlow",            nFlow);
			rs.add("col",              out_col);
			rs.add("row",              out_row);
			rs.add("length",           out_length);
			rs.add("fullLength",       out_fullLength);
			rs.add("clipQualLeft",     out_clipQualLeft);
			rs.add("clipQualRight",    out_clipQualRight);
			rs.add("clipAdapterLeft",  out_clipAdapterLeft);
			rs.add("clipAdapterRight", out_clipAdapterRight);
			rs.add("flow",             out_flow);
			rs.add("base",             out_base);
			rs.add("qual",             out_qual);
			rs.add("flowIndex",        out_flowIndex);
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
