/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */

#ifndef BEDVALIDATOR_H_
#define BEDVALIDATOR_H_

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

using namespace std;

/* Column constants  */
#define  CHROM_IDX 		1
#define  CHROMSTART_IDX 	2
#define  CHROMEND_IDX 		3
#define  NAME_IDX 		4
#define  SCORE_IDX 		5
#define  STRAND_IDX 		6
#define  THICKSTART_IDX 	7
#define  THICKEND_IDX 		8
#define  ITEMRGB_IDX 		9
#define  BLOCKCOUNT_IDX 	10
#define  BLOCKSIZES_IDX 	11
#define  BLOCKSTARTS_IDX 	12

/* Constants used in code */
#define SCORE_MIN 		0
#define SCORE_MAX		1000
#define MIN_COLUMNS 		3

#define int64 int64_t

const char STRAND_PLUS		= '+';
const char STRAND_MINUS 	= '-';

namespace lifetechnologies {

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.BedValidator");

class BedValidator
{

public:

	BedValidator( const string &bedFnIn ) { mBedFn = bedFnIn; }
	~BedValidator() {}

	void setExpectedChromDataSource( const BamHeader &bamHeader, const string &bamFile = "" ) {
	    vBamHeaders.clear();
	    vBamFiles.clear();
		vBamHeaders.push_back( bamHeader );
		vBamFiles.push_back( bamFile );
	}

	void setExpectedChromDataSource( const vector<BamHeader> &vecBamHeaders, const vector<string> &vecBamFiles = vector<string>() ) {
	    vBamHeaders = vecBamHeaders;
		if ( vecBamFiles.size() == vecBamHeaders.size() )
			vBamFiles = vector<string>( vecBamFiles );
		else
			vBamFiles = vector<string>( vecBamHeaders.size(), "" );
	}

	void setExpectedChromDataSource( AlignReader &alignReader ) {
	    // combine the BAM file-names into one string
		string locBamFile = alignReader.getBamReader(0).getFilename();
	    for ( size_t bamId = 1; bamId < alignReader.getNumBamReaders(); bamId++ )
	    	locBamFile += "," + alignReader.getBamReader(bamId).getFilename();
	    vBamHeaders.clear();
	    vBamFiles.clear();
		vBamHeaders.push_back( alignReader.getHeader() );
		vBamFiles.push_back( locBamFile );
	}
	
	bool validate() {
		vector<BamHeader>::const_iterator headerIter = vBamHeaders.begin();
		vector<string>::const_iterator bamFileIter = vBamFiles.begin();
		bool allValid = true;
		for ( ; headerIter != vBamHeaders.end(); ++headerIter, ++bamFileIter ) {
			mBamHeader = *headerIter;
			mBamFile   = *bamFileIter;
			bool isValid = validateOne();
			allValid = allValid && isValid;
		}
		return allValid;
	}

private:

	bool validateOne()
	{
		bool isValid = true;
		int colCount   = -1;
		int trackCount =  1;

		string line, token;
		string traceChromName = "";
		string trackName = "track1";

		const string TRACK = "track";
		const char lineDelim = '\n';

		string bedBamFiles = "BED File: " + mBedFn;
		bedBamFiles += (mBamFile.size() != 0) ? ("; BAM File(s): " + mBamFile) : "";

		ifstream bedStream;
		// Open BED file for validation
		bedStream.open(mBedFn.c_str(), ifstream::in);
		if( !bedStream.is_open() ) {
			string msg = "BED file cannot be opened.";
			msg += " " + bedBamFiles;
			cerr << "ERROR: " << msg << endl;
			LOG4CXX_ERROR(g_log, msg);
			return false;
		}

		while( getline(bedStream, line, lineDelim) )
		{
			size_t pos = line.find(TRACK);
			if( pos < line.length() )
			{	// Track is present
				if( !getTrackName(trackName, line, trackCount) ) {
					string msg = "ERROR while reading track name.";
					msg += " " + bedBamFiles;
					cerr << msg << endl;
					LOG4CXX_ERROR(g_log, msg);
				}
				// Reset column count because new track is starting
				colCount = -1;
				++trackCount; // Increment track count for every track
				// but what if the first record in the file is a track? then trackCount set to 2...?
			}
			else
			{	// process line containing BED data
				bool dataLineFound = false;
				int colIdx = 1;
				int64 chromStart = -1, chromEnd = -1, score = -1;
				string chromName, strand;
				const char colDelim = '\t';

				std::stringstream ssLine(line); // push the line into stringstream line for another round of getline
				for( ; getline(ssLine, token, colDelim); ++colIdx )
				{
					dataLineFound = true;
					switch(colIdx)
					{	// Validate Column data from BED file
						case CHROM_IDX:
							chromName = token;
							break;

						case CHROMSTART_IDX:
							chromStart = atol( token.c_str() );
							if( chromStart < 0 ) {
								string msg = "chromStart cannot be less than 0. Track: " + trackName + "; chromName: " + chromName + "; chromStart: " + token;
								msg += "; " + bedBamFiles;
								cerr << "ERROR: " << msg << endl;
								LOG4CXX_ERROR(g_log, msg);
								isValid = false;
							}
							break;

						case CHROMEND_IDX:
							chromEnd = atol( token.c_str() );
							if( chromEnd < 1 ) {
								string msg =  "chromEnd cannot be less than 1. Track: " + trackName + "; chromName: " + chromName + "; chromEnd: " + token;
								msg += "; " + bedBamFiles;
								cerr << "ERROR: " << msg << endl;
								LOG4CXX_ERROR(g_log, msg);
								isValid = false;
							}
							if( chromStart > chromEnd ) {
								string msg =  "chromStart cannot be greater than chromEnd. Track: " + trackName + "; chromName: " + chromName;
								stringstream chromInfo;
								chromInfo << "; chromStart: " << chromStart << "; chromEnd: " << chromEnd;
								msg += chromInfo.str();
								msg += "; " + bedBamFiles;
								cerr << "ERROR: " << msg << endl;
								LOG4CXX_ERROR(g_log, msg);
								isValid = false;
							}
							break;

						case SCORE_IDX:
							score = atol( token.c_str() );
							if( score < SCORE_MIN || score > SCORE_MAX ) {
								string msg =  "score must be between 0 and 1000. Track: " + trackName + "; chromName: " + chromName + "; score: " + token;
								msg += "; " + bedBamFiles;
								cerr << "ERROR: " << msg << endl;
								LOG4CXX_ERROR(g_log, msg);
								isValid = false;
							}
							break;

						case STRAND_IDX:
							strand = token;
							if( token[0] != STRAND_PLUS && token[0] != STRAND_MINUS ) {
								string msg =  "strand must begin with a + or - sign. Track: " + trackName + "; chromName: " + chromName + "; strand: " + token;
								msg += "; " + bedBamFiles;
								cerr << "ERROR: " << msg << endl;
								LOG4CXX_ERROR(g_log, msg);
								isValid = false;
							}
							break;

						case NAME_IDX:
						case THICKSTART_IDX:
						case THICKEND_IDX:
						case ITEMRGB_IDX:
						case BLOCKCOUNT_IDX:
						case BLOCKSIZES_IDX:
						case BLOCKSTARTS_IDX:
						default:
							break;
					} // switch()
				} // for()

				if( colCount > 0 )	// this is NOT the first BED record for this track
				{	// Verify the column count with the previous column count
					if( dataLineFound )
					{
						--colIdx;
						if( colCount < MIN_COLUMNS ) // Minimum three columns must be present
						{
							string msg =  "BED data contains fewer than three columns. Track: " + trackName + "; chromName: " + chromName;
							msg += "; " + bedBamFiles;
							cerr << "ERROR: " << msg << endl;
							LOG4CXX_ERROR(g_log, msg);
							isValid = false;
						}
						if( colCount != colIdx ) { // inconsistent column counts?
							string msg =  "inconsistent number of columns in same track. Track: " + trackName + "; chromName: " + chromName;
							msg += "; " + bedBamFiles;
							cerr << "ERROR: " << msg << endl;
							LOG4CXX_ERROR(g_log, msg);
							isValid = false;
						}
					}
				}
				else	// this is the first BED record for this track
				{	// Initialize Column count for current Track
					stringstream szColcount;
					colCount = --colIdx;
					szColcount << colCount;
					string msg = "Track: " + trackName + "; Column count: " + szColcount.str();
					msg += "; " + bedBamFiles;
					cout << msg << endl;
					LOG4CXX_INFO(g_log, msg);
				}
				// Validating Bam Header data
				if( dataLineFound )
				{
					bool retVal = checkWithBamHeader(trackName, chromName, chromStart, chromEnd);
					if( !retVal )
						isValid = retVal;
				}
			}
		} // while()

		//Close the BED file
		bedStream.close();

		return isValid;
	}

	bool checkWithBamHeader( const string trackName, const string chromName, const int64 chromStart, const int64 chromEnd )
	{
		string bedBamFiles = "BED File: " + mBedFn;
		bedBamFiles += (mBamFile.size() != 0) ? ("; BAM File(s): " + mBamFile) : "";

		bool isValid = mBamHeader.isValidSequenceInterval( chromName, chromStart+1, chromEnd );

		if( !isValid ) {
			stringstream szValues;
			string msg =  "invalid sequence interval. Track: " + trackName + "; chromName: " + chromName;
			szValues << "; chromStart: " << chromStart << "; chromEnd: " << chromEnd << "; chromLength: " << mBamHeader.getSequenceLength(chromName.c_str());
			msg += szValues.str();
			msg += "; " + bedBamFiles;
			cerr << "ERROR: " << msg << endl;
			LOG4CXX_ERROR(g_log, msg);
		}
		return isValid;
	}

	bool getTrackName( string &trackName, const string trackLine, int &trackCount )
	{ // Read BED file to get TrackName if track name is not present then Create Implicit name like Track1 Track2 ...
		char colDelim;

		/* 	track name is one of two types as follows
		*	Track name=pairedReads and
		*	Track name="Human All Exon 50Mb targets"
		*	To read the name identify the name is with multi word or single word by '"' quote
		*/
		const char colDelimQuote = '"';
		const char colDelimSpace = ' ';

		const string TRACK_NAME = "name=";
		string token;

		int trackNameIndex = TRACK_NAME.length();

		size_t pos = trackLine.find(TRACK_NAME);
		if (pos < trackLine.length())
		{	// Read the name of the track from BED file
			if(trackLine[pos + trackNameIndex] == colDelimQuote) {
				colDelim = colDelimQuote;
				++trackNameIndex;
			} else
				colDelim = colDelimSpace;

			string subLine = trackLine.substr(pos + trackNameIndex, trackLine.length());
			std::stringstream ssLine(subLine);
			getline(ssLine, token, colDelim);

			trackName = token;

			string msg = "Track name is " + token;
			msg += "; BED File: " + mBedFn;
			LOG4CXX_INFO(g_log, msg);
		} else {	// Track name is not present so use following IMPLICIT name
			stringstream tempTrackName;

			tempTrackName << "Track" << trackCount;
			trackName.assign(tempTrackName.str());

			string msg = "Track name is " + tempTrackName.str();
			msg += "; BED File: " + mBedFn;
			LOG4CXX_INFO(g_log, msg);
		}

		return true;
	}

	/*Input Bed file name*/
	string mBedFn;

	/* BAM file header and name */
	BamHeader mBamHeader;
	string mBamFile;

	/* vectors of BAM file headers and names */
	vector<BamHeader> vBamHeaders;
	vector<string>    vBamFiles;
};

}
#endif
