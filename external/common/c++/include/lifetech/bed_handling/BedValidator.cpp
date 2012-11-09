/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */

#include "BedValidator.h"

LoggerPtr pLogger(Logger::getLogger("BedValidator"));

BedValidator::BedValidator(const string &bedFnIn) 
{
	mBedFn = bedFnIn;
	dumpmessage(DEBUG, __FILE__, " In BedValidator::BedValidator()");	
}


BedValidator::~BedValidator() 
{
        dumpmessage(DEBUG, __FILE__, " In BedValidator::~BedValidator()");
}


void BedValidator::setExpectedChromNames ( const vector<string> &bamChromNamesIn ) 
{
        mBamChromNames = bamChromNamesIn;
}


void BedValidator::setExpectedChromSizes (const vector<int64> &bamChromSizesIn ) 
{
        mBamChromSizes = bamChromSizesIn;
}


bool BedValidator::validate() 
{
	dumpmessage(DEBUG, __FILE__, " In BedValidator::validate()");

	bool isValid = true;	

	int colCount = -1;
	int trackCount = 1;

	string line, token;
	string traceChromName = "";
	string trackName = "track1";
	string dumpMsg;
			
	const string TRACK = "track";
	const char lineDelim = '\n';

	if(mBamChromNames.size() != mBamChromSizes.size()) 
	{
		dumpmessage(ERROR, __FILE__, " Validation failed. Size of input chrom names is not the same as the chrom sizes.");
		return false;
	}
	
	ifstream bedStream;
	// Open BED file for validation
	bedStream.open(mBedFn.c_str(), ifstream::in);
	if( !bedStream.is_open() )
	{
		dumpMsg = " BED file: " + mBedFn + ". BED file can not be opened";
		dumpmessage(ERROR, __FILE__, dumpMsg);
		return false;
	}
	
	dumpMsg = " BED file: " + mBedFn;
	dumpmessage(INFO, __FILE__, dumpMsg);
	
	while( getline(bedStream, line, lineDelim) )
	{		
		size_t pos = line.find(TRACK);
		if (pos < line.length())
		{	//Track is present 

			if(!getTrackName(trackName, line, trackCount))
			{
				dumpmessage(ERROR, __FILE__, " ERROR while reading track name.");
			}

			// Reset column count as new track starting
			colCount = -1;
			//Increment track count for every track
			++trackCount; 
		}
		else
		{	// Process BED file data
			bool dataLineFound = false;
				
			int colIdx = 1;
			int64 chromStart = -1, chromEnd = -1, score = -1;
			
			string chromName, strand;

			const char colDelim = '\t';
			
			// push the line into stringstream line for another round of getline
			stringstream ssLine(line); 
			for(; getline(ssLine, token, colDelim); ++colIdx)
			{
				stringstream ssToken(token);
				dataLineFound = true;
				
				switch(colIdx)
				{	// Validate Column data from BED file
					case CHROM_IDX:
						chromName = token;
						break;
							
					case CHROMSTART_IDX:
						ssToken >> chromStart;
						
						if(chromStart < 0 || ssToken.fail())
						{
							dumpMsg = " Track: " + trackName + " Chrom: " + chromName + " chromStart:"+ token + " Error: chromStart can not be less than 0";
							dumpmessage(ERROR, __FILE__, dumpMsg);
							isValid = false;
						}
						break;	
			
					case CHROMEND_IDX:
						ssToken >> chromEnd;

						if(chromEnd < 1 || ssToken.fail())
						{
							dumpMsg = " Track: " + trackName + " Chrom: " + chromName + " chromEnd:" + token + " Error: chromEnd can not be less than 1";
							dumpmessage(ERROR, __FILE__, dumpMsg);
							isValid = false;
						}

						if( chromStart > chromEnd)
						{
							stringstream ssValue;
							
							ssValue << " ChromStart: " << chromStart << ", ChromEnd: " << chromEnd;
							dumpMsg = " Track: " + trackName + " Chrom: " + chromName + ssValue.str() + " Error: chromStart can not be greater than chromEnd";
							dumpmessage(ERROR, __FILE__, dumpMsg);
							isValid = false;
						}
						
						break;
							
					case SCORE_IDX:
						ssToken >> score;

						if(score < SCORE_MIN || score > SCORE_MAX || ssToken.fail())
						{
							stringstream ssValue;
						
							ssValue << " ChromStart: " << chromStart;
							dumpMsg = " Track: " + trackName + " Chrom: " + chromName + ssValue.str() + " score:" + token + " Error: Score is not in 0 to 1000 limit";
							dumpmessage(ERROR, __FILE__, dumpMsg);
							isValid = false;
						}
						break;
							
					case STRAND_IDX:
						strand = token;
						if(token[0] != STRAND_PLUS && token[0] != STRAND_MINUS)
						{
							stringstream ssValue;
						
							ssValue << " ChromStart: " << chromStart;
							dumpMsg = " Track: " + trackName + " Chrom: " + chromName + ssValue.str() + " strand:" + token + " Error: Strand value is not + or -";
							dumpmessage(ERROR, __FILE__, dumpMsg);			
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
				
			if( colCount > 0 )
			{	// Verify the column count with the previous column count
				if( dataLineFound )
				{
					--colIdx;
						
					if(colCount < MIN_COLUMNS) 
					{ // Minimum three columns must be present	
						dumpMsg = " Track: " + trackName + " Chrom: " + chromName + " ERROR - Do not match the minimum three columns criteria";
						dumpmessage(ERROR, __FILE__, dumpMsg);
						isValid = false;
					}
					
					if( colCount != colIdx )
					{ // If the column count do not match		
						stringstream ssValue;
						
						ssValue << " ChromStart: " << chromStart;
						dumpMsg = " Track: " + trackName + " Chrom: " + chromName + ssValue.str() + " ERROR - Columns do not match from same track";						
						dumpmessage(ERROR, __FILE__, dumpMsg);
						isValid = false;
					}
				}
			}
			else
			{	// Initialize Column count for current Track
				stringstream ssColcount;

				colCount = --colIdx;
				ssColcount << colCount;
				dumpMsg = " Track: " + trackName + " - Column count: " + ssColcount.str();
				dumpmessage(INFO, __FILE__, dumpMsg);
			}
				
			// Validating Bam Header data if present otherwise return the validation of BED file
			if( dataLineFound && mBamChromNames.size() > 0)
			{
				bool retVal;

				retVal = checkWithBAMHeader(trackName, chromName, chromStart, chromEnd);

				if(!retVal)
					isValid = retVal;
			}
				
		} // else{} process BED data
	} // while()
	
	//Close the BED file
	bedStream.close();

	dumpmessage(DEBUG, __FILE__, " Out BedValidator::validate()");	
	return isValid;
}


bool BedValidator::checkWithBAMHeader(const string trackName, const string chromName, const int64 chromStart, const int64 chromEnd)
{
	bool isValid = false;
	string dumpMsg;

	//Get index of found chrom name
	vector<string>::iterator itFindChromName = find (mBamChromNames.begin(), mBamChromNames.end(), chromName);
	unsigned int iNameidx = itFindChromName - mBamChromNames.begin();

	if(iNameidx < mBamChromNames.size())
	{	// Check limit of chrom length
		if((chromStart < mBamChromSizes.at(iNameidx)) && (chromEnd <= mBamChromSizes.at(iNameidx)))				
			isValid = true;
		else
		{
			stringstream ssValue;
				
			dumpMsg = " Track: " + trackName + " Chrom: " + chromName + " ERROR - Not satisfied the condition: chromStart< lengthChrom and chromEnd<=lengthChrom";					
			dumpmessage(ERROR, __FILE__, dumpMsg);
			
			ssValue << " ChromStart: " << chromStart << ", ChromEnd: " << chromEnd << ", ChromLength: " << mBamChromSizes.at(iNameidx);
			dumpMsg = ssValue.str();
			dumpmessage(ERROR, __FILE__, dumpMsg);
				
			isValid = false;
		}
	}
	else
	{
		dumpMsg = " Track: " + trackName + " Chrom: " + chromName + " ERROR - Chrom name not found in BAM header";					
		dumpmessage(ERROR, __FILE__, dumpMsg);
	}

	return isValid;
}


bool BedValidator::getTrackName(string &trackName, const string trackLine, int &trackCount)
{ // Read BED file to get TrackName if track name is not present then Create Implicit name like Track1 Track2 ...
	char colDelim;
	
	/* 	track name are of two types as follows
	*	Track name=pairedReads and
	*	Track name="Human All Exon 50Mb targets"
	*	To read the name identify the name is with multi word or single word by '"' quote
	*/
	const char colDelimQuote = '"';
	const char colDelimSpace = ' ';

	const string TRACK_NAME = "name=";
	string token, dumpMsg;
	
	stringstream ssTrackName;

	int trackNameIndex = TRACK_NAME.length();
		
	size_t pos = trackLine.find(TRACK_NAME);
	if (pos < trackLine.length())
	{	// Read the name of the track from BED file
		if(trackLine[pos + trackNameIndex] == colDelimQuote)
		{
			colDelim = colDelimQuote;
			++trackNameIndex;
		}
		else
		{
			colDelim = colDelimSpace;
		}

		string subLine = trackLine.substr(pos + trackNameIndex, trackLine.length());
		
		ssTrackName << subLine;
		getline(ssTrackName, token, colDelim);

		trackName = token;

		dumpMsg = " Track name is " + token;	
		dumpmessage(INFO, __FILE__, dumpMsg);					
	}
	else
	{	// Track name is not present so use following IMPLICIT name
		ssTrackName << "Track" << trackCount;				
		trackName.assign(ssTrackName.str());
		
		dumpMsg = " Track name is " + ssTrackName.str();
		dumpmessage(INFO, __FILE__, dumpMsg);
	}

	return true;
}

bool readValidationBAMInput(const string bamHeaderFile, vector<string> &bamChromNamesIn, vector<int64> &bamChromSizesIn)
{
	string line, dumpMsg;
	
	char lineDelim = '\n';
	
	ifstream validationStream;
	// Open input data file
	validationStream.open(bamHeaderFile.c_str(), ifstream::in);
	if( !validationStream.is_open() )
	{
		dumpMsg = " BAM header file: " + bamHeaderFile + ". Validation data file can not be opened";
		dumpmessage(ERROR, __FILE__, dumpMsg);
		return false;
	}
	
	while( getline(validationStream, line, lineDelim) )
	{
		if(line != "")
		{
			stringstream ssVStream(line);
			string chromName;
			
			int64 lengthChrom;
			
			ssVStream >> chromName >> lengthChrom;
			
			if(ssVStream.fail())
			{
				dumpMsg = " BAM header file: " + bamHeaderFile + ". Please check BAM header input file for proper data";
				dumpmessage(ERROR, __FILE__, dumpMsg);
				return false;
			}
			
			bamChromNamesIn.push_back(chromName);
			bamChromSizesIn.push_back(lengthChrom);
		}
	}
	// Close file
	validationStream.close();
	return true;
}


int main(int argc, char *argv[])
{
	string bedFnIn;
	string bamHeaderFile;

	vector<string> bamChromNamesIn;
	vector<int64> bamChromSizesIn;

	const string logFileName = "BedValidator.log";
	
	bool retVal = false;
	
	if ( argc < 2)
	{
		dumpmessage(INFO, __FILE__, " Please provide BED file name and BAM validation data file name");
		return 1;
	}
	
	/* Configure Log4cxx conf file name */	
	ConfigureLog4cxx(logFileName.c_str());
	
	bedFnIn = argv[1];
	
	if(argc >= 3)
	{
		bamHeaderFile = argv[2];
		retVal = readValidationBAMInput(bamHeaderFile, bamChromNamesIn, bamChromSizesIn);
	}
	
	// Instantiate BedValidator object
	BedValidator bedV(bedFnIn);
	
	if(retVal)
	{ // Set expected data
		bedV.setExpectedChromNames (bamChromNamesIn);
		bedV.setExpectedChromSizes (bamChromSizesIn);
	}
	
	// Validate BED file
	retVal = bedV.validate();
	
	if(retVal)
		dumpmessage(INFO, __FILE__, " BED file validation result: SUCCESS");
	else
		dumpmessage(ERROR, __FILE__, " BED file validation result: FAIL");
	
	return 0;
}

//#endif
