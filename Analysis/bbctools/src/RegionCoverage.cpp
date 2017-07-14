// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * RegionCoverage.cpp
 *
 *  Created on: Aug 27, 2015
 *      Author: Guy Del Mistro
 */

#include "RegionCoverage.h"

#include "BbcUtils.h"
using namespace BbcUtils;

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>

RegionCoverage::RegionCoverage( const BamTools::RefVector& references )
	: m_contigList(NULL)
	, m_bcovRegion(NULL)
	, m_rcovRegion(NULL)
    , m_lastRegionAssigned(NULL)
	, m_numAuxFields(0)
	, m_ncovDepths(0)
{
	m_numRefContigs = references.size();
	if( m_numRefContigs ) {
		m_contigList = new TargetContig*[m_numRefContigs];
		for( size_t i = 0; i < m_numRefContigs; ++i ) {
			m_contigList[i] = new TargetContig( references[i].RefName, references[i].RefLength );
			m_contigIdx[ references[i].RefName ] = i;
		}
	}
	// initial values force set up on first call to 'iterator'
	m_bcovContigIdx = m_rcovContigIdx = m_numRefContigs;
	m_bcovRegionPos = 0;
}

RegionCoverage::~RegionCoverage()
{
	Clear();
}

//
// ---- Public Non-virtual Methods ---
//

void RegionCoverage::Clear()
{
	m_headerLine.empty();
	m_contigIdx.clear();
	if( m_contigList ) {
		for( size_t i = 0; i < m_numRefContigs; ++i ) {
			delete m_contigList[i];
		}
		delete [] m_contigList;
		m_contigList = NULL;
	}
	m_bcovContigIdx = m_rcovContigIdx = m_numRefContigs;
	m_lastRegionAssigned = NULL;
}

uint32_t *RegionCoverage::CreateTargetBinSizes( uint32_t &numBins, uint32_t binSize,
	uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// since this method is public default options are handled
	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_contigList[endContig]->length;

	// determine the number of regions covered by window
	if( !SetCursorOnRegion( srtContig, srtPosition ) ) return NULL;
	uint32_t numRegions = 0;
	for( TargetRegion *cur = m_bcovRegion; cur; cur = cur->next ) {
		if( srtContig == endContig && cur->trgSrt > endPosition ) break;
		++numRegions;
	}
	for( uint32_t contig = srtContig+1; contig <= endContig; ++contig ) {
		for( TargetRegion *cur = m_contigList[contig]->targetRegionHead; cur; cur = cur->next ) {
			if( contig == endContig && cur->trgSrt > endPosition ) break;
			++numRegions;
		}
	}
	// make local copy of contig sizes required, noting partial srt/end contig lengths
	uint32_t *csizes = new uint32_t[numRegions];
	uint32_t lstContig = 0, srtPos = 0, endPos = 0;
	bool firstContig = true;
	for( size_t n = 0; n < numRegions; ++n ) {
		csizes[n] = PullSubRegion( 0, lstContig, srtPos, endPos, firstContig );
	}
	// adjust last region size (if clipped)
	if( lstContig == endContig && endPos > endPosition ) {
		csizes[numRegions-1] -= endPos - endPosition;
	}
	uint32_t *cnbins = new uint32_t[numRegions];
	if( binSize > 0 ) {
		// get number of binSizes in each contig
		numBins = 0;
		for( size_t n = 0, c = 0; n < numRegions; ++n ) {
			uint32_t nbins = csizes[n] / binSize;
			if( csizes[c] % binSize ) ++nbins;
			cnbins[c++] = nbins;
			numBins += nbins;
		}
		// spread coverage in equal chunks, except last for each contig
		uint32_t *binWidths = new uint32_t[numBins];
		for( size_t n = 0, c = 0, i = 0; n < numRegions; ++n, ++c ) {
			for( uint32_t n = 0; n < cnbins[c]-1; ++n ) {
				binWidths[i++] = binSize;
			}
			uint32_t lastBinSize = csizes[c] % binSize;
			if( lastBinSize == 0 ) lastBinSize = binSize;
			binWidths[i++] = lastBinSize;
		}
		delete [] csizes;
		delete [] cnbins;
		return binWidths;
	}
	// a copy of the effective contig sizes is required for reference to starting sizes but as a double
	double *esizes = new double[numRegions];
	for( size_t i = 0; i < numRegions; ++i ) {
		esizes[i] = (double)csizes[i];
	}
	// Every contig will be represented by N >= 1 bins containing a roughly even amount of base coverage, with
	// the number of bins for each contig distributed as proportionately as possible according to the relative
	// sizes of the contigs. The calling algorithm is expected to know starting position of first contig and to
	// tally sizes for each bin to track contig changes, so that return need only be a list of contiguous widths.
	if( numBins < numRegions ) numBins = numRegions;
	for( size_t i = 0; i < numRegions; ++i ) {
		cnbins[i] = 1;
	}
	uint32_t xbins = numBins - numRegions;				// number of bins left to distribute
	while( xbins ) {
		// get largest effective contig size remaining and give that an extra bin
		uint32_t m = 0;
		for( uint32_t i = 1; i < numRegions; ++i ) {
			if( esizes[i] > esizes[m] ) m = i;
		}
		// incr #bins for the largest contig and reduce its effective size as if spread over N bins
		++cnbins[m];
		esizes[m] = (double)csizes[m] / cnbins[m];
		--xbins;
 	}
	// spread the final contig bins into linear array of uniformly spread bin widths
	uint32_t *binWidths = new uint32_t[numBins];
	for( size_t i = 0, bin = 0; i < numRegions; ++i ) {
		uint64_t csz = csizes[i], binSrt = 0;
		uint32_t nBins = cnbins[i];
		for( size_t b = 1; b <= nBins; ++b ) {
			int64_t binEnd = (b * csz) / nBins;
			binWidths[bin++] = binEnd - binSrt;
			binSrt = binEnd;
		}
	}
	delete [] csizes;
	delete [] esizes;
	delete [] cnbins;
	return binWidths;
}

uint32_t RegionCoverage::GetTargetedWindowSize(
	uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// reads to next covered base after or at srtPosition also checks srtContig is within bounds
	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_contigList[endContig]->length;

	// add lengths for first (partial) regions, middle, and end (partial) contig regions covered
	uint32_t length = 0;
	for( uint32_t contig = srtContig; contig <= endContig; ++contig ) {
		// again have to track end of last region to avoid counting twice
		uint32_t srtPos = contig > srtContig ? 1 : srtPosition;
		uint32_t endPos = contig < endContig ? m_contigList[contig]->length : endPosition;
		for( TargetRegion *cur = m_contigList[contig]->targetRegionHead; cur; cur = cur->next ) {
			if( cur->trgEnd < srtPos ) continue;
			if( cur->trgSrt > endPos ) break;
			uint32_t csrt = cur->trgSrt > srtPos ? cur->trgSrt : srtPos;
			uint32_t cend = cur->trgEnd > endPos ? endPos : cur->trgEnd;
			length += cend - csrt + 1;
			// for multiple overlaps do not count same region coverage more than once!
			srtPos = cend+1;
		}
	}
	return length;
}

string RegionCoverage::FieldsOnRegion( uint32_t contigIdx, uint32_t readSrt, uint32_t readEnd, uint32_t maxValues )
{
	if( !m_numAuxFields || !maxValues || !ReadOnRegion( contigIdx, readSrt, readEnd ) ) {
		string fieldVals;
		for( int i = 0; i < (int)m_numAuxFields-1; ++i ) {
			fieldVals += "\t";
		}
		return fieldVals;
	}
	uint32_t numVals = 0;
	string last_tsv;
	vector<string> all;
	for( TargetRegion *cur = m_rcovRegion; cur; cur = cur->next ) {
		if( readEnd < cur->trgSrt ) break;
		last_tsv = cur->auxData[0];
		for( size_t i = 1; i < m_numAuxFields; ++i ) {
			last_tsv += "\t" + cur->auxData[i];
		}
		if( ++numVals < maxValues ) {
			all.push_back( last_tsv );
		}
	}
	// invert the fields
	string fieldVals;
	if( numVals > maxValues ) {
		const string sep = ",...(" + BbcUtils::integerToString(numVals-2) + ")...,";
		vector<string> v1 = stringTrimSplit( all[0], '\t' );
		vector<string> v2 = stringTrimSplit( last_tsv, '\t' );
		for( size_t i = 0; i < m_numAuxFields; ++i ) {
			if( i ) fieldVals += "\t";
			fieldVals += v1[i] + sep + v2[i];
		}
	} else {
		all.push_back( last_tsv );
		for( size_t i = 0; i < m_numAuxFields; ++i ) {
			vector<string> v = stringTrimSplit( all[0], '\t' );
			if( i ) fieldVals += "\t";
			fieldVals += v[i];
			for( size_t j = 1; j < numVals; ++j ) {
				v = stringTrimSplit( all[j], '\t' );
				fieldVals += "," + v[i];
			}
		}
	}
	return fieldVals;
}

// Return the target index for the region the last call to ReadOnRegion() mapped to, or 0 if none
uint32_t RegionCoverage::GetLastReadOnRegionIdx() const {
  return m_lastRegionAssigned ? m_lastRegionAssigned->trgIdx : 0;
}

bool RegionCoverage::GetNextRegion( int &contigIdx, int &srtPosition, int &endPosition, bool start )
{
    static uint32_t s_contig = 0;
    static TargetRegion *s_region = NULL;
    // start iteration check
    if( start ) {
    	s_contig = 0;
    	s_region = NULL;
    }
    // ended iteration check
    if( s_contig >= m_numRefContigs ) {
		return false;
    }
    if( s_region ) {
    	s_region = s_region->next;
    } else {
    	// first iteration only
    	s_region = m_contigList[s_contig]->targetRegionHead;
    }
    while( !s_region ) {
    	if( ++s_contig >= m_numRefContigs ) return false;
    	s_region = m_contigList[s_contig]->targetRegionHead;
	}
    contigIdx = s_contig;
    srtPosition = s_region->trgSrt - 1;
    endPosition = s_region->trgEnd;
	return true;
}

string RegionCoverage::Load(
	const string& fileName, const string& fileType, const string& auxFields, const string& trgFields )
{
	if( !m_numRefContigs ) {
		throw runtime_error("RegionCoverage::Load() requires a non-empty list of reference sequences.");
	}
	// fileType is holder for future expansion
	m_headerLine.empty();
	int filetype = 0, srtPosAdj = 0;
	if( fileType == "BED" ) {
		filetype = 1;
		srtPosAdj = 1;
	} else if( fileType == "TAB" ) {
		filetype = 2;
	}
	if( !filetype ) {
		throw runtime_error("RegionCoverage::Load() Unsupported target region file type: "+fileType);
	}
	// C++ does not support passing anonymous arrays or maps
    vector<int> trgFieldsIdx = stringToIntVector(trgFields);
    if( trgFieldsIdx.size() != 3 ) {
    	throw runtime_error("RegionCoverage::Load() Exactly 3 target fields indexes cannot be parsed from list "+trgFields);
    }
    vector<int> auxFieldsIdx;
    try {
    	auxFieldsIdx = stringToIntVector(auxFields);
    } catch (runtime_error& e) {
    	return string("Parsing target regions field index list '") + auxFields + "':\n  " + e.what();
    }
    m_numAuxFields = auxFieldsIdx.size();
    vector<string> auxFieldValues(m_numAuxFields,"");
    size_t numAuxFields = auxFieldValues.size();
    bool auxWarn = true;

    ifstream trgfile(fileName.c_str());
    if( !trgfile.is_open() ) {
    	return "Failed to open region file "+fileName;
    }
    string line, lastContig("");
    int lineNum = 0, numTracks = 0, numFields = 0;
    uint32_t srt, end;
    size_t contigIdx = 0;
    TargetContig *currentContig = NULL;
    while( getline(trgfile,line) ) {
    	++lineNum;
    	vector<string> fields = stringTrimSplit(line,'\t');
    	int fsize = fields.size();
    	if( !fsize ) continue;
    	// handle header line - may later depend on fileType
    	if( filetype == 1 && fields[0].substr(0,5) == "track" ) {
    		if( ++numTracks > 1 ) {
    			fprintf(stderr,"WARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n");
    			break;
    		}
    		if( lineNum > 1 ) {
    			return "Bed file has records before first track line.";
    		}
    		m_headerLine = line;
    		continue;
    	} else if( lineNum == 1 && filetype == 2 ) {
    		m_headerLine = line;
    		continue;
    	}
    	// check primary data fields
    	for( size_t i = 0; i < 3; ++i ) {
    		if( trgFieldsIdx[i] >= fsize || fields[trgFieldsIdx[i]].empty() ) {
    			ostringstream ss;
    			ss << "Missing or blank data field (" << (trgFieldsIdx[i]+1) << ") while reading '";
    			ss << fileName << "' at line " << lineNum;
    			return ss.str();
    		}
    	}
    	// check number of fields consistency
    	if( numFields != fsize ) {
    		if( !numFields ) {
    			numFields = fsize;
    		} else {
				ostringstream ss;
				ss << "Inconsistent number of fields in regions file " << fileName << " at line " << lineNum;
				return ss.str();
    		}
    	}
    	// check for new contig start
    	string contig = fields[trgFieldsIdx[0]];
    	if( lastContig != contig ) {
    		map<string,size_t>::iterator mi = m_contigIdx.find(contig);
    		if( mi == m_contigIdx.end() ) {
    			ostringstream ss;
    			ss << "Contig '" << contig << "' not in reference while reading '";
    			ss << fileName << "' at line " << lineNum;
    			return ss.str();
    		} else {
    			contigIdx = mi->second;
    		}
    		lastContig = contig;
    		currentContig = m_contigList[contigIdx];
    	}
    	// get both coordinates to 1-base
    	try {
			srt = stringToInteger(fields[trgFieldsIdx[1]]) + srtPosAdj;
			end = stringToInteger(fields[trgFieldsIdx[2]]);
    	} catch( runtime_error &e ) {
    		return "Format error in regions file "+fileName+": "+e.what();
    	}
		if( srt == end+1 ) {
			fprintf(stderr,"WARNING: Bed file region has zero length at line %d. Discarded.\n",lineNum);
			continue;
		}
    	if( srt <= 0 || end <= 0 || end < srt ) {
			ostringstream ss;
			ss << "invalid region coordinate " << contig << ":" << (srt-srtPosAdj) << "-" << end;
			ss <<  ") while reading '" << fileName << "' at line " << lineNum;
			return ss.str();
    	}
    	// grab auxiliary fields for new region data
    	// - missing auxiliary fields are given value "" with a warning for first occurrence
    	if( numAuxFields ) {
			for( size_t i = 0; i < m_numAuxFields; ++i ) {
				int idx = auxFieldsIdx[i];
				if( idx < 0 ) idx += fsize;
				if( idx >= fsize && auxWarn ) {
					cerr << "Warning: No value found for auxiliary field " << idx << " at line " << lineNum;
					cerr << " of regions file " << fileName << ". Missing values will default to empty.\n";
					auxWarn = false;
				}
				auxFieldValues[i] = (idx >= 0 && idx < fsize) ? fields[idx] : "";
			}
    	}
    	currentContig->AddRegion( new TargetRegion(srt,end,auxFieldValues) );
    }
    trgfile.close();
    // create array of sorted regions for mapping
    uint32_t rgnIdx = 0;
    for( size_t i = 0; i < m_numRefContigs; ++i ) {
    	m_contigList[i]->ReverseSort(rgnIdx);
    	rgnIdx += m_contigList[i]->numRegions;
    }
	return "";
}

uint32_t RegionCoverage::PullSubRegion(
	uint32_t pullSize, uint32_t &contig, uint32_t &srtPos, uint32_t &endPos, bool &firstContigRegion )
{
	// Assumes an initial call to SetCursorOnRegion() but could be used with other iterators
	// 0 return may be  valid if no more regions to pull (depending on whether expected)
	if( !m_bcovRegion || m_bcovContigIdx >= m_numRefContigs ) return 0;
	// set start to current locus and length to end of current region at cursor
	contig = m_bcovContigIdx;
	srtPos = m_bcovRegionPos;
	firstContigRegion = m_bcovRegion == m_contigList[m_bcovContigIdx]->targetRegionHead;
	uint32_t reglen = m_bcovRegion->trgEnd - m_bcovRegionPos + 1;	// +1 for inclusive
	// Note: exception case for pullSize == 0; also guards against infinite loops for usage
	if( pullSize && reglen > pullSize ) {
		// return sub-region of current region
		m_bcovRegionPos += pullSize;
		endPos = m_bcovRegionPos - 1; // -1 for inclusive
		reglen = pullSize;
	} else {
		// return end of current region
		endPos = m_bcovRegion->trgEnd;
		m_bcovRegionPos = endPos + 1;	// assume next available region overlaps this one
		// move cursor to start of next available region
		m_bcovRegion = m_bcovRegion->next;
		// skip regions that may have already been covered by last region
		while( m_bcovRegion && m_bcovRegion->trgEnd <= endPos ) {
			m_bcovRegion = m_bcovRegion->next;
		}
		// move to next available contig if no more regions on current
		while( !m_bcovRegion ) {
			m_bcovRegionPos = 0;	// note move to new contig
			if( ++m_bcovContigIdx >= m_numRefContigs ) break;
			m_bcovRegion = m_contigList[m_bcovContigIdx]->targetRegionHead;
		}
		// only set to start of next position if not moving backwards to ensure same region is not covered twice
		if( m_bcovRegion && m_bcovRegion->trgSrt > m_bcovRegionPos ) {
			m_bcovRegionPos = m_bcovRegion->trgSrt;
		}
	}
	return reglen;
}

void RegionCoverage::SetCovAtDepths( const string &depths ) {
	m_covAtDepth = BbcUtils::stringToIntVector(depths);
	sort( m_covAtDepth.begin(), m_covAtDepth.end(), less<int>() );
	m_ncovDepths = m_covAtDepth.size();
}

bool RegionCoverage::SetCursorOnRegion( uint32_t contigIdx, uint32_t position )
{
	// safety check
	if( !m_contigList || contigIdx > m_numRefContigs ||
		position == 0 || position > (uint32_t)m_contigList[contigIdx]->length ) {
		return false;
	}
	m_bcovContigIdx = contigIdx;
	while( m_bcovContigIdx < m_numRefContigs ) {
		m_bcovRegion = m_contigList[m_bcovContigIdx]->targetRegionHead;
		for( ; m_bcovRegion; m_bcovRegion = m_bcovRegion->next ) {
			if( position <= m_bcovRegion->trgEnd ) {
				m_bcovRegionPos = position > m_bcovRegion->trgSrt ? position : m_bcovRegion->trgSrt;
				return true;
			}
		}
		++m_bcovContigIdx;
		position = 0;
	}
	// not an error if no region found (within window)
	return true;
}

void RegionCoverage::SetWholeContigTargets()
{
	if( !m_numRefContigs ) {
		throw runtime_error("RegionCoverage::SetWholeContigTargets() requires a non-empty list of reference sequences.");
	}
	vector<string> auxFieldValues;
    for( size_t i = 0; i < m_numRefContigs; ++i ) {
    	m_contigList[i]->AddRegion( new TargetRegion( 1, m_contigList[i]->length, auxFieldValues ) );
    	m_contigList[i]->ReverseSort(i);
    }
}

void RegionCoverage::Write( const string &filename, const string &columnTitles )
{
	// silent do nothing conditions
	if( columnTitles.empty() || filename.empty() ) {
		return;
	}
	// open file or set up for print
	bool tofile = false;
	FILE *fout;
	if( filename == "STDERR" ) {
		fout = stderr;
	} else if( filename == "STDOUT" || filename == "-" ) {
		fout = stdout;
	} else {
		fout = fopen( filename.c_str(), "w" );
		if( !fout ) {
			fprintf(stderr,"ERROR: Failed to open '%s' to write region coverage output.\n",filename.c_str());
			return;
		}
		tofile = true;
	}
	// add header line to output
	if( columnTitles == "BED" ) {
		if( !m_headerLine.empty() ) {
			fprintf( fout, "%s\n", m_headerLine.c_str() );
		}
	} else {
		string headerLine = BbcUtils::replaceAll( columnTitles, ",", "\t" );
		headerLine = BbcUtils::replaceAll( headerLine, "\\\t", "," );
		bool haveHeader = (m_ncovDepths > 0);
		if( !headerLine.empty() ) {
			fprintf( fout, "%s", headerLine.c_str() );
			haveHeader = true;
		}
		for( size_t i = 0; i < m_ncovDepths; ++i ) {
			fprintf( fout, "\tcov%dx", m_covAtDepth[i] );
		}
		if( haveHeader ) fprintf( fout, "\n" );
	}
	for( size_t i = 0; i < m_numRefContigs; ++i ) {
		const char *contig = m_contigList[i]->id.c_str();
		for( TargetRegion *tr = m_contigList[i]->targetRegionHead; tr; tr = tr->next ) {
			fprintf( fout, "%s\t%d\t%d", contig, tr->trgSrt, tr->trgEnd );
			vector<string> &auxData = tr->auxData;
			for( size_t k = 0; k < auxData.size(); ++k ) {
				fprintf( fout, "\t%s", auxData[k].c_str() );
			}
			fprintf( fout, "%s", ReportRegionStatistics(tr).c_str() );
			for( size_t k = 0; k < m_ncovDepths; ++k ) {
				fprintf( fout, "\t%u", (tr->covAtReads ? tr->covAtReads[k] : 0) );
			}
			fprintf( fout, "\n" );
		}
	}
	if( tofile ) fclose(fout);
}

void RegionCoverage::WriteSummary(
	const string &filename, bool invertReadsOnTarget, const string &columnTitles )
{
	// silent do nothing conditions
	if( columnTitles.empty() || filename.empty() ) {
		return;
	}
	// open file or set up for print
	bool tofile = false;
	FILE *fout;
	if( filename == "STDERR" ) {
		fout = stderr;
	} else if( filename == "STDOUT" || filename == "-" ) {
		fout = stdout;
	} else {
		fout = fopen( filename.c_str(), "w" );
		if( !fout ) {
			fprintf(stderr,"ERROR: Failed to open '%s' to write region coverage output.\n",filename.c_str());
			return;
		}
		tofile = true;
	}
	// add header line to output
	string headerLine = BbcUtils::replaceAll( columnTitles, ",", "\t" );
	headerLine = BbcUtils::replaceAll( headerLine, "\\\t", "," );
	if( !headerLine.empty() ) {
		fprintf( fout, "%s\n", headerLine.c_str() );
	}
	for( size_t i = 0; i < m_numRefContigs; ++i ) {
		TargetContig *c = m_contigList[i];
		fprintf( fout, "%s\t%lu\t%lu", c->id.c_str(), c->fwdReads, c->revReads );
		if( invertReadsOnTarget ) {
			fprintf( fout, "\t%lu\t%lu", c->fwdReads - c->fwdTrgReads, c->revReads - c->revTrgReads );
		} else {
			fprintf( fout, "\t%lu\t%lu", c->fwdTrgReads, c->revTrgReads );
		}
		fprintf( fout, "\n" );
	}
	if( tofile ) fclose(fout);
}

//
// ---- Public Virtual Methods for derived methods tacking on base/read statistics per region ---
//

// Pass forward and reverse coverage at base position for collection
// and return region type covered (on or off target).
// The base method handles optional on-target coverage at depth
uint32_t RegionCoverage::BaseDepthOnRegion(
	uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads )
{
	if( !m_ncovDepths ) {
		return BaseOnRegion( contigIdx, position );
	}
	uint32_t covType = BaseOnRegion( contigIdx, position );
	int totReads = fwdReads + revReads;
	if( covType && totReads ) {
		// record coverage for ALL regions overlapping base position
		for( TargetRegion *cur = m_bcovRegion; cur; cur = cur->next ) {
			if( position < cur->trgSrt ) break;
			if( position > cur->trgEnd ) continue;
			AddCovAtDepth( cur, totReads );
		}
	}
	return covType;
}

void RegionCoverage::AddCovAtDepth( TargetRegion *tr, int totReads ) {
	if( !tr->covAtReads ) {
		tr->covAtReads = new uint32_t[m_ncovDepths];
		for( size_t i = 0; i < m_ncovDepths; ++i ) {
			tr->covAtReads[i] = 0;
		}
	}
	for( size_t i = 0; i < m_ncovDepths; ++i ) {
		if( totReads >= m_covAtDepth[i] ) {
			++tr->covAtReads[i];
		}
	}
}

void RegionCoverage::TrackReadsOnRegion( const BamTools::BamAlignment &aread, uint32_t endPos )
{
	// track total and on-target reads
	uint32_t readEnd = endPos ? endPos : aread.GetEndPosition();
	uint32_t covType = ReadOnRegion( aread.RefID, aread.Position + 1, readEnd );
	TargetContig *contig = m_contigList[m_rcovContigIdx];
	if( aread.IsReverseStrand() ) {
		++contig->fwdReads;
		if( covType & 1 ) ++contig->fwdTrgReads;
	} else {
		++contig->revReads;
		if( covType & 1 ) ++contig->revTrgReads;
	}
}

// Return a set of statistics for the given contig, starting and separated by tab, if any
string RegionCoverage::ReportRegionStatistics( TargetRegion *region )
{
	return "";
}

//
// ---- Protected Methods ----
//

// Linear search for base coverage on target (return 0 or 1 for off/on target)
// Tracks first on-target in instance object so can be iterated over all hit targets.
// Reset can be forced by using position = 0
uint32_t RegionCoverage::BaseOnRegion( uint32_t contigIdx, uint32_t position )
{
	if( contigIdx != m_bcovContigIdx && contigIdx < m_numRefContigs ) {
		m_bcovContigIdx = contigIdx;
		m_bcovRegion = m_contigList[m_bcovContigIdx]->targetRegionHead;
	} else if( !position ) {
		m_bcovRegion = m_contigList[m_bcovContigIdx]->targetRegionHead;
	}
	for( ; m_bcovRegion; m_bcovRegion = m_bcovRegion->next ) {
		if( position < m_bcovRegion->trgSrt ) break;
		if( position <= m_bcovRegion->trgEnd ) return 1;
	}
	return 0;
}

// Linear search for read coverage on target (return 0 or 1 for off/on target)for given contig.
// Tracks first on-target in instance object so can be iterated over all hit targets.
// Assumes reads are given in order (for a single contig) so the search starts where it left off
// or at the start of new contig. A reset for the current contig can be forced by using readEnd = 0
// Side-effect: Record the (reference sorted) index of the overlapped region or 0 if none.
uint32_t RegionCoverage::ReadOnRegion( uint32_t contigIdx, uint32_t readSrt, uint32_t readEnd )
{
	if( contigIdx != m_rcovContigIdx && contigIdx < m_numRefContigs ) {
		m_rcovContigIdx = contigIdx;
		m_rcovRegion = m_contigList[m_rcovContigIdx]->targetRegionHead;
	} else if( !readEnd ) {
		m_rcovRegion = m_contigList[m_rcovContigIdx]->targetRegionHead;
	}
	// here we want the first region overlapped by the read
	for( ; m_rcovRegion; m_rcovRegion = m_rcovRegion->next ) {
		if( readEnd < m_rcovRegion->trgSrt ) break;
		if( readSrt <= m_rcovRegion->trgEnd ) {
			m_lastRegionAssigned = m_rcovRegion;
			return 1;
		}
	}
	m_lastRegionAssigned = NULL;
	return 0;
}
