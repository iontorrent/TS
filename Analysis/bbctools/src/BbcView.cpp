// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcView.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: Guy Del Mistro
 */

#include "BbcView.h"

#include <cstdlib>
#include <sstream>
#include <cstdio>

#include "BbcUtils.h"

const uint16_t s_maxVersionNumber = 1000;

BbcView::BbcView()
	: m_bbcfile(NULL)
	, m_noOffTargetPositions(false)
	, m_showOnTargetCoverage(true)
	, m_showContigNames(true)
	, m_showRegionCoordinates(true)
	, m_outputBedCoordinates(false)
	, m_showLociOnly(false)
	, m_showZeroCoverage(false)
	, m_sumFwdRevCoverage(false)
	, m_useRegionAnnotation(false)
	, m_cbcLoaded(false)
	, m_versionNumber(0)
	, m_numContigs(0)
	, m_contigIdx(0)
    , m_firstContigIdx(0)
	, m_position(0)
	, m_wordsize(0)
	, m_readlen(0)
	, m_fcov(0)
	, m_rcov(0)
	, m_ontarg(0)
	, m_ontargInvert(0)
    , m_lastSeekContig(0)
	, m_lastSeekPos(0)
	, m_cbcMinorWidth(0)
	, m_cbcMajorWidth(0)
	, m_fcovSum(0)
	, m_rcovSum(0)
	, m_fcovSumTrg(0)
	, m_rcovSumTrg(0)
	, m_windowSize(0)
    , m_bbcfileRewindPos(0)
	, m_regionCoverage(NULL)
	, m_bbcCreate(NULL)
	, m_bbcIndex(NULL)
	, m_bbcCoarse(NULL)
	, m_bcStream(NULL)
{
	m_contigStr = "";
}

BbcView::~BbcView()
{
	Close();
}

void BbcView::Close()
{
	if( m_bbcfile ) {
		fclose(m_bbcfile);
		m_bbcfile = NULL;
	}
	m_references.clear();
	m_headerLine = "";
}

bool BbcView::CreateCbc( BbcCoarse &bbcCoarse )
{
	// Create a coarse BBC (CBC) file by passing chunks of summed coverage over the whole reference
	if( !m_bbcfile || !bbcCoarse.Open(true) ) return false;
	Rewind();
	if( !bbcCoarse.SetReference( m_references ) ) return false;
	uint32_t binsize = bbcCoarse.GetMinorBinSize();
	bool readOk = true;
	for( uint32_t contig = 0; contig < m_numContigs && readOk; ++contig ) {
		bbcCoarse.SetContig(contig);
		uint32_t nbins = m_references[contig].RefLength / binsize;
		if( m_references[contig].RefLength % binsize ) ++nbins;	// round up #bins
		for( uint32_t bin = 0; bin < nbins; ++bin ) {
			uint32_t binSrt = 1 + bin * binsize;
			if( !ReadSum( contig, binSrt, binSrt+binsize ) ||
				!bbcCoarse.PassCoverage( m_fcovSum, m_rcovSum, m_fcovSumTrg, m_rcovSumTrg ) ) {
				readOk = false;
				break;
			}
		}
	}
	// flush the CBC buffer by resetting the contig (to last contig - otherwise error produced)
	bbcCoarse.SetContig(m_numContigs-1);
	bbcCoarse.Close(readOk);
	Rewind();
	return readOk;
}

bool BbcView::CreateIndex( BbcIndex &indexer )
{
	// Similar to Read() but only reads file anchor points, not actual coverage data
	if( !m_bbcfile || !indexer.Open(true) ) return false;
	Rewind();
	if( !indexer.SetReference( m_references ) ) return false;
	bool readOk = m_versionNumber == 1000 ? CreateIndexVersion1000(indexer) : CreateIndexVersion0(indexer);
	indexer.Close(readOk);
	Rewind();
	return readOk;
}

const BamTools::RefVector BbcView::GetReferenceData() const
{
	return m_references;
}

bool BbcView::GetTotalBaseReads( uint64_t &fwdReads, uint64_t &revReads, uint64_t &fwdTrgReads, uint64_t &revTrgReads )
{
	if( !m_bbcfile ) return false;
	Rewind();
	if( !ReadSum(0,1,0,m_numContigs-1) ) return false;
	fwdReads = m_fcovSum;
	revReads = m_rcovSum;
	fwdTrgReads = m_fcovSumTrg;
	revTrgReads = m_rcovSumTrg;
	return true;
}

uint64_t BbcView::GetWindowSize() const
{
	return m_windowSize;
}

uint64_t BbcView::GetWindowSize( uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition )
{
	// Return the number of bases covered between the current reference limits,
	// limited to just targeted regions if this is provided
	// Sets m_windowSize for calls to recall last window size used.
	if( m_regionCoverage && !m_useRegionAnnotation ) {
		m_windowSize = m_regionCoverage->GetTargetedWindowSize( srtContig, srtPosition, endPosition, endContig );
	} else if( endContig == srtContig ) {
		m_windowSize = endPosition - srtPosition + 1;
	} else {
		uint32_t contig = srtContig;
		m_windowSize = m_references[contig].RefLength - srtPosition + 1;
		while( ++contig < endContig ) {
			m_windowSize += m_references[contig].RefLength;
		}
		m_windowSize += endPosition;
	}
	return m_windowSize;
}

bool BbcView::Open( const string &filename, bool test )
{
	Close();
	m_bbcfile = fopen( filename.c_str(), "rb" );
	if( !m_bbcfile ) {
		if( !test ) {
			fprintf( stderr, "ERROR: Failed to open BBC file '%s' for reading.\n",filename.c_str());
		}
		return false;
	}
	if( !ReadFileHeader() ) {
		if( !test ) {
			fprintf( stderr, "ERROR: File '%s' does not have BBC file format.\n",filename.c_str());
		}
		fclose(m_bbcfile);
		m_bbcfile = NULL;
		return false;
	}
	return true;
}

string BbcView::ParseRegionRange(
	const string &range, uint32_t &srtContig, uint32_t &srtPosition, uint32_t &endContig, uint32_t &endPosition )
{
	// set range to view limits from a range string to instance fields
	// this method scans all contig names against the range string first
	// to avoid possible issues with format punctuation used in contig names
	srtContig = 0;
	srtPosition = 1;
	endContig = m_numContigs - 1;
	endPosition = m_references[endContig].RefLength;
	size_t rangeLen = range.size();
	string range2;
	// while() here just used as if() block with multiple exits
	while( rangeLen ) {
		size_t bestMatchLen = 0;
		for( size_t i = 0; i < m_numContigs; ++i ) {
			size_t len = m_references[i].RefName.size();
			if( range.compare(0,len,m_references[i].RefName) ) continue;
			// check for valid separator for contig name
			if( len != rangeLen && range.compare(len,2,"..") && range.compare(len,1,":") ) {
				continue;
			}
			// in case of ambiguity, record longest contig name match
			if( len > bestMatchLen ) {
				bestMatchLen = len;
				srtContig = i;
			}
		}
		if( !bestMatchLen ) {
			return "Could not identify contig name in current reference.";
		}
		// default if end locus not specified
		endContig = srtContig;
		endPosition = m_references[endContig].RefLength;
		// check/get 1st position range
		size_t len = bestMatchLen;
		if( len == rangeLen ) break;
		if( !range.compare(len,2,"..") ) {
			range2 = range.substr(len+2);
			if( !range2.empty() ) break;
			return "Missing ending contig name after '..'.";
		}
		range2 = BbcUtils::collectDigits(range,++len);
		if( range2.empty() ) {
			return "Missing positional range after ':'.";
		}
		srtPosition = BbcUtils::stringToInteger(range2);
		len += range2.size();
		if( len == rangeLen ) {
			range2 = "";
			break;
		}
		if( !range.compare(len,2,"..") ) {
			range2 = range.substr(len+2);
			if( !range2.empty() ) break;
			return "Missing ending contig name after '..'.";
		}
		// check/get 2nd position range
		if( range.compare(len,1,"-") ) {
			return "Invalid character '"+range.substr(len,1)+"' after 1st coordinate.";
		}
		range2 = BbcUtils::collectDigits(range,++len);
		if( range2.empty() ) {
			return "Missing 2nd coordinate after '-'.";
		}
		endPosition = BbcUtils::stringToInteger(range2);
		len += range2.size();
		if( len == rangeLen ) {
			range2 = "";
			break;
		}
		if( !range.compare(len,2,"..") && len+2 > rangeLen ) {
			return "Invalid end contig ('"  + range.substr(len) + "') after single contig locus.";
		}
		return "Invalid characters ('" + range.substr(len) + "') after coordinates.";
	}
	// repeat similar analysis if a second contig is named for the end locus
	rangeLen = range2.size();
	while( rangeLen ) {
		size_t bestMatchLen = 0;
		for( size_t i = 0; i < m_numContigs; ++i ) {
			size_t len = m_references[i].RefName.size();
			if( range2.compare(0,len,m_references[i].RefName) ) continue;
			// check for valid separator for contig name
			if( len != rangeLen && range2.compare(len,1,":") ) {
				continue;
			}
			// in case of ambiguity, record longest contig name match
			if( len > bestMatchLen ) {
				bestMatchLen = len;
				endContig = i;
			}
		}
		if( !bestMatchLen ) {
			return "Could not identify 2nd contig name in current reference.";
		}
		// check/get 1st position range
		size_t len = bestMatchLen;
		if( len == rangeLen ) {
			endPosition = m_references[endContig].RefLength;
			break;
		}
		string range3 = BbcUtils::collectDigits(range2,++len);
		if( range3.empty() ) {
			return "Missing end contig coordinate after ':'.";
		}
		endPosition = BbcUtils::stringToInteger(range3);
		len += range3.size();
		if( len == rangeLen ) break;
		return "Invalid characters '("+range2.substr(len)+")' after end contig coordinate.";
	}
	// test coordinates against contig lengths
	if( endContig < srtContig ) {
		return "First contig must be indexed before second contig in reference.";
	}
	if( srtPosition < 1 || srtPosition > (uint32_t)m_references[srtContig].RefLength ) {
		return "Start coordinate '"+BbcUtils::integerToString(srtPosition)+"' must in range 1-" +
			BbcUtils::integerToString(m_references[srtContig].RefLength) + " for contig '" +
			m_references[srtContig].RefName + "'.";
	}
	if( endPosition < 1 || endPosition > (uint32_t)m_references[endContig].RefLength ) {
		return "End coordinate '"+BbcUtils::integerToString(endPosition)+"' must in range 1-" +
				BbcUtils::integerToString(m_references[endContig].RefLength) + " for contig '" +
				m_references[endContig].RefName + "'.";
	}
	if( endContig == srtContig && endPosition < srtPosition ) {
		return "End coordinate '"+BbcUtils::integerToString(endPosition) +
			"' is less than start coordinate '" + BbcUtils::integerToString(srtPosition) + "'.";
	}
	return "";	// no error
}

void BbcView::PrintBaseCoverage(
	const char* contig, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	if( !contig || !m_bcStream ) return;
	(this->*m_bcStream)( contig, position, fwdReads, revReads, covType );
}

bool BbcView::ReadAll()
{
	if( !m_bbcfile ) return false;
	uint32_t lastContig = m_numContigs-1;
	uint32_t lastPos = m_references[lastContig].RefLength+1;
	Rewind();
	// using ReadRegions() here is only a performance enhancement
	bool success = m_regionCoverage && !m_useRegionAnnotation && m_noOffTargetPositions && !m_ontargInvert
		? ReadRegions( 0, 1, lastPos, lastContig )
		: ReadRegion(  0, 1, lastPos, lastContig );
	return success;
}

string BbcView::ReadRange( const string &range, bool isolateRegions,
	uint32_t numBins, uint32_t binSize, int32_t srtBin, int32_t endBin )
{
	// using both numBins and binSize is not allowed here
	if( numBins && binSize ) {
		return "ERROR: BbcView::ReadRange() may not be used with both numBins > 0 and binSize > 0.";
	}
	uint32_t rangeSrtContig, rangeSrtPos, rangeEndPos, rangeEndContig;
	string errMsg = ParseRegionRange( range, rangeSrtContig, rangeSrtPos, rangeEndContig, rangeEndPos );
	if( !errMsg.empty() ) {
		return "Error: Parsing window range '" + range + "': " + errMsg;
	}
	// due regions/range overlap it is possible there will be no window to view
	GetWindowSize( rangeSrtContig, rangeSrtPos, rangeEndContig, rangeEndPos );
	if( m_windowSize == 0 ) return "";

	// select and check binning options
	bool viewToRegions = m_regionCoverage && !m_useRegionAnnotation;
	bool fixedWidth = (binSize >= numBins);	// either or both are 0
	uint32_t *contigBins = NULL;
	if( isolateRegions ) {
		// if binSize > 0 then divide regions evenly by this value (except overflow in last bin)
		// else use numBins - which defaults to number of regions if less
		uint32_t numBinsUsed = numBins;
		contigBins = CreateContigBinSizes( numBinsUsed, binSize, rangeSrtContig, rangeSrtPos, rangeEndContig, rangeEndPos );
		// detect and warn for automatic resets
		if( numBinsUsed != numBins && numBins > 0 && binSize == 0 ) {
			cerr << "WARNING: --numbins (" << numBins << ") increased to number of ";
			cerr << (viewToRegions ? "regions" : "contigs") << " within the window (";
			cerr << numBinsUsed << ") as required for the --binContigs option." << endl;
		}
		numBins = numBinsUsed;
	} else if( fixedWidth ) {
		// deal with by-base coverage view exception case
		if( !binSize ) {
			if( !m_headerLine.empty() ) {
				cout << m_headerLine << endl;
			}
			bool success = viewToRegions
				? ReadRegions( rangeSrtContig, rangeSrtPos, rangeEndPos+1, rangeEndContig )
				: ReadRegion(  rangeSrtContig, rangeSrtPos, rangeEndPos+1, rangeEndContig );
			if( success ) return "";
			return "ERROR: Unexpected issue in BbcView::Read().";
		}
		numBins = m_windowSize / binSize;
		if( m_windowSize % binSize ) ++numBins;	// last bin typically smaller
	} else if( numBins > m_windowSize ) {
		cerr << "WARNING: --numbins (" << numBins << ") decreased to the number of bases in the specified ";
		cerr << "contiguous region of the reference (" << m_windowSize << ")." << endl;
		numBins = m_windowSize;
	}
	// set/check bin output range
	if( srtBin ) {
		if( srtBin < 0 ) srtBin += numBins+1;
		if( srtBin < 0 ) srtBin = 0;
		else if( srtBin > (int32_t)numBins ) {
			cerr << "WARNING: Effective first bin (" << srtBin << ") is greater than last bin (" << numBins;
			cerr << "). No coverage bins were output." << endl;
			delete [] contigBins;
			return "";
		}
	}
	if( endBin ) {
		if( endBin < 0 ) endBin += numBins+1;
		if( endBin < 0 ) endBin = 0;
		else if( endBin > (int32_t)numBins ) endBin = (int32_t)numBins;
	}
	if( endBin && endBin < srtBin ) {
		cerr << "WARNING: Effective last bin (" << endBin << ") is less than first bin (" << srtBin;
		cerr << "). No coverage bins were output." << endl;
		delete [] contigBins;
		return "";
	}
	uint32_t firstBin = srtBin ? srtBin : 1;
	uint32_t lastBin  = endBin ? endBin : numBins;

	// output header line => always printed even if no coverage bins output
	if( !m_headerLine.empty() ) {
		cout << m_headerLine << endl;
	}
	// for target regions non-contiguous reference coverage is driven by slices of concatenated targets
	if( viewToRegions ) {
		// set region base cursor to first base in window
		if( !m_regionCoverage->SetCursorOnRegion( rangeSrtContig, rangeSrtPos ) ) {
			delete [] contigBins;
			return "ERROR: Unexpected failure returned from SetCursorOnRegion() in BbcView::Read().";
		}
		uint32_t srtContig = rangeSrtContig, srtPosition = 0;
		// for N bins pull reference regions covered
		uint32_t holder = 0, pullSize;
		for( uint32_t bin = 1; bin <= numBins; ++bin ) {
			if( contigBins ) {
				pullSize = contigBins[bin-1];
				if( bin == numBins ) holder = bin;
			} else if( fixedWidth ) {
				// code below can error where it expects the whole windowSize to be extracted so...
				if( bin < numBins ) pullSize = binSize;
				else pullSize = m_windowSize - holder;
				holder += pullSize;
			} else {
				// code below deals with fractional window size
				// Note: using binEnd+1 mimics strategy used for bbcOverview.pl but that was a 1-off error
				int64_t binEnd = (bin * m_windowSize) / numBins;
				pullSize = binEnd - holder;
				holder = binEnd;
			}
			// collect the total coverage for the targeted regions covered by this bin section
			uint64_t fcovSum = 0, rcovSum = 0, fcovSumTrg = 0, rcovSumTrg = 0;
			uint32_t endContig, srtPos, endPos;
			bool newContigRegion = true, firstPull = true;
			bool outputBin = bin >= firstBin && bin <= lastBin;
			while( pullSize > 0 ) {
				// endContig here is contig for the targeted sub-region just pulled
				uint32_t reglen = m_regionCoverage->PullSubRegion( pullSize, endContig, srtPos, endPos, newContigRegion );
				if( !reglen ) {
					delete [] contigBins;
					return "ERROR: Unexpected failure to collect data from targeted sub-region\n";
				}
				if( outputBin ) {
					if( !ReadSum( endContig, srtPos, endPos+1 ) ) {
						delete [] contigBins;
						return "ERROR: Unexpected failure to collect data from targeted sub-region\n";
					}
					fcovSum += m_fcovSum;
					rcovSum += m_rcovSum;
					fcovSumTrg += m_fcovSumTrg;
					rcovSumTrg += m_rcovSumTrg;
				}
				// record start position for target region - little use when sub-region spans regions
				if( !srtPosition ) srtPosition = srtPos;
				// avoid contig crossing issue where pulled amount starts exactly on start of new contig
				if( firstPull ) {
					firstPull = false;
					if( newContigRegion ) srtContig = endContig;
				}
				pullSize -= reglen;
			}
			if( outputBin ) {
				PrintRegionCoverage( srtContig, srtPosition, endContig, endPos, fcovSum, rcovSum, fcovSumTrg, rcovSumTrg );
			}
			srtContig = endContig;
			srtPosition = 0;
		}
	} else {
		// the range directly corresponds to the reference but is binned vs a contiguous sequence of contigs
		string annotation;
		uint32_t srtContig = rangeSrtContig, srtPos = rangeSrtPos;
		uint32_t holder = 0, pullSize = binSize;
		for( uint32_t bin = 1; bin <= numBins; ++bin ) {
			// end of current contig used to check for extension from start over end of contig(s)
			uint32_t lstPos = m_references[srtContig].RefLength;
			if( contigBins ) {
				pullSize = contigBins[bin-1];
				if( srtPos + pullSize - 1 > lstPos ) {
					// in this case the contig boundaries are not spanned but reset
					lstPos = m_references[++srtContig].RefLength;
					srtPos = 1;
				}
			} else if( fixedWidth ) {
				// code below can error where it expects the whole windowSize to be extracted so...
				if( bin == numBins ) pullSize = m_windowSize - holder;
				holder += pullSize;
			} else {
				// code below deals with fractional window size
				int64_t binEnd = (bin * m_windowSize) / numBins;
				pullSize = binEnd - holder;
				holder = binEnd;
			}
			uint32_t endPos = srtPos + pullSize - 1;	// inclusive
			uint32_t endContig = srtContig;
			while( endPos > lstPos ) {
				endPos = endPos - lstPos;	// should be correct for inclusive 1-based coords
				if( ++endContig == m_numContigs ) {
					delete [] contigBins;
					return "ERROR: Unexpected coverage beyond end of last contig\n";
				}
				lstPos = m_references[endContig].RefLength;
			}
			if( bin >= firstBin && bin <= lastBin ) {
				// assume if a region is used here then it is for annotation
				if( m_regionCoverage ) {
					annotation = m_regionCoverage->FieldsOnRegion( srtContig, srtPos, endPos );
				}
				if( !ReadSum( srtContig, srtPos, endPos+1, endContig ) ) {
					delete [] contigBins;
					return "ERROR: Unexpected failure to collect data from targeted sub-region\n";
				}
				PrintRegionCoverage( srtContig, srtPos, endContig, endPos,
					m_fcovSum, m_rcovSum, m_fcovSumTrg, m_rcovSumTrg, annotation );
			}
			srtContig = endContig;
			srtPos = endPos + 1;
		}
	}
	delete [] contigBins;
	return "";
}

bool BbcView::ReadRegion( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// Note: for consistency with other methods, endPosition is +1 of last base position to include
	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_references[endContig].RefLength+1;

	// move cursor to first base location within range window
	if( !SeekStart( srtContig, srtPosition ) ) return false;

	uint32_t endPos = srtContig < endContig ? m_references[srtContig].RefLength+1 : endPosition;
	while( srtContig <= endContig ) {
		// check for seeking beyond end of one contig in to another
		while( m_contigIdx > srtContig ) {
			// output 0' for gap at end of region - need to temporarily reset contig string
			// NOTE: On target coverage comes from bases flagged in BBC file
			// IF 0 coverage regions were flagged this would not catch that that
			if( m_showZeroCoverage ) {
				m_contigStr = m_references[srtContig].RefName.c_str();
				while( srtPosition < endPos ) {
					StreamCoverage( srtPosition++, 0, 0, 0 );
				}
			}
			if( ++srtContig > endContig ) break;
			srtPosition = 1;
			endPos = srtContig < endContig ? m_references[srtContig].RefLength+1 : endPosition;
			m_contigStr = m_references[srtContig].RefName.c_str();
		}
		// output 0's for start gaps (m_position should be at least 1 short of next base covered)
		if( m_showZeroCoverage ) {
			while( srtPosition < m_position ) {
				StreamCoverage( srtPosition++, 0, 0, 0 );
				if( srtPosition == endPos ) return true;
			}
			if( srtContig > endContig || m_position >= endPos ) break;
		} else {
			if( srtContig > endContig || m_position >= endPos ) break;
			srtPosition = m_position;
		}
		// stream single base coverage currently read to buffer
		if( m_showZeroCoverage || m_fcov || m_rcov ) {
			StreamCoverage( m_position, m_fcov, m_rcov, m_ontarg );
		}
		// read next covered base location
		if( !(m_versionNumber == 1000 ? ReadBaseCov1000() : ReadBaseCov0()) ) return false;
		++srtPosition;	// expected m_position if no jump ahead (over 0 coverage)
	}
	return true;
}

bool BbcView::ReadRegions( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// Note: for consistency with other methods, endPosition is +1 of last base position to include
	if( !m_regionCoverage ) return false;	// safety

	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_references[endContig].RefLength+1;

	// set region base cursor to first base in window
	if( !m_regionCoverage->SetCursorOnRegion( srtContig, srtPosition ) ) return false;

	// slightly kludgy way to prevent StreamCoverage() from using m_regionCoverage for defining on-target
	RegionCoverage *regionCoverage = m_regionCoverage;
	m_regionCoverage = NULL;

	uint32_t contig, srtPos, endPos;
	bool firstContigRegion;	// not needed here
	while( regionCoverage->PullSubRegion( 0, contig, srtPos, endPos, firstContigRegion ) ) {
		if( contig > endContig ) break;
		if( contig == endContig && endPos >= endPosition ) {
			if( srtPos >= endPosition ) break;
			endPos = endPosition-1;	// for partial region coverage at right of window
		}
		if( !ReadRegion( contig, srtPos, endPos+1, contig ) ) {
			m_regionCoverage = regionCoverage;
			return false;
		}
	}
	m_regionCoverage = regionCoverage;
	return true;
}

void BbcView::Rewind()
{
	if( !m_bbcfile ) return;
	m_contigIdx = m_firstContigIdx;
	m_contigStr = m_references[m_contigIdx].RefName.c_str();
	fseek( m_bbcfile, m_bbcfileRewindPos, SEEK_SET );
	m_wordsize = m_readlen = m_ontarg = 0;
	m_lastSeekContig = m_lastSeekPos = 0;
	m_position = 0;
}

void BbcView::SelectPrintStream( const string &streamID )
{
	if( streamID == "" || streamID == "NONE" ) {
		m_bcStream = NULL;
	} else if( streamID == "SAMDEPTH" ) {
		m_bcStream = &BbcView::SamDepthPrint;
	} else if( streamID == "BBCVIEW" ) {
		m_bcStream = &BbcView::BbcViewPrint;
	} else if( streamID == "COMPACT" ) {
		m_bcStream = &BbcView::CompactPrint;
	} else {
		throw runtime_error("Unknown print stream ID '"+streamID+"' passed.");
	}
}

void BbcView::SetBbcCreate( BbcCreate *bbcCreate )
{
	m_bbcCreate = bbcCreate;
}

void BbcView::SetBbcCoarse( BbcCoarse *bbcCoarse )
{
	m_bbcCoarse = bbcCoarse;
	m_cbcLoaded = false;	// only fully loaded on demand
	m_cbcMinorWidth = m_bbcCoarse->GetMinorBinSize();
	m_cbcMajorWidth = m_bbcCoarse->GetMajorBinSize();
}

void BbcView::SetBbcIndex( BbcIndex *bbcIndex )
{
	m_bbcIndex = bbcIndex;
}

void BbcView::SetRegionCoverage( RegionCoverage *regionCoverage )
{
	m_regionCoverage = regionCoverage;
}

void BbcView::SetHeaderLine( const string &headerLine, bool commasToTabs )
{
	m_headerLine = headerLine;
	if( commasToTabs ) {
		m_headerLine = BbcUtils::replaceAll( m_headerLine, ",", "\t" );
		// put back escaped (using \\) comma's
		m_headerLine = BbcUtils::replaceAll( m_headerLine, "\\\t", "," );
	}
}

void BbcView::SetHideOnTargetCoverage( bool hide )
{
	m_showOnTargetCoverage = !hide;
}

void BbcView::SetHideContigNames( bool hide )
{
	m_showContigNames = !hide;
}

void BbcView::SetHideRegionCoordinates( bool hide )
{
	m_showRegionCoordinates = !hide;
}

void BbcView::SetInvertOnTarget( bool invert )
{
	m_ontargInvert = invert ? 1 : 0;
}

void BbcView::SetNoOffTargetPositions( bool hide )
{
	m_noOffTargetPositions = hide;
}

void BbcView::SetOutputBedCoordinates( bool bed )
{
	m_outputBedCoordinates = bed;
}

void BbcView::SetShowLociOnly( bool show )
{
	m_showLociOnly = show;
}

void BbcView::SetShowZeroCoverage( bool show )
{
	m_showZeroCoverage = show;
}

void BbcView::SetSumFwdRevCoverage( bool sum )
{
	m_sumFwdRevCoverage = sum;
}

void BbcView::SetUseRegionAnnotation( bool use )
{
	m_useRegionAnnotation = use;
}

string BbcView::VersionString()
{
	double vnum = (double)(m_bbcfile ? m_versionNumber : s_maxVersionNumber)/1000;
	return BbcUtils::numberToString( vnum, 3 );
}

// ---- Private methods ----

void BbcView::BaseCoveragePrint( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	// helper function to avoid code repetition
	if( m_showRegionCoordinates ) {
		if( m_outputBedCoordinates ) --position;
		printf( "%u\t", position );
	}
	if( m_showLociOnly ) {
		if( m_showContigNames | m_showRegionCoordinates ) printf("\n");
		return;
	}
	if( m_showOnTargetCoverage ) {
		printf( "%u\t", covType & 1 );
	}
	if( m_sumFwdRevCoverage ) printf( "%u\n", fwdReads+revReads );
	else printf( "%u\t%u\n", fwdReads, revReads );
}

uint32_t *BbcView::CreateContigBinSizes( uint32_t &numBins, uint32_t binSize,
	uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition )
{
	// If binSize == 0 this method returns an array of size numBins containing the contiguous number
	// of bases covered from the starting regions, or NULL if an error occurs (case not implemented).
	// numBins is automatically reset to the number of bins in the window if initially smaller.
	// If binSize > 0 each contig/region has bins assigned to cover to this size equally, excect the
	// round-off in the last bin.

	// defer to similar method for target regions
	if( m_regionCoverage && !m_useRegionAnnotation ) {
		return m_regionCoverage->CreateTargetBinSizes( numBins, binSize, srtContig, srtPosition, endPosition, endContig );
	}
	// make local copy of contig sizes required, noting partial srt/end contig lengths
	uint32_t numContigs = endContig - srtContig + 1;
	uint32_t *csizes = new uint32_t[numContigs];
	if( endContig == srtContig ) {
		csizes[0] = endPosition - srtPosition + 1;
	} else {
		csizes[0] = m_references[srtContig].RefLength - srtPosition + 1;
		uint32_t pContig = 1;
		for( uint32_t contig = srtContig+1; contig < endContig; ++contig ) {
			csizes[pContig++] = m_references[contig].RefLength;
		}
		csizes[pContig] = endPosition;
	}
	// this hold the number of bins assigned to each contig
	uint32_t *cnbins = new uint32_t[numContigs];
	if( binSize > 0 ) {
		// get number of binSizes in each contig
		numBins = 0;
		for( uint32_t contig = srtContig, c = 0; contig <= endContig; ++contig ) {
			uint32_t nbins = csizes[c] / binSize;
			if( csizes[c] % binSize ) ++nbins;
			cnbins[c++] = nbins;
			numBins += nbins;
		}
		// spread coverage in equal chunks, except last for each contig
		uint32_t *binWidths = new uint32_t[numBins];
		for( uint32_t contig = srtContig, c = 0, i = 0; contig <= endContig; ++contig, ++c ) {
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
	double *esizes = new double[numContigs];
	for( size_t i = 0; i < numContigs; ++i ) {
		esizes[i] = (double)csizes[i];
	}
	// Every contig will be represented by N >= 1 bins containing a roughly even amount of base coverage, with
	// the number of bins for each contig distributed as proportionately as possible according to the relative
	// sizes of the contigs. The calling algorithm is expected to know starting position of first contig and to
	// tally sizes for each bin to track contig changes, so that return need only be a list of contiguous widths.
	for( size_t i = 0; i < numContigs; ++i ) {
		cnbins[i] = 1;
	}
	if( numBins < numContigs ) numBins = numContigs;
	uint32_t xbins = numBins - numContigs;				// number of bins left to distribute
	while( xbins ) {
		// get largest effective contig size remaining and give that an extra bin
		uint32_t m = 0;
		for( uint32_t i = 1; i < numContigs; ++i ) {
			if( esizes[i] > esizes[m] ) m = i;
		}
		// incr #bins for the largest contig and reduce its effective size as if spread over N bins
		++cnbins[m];
		esizes[m] = (double)csizes[m] / cnbins[m];
		--xbins;
 	}
	// spread the final contig bins into linear array of uniformly spread bin widths
	uint32_t *binWidths = new uint32_t[numBins];
	for( size_t i = 0, bin = 0; i < numContigs; ++i ) {
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

bool BbcView::CreateIndexVersion0( BbcIndex &indexer )
{
	// inform indexer of first contig position with coverage
	if( !indexer.SetContig(m_firstContigIdx) ) return false;
	uint32_t position, reg_head, wordsize, reglen;
	while(1) {
		long fpos = ftell(m_bbcfile);
		// assume correctly read to the end of the file if can't grab another more bytes at this point
		if( fread( &position, sizeof(uint32_t), 1, m_bbcfile) != 1 ) return true;
		// expecting data here so failure to read is an error
		if( fread( &reg_head, sizeof(uint32_t), 1, m_bbcfile) != 1 ) break;
		if( !position ) {
			m_contigIdx = reg_head - 1;
			if( !indexer.SetContig(m_contigIdx) ) break;
			continue;
		}
		wordsize = reg_head & 6;
		if( !wordsize ) continue;
		if( wordsize == 6 ) wordsize = 8;
		reglen = reg_head >> 3;
		indexer.PassAnchor( position+reglen, fpos );
		fseek( m_bbcfile, wordsize * reglen, SEEK_CUR );
	}
	return false;
}

bool BbcView::CreateIndexVersion1000( BbcIndex &indexer )
{
	// inform indexer of first contig position with coverage
	if( !indexer.SetContig(m_firstContigIdx) ) return false;
	uint32_t position, wordsize, reglen;
	uint16_t reg_head;
	long anchorFpos = 0;
	while(1) {
		long fpos = ftell(m_bbcfile);
		// assume correctly read to the end of the file if can't grab another more bytes at this point
		if( fread( &reg_head, 1, sizeof(uint16_t), m_bbcfile) != sizeof(uint16_t) ) return true;
		if( reg_head & 0x8000 ) {
			reg_head &= 0x7FFF;
		} else {
			// expecting data here so failure to read is an error
			if( fread( &position, 1, sizeof(uint32_t), m_bbcfile) != sizeof(uint32_t) ) break;
			anchorFpos = fpos;	// last valid anchor prior to region length overlapping an index site
		}
		if( !reg_head ) {
			m_contigIdx = position;
			if( !indexer.SetContig(m_contigIdx) ) break;
			continue;
		}
		wordsize = reg_head & 6;
		if( wordsize == 6 ) wordsize = 8;
		else if( wordsize == 0 ) wordsize = 1;
		reglen = reg_head >> 3;
		// record last anchor if this region overlaps an index site
		position += reglen;
		indexer.PassAnchor( position, anchorFpos );
		fseek( m_bbcfile, wordsize * reglen, SEEK_CUR );
	}
	return false;
}

void BbcView::PrintRegionCoverage(
	uint32_t srtContig, uint32_t srtPosition, uint32_t endContig, uint32_t endPosition,
	uint64_t fcovSum, uint64_t rcovSum, uint64_t fcovSumTrg, uint64_t rcovSumTrg, const string annotation )
{
	// print the coverage for the given region employing formating options
	if( m_showContigNames ) {
		string contigId = m_references[srtContig].RefName;
		if( endContig != srtContig ) {
			contigId += "--" + m_references[endContig].RefName;
		}
		printf( "%s\t", contigId.c_str() );
	}
	if( m_showRegionCoordinates ) {
		if( m_outputBedCoordinates ) --srtPosition;
		printf( "%u\t%u\t", srtPosition, endPosition );
	}
	if( m_showLociOnly ) {
		if( m_showContigNames | m_showRegionCoordinates ) {
			if( annotation.empty() ) printf("\n");
			else printf( "\t%s\n", annotation.c_str() );
		}
		return;
	}
	if( m_ontargInvert ) {
		fcovSumTrg = fcovSum - fcovSumTrg;
		rcovSumTrg = rcovSum - rcovSumTrg;
	}
	if( m_sumFwdRevCoverage ) {
		printf( "%lu", fcovSum+rcovSum );
	} else {
		printf( "%lu\t%lu", fcovSum, rcovSum );
	}
	if( m_showOnTargetCoverage ) {
		if( m_sumFwdRevCoverage ) printf( "\t%lu", fcovSumTrg+rcovSumTrg );
		else printf( "\t%lu\t%lu", fcovSumTrg, rcovSumTrg );
	}
	if( m_useRegionAnnotation ) {
		printf( "\t%s\n", annotation.c_str() );
	}
	else {
		printf("\n");
	}
}

bool BbcView::ReadBaseCov0( uint32_t skipToPosition ) {
	// Read 1 base coverage from the current file location (cursor) to m_fcov, m_rcov & m_ontarg
	// with m_position set to the base position just read
	// OR if skipToPosition > 0 move cursor to just before the first coverage region >= skipToPosition
	// To read to the end of current contig have skipToPosition > length of contig.
	//
    while(1) {
    	// check if attempting to skip to position in front of or at cursor
		// - has to be checked before pending read length as single base could have been read at cursor
		if( skipToPosition && skipToPosition <= m_position ) {
			return true;
		}
    	// first read any pending coverage region
		if( m_readlen ) {
			// m_position is one before the loci of next read
			if( skipToPosition ) {
				uint32_t ws = m_wordsize ? m_wordsize << 1 : 1;
				if( m_position + m_readlen < skipToPosition ) {
					// skip over current segment and look at next
					fseek( m_bbcfile, ws * m_readlen, SEEK_CUR );
					m_position += m_readlen;
					m_readlen = 0;
					continue;
				}
				// skip partial segment (>= 0) and read next base coverage
				uint32_t skiplen = skipToPosition - m_position - 1;
				fseek( m_bbcfile, ws * skiplen, SEEK_CUR );
				m_position += skiplen;
				m_readlen -= skiplen;
			}
			if( m_wordsize ) {
				if( fread( &m_fcov, m_wordsize, 1, m_bbcfile) != 1 ) break;
				if( fread( &m_rcov, m_wordsize, 1, m_bbcfile) != 1 ) break;
			} else {
				if( fread( &m_fcov, 1, 1, m_bbcfile) != 1 ) break;
				m_rcov = m_fcov >> 4 & 15;
				m_fcov &= 15;
			}
			--m_readlen;
			++m_position;
			return true;
		}
		// a non-resumed call begins here
		uint32_t reg_head;
		if( fread( &m_position, sizeof(uint32_t), 1, m_bbcfile) != 1 ) {
			// assume if it break here then this is EOF (otherwise need ftell() vs. file size)
			m_contigIdx = m_numContigs;
			return true;
		}
		if( fread( &reg_head, sizeof(uint32_t), 1, m_bbcfile) != 1 ) break;
		if( !m_position ) {
			m_contigIdx = reg_head - 1;
			if( m_contigIdx >= m_numContigs ) return false;
			m_contigStr = m_references[m_contigIdx].RefName.c_str();
			skipToPosition = 0;
			continue;
		}
		--m_position;	// unfortunate trick to have ++m_position correct above
		m_wordsize = (reg_head & 6) >> 1;
		if( !m_wordsize ) continue; // empty regions ignored, if ever produced
		if( m_wordsize == 3 ) m_wordsize = 4;
		m_readlen = reg_head >> 3;
		m_ontarg = reg_head & 1;
		m_fcov = m_rcov = 0;	// clear top bytes & for skip to just prior to coverage
    }
    return false;
}

bool BbcView::ReadBaseCov1000( uint32_t skipToPosition ) {
	// Read 1 base coverage from the current file location (cursor) to m_fcov, m_rcov & m_ontarg
	// with m_position set to the base position just read
	// OR if skipToPosition > 0 move cursor to just before the first coverage region >= skipToPosition
	// To read to the end of current contig have skipToPosition > length of contig.
	//
    while(1) {
    	// check if attempting to skip to position in front of or at cursor
		// - has to be checked before pending read length as single base could have been read at cursor
		if( skipToPosition && skipToPosition <= m_position ) {
			return true;
		}
		// first read any pending coverage region
		if( m_readlen ) {
			// m_position is one before the loci of next read
			if( skipToPosition ) {
				uint32_t ws = m_wordsize ? m_wordsize << 1 : 1;
				if( m_position + m_readlen < skipToPosition ) {
					// skip over current segment and look at next
					fseek( m_bbcfile, ws * m_readlen, SEEK_CUR );
					m_position += m_readlen;
					m_readlen = 0;
					continue;
				}
				// skip partial segment (>= 0) and read next base coverage
				uint32_t skiplen = skipToPosition - m_position - 1;
				fseek( m_bbcfile, ws * skiplen, SEEK_CUR );
				m_position += skiplen;
				m_readlen -= skiplen;
			}
			if( m_wordsize ) {
				if( fread( &m_fcov, m_wordsize, 1, m_bbcfile) != 1 ) break;
				if( fread( &m_rcov, m_wordsize, 1, m_bbcfile) != 1 ) break;
			} else {
				if( fread( &m_fcov, 1, 1, m_bbcfile) != 1 ) break;
				m_rcov = m_fcov >> 4 & 15;
				m_fcov &= 15;
			}
			--m_readlen;
			++m_position;
			return true;
		}
		// a non-resumed call begins here
		uint16_t reg_head;
		if( fread( &reg_head, 1, sizeof(uint16_t), m_bbcfile) != sizeof(uint16_t) ) {
			// assume if it break here then this is EOF (otherwise need ftell() vs. file size)
			m_contigIdx = m_numContigs;
			return true;
		}
		if( reg_head ) {
			// only read position data if top bit of reg_head is not set
			if( reg_head & 0x8000 ) {
				reg_head &= 0x7FFF;	// remove to get region length
			} else {
				if( fread( &m_position, 1, sizeof(uint32_t), m_bbcfile) != sizeof(uint32_t) ) break;
				--m_position;	// unfortunate trick to have ++m_position correct above
			}
			m_wordsize = (reg_head & 6) >> 1;
			if( m_wordsize == 3 ) m_wordsize = 4;
			m_readlen = reg_head >> 3;
			m_ontarg = reg_head & 1;
			m_fcov = m_rcov = 0;	// clear top bytes & for skip to just prior to coverage
		} else {
			// reg_head == 0  =>  grab new position
			if( fread( &m_contigIdx, 1, sizeof(uint32_t), m_bbcfile) != sizeof(uint32_t) ) break;
			if( m_contigIdx >= m_numContigs ) break;	// fail safe
			m_contigStr = m_references[m_contigIdx].RefName.c_str();
			skipToPosition = 0;
		}
    }
    // end of file cannot be
    return false;
}

bool BbcView::ReadSum(
	uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// Performs the same task as ReadSum() but employing CBC for fast sums over large ranges
	// NOTE: endPosition is +1 of the last position covered.

	// These retain the residual end coverage from a previous call to enhance performance
	// when binning over contiguous adjacent regions
	static uint32_t s_endContig = 0, s_endPosition = 0;
	static uint64_t s_fcovSum = 0, s_rcovSum = 0, s_fcovSumTrg = 0, s_rcovSumTrg = 0;

	// if no CBC file is provided then use standard method
	if( !m_bbcCoarse ) return ReadSumNoCbc( srtContig, srtPosition, endPosition, endContig );

	// retain original values in case loading CBC fails
	uint32_t c_srtContig = srtContig, c_srtPosition = srtPosition;
	uint32_t c_endContig = endContig, c_endPosition = endPosition;

	// check for default arg. overrides
	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_references[endContig].RefLength+1;

	// check for continuation from last call (to avoid re-calculation of leading over hang)
	bool loadLeading = (s_endContig != srtContig || s_endPosition != srtPosition);
	s_endContig = endContig;
	s_endPosition = endPosition;

	m_fcovSum = m_rcovSum = m_fcovSumTrg = m_rcovSumTrg = 0;
	bool cbcIssue = false;
	for( ; srtContig <= endContig; ++srtContig, srtPosition = 1, loadLeading = true ) {
		// get size of contiguous region to assess
		uint32_t endPos = srtContig < endContig ? m_references[srtContig].RefLength+1 : endPosition;
		uint32_t width = endPos - srtPosition;
		// check for inclusion of whole last bin
		bool addLastMajorBin = endPos > (uint32_t)m_references[srtContig].RefLength;
		bool addWholeBin = (addLastMajorBin && srtPosition == 1);
		// CBC file no use if region is smaller than minor bin size (or perhaps even a multiple of this?)
		if( width <= m_cbcMinorWidth && !addWholeBin ) {
			// add in the base coverage region directly
			if( !ReadSumFragment( srtContig, srtPosition, endPos,
					s_fcovSum, s_rcovSum, s_fcovSumTrg, s_rcovSumTrg ) ) {
				cbcIssue = true;
				break;
			}
		} else {
			// loading of CBC file to memory is deferred until/unless required
			if( !m_cbcLoaded ) {
				if( !m_bbcCoarse->SetReference(m_references) || !m_bbcCoarse->LoadAll() ) {
					cbcIssue = true;
					break;
				}
				m_cbcLoaded = true;
			}
			// add in major block coverage (if any provided and spanned) up to srtPosition
			m_bbcCoarse->SumMajorCoverage( srtContig, srtPosition, endPos, addLastMajorBin,
				m_fcovSum, m_rcovSum, m_fcovSumTrg, m_rcovSumTrg );

			// get new leading minor coverage interval, if not recorded as last trailing minor interval
			if( loadLeading ) {
				// add in base coverage segment unless region is aligned to a minor block (c0 == 0) and
				// add in minor block coverage (where minor block is not on a major block boundary)
				uint32_t c0 = (srtPosition-1) % m_cbcMinorWidth;
				if( !ReadSumFragment( srtContig, srtPosition-c0, srtPosition,
						s_fcovSum, s_rcovSum, s_fcovSumTrg, s_rcovSumTrg ) ||
					!m_bbcCoarse->SumMajorMinorCoverage( srtContig, srtPosition,
						s_fcovSum, s_rcovSum, s_fcovSumTrg, s_rcovSumTrg ) ) {
					cbcIssue = true;
					break;
				}
			}
			// subtract leading minor coverage interval
			m_fcovSum -= s_fcovSum;
			m_rcovSum -= s_rcovSum;
			m_fcovSumTrg -= s_fcovSumTrg;
			m_rcovSumTrg -= s_rcovSumTrg;
			// no need for trailing bin if collecting coverage over end of contig
			if( addLastMajorBin ) continue;

			// get trailing minor coverage - becomes potential leading interval for next call
			uint32_t c0 = (endPos-1) % m_cbcMinorWidth;
			if( !ReadSumFragment( srtContig, endPos-c0, endPos,
					s_fcovSum, s_rcovSum, s_fcovSumTrg, s_rcovSumTrg ) ||
				!m_bbcCoarse->SumMajorMinorCoverage( srtContig, endPos,
					s_fcovSum, s_rcovSum, s_fcovSumTrg, s_rcovSumTrg ) ) {
				cbcIssue = true;
				break;
			}
		}
		// add in trailing or whole small coverage interval
		m_fcovSum += s_fcovSum;
		m_rcovSum += s_rcovSum;
		m_fcovSumTrg += s_fcovSumTrg;
		m_rcovSumTrg += s_rcovSumTrg;
	}
	// check for issues with old CBC files or reference mismatch,
	if( cbcIssue ) {
		// proceed as if there was no CBC available
		m_bbcCoarse->Close();
		m_bbcCoarse = NULL;
		return ReadSumNoCbc( c_srtContig, c_srtPosition, c_endPosition, c_endContig );
	}
	return true;
}

bool BbcView::ReadSumFragment( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition,
		uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg )
{
	// returns summed base coverage over (short) region on single contig
	fcovSum = rcovSum = fcovSumTrg = rcovSumTrg = 0;

	// no bases covered, e.g. called on cbc boundary
	if( srtPosition == endPosition ) return true;
	if( !SeekStart( srtContig, srtPosition ) ) return false;
	while(1) {
		// check for overrun on current contig, e.g. no more coverage
		srtPosition = m_position;
		if( m_contigIdx > srtContig || srtPosition >= endPosition ) break;
		fcovSum += m_fcov;
		rcovSum += m_rcov;
		if( m_ontarg ) {
			fcovSumTrg += m_fcov;
			rcovSumTrg += m_rcov;
		}
		if( !(m_versionNumber == 1000 ? ReadBaseCov1000() : ReadBaseCov0()) ) return false;
	}
	return true;
}

bool BbcView::ReadSumNoCbc(
	uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, uint32_t endContig )
{
	// Similar to Read() except whole coverage of region summed to following member vars
	// NOTE: endPosition is +1 of the last position covered.
	m_fcovSum = m_rcovSum = m_fcovSumTrg = m_rcovSumTrg = 0;

	// check for default arg. overrides
	if( endContig < srtContig ) endContig = srtContig;
	if( endPosition == 0 ) endPosition = m_references[endContig].RefLength+1;

	if( !SeekStart( srtContig, srtPosition ) ) return false;

	uint32_t endPos = srtContig < endContig ? m_references[srtContig].RefLength+1 : endPosition;
	while( srtContig <= endContig ) {
		// check for seeking beyond end of one contig in to another
		if( m_contigIdx > srtContig ) {
			// this checks for end of file too
			if( m_contigIdx > endContig || m_contigIdx >= m_numContigs ) break;
			srtPosition = 1;
			srtContig = m_contigIdx;
			endPos = srtContig < endContig ? m_references[srtContig].RefLength+1 : endPosition;
		}
		// check for move beyond end position
		if( endPos-1 < m_position ) {
			return true;
		}
		// add base coverage to sums
		m_fcovSum += m_fcov;
		m_rcovSum += m_rcov;
		if( m_ontarg ) {
			m_fcovSumTrg += m_fcov;
			m_rcovSumTrg += m_rcov;
		}
		srtPosition = m_position;
		// read next covered base location (to be consistent for start of next call)
		if( !(m_versionNumber == 1000 ? ReadBaseCov1000() : ReadBaseCov0()) ) return false;
		if( srtPosition >= endPos ) break;
	}
	return true;
}

bool BbcView::ReadFileHeader()
{
	// Attempt to read reference/contigs list header and determine version format
	// - most likely place to fail if file is not BBC format (return false)
	stringstream ss;
	int32_t ic;
	vector<string> refNames;
	vector<int32_t> refLengths;
	bool sepV0 = false, sepV1 = false, idToken = true, validHead = false;
	while( (ic = fgetc(m_bbcfile)) != EOF ) {
		unsigned char c = (unsigned char)ic;
		// V0 format: [<name>$<length>\t]...  + \n
		// V1 format: [<name>\t<length>\n]...[<name>\t<length>\0]
		if( !c ) {
			if( sepV1 ) {
				if( idToken || ss.eof() ) break;
				ss >> ic;
				refLengths.push_back( ic );
			}
			validHead = sepV1;
			break;
		} else if( c == '\t' ) {
			if( ss.eof() ) break;
			if( idToken ) {
				if( sepV0 ) break;
				sepV1 = true;
				refNames.push_back( ss.str() );
			} else {
				if( sepV1 ) break;
				ss >> ic;
				refLengths.push_back( ic );
			}
		} else if( c == '$' ) {
			if( !idToken || ss.eof() ) break;
			// allow as part (but not whole) of name for V1 format
			if( !sepV1 ) sepV0 = true;
			refNames.push_back( ss.str() );
		} else if( c == '\n' ) {
			if( idToken || ss.eof() ) break;
			ss >> ic;
			refLengths.push_back( ic );
			if( sepV0 ) {
				validHead = true;
				break;
			}
		} else {
			// always invalid characters where not handled above
			if( c < 32 || c >= 127 ) break;
			// non-digits invalid for reference length token
			if( !idToken && (c < '0' || c > '9') ) break;
			// valid data character collection
			ss << c;
			continue;
		}
		// swap expected token type and clear stringstream
		idToken = !idToken;
		ss.str(string());
		ss.clear();
	}
	// check for unexpected EOF, etc.
	if( !validHead ) return false;
	m_numContigs = refNames.size();
	if( m_numContigs == 0 ) return false;
	if( refNames.size() != refLengths.size() ) return false;
	// ensure the version number and initial contig idx is read given version
	if( sepV1 ) {
		if( fread( &m_versionNumber, 1, sizeof(uint16_t), m_bbcfile) != sizeof(uint16_t) ) return false;
		if( m_versionNumber > s_maxVersionNumber ) return false;
		if( fread( &m_contigIdx, 1, sizeof(uint32_t), m_bbcfile) != sizeof(uint32_t) ) return false;
	} else {
		if( fread( &m_contigIdx, sizeof(uint32_t), 1, m_bbcfile) != 1 ) return false;
		if( m_contigIdx ) return false;	// should be 0 for V0
		if( fread( &m_contigIdx, sizeof(uint32_t), 1, m_bbcfile ) != 1 ) return false;
		--m_contigIdx; // saved with +1 in V0 format
	}
	if( m_contigIdx >= m_numContigs ) return false;
	// create references object for compatibility with bamtools
	for( size_t i = 0; i < refNames.size(); ++i ) {
		m_references.push_back( BamTools::RefData( refNames[i], refLengths[i] ) );
	}
	m_contigStr = m_references[m_contigIdx].RefName.c_str();
	// save start of file data markers for rewind
	m_firstContigIdx = m_contigIdx;
	m_bbcfileRewindPos = ftell(m_bbcfile);
	return true;
}

bool BbcView::SeekStart( uint32_t contigIdx, uint32_t position )
{
	// Set current file read position (the cursor) to the given start locus, using indexer if provided
	// endPos is only required for testing if a Rewind() is necessary and where no indexer is provided
	// Check contigIdx is within range
	if( contigIdx >= m_numContigs ) {
		// reset last seek locus so a call can be made to explicitly reset these
		m_lastSeekPos = m_lastSeekContig = 0;
		return false;
	}
	// Best case is continuing from where last left off
	if( contigIdx == m_contigIdx && position == m_position ) {
		m_lastSeekPos = position;
		m_lastSeekContig = contigIdx;
		return true;
	}
	// If not an explicit back seek BUT cursor contig is ahead of requested seek assume coverage run-off
	bool backSeek = contigIdx < m_lastSeekContig || (contigIdx == m_lastSeekContig && position <= m_lastSeekPos);
	if( !backSeek && m_contigIdx > contigIdx ) {
		return true;
	}
	if( m_bbcIndex ) {
		// determine if a forward file seek is necessary
		bool fwdSeek = contigIdx > m_contigIdx || (contigIdx == m_contigIdx && position > m_position);
		// indexing allows random access file seeking
		long fpos = (fwdSeek || backSeek) ? m_bbcIndex->GetFilePos( contigIdx, position, backSeek ) : 0;
		// 0 return means either (1) no need to reset file pointer (within range)
		// (2) no more coverage at or beyond requested locus or (3) an (unlikely) coding issue
		if( fpos ) {
			fseek( m_bbcfile, fpos, SEEK_SET );
			m_contigIdx = m_bbcIndex->GetContigIdx();
			m_contigStr = m_references[m_contigIdx].RefName.c_str();
			// ensure the new position is read from file
			m_readlen = m_position = 0;
			// if contig is not that requested then end or more of contig was skipped
			// - set skip to force next base read to buffer
			if( contigIdx != m_contigIdx ) position = 0;
		}
	} else if( backSeek ) {
		// without indexing backwards seeking requires full rewind - a fail safe for V0 indexing format
		Rewind();
	}
	// record last seek target or detecting backwards access
	// - if seeks to a new contig the previous seek is a better test for backwards seeks
	if( m_contigIdx <= contigIdx ) {
		m_lastSeekPos = position;
		m_lastSeekContig = contigIdx;
	}
	// force read over skipped whole contigs (non-indexed seek)
	while( m_contigIdx < contigIdx ) {
		uint32_t lastPos = m_references[m_contigIdx++].RefLength+1;
		if( !(m_versionNumber == 1000 ? ReadBaseCov1000(lastPos) : ReadBaseCov0(lastPos)) ) return false;
	}
	return m_versionNumber == 1000 ? ReadBaseCov1000(position) : ReadBaseCov0(position);
}

void BbcView::StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	// RegionCoverage is used for overriding streamed coverage to new targets, e.g. for new BBC
	if( m_regionCoverage ) {
		covType = m_regionCoverage->BaseDepthOnRegion( m_contigIdx, position, fwdReads, revReads );
	}
	// invert on target status (if option set)
	covType ^= m_ontargInvert;
	// BbcCreate is for making a new BBC file from a view of a BBC file
	if( m_bbcCreate ) {
		m_bbcCreate->CollectBaseCoverage( m_contigIdx, position, fwdReads, revReads, covType );
	}
	// Typically there should be no print stream if there is a BbcCreate() but this is not prevented here
	if( m_bcStream ) {
		PrintBaseCoverage( m_contigStr, position, fwdReads, revReads, covType );
	}
}

//
// ---- Static standard print streamers of type BbcViewStream for non-binned views ----
//

void BbcView::BbcViewPrint(
	const char* contig, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	if( m_noOffTargetPositions && !(covType & 1) ) return;
	if( m_showZeroCoverage || fwdReads || revReads) {
		if( m_showContigNames ) printf( "%s\t", contig );
		BaseCoveragePrint( position, fwdReads, revReads, covType );
	}
}

void BbcView::CompactPrint(
	const char* contig, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	static const char* lastContig = NULL;
	if( m_noOffTargetPositions && !(covType & 1) ) return;
	if( m_showZeroCoverage || fwdReads || revReads) {
		if( contig != lastContig ) {
			if( m_showContigNames ) printf( ">%s\n", contig );
			lastContig = contig;
		}
		BaseCoveragePrint( position, fwdReads, revReads, covType );
	}
}

void BbcView::SamDepthPrint(
	const char* contig, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	// should not be used with on-target inversion
	// covType == 2  =>  on target as whole genome
	if( covType ) {
		printf( "%s\t%u\t%u\n", contig, position, fwdReads+revReads );
	}
}

