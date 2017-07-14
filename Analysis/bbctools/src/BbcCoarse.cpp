// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcCoarse.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: Guy Del Mistro
 */

#include "BbcCoarse.h"
#include "BbcUtils.h"

#include <cstdio>
#include <sys/stat.h>

const uint16_t s_maxVersionNumber = 1000;
const uint16_t s_minorBinSize  = 1000;		// default number of bases averaged per minor bin
const uint16_t s_minorPerMajor = 250;		// default number of minor bins per major bin

BbcCoarse::BbcCoarse( const string &filename )
	: m_filename(filename)
	, m_write(false)
	, m_cbcfile(NULL)
	, m_versionNumber(s_maxVersionNumber)
	, m_minorBinSize(s_minorBinSize)
	, m_minorPerMajor(s_minorPerMajor)
	, m_majorBinSize(s_minorPerMajor * s_minorBinSize)
	, m_contigIdx(0)
	, m_numContigs(0)
	, m_numMajorBins(0)
	, m_minorBinPackSize(0)
	, m_minorBinPackBinHead(0)
	, m_minorBinPackHead(0)
	, m_minorBinPackTail(0)
	, m_majorBinCount(0)
	, m_minorBinCount(0)
	, m_numContigBins(0)
	, m_v0_numcov(0)
	, m_contigBinIndex(NULL)
	, m_v0_minorBins(NULL)
	, m_majorBins(NULL)
	, m_minorBinPack(NULL)
{
}

BbcCoarse::~BbcCoarse()
{
	Close();
}

void BbcCoarse::Close( bool keepFile )
{
	// completes file output for file write
	if( m_cbcfile ) {
		if( keepFile && m_write ) WriteAll();
		fclose(m_cbcfile);
		m_cbcfile = NULL;
		if( !keepFile ) remove(m_filename.c_str());
	}
	// reset to initial values
	m_versionNumber = s_maxVersionNumber;
	m_contigIdx = m_numMajorBins = m_minorBinPackSize = 0;
	m_minorBinPackBinHead = m_minorBinPackHead = m_minorBinPackTail;
	m_majorBinCount = m_minorBinCount = 0;
	// free workspace memory
	delete [] m_contigBinIndex;
	m_contigBinIndex = NULL;
	delete [] m_v0_minorBins;
	m_v0_minorBins = NULL;
	delete [] m_majorBins;
	m_majorBins = NULL;
	if( m_minorBinPack ) {
		free(m_minorBinPack);
		m_minorBinPack = NULL;
	}
}

uint32_t BbcCoarse::GetMinorBinSize() {
	return m_minorBinSize;
}

uint32_t BbcCoarse::GetMajorBinSize() {
	return m_majorBinSize;
}

bool BbcCoarse::LoadAll()
{
	// Loads all of the CBC into memory for quick usage.
	// Called after SetReference().
	if( m_write || !m_numMajorBins ) return false;

	// currently knowledge of file size is required
	struct stat st;
	stat( m_filename.c_str(), &st );
	uint64_t filesize = st.st_size;
	bool sizeIssue = true;

	// For V0 this is 47Mb for hg19 regardless of coverage.
	// For V1 this can vary largely depending on coverage but is often < 500Kb.
	while(sizeIssue) {
		if( m_versionNumber < 1000 ) {
			// for V0 last contig #bins is retrieved from file size
			// file size must be whole number of integers and later match size reserved for reference
			if( filesize % sizeof(uint32_t) != 0 ) break;
			uint32_t binsize = m_numMajorBins * sizeof(uint32_t);
			m_v0_numcov = filesize / binsize;	// integers per minor bin => exactly 2 or 4
			if( (filesize % binsize) || (m_v0_numcov != 4 && m_v0_numcov != 2) ) break;
			// load whole file as one giant array
			delete [] m_v0_minorBins;
			binsize = filesize / sizeof(uint32_t);
			m_v0_minorBins = new uint32_t[binsize];
			if( fread( m_v0_minorBins, sizeof(uint32_t), binsize, m_cbcfile) != binsize ) break;
		} else {
			uint32_t nreads = 0;	// assume no major bin block
			// optionally load major bins
			if( m_minorPerMajor > 1 ) {
				// would change here if know whether CBC has on-target coverage
				nreads = m_numMajorBins*4;
				if( fread( m_majorBins, sizeof(uint64_t), nreads, m_cbcfile) != nreads ) break;
				nreads *= sizeof(uint64_t);
			}
			nreads += 2 * sizeof(uint32_t);	// header words
			if( filesize < nreads ) break;
			// the remainder of file is assumed to be whole of the minor bins
			m_minorBinPackSize = filesize - nreads;
			if( m_minorBinPack ) free(m_minorBinPack);
			m_minorBinPack = (uint8_t *)malloc( m_minorBinPackSize );
			if( fread( m_minorBinPack, 1, filesize, m_cbcfile) != m_minorBinPackSize ) break;
			// load contig offset array
			if( m_minorPerMajor == 1 && !LoadContigMinorOffsets() ) {
				cerr << "WARNING: CBC v" << (float)(m_versionNumber/1000) << " file does not align to reference.\n";
				return false;
			}
		}
		sizeIssue = false;
	}
	if( sizeIssue ) {
		cerr << "WARNING: CBC v" << (float)(m_versionNumber/1000) << " file does not have the expected size.\n";
		return false;
	}
	return true;
}

bool BbcCoarse::Open( bool write )
{
	Close();
	m_write = write;
	if( write ) {
		m_cbcfile = fopen( m_filename.c_str(), "wb" );
		m_versionNumber = s_maxVersionNumber;
		m_minorPerMajor = s_minorPerMajor;
		m_minorBinSize  = s_minorBinSize;
		m_majorBinSize  = m_minorPerMajor * m_minorBinSize;
	} else {
		m_cbcfile = fopen( m_filename.c_str(), "rb" );
		if( m_cbcfile && !ReadVersion() ) Close();
	}
	return (m_cbcfile != NULL);
}

bool BbcCoarse::PassCoverage( uint64_t fcov, uint64_t rcov, uint64_t fcovTrg, uint64_t rcovTrg )
{
	// Returns false if expected number of bins exceeded
	static uint64_t s_fcov = 0, s_rcov = 0, s_fcovTrg = 0, s_rcovTrg = 0;
	static uint8_t lastFcode = 0;
	if( m_minorPerMajor > 1 ) {
		// collected enough minor bin values to complete large bin?
		if( m_minorBinCount == m_minorPerMajor ) {
			// compact both coverage sums and minor bin offsets to major bin
			if( m_majorBinCount < m_numContigBins) {
				uint32_t idx = (m_contigBinIndex[m_contigIdx] + m_majorBinCount) << 2;	// * 4
				m_majorBins[idx]   =    (s_fcov << 8) | (m_minorBinPackBinHead & 0xFF);
				m_majorBins[++idx] =    (s_rcov << 8) | ((m_minorBinPackBinHead >> 8) & 0xFF);
				m_majorBins[++idx] = (s_fcovTrg << 8) | ((m_minorBinPackBinHead >> 16) & 0xFF);
				m_majorBins[++idx] = (s_rcovTrg << 8) | (m_minorBinPackBinHead >> 24);
			}
			// call for either first data of next bin or to finalize
			s_fcov    = fcov;
			s_rcov    = rcov;
			s_fcovTrg = fcovTrg;
			s_rcovTrg = rcovTrg;
			m_minorBinPackBinHead = m_minorBinPackTail;
			m_minorBinCount = 0;
			// called too many times before next reset by SetContig()
			if( m_majorBinCount >= m_numContigBins ) return false;
			// check if this was a call (from SetContig()) to flush buffer
			if( ++m_majorBinCount == m_numContigBins ) return true;
		} else {
			s_fcov    += fcov;
			s_rcov    += rcov;
			s_fcovTrg += fcovTrg;
			s_rcovTrg += rcovTrg;
		}
	} else if( m_minorBinCount >= m_numContigBins ) {
		// called too many times before next reset by SetContig()
		return false;
    }
	// collect data to tail of workspace...
	// 8b format code = (2b) 0-3 arrangement C | (3b) 0-7 word size B | (3b) 0-7 word size A
	// C == 0 => A = F, B = R       : F+R total target reads, no on-target (different word sizes)
	// C == 1 => not used (could be for Ft/Rt but F >= Ft, R >= Rt)
	// C == 2 => A = F & Ft, B = R & Rt  : fwd and/or rev total follow by on-target reads
	// C == 3 => A = F & R,  B = Ft & Rt : total and/or on-target fwd reads followed by same size for rev reads
	// C(0) and C(2) are better for regions with F/R strand bias; C(0) only useful for 100% bias and no on-target reads
	// C(3) is assumed to be the more general case, where the region is unbalanced for total vs. on-target reads
	uint8_t a = MinWordSize(fcov);
	uint8_t b = MinWordSize(rcov);
	uint8_t c = 0;
    if( fcovTrg | rcovTrg ) {
    	// have some on-target coverage - choice of C(3) over C(2) depends on least total word size
    	uint8_t a3 = MinWordSize(fcovTrg);
    	uint8_t b3 = MinWordSize(rcovTrg);
    	uint8_t s2a = a > a3 ? a : a3;
    	uint8_t s2b = b > b3 ? b : b3;
    	uint8_t s3a = a > b ? a : b;
    	uint8_t s3b = a3 > b3 ? a3 : b3;
    	if( s2a+s2b <= s3a+s3b ) {
    		a = s2a;
    		b = s2b;
    		c = 2;
    	} else {
    		a = s3a;
    		b = s3b;
    		c = 3;
    	}
    }
    uint8_t fcode = (c << 6) | (b << 3) | a;
    if( m_minorBinCount == 0 || fcode != lastFcode ) {
		// initiate format or force format change
    	// Note: Unlike BBC format, looking ahead to prevent wasteful fcode changes is much more
    	// complicated and mostly not worth it (few cases lose more than 1 byte).
    	m_minorBinPackHead = m_minorBinPackTail;
    	CheckMinorBinPack(2);
    	m_minorBinPack[m_minorBinPackTail++] = 1;	// number of values at fcode
    	m_minorBinPack[m_minorBinPackTail++] = lastFcode = fcode;
    } else {
    	// continue with same format - assuming regions have similar summed coverage depths
    	++m_minorBinPack[m_minorBinPackHead];	// max of 255 here limits values for s_minorPerMajor
    }
    // pack the data according to the fcode sizes and order F, R, Ft, Rt
    CheckMinorBinPack((a+b)<<1);
    PackBytes(fcov,a);
    if( c == 3 ) {
    	PackBytes(rcov,a);
    	PackBytes(fcovTrg,b);
    	PackBytes(rcovTrg,b);
    } else {
        PackBytes(rcov,b);
        if( c == 2 ) {
        	PackBytes(fcovTrg,a);
        	PackBytes(rcovTrg,b);
        }
    }
	++m_minorBinCount;
	return true;
}

bool BbcCoarse::SetContig( uint32_t contigIdx )
{
	// error conditions
	if( contigIdx >= m_numContigs || !m_write ) return false;
	// no backwards contig stepping allowed
	if( contigIdx < m_contigIdx ) return false;
	// collect and store last major bin data
	if( m_minorPerMajor > 1 && m_minorBinCount > 0 ) {
		// check if penultimate bin was encountered
		if( m_majorBinCount != m_numContigBins-1 ) return false;
		// this forces output and reset for pending major bin data - which typically sums fewer samples
		m_minorBinCount = m_minorPerMajor;
		PassCoverage(0,0,0,0);
	}
	// set number of minor bins expected for this contig
	m_contigIdx = contigIdx;
	if( m_minorPerMajor == 1 ) {
		// use m_contigBinIndex[] directly before overwritten by pointer into data
		m_numContigBins = m_contigBinIndex[m_contigIdx];
	} else {
		// use difference of major bins recorded per conig
		m_numContigBins = (m_contigIdx == m_numContigs-1) ? m_numMajorBins : m_contigBinIndex[m_contigIdx+1];
		m_numContigBins -= m_contigBinIndex[m_contigIdx];
	}
	m_majorBinCount = m_minorBinCount = 0;
	return true;
}

bool BbcCoarse::SetReference( const BamTools::RefVector &references )
{
	// called after Open(true) for read or write to allocate work space
	m_numContigs = references.size();
	if( !m_cbcfile || !m_numContigs ) {
		Close(false);
		return false;
	}
	// create contig indexing array - always be created on the fly
	delete [] m_contigBinIndex;
	m_contigBinIndex = new uint32_t[m_numContigs+1];	// one extra to avoid boundary checking
	// using array m_majorBins could waste space/performance for many tiny contigs
	// note: for reading V1 files this logic will identify this bin should not be loaded
	if( m_minorPerMajor > 1 ) {
		// pre-check contig sizes for a sufficient number of over-sized contigs
		m_numMajorBins = 0;
		for( size_t i = 0; i < m_numContigs; ++i ) {
			uint32_t nbin = references[i].RefLength / m_majorBinSize;
			m_numMajorBins += nbin;	// nbin == 0 if contig length < binsize
		}
		// reset to avoid up to N contigs longer than binsize (N = m_minorPerMajor/50 = 5)
		// i.e. requiring up to N * m_minorPerMajor extra sums w/o major bins array
		if( m_numMajorBins < m_minorPerMajor/50 ) {
			m_minorPerMajor = 1;
			m_majorBinSize = m_minorBinSize;
		}
	}
	// pre-allocate for contig offsets in to coverage data (to major bins or minor bins directly)
	if( m_versionNumber < 1000 || m_minorPerMajor > 1 ) {
		m_numMajorBins = 0;
		for( size_t i = 0; i < m_numContigs; ++i ) {
			m_contigBinIndex[i] = m_numMajorBins;
			uint32_t nbin = references[i].RefLength / m_majorBinSize;
			if( references[i].RefLength % m_majorBinSize ) ++nbin;	// round up #bins
			m_numMajorBins += nbin;
		}
		// add total bin count to the end - avoid array bound checking when using last contig
		m_contigBinIndex[m_numContigs] = m_numMajorBins;
	} else {
		// here save the number of (minor) bins per reference contig
		// - this data is replaced by direct pointers into the compressed data
		for( size_t i = 0; i < m_numContigs; ++i ) {
			m_contigBinIndex[i] = references[i].RefLength / m_minorBinSize;
			if( references[i].RefLength % m_majorBinSize ) ++m_contigBinIndex[i];
		}
		// m_minorPerMajor == 1 is a special case where major substitutes for minor bins
		m_numMajorBins = m_contigBinIndex[m_numContigs-1];
	}
	// allocate memory and sizes to be filled out or loaded (unused for V0 or V1 w/ minor contigs)
	delete [] m_majorBins;
	m_majorBins = m_minorPerMajor > 1 ? new uint64_t[m_numMajorBins*4] : NULL;
	// memory required for holding compressed data before write
	free(m_minorBinPack);
	m_minorBinPack = NULL;
	if( m_write ) {
		// sizeof(uint32_t) assumes sparse coverage: average 1 byte per targeted base coverage (or 2 for WGNM)
		// realloc() may be necessary for high coverage references, doubling storage as necessary
		m_minorBinPackSize = m_numMajorBins * m_minorPerMajor * sizeof(uint32_t);	// ~47Mb for human
		m_minorBinPack = (uint8_t *)malloc( m_minorBinPackSize );
	}
	return true;
}

bool BbcCoarse::SumMajorCoverage( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, bool addEndBin,
	uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg )
{
	// if major bin data is not loaded/used sum over minor bin range
	if( !m_majorBins ) {
		return SumMinorCoverage(
			srtContig, srtPosition, endPosition, addEndBin, fcovSum, rcovSum, fcovSumTrg, rcovSumTrg );
	}
	uint32_t a0 = (srtPosition-1) / m_majorBinSize;
	uint32_t a1 = (endPosition-1) / m_majorBinSize;
	// check if adding in partial end bin requested and valid
	if( addEndBin && a1 < m_contigBinIndex[srtContig+1]-m_contigBinIndex[srtContig] ) a1++;
	// do nothing if a major bin is not covered or data is not recorded
	if( a1 <= a0 ) return true;
	uint32_t bin = (m_contigBinIndex[srtContig] + a0) << 2;
	for( uint32_t a = a0; a < a1; ++a ) {
		fcovSum += m_majorBins[bin++] >> 8;
		rcovSum += m_majorBins[bin++] >> 8;
		fcovSumTrg += m_majorBins[bin++] >> 8;
		rcovSumTrg += m_majorBins[bin++] >> 8;
	}
	return true;
}

bool BbcCoarse::SumMajorMinorCoverage( uint32_t srtContig, uint32_t srtPosition,
	uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg )
{
	// For use to get overlap coverage when SumMajorCoverage() is effective to get bulk coverage.
	// Hence, do nothing if there is no major bin data.
	if( !m_majorBins ) return true;

	// calculate bin srtPosition falls in - no need for any work if on major bin boundary
	uint32_t b1 = ((srtPosition-1) % m_majorBinSize) / m_minorBinSize;
	if( !b1 ) return true;

	// determine offset to minor bin data given relative to major bin
	uint32_t a0 = (srtPosition-1) / m_majorBinSize;

	uint32_t bb = (m_contigBinIndex[srtContig] + a0) << 2;
	uint32_t dpos = m_majorBins[bb] & 0xFF;
	dpos |= (m_majorBins[++bb] & 0xFF) << 8;
	dpos |= (m_majorBins[++bb] & 0xFF) << 16;
	dpos |= (m_majorBins[++bb] & 0xFF) << 24;
	return ReadSum( dpos, 0, b1, fcovSum, rcovSum, fcovSumTrg, rcovSumTrg );
}

bool BbcCoarse::SumMinorCoverage( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, bool addEndBin,
	uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg )
{
	// floor minor bin indexes
	uint32_t dpos = m_contigBinIndex[srtContig];
	uint32_t b0 = (srtPosition-1) / m_minorBinSize;
	uint32_t b1 = (endPosition-1) / m_minorBinSize;
	// check if adding in partial end bin requested and necessary (i.e. not already rounded up)
	if( addEndBin && ((endPosition-1) % m_minorBinSize) ) b1++;
	// this method deals with old version
	if( m_versionNumber < 1000 ) {
		// do nothing if data not loaded
		if( !m_v0_minorBins ) {
			cerr << "ERROR: SumMinorCoverage() called without data loaded.\n";
			return false;
		}
		// data is directly memory after offset for contig
		dpos = m_v0_numcov * (dpos + b0);
		if( m_v0_numcov == 4 ) {
			while( b0++ < b1 ) {
				fcovSum += m_v0_minorBins[dpos++];
				rcovSum += m_v0_minorBins[dpos++];
				fcovSumTrg += m_v0_minorBins[dpos++];
				rcovSumTrg += m_v0_minorBins[dpos++];
			}
		} else if( m_v0_numcov == 2 ) {
			while( b0++ < b1 ) {
				fcovSum += m_v0_minorBins[dpos++];
				rcovSum += m_v0_minorBins[dpos++];
			}
		}
		return true;
	}
	// do nothing if data not loaded
	if( !m_minorBinPack ) {
		cerr << "ERROR: SumMinorCoverage() called without data loaded.\n";
		return false;
	}
	// if called with major bin data then recalculate data and start/end bin offsets
	if( m_majorBins ) {
		uint32_t a0 = (srtPosition-1) / m_majorBinSize;
		uint32_t bb = (dpos + a0) << 2;
		dpos = m_majorBins[bb] & 0xFF;
		dpos |= (m_majorBins[++bb] & 0xFF) << 8;
		dpos |= (m_majorBins[++bb] & 0xFF) << 16;
		dpos |= (m_majorBins[++bb] & 0xFF) << 24;
		b0 = ((srtPosition-1) % m_majorBinSize) / m_minorBinSize;
		b1 = ((endPosition-1) % m_majorBinSize) / m_minorBinSize;
		if( addEndBin && b1 < m_contigBinIndex[srtContig+1]-m_contigBinIndex[srtContig] ) b1++;
	}
	// collect minor bin data given starting data position, offsets to start and end bins to sum
	return ReadSum( dpos, b0, b1, fcovSum, rcovSum, fcovSumTrg, rcovSumTrg );
}

string BbcCoarse::VersionString()
{
	double vnum = (double)(m_cbcfile ? m_versionNumber : s_maxVersionNumber)/1000;
	return BbcUtils::numberToString( vnum, 3 );
}

//
// --------------- Private methods ---------------
//

bool BbcCoarse::LoadContigMinorOffsets()
{
	// strictly for small contigs compression
	// returns false if there appears to be a mismatch vs. expected bin counts
	uint32_t dataPos = 0;
	for( size_t i = 0; i < m_numContigs; ++i ) {
		int32_t numBins = (int32_t)m_contigBinIndex[i];
		if( !numBins ) return false;
		m_contigBinIndex[i] = dataPos;
		while( numBins ) {
			// see PassCoverage() for details of coding
			uint8_t rglen = m_minorBinPack[dataPos++];
			if( !rglen ) {
				return false;
			}
			uint8_t fcode = m_minorBinPack[dataPos++];
			uint8_t a = fcode & 7;
			uint8_t b = (fcode >> 3) & 7;
			uint8_t c = fcode >> 6;
			dataPos += rglen * ((a + b) * (c > 1 ? 2 : 1));
			numBins -= rglen;
			if( numBins < 0 ) return false;
		}
	}
	// for consistency last value in array indicates end of data
	if( dataPos != m_minorBinPackSize ) return false;
	m_contigBinIndex[m_numContigs] = dataPos;
	return true;
}

bool BbcCoarse::ReadSum( uint32_t dataPos, uint32_t srtBin, uint32_t endBin,
		uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg )
{
	// read and sum coverage for current bin at dataPos, skipping srtBin values up to endBin value
	uint32_t binsRead = 0;
	while( binsRead < endBin ) {
		// see PassCoverage() for details of coding
		uint8_t rglen = m_minorBinPack[dataPos++];
		// check alignment error - if this is 0 would otherwise get into endless loop
		if( !rglen || dataPos >= m_minorBinPackSize ) {
			cerr << "WARNING: CBC v" << (float)(m_versionNumber/1000) << " file does not align to reference.\n";
			return false;
		}
		uint8_t fcode = m_minorBinPack[dataPos++];
		uint8_t a = fcode & 7;
		uint8_t b = (fcode >> 3) & 7;
		uint8_t c = fcode >> 6;
		uint8_t nbytes = (a + b) * (c > 1 ? 2 : 1);
		// skip zero regions
		if( !nbytes ) {
			binsRead += rglen;
			continue;
		}
		// full skip bins of data prior to srtBin
		if( binsRead + rglen < srtBin ) {
			binsRead += rglen;
			dataPos += nbytes * rglen;
			continue;
		}
		// partial skip for overlapping region
		if( binsRead < srtBin ) {
			dataPos += nbytes * (srtBin - binsRead);
			rglen -= srtBin - binsRead;
			binsRead = srtBin;
		}
		// set unpack codes for coverage types - see PassCoverage() for details of coding
		uint8_t fBytes = a, rBytes = (c == 3) ? a : b;
		uint8_t ftBytes = (c == 3) ? b : a, rtBytes = b;
		if( c < 2 ) ftBytes = rtBytes = 0;
		// process remaining region coverage (up to endBin)
		while( rglen ) {
			fcovSum += UnpackBytes(dataPos,fBytes);
			rcovSum += UnpackBytes(dataPos,rBytes);
			fcovSumTrg += UnpackBytes(dataPos,ftBytes);
			rcovSumTrg += UnpackBytes(dataPos,rtBytes);
			if( ++binsRead >= endBin ) break;
			--rglen;
		}
	}
	return true;
}

bool BbcCoarse::ReadVersion()
{
	// Pretty much impossible that first word of coverage for 1st 1000 bases was at maximum
	// so this value is set to distinguish between versions
	uint32_t word;
	if( fread( &word, sizeof(uint32_t), 1, m_cbcfile ) != 1 ) return false;
	if( word == 0xFFFFFFFF ) {
		if( fread( &word, sizeof(uint32_t), 1, m_cbcfile ) != 1 ) return false;
		m_versionNumber = word >> 16;
		m_minorPerMajor = word & 0xFFFF;
		m_minorBinSize  = s_minorBinSize;
	} else {
		// m_minorBinSize was not stored in V0 format but 1000 was always used by coverageAnalysis
		m_versionNumber = 0;
		m_minorPerMajor = 1;
		m_minorBinSize  = 1000;
		fseek( m_cbcfile, 0, SEEK_SET );	// whole file is one major array
	}
	m_majorBinSize = m_minorPerMajor * m_minorBinSize;
	return true;
}

bool BbcCoarse::WriteAll()
{
	// write current version number
	// Note: minor bin size is assumed to be 1000 for V0 and V1 (it could be saved here in future versions)
	uint32_t word = 0xFFFFFFFF;
	fwrite( &word, sizeof(uint32_t), 1, m_cbcfile );
	word = (m_versionNumber) << 16 | m_minorPerMajor;
	fwrite( &word, sizeof(uint32_t), 1, m_cbcfile );

	// dump contents of all work space arrays to the file
	// - except m_contigBinIndex as this is always created on fly from provided references
	// - major bin size array is optional (not used where contig sizes are small)
	if( m_minorPerMajor > 1 ) {
		// a bit wasteful if there is no on-target coverage (could compact max sum cov to 6 bytes)
		// - requires knowledge that BBC contains any on-target coverage or not
		fwrite( m_majorBins, sizeof(uint64_t), m_numMajorBins*4, m_cbcfile );
	}
	// highly variable work space for minor bins
	if( m_minorBinPack ) {
		fwrite( m_minorBinPack, 1, m_minorBinPackTail, m_cbcfile );
	}
	return true;
}
