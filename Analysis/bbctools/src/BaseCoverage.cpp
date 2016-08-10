// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BaseCoverage.cpp
 *
 *  Created on: Sep 4, 2015
 *      Author: Guy Del Mistro
 */

#include "BaseCoverage.h"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
using namespace std;

BaseCoverage::BaseCoverage( const BamTools::RefVector& references )
    : m_references(references)
	, m_bufSize(1<<12) // 4096
	, m_bufHead(0)
	, m_bufTail(0)
	, m_contigIdx(0)
	, m_lastSrt(0)
    , m_lastEnd(0)
	, m_ontargInvert(0)
	, m_regionCoverage(NULL)
	, m_bbcCreate(NULL)
	, m_bbcView(NULL)
{
	m_fwdDepth = (uint32_t *)calloc(m_bufSize,sizeof(uint32_t));
	m_revDepth = (uint32_t *)calloc(m_bufSize,sizeof(uint32_t));
	m_bufMask = m_bufSize-1;
	m_contigStr = m_references[m_contigIdx].RefName.c_str();
}

BaseCoverage::~BaseCoverage(void)
{
	free(m_fwdDepth);
	free(m_revDepth);
}

// Add an alignment to the coverage collector.
// Optional argument endPosition specifies the (effective) end of the read and save a call to GetEndPosition().
// Return the alignment end alignment position, 0 to ignore read or -1 for error.
int BaseCoverage::AddAlignment( BamTools::BamAlignment &aread, int32_t endPosition )
{
	// Sanity checks for usage and to ensure consistent reference coverage
	if( aread.RefID < (int32_t)m_contigIdx ) {
		//throw runtime_error("Illegal call for reference contig out of order.");
		return -1;
	}
    int32_t srt = aread.Position;
    int32_t end = endPosition ? endPosition : aread.GetEndPosition();
	if( aread.RefID == (int32_t)m_contigIdx && srt < (int32_t)m_lastSrt ) {
		//throw runtime_error("Illegal call for read start out of order.");
		return -1;
	}
	// silently ignore or correct reads beyond ends of reference (shouldn't happen)
	if( srt < 0 ) {
		//throw runtime_error("Read start < 1.");
		srt = 0;
	} else if( srt >= m_references[m_contigIdx].RefLength ) {
		//throw runtime_error("Read start beyond length of reference contig.");
		return 0;
	}
	if( end > m_references[m_contigIdx].RefLength ) {
		//throw runtime_error("Read end beyond length of reference contig.");
		end = m_references[m_contigIdx].RefLength;
	}
	// update coverage buffer for new read
	FlushBuffer( aread.RefID, srt );
	CheckBufferSize( end-srt );

    // add to base reads over covered segments using CIGAR data
	uint32_t *baseDepth = aread.IsReverseStrand() ? m_revDepth : m_fwdDepth;
    vector<BamTools::CigarOp>::const_iterator cigarIt = aread.CigarData.begin();
    vector<BamTools::CigarOp>::const_iterator cigarEnd = aread.CigarData.end();
    uint32_t bufCurs = m_bufHead;
    for( end = srt; cigarIt != cigarEnd; ++cigarIt ) {
        const BamTools::CigarOp& op = (*cigarIt);
        switch( op.Type ) {
            // increase end position for coverage chars [MXN=]
            case BamTools::Constants::BAM_CIGAR_MATCH_CHAR    :
            case BamTools::Constants::BAM_CIGAR_MISMATCH_CHAR :
            case BamTools::Constants::BAM_CIGAR_REFSKIP_CHAR  :
            case BamTools::Constants::BAM_CIGAR_SEQMATCH_CHAR :
                end += op.Length;
                break;
            // segments of coverage separated by deletions char [D]
            case BamTools::Constants::BAM_CIGAR_DEL_CHAR      :
            	while( srt++ < end ) {
            		++baseDepth[ bufCurs & m_bufMask ];
            		++bufCurs;
             	}
            	bufCurs += op.Length;
            	srt = (end += op.Length);
                break;
            // other CIGAR chars not affecting coverage [SIPH]
            // - assumes padded not used in alignment length (and HS clips not internal)
            default :
                break;
        }
    }
    // complete coverage for alignments (not ending in deletion)
	while( srt++ < end ) {
		++baseDepth[ bufCurs & m_bufMask ];
		++bufCurs;
	}
    // track longest alignment in buffer
    if( end > (int32_t)m_lastEnd ) {
    	m_lastEnd = (uint32_t)end;
    	m_bufTail = bufCurs;
    }
	//if( end != aread.GetEndPosition() ) throw runtime_error("Read end != GetEndPosition().");
    return end;
}

// Call to ensure last coverage region is processed
void BaseCoverage::Flush() {
	FlushBuffer( m_contigIdx, m_lastEnd+1 );
}

//
// ---- Private methods ----
//

void BaseCoverage::CheckBufferSize( uint32_t readLen )
{
	// it is expected that all reads will be below the 4K buffer size
	if( readLen <= m_bufSize ) return;
	if( readLen > MAXBUFSIZE )
		throw runtime_error("Read length exceeds maximum buffer size!");
	uint32_t osz = m_bufSize;
	while( (m_bufSize <<= 1) < readLen );
	m_fwdDepth = (uint32_t *)realloc( m_fwdDepth, m_bufSize*sizeof(uint32_t) );
	m_revDepth = (uint32_t *)realloc( m_revDepth, m_bufSize*sizeof(uint32_t) );
	for( size_t i = osz; i < m_bufSize; ++ i ) {
		m_fwdDepth[i] = m_revDepth[i] = 0;
	}
	// this simple modulo utility is why code m_bufSize and MAXBUFSIZE are powers of 2
	m_bufMask = m_bufSize-1;
}

void BaseCoverage::FlushBuffer( uint32_t contigIdx, uint32_t readStart )
{
	bool flushAll = (contigIdx != m_contigIdx || readStart > m_lastEnd);
	int32_t nflush = flushAll ? m_bufTail-m_bufHead : readStart-m_lastSrt;
	while( nflush-- ) {
		// pre-increment of position accounts for 0 base of BAM position coordinates
		// note: coding by incrementing array pointers showed no appreciable difference in performance here
		StreamCoverage( ++m_lastSrt, m_fwdDepth[m_bufHead], m_revDepth[m_bufHead] );
		m_fwdDepth[m_bufHead] = m_revDepth[m_bufHead] = 0;
		// handle wrap-around buffer
		if( ++m_bufHead == m_bufSize ) {
			m_bufTail -= m_bufSize;
			m_bufHead = 0;
		}
	}
	if( flushAll ) {
		if( contigIdx != m_contigIdx ) {
			m_contigStr = m_references[ m_contigIdx = contigIdx ].RefName.c_str();
		}
		m_bufTail = m_bufHead = 0;
		m_lastSrt = m_lastEnd = readStart;
	}
}

void BaseCoverage::SetBbcCreate( BbcCreate *bbcCreate ) {
	m_bbcCreate = bbcCreate;
}

void BaseCoverage::SetBbcView( BbcView *bbcView ) {
	// currently for access to PrintBaseCoverage() only
	m_bbcView = bbcView;
}

void BaseCoverage::SetInvertOnTarget( bool invert )
{
	m_ontargInvert = invert ? 1 : 0;
}

void BaseCoverage::SetRegionCoverage( RegionCoverage *regionCoverage ) {
	m_regionCoverage = regionCoverage;
}

void BaseCoverage::StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads )
{
	// m_bbcView methods could used here, in which case BaseCoverage should be derived from BbcView
	// Note: covType == 2  =>  on target for whole genome
	uint32_t covType = m_regionCoverage ?
		m_regionCoverage->BaseDepthOnRegion( m_contigIdx, position, fwdReads, revReads ) : 2;

	// invert on target status (if option set)
	covType ^= m_ontargInvert;
	if( m_bbcCreate ) {
		m_bbcCreate->CollectBaseCoverage( m_contigIdx, position, fwdReads, revReads, covType );
	}
	if( m_bbcView ) {
		m_bbcView->PrintBaseCoverage( m_contigStr, position, fwdReads, revReads, covType );
	}
}

