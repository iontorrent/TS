// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * bbcCreate.cpp
 *
 *  Created on: Sep 16, 2015
 *      Author: Guy Del Mistro
 */

#include "BbcCreate.h"
#include <cstdlib>

const uint16_t s_versionNumber = 1000;

const uint32_t s_wordSizeToggle = 2 * sizeof(uint16_t);
const uint32_t s_recHeadSize = sizeof(uint16_t) + sizeof(uint32_t);
const uint32_t s_flushAtIndexBlockSize = 100000;

BbcCreate::BbcCreate( const BamTools::RefVector& references, uint32_t bufferSize )
	: m_references(references)
    , m_bufsize(bufferSize)
	, m_bbcfile(NULL)
{
	m_totalReads = m_contigReads = m_reads = m_wordsize = m_covtype = 0;
    m_srtPos = m_lstPos = m_curPos = m_curCov = m_curWsz = m_backWrdsz = m_backStep = 0;
    m_markAnchor = true;
    m_printOutput = m_onTargetOnly = false;
    m_newContig = s_versionNumber;			// reset after first used
    m_contigIdx = references.size() + 1;	// forces initialization for first contig
	m_buffer = (uint32_t *)malloc( 2 * m_bufsize * sizeof(uint32_t) );
}

BbcCreate::~BbcCreate()
{
	Close();
	free(m_buffer);
}

void BbcCreate::Close() {
	if( m_reads ) {
		FlushReads();
		m_totalReads += m_contigReads;
		m_reads = 0;
	}
	if( m_bbcfile ) {
		fclose(m_bbcfile);
		m_bbcfile = NULL;
	}
}

void BbcCreate::CollectBaseCoverage(
	uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	// option ignore all off-target reads
	if( m_onTargetOnly && (covType & 1) == 0 ) {
		return;
	}
	// this code is written for performance - hence numerous data shared by member variables
	// (private methods share data via private variables)
	m_curPos = position;
	m_curCov = covType & 1;
	m_curWsz = fwdReads >= revReads ? fwdReads : revReads;
	if( m_curWsz >= 65536 )    m_curWsz = 8;
	else if( m_curWsz >= 256 ) m_curWsz = 4;
	else if( m_curWsz >= 16 )  m_curWsz = 2;
	else                       m_curWsz = 1;

	uint32_t move = m_curPos - m_lstPos;
	if( contigIdx != m_contigIdx ) {
		// Flush buffer for start of new contig (forced for first contig)
		FlushReads();
		m_totalReads += m_contigReads;
		m_contigReads = 0;
		m_contigIdx = contigIdx;
		if( m_bbcfile ) {
			// leading 4 byte 0 indicates start of new contig, followed by contig number
			// - except for first time where it indicates the version number
			fwrite( &m_newContig, 1, sizeof(uint16_t), m_bbcfile );
			fwrite( &m_contigIdx, 1, sizeof(uint32_t), m_bbcfile );
			m_newContig = 0;
		} else if( m_printOutput ) {
			printf( ":\t%s\n", m_references[contigIdx].RefName.c_str() );
		}
	} else if( m_reads + move >= m_bufsize || move * m_wordsize >= s_recHeadSize ) {
		// Flush buffer if not enough space left in buffer for move or can insert new record
		FlushReads();
	} else if( m_curWsz > m_wordsize || m_covtype != m_curCov ) {
		// If word size increases or coverage type changes just insert new length/size marker
		// But first, tack any short skipped (0 coverage gap) region to previous (where less bytes)
		if( m_curWsz >= m_wordsize && move > 1 ) {
			if( m_wordsize == 1 ) {
				while( ++m_lstPos < m_curPos ) {
					m_buffer[m_reads++] = 0;
				}
			} else {
				uint32_t idx = m_reads << 1; // *2
				while( ++m_lstPos < m_curPos ) {
					m_buffer[idx++] = 0;
					m_buffer[idx++] = 0;
					++m_reads;
				}
			}
			move = 1;	// prevent 0's tacked on to start of next position
		}
		FlushReads(false);
		// Tack 0's ton to start of new region in rare case where on-target has low cov. after high cov. off-target
		m_lstPos = m_curPos - move;
	} else if( m_curWsz < m_wordsize ) {
		// see if most recent coverage could be saved more efficiently with a scale change
		// => difference in bytes is greater than 2 inserted headers == change up and down scale
		// (multiple scale changes less efficient than just one to next largest size)
		if( m_curWsz > m_backWrdsz ) m_backWrdsz = m_curWsz;
		if( (++m_backStep) * (m_wordsize - m_backWrdsz) >= s_wordSizeToggle ) {
			BackFlushReads();
		}
	} else {
	    // ignore non-contiguous drops in counts scale
		m_backWrdsz = m_backStep = 0;
	}
	// half byte coverage - significant since often lots of v.low coverage regions
	if( m_wordsize == 1 ) {
		while( ++m_lstPos < m_curPos ) {
			m_buffer[m_reads++] = 0;
		}
		m_buffer[m_reads] = fwdReads | (revReads << 4);
	} else {
		// set coverage to 0 for small gaps in depth output (that do not merit a scale change)
		uint32_t idx = m_reads << 1; // *2
		while( ++m_lstPos < m_curPos ) {
			m_buffer[idx++] = 0;
			m_buffer[idx++] = 0;
			++m_reads;
		}
		// save the pair of reads to the buffer
		m_buffer[idx] = fwdReads;
		m_buffer[++idx] = revReads;
	}
	// store coverage (read depth) for given base position
	m_contigReads += fwdReads + revReads;
	++m_reads;
}

bool BbcCreate::Open( const string &filename ) {
	Close();
	if( !m_references.size() ) {
		fprintf( stderr, "ERROR: Cannot create BBC file given empty list of reference contigs.\n");
		return false;
	}
	if( filename == "-" ) {
		// default to printing output, omitting reference header info
		m_printOutput = true;
		return true;
	}
	m_bbcfile = fopen( filename.c_str(), "wb" );
	if( !m_bbcfile ) {
		fprintf( stderr, "ERROR: Failed to open BBC file '%s' for output.\n",filename.c_str());
		return false;
	}
	// Construct and output header line to BBC file
	// Note: This header information is NOT printed for the default text streaming
	char strbuf[31], tab = '\t', eol = '\n', eos = '\0';
	size_t j = m_references.size()-1;
	for( size_t i = 0; i <= j; ++i ) {
		if( !m_references[i].RefName.size() || m_references[i].RefName == "$" ) {
			fprintf( stderr, "ERROR: Illegal reference name '%s' for BBC file creation.\n",m_references[i].RefName.c_str() );
			Close();
			return false;
		}
		fwrite( m_references[i].RefName.c_str(), 1, m_references[i].RefName.size(), m_bbcfile );
		sprintf( strbuf, "%c%d%c", tab, m_references[i].RefLength, (i < j ? eol : eos) );
		fwrite( strbuf, 1, strlen(strbuf)+(i < j ? 0 : 1), m_bbcfile );
	}
	return true;
}

void BbcCreate::SetNoOffTargetPositions( bool hide )
{
	m_onTargetOnly = hide;
}

float BbcCreate::VersionNumber() {
	return (float)s_versionNumber/1000;
}

// ---- Private methods ----

void BbcCreate::FlushReads( bool markAnchor )
{
	static const uint16_t NOP = 0x8000;
	static uint32_t s_lastAnchorPos = 0;
	if( m_reads ) {
		uint32_t ws = m_wordsize == 8 ? 3 : (m_wordsize >> 1);
		if( m_bbcfile ) {
			// if writing region spanning index boundary ensure an anchor is placed
			// - only necessary where whole index block (100K) are covered (and typically no target regions)
			if( (m_srtPos+m_reads-1)/s_flushAtIndexBlockSize > (m_srtPos-1)/s_flushAtIndexBlockSize ) {
				// this saves need for unnecessary extra anchor insertions - e.g. 2-3K for AmpliSeq Exome
				if( s_lastAnchorPos - m_srtPos > m_bufsize ) {
					m_markAnchor = true;
				}
			}
			// pack flag-pos.length.wordSizeCode.onTargetBit [+ anchor position]
			uint16_t head = (m_reads << 3) | (ws << 1) | m_covtype;
			if( m_markAnchor ) {
				if( !(ftell(m_bbcfile) & 0xFFFFFFFF) ) {
					// anchor points must not be on 32bit boundary due to conflict with indexing
					// (64bit wrap around vs. 0-coverage blocks). Insert NOP -> 0 read length region
					fwrite( &NOP, 1, sizeof(uint16_t), m_bbcfile );
				}
				fwrite( &head, 1, sizeof(uint16_t), m_bbcfile );
				fwrite( &m_srtPos, 1, sizeof(uint32_t), m_bbcfile );
				s_lastAnchorPos = m_srtPos;
			} else {
				head |= 0x8000;
				fwrite( &head, 1, sizeof(uint16_t), m_bbcfile );
			}
			// pack length of array according to word size (none if wordSize == 0)
			// to get the right packing this has to be done one element at a time
			uint32_t *ptr = m_buffer;
			ws = m_wordsize >> 1;
			if( ws ) {
				while( m_reads-- ) {
					fwrite( ptr++, ws, 1, m_bbcfile );
					fwrite( ptr++, ws, 1, m_bbcfile );
				}
			} else {
				while( m_reads-- ) {
					fwrite( ptr++, 1, 1, m_bbcfile );
				}
			}
		} else if( m_printOutput ) {
			// default option to view how coverage regions are stored
			if( m_markAnchor ) printf( "%u ", m_srtPos );
			printf( "| %u x %u (%u)\n", m_reads, m_wordsize, m_covtype );
			// just show interesting coverage regions
			if( !m_markAnchor || !markAnchor ) {
				for( uint32_t i = 0, idx = 0; i < m_reads; ++i,++idx ) {
					if( ws ) {
						printf( "F=%u\t", m_buffer[idx] );
						printf( "R=%u\n", m_buffer[++idx] );
					} else {
						printf( "F=%u\tR=%u\n", (m_buffer[idx] & 15), m_buffer[idx] >> 4 );
					}
				}
			}
		}
		m_reads = 0;
	}
	// reset for processing next region
	m_covtype  = m_curCov;
	m_wordsize = m_curWsz;
	m_srtPos   = m_curPos;
	m_lstPos   = m_curPos-1;
	m_backWrdsz = m_backStep = 0;
	m_markAnchor = markAnchor;
}

void BbcCreate::BackFlushReads()
{
	// flush back to previous cursor for down-scale change
	uint32_t backReads = m_backStep-1;	// -1 because last read processed on return
	m_reads -= backReads;				// number of reads to be flushed
	uint32_t j = m_reads << 1;			// save buffer index for retained read depths
	m_curWsz = m_backWrdsz;				// becomes the new m_wordsize
	FlushReads(false);					// lesser flush reads to position behind head
	m_reads   = backReads;				// reads left in the buffer
	m_srtPos -= backReads;				// start position for in buffer; m_lstPos still correct
	// transfer remaining reads to front of buffer
	if( m_curWsz == 1 ) {
		// scaling down 2 values to 2x half-bytes
		for( uint32_t i = 0; i < m_reads; ++i ) {
			m_buffer[i] = m_buffer[j++];
			m_buffer[i] |= m_buffer[j++] << 4;
		}
	} else {
		for( uint32_t i = 0; i < (m_reads << 1); ++i ) {
			m_buffer[i] = m_buffer[j++];
		}
	}
}

