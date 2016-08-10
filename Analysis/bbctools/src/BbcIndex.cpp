// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcIndex.cpp
 *
 *  Created on: Sep 29, 2015
 *      Author: Guy Del Mistro
 */

#include "BbcIndex.h"
#include "BbcUtils.h"

#include <cstdio>
#include <sys/stat.h>

// Write V0 format if s_maxVersionNumber is set to 0
const uint16_t s_maxVersionNumber = 1000;
const uint32_t s_defaultBlocksize = 100000;

BbcIndex::BbcIndex( const string &filename )
	: m_filename(filename)
	, m_write(false)
	, m_bcifile(NULL)
	, m_versionNumber(s_maxVersionNumber)
	, m_contigIdx(0)
	, m_numContigs(0)
	, m_blocksize(s_defaultBlocksize)
	, m_nextAnchor(1)
	, m_contigBlocks(NULL)
	, m_posArraySize(0)
	, m_bbcFilePosArray(NULL)
{
}

BbcIndex::~BbcIndex()
{
	Close();
}

void BbcIndex::Close( bool keepFile )
{
	// completes file output for file write
	if( m_bcifile ) {
		if( keepFile && m_write ) WriteAll();
		fclose(m_bcifile);
		m_bcifile = NULL;
		if( !keepFile ) remove(m_filename.c_str());
	}
	// reset to initial values
	m_contigIdx = m_numContigs = m_posArraySize = 0;
	m_nextAnchor = 1;
	m_versionNumber = s_maxVersionNumber;
	m_blocksize = s_defaultBlocksize;
	delete [] m_contigBlocks;
	m_contigBlocks = NULL;
	delete [] m_bbcFilePosArray;
	m_bbcFilePosArray = NULL;
}

uint32_t BbcIndex::GetContigIdx()
{
	return m_contigIdx;
}

long BbcIndex::GetFilePos( uint32_t contigIdx, uint32_t position, bool forceSeek )
{
	// fail safes
	if( m_write || !m_bbcFilePosArray || contigIdx >= m_numContigs ) {
		return 0;
	}
	// unless forceSeek, do not reset file pointer if already close
	if( !forceSeek && contigIdx == m_contigIdx &&
		position >= m_nextAnchor && position < m_nextAnchor+m_blocksize ) {
		return 0;
	}
	if( position > 0 ) --position;
	uint32_t contigBlock = position / m_blocksize;
	uint32_t blockIdx = m_contigBlocks[contigIdx] + contigBlock;
	if( blockIdx >= m_contigBlocks[contigIdx+1] ) {
		return 0;	// out-of-bounds position set request for given contig
	}
	// NULL file pointer indicates no further coverage on this contig
	// => find the next available contig start where there is coverage
	while( !m_bbcFilePosArray[blockIdx] ) {
		if( ++contigIdx >= m_numContigs ) {
			return 0;
		}
		contigBlock = 0;
		blockIdx = m_contigBlocks[contigIdx];
	}
	m_contigIdx = contigIdx;
	m_nextAnchor = 1 + (contigBlock * m_blocksize);
	return m_bbcFilePosArray[blockIdx];
}

bool BbcIndex::Open( bool write, bool test )
{
	Close();
	m_write = write;
	if( write ) {
		m_bcifile = fopen( m_filename.c_str(), "wb" );
		return (m_bcifile != NULL);
	}
	m_bcifile = fopen( m_filename.c_str(), "rb" );
	if( !ReadFileHeader() ) return false;
	if( test ) return true;
	// LoadAll() always closes the file but leaves memory intact
	if( LoadAll() ) return true;
	Close();
	return false;
}

void BbcIndex::PassAnchor( uint32_t regionEndPos, long filePos )
{
	// Process next region end that crosses next block boundary
	// Note: No function for non-increasing values (no error check).
	// regionEndPos is 1 beyond last base, hence <= in test
	if( regionEndPos <= m_nextAnchor || !m_write ) return;
	// no need to pad skipped regions as 0'd already
	uint32_t block = (regionEndPos - 1) / m_blocksize;
	m_bbcFilePosArray[ m_contigBlocks[m_contigIdx]+block ] = filePos;
	// next anchor should be 1 after this one
	m_nextAnchor = 1 + ((block+1) * m_blocksize);
}

bool BbcIndex::SetContig( uint32_t contigIdx )
{
	// error conditions
	if( contigIdx >= m_numContigs || !m_write ) return false;
	// no backwards contig stepping allowed
	if( contigIdx < m_contigIdx ) return false;
	// no need to fill in for jumped over contigs as array already 0'd
	m_contigIdx = contigIdx;
	m_nextAnchor = 1;
	return true;
}

bool BbcIndex::SetReference( BamTools::RefVector &references )
{
	// called after Open(true) and before WriteAll() (at Close())
	// it is not allowed to open a file for writing if already open for reading
	m_numContigs = references.size();
	if( !m_bcifile || !m_numContigs || !m_write ) {
		Close(false);
		return false;
	}
	// allocate memory and sizes to be filled out prior to WriteAll() call
	delete [] m_contigBlocks;
	m_contigBlocks = new uint32_t[m_numContigs];
	m_posArraySize = 0;
	// create list of cumulative block sizes (number of file seek positions) for each contig
	for( size_t i = 0; i < m_numContigs; ++i ) {
		m_contigBlocks[i] = m_posArraySize;
		uint32_t nblock   = references[i].RefLength / m_blocksize;
		if( references[i].RefLength % m_blocksize ) ++nblock;	// round up #blocks
		m_posArraySize += nblock;
	}
	// create space for index in memory
	m_bbcFilePosArray = new long[m_posArraySize];
	memset( m_bbcFilePosArray, 0, m_posArraySize*sizeof(long) );
	return true;
}

string BbcIndex::VersionString()
{
	double vnum = (double)(m_bcifile ? m_versionNumber : s_maxVersionNumber)/1000;
	return BbcUtils::numberToString( vnum, 3 );
}

//
// --------------- Private methods ---------------
//

bool BbcIndex::LoadAll( bool loadFilePosArray ) {
	// loads all data into memory - assuming the header was first read
	bool loadOk = LoadReference();

	// LoadFilePosArray() may become as needed, since it may become more
	// efficient to use fseek() on BCI for just a single indexing.
	// But for now LoadFilePosArray() also resolves a few V0 / V1 differences.
	if( loadOk && loadFilePosArray ) {
		loadOk = LoadFilePosArray(true);
	}
	fclose(m_bcifile);
	m_bcifile = NULL;
	return loadOk;
}

bool BbcIndex::LoadFilePosArray( bool optimize )
{
	// Load all contents of a BCI indexed file pointers beyond the header.
	// This method is necessary for V0 to avoid 32bit file pointer clock overs.
	m_bbcFilePosArray = new long[m_posArraySize];
	long highWord = 0;
	uint32_t lastfpos = 0, fpos = 0;
	for( uint32_t i = 0; i < m_posArraySize; ++i ) {
		if( fread( &fpos, sizeof(uint32_t), 1, m_bcifile ) != 1 ) return false;
		if( fpos ) {
			// check for 32bit overflow
			if( fpos < lastfpos ) {
				highWord += 0x100000000;
			}
			m_bbcFilePosArray[i] = fpos | highWord;
			lastfpos = fpos;
		} else {
			m_bbcFilePosArray[i] = 0;	// retain 0's => no coverage for this block
		}
	}
	if( optimize ) OptimizeFilePosArray();
	return true;
}

bool BbcIndex::LoadReference()
{
	// Loads the contigs section of the BCI file
	if( !m_numContigs ) return false;
	delete [] m_contigBlocks;
	m_contigBlocks = new uint32_t[m_numContigs+1];
	if( fread( m_contigBlocks, sizeof(uint32_t), m_numContigs, m_bcifile ) != m_numContigs ) return false;
	if( m_versionNumber ) {
		// for V1 last contig #blocks stored in first offset
		m_posArraySize = m_contigBlocks[0];
		m_contigBlocks[0] = 0;
	} else {
		// for V0 last contig #blocks is retrieved from file size
		struct stat st;
		stat( m_filename.c_str(), &st );
		m_posArraySize = (st.st_size / sizeof(uint32_t)) - 2 - m_numContigs;
	}
	// extra contig offset for easier bounds checking (=> #blocks for last contig)
	m_contigBlocks[m_numContigs] = m_posArraySize;
	return true;
}

void BbcIndex::OptimizeFilePosArray()
{
	// Blocks with 0 coverage have NULL file pointers, which were used in script version directly
	// for counting. Retaining these would force the general indexing procedure to search forward
	// (inefficient for long jumps) or Rewind and search from the beginning (for backward access).
	// This method replaces NULL file pointers with the next valid forward pointer, except for the
	// last NULL(s) encountered. Hence any NULL then encountered indicates that there is no more
	// coverage for the whole contig so that the appropriate action can be taken.
	if( !m_bbcFilePosArray || !m_numContigs ) return;
	m_contigBlocks[0] = 0;	// in case modified for V1 output
	for( size_t i = 0; i < m_numContigs; ++i ) {
		long fpos = 0;
		int bSrt = m_contigBlocks[i];
		int bEnd = (i+1 == m_numContigs) ? m_posArraySize : m_contigBlocks[i+1];
		for( int block = bEnd-1; block >= bSrt; --block ) {
			if( m_bbcFilePosArray[block] ) {
				fpos = m_bbcFilePosArray[block];
			} else {
				m_bbcFilePosArray[block] = fpos;
			}
		}
	}
}

bool BbcIndex::ReadFileHeader()
{
	// Reads file header
	if( !m_bcifile || m_write ) return false;
	uint32_t versionKey;
	if( fread( &versionKey, sizeof(uint32_t), 1, m_bcifile) != 1 ) return false;
	if( versionKey & 0x80000000 ) {
		m_versionNumber = (uint16_t)versionKey;
		m_blocksize = s_defaultBlocksize;
	} else {
		m_versionNumber = 0;
		m_blocksize = versionKey;
	}
	if( fread( &m_numContigs, sizeof(uint32_t), 1, m_bcifile ) != 1 ) return false;
	return true;
}

void BbcIndex::WriteAll()
{
	// Primary header includes version and number of contigs
	if( !m_numContigs ) return;
	// for V0 use a fixed position block size of 100000 as used to create old index files
	m_versionNumber = s_maxVersionNumber;
	if( s_maxVersionNumber == 0 ) {
		fwrite( &m_blocksize, sizeof(uint32_t), 1, m_bcifile );
	} else {
		uint32_t versionKey = s_maxVersionNumber | 0x80000000;
		fwrite( &versionKey, sizeof(uint32_t), 1, m_bcifile );
	}
	fwrite( &m_numContigs, sizeof(uint32_t), 1, m_bcifile );

	// V1 removes need for file stat on read (by using redundant 0 first value)
	if( m_versionNumber ) m_contigBlocks[0] = m_posArraySize;
	fwrite( m_contigBlocks, sizeof(uint32_t), m_numContigs, m_bcifile );

	// (optimized) indexed file pointers array (for V1)
	if( m_versionNumber ) OptimizeFilePosArray();

	// NOTE: To save (typically ~half) file space only use low word of long array output
	// The long pointers are reconstructed on load (for consistency with V0).
	for( size_t i = 0; i < m_posArraySize; ++i ) {
		uint32_t fpos = (uint32_t)m_bbcFilePosArray[i];
		fwrite( &fpos, sizeof(uint32_t), 1, m_bcifile );
	}
}
