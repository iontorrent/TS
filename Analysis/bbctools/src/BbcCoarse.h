// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcCoarse.h
 *
 *  Created on: Oct 8, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BbcCoarse_H_
#define BbcCoarse_H_

#include "api/BamAux.h"

#include <cstdlib>

#include <string>
#include <vector>
#include <stdint.h>
#include <cstdio>
using namespace std;

class BbcCoarse
{
    public:
		BbcCoarse( const string &filename );
		~BbcCoarse(void);

		// Close the currently open index file and free all associated member memory
		void Close( bool keepFile = true );

		// Get small bin size (e.g. for sampling reference and building CBC)
		uint32_t GetMinorBinSize(void);

		// Get major bin size (e.g. for long range sampling reference)
		uint32_t GetMajorBinSize(void);

		// Load the whole CBC file into memory. May be called after Open() and SetReference().
		bool LoadAll(void);

		// Open a BCI file for read or write
		bool Open( bool write = false );

		// Pass coverage for next target bin, initialized by call to SetContig()
		bool PassCoverage( uint64_t fcov, uint64_t rcov, uint64_t fcovTrg, uint64_t rcovTrg );

		// Prepare for collecting summed coverage via subsequent PassCoverage() calls
		bool SetContig( uint32_t contigIdx );

		// Sets references (contigs) for reserving memory, etc. (for index writing)
		bool SetReference( const BamTools::RefVector &references );

		// Add in the major bin coverage for the locus if at least one major bin boundary is overlapped, for bins
		// spanning srtPosition up to endPosition, last bin being inclusive or exclusive given by addEndBin.
		// endPosition is +1 to the last base coverage considered.
		// No error checking performed on locus but error returned if CBC data does not align with reference.
		bool SumMajorCoverage( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, bool addEndBin,
			uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg );

		// Add in the small bin coverage for the major bin boundary closest to and up to locus.
		// No error checking performed on locus but error returned if CBC data does not align with reference.
		bool SumMajorMinorCoverage( uint32_t srtContig, uint32_t srtPosition,
			uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg );

		// Add in coverage over small bins where no major bin is defined, for bins spanning
		// srtPosition up to the bin spanning endPosition, last bin inclusive or exclusive given by addEndBin.
		// endPosition is +1 to the last base coverage considered.
		// No error checking performed on locus but error returned if CBC data does not align with reference.
		bool SumMinorCoverage( uint32_t srtContig, uint32_t srtPosition, uint32_t endPosition, bool addEndBin,
			uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg );

		// Return version number/string of open CBC file, or maximum version readable if none open.
		string VersionString(void);

    private:
		BamTools::RefVector m_references;
		string    m_filename;
		bool      m_write;
		FILE     *m_cbcfile;
		uint16_t  m_versionNumber;
		uint16_t  m_minorBinSize;
		uint16_t  m_minorPerMajor;
		uint32_t  m_majorBinSize;
		uint32_t  m_contigIdx;
		uint32_t  m_numContigs;
		uint32_t  m_numMajorBins;
		uint32_t  m_minorBinPackSize;
		uint32_t  m_minorBinPackBinHead;
		uint32_t  m_minorBinPackHead;
		uint32_t  m_minorBinPackTail;
		uint32_t  m_majorBinCount;
		uint32_t  m_minorBinCount;
		uint32_t  m_numContigBins;
		uint8_t   m_v0_numcov;

		uint32_t *m_contigBinIndex;
		uint32_t *m_v0_minorBins;
		uint64_t *m_majorBins;
		uint8_t  *m_minorBinPack;

		bool LoadContigMinorOffsets(void);

		bool ReadSum( uint32_t dataPos, uint32_t srtBin, uint32_t endBin,
				uint64_t &fcovSum, uint64_t &rcovSum, uint64_t &fcovSumTrg, uint64_t &rcovSumTrg );

		bool ReadVersion(void);

		bool WriteAll(void);

		inline uint8_t MinWordSize( uint64_t data ) {
			uint8_t minBytes = 0;
			while( data ) {
				++minBytes;
				data >>= 8;
			}
			return minBytes;
		};

		inline void PackBytes( uint64_t data, uint8_t numbytes ) {
	        while( numbytes ) {
	        	m_minorBinPack[m_minorBinPackTail++] = data & 0xFF;
	        	data >>= 8;
	        	--numbytes;
	        }
		}

		inline uint64_t UnpackBytes( uint32_t &pos, uint8_t numbytes ) {
			uint64_t data = 0;
			uint8_t shift = 0;
	        while( numbytes ) {
	        	data |= m_minorBinPack[pos++] << shift;
	        	shift += 8;
	        	--numbytes;
	        }
	        return data;
		}

		inline void CheckMinorBinPack( int siz ) {
			if( m_minorBinPackTail + siz > m_minorBinPackSize ) {
				// double memory size until enough
				while( m_minorBinPackTail + siz > m_minorBinPackSize )
					m_minorBinPackSize <<= 1;
				m_minorBinPack = (uint8_t *)realloc( m_minorBinPack, m_minorBinPackSize );
			}
		}
};

#endif /* BbcCoarse_H_ */
