// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcIndex.h
 *
 *  Created on: Sep 29, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BBCINDEX_H_
#define BBCINDEX_H_

#include "api/BamAux.h"

#include <string>
#include <vector>
#include <stdint.h>
#include <cstdio>
using namespace std;


class BbcIndex
{
    public:
		BbcIndex( const string &filename );
		~BbcIndex(void);

		// Close the currently open index file and free all associated member memory
		void Close( bool keepFile = true );

		// Return the current contig index, as may be reset by a GetFilePos() call
		uint32_t GetContigIdx(void);

		// Return the indexed file position for correspond locus or 0 if not appropriate
		long GetFilePos( uint32_t contigIdx, uint32_t position, bool forceSeek = false );

		// Pass a position location at filePos of BBC of (possible) use in indexing
		void PassAnchor( uint32_t regionEndPos, long filePos );

		// Open a BCI file for read or write.
		// If test == true only the header of the file is read.
		bool Open( bool write = false, bool test = false );

		// Prepare for writing file position anchors for a new contig
		bool SetContig( uint32_t contigIdx );

		// Sets references (contigs) for reserving memory, etc. (for index writing)
		bool SetReference( BamTools::RefVector &references );

		// Return version number/string of open BCI file, or maximum version readable if none open.
		string VersionString(void);

    private:
		BamTools::RefVector m_references;
		string    m_filename;
		bool      m_write;
		FILE     *m_bcifile;
		uint16_t  m_versionNumber;
		uint32_t  m_contigIdx;
		uint32_t  m_numContigs;
		uint32_t  m_blocksize;
		uint32_t  m_nextAnchor;
		uint32_t *m_contigBlocks;
		uint32_t  m_posArraySize;
		long int *m_bbcFilePosArray;

		bool LoadAll( bool loadFilePosArray = true );
		bool LoadFilePosArray( bool optimize );
		bool LoadReference(void);
		void OptimizeFilePosArray(void);
		bool ReadFileHeader(void);
		void WriteAll(void);
};

#endif /* BBCINDEX_H_ */
