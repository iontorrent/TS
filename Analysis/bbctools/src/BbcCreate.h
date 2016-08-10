// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * bbcCreate.h
 *
 *  Created on: Sep 16, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BBCCREATE_H_
#define BBCCREATE_H_

#include "api/BamAux.h"

#include <string>
#include <stdint.h>
#include <cstdio>
using namespace std;

class BbcCreate
{
    public:
		BbcCreate( const BamTools::RefVector& references, uint32_t bufferSize = 4096 );
		~BbcCreate(void);

		void Close(void);

		void CollectBaseCoverage(
			uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType );

		bool Open( const string &filename );

		void SetNoOffTargetPositions( bool hide = true );

		float VersionNumber(void);

    private:
		const BamTools::RefVector &m_references;
		uint32_t m_bufsize;
		uint32_t m_totalReads;
		uint32_t m_contigReads;
		uint32_t m_contigIdx;
		uint32_t m_reads;
		uint32_t m_wordsize;
		uint32_t m_covtype;
		uint32_t m_srtPos;
		uint32_t m_lstPos;
		uint32_t m_curWsz;
		uint32_t m_curPos;
		uint32_t m_curCov;
		uint32_t m_backWrdsz;
		uint32_t m_backStep;
		uint16_t m_newContig;
		bool m_onTargetOnly;
		bool m_markAnchor;
		bool m_printOutput;


		uint32_t *m_buffer;
		FILE *m_bbcfile;

		void FlushReads( bool markPosition = true );
		void BackFlushReads(void);
};

#endif /* BBCCREATE_H_ */
