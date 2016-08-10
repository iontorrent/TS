// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * baseCoverage.h
 *
 *  Created on: Sep 4, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BASECOVERAGE_H_
#define BASECOVERAGE_H_

#include "api/BamAlignment.h"
#include "RegionCoverage.h"
#include "BbcCreate.h"
#include "BbcView.h"

#include <stdint.h>
using namespace std;

class BaseCoverage
{
    public:
		BaseCoverage( const BamTools::RefVector& references );
		~BaseCoverage(void);

		int  AddAlignment( BamTools::BamAlignment &aread, int32_t endPosition = 0 );
		void Flush(void);

		//
		void SetBbcCreate( BbcCreate *bbcCreate );

		void SetBbcView( BbcView *bbcView );

		void SetRegionCoverage( RegionCoverage *regionCoverage );

		// Set flag to invert on/off target base coverage status.
		// NOTE: This is a streaming operation so care should be taken that option is not
		// also (re)inverted with BbcView().
		void SetInvertOnTarget( bool invert = true );

    private:
		const BamTools::RefVector &m_references;
		const char *m_contigStr;
		uint32_t  m_bufSize;
		uint32_t  m_bufHead;
		uint32_t  m_bufTail;
		uint32_t  m_bufMask;
		uint32_t  m_contigIdx;
		uint32_t  m_lastSrt;
		uint32_t  m_lastEnd;
		uint32_t  m_ontargInvert;
		uint32_t *m_fwdDepth;
		uint32_t *m_revDepth;

		RegionCoverage *m_regionCoverage;
		BbcCreate      *m_bbcCreate;
		BbcView        *m_bbcView;

		static const uint32_t MAXBUFSIZE = 1<<24;

    private:
		void CheckBufferSize( uint32_t readLen );
		void FlushBuffer( uint32_t contigIdx, uint32_t readStart );
		void StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads );
};

#endif /* BASECOVERAGE_H_ */
