// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * RegionStatistics.h
 *
 *  Created on: Sep 11, 2015
 *      Author: Guy Del Mistro
 */

#ifndef REGIONSTATISTICS_H_
#define REGIONSTATISTICS_H_

#include "RegionCoverage.h"

class RegionStatistics: public RegionCoverage
{
    public:
		RegionStatistics( const BamTools::RefVector& references );
		~RegionStatistics(void);

	// extra data structures for data collected per target read/base coverage
    protected:
        struct StatsData {
        	uint32_t   fwdReads;
        	uint32_t   revReads;
        	uint32_t   basesCovered;
        	uint32_t   uncov5p;
        	uint32_t   uncov3p;
            StatsData *next;

            StatsData( int regionLen = 0 )
        		: fwdReads(0)
        		, revReads(0)
                , basesCovered(0)
                , uncov5p(regionLen)
            	, uncov3p(regionLen)
            	, next(NULL) {}

            ~StatsData(void) {}
        };

        StatsData  *m_statsLinkList;

        StatsData *GetStats( TargetRegion *region );

        virtual uint32_t BaseDepthOnRegion(
        	uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads );

        virtual string ReportRegionStatistics( RegionCoverage::TargetRegion *region );
};

#endif /* REGIONSTATISTICS_H_ */
