// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * AmpliconRegionStatistics.h
 *
 *  Created on: Sep 15, 2015
 *      Author: Guy Del Mistro
 */

#ifndef AmpliconRegionStatistics_H_
#define AmpliconRegionStatistics_H_

#include "RegionCoverage.h"

class AmpliconRegionStatistics: public RegionCoverage
{
    public:
		AmpliconRegionStatistics( const BamTools::RefVector& references );
		~AmpliconRegionStatistics(void);

		void SetGenericReads( bool genericReads = true );
		void SetSigFacCoverage( double val );
        void SetMaxUpstreamPrimerStart( int32_t dist );
        void SetMaxE2eEndDistance( int32_t dist );

	// extra data structures for data collected per target read/base coverage
    protected:
        struct StatsData {
        	uint32_t  overlaps;
        	uint32_t  fwd_e2e;	// or fwd_cov
        	uint32_t  rev_e2e;
        	uint32_t  fwdReads;
        	uint32_t  revReads;
            StatsData *next;

            StatsData()
        		: overlaps(0)
        		, fwd_e2e(0)
                , rev_e2e(0)
                , fwdReads(0)
            	, revReads(0)
            	, next(NULL) {}

            ~StatsData(void) {}
        };

        StatsData  *m_statsLinkList;

        bool     m_ampliconReads;
        int32_t  m_maxUpstreamPrimerStart;
        int32_t  m_maxE2eEndDist;
        double   m_sigFacCoverage;
        uint32_t m_regionStackSize;
        TargetRegion **m_regionStack;

        StatsData *GetStats( TargetRegion *region );

        virtual void TrackReadsOnRegion( const BamTools::BamAlignment &aread, uint32_t endPos = 0 );

        virtual string ReportRegionStatistics( RegionCoverage::TargetRegion *region );
};

#endif /* AmpliconRegionStatistics_H_ */
