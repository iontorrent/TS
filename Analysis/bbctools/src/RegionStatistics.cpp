// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * RegionStatistics.cpp
 *
 *  Created on: Sep 11, 2015
 *      Author: Guy Del Mistro
 */

#include "RegionStatistics.h"
#include "BbcUtils.h"
#include <sstream>
#include <iomanip>

RegionStatistics::RegionStatistics( const BamTools::RefVector& references )
	: RegionCoverage(references)
	, m_statsLinkList(NULL)
{
}

RegionStatistics::~RegionStatistics(void)
{
	while( m_statsLinkList ) {
		StatsData *stats = m_statsLinkList->next;
		delete m_statsLinkList;
		m_statsLinkList = stats;
	}
}

RegionStatistics::StatsData *RegionStatistics::GetStats( TargetRegion *region )
{
	if( region && region->extData ) {
		return (StatsData *)region->extData;
	}
	StatsData *stats = new StatsData( region->trgEnd - region->trgSrt + 1 );
	if( region ) {
		region->extData = (void *)stats;
	}
	// the link list is for delete
	stats->next = m_statsLinkList;
	return m_statsLinkList = stats;
}

//
// Note: Since these derived methods create data objects they cannot be const
// Therefore to match prototypes neither can the base methods

uint32_t RegionStatistics::BaseDepthOnRegion(
	uint32_t contigIdx, uint32_t position, uint32_t fwdReads, uint32_t revReads )
{
	int totReads = fwdReads+revReads;
	if( !totReads ) return 0;
	uint32_t covType = BaseOnRegion( contigIdx, position );
	if( covType & 1 ) {
		// record coverage for ALL regions overlapping base position
		for( TargetRegion *cur = m_bcovRegion; cur; cur = cur->next ) {
			if( position < cur->trgSrt ) break;
			if( position > cur->trgEnd ) continue;
			AddCovAtDepth( cur, totReads );

			// this collects all the stats given region definition and current (on target) position data
			StatsData *stats = GetStats(cur);
			++stats->basesCovered;
			if( position - cur->trgSrt < stats->uncov5p ) {
				stats->uncov5p = position - cur->trgSrt;
			}
			if( cur->trgEnd - position < stats->uncov3p ) {
				stats->uncov3p = cur->trgEnd - position;
			}
			stats->fwdReads += fwdReads;
			stats->revReads += revReads;
		}
	}
	return covType;
}

// Create the statistics line per region
string RegionStatistics::ReportRegionStatistics( TargetRegion *region )
{
	StatsData *stats = GetStats(region);
	uint32_t regLength = region->trgEnd - region->trgSrt + 1;
	char sep = '\t';
	ostringstream ss;
	ss << sep << stats->basesCovered << sep << stats->uncov5p << sep << stats->uncov3p;
	ss << sep << setprecision(3) << fixed << float(stats->fwdReads+stats->revReads)/regLength;
	ss << sep << stats->fwdReads << sep << stats->revReads;
	return ss.str();
}

