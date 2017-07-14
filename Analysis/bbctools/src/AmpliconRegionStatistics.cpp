// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * AmpliSeqAmpliconRegionStatistics.cpp
 *
 *  Created on: Sep 15, 2015
 *      Author: Guy Del Mistro
 */

#include "AmpliconRegionStatistics.h"
#include "BbcUtils.h"
#include <sstream>
#include <iomanip>

AmpliconRegionStatistics::AmpliconRegionStatistics( const BamTools::RefVector& references )
	: RegionCoverage(references)
	, m_statsLinkList(NULL)
	, m_ampliconReads(true)
	, m_maxUpstreamPrimerStart(30)
	, m_maxE2eEndDist(2)
	, m_sigFacCoverage(0)
	, m_regionStackSize(32)
{
	m_regionStack = (TargetRegion **)malloc( m_regionStackSize * sizeof(TargetRegion *) );
}

AmpliconRegionStatistics::~AmpliconRegionStatistics(void)
{
	while( m_statsLinkList ) {
		StatsData *stats = m_statsLinkList->next;
		delete m_statsLinkList;
		m_statsLinkList = stats;
	}
	free( m_regionStack );
}

void AmpliconRegionStatistics::SetGenericReads( bool genericReads )
{
	m_ampliconReads = !genericReads;
}

void AmpliconRegionStatistics::SetSigFacCoverage( double val )
{
	// value is slightly reduced to counter the effect of round-off accuracy using a >= comparison
	// - it was noted that and exact value comparison often misses the == test for doubles here
	// A value of 10^-7 is more than small enough for the length ratios being considered here.
	m_sigFacCoverage = val <= 0 ? 0 : val - 0.0000001;
}

void AmpliconRegionStatistics::SetMaxUpstreamPrimerStart( int32_t dist )
{
	m_maxUpstreamPrimerStart = dist < 0 ? 0 : dist;
}

void AmpliconRegionStatistics::SetMaxE2eEndDistance( int32_t dist )
{
	m_maxE2eEndDist = dist < 0 ? 0 : dist;
}

AmpliconRegionStatistics::StatsData *AmpliconRegionStatistics::GetStats( TargetRegion *region )
{
	if( region && region->extData ) {
		return (StatsData *)region->extData;
	}
	StatsData *stats = new StatsData();
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
//

// This only records data associated with base coverage

void AmpliconRegionStatistics::TrackReadsOnRegion( const BamTools::BamAlignment &aread, uint32_t endPos )
{
	// pseudo-random number generator 'seed' for resolving equivalent read assignments
	static uint16_t clockSeed = 0;
	static int32_t lastRefID = 0;
	// always reset clockSeed for new contig to allow consistency with BAM split up by contig vs. whole
	if( lastRefID != aread.RefID ) {
		clockSeed = 0;
		lastRefID = aread.RefID;
	}
	// check/set first region read overlaps
	uint32_t readSrt = aread.Position + 1;
	uint32_t readEnd = endPos ? endPos : aread.GetEndPosition();
	uint32_t covType = ReadOnRegion( aread.RefID, readSrt, readEnd );
	// maintain base method of tracking total reads
	TargetContig *contig = m_contigList[m_rcovContigIdx];
	bool isRev = aread.IsReverseStrand();
	if( isRev ) {
		++contig->revReads;
	} else {
		++contig->fwdReads;
	}
	// Tracking of reads on target
	if( covType & 1 ) {
		// iterate over all regions overlapping read...
		int32_t bestEndDist = -m_maxUpstreamPrimerStart;
		int32_t bestOverlap = 0;
		uint32_t numBestRegions = 0;
		bool haveBestEnd = false;
		for( TargetRegion *cur = m_rcovRegion; cur; cur = cur->next ) {
			if( readEnd < cur->trgSrt ) break;
			if( readSrt > m_rcovRegion->trgEnd ) continue;
			// save stats for all overlapped reads
			++(GetStats(cur)->overlaps);
			// find most likely AmpliSeq primed region of those overlapped
			// NOTE: can still be wrong for regions starting very close together, given 5' digestion uncertainty,
			// coupled with read length and digestion uncertainty at 3'
			int32_t dSrt = readSrt - cur->trgSrt;
			int32_t dEnd = cur->trgEnd - readEnd;
			int32_t endDist5p = isRev ? dEnd : dSrt;
			// for non-amplicon reads, ends are ignored and only maximum overlap is employed to distinguish target region
			if( m_ampliconReads ) {
				// always select region that is closest start before 5p primer
				if( endDist5p < 0 && endDist5p > bestEndDist ) {
					haveBestEnd = true;
					bestEndDist = endDist5p;
					bestOverlap = 0; // force record best below
				} else if( haveBestEnd && endDist5p != bestEndDist ) {
					// region is not closer primed or same distance from false priming site
					continue;
				}
			}
			// save region based on max overlap for equivalent regions
			if( dSrt < 0 ) dSrt = 0;
			if( dEnd < 0 ) dEnd = 0;
			int32_t overlap = cur->trgEnd - cur->trgSrt - dSrt - dEnd; // +1
			if( overlap >= bestOverlap ) {
				// if overlaps also match then default to region starting most 3'
				// - cannot do better w/o knowing exact priming location, or possibly using ZA tag value
				if( overlap == bestOverlap ) {
					// stack multiple equivalent solutions
					if( numBestRegions >= m_regionStackSize ) {
						// safety code - only triggered if many targets overlapping read
						m_regionStackSize <<= 1;	// *2
						m_regionStack = (TargetRegion **)realloc(
							m_regionStack, m_regionStackSize * sizeof(TargetRegion *) );
					}
				} else {
					// save new best solution - these values are the same for all equivalent solutions
					bestOverlap = overlap;
					numBestRegions = 0;
				}
				m_regionStack[numBestRegions++] = cur;
			}
		}
		// pseudo-randomly choose best region of equivalent best regions
		TargetRegion *bestRegion = m_regionStack[ clockSeed % numBestRegions ];
		m_lastRegionAssigned = bestRegion;
		bool e2e_or_cov;
		if( m_sigFacCoverage ) {
			int32_t trgLen = bestRegion->trgEnd - bestRegion->trgSrt + 1;
			e2e_or_cov = (double(bestOverlap+1)/trgLen >= m_sigFacCoverage);
		} else {
			int32_t dSrt = readSrt - bestRegion->trgSrt;
			int32_t dEnd = bestRegion->trgEnd - readEnd;
			if( dSrt < 0 ) dSrt = 0;
			if( dEnd < 0 ) dEnd = 0;
			e2e_or_cov = ((dSrt > dEnd ? dSrt : dEnd) <= m_maxE2eEndDist);
		}
		StatsData *stats = GetStats(bestRegion);
		if( isRev ) {
			++contig->revTrgReads;
			++stats->revReads;
			if( e2e_or_cov ) ++stats->rev_e2e;
		} else {
			++contig->fwdTrgReads;
			++stats->fwdReads;
			if( e2e_or_cov ) ++stats->fwd_e2e;
		}
	}
	++clockSeed;
}

// Create the statistics line per region
string AmpliconRegionStatistics::ReportRegionStatistics( TargetRegion *region )
{
	StatsData *stats = GetStats(region);
	char sep = '\t';
	ostringstream ss;
	ss << sep << stats->overlaps << sep << stats->fwd_e2e << sep << stats->rev_e2e;
	ss << sep << (stats->fwdReads + stats->revReads);
	ss << sep << stats->fwdReads << sep << stats->revReads;
	return ss.str();
}
