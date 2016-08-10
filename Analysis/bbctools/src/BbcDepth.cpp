// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcDepth.cpp
 *
 *  Created on: Nov 11, 2015
 *      Author: Guy Del Mistro
 */

#include "BbcDepth.h"

#include "BbcUtils.h"
#include <algorithm>

BbcDepth::BbcDepth()
	: m_ignoreOnTarget(false)
	, m_genomeReport(false)
	, m_uniformityMeanThreshold(0.2)
	, m_strandBiasThreshold(0.7)
	, m_strandBiasMinCount(10)
	, m_ncovDepths(0)
	, m_basesCovered(0)
	, m_totalReads(0)
	, m_totalTargetReads(0)
	, m_biasedTargetReads(0)
{
	memset( m_readDepthCounts, 0, BBC_DEPTH_COUNTS*sizeof(uint32_t) );
}

void BbcDepth::Report( bool usingRange )
{
	// total aligned bases here is all bases, regardless of report window
	uint64_t windowSize = GetWindowSize();
	uint64_t fwdReads, revReads, fwdTrgReads, revTrgReads;
	if( !GetTotalBaseReads( fwdReads, revReads, fwdTrgReads, revTrgReads ) ) return;
	uint64_t totalBbcReads = fwdReads + revReads;
	uint64_t totalOnTargetReads = fwdTrgReads + revTrgReads;

	// scan read depths for specified coverage and uniformity depths
	uint64_t targetSize = m_ignoreOnTarget ? windowSize : m_basesCovered;
	uint64_t onTargetReads = m_ignoreOnTarget ? m_totalReads : m_totalTargetReads;
	double mrd = targetSize == 0 ? 0 : (double)onTargetReads/targetSize;
	double lambda = m_uniformityMeanThreshold * mrd;
	uint32_t mrd_low = int(lambda), mrd_high = mrd_low+1;
	uint64_t ucov_low = 0, ucov_high = 0;
	uint32_t mrd_max = mrd_high;
	uint64_t *covDepths = new uint64_t[m_ncovDepths];
	size_t firstCovDepth = 0;
	if( m_ncovDepths ) {
		if( (uint32_t)m_covAtDepth[m_ncovDepths-1] > mrd_max ) {
			mrd_max = (uint32_t)m_covAtDepth[m_ncovDepths-1];
		}
		memset( covDepths, 0, m_ncovDepths * sizeof(uint64_t) );
	}
	// set actual 0x coverage in case of very low mean read depth
	uint32_t covlen = targetSize;
	uint32_t save0x = m_readDepthCounts[0];
	bool check_ucov = true;
	m_readDepthCounts[0] += targetSize - m_basesCovered;
	for( uint32_t rd = 0; rd <= mrd_max && rd < BBC_DEPTH_COUNTS; ++rd ) {
		uint32_t ct = m_readDepthCounts[rd];
		// check for specific coverage depths
		if( rd == mrd_low ) {
			ucov_low = covlen;
		} else if( rd == mrd_high ) {
			ucov_high = covlen;
			check_ucov = false;
		}
		while( firstCovDepth < m_ncovDepths ) {
			if( (uint32_t)m_covAtDepth[firstCovDepth] > rd ) break;
			covDepths[firstCovDepth] = covlen;
			++firstCovDepth;
		}
		covlen -= ct;
	}
	// do the same for (sparse) mapped read depths - supposedly in numeric order
	for( map<uint32_t,uint32_t>::iterator it = m_readDepthMap.begin(); it != m_readDepthMap.end(); ++it ) {
		uint32_t rd = it->first;
		uint32_t ct = it->second;
		// check for specific coverage depths
		if( check_ucov ) {
			if( rd == mrd_low ) {
				ucov_low = covlen;
			} else if( rd >= mrd_high ) {
				// unless the low threshold is specifically caught both are at the current depth
				if( !ucov_low ) ucov_low = covlen;
				ucov_high = covlen;
				mrd_high = mrd_max+1;	// prevent finding again
			}
		}
		while( firstCovDepth < m_ncovDepths ) {
			if( (uint32_t)m_covAtDepth[firstCovDepth] < rd ) break;
			covDepths[firstCovDepth] = covlen;
			++firstCovDepth;
		}
		covlen -= ct;
		// exit early if collected all at-depth stats required
		if( rd > mrd_max ) break;
	}
	// linearly interpolate uniformity at uniformity mean threshold
	lambda -= mrd_low;		// = frac( m_uniformityMeanThreshold * mrd )
	double pcScale = targetSize ? (double)100/targetSize : 0;
	// for consistency, set uniformity to 100% if there is 0 coverage
	double uniformity = mrd > 0 ? pcScale * ( (1.0-lambda)*ucov_low + lambda*ucov_high ) : 100;

	// this report is basically for the coverageAnalysis plugin
	if( m_genomeReport ) {
		const char *tag = usingRange ? "Target" : "Genome";
		if( usingRange ) {
			printf( "Total aligned base reads:          %lu\n", totalBbcReads );
			printf( "Total base reads on target:        %lu\n", m_totalTargetReads );
			printf( "Bases in target regions:           %lu\n", targetSize );
			printf( "Percent base reads on target:      %s%%\n",
				BbcUtils::sigfig(totalBbcReads == 0 ? 0 : (double)100*m_totalTargetReads/totalBbcReads ).c_str() );
		} else {
			printf( "Total base reads on target:        %lu\n", totalBbcReads );
			printf( "Bases in reference genome:         %lu\n", targetSize );
		}
		printf( "Average base coverage depth:       %s\n", BbcUtils::sigfig(mrd).c_str() );
		printf( "Uniformity of base coverage:       %.2f%%\n", uniformity );
		for( size_t i = 0; i < m_ncovDepths; ++i ) {
			string sti = BbcUtils::numberToString( m_covAtDepth[i], 0 );
			int nSpc = 9 - sti.size();
			string spc( (nSpc > 0 ? nSpc : 0), ' ' );
			printf( "%s base coverage at %sx:%s%.2f%%\n", tag, sti.c_str(), spc.c_str(), pcScale*covDepths[i] );
		}
		printf( "%s bases with no strand bias:  %.2f%%\n", tag, pcScale*(targetSize-m_biasedTargetReads) );
	} else {
		// output collected stats
		const char *tag = m_ignoreOnTarget ? "Target" : "On-target";
		const char *psp = m_ignoreOnTarget ? "" : "  ";
		const char *isp = m_ignoreOnTarget ? " " : "";
		printf( "Total aligned base reads:%s            %lu\n", psp, totalBbcReads );
		printf( "Total on-target base reads:%s          %lu\n", psp, totalOnTargetReads );
		if( m_ignoreOnTarget ) {
			printf( "Bases in targeted regions:           %lu\n", targetSize );
		} else {
			printf( "On-target bases read in trg. regions:  %lu\n", targetSize );
		}
		if( usingRange ) {
			printf( "Base reads in targeted regions:%s      %lu\n", psp, m_totalReads );
			printf( "On-target reads in targeted regions:%s %lu\n", psp, m_totalTargetReads );
			printf( "Pct on-target reads in trg. regions:%s %s%%\n", psp,
				BbcUtils::sigfig(m_totalReads == 0 ? 0 : (double)100*m_totalTargetReads/m_totalReads ).c_str() );
			printf( "Pct on-target reads of total reads:%s  %s%%\n", psp,
				BbcUtils::sigfig(totalBbcReads == 0 ? 0 : (double)100*m_totalTargetReads/totalBbcReads ).c_str() );
		} else {
			printf( "Percent base reads on target:%s        %s%%\n", psp,
				BbcUtils::sigfig(totalBbcReads == 0 ? 0 : (double)100*m_totalTargetReads/totalBbcReads ).c_str() );
		}
		printf( "%s average base read depth:%s     %s\n", tag, isp, BbcUtils::sigfig(mrd).c_str() );
		printf( "%s uniformity of base coverage:%s %.2f%%\n", tag, isp, uniformity );
		for( size_t i = 0; i < m_ncovDepths; ++i ) {
			string sti = BbcUtils::numberToString( m_covAtDepth[i], 0 );
			int nSpc = (m_ignoreOnTarget ? 11 : 10) - sti.size();
			string spc( (nSpc > 0 ? nSpc : 0), ' ' );
			printf( "%s base coverage at %sx:%s%.2f%%\n", tag, sti.c_str(), spc.c_str(), pcScale*covDepths[i] );
		}
		printf( "%s bases with no strand bias:%s   %.2f%%\n", tag, isp, pcScale*(targetSize-m_biasedTargetReads) );
	}
	// clean up
	m_readDepthCounts[0] = save0x;
	delete [] covDepths;
}

void BbcDepth::SetCovAtDepths( const string &depths ) {
	m_covAtDepth = BbcUtils::stringToIntVector(depths);
	sort( m_covAtDepth.begin(), m_covAtDepth.end(), less<int>() );
	m_ncovDepths = m_covAtDepth.size();
}

void BbcDepth::SetGenomeReport( bool set )
{
	m_genomeReport = set;
}

void BbcDepth::SetIgnoreOnTarget( bool ignore )
{
	m_ignoreOnTarget = ignore;
}

void BbcDepth::SetStrandBiasThreshold( double val )
{
	if( val < 0 ) val = 0;
	m_strandBiasThreshold = val;
}

// Set the minimum number of base reads collected before strand biased coverage is judged
void BbcDepth::SetStrandBiasMinCount( uint32_t val )
{
	if( val < 1 ) val = 1;
	m_strandBiasMinCount = val;
}

void BbcDepth::SetUniformityMeanThreshold( double val ) {
	if( val < 0 ) val = 0;
	m_uniformityMeanThreshold = val;
}

void BbcDepth::Write( const string &filename, const string &columnTitles )
{
	// silent do nothing conditions
	if( columnTitles.empty() || filename.empty() ) {
		return;
	}
	// open file or set up for print
	bool tofile = false;
	FILE *fout;
	if( filename == "STDERR" ) {
		fout = stderr;
	} else if( filename == "STDOUT" || filename == "-" ) {
		fout = stdout;
	} else {
		fout = fopen( filename.c_str(), "w" );
		if( !fout ) {
			fprintf(stderr,"ERROR: Failed to open '%s' to write region coverage output.\n",filename.c_str());
			return;
		}
		tofile = true;
	}
	// add header line to output
	string headerLine = BbcUtils::replaceAll( columnTitles, ",", "\t" );
	headerLine = BbcUtils::replaceAll( headerLine, "\\\t", "," );
	if( !headerLine.empty() ) {
		fprintf( fout, "%s\n", headerLine.c_str() );
	}
	uint64_t windowSize = GetWindowSize();
	if( !windowSize ) return;

	// m_readDepthCounts[0] contains count of base explicitly covered at 0 reads
	// - reset to actual base coverage at 0x given full window size
	uint32_t save0x = m_readDepthCounts[0];
	m_readDepthCounts[0] += windowSize - m_basesCovered;
	// output coverage data
	double norm = (double)windowSize/m_totalReads;	// 1 / mean base read depth
	double pcCC = (double)100.0 / windowSize;
	for( unsigned int rd = 0; rd < BBC_DEPTH_COUNTS; ++rd ) {
		uint32_t ct = m_readDepthCounts[rd];
		if( !ct ) continue;
		fprintf( fout, "%u\t%u\t%lu\t%.4f\t%.2f\n", rd, ct, windowSize, norm*rd, pcCC*windowSize );
		windowSize -= ct;
	}
	// do the same for (sparse) mapped read depths - supposedly in numeric order
	for( map<uint32_t,uint32_t>::iterator it = m_readDepthMap.begin(); it != m_readDepthMap.end(); ++it ) {
		uint32_t rd = it->first;
		uint32_t ct = it->second;
		fprintf( fout, "%u\t%u\t%lu\t%.4f\t%.2f\n", rd, ct, windowSize, norm*rd, pcCC*windowSize );
		windowSize -= ct;
	}
	// restore entry 0x value just in case this method is called again
	m_readDepthCounts[0] = save0x;
	if( tofile ) fclose(fout);
}

void BbcDepth::StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType )
{
	// Below a set read depth (BBC_DEPTH_COUNTS) coverage is collected in arrays for performance.
	// At or above this read depth is it kept in a map() to avoid large sparse array use.
	// Note: Explicit 0 base coverage is collected here so should be subtracted from m_basesCovered
	uint32_t readDepth = fwdReads + revReads;
	if( m_ignoreOnTarget || (covType & 1) ) {
		if( readDepth < BBC_DEPTH_COUNTS ) {
			++m_readDepthCounts[readDepth];
		} else {
			++m_readDepthMap[readDepth];
		}
		//m_totalTargetReads += readDepth;
		if( readDepth >= m_strandBiasMinCount ) {
			double sfq = (double)(fwdReads > revReads ? fwdReads : revReads) / readDepth;
			if( sfq > m_strandBiasThreshold ) ++m_biasedTargetReads;
		}
		++m_basesCovered;			// includes positions passed with no reads (readDepth == 0)
	}
	if( covType & 1 ) {
		m_totalTargetReads += readDepth;
	}
	m_totalReads += readDepth;	// includes off-target bases in region
}

