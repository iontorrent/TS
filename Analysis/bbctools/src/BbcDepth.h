// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
/*
 * BbcDepth.h
 *
 *  Created on: Nov 11, 2015
 *      Author: Guy Del Mistro
 */

#ifndef BBCDEPTH_H_
#define BBCDEPTH_H_

// Uses inheritance to override BbcView::StreamCoverage() data collection

#include "BbcView.h"

#include <map>
using namespace std;

#define BBC_DEPTH_COUNTS 8192

class BbcDepth: public BbcView
{
    public:
		BbcDepth(void);

		// Output report based on depths of coverage and some other stats collected
		// usingRange = true means the total aligned are for whole reference rather than a window
		void Report( bool usingRange = true );

        // Sets the specific depths at which numbers of bases at those depths are reported by Report()
        // Should be comma-separated list of integers depths or "".
		void SetCovAtDepths( const string &depths );

		// Configure printing formats to not include the on-target data field(s).
		void SetIgnoreOnTarget( bool ignore = true );

		// Set the ratio of fwd/total or rev/total equating to a definite strand bias
		void SetStrandBiasThreshold( double val = 0.7 );

		// Set the minimum number of base reads collected before strand biased coverage is judged
		void SetStrandBiasMinCount( uint32_t val = 10 );

		// Set the report format to be "genome report" (basically the old coverageAnalsis report)
		void SetGenomeReport( bool set = true );

		// Set the uniformity fraction of mean threshold
		void SetUniformityMeanThreshold( double val = 0.2 );

        // Print all target regions and all statistics attached to them
        void Write(
            const string &filename = "STDOUT",
        	const string &columnTitles = "read_depth,base_cov,base_cum_cov,norm_read_depth,pc_base_cum_cov" );

    private:
        bool     m_ignoreOnTarget;
        bool     m_genomeReport;
        double   m_uniformityMeanThreshold;
        double   m_strandBiasThreshold;
        uint32_t m_strandBiasMinCount;
        uint32_t m_ncovDepths;
		uint32_t m_basesCovered;
		uint64_t m_totalReads;
		uint64_t m_totalTargetReads;
		uint64_t m_biasedTargetReads;
		uint32_t m_readDepthCounts[BBC_DEPTH_COUNTS];
		vector<int> m_covAtDepth;
		map<uint32_t,uint32_t> m_readDepthMap;

		// Collects streamed base coverage for read depth analysis (virtual bbcView::StreamCoverage)
		void StreamCoverage( uint32_t position, uint32_t fwdReads, uint32_t revReads, uint32_t covType );
};

#endif /* BBCDEPTH_H_ */
