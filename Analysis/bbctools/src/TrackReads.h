// Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved.
/*
 * TrackReads.h
 *
 *  Created on: Sep 8, 2016
 *      Author: Guy Del Mistro
 */

// The TrackReads class is defined with access to the BAM read and target region data
// so that this read tracking output can be expanded on later.

#ifndef TRACKREADS_H_
#define TRACKREADS_H_

#include "RegionCoverage.h"

class TrackReads
{
    public:
		TrackReads( const string& filename, const RegionCoverage* regions = NULL );
		virtual ~TrackReads(void);

		void Write( const BamTools::BamAlignment &aread, uint32_t endPos );

    private:
		const string& m_filename;
		const RegionCoverage* m_regions;
		FILE  *m_outfile;
};

#endif /* TRACKREADS_H_ */
