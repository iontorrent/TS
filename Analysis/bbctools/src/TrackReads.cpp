// Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved.
/*
 * TrackReads.cpp
 *
 *  Created on: Sep 8, 2016
 *      Author: Guy Del Mistro
 */

#include "TrackReads.h"

TrackReads::TrackReads( const string& filename, const RegionCoverage* regions )
	: m_filename(filename)
	, m_regions(regions)
	, m_outfile(NULL)
{
	if( !filename.empty() && filename != "-" ) {
		m_outfile = fopen( m_filename.c_str(), "w" );
		if( !m_outfile ) throw std::runtime_error("TrackReads(): Failed to open file for write: "+m_filename);
	}
}

TrackReads::~TrackReads()
{
	if( m_outfile ) fclose(m_outfile);
}

void TrackReads::Write( const BamTools::BamAlignment &aread, uint32_t endPos )
{
	uint32_t rend = endPos ? endPos : aread.GetEndPosition();
	uint32_t rlen = rend - aread.Position;
	FILE *out = m_outfile ? m_outfile : stdout;
	if( m_regions ) {
		fprintf( out, "%u\t%u\n",m_regions->GetLastReadOnRegionIdx(),rlen);
	} else {
		// Default output if no target regions resource was provided
		fprintf( out, "%u\n",rlen);
	}
}

