/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */

/*
 * BeadIDReadFilter.cpp
 *
 *  Created on: Feb 25, 2011
 *      Author: kerrs1
 */

#include <string>

#include <samita/common/types.hpp>
#include <samita/align/align_reader.hpp>
#include <samita/align/align_reader_util.hpp>
#include <samita/align/align_writer.hpp>

#include "BeadIDReadFilter.h"

using namespace lifetechnologies;
using namespace std;

int main( int argc, char *argv[] ) {
	BeadIDReadFilter beadIdReadFilter( argv[1] );
	AlignReader alignReader;
	prepareAlignReader( alignReader, 1, &argv[2] );
	AlignWriter alignWriter( argv[3], alignReader.getHeader() );
	float p = -1;
	if( argc > 4 )
		p = atof( argv[4] );
	int inList = 0, totalReads = 0, filteredReads = 0;
	if( p < 1 )
		for( AlignReader::const_iterator alignIter = alignReader.begin();
				alignIter != alignReader.end(); ++alignIter, ++totalReads ) {
			if( totalReads % 100000 == 0 )
				cout << totalReads << " records read." << endl;
			if( beadIdReadFilter( alignIter->getName() ) ) {
				++inList;
				if( p <= 0  ||  RAND_MAX * p < rand() ) {
					++filteredReads;
					alignWriter.write( *alignIter );
				}
			}
		}
	cout << "Total reads: " << totalReads << endl;
	cout << "Number of reads in list: " << inList << endl;
	cout << "Number of reads in filter output: " << filteredReads << endl;
}
