/*
 *  Created on: 03-11-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <cassert>
#include <climits>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <samita/align/align_reader.hpp>
#include "lifetech/string/util.hpp"

using namespace std;
using namespace lifetechnologies;

int main (int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "Usage: performance_test input.bam [num records] [range]" << endl;
		cout << "       where <range> is of the form \"chr1:1-1000\"" << endl;
		return 0;
	}

    const char* filename = argv[1];

    size_t maxRecords = ULONG_MAX;
	if (argc == 3)
		maxRecords = atoi(argv[2]);

	const char* range = NULL;
	if (argc == 4)
		range = argv[3];


    AlignReader sam(filename);
    size_t nRecords = 0;

    if (range)
    	sam.select(range);
    AlignReader::iterator iter = sam.begin();
    AlignReader::iterator end = sam.end();
    while(iter != end) {
        ++iter;
        nRecords++;
        if ((nRecords % 1000000) == 0)
            cout << "processed: " << nRecords << endl;
        if (nRecords >= maxRecords)
        	break;
    }

    cout << "Total records : " << nRecords << endl;
    return 0;
}

