/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <iostream>
#include <iomanip>
#include "BAMReader.h"

using namespace std;

int main(int argc, char* argv[])
{
	// Open a BAM file:
	char* bamFile  = argv[1];
	char* bamIndex = argv[2];

	BAMReader reader(bamFile, bamIndex);
	assert(reader);

	// Print out list of reference sequences, and their lengths:
	cout << "Found " << reader.numRefs() << " reference sequences:" << endl;
	for(int i=0; i<reader.numRefs(); ++i)
		cout << setw(9) << reader.refs()[i] << "    " << reader.lens()[i] << endl;

	// Print out list of reads:
	for(BAMReader::iterator i=reader.get_iterator(); i.good(); i.next())
		cout << i.qname() << endl;
}

