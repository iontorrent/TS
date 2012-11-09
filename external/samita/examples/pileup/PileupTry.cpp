/*
 * Created on: 
 * Author: Jonathan Manning
 * Latest revision: $Revision: 49984$
 * Last changed by: $AUTHOR: EDWARD_WAUGH $
 * Last change date: $Date: 2010-10-01 14:54:43 -0400 (Fri, 01 Oct 2010) $ 
 * Copyright 2010 Life Technologies. All rights reserved. Use is subject to license terms.
 */
#include <cstdio>
#include <cstdlib>

#include <samita/pileup/pileup_builder.hpp>
#include <samita/filter/filter.hpp>

using namespace std;
using namespace lifetechnologies;

int main (int argc, char * argv[])
{
	char defaultInput[] = "../../test/data/test.pileup.bam";
	char *input = defaultInput;
	if(argc > 1) {
	    input = argv[1];
	}

	cerr << "Opening: " << input << endl;
	AlignReader sam(input);
	BamHeader &header = sam.getHeader();

	typedef AlignReader::filter_iterator<StandardFilter> MyIterator;
	StandardFilter filter;
	filter.setFilteringFlags(BAM_DEF_MASK);
	PileupBuilder<MyIterator> pb (MyIterator(filter, sam.begin(), sam.end()), MyIterator(filter, sam.end(), sam.end()));
	int nPileup = 0;

	for (PileupBuilder<MyIterator>::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++)
	{
            std::cout << (*plp)->getPileupStr(header) << std::endl;
	    nPileup++;

	}

	return 1;
}
