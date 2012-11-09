/*
 * Created on: 8-25-20010 Author: Keith Moulton Latest revision: $Revision: 49984
 * $ Last changed by: $AUTHOR: EDWARD_WAUGH $ Last change date: $Date: 2010-10-01
 * 14:54:43 -0400 (Fri, 01 Oct 2010) $ Copyright 2010 Life Technologies. All
 * rights reserved. Use is subject to license terms.
 */

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include <vector>
#include <iostream>

#include <samita/align/align_reader.hpp>
#include <samita/align/align_reader_util.hpp>
#include <samita/sam/bam_metadata.hpp>

using namespace std;
using namespace lifetechnologies;

/* =====================================================================================================================
 ======================================================================================================================= */

int main (int argc, char** argv)
{
	if(argc < 1) {
		cerr << "Usage: ExtractHeader input.bam" << endl;
		return EXIT_FAILURE;
	}

	AlignReader sam;
	prepareAlignReader(sam, argc - 1, argv + 1);
	
	BamHeader const &header = sam.getHeader();
	
	copy(header.getSequenceDictionary().begin(), header.getSequenceDictionary().end(), ostream_iterator<SQ>(cout));
	copy(header.getReadGroups().begin(), header.getReadGroups().end(), ostream_iterator<RG>(cout));
	copy(header.getPrograms().begin(), header.getPrograms().end(), ostream_iterator<PG>(cout));

	//copy(header.getComments().begin(), header.getComments().end(), ostream_iterator<std::string>(cout));
	for(std::vector<std::string>::const_iterator i = header.getComments().begin(); i != header.getComments().end(); ++i)
	{
	    cout << (*i) << endl; // Comments need newline
	}

	cout << " ******** End Header ************" << endl;

	BamMetadata metadata(header);   
	if(! metadata.hasMetadata() )
	{
		cerr << "Does not have extended metadata" << endl;
		return EXIT_SUCCESS;
	}
	
	for(std::vector<RG>::const_iterator rg = header.getReadGroups().begin();
		rg != header.getReadGroups().end(); ++rg)
	{
		RGExtended const & rgext = metadata.getReadGroupExtended((*rg).ID);

		cerr << "Is ECC" << rgext.EC << endl;
		cerr << "RG Record: "<< rgext.ID << endl;
		cerr << rgext;
	}
	

	return EXIT_SUCCESS;
}
