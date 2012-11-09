/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */
#include <samita/align/align_reader.hpp>
#include <log4cxx/logger.h>

#include "samita/common/BedValidator.hpp"

using namespace lifetechnologies;

int main(int argc, char *argv[])
{
	string bedFnIn;
	string validationFile;

	if ( argc < 3 )
	{
		cout << "Please provide BED file name and BAM validation data file name\n";
		return 1;
	}
	
	bedFnIn = argv[1];
	validationFile = argv[2];

	BedValidator bedV( bedFnIn );
	AlignReader bamReader( validationFile.c_str() );
//	bedV.setExpectedChromDataSource( bamReader );
	vector<BamHeader> vecBamHeaders = vector<BamHeader>( 3, bamReader.getHeader() );
	vector<string>    vecBamFiles   = vector<string>( 3, validationFile );
	bedV.setExpectedChromDataSource( vecBamHeaders, vecBamFiles );

	bool isValid = bedV.validate();
	string msg = "BED file validation result: ";
	msg += ( isValid ? "SUCCESS." : "FAIL.");
	cout << msg << endl;
	LOG4CXX_INFO(g_log, msg);

	return 0;
}
