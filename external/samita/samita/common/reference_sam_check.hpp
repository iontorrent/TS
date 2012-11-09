/*
 * Created on:  02-07-2011
 *    Authors:
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef REFERENCE_SAM_CHECK_H_
#define REFERENCE_SAM_CHECK_H_

// query
// source

#include <samita/reference/reference.hpp>
#include <samita/align/align_reader.hpp>

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <log4cxx/logger.h>
#include <samita/common/types.hpp>
#include <boost/lexical_cast.hpp>

namespace lifetechnologies
{

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.reference_sam_check");

bool sourceHasAllNamesLengths(
		const BamHeader &header,
		const ReferenceSequenceReader &sourceRef,
		const string fileName ) {
	//This checks all the SQ's in the header and makes sure all the names contained
	//in the BAM header are also contained in the source reference reader and that
	//the lengths are the same for each name.

	bool isConsistent = true;
	std::vector < SQ > const &seqDict = header.getSequenceDictionary();
	std::vector < SQ >::const_iterator seqDictItr = seqDict.begin();
	for( ; seqDictItr != seqDict.end(); seqDictItr++ ) {
		long sourceLength = sourceRef.getLength( seqDictItr->SN );
		if ( sourceLength < 0 ) {
		    isConsistent = false;
			string msg = "sequence name " + seqDictItr->SN + " was not found in reference file";
			msg += (fileName.length() > 0) ? ("; BAM file(s): " + fileName) : "";
			LOG4CXX_ERROR(g_log, msg);
		} else if ( seqDictItr->LN != sourceLength ) {
			isConsistent = false;
			string msg = "for sequence name " + seqDictItr->SN
					+ "; the lengths do not match.  Sequence length: "
					+ boost::lexical_cast<string>( seqDictItr->LN )
					+ "; Reference length: "
					+ boost::lexical_cast<string>( sourceLength );
			msg += (fileName.length() > 0) ? ("; BAM file(s): " + fileName) : "";
			LOG4CXX_ERROR(g_log, msg);
		}
	}
	return isConsistent;
}

bool isHeaderConsistentWithReference(
		const BamHeader &header,
		const ReferenceSequenceReader &refReader,
		const string fileName = "") {
	bool isConsistent = sourceHasAllNamesLengths( header, refReader, fileName );
	string fileNameLog = (fileName.length() > 0) ? ("; BAM file(s): " + fileName) : "";
	if(! isConsistent) {
		LOG4CXX_ERROR( g_log, "Error: input-file header is not consistent with reference" + fileNameLog );
	} else {
		LOG4CXX_INFO( g_log, "Input-file header is consistent with reference" + fileNameLog );
	}
	return isConsistent;
}

bool isHeaderConsistentWithReference(
		const vector<BamHeader> &vecHeaders,
		const ReferenceSequenceReader &refReader,
		const vector<string> &fileNames = vector<string>()) {
	vector<string> myFileNames;
	if ( fileNames.size() == vecHeaders.size())
		myFileNames = vector<string>(fileNames);
	else
		myFileNames = vector<string>(vecHeaders.size(),"");
	vector<BamHeader>::const_iterator headerIter = vecHeaders.begin();
	vector<string>::const_iterator fileNameIter = myFileNames.begin();
	bool allConsistent = true;
	for ( ; headerIter != vecHeaders.end(); ++headerIter, ++fileNameIter ) {
		// we need to save the return in the temporary variable isConsistent
		// otherwise the loop exits as soon as allConsistent turns false!
		bool isConsistent = isHeaderConsistentWithReference( *headerIter, refReader, *fileNameIter );
		allConsistent = allConsistent && isConsistent;
	}
	return allConsistent;
}

bool isHeaderConsistentWithReference(
		AlignReader &alignReader,
		const ReferenceSequenceReader &refReader ) {
    size_t numBamReaders = alignReader.getNumBamReaders();
    if ( numBamReaders == 0 ) {
		LOG4CXX_ERROR( g_log, "Error: no BamReaders in AlignReader." );
		return false;
    }
    // combine the BAM file-names into one string
	string fileNames = alignReader.getBamReader(0).getFilename();
    for ( size_t bamId = 1; bamId < numBamReaders; bamId++ )
    	fileNames += "," + alignReader.getBamReader(bamId).getFilename();
	return isHeaderConsistentWithReference( alignReader.getHeader(), refReader, fileNames );
}

}

#endif
