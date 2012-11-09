/*
 * Copyright (c) 2011 Life Technologies Corporation. All rights reserved.
 */

/*
 * BeadIDReadFilter.h
 *
 *  Created on: Feb 25, 2011
 *      Author: kerrs1
 */

/*! \class BeadIDReadFilter
 *  \desc Predicate Class used as Read Filter for pulling out named reads
 */

#ifndef BEADIDREADFILTER_H_
#define BEADIDREADFILTER_H_

#include <fstream>
#include <string>
#include <set>
#include <samita/common/types.hpp>

namespace lifetechnologies {

class BeadIDReadFilter {
private:
	std::set<std::string> m_lookup; // Read Group ID
public:
	BeadIDReadFilter() {}
	virtual ~BeadIDReadFilter() {}
    BeadIDReadFilter( const std::set<std::string> & beadids )
      : m_lookup(beadids) {};
	BeadIDReadFilter( const char * beadFileName ) {
		std::ifstream beadFile;
		std::string line;
		beadFile.open( beadFileName, std::ios::in );
		while( getline( beadFile, line ) )
			m_lookup.insert( line );
		beadFile.close();
	}
    bool operator() ( Align const &a ) const {
    	return m_lookup.find(a.getName()) != m_lookup.end();
    }
    bool operator() ( std::string beadID ) const {
    	return m_lookup.find(beadID) != m_lookup.end();
    }
};

}

#endif /* BEADIDREADFILTER_H_ */
