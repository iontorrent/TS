/*
 *  Created on: 10-28-2010
 *      Author: Jonathan Manning
 *
 *  Latest revision:  $Revision: 67503 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2010-10-01 14:54:43 -0400 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef metadata_TEST_HPP_
#define metadata_TEST_HPP_

#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>

#include "samita/align/align_reader.hpp"
#include "samita/sam/bam_metadata.hpp"

using namespace std;
using namespace lifetechnologies;

class MetadataTest: public CppUnit::TestFixture
{

	CPPUNIT_TEST_SUITE(MetadataTest);
	CPPUNIT_TEST(metadataMissingTest);
    //      CPPUNIT_TEST(metadataParseTest);
    //      CPPUNIT_TEST(metadataRequiredTest);
	CPPUNIT_TEST_SUITE_END();
	
public:
	
	void setUp()
	{
	}
	
	void tearDown()
	{
	}
	
	//*************************************************************
	// test basic iteration of metadata records over a known input
	//*************************************************************
	void metadataMissingTest()
	{
		const char* input = "data/test.sorted.bam";
		AlignReader sam(input);
		BamHeader & header = sam.getHeader();

		BamMetadata headerplus(header);
		//CPPUNIT_ASSERT(headerplus != NULL);
		CPPUNIT_ASSERT(!headerplus.hasMetadata()); // This input does not have extended headers

		// Parent BamHeader methods should work just fine.
		
		RG const & rg1 = headerplus.getReadGroup(1);
		const std::string rg1expected("S1");
		CPPUNIT_ASSERT_EQUAL(rg1.ID, rg1expected);
		
		RG const & rg2 = headerplus.getReadGroup(2);
		const std::string rg2expected("AJAJAJAJAJAJAJAJAJ");
		CPPUNIT_ASSERT_EQUAL(rg2.ID, rg2expected);
		
		// Fetch existing
		RGExtended const & rg1ext = headerplus.getReadGroupExtended(rg1.ID);
		CPPUNIT_ASSERT(!rg1ext.hasMetadata());
		
		// Self Upgrade
		//RGExtended rg2ext(rg2);
		//CPPUNIT_ASSERT(!rg2ext.hasMetadata());
		//CPPUNIT_ASSERT_EQUAL(rg2ext, headerplus.getReadGroupExtended(rg2.ID));
		
	}
	
};

#endif //metadata_TEST_HPP

