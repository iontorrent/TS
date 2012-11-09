/*
 *  Created on:
 *      Author: Matthew Muller
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef LTS_TEST_HPP_
#define LTS_TEST_HPP_

#include <stdexcept>
#include <stdio.h>
#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <log4cxx/logger.h>
#include "lts/lts_reader.hpp"

using namespace std;
using namespace ltstools;

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.lts");

class LtsTest: public CppUnit::TestFixture
{
private:
    std::string m_inputBas;
    std::string m_outputBas;
    size_t m_numExpectedRecords;
    std::string m_inputBam;
public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("Test");
        suiteOfTests->addTest(new CppUnit::TestCaller<LtsTest>("noNonExistingFiles", &LtsTest::noNonExistingFiles));
        suiteOfTests->addTest(new CppUnit::TestCaller<LtsTest>("noDuplicateFiles", &LtsTest::noDuplicateFiles));
        suiteOfTests->addTest(new CppUnit::TestCaller<LtsTest>("iterateOverLanes", &LtsTest::iterateOverLanes));
        suiteOfTests->addTest(new CppUnit::TestCaller<LtsTest>("readingBarcodes", &LtsTest::readingBarcodes));
        suiteOfTests->addTest(new CppUnit::TestCaller<LtsTest>("readingPanels", &LtsTest::readingPanels));
        return suiteOfTests;
    }
    void setUp()
    {
//        m_inputBas = "sam/input.bas";
//        m_outputBas = "sam/output.bas";
//        m_numExpectedRecords = 3;
//        m_inputBam = "data/test.bam";
    }

    void tearDown()
    {
//        remove(m_outputBas.c_str());
    }

    void noNonExistingFiles()
    {
    	//Lts() shouldn't work with non-existent files.
    	file_not_found_exception* ex = NULL;
    	//Temporarily redirect stderr so the expected HDF warnings don't appear on the console.
		FILE orig_stderr = *stderr;
		*stderr = *(fopen("/dev/null","w"));
    	try {
			vector<string> filenames;
			filenames.push_back("test/data/minimal_lts_barcode_0.h5");
			filenames.push_back("test/data/minimal_lts_barcode_2.h5");
			filenames.push_back("test/data/minimal_lts_barcode_1.h5");
			filenames.push_back("test/data/minimal_lts_barcode_#.h5");
			Lts lts(filenames);
    	} catch (file_not_found_exception e) {
    		ex = &e;
    	}
    	//Restore stderr
    	fclose(stderr);
		*stderr = orig_stderr;
    	CPPUNIT_ASSERT(ex != NULL);
    }

    void noDuplicateFiles()
    {
    	//Lts() shouldn't work with redundant files.
    	invalid_argument* iaex = NULL;
    	try {
			vector<string> filenames;
			filenames.push_back("test/data/minimal_lts_barcode_0.h5");
			filenames.push_back("test/data/minimal_lts_barcode_1.h5");
			filenames.push_back("test/data/minimal_lts_barcode_2.h5");
			filenames.push_back("test/data/minimal_lts_barcode_1.h5");
			Lts lts(filenames);
    	} catch (invalid_argument e) {
    		iaex = &e;
    	}
    	CPPUNIT_ASSERT(iaex != NULL);
    }

    void iterateOverLanes() {
    	vector<string> filenames;
		filenames.push_back("test/data/minimal_lts_barcode_0.h5");
		filenames.push_back("test/data/minimal_lts_barcode_1.h5");
		filenames.push_back("test/data/minimal_lts_barcode_2.h5");
		Lts lts(filenames);
		vector<Lane> lanes = lts.getLanes();
		CPPUNIT_ASSERT_EQUAL(3, (int)lanes.size());
		CPPUNIT_ASSERT_EQUAL(filenames[0], lanes[0].getFilename());
		CPPUNIT_ASSERT_EQUAL(filenames[1], lanes[1].getFilename());
		CPPUNIT_ASSERT_EQUAL(filenames[2], lanes[2].getFilename());
    }

    void readingBarcodes() {
    	vector<string> filenames;
		filenames.push_back("test/data/minimal_lts_barcode_0.h5");
		Lts lts(filenames);
		vector<string> barcodes = lts.getLanes()[0].getBarcodes();
		CPPUNIT_ASSERT_EQUAL(3, (int)barcodes.size());
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/Barcode_1"), barcodes[0]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_2"), barcodes[1]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_3"), barcodes[2]);

		filenames.clear();
		filenames.push_back("test/data/minimal_lts_no_barcode_0.h5");
		lts = Lts(filenames);
		barcodes = lts.getLanes()[0].getBarcodes();
		CPPUNIT_ASSERT_EQUAL(1, (int)barcodes.size());
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/nobarcode"), barcodes[0]);
    }

    void readingPanels() {
    	vector<string> filenames;
		filenames.push_back("test/data/minimal_lts_barcode_0.h5");
		Lts lts(filenames);
		vector<string> panels = lts.getLanes()[0].getPanels();
		CPPUNIT_ASSERT_EQUAL(9, (int)panels.size());
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/Barcode_1/0001"), panels[0]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/Barcode_1/0002"), panels[1]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/Barcode_1/0003"), panels[2]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_2/0004"), panels[3]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_2/0005"), panels[4]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_2/0006"), panels[5]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_3/0007"), panels[6]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_3/0008"), panels[7]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_b/Barcode_3/0009"), panels[8]);

		filenames.clear();
		filenames.push_back("test/data/minimal_lts_no_barcode_0.h5");
		lts = Lts(filenames);
		panels = lts.getLanes()[0].getPanels();
		CPPUNIT_ASSERT_EQUAL(3, (int)panels.size());
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/nobarcode/0001"), panels[0]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/nobarcode/0002"), panels[1]);
		CPPUNIT_ASSERT_EQUAL(std::string("/library_a/nobarcode/0003"), panels[2]);
    }


//        BasReader reader;
//        BasReader::iterator iter;
//        BasReader::iterator end;
//        size_t nRecords;
//
//        reader.open(m_inputBas.c_str());
//        iter = reader.begin();
//        end = reader.end();
//        nRecords = 0;
//        while (iter != end)
//        {
//            BasRecord const& record = *iter;
//            cerr << record << endl;
//            if (nRecords == 0)
//            {
//                CPPUNIT_ASSERT_EQUAL(std::string("S1"), record.getReadGroup());
//
//                CPPUNIT_ASSERT_EQUAL((size_t)1, record.getTotalBases());
//                CPPUNIT_ASSERT_EQUAL((size_t)2, record.getMappedBases());
//                CPPUNIT_ASSERT_EQUAL((size_t)3, record.getTotalReads());
//                CPPUNIT_ASSERT_EQUAL((size_t)4, record.getMappedReads());
//                CPPUNIT_ASSERT_EQUAL((size_t)5, record.getMappedReadsPairedInSequencing());
//                CPPUNIT_ASSERT_EQUAL((size_t)6, record.getMappedReadsProperlyPaired());
//
//                CPPUNIT_ASSERT_EQUAL(2.0, record.getAvgQualityMappedBases());
//                CPPUNIT_ASSERT_EQUAL(3.0, record.getMeanInsertSize());
//                CPPUNIT_ASSERT_EQUAL(4.0, record.getSdInsertSize());
//                CPPUNIT_ASSERT_EQUAL(5.0, record.getMedianInsertSize());
//                CPPUNIT_ASSERT_EQUAL(6.0, record.getAdMedianInsertSize());
//            }
//
//            nRecords++;
//            ++iter;
//        }
//        CPPUNIT_ASSERT_EQUAL(m_numExpectedRecords, nRecords);
//        reader.close();
//    }

};

#endif //TEST_HPP_
