/*
 *  Created on: 06-16-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 48965 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-08-30 09:53:49 -0700 (Mon, 30 Aug 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef TEST_HPP_
#define TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <log4cxx/logger.h>

using namespace std;

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.sam");

class Test: public CppUnit::TestFixture
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
        suiteOfTests->addTest(new CppUnit::TestCaller<Test>("firstTest", &Test::firstTest));
        return suiteOfTests;
    }
    void setUp()
    {
    	LOG4CXX_INFO(g_log, "setUp");
//        m_inputBas = "sam/input.bas";
//        m_outputBas = "sam/output.bas";
//        m_numExpectedRecords = 3;
//        m_inputBam = "data/test.bam";
    }

    void tearDown()
    {
//        remove(m_outputBas.c_str());
    }

    void firstTest()
    {
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
    }

};

#endif //TEST_HPP_
