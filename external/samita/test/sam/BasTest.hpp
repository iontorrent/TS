/*
 *  Created on: 06-16-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 76065 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-11 13:57:58 -0800 (Fri, 11 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef BAS_TEST_HPP_
#define BAS_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/sam/bas.hpp>
#include <samita/align/align_reader.hpp>

using namespace std;
using namespace lifetechnologies;

class BasTest: public CppUnit::TestFixture
{
private:
    std::string m_inputBas;
    std::string m_outputBas;
    size_t m_numExpectedRecords;
    std::string m_inputBam;
public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BasTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<BasTest>("testBasReader", &BasTest::testBasReader));
        suiteOfTests->addTest(new CppUnit::TestCaller<BasTest>("testBasWriter", &BasTest::testBasWriter));
        suiteOfTests->addTest(new CppUnit::TestCaller<BasTest>("testBasMerge", &BasTest::testBasMerge));
        return suiteOfTests;
    }
    void setUp()
    {
        m_inputBas = "sam/input.bas";
        m_outputBas = "sam/output.bas";
        m_numExpectedRecords = 3;
        m_inputBam = "data/test.bam";
    }

    void tearDown()
    {
        remove(m_outputBas.c_str());
    }

    void testBasReader()
    {
        BasReader reader;
        BasReader::iterator iter;
        BasReader::iterator end;
        size_t nRecords;

        reader.open(m_inputBas.c_str());
        iter = reader.begin();
        end = reader.end();
        nRecords = 0;
        while (iter != end)
        {
            BasRecord const& record = *iter;
            //cerr << record << endl;
            if (nRecords == 0)
            {
                CPPUNIT_ASSERT_EQUAL(std::string("S1"), record.getReadGroup());

                CPPUNIT_ASSERT_EQUAL((size_t)1, record.getTotalBases());
                CPPUNIT_ASSERT_EQUAL((size_t)2, record.getMappedBases());
                CPPUNIT_ASSERT_EQUAL((size_t)3, record.getTotalReads());
                CPPUNIT_ASSERT_EQUAL((size_t)4, record.getMappedReads());
                CPPUNIT_ASSERT_EQUAL((size_t)5, record.getMappedReadsPairedInSequencing());
                CPPUNIT_ASSERT_EQUAL((size_t)6, record.getMappedReadsProperlyPaired());

                CPPUNIT_ASSERT_EQUAL(2.0, record.getAvgQualityMappedBases());
                CPPUNIT_ASSERT_EQUAL(3.0, record.getMeanInsertSize());
                CPPUNIT_ASSERT_EQUAL(4.0, record.getSdInsertSize());
                CPPUNIT_ASSERT_EQUAL(5.0, record.getMedianInsertSize());
                CPPUNIT_ASSERT_EQUAL(6.0, record.getAdMedianInsertSize());
            }

            nRecords++;
            ++iter;
        }
        CPPUNIT_ASSERT_EQUAL(m_numExpectedRecords, nRecords);
        reader.close();
    }

    void testBasWriter()
    {
        BasWriter writer;
        BasRecord record;

        record.setMeanInsertSize(100);
        record.setSdInsertSize(3.1415);

        writer.open(m_outputBas.c_str());
        writer << bas_comment << "Test comment" << std::endl;
        writer << record << std::endl;
        writer.close();

    }

    void testBasMerge()
    {
        BasReader bas;
        bas.open(m_inputBas.c_str());
        BasReader::iterator iter = bas.begin();
        BasRecord const& basRecord = *iter;
        CPPUNIT_ASSERT_EQUAL(std::string("S1"), basRecord.getReadGroup());

        AlignReader sam(m_inputBam.c_str());
        BamHeader & hdr = sam.getHeader();
        hdr.setRGStats(basRecord.getReadGroup(), basRecord);
        // verify values from header
        RG rg = hdr.getReadGroup("S1");

        CPPUNIT_ASSERT_EQUAL(std::string("NA19240"), rg.SM);

        CPPUNIT_ASSERT(rg.Stats.empty() == false);
        CPPUNIT_ASSERT_EQUAL((size_t)1, rg.Stats.getTotalBases());
        CPPUNIT_ASSERT_EQUAL((size_t)2, rg.Stats.getMappedBases());
        CPPUNIT_ASSERT_EQUAL((size_t)3, rg.Stats.getTotalReads());
        CPPUNIT_ASSERT_EQUAL((size_t)4, rg.Stats.getMappedReads());
        CPPUNIT_ASSERT_EQUAL((size_t)5, rg.Stats.getMappedReadsPairedInSequencing());
        CPPUNIT_ASSERT_EQUAL((size_t)6, rg.Stats.getMappedReadsProperlyPaired());

        CPPUNIT_ASSERT_EQUAL(2.0, rg.Stats.getAvgQualityMappedBases());
        CPPUNIT_ASSERT_EQUAL(3.0, rg.Stats.getMeanInsertSize());
        CPPUNIT_ASSERT_EQUAL(4.0, rg.Stats.getSdInsertSize());
        CPPUNIT_ASSERT_EQUAL(5.0, rg.Stats.getMedianInsertSize());
        CPPUNIT_ASSERT_EQUAL(6.0, rg.Stats.getAdMedianInsertSize());

        bas.close();
        sam.close();
    }

};

#endif //BAS_TEST_HPP_
