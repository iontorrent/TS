/*
 *  Created on: 04-15-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef REFERENCE_TEST_HPP_
#define REFERENCE_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/common/interval.hpp>
#include <samita/reference/reference.hpp>

using namespace std;
using namespace lifetechnologies;

class ReferenceTest: public CppUnit::TestFixture
{
private:
    string m_uncompressedFilename;
    string m_compressedFilename;
    size_t m_numExpectedContigs;
    string m_lastContigName;
    size_t m_lastContigLength;
    size_t m_lastContigIndex;
    char m_lastContigFirstBase;
    char m_lastContigTenthBase;
    char m_lastContigLastBase;
    string m_queryStr;
    string m_queryContigName;
    size_t m_queryContigBegin;
    size_t m_queryContigEnd;
    size_t m_queryContigLength;
    string m_queryResult;
    string m_getResult;

public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ReferenceTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<ReferenceTest>("testUncompressedReferenceSequenceReader",
                &ReferenceTest::testUncompressedReferenceSequenceReader));
        suiteOfTests->addTest(new CppUnit::TestCaller<ReferenceTest>("testCompressedReferenceSequenceReader",
                &ReferenceTest::testCompressedReferenceSequenceReader));
        return suiteOfTests;
    }

    void setUp()
    {
        m_uncompressedFilename = "reference/reference.fasta";
        m_compressedFilename = "reference/reference.fasta.rz";
        m_numExpectedContigs = 5;
        m_lastContigIndex = 5;
        m_lastContigName = "chr5";
        m_lastContigLength = 5567;
        m_lastContigFirstBase = 'A';
        m_lastContigTenthBase = 'T';
        m_lastContigLastBase = 'G';
        m_queryContigName = "chr2";
        m_queryContigBegin = 10;
        m_queryContigEnd = 20;
        m_getResult = "GGCGACCAGC";
        m_queryContigLength = m_queryContigEnd - m_queryContigBegin + 1;
        stringstream sstrm;
        sstrm << m_queryContigName << ":" << m_queryContigBegin << "-" << m_queryContigEnd;
        m_queryStr = sstrm.str();
        m_queryResult = "GATCAACCGCC";
    }

    void tearDown()
    {
        string uncompressedIndexFilename = m_uncompressedFilename + ".fai";
        remove(uncompressedIndexFilename.c_str());
        string compressedIndexFilename = m_compressedFilename + ".fai";
        remove(compressedIndexFilename.c_str());
    }

    void testUncompressedReferenceSequenceReader()
    {
        testReferenceSequenceReader(m_uncompressedFilename);
        testReferenceSequenceReader(m_compressedFilename);
    }

    void testCompressedReferenceSequenceReader()
    {

    }

    void testReferenceSequenceReader(string const& filename)
    {
        ReferenceSequenceReader reader;

        reader.open(filename.c_str());

        ReferenceSequenceReader::iterator iter = reader.begin();
        ReferenceSequenceReader::iterator end = reader.end();
        size_t nContigs = 0;

        while (iter != end)
        {
            ReferenceSequence const& refseq = *iter;
            nContigs++;
            ++iter;
            if (nContigs == m_lastContigIndex)
            {
                CPPUNIT_ASSERT_EQUAL(m_lastContigName, refseq.getName());
                CPPUNIT_ASSERT_EQUAL(m_lastContigIndex, refseq.getContigIndex());
                CPPUNIT_ASSERT_EQUAL(m_lastContigFirstBase, refseq[1]);
                CPPUNIT_ASSERT_EQUAL(m_lastContigTenthBase, refseq[10]);
                string subseq = refseq.get(20,30);
                CPPUNIT_ASSERT_EQUAL(m_getResult, subseq);
                CPPUNIT_ASSERT_EQUAL(m_lastContigLastBase, refseq[refseq.getLength()]);
                CPPUNIT_ASSERT_THROW(refseq[0], reference_sequence_index_out_of_bounds);
                CPPUNIT_ASSERT_THROW(refseq[refseq.getLength()+1], reference_sequence_index_out_of_bounds);
            }
        }
        CPPUNIT_ASSERT_EQUAL(m_numExpectedContigs, nContigs);

        // test overloaded selects
        ReferenceSequence const& refseq1 = reader.getSequence(m_queryStr.c_str());
        CPPUNIT_ASSERT_EQUAL(m_queryResult, std::string(refseq1.getBases()));
        CPPUNIT_ASSERT_EQUAL(m_queryContigName, refseq1.getName());
        CPPUNIT_ASSERT_EQUAL(m_queryContigLength, refseq1.getLength());

        ReferenceSequence const& refseq2 = reader.getSequence(m_queryContigName.c_str(), m_queryContigBegin, m_queryContigEnd);
        CPPUNIT_ASSERT_EQUAL(m_queryResult, std::string(refseq2.getBases()));
        CPPUNIT_ASSERT_EQUAL(m_queryContigName, refseq2.getName());
        CPPUNIT_ASSERT_EQUAL(m_queryContigLength, refseq2.getLength());

        SequenceInterval interval(m_queryContigName.c_str(), m_queryContigBegin, m_queryContigEnd);
        ReferenceSequence const& refseq3 = reader.getSequence(interval);
        CPPUNIT_ASSERT_EQUAL(m_queryResult, std::string(refseq3.getBases()));
        CPPUNIT_ASSERT_EQUAL(m_queryContigName, refseq3.getName());
        CPPUNIT_ASSERT_EQUAL(m_queryContigLength, refseq3.getLength());

        reader.close();

    }
};

#endif //REFERENCE_TEST_HPP_
