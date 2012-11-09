/*
 *  Created on: 04-19-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 76065 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-11 13:57:58 -0800 (Fri, 11 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FASTQ_TEST_HPP_
#define FASTQ_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/fastq/fastq.hpp>

using namespace std;
using namespace lifetechnologies;

class FastqTest: public CppUnit::TestFixture
{
private:
    std::string m_uncompressedInputFilename;
    std::string m_compressedInputFilename;
    std::string m_uncompressedOutputFilename;
    std::string m_compressedOutputFilename;
    std::string m_invalidFilename;
    std::string m_missingFilename;
    size_t m_numExpectedRecords;
public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("FastqTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<FastqTest>("testFastqReader", &FastqTest::testFastqReader));
        suiteOfTests->addTest(new CppUnit::TestCaller<FastqTest>("testFastqWriter", &FastqTest::testFastqWriter));
        suiteOfTests->addTest(new CppUnit::TestCaller<FastqTest>("testMissingFile", &FastqTest::testMissingFile));
        suiteOfTests->addTest(new CppUnit::TestCaller<FastqTest>("testInvalidFile", &FastqTest::testInvalidFile));
        return suiteOfTests;
    }
    void setUp()
    {
        m_uncompressedInputFilename = "fastq/input.fastq";
        m_compressedInputFilename = "fastq/input.fastq.gz";
        m_uncompressedOutputFilename = "fastq/output.fastq";
        m_compressedOutputFilename = "fastq/output.fastq.gz";
        m_invalidFilename = "invalid.fastq";
        m_missingFilename = "missing.fastq";
        m_numExpectedRecords = 4;

        //create the invalid file
        std::ofstream inv(m_invalidFilename.c_str());
        inv << "ain't no fastq data in this file" << std::endl;
        inv.close();
        // make sure the missing file does not exist
        remove(m_missingFilename.c_str());
    }

    void tearDown()
    {
        remove(m_uncompressedOutputFilename.c_str());
        remove(m_compressedOutputFilename.c_str());
        remove(m_invalidFilename.c_str());
    }

    void testMissingFile()
    {
        FastqReader reader;

        if (reader.open(m_missingFilename.c_str()))
        {
            reader.close();
            CPPUNIT_ASSERT(false);
        }
    }

    void testInvalidFile()
    {
        FastqReader reader(m_invalidFilename.c_str());
        CPPUNIT_ASSERT_THROW(reader.begin(), invalid_input_record);
        reader.close();
    }

    void testFastqReader()
    {
        FastqReader reader;
        FastqReader::iterator iter;
        FastqReader::iterator end;
        size_t nRecords;

        // test an uncompressed file
        reader.open(m_uncompressedInputFilename.c_str());
        iter = reader.begin();
        end = reader.end();
        nRecords = 0;
        while (iter != end)
        {
            FastqRecord const& record = *iter;
            //cerr << record << endl;
            nRecords++;
            ++iter;
            if (nRecords == 1)
            {
                QualityValueArray const& qvs = record.getQvArray();
                for (size_t i=0; i<qvs.size(); i++)
                {
                    CPPUNIT_ASSERT_EQUAL( (uint8_t)i, qvs[i]);
                }
            }
        }
        CPPUNIT_ASSERT_EQUAL(m_numExpectedRecords, nRecords);
        reader.close();

        // test a compressed file
        reader.open(m_compressedInputFilename.c_str());
        iter = reader.begin();
        end = reader.end();
        nRecords = 0;
        while (iter != end)
        {
            nRecords++;
            ++iter;
        }
        CPPUNIT_ASSERT_EQUAL(m_numExpectedRecords, nRecords);
        reader.close();
    }

    void testFastqWriter()
    {
        FastqWriter writer;
        FastqRecord record;

        record.setId("1_1_1");
        record.setDescription("test read");
        record.setSequence("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        record.setQvs("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`");

        // test uncompressed
        writer.open(m_uncompressedOutputFilename.c_str());
        writer << fastq_comment << "Test comment" << std::endl;
        writer << record << std::endl;
        writer.close();

        // test uncompressed
        writer.open(m_compressedOutputFilename.c_str(), true);
        writer << fastq_comment << "Test comment" << std::endl;
        writer << record << std::endl;
        writer.close();
    }
};

#endif //FASTQ_TEST_HPP_
