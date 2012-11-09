/*
 *  Created on: 04-12-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 78915 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-25 08:30:00 -0800 (Fri, 25 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef GFF_TEST_HPP_
#define GFF_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/gff/gff.hpp>

using namespace std;
using namespace lifetechnologies;

class GffTest: public CppUnit::TestFixture
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
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("GffTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<GffTest>("testGffReader", &GffTest::testGffReader));
        suiteOfTests->addTest(new CppUnit::TestCaller<GffTest>("testGffWriter", &GffTest::testGffWriter));
        suiteOfTests->addTest(new CppUnit::TestCaller<GffTest>("testMissingFile", &GffTest::testMissingFile));
        suiteOfTests->addTest(new CppUnit::TestCaller<GffTest>("testInvalidFile", &GffTest::testInvalidFile));
        return suiteOfTests;
    }
    void setUp()
    {
        m_uncompressedInputFilename = "gff/input.gff";
        m_compressedInputFilename = "gff/input.gff.gz";
        m_uncompressedOutputFilename = "gff/output.gff";
        m_compressedOutputFilename = "gff/output.gff.gz";
        m_invalidFilename = "invalid.gff";
        m_missingFilename = "missing.gff";
        m_numExpectedRecords = 3;

        //create the invalid file
        std::ofstream inv(m_invalidFilename.c_str());
        inv << "ain't no gff data in this file" << std::endl;
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
        GffReader reader;

        if (reader.open(m_missingFilename.c_str()))
        {
            reader.close();
            CPPUNIT_ASSERT(false);
        }
    }

    void testInvalidFile()
    {
        GffReader reader(m_invalidFilename.c_str());
        CPPUNIT_ASSERT_THROW(reader.begin(), invalid_input_record);
        reader.close();
    }

    void testGffReader()
    {
        GffReader reader;
        GffReader::const_iterator iter;
        GffReader::const_iterator end;
        size_t nRecords;

        // test an uncompressed file
        reader.open(m_uncompressedInputFilename.c_str());
        iter = reader.begin();
        end = reader.end();
        nRecords = 0;
        while (iter != end)
        {
            GffFeature const& record = *iter;
            std::stringstream oss;
            oss << record << endl;
            nRecords++;
            ++iter;
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

    void testGffWriter()
    {
        GffWriter writer;
        GffFeature record;

        record.setSequence("foo");
        record.setSource("src");
        record.setType("feat1");
        record.setStart(1);
        record.setEnd(100);
        record.setScore(3.1415);

        // test uncompressed
        writer.open(m_uncompressedOutputFilename.c_str());
        writer << gff_comment << "Test comment" << std::endl;
        writer << record << std::endl;
        writer.close();

        // test uncompressed
        writer.open(m_compressedOutputFilename.c_str(), true);
        writer << gff_comment << "Test comment" << std::endl;
        writer << record << std::endl;
        writer.close();
    }
};

#endif //GFF_TEST_HPP_
