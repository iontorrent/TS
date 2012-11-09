/*
 *  Created on: 04-19-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 77364 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-17 12:53:34 -0800 (Thu, 17 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef INTERVAL_TEST_HPP_
#define INTERVAL_TEST_HPP_

#include <boost/lexical_cast.hpp>
#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/common/interval.hpp>

using namespace std;
using namespace lifetechnologies;

class IntervalTest: public CppUnit::TestFixture
{
private:

public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("IntervalTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<IntervalTest>("testGeneral",
                &IntervalTest::testGeneral));
        suiteOfTests->addTest(new CppUnit::TestCaller<IntervalTest>("testBisect",
                &IntervalTest::testBisect));
        suiteOfTests->addTest(new CppUnit::TestCaller<IntervalTest>("testAbuts",
                &IntervalTest::testAbuts));
        suiteOfTests->addTest(new CppUnit::TestCaller<IntervalTest>("testIntersect",
                &IntervalTest::testIntersect));
        suiteOfTests->addTest(new CppUnit::TestCaller<IntervalTest>("testParse",
                &IntervalTest::testParse));
        return suiteOfTests;
    }

    void setUp()
    {
    }

    void tearDown()
    {
    }

    void testGeneral()
    {
        SequenceInterval int1("foo", 1, 100);
        SequenceInterval int2("foo", 1, 100);
        SequenceInterval int3("foo", 10, 1000);
        SequenceInterval int4("bar", 1, 100);

        // check equality
        CPPUNIT_ASSERT(int1 == int1);
        CPPUNIT_ASSERT(int1 == int2);
        CPPUNIT_ASSERT(!(int1 == int3));
        CPPUNIT_ASSERT(!(int1 == int4));

        // check methods
        CPPUNIT_ASSERT_EQUAL(10, int3.getStart());
        CPPUNIT_ASSERT_EQUAL(1000, int3.getEnd());
        CPPUNIT_ASSERT_EQUAL((size_t)(1000-10+1), int3.getLength());
        CPPUNIT_ASSERT_EQUAL(std::string("bar"), int4.getSequence());
    }

    void testBisect()
    {
        SequenceInterval int1("foo", 100, 1000);
        SequenceInterval int2 = int1.split(1);

        CPPUNIT_ASSERT(int1.abuts(int2));
        CPPUNIT_ASSERT_EQUAL(100, int1.getStart());
        CPPUNIT_ASSERT_EQUAL(550, int1.getEnd());
        CPPUNIT_ASSERT_EQUAL(551, int2.getStart());
        CPPUNIT_ASSERT_EQUAL(1000, int2.getEnd());

        SequenceInterval int3("bar", 16999, 116999);
        SequenceInterval int4 = int3.split();

        CPPUNIT_ASSERT(int3.abuts(int4));
        CPPUNIT_ASSERT_EQUAL(16999, int3.getStart());
        CPPUNIT_ASSERT_EQUAL(65536, int3.getEnd());
        CPPUNIT_ASSERT_EQUAL(65536+1, int4.getStart());
        CPPUNIT_ASSERT_EQUAL(116999, int4.getEnd());
    }

    void testAbuts()
    {
        SequenceInterval int1("foo", 1, 100);
        SequenceInterval int2("foo", 1, 100);
        SequenceInterval int3("foo", 100, 1000);
        SequenceInterval int4("foo", 101, 1000);
        SequenceInterval int5("bar", 101, 1000);

        CPPUNIT_ASSERT(int1.abuts(int2) == false);
        CPPUNIT_ASSERT(int1.abuts(int3) == false);
        CPPUNIT_ASSERT(int1.abuts(int4) == true);
        CPPUNIT_ASSERT(int1.abuts(int5) == false);
    }

    void testIntersect()
    {
        SequenceInterval int1("foo", 1, 100);
        SequenceInterval int2("foo", 80, 1000);

        SequenceInterval int3(int1);
        SequenceInterval int4(int2);

        SequenceInterval int5(int1);
        SequenceInterval int6("foo", 101, 1000);

        SequenceInterval int7("foo", 200, 1000);

        CPPUNIT_ASSERT(int1.intersects(int2));

        Interval int12 = int1.intersect(int2);
        CPPUNIT_ASSERT_EQUAL(80, int12.getStart());
        CPPUNIT_ASSERT_EQUAL(100, int12.getEnd());

        int3 += int4;
        CPPUNIT_ASSERT_EQUAL(1, int3.getStart());
        CPPUNIT_ASSERT_EQUAL(1000, int3.getEnd());

        Interval tint = int3 + int4;
        CPPUNIT_ASSERT_EQUAL(1, tint.getStart());
        CPPUNIT_ASSERT_EQUAL(1000, tint.getEnd());

        int5 += int6;
        CPPUNIT_ASSERT_EQUAL(1, int5.getStart());
        CPPUNIT_ASSERT_EQUAL(1000, int5.getEnd());

        CPPUNIT_ASSERT(int1.intersects(int7, 100));
    }

    void testParse()
    {
        string seq;
        int begin;
        int end;

        seq = "";
        begin = -1;
        end = -1;
        SequenceInterval::parse("chr1:123-456", seq, begin, end);
        CPPUNIT_ASSERT_EQUAL(std::string("chr1"), seq);
        CPPUNIT_ASSERT_EQUAL(begin, 123);
        CPPUNIT_ASSERT_EQUAL(end, 456);

        seq = "";
        begin = -1;
        end = -1;
        SequenceInterval::parse("chr1:123", seq, begin, end);
        CPPUNIT_ASSERT_EQUAL(std::string("chr1"), seq);
        CPPUNIT_ASSERT_EQUAL(123, begin);
        CPPUNIT_ASSERT_EQUAL(-1, end);

        seq = "";
        begin = -1;
        end = -1;
        SequenceInterval::parse("chr1", seq, begin, end);
        CPPUNIT_ASSERT_EQUAL(std::string("chr1"), seq);
        CPPUNIT_ASSERT_EQUAL(-1, begin);
        CPPUNIT_ASSERT_EQUAL(-1, end);
        CPPUNIT_ASSERT_THROW(SequenceInterval::parse("chr1:q123_456", seq, begin, end), boost::bad_lexical_cast);
    }
};

#endif //INTERVAL_TEST_HPP_
