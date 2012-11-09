/*
 *  Created on: 8-25-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 78915 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-25 08:30:00 -0800 (Fri, 25 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FILTER_TEST_HPP_
#define FILTER_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/common/interval.hpp>
#include <samita/align/align_reader.hpp>
#include <samita/filter/filter.hpp>
#include <samita/filter/mate_filter.hpp>
#include <lifetech/string/util.hpp>

using namespace std;
using namespace lifetechnologies;

class FilterTest: public CppUnit::TestFixture
{
    public:
        static CppUnit::Test *suite()
        {
            CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("FilterTest");

            suiteOfTests->addTest(new CppUnit::TestCaller<FilterTest>("matesFilterTest", &FilterTest::matesFilterTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<FilterTest>("pairFilterTest", &FilterTest::pairFilterTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<FilterTest>("tripleFilterTest", &FilterTest::tripleFilterTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<FilterTest>("chainFilterTest", &FilterTest::chainFilterTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<FilterTest>("standardFilterTest", &FilterTest::standardFilterTest));

            return suiteOfTests;
        }

        void setUp()
        {
        }

        void tearDown()
        {
        }

        //*************************************************************
        // test basic iteration of AlignMates records over a known input
        //   with a filter
        //*************************************************************
        void matesFilterTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 25;

            AlignMates mates;
            MateFilter filter(&mates);
            AlignReader sam(input);
            size_t nRecords = 0;

            AlignReader::filter_iterator<MateFilter> iter(filter, sam.begin(), sam.end());
            AlignReader::filter_iterator<MateFilter> end(filter, sam.end(), sam.end());

            while(iter != end)
            {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test basic iteration of records over a known input
        //   with a filter chain
        //*************************************************************
        class TestMinQualFilter : SamitaAlignFilter
        {
            public:
                TestMinQualFilter(bool *called, int q=0) : m_called(called), m_minQual(q) {}
                bool operator() (Align const &a) const
                {
                    *m_called = true;
                    int32_t mq = a.shouldHaveMate() ? a.getMapQual() : a.getQual();
                    if(mq < 0) mq = a.getQual();
                    return (mq >= m_minQual);
                }
            private:
                bool *m_called;
                int m_minQual;
        };
        class TestMaxQualFilter : SamitaAlignFilter
        {
            public:
                TestMaxQualFilter(bool *called, int q=255) : m_called(called), m_maxQual(q) {}
                bool operator() (Align const &a) const
                {
                    *m_called = true;
                    return (a.getMapQual() <= m_maxQual);
                }
            private:
                bool *m_called;
                int m_maxQual;
        };
        class TestFlagFilter : SamitaAlignFilter
        {
            public:
                TestFlagFilter(bool *called, int f=0) : m_called(called), m_flag(f) {}
                bool operator() (Align const &a) const
                {
                    *m_called = true;
                    return ((a.getFlag() & m_flag) == m_flag);
                }
            private:
                bool *m_called;
                int m_flag;
        };

        void pairFilterTest()
        {
            const char* input = "data/test.bam";
            size_t nExpectedRecords = 85;

            bool minCalled = false;
            bool maxCalled = false;

            TestMinQualFilter minFilter1(&minCalled, 0);
            TestMaxQualFilter maxFilter1(&maxCalled, 255);
            AlignReader sam(input);
            size_t nRecords = 0;

            FilterPair<TestMinQualFilter, TestMaxQualFilter> filter1(minFilter1, maxFilter1);
            AlignReader::filter_iterator< FilterPair<TestMinQualFilter, TestMaxQualFilter>  > iter1(filter1, sam.begin(), sam.end());
            AlignReader::filter_iterator< FilterPair<TestMinQualFilter, TestMaxQualFilter>  > end1(filter1, sam.end(), sam.end());

            while(iter1 != end1)
            {
                ++iter1;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);

            CPPUNIT_ASSERT(minCalled);
            CPPUNIT_ASSERT(maxCalled);

            nExpectedRecords = 0;
            nRecords = 0;

            TestMinQualFilter minFilter2(&minCalled, 256);
            TestMaxQualFilter maxFilter2(&maxCalled, 500);

            FilterPair<TestMinQualFilter, TestMaxQualFilter> filter2(minFilter2, maxFilter2);
            AlignReader::filter_iterator< FilterPair<TestMinQualFilter, TestMaxQualFilter>  > iter2(filter2, sam.begin(), sam.end());
            AlignReader::filter_iterator< FilterPair<TestMinQualFilter, TestMaxQualFilter>  > end2(filter2, sam.end(), sam.end());

            while(iter2 != end2)
            {
                ++iter2;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }
        void tripleFilterTest()
        {
            const char* input = "data/test.bam";
            size_t nExpectedRecords = 44;

            bool minCalled = false;
            bool maxCalled = false;
            bool flagCalled = false;

            TestMinQualFilter minFilter(&minCalled, 0);
            TestMaxQualFilter maxFilter(&maxCalled, 255);
            TestFlagFilter flagFilter(&flagCalled, 64);
            AlignReader sam(input);
            size_t nRecords = 0;

            FilterTriple<TestMinQualFilter, TestMaxQualFilter, TestFlagFilter> filter(minFilter, maxFilter, flagFilter);
            AlignReader::filter_iterator< FilterTriple<TestMinQualFilter, TestMaxQualFilter, TestFlagFilter>  > iter(filter, sam.begin(), sam.end());
            AlignReader::filter_iterator< FilterTriple<TestMinQualFilter, TestMaxQualFilter, TestFlagFilter>  > end(filter, sam.end(), sam.end());

            while(iter != end)
            {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);

            CPPUNIT_ASSERT(minCalled);
            CPPUNIT_ASSERT(maxCalled);
            CPPUNIT_ASSERT(flagCalled);
        }

        void chainFilterTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 40;

            bool minCalled = false;
            bool maxCalled = false;
            bool flagCalled = false;

            TestMinQualFilter minFilter(&minCalled, 0);
            TestMaxQualFilter maxFilter(&maxCalled, 255);
            TestFlagFilter flagFilter(&flagCalled, 128);
            AlignReader sam(input);
            size_t nRecords = 0;

            FilterChain chain;

            chain.add(minFilter);
            chain.add(maxFilter);
            chain.add(flagFilter);

            AlignReader::filter_iterator< FilterChain  > iter(chain, sam.begin(), sam.end());
            AlignReader::filter_iterator< FilterChain  > end(chain, sam.end(), sam.end());

            while(iter != end)
            {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);

            CPPUNIT_ASSERT(minCalled);
            CPPUNIT_ASSERT(maxCalled);
            CPPUNIT_ASSERT(flagCalled);
        }

        void standardFilterTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 55;

            AlignReader sam(input);
            size_t nRecords = 0;

            StandardFilter filter(true, true, true, true, true, 40);

            AlignReader::filter_iterator< StandardFilter  > iter(filter, sam.begin(), sam.end());
            AlignReader::filter_iterator< StandardFilter  > end(filter, sam.end(), sam.end());

            while(iter != end)
            {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }
};

#endif //FILTER_TEST_HPP_
