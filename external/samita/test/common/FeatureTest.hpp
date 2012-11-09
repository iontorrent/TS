/*
 *  Created on: 9-8-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 77367 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-17 13:03:01 -0800 (Thu, 17 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FEATURE_TEST_HPP_
#define FEATURE_TEST_HPP_

#include <stdexcept>
#include <boost/lexical_cast.hpp>
#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/common/feature.hpp>

using namespace std;
using namespace lifetechnologies;

class FeatureTest: public CppUnit::TestFixture
{
private:

public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("FeatureTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<FeatureTest>("testIntersect",
                &FeatureTest::testIntersect));
        suiteOfTests->addTest(new CppUnit::TestCaller<FeatureTest>("testBadIntersect",
                &FeatureTest::testBadIntersect));
        return suiteOfTests;
    }

    void setUp()
    {
    }

    void tearDown()
    {
    }

    void testBadIntersect()
    {
        Feature a;
        a.setStart(10);
        a.setEnd(100);

        Feature b;
        b.setStart(200);
        b.setEnd(300);

        CPPUNIT_ASSERT_THROW(a.intersect(b), std::invalid_argument);
    }

    void testIntersect()
    {
        Feature a;
        a.setStart(10);
        a.setEnd(100);
        a.setAttribute("attr1", "val1");
        a.setAttribute("attr2", "val2a");


        Feature b;
        b.setStart(90);
        b.setEnd(200);
        a.setAttribute("attr1", "val1");
        a.setAttribute("attr2", "val2b");
        b.setAttribute("attr3", "val3");
        b.setAttribute("attr4", "val4");

        Feature c = a.intersect(b);

        FeatureAttributeMap const& attrs = c.getAttributes();
        CPPUNIT_ASSERT_EQUAL((size_t)4, attrs.size());

        FeatureAttributeMap::const_iterator attrs_iter = attrs.begin();
        FeatureAttributeMap::const_iterator attrs_end = attrs.end();
        size_t totalValues = 0;
        while (attrs_iter != attrs_end)
        {
            std::string const& name = attrs_iter->first;
            FeatureAttributeValues const& values = attrs_iter->second;
            size_t nValues = values.size();
            totalValues += nValues;

            if (name == "attr1")
                CPPUNIT_ASSERT_EQUAL((size_t)1, nValues);
            else if (name == "attr2")
                CPPUNIT_ASSERT_EQUAL((size_t)2, nValues);
            else if (name == "attr3")
                CPPUNIT_ASSERT_EQUAL((size_t)1, nValues);
            else if (name == "attr4")
                CPPUNIT_ASSERT_EQUAL((size_t)1, nValues);
            ++attrs_iter;
        }
        CPPUNIT_ASSERT_EQUAL((size_t)5, totalValues);
    }

};

#endif //FEATURE_TEST_HPP_
