/*        File: IntervalLatticeTest.hpp
 *  Created on: Jul 27, 2010
 *      Author: Caleb J Kennedy (caleb.kennedy@lifetech.com)
 * 
 *  Latest revision:  $Revision: $
 *  Last changed by:  $Author: kennedcj $
 *  Last change date: $Date: $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include "interval_lattice.hpp"
#include "feature.hpp"
#include <string>
#include <map>

using namespace lifetechnologies;

typedef Print<Interval> PRINT;

class IntervalLatticeTest : public CppUnit::TestFixture
{
  public:

    static CppUnit::Test *suite()
    {
      CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("IntervalLatticeTest");

      suiteOfTests->addTest(new CppUnit::TestCaller<IntervalLatticeTest>("partTest", &IntervalLatticeTest::partTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<IntervalLatticeTest>("insertTest", &IntervalLatticeTest::insertTest));

      return suiteOfTests;
    }

    void setUp()
    {

    }

    void tearDown()
    {

    }

    void partTest()
    {
    	Interval n("", 0, 0), x("", 1, 2), y("", 3, 5), z("", 0, 4);

    	CPPUNIT_ASSERT(Interval::Contains(n, x) && !Interval::Contains(x, n));
    	CPPUNIT_ASSERT(contains(x, z) && !contains(y, z) && !contains(x, y));
    }

    void insertTest()
    {
    	// Interval x("chr1", 2, 3), y("chr1", 4, 5), z("chr1", 1, 4), w("chr1", 2, 10);
    	Feature x, y, z, w;

    	x.setSequence("chr1");
    	x.setInterval(2, 3);
    	x.setSource("whatever");

    	y.setSequence("chr1");
    	y.setInterval(4, 5);

    	z.setSequence("chr1");
    	z.setInterval(1, 4);

    	w.setSequence("chr1");
    	w.setInterval(2, 10);

      IntervalLattice lattice("chr1");

      lattice.insert(x, 0, m_print);
      lattice.insert(z, 0, m_print);
      lattice.insert(y, 0, m_print);
      lattice.insert(w, 0, m_print);

      PRINT print(PRINT::DOT, "lattice.dot");
      std::stack< IntervalLattice::iterator<PRINT> > path = lattice.traverse(lattice.begin(true, print), lattice.end(print));
    }

  private:
    PRINT m_print;
};
