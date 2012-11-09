/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * interval_lattice_test.cpp
 *
 *  Created on: July 27, 2010
 *      Author: kennedcj
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include "interval_lattice_test.hpp"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(IntervalLatticeTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(IntervalLatticeTest, "IntervalLatticeTest");

int main (int argc, char *argv[])
{

	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();

	if (argc == 1)
	        registry.addRegistry("All");
	    else
	    {
	        // run just the specified tests
	        for (int i=1; i<argc; i++)
	            registry.addRegistry(argv[i]);
	    }

	    runner.addTest(registry.makeTest());
	    if (runner.run())
	        return 0;
	    return 1;
}
