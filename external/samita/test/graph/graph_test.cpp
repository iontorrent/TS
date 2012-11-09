/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * relation_test.cpp
 *
 *  Created on: May 4, 2010
 *      Author: kennedcj
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include "NodeTest.hpp"
#include "GraphTest.hpp"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(NodeTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(NodeTest, "Node");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(GraphTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(GraphTest, "Graph");

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
