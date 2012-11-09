/*
 *  Created on: 8-31-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49961 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:31:15 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

// test source files
#include "logging/LoggingTest.hpp"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(LoggingTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(LoggingTest, "Logging");

int main (int argc, char *argv[])
{
    // set up test runner
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
