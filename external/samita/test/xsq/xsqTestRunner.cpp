/*
 *  Created on: 12-28-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49209 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-09-09 06:24:56 -0700 (Thu, 09 Sep 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>

// test source files
//#include <log4cxx/logger.h>
//#include <log4cxx/basicconfigurator.h>
#include "XsqTest.hpp"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(XsqTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(ChenyTest, "cheny");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(MullermwTest, "mullermw");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(DimaTests, "dima");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(UnitTests, "unit");


int main (int argc, char *argv[])
{
    // set up logger
    log4cxx::BasicConfigurator::configure();

    // set up test runner
    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();

    if (argc == 1)
        registry.addRegistry("unit");
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
