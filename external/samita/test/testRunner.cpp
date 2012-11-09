/*
 *  Created on: 12-28-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 78915 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-25 08:30:00 -0800 (Fri, 25 Feb 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <fstream>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
//#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/TextOutputter.h>
#include <cppunit/CompilerOutputter.h>

// test source files
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include "common/IntervalTest.hpp"
#include "common/FeatureTest.hpp"
#include "align/AlignTest.hpp"
#include "gff/GffTest.hpp"
#include "fastq/FastqTest.hpp"
#include "reference/ReferenceTest.hpp"
#include "sam/BasTest.hpp"
#include "filter/FilterTest.hpp"
#include "pileup/PileupTest.hpp"
#include "sam/MetadataTest.hpp"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(IntervalTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(IntervalTest, "Interval");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FeatureTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FeatureTest, "Feature");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(AlignTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(AlignTest, "Align");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(ReferenceTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(ReferenceTest, "Reference");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(GffTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(GffTest, "Gff");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FastqTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FastqTest, "Fastq");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(BasTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(BasTest, "Bas");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FilterTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(FilterTest, "Filter");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(PileupTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(PileupTest, "Pileup");

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(MetadataTest, "All");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(MetadataTest, "Metadata");

int main (int argc, char *argv[])
{
    // set up logger
    //log4cxx::BasicConfigurator::configure();

    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    if (argc == 1)
        registry.addRegistry("All");
    else
    {
        // run just the specified tests
        for (int i=1; i<argc; i++)
            registry.addRegistry(argv[i]);
    }

    CppUnit::TextUi::TestRunner runner;
    runner.addTest(registry.makeTest());

    // Errors have line numbers
    runner.setOutputter(
     CppUnit::CompilerOutputter::defaultOutputter( 
     &runner.result(),
     std::cerr ) );

    CppUnit::TestResult controller;
    CppUnit::TestResultCollector result;
    controller.addListener(&result);

    // register listener for per-test progress output
    // verbose, per-test output: test : OK
    CppUnit::BriefTestProgressListener progress;
    controller.addListener (&progress);

    runner.run(controller);

    // output stderr summary report
    CppUnit::TextOutputter compilerOutputter(&result, std::cerr);
    compilerOutputter.write();

    // output xml report
    std::ofstream strm("result.xml");
    CppUnit::XmlOutputter xmlOutputter(&result, strm);
    xmlOutputter.write();

    return result.wasSuccessful() ? 0 : 1;
}
