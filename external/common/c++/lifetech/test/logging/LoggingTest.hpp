/*
 *  Created on: 08-31-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49961 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:31:15 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef LOGGING_TEST_HPP_
#define LOGGING_TEST_HPP_

#include <cstdlib>
#include <stdexcept>
#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/log4cxx.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>

#include <lifetech/logging/util.hpp>

using namespace std;
using namespace lifetechnologies;

class LoggingTest: public CppUnit::TestFixture
{
public:
    static CppUnit::Test *suite()
    {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("LoggingTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testTerminateLogger",
                &LoggingTest::testTerminateLogger));
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testTerminatePrinter",
                &LoggingTest::testTerminatePrinter));
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testFileConfiguration",
                &LoggingTest::testFileConfiguration));
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testDefaultConfiguration",
                &LoggingTest::testDefaultConfiguration));
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testEnviromentConfiguration",
                &LoggingTest::testEnviromentConfiguration));
        suiteOfTests->addTest(new CppUnit::TestCaller<LoggingTest>("testMissingFileConfiguration",
                &LoggingTest::testMissingFileConfiguration));
        return suiteOfTests;
    }

    void setUp()
    {
    }

    void tearDown()
    {
        log4cxx::LogManager::resetConfiguration();
    }

    void testTerminateLogger()
    {
        logging::inititialize_terminate_logging();
    }

    void testTerminatePrinter()
    {
        logging::inititialize_terminate_logging(true);
    }

    void testMissingFileConfiguration()
    {
        CPPUNIT_ASSERT_THROW(logging::initialize_logging("logging/missing.properties"), std::invalid_argument);
    }

    void testFileConfiguration()
    {
        logging::initialize_logging("logging/test.properties");
        log4cxx::AppenderList const& appenders = log4cxx::Logger::getRootLogger()->getAllAppenders();
        CPPUNIT_ASSERT_EQUAL((size_t)1, appenders.size());
        CPPUNIT_ASSERT_EQUAL(std::string("config_test"), appenders[0]->getName());
    }

    void testEnviromentConfiguration()
    {
        setenv("LOG4CXX_CONFIGURATION", "logging/test.properties", 1);
        logging::initialize_logging();
        log4cxx::AppenderList const& appenders = log4cxx::Logger::getRootLogger()->getAllAppenders();
        CPPUNIT_ASSERT_EQUAL((size_t)1, appenders.size());
        CPPUNIT_ASSERT_EQUAL(std::string("config_test"), appenders[0]->getName());
        unsetenv("LOG4CXX_CONFIGURATION");
    }

    void testDefaultConfiguration()
    {
        logging::initialize_logging();
        log4cxx::AppenderList const& appenders = log4cxx::Logger::getRootLogger()->getAllAppenders();
        CPPUNIT_ASSERT_EQUAL((size_t)1, appenders.size());
        CPPUNIT_ASSERT_EQUAL(std::string("stderr"), appenders[0]->getName());
    }

};

#endif //LOGGING_TEST_HPP_
