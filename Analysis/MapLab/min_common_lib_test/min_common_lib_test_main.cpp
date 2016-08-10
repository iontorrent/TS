/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <test_suite.h>
#include <comma_locale.h>

#include "test_default.h"
#include "test_test.h"
#include "test_benchmark.h"
#include "test_facets.h"

#include "test_cmdline_parser.h"
#include "test_error_handling.h"
#include "test_loggers.h"
#include "test_resource.h"
#include "test_file_utils.h"
#include "test_tracer.h"
#include "test_time_counter.h"
#include "test_temp_file.h"




class CommonLibTest : public TestSuite
{
    TestDefault        testDefault;
    TestTest           testTest;
    TestBenchmark      testBenchmark;
    TestFacets         testFacets;
    TestErrorHandling  testErrorHandling;
    TestLoggers        testLoggers;
    TestResource       testResource;
    TestTracer         testTracer;
    TestTimeCounter    testTimeCounter;
    TestTempFile       testTempFile;
    TestCmdlineParser  testCmdlineParser;
    TestFileUtils      testFileutils;
    TestPathops        testPathops;
    TestListdir        testListdir;
    TestReadlist       testReadlist;

public:
    CommonLibTest (int argc, char* argv [], char* envp [] = NULL)
    :
    TestSuite (argc, argv, envp)
    {
    }
    void init ()
    {
        reg (testDefault);
        reg (testTest);
        reg (testBenchmark);
        reg (testFacets);
        reg (testErrorHandling);
        reg (testLoggers);
        reg (testResource);
        reg (testTracer);
        reg (testTimeCounter);
        reg (testTempFile);
        reg (testCmdlineParser);
        reg (testFileutils);
        reg (testPathops);
        reg (testListdir);
        reg (testReadlist);
    }
};

int main (int argc, char** argv)
{
    std::cerr.imbue (deccomma_locale);
    std::cout.imbue (deccomma_locale);
    ers.imbue (deccomma_locale);

    CommonLibTest test (argc, argv);
    bool retcode = !test.run ();
    return retcode;
}

