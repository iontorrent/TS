/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <test_suite.h>
#include <comma_locale.h>
#include <runtime_error.h>

#include "test_tmap_binary_search.h"

class TmapLibUnitTests : public TestSuite
{
    TestTmapBinarySearch testTmapBinarySearch;
public:
    TmapLibUnitTests (int argc, char* argv [], char* envp [] = NULL)
    :
    TestSuite (argc, argv, envp)
    {
    }
    void init ()
    {
        reg (testTmapBinarySearch);
    }
};

int main (int argc, char** argv)
{
    std::cerr.imbue (deccomma_locale);
    std::cout.imbue (deccomma_locale);
    ers.imbue (deccomma_locale);

    TmapLibUnitTests test (argc, argv);
    bool retcode = !test.run ();
    return retcode;
}



