/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <test_suite.h>
#include <comma_locale.h>
#include "test_cigar_lib.h"
#include "test_align_util.h"

class CigarLibTest : public TestSuite
{
    TestCigarLib           testCigarLib;
    TestAlign              testAlign;

public:
    CigarLibTest (int argc, char* argv [], char* envp [] = NULL)
    :
    TestSuite (argc, argv, envp)
    {
    }
    void init ()
    {
        reg (testCigarLib);
        reg (testAlign);
    }
};

int main (int argc, char** argv)
{
    std::cerr.imbue (deccomma_locale);
    std::cout.imbue (deccomma_locale);
    ers.imbue (deccomma_locale);

    CigarLibTest test (argc, argv);
    bool retcode = !test.run ();
    return retcode;
}


