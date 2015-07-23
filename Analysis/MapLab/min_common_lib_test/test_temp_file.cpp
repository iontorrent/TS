/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_temp_file.h"
#include <fileutils.h>
#include <portability.h>

bool TestTempFile::process ()
{
    std::string default_tmpdir = temp_dir ();
    o_ << "Default temp directory is " << default_tmpdir << std::endl;
    std::string current_tmpdir = temp_dir (".");
    o_ << "Temp directory for passed '.' is " << current_tmpdir << std::endl;

    std::string default_fname = make_temp_fname ();
    o_ << "Default temp file name is " << default_fname << std::endl;

    std::string kokos_fname = make_temp_fname (NULL, "kokos_");
    o_ << "Temp file name for prefix 'kokos_' is " << kokos_fname << std::endl;

    std::string dest;
    int fhandle = make_linked_temp_file (dest);
    o_ << "Temp file created, name is " << dest << ", fhandle is " << fhandle << std::endl;
    const char TESTDATA [] = "This Is Test Data!";
    size_t wr = ::sci_write (fhandle, TESTDATA, sizeof (TESTDATA));
    if (wr != sizeof (TESTDATA))
        ers << "Unable to write complete test string" << Throw;
    if (::sci_close (fhandle))
        ers << "Unable to close file" << Throw;
    o_ << wr << " bytes written, file closed" << std::endl;
    char buf [sizeof (TESTDATA)];
    fhandle = ::sci_open (dest.c_str (), O_RDONLY);
    if (fhandle == -1)
        ers << "Unable to reopen file for reading" << Throw;
    size_t rd = ::sci_read (fhandle, buf, sizeof (TESTDATA));
    if (rd != sizeof (TESTDATA))
        ers << "Can not fully read data from file" << Throw;
    if (strcmp (buf, TESTDATA))
        ers << "Wrong data read: " << buf << ", expected " << TESTDATA << Throw;
    if (::sci_close (fhandle))
        ers << "Unable to close file" << Throw;
    if (::unlink (dest.c_str ()))
        ers << "Unable to unlink file " << dest << Throw;
    o_ << "write / close / reopen / read / verify / unlink test successfull" << std::endl;

    return true;
}
