/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_default_h__
#define __test_default_h__
#include <extesting.h>
// Below is a template for a derivative from Test
// To add a case,
//   - copy class below and substitute name/constructor name/ heading with proper values
//   - instantiate an instance of this class wiin CommonLibTest object
//   - register the instance in the init method of the CommonLibTest class
//   - fill in "process" and add "init", "report" and "cleanup: if needed

class TestDefault : public Test
{
public:
    TestDefault ()
    :
    Test ("TestDefault : basic test system operability")
    {
    }
    bool process () {return true;}
};
#endif // __test_default_h__
