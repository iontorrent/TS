#ifndef __CIGAR_TEST_LIB_H__
#define __CIGAR_TEST_LIB_H__

#include <extesting.h>

// Below is a template for a derivative from Test
// To add a case,
//   - copy class below and substitute name/constructor name/ heading with proper values
//   - instantiate an instance of this class wiin CommonLibTest object
//   - register the instance in the init method of the CommonLibTest class
//   - fill in "process" and add "init", "report" and "cleanup: if needed

class TestCigarTranslator : public Test
{
public:
    TestCigarTranslator ();
    bool process ();
};

class TestDecimalPositions : public Test
{
public:
    TestDecimalPositions ();
    bool process ();
};

class TestCigarFootprint : public Test
{
public:
    TestCigarFootprint ();
    bool process ();
};

class TestNextCigarPos : public Test
{
public:
    TestNextCigarPos ();
    bool process ();
};


class TestCigarLib : public Test
{
    TestDecimalPositions testDecimalPositions;
    TestCigarTranslator  testCigarTranslator;
    TestCigarFootprint   testCigarFootprint;
    TestNextCigarPos     testNextCigarPos;

public:
    TestCigarLib ()
    :
    Test ("CigarLib")
    {
        add_subord (&testDecimalPositions);
        add_subord (&testCigarTranslator);
        add_subord (&testCigarFootprint);
        add_subord (&testNextCigarPos);
    }
    bool init ();
    bool process () 
    {
        return true;
    }
};





#endif

