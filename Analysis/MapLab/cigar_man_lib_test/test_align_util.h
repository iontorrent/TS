#ifndef __TEST_ALIGN_UTIL_H__
#define __TEST_ALIGN_UTIL_H__

#include <extesting.h>
#include "test_alignment_generator.h"

// Below is a template for a derivative from Test
// To add a case,
//   - copy class below and substitute name/constructor name/ heading with proper values
//   - instantiate an instance of this class wiin CommonLibTest object
//   - register the instance in the init method of the CommonLibTest class
//   - fill in "process" and add "init", "report" and "cleanup" if needed

class TestValidateAlignment : public Test
{
public:
    TestValidateAlignment ();
    bool process ();
};

class TestNormalizeSegment : public Test
{
public:
    TestNormalizeSegment ();
    bool process ();
};

class TestSegmentScore : public Test
{
public:
    TestSegmentScore ();
    bool process ();
};

class TestWorstScorePosFinder : public Test
{
public:
    TestWorstScorePosFinder ();
    bool process ();
};

class TestSegmentMoveBases : public Test
{
public:
    TestSegmentMoveBases ();
    bool process ();
};

class TestClipBasesFromSegment : public Test
{
public:
    TestClipBasesFromSegment ();
    bool process ();
};

class TestClipSegmentByScore : public Test
{
public:
    TestClipSegmentByScore ();
    bool process ();
};

class TestClipSegmentToRefBase : public Test
{
    bool deterministic_test ();
    bool random_test ();
public:
    TestClipSegmentToRefBase ();
    bool process ();
};


class TestAlignUtil : public Test
{
    TestValidateAlignment     testValidateAlignment;
    TestNormalizeSegment      testNormalizeSegment;
    TestSegmentScore          testSegmentScore;
    TestWorstScorePosFinder   testWorstScorePosFinder;
    TestSegmentMoveBases      testSegmentMoveBases;
    TestClipBasesFromSegment  testClipBasesFromSegment;
    TestClipSegmentByScore    testClipSegmentByScore;
    TestClipSegmentToRefBase  testClipSegmentToRefBase;

public:
    TestAlignUtil ()
    :
    Test ("AlignUtil")
    {
        add_subord (&testValidateAlignment);
        add_subord (&testNormalizeSegment);
        add_subord (&testSegmentScore);
        add_subord (&testWorstScorePosFinder);
        add_subord (&testSegmentMoveBases);
        add_subord (&testClipBasesFromSegment);
        add_subord (&testClipSegmentByScore);
        add_subord (&testClipSegmentToRefBase);
    }
    bool process () 
    {
        return true;
    }
};


class TestAlign : public Test
{
    TestAlignmentGenerator  testAlignmentGenerator;
    TestAlignUtil           testAlignUtil;
public:
    TestAlign ()
    :
    Test ("Align")
    {
        add_subord (&testAlignmentGenerator);
        add_subord (&testAlignUtil);
    }
    bool init ();
    bool process () 
    {
        return true;
    }
};



#endif


