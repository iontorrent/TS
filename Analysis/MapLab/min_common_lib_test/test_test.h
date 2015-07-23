/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_test_h__
#define __test_test_h__
#include <stdexcept>
#include <extesting.h>

class TestTest : public Test
{
    class Sub : public Test
    {
        bool expected_result_;
        bool should_throw_;
    public:
        Sub (const char* name, bool result = true, bool do_throw = false)
        :
        Test (name),
        expected_result_ (result),
        should_throw_ (do_throw)
        {
        }
        bool process () 
        {
            o_ << "Running " << name () << std::endl;
            if (should_throw_)
            {
                o_ << "Throwing an exeption" << std::endl;
                throw std::runtime_error ("TestException");
            }
            o_ << "Exiting with " << (expected_result_ ? "Success" : "Failure") << " status" << std::endl;
            return expected_result_;
        }
    };
    Sub sub1, sub2, sub21, sub22, sub23, sub24;
public:
    TestTest ()
    :
    Test ("TestTest-ShouldFail : TestCase subordination"),
    sub1 ("Sub1-ShouldFail"),
    sub2 ("Sub2"),
    sub21 ("Sub2Sub1"),
    sub22 ("Sub2Sub2-ShouldFail", false),
    sub23 ("Sub2Sub3-ShouldFail", true, true),
    sub24 ("Sub2Sub4")
    {
        add_subord (&sub1);
        add_subord (&sub2);
        sub2.add_subord (&sub21);
        sub2.add_subord (&sub22);
        sub2.add_subord (&sub23);
        sub2.add_subord (&sub24);
        
    }
    bool process () 
    {
        o_ << "Top test: " << name () << std::endl;
        return true;
    }
};
#endif // __test_test_h__
