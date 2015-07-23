/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __test_facets_h__
#define __test_facets_h__
#include <extesting.h>
#include <test_facet.h>

extern const char* TFDER_name;
class TestFacets : public Test
{
    class TFacet : public TestFacetSet 
    {
        std::ostream& o_;
    public:
        TFacet (const char* nm, std::ostream& o)
        :
        TestFacetSet (nm),
        o_ (o)
        {
            o_ << "TestFacet " << name () << " (at " << (void*) this << ") constructed" << std::endl;
        }
        ~TFacet ()
        {
            o_ << "TestFacet " << name () << " (at " << (void*) this << ") destroyed" << std::endl;
        }
    };
    class SubTest : public Test
    {
        TestFacet* extr_;
    public:
        SubTest (const char* nm, TestFacet* extr)
        :
        Test (nm),
        extr_ (extr)
        {
        }
        bool init ()
        {
            TFacet* tfbase = new TFacet ("subTestFacet-base", o_);
            TFacet* tfderv = new TFacet ("subTestFacet-derv", o_);
            tfbase->add (tfderv);
            add_facet (tfbase);
            add_facet (extr_, true);
            return true;
        }
        bool process ()
        {
            o_ << "Running process () for SubTest " << name () << std::endl;
            TestFacet* tf = find_facet (TFDER_name);
            TEST_ASSERTX (tf != NULL, "Failed to fine facet by name");
            if (tf)
                o_ << "Found facet by name: " << tf->name () << std::endl;
            else
                o_ << "Facet " << TFDER_name << " not found" << std::endl;
            return true;
        }
        bool cleanup ()
        {
            Test::cleanup ();
            return true;
        }
    };
    TFacet* extr_;
public:
    TestFacets ()
    :
    Test ("TestFacets : facets life cycle, subordination and by-name search"),
    extr_ (NULL)
    {
    }
    bool init ()
    {
        TFacet* tf1 = new TFacet ("testFacetOne-base", o_);
        TFacet* tf2 = new TFacet ("testFacetTwo-base", o_);
        TFacet* tf11 = new TFacet ("testFacetOne-derived-1", o_);
        TFacet* tf12 = new TFacet ("testFacetOne-derived-2", o_);
        TFacet* tf111 = new TFacet (TFDER_name, o_);
        extr_ = new TFacet ("externalFacet", o_);

        tf11->add (tf111);
        tf1->add (tf11);
        tf1->add (tf12);
        add_facet (tf1);
        add_facet (tf2);
        add_facet (extr_, true);

        SubTest *subtest1 = new SubTest ("SubTest1", extr_);
        SubTest *subtest2 = new SubTest ("SubTest2", extr_);
        add_subord (subtest1);
        add_subord (subtest2);
        return true;
    }
    bool process ()
    {
        o_ << "Running process () for TestFacet " << name () << std::endl;
        return true;
    }
    bool cleanup ()
    {
        o_ << "Cleanup - removing external facet" << std::endl;
        delete extr_;
        extr_ = NULL;
        return Test::cleanup ();
    }
};
#endif // __test_facets_h__
