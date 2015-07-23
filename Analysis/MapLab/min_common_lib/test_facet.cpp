/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_facet.h"
#include <iostream>

// #define TRACE_TEST_FACET

const char* TestFacet::separator = "/";

TestFacetSet::~TestFacetSet ()
{
    for (FVIter iter = facets_.begin (), sent = facets_.end (); iter != sent; ++iter)
    {
        delete *iter;
        *iter = NULL; // prevents double deletion for non-trivial hierarchies
    }
}

TestFacet* TestFacetSet::find (const char* nm, bool force_segmented)
{
    const char* brk = strpbrk (nm, TestFacet::separator);
    if (brk)
    {
        char temp [brk - nm + 1];
        memcpy (temp, nm, brk - nm);
        temp [brk - nm] = 0;
        FSIter itr = name2facet_.find (temp);
        if (itr != name2facet_.end ())
        {
            TestFacetSet* subtree = dynamic_cast <TestFacetSet*> ((*itr).second);
            if (subtree)
            {
                size_t span = strspn (brk, TestFacet::separator);
                const char* subname = brk + span;
                if (*subname)
                    return subtree->find (subname, true);
            }
        }
    }
    else
    {
#ifdef TRACE_TEST_FACET
        std::cerr << "Searching among " << name2facet_.size () << " subords of " << name () << std::endl;
        for (FSIter i = name2facet_.begin (), s = name2facet_.end (); i != s; ++i)
            std::cerr << "    " << (*i).first << " : " << (*i).second->name () << std::endl;
#endif
        FSIter itr = name2facet_.find (nm);
        if (itr != name2facet_.end ())
            return (*itr).second;
#ifdef TRACE_TEST_FACET
        std::cerr << "Not found" << std::endl;
#endif
        if (!force_segmented)
        {
#ifdef TRACE_TEST_FACET
            std::cerr << "Searching for " << nm << " in " << facets_.size () << " subords of " << name () << std::endl;
#endif
            for (FVIter itr = facets_.begin (), sent = facets_.end (); itr != sent; ++itr)
            {
                TestFacetSet* subtree = dynamic_cast <TestFacetSet*> (*itr);
                if (subtree)
                {
#ifdef TRACE_TEST_FACET
                    std::cerr << "    Searching in " << (*itr)->name () << std::endl;
#endif
                    return subtree->find (nm, false);
                }
            }
        }
    }
    return NULL;
}

void TestFacetSet::add (TestFacet* facet)
{
    FSIter itr = name2facet_.find (facet->name ());
    assert (itr == name2facet_.end ());
    name2facet_.insert (itr, std::make_pair (facet->name (), facet));
    facets_.push_back (facet);
}
