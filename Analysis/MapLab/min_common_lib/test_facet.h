/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

#ifndef __test_facet_h__
#define __test_facet_h__

#include <cassert>
#include <vector>
#include <map>
#include <set>
#include <cstring>
#include <string>

/// \file
/// Base classes for test facets are defined here.\n
/// The facet is a resource that is acquired (or created) before running the test or benchmark, 
/// used within a test/benchmark and then destroyed / released. \n
/// The example of a facet-controlled resource is a file with certain context used by a test case that 
/// checks functionaluty of file reading function.\n
/// The reason for mantaining a base classe for facets is lifetime management (ordered automated destruction).

/// Base class for test facets
/// 
/// derived classes should perform any resource acquisition and initialization in constructor, 
/// and cleanup and resource releasing in destructor,
/// preferrably using automatic destruction of members.

class TestFacet
{
    std::string name_;
public:
    static const char* separator;
    /// Constructor\n
    /// All facets are named to facilitate search in combined facet sets
    /// \param name the name of a facet. The uniquiness is not enforced, but duplicate names 
    /// would prevent access to all but one similarly named facets
    TestFacet (const char* name)
    {
        // make sure no separator is in the name - Ok to drop in 'release' test
        if (name)
        {
            assert (strpbrk (name, separator) == NULL);
            const char* end = name;
            while (*end && !isspace (*end))
                ++end;
            name_.assign (name, end - name);
        }
        // else
        // {
        //    std::cerr << "WARNING: name not passed to the created Facet - it is not searchable" << std::endl;
        // }
    }
    /// virtual destructor - ensures destructor chaining in derived classes
    virtual ~TestFacet () 
    {
    }
    /// name accessoe
    const char* name () const
    {
        return name_.c_str ();
    }
};

typedef std::set <TestFacet*> FaSet;
typedef std::vector <TestFacet*> FaVec;

class lexgr_charp_comparator
{
public:
    bool operator () (const char* s1, const char* s2) const
    {
        return strcmp (s1, s2) < 0;
    }
};

/// Base class for complex test facet 
/// that can have subordinate facets (forming a tree-like structure)
///
/// The root node of the tree of TestFacetSets controls lifetime of the entire tree; 
/// its' destruction causes destruction of all subordinate facets
/// The subordinates in the TestFacetSet are ordered; they are destroyed in the order opposite to their addition
/// to the master\n
/// For use of TestFacetSet, all the facets should be allocated on heap using new operator
class TestFacetSet : public TestFacet
{
public:
    typedef std::map <const char*, TestFacet*, lexgr_charp_comparator> Name2Facet;
    typedef FaVec::iterator FVIter;
    typedef Name2Facet::iterator FSIter;
private:
    FaVec facets_;
    Name2Facet name2facet_;
public:
    /// Constructor
    TestFacetSet (const char* name)
    :
    TestFacet (name)
    {
    }
    /// Destructor: destroys subordinate objects in the order opposite to their addition
    ~TestFacetSet ();
    /// adds facet to a tree. TestFacetSet can be added and will become a facet tree component 
    void add (TestFacet* facet);
    /// name-based access
    /// the name of the facet may be segmented, in this case the levels are searched explicitely.
    /// segments are separated with (spans of) "/" character (as in unix path)
    /// the non-segmented names are searched in entire tree
    /// \param name   the name for the searched facet
    /// \param force_segmented disables global search, searches only in immediate subordinates
    TestFacet* find (const char* name, bool force_segmented = false);
    FVIter begin () { return facets_.begin (); }
    FVIter end () { return facets_.end (); }
};


#endif // __test_facet_h__