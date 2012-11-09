/*
 *  Created on: 09-20-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 11:54:43 -0700 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <samita/align/align_reader.hpp>
#include <samita/filter/filter.hpp>
#include <samita/filter/mate_filter.hpp>

using namespace std;
using namespace lifetechnologies;

// predicate class to pass all alignments through filter >= user specified mapping quality
class ExampleMinQualFilter
{
public:
    ExampleMinQualFilter(int q=0) : m_minQual(q) {}
    bool operator() (Align const &a) const
    {
        return (a.getMapQual() >= m_minQual);
    }
private:
    int m_minQual;
};

// predicate class to pass all alignments through filter <= user specified mapping quality
class ExampleMaxQualFilter
{
public:
    ExampleMaxQualFilter(int q=255) : m_maxQual(q) {}
    bool operator() (Align const &a) const
    {
        return (a.getMapQual() <= m_maxQual);
    }
private:
    int m_maxQual;
};

// predicate class to pass all alignments through filter containing a user specified mapping flag
class ExampleFlagFilter
{
public:
    ExampleFlagFilter(int f=0) : m_flag(f) {}
    bool operator() (Align const &a) const
    {
        return ((a.getFlag() & m_flag) == m_flag);
    }
private:
    int m_flag;
};

class ExampleDynamicFilter
{
public:
    ExampleDynamicFilter(bool *isOpen) : m_isOpenPtr(isOpen) {}
    bool operator() (Align const &a) const
    {
        bool isOpen = *m_isOpenPtr;
        if (isOpen)
            cout << "filter is open" << endl;
        else
            cout << "filter is closed" << endl;
        return isOpen;
    }
private:
    bool *m_isOpenPtr;
};


int main (int argc, char *argv[])
{

    char *bamFilename = "../data/example.sorted.bam";

    if (argc > 1 )
    {
        bamFilename = argv[1];
    }

    //NOTE: I've used braces below to reuse variable names.
    //      I know this is dangerous but it is appropriate in
    //      this case to make the examples a bit easier to follow.

    // iterate over all Align's using a filter
    {
        cout << "**************************************" << endl;
        cout << "* Align Example w/ filter            *" << endl;
        cout << "**************************************" << endl;
        ExampleFlagFilter filter(64);

        AlignReader sam(bamFilename);  // client can also specify filename using constructor

        AlignReader::filter_iterator<ExampleFlagFilter> iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator<ExampleFlagFilter> end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Select Example w/ filter           *" << endl;
        cout << "**************************************" << endl;
        ExampleFlagFilter filter(64);

        AlignReader sam(bamFilename);

        sam.select("1:5000-9000");
        AlignReader::filter_iterator<ExampleFlagFilter> iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator<ExampleFlagFilter> end(filter, sam.end(), sam.end());
        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ mates filter            *" << endl;
        cout << "**************************************" << endl;
        AlignMates mates;
        MateFilter filter(&mates);

        AlignReader sam(bamFilename);

        AlignReader::filter_iterator<MateFilter> iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator<MateFilter> end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            cout << "Found pair: " << endl;
            cout << "     " << mates.first << endl;
            cout << "     " << mates.second << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ two filters             *" << endl;
        cout << "**************************************" << endl;
        ExampleMinQualFilter minFilter(0);
        ExampleMaxQualFilter maxFilter(255);

        AlignReader sam(bamFilename);

        FilterPair<ExampleMinQualFilter, ExampleMaxQualFilter> filter(minFilter, maxFilter);
        AlignReader::filter_iterator< FilterPair<ExampleMinQualFilter, ExampleMaxQualFilter>  > iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator< FilterPair<ExampleMinQualFilter, ExampleMaxQualFilter>  > end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ three filters           *" << endl;
        cout << "**************************************" << endl;
        ExampleMinQualFilter minFilter(0);
        ExampleMaxQualFilter maxFilter(255);
        ExampleFlagFilter flagFilter(64);

        AlignReader sam(bamFilename);

        FilterTriple<ExampleMinQualFilter, ExampleMaxQualFilter, ExampleFlagFilter> filter(minFilter, maxFilter, flagFilter);

        AlignReader::filter_iterator< FilterTriple<ExampleMinQualFilter, ExampleMaxQualFilter, ExampleFlagFilter>  > iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator< FilterTriple<ExampleMinQualFilter, ExampleMaxQualFilter, ExampleFlagFilter>  > end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ N filters               *" << endl;
        cout << "**************************************" << endl;
        ExampleMinQualFilter minFilter(0);
        ExampleMaxQualFilter maxFilter(255);
        ExampleFlagFilter flagFilter(64);

        AlignReader sam(bamFilename);

        FilterChain chain;
        chain.add(minFilter);
        chain.add(maxFilter);
        chain.add(flagFilter);

        AlignReader::filter_iterator< FilterChain  > iter(chain, sam.begin(), sam.end());
        AlignReader::filter_iterator< FilterChain > end(chain, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ standard filter         *" << endl;
        cout << "**************************************" << endl;
        StandardFilter filter(true, true, true, true, true, 40);

        AlignReader sam(bamFilename);

        AlignReader::filter_iterator< StandardFilter  > iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator< StandardFilter  > end(filter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    {
        cout << "**************************************" << endl;
        cout << "* Example w/ a dynamic filter        *" << endl;
        cout << "**************************************" << endl;

        AlignReader sam(bamFilename);

        bool isOpen = true;

        ExampleDynamicFilter filter(&isOpen);

        AlignReader::filter_iterator< ExampleDynamicFilter  > iter(filter, sam.begin(), sam.end());
        AlignReader::filter_iterator< ExampleDynamicFilter  > end(filter, sam.end(), sam.end());

        int nRecords = 0;
        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            nRecords++;
            if (nRecords > 5)
                isOpen = false; // close the filter
            ++iter;
        }
    }

    return 0;
}
