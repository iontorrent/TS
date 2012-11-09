/*
 *  Created on: 12-28-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 64019 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-01-04 15:28:27 -0800 (Tue, 04 Jan 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <samita/align/align_reader.hpp>
#include <samita/common/interval.hpp>
#include <samita/filter/filter.hpp>

using namespace std;
using namespace lifetechnologies;

// worker to handle each alignment
class ExampleWorker

{
    public:
        ExampleWorker() {}
        void operator() (Align const &a)
        {
            cout << a << endl;
        }
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

    // iterate over all Align's using no filter
    {
        cout << "**************************************" << endl;
        cout << "* Align Example w/ no filter    *" << endl;
        cout << "**************************************" << endl;

        AlignReader sam;
        sam.open(bamFilename);
        AlignReader::iterator iter = sam.begin();
        AlignReader::iterator end = sam.end();
        vector<Align> savedAlignments;
        while (iter != end)
        {
            Align const& a = *iter;

            // do some work with a...
            cout << a << endl;
            savedAlignments.push_back(a);  // this makes a copy
            ++iter;
        }
        cout << savedAlignments.size() << " alignments saved" << endl;
    }

    // iterate over all Align's using no filter
    {
        cout << "**************************************" << endl;
        cout << "* Align Example w/ no filter         *" << endl;
        cout << "**************************************" << endl;
        AlignReader sam;
        sam.open(bamFilename);
        AlignReader::const_iterator iter = sam.begin();
        AlignReader::const_iterator end = sam.end();
        while (iter != end)
        {
            Align const& a = *iter;
            // do some work with a...
            cout << a << endl;
            ++iter;
        }
    }

    // count using std::for_each
    {
        cout << "**************************************" << endl;
        cout << "* Print alignments using for_each    *" << endl;
        cout << "**************************************" << endl;
        AlignReader sam;
        sam.open(bamFilename);

        ExampleWorker worker;
        std::for_each(sam.begin(), sam.end(), worker);
    }

    {
        cout << "**************************************" << endl;
        cout << "* Select Example w/ query string     *" << endl;
        cout << "**************************************" << endl;
        AlignReader sam(bamFilename);

        sam.select("1:5000-9000");
        AlignReader::iterator iter = sam.begin();
        AlignReader::iterator end = sam.end();
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
        cout << "* Select Example w/ interval         *" << endl;
        cout << "**************************************" << endl;
        AlignReader sam(bamFilename);
        SequenceInterval interval("1", 5000, 9000);

        sam.select(interval);
        AlignReader::iterator iter = sam.begin();
        AlignReader::iterator end = sam.end();
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
        cout << "* BAM Header Example                 *" << endl;
        cout << "**************************************" << endl;
        AlignReader sam(bamFilename);
        BamHeader hdr = sam.getHeader();
        cout << hdr << endl; // print the header

        FlagFilter mappedFilter(BAM_FUNMAP);

        AlignReader::filter_iterator<FlagFilter> iter(mappedFilter, sam.begin(), sam.end());
        AlignReader::filter_iterator<FlagFilter> end(mappedFilter, sam.end(), sam.end());

        while (iter != end)
        {
            Align const& a = *iter;

            std::string const& rgID = a.getReadGroupId();
            int32_t refID = a.getRefId();

            // get the RG class for this alignments read group ID
            RG const& rg = hdr.getReadGroup(rgID);

            // get the SQ class for this alignments ref ID
            SQ const& sq = hdr.getSequence(refID);

            // get the library type
            LibraryType libType = getLibType(rg.LB);

            // get the bam file id (aka index into the vector of bam files)
            int32_t fileID = a.getFileId();

            // print out some arbitrary stuff
            cout << a.getName() << " : "
                 << "\tRead group sample = " << rg.SM
                 << "\tLibraryType = " << libType
                 << "\tReference name = " << sq.SN
                 << "\tFile id = " << fileID << endl;
            ++iter;
        }
    }

    return 0;
}
