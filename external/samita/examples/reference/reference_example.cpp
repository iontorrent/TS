/*
 *  Created on: 12-28-2009
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
#include <samita/reference/reference.hpp>
#include <samita/common/interval.hpp>

using namespace std;
using namespace lifetechnologies;

int main (int argc, char *argv[])
{
    //NOTE: I've used braces below to reuse variable names.
    //      I know this is dangerous but it is appropriate in
    //      this case to make the examples a bit easier to follow.
    {
        cout << "**************************************" << endl;
        cout << "* Uncompressed reference example     *" << endl;
        cout << "**************************************" << endl;
        ReferenceSequenceReader reader;

        reader.open("reference_example.fasta");

        // iterate over all contigs
        ReferenceSequenceReader::iterator iter = reader.begin();
        ReferenceSequenceReader::iterator end = reader.end();
        while (iter != end)
        {
            ReferenceSequence const& refseq = *iter;
            cout << "Name = " << refseq.getName() << "\t" << "3rd base = " << refseq[3] << endl;
            ++iter;
        }

        // select and print out a sequence
        SequenceInterval interval("chr3", 5, 20);
        ReferenceSequence const& refseq = reader.getSequence(interval);
        cout << interval << " = " << refseq.getBases() << endl;
        reader.close();
    }

    {
        // repeat examples only with a compressed reference
        cout << "**************************************" << endl;
        cout << "* Compressed reference example       *" << endl;
        cout << "**************************************" << endl;
        ReferenceSequenceReader reader;
        reader.open("reference_example.fasta.rz");

        // iterate over all contigs
        ReferenceSequenceReader::iterator iter = reader.begin();
        ReferenceSequenceReader::iterator end = reader.end();
        while (iter != end)
        {
            ReferenceSequence const& refseq = *iter;
            cout << "Name = " << refseq.getName() << "\t" << "3rd base = " << refseq[3] << endl;
            ++iter;
        }

        // select and print out a sequence
        SequenceInterval interval("chr3", 5, 20);
        ReferenceSequence const& refseq = reader.getSequence(interval);
        cout << interval << " = " << refseq.getBases() << endl;
        reader.close();
    }

    return 0;
}
