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
#include <samita/gff/gff.hpp>
#include <samita/common/feature.hpp>

using namespace std;
using namespace lifetechnologies;

int main (int argc, char *argv[])
{
    // write a gff file
    GffWriter writer;
    GffFeature feature;

    feature.setSequence("chr1");
    feature.setSource("gff_example");
    feature.setType("example_feature");
    feature.setStart(1);
    feature.setEnd(100);
    feature.setScore(3.1415);

    writer.open("gff_example.gff");
    writer << feature << std::endl;
    writer.close();

    // read a gff file
    GffReader reader;
    GffReader::const_iterator iter;
    GffReader::const_iterator end;

    reader.open("gff_example.gff");
    iter = reader.begin();
    end = reader.end();
    while (iter != end)
    {
        GffFeature const& feature = *iter;
        cout << feature << endl;
        ++iter;
    }
    reader.close();

    return 0;
}
