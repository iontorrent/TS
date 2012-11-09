/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
/*
 *  Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 *
 *  Created on: Summer 2010
 *      Author: Jonathan Manning <jonathan.manning@lifetech.com>
 *  Latest revision:  $Revision: 76437 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-02-05 18:27:55 -0500 (Sat, 05 Feb 2011) $
 */

/** Enrichment Analysis
 * Run analysis on an enriched dataset.
 * Target Filter
 * Enrichment Stats
 * Per Target Coverage Report
 * Coverage Frequency Histograms
 * Coverage as BEDGRAPH
 * Optional Whole Genome Coverage Frequency Histogram and BEDGRAPH
 */

#include <iostream>
#include <fstream>
#include <string>

#include <log4cxx/logger.h>

#include <samita/common/types.hpp>
#include <samita/align/align_writer.hpp>
#include <samita/align/align_reader.hpp>
#include <samita/align/align_reader_util.hpp>

//#include "errorhandler.hpp"

#include "samdiff_cmdline.h"

/* ================================================== */
namespace lifetechnologies {

struct AlignLess
{
    bool operator()(Align const & a1, Align const & a2) const
    {
        if (a1.getStart() == a2.getStart())
            //return (&a1 > &a2);  // just arbitrarily pick the one with higher pointer
            return a1.getName() > a2.getName();
        else
            return (a1.getStart() > a2.getStart());
    }
};

} //namespace lifetechnologies

/* ================================================== */

/* Main Application */
int main(int argc, char * argv[]) {

    using namespace std;
    using namespace lifetechnologies;

//#ifdef SOLID_DEBUG
//    // Assign C and C++ error handler
//    signal(SIGSEGV, handler);
//    set_terminate(handler);
//#endif

    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samdiff");

    // Parse command line
    gengetopt_args_info args;
    if(::cmdline_parser(argc, argv, &args) != 0) return EXIT_FAILURE;

    bool wantA = (args.mode_arg == mode_arg_Aonly) || (args.mode_arg == mode_arg_Disjoint);
    bool wantB = (args.mode_arg == mode_arg_Bonly) || (args.mode_arg == mode_arg_Disjoint);
    bool wantBoth = (args.mode_arg == mode_arg_Union);

    // Prepare input file(s)
    AlignReader a, b;
    // Open and validate input BAM files
    char * inputA[] = { args.input_first_arg };
    char * inputB[] = { args.input_second_arg };

    if( ! prepareAlignReader(a, 1, inputA) || ! prepareAlignReader(b,1,inputB) )
    {
        LOG4CXX_FATAL(g_log, "Failed to open BAM file(s) for input.)");
        //cerr << "[FATAL] Failed to open BAM file(s) for input." << endl;
        return EXIT_FAILURE;
    }

    std::string outputFilename(args.output_file_arg);
    AlignWriter output(outputFilename, a.getHeader());

    //try {

    AlignReader::const_iterator ia = a.begin();
    AlignReader::const_iterator ib = b.begin();
    // Assume both sorted. Merge sort with delta
    AlignLess aless;

    while(ia != a.end())
    {
        if(ib == b.end())
        {
                // Finish A
            if(wantA) output.write(*ia); // If mode disjoint or a-only
            ++ia;
            continue;
        }

        if(ia->getName() == ib->getName() && ia->getStart() == ib->getStart()) {
            if(wantBoth) output.write(*ia); // If mode union
            // No write of *ib - they are same, only write one
            ++ia;
            ++ib;
            continue;
        }
        if (aless(*ia, *ib))
        {
            if(wantA) output.write(*ia); // If mode disjoint or a-only
            ++ia;
        }
        else
        {
            if(wantB) output.write(*ib); // If mode disjoint or b-only
            ++ib;
        }
    }
    cerr << "Done A" << endl;
    while(ib != b.end()) {
        // Finish B
        if(wantB) output.write(*ib); // If mode disjoint or b-only
        ++ib;
    }
    cerr << "Done B" << endl;
//     catch (exception & e)
//     {
//         LOG4CXX_FATAL(g_log, "Unable to run analysis: " << e.what());
//         cerr << "[FATAL] Unable to run analysis: " << e.what() << endl;
// #ifdef SOLID_DEBUG
//         abort();
// #endif // SOLID_DEBUG
//         return EXIT_FAILURE;
//     }

    return EXIT_SUCCESS;
}

