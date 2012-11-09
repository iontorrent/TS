/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
/*
 *  Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 *
 *  Created on: Summer 2010
 *      Author: Jonathan Manning <jonathan.manning@lifetech.com>
 *  Latest revision:  $Revision: 89251 $
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

#include "headermerge_cmdline.h"

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


    // Parse command line
    gengetopt_args_info args;
    if(::cmdline_parser(argc, argv, &args) != 0) return EXIT_FAILURE;

    if(args.debug_given) {
        log4cxx::LoggerPtr log = log4cxx::Logger::getRootLogger();
        // Force debug level logging
        log->setLevel(log4cxx::Level::getDebug());
    }
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.headermerge");

    if(args.input_given < 2)
    {
        cerr << "Only one input file, no merge." << endl;
    }

    // Prepare input file(s)
    AlignReader bam;
    // Open and validate input BAM files
    if( ! prepareAlignReader(bam, args.input_given, args.input_arg) )
    {
        LOG4CXX_FATAL(g_log, "Failed to open BAM file(s) for input.");
        //cerr << "[FATAL] Failed to open BAM file(s) for input." << endl;
        ::cmdline_parser_free(&args);
        return EXIT_FAILURE;
    }


    cerr << "********* Merged Header: *********" << endl;
    cout << bam.getHeader() << endl;

    bam.close();

    return EXIT_SUCCESS;
}

