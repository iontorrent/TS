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

#include "remarkpairs_cmdline.h"

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

    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.remarkpairs");

    // Parse command line
    gengetopt_args_info args;
    if(::cmdline_parser(argc, argv, &args) != 0) return EXIT_FAILURE;

    int32_t minInsertSize = args.min_arg;
    int32_t maxInsertSize = args.max_arg;
    if(minInsertSize > maxInsertSize) {
        cerr << "Minimum must be less than Maximum" << endl;
        return EXIT_FAILURE;
    }

    // Prepare input file(s)
    AlignReader input;
    // Open and validate input BAM files
    char * inputFiles[] = { args.input_arg };

    if( ! prepareAlignReader(input, 1, inputFiles))
    {
        LOG4CXX_FATAL(g_log, "Failed to open BAM file for input.)");
        return EXIT_FAILURE;
    }

    const BamHeader & header = input.getHeader();
    // WARNING - only looks at first read group
    const LibraryType lt = getLibType(header.getReadGroups()[0]);
    if(lt == LIBRARY_TYPE_FRAG) {
        cerr << "Refusing to mark proper pairs on a FRAG library." << endl;
        return EXIT_FAILURE;
    }

    std::string outputFile(args.output_arg);
    try {
        AlignWriter output(outputFile, header);

        for(AlignReader::iterator it = input.begin(); it != input.end(); ++it)
        {
            Align & align = *it;
            const bool isAAA = (getCategory(align, lt, minInsertSize, maxInsertSize) == "AAA");
            int32_t flag = align.getFlag();
            if (isAAA)
                flag |= BAM_FPROPER_PAIR; // set bit
            else
                flag &= ~BAM_FPROPER_PAIR; // clear bit
            align.getBamPtr()->core.flag = flag; //align.setFlag(flag);
            output.write(align);
        }

        //output.close();
    }
    catch (exception & e)
    {
        LOG4CXX_FATAL(g_log, "Unable to run analysis: " << e.what());
        cerr << "[FATAL] Unable to run analysis: " << e.what() << endl;
#ifdef SOLID_DEBUG
        abort();
#endif // SOLID_DEBUG
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

