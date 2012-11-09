/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*- vim: set expandtab ts=4 sw=4: */
/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */
#ifndef ALIGN_READER_UTIL_HPP_
#define ALIGN_READER_UTIL_HPP_

#include <log4cxx/logger.h>
#include <boost/filesystem.hpp>

#include <samita/align/align_reader.hpp>

namespace lifetechnologies {

    /** \function prepareAlignReader
     * Utility function to validate an array of filenames
     * for input as BAM.
     */

    bool prepareAlignReader(AlignReader & alignReader,
                            unsigned int const nInputFiles,
                            char ** const inputFiles)
    {
        namespace fs = boost::filesystem;
        log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("lifetechnologies.prepareAlignReader");

        if(nInputFiles == 0)
        {
            LOG4CXX_FATAL(log, "No input BAM files given.");
            return false;
        }

        bool allValid = true; // Set false if any file fails validation

        //AlignReader alignReader
        // push each input file -- support multiple files
        for(unsigned int i = 0; i < nInputFiles; i++) {
            // Stat file - make sure it exists and is non-zero size
            fs::path inputFilePath(inputFiles[i]);

            // Error conditions are not fatal until end - all files are checked.
            if(inputFilePath.empty())
            {
                LOG4CXX_FATAL(log, "Empty filename given?");
                allValid=false;
                continue;
            }

            if( ! fs::exists(inputFilePath) )
            {
                LOG4CXX_FATAL(log, "BAM file does not exist: '" <<
                              inputFilePath.filename() << "'");
                allValid=false;
                continue;
            }

            if( fs::is_empty(inputFilePath) )
            {
                LOG4CXX_FATAL(log, "File is empty: '" <<
                              inputFilePath.filename() << "'");
                allValid=false;
                continue;
            }


            // Open it - adds it to alignreader iterator
            if(! alignReader.open(inputFiles[i]))
            {
                /* Can fail for several reasons, including header merge */
                LOG4CXX_FATAL(log, "Failed to open BAM file: '" <<
                              inputFilePath.filename() << "'");
                allValid=false;
                continue;
            }

            LOG4CXX_INFO(log, "Opening BAM file for input: " << inputFiles[i]);

            // Probe header to force open and validate
            BamHeader test = alignReader.getHeader();
            if(test.empty()) {
                LOG4CXX_FATAL(log, "Failed to read header from input BAM");
                /* NB: This is immediately fatal, no other files are checked.
                * If header can't be read, all later files will also fail.
                * (Perhaps through no fault of their own...) */
                return false;
            }
        }


        // //Probe header to force open and validate
        //BamHeader test = alignReader.getHeader();
        //if(test.empty()) {
        //    LOG4CXX_FATAL(log, "Failed to read header from input BAM");
        //    /* NB: This is immediately fatal, no other files are checked.
        //     * If header can't be read, all later files will also fail.
        //     * (Perhaps through no fault of their own...) */
        //    return false;
        //}

        // True only if all files passed validation
        return allValid;
    }

} // namespace lifetechnologies
#endif // ALIGNREADERUTIL_HPP_
