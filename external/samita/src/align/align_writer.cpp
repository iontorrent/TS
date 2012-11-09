/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
/* Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

#include <sam.h>
#include <log4cxx/logger.h>
#include <boost/filesystem.hpp>

#include "samita/align/align_writer.hpp"

// Forward decl from sam.c, exported properly in 0.1.12
//extern bam_header_t *bam_header_dup(const bam_header_t *h0);

namespace lifetechnologies {

    static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.alignwriter");
    namespace fs = boost::filesystem;

    AlignWriter::AlignWriter(const std::string & output, const BamHeader & header)
        : m_filename(output),
          m_bam(NULL)
    {
        if(header.empty())
        {
            LOG4CXX_FATAL(g_log, "Invalid BAM Header given for output file. Unable to open or merge input BAM headers");
            return;
        }
        assert(!header.empty());
        bam_header_t * bheader = buildBamHeader(header);
        assert(bheader != NULL);

        { // Create leading output folders
            fs::path outputFile(m_filename);
            fs::create_directories(outputFile.parent_path());
        }

        m_bam = samopen(m_filename.c_str(), "wb", bheader);
        LOG4CXX_DEBUG(g_log, "Opened BAM file for writing: " << m_filename);
        //bam_header_destroy(bheader); // NB pair with dup below
    }

    AlignWriter::~AlignWriter()
    {
        if(m_bam == NULL) return;
        samclose(m_bam);
        m_bam = NULL;
        LOG4CXX_DEBUG(g_log, "Closed BAM file after writing: " << m_filename);

        if(m_filename.empty()) return;
        fs::file_status s = fs::status(m_filename); // Stat once for multiple tests
        if(!fs::exists(s) || !fs::is_regular_file(s) || fs::is_empty(m_filename))
        {
            LOG4CXX_INFO(g_log, "Non-existent, not a regular file, or empty BAM file: '"<< m_filename <<"'. Skipping automatic indexing operation.");
            return;
        }

        // Build index on new file, if possible.
        LOG4CXX_DEBUG(g_log, "Generating index on BAM file: '" << m_filename << "'");
        int ret = bam_index_build(m_filename.c_str());
        if (ret == 0 )
        {
            LOG4CXX_INFO(g_log, "Generated index on BAM: " << m_filename);
        }
        else
        {
            LOG4CXX_ERROR(g_log, "Failed to generate index on BAM: " << m_filename);
        }
        assert( ret == 0 );
    }

    bool AlignWriter::write(const bam1_t * align)
    {
        return (samwrite(m_bam, align) > 0);
    };


    // FIXME - move to BamHeader
    bam_header_t * AlignWriter::buildBamHeader(const BamHeader & header)
    {
        // Punt - return original raw header, not composite header.
        bam_header_t * hdr = header.getRawHeader();
        assert(hdr != NULL);
        //return ::bam_header_dup(hdr); dup plus destroy, or neither dup nor //destroy
        return hdr;
    }

} // namespace lifetechnologies
