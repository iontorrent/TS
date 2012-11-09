/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4 -*-   vi:set expandtab ts=4 sw=4: */
#ifndef ALIGNWRITER_HPP
#define ALIGNWRITER_HPP
/* Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

#include <string>

#include <sam.h>

#include <samita/align/align_reader.hpp>
#include <samita/common/types.hpp>

namespace lifetechnologies {

// Consider moving to samita++
class AlignWriter
{
    const std::string m_filename;
    samfile_t * m_bam;

public:
    //AlignWriter(const std::string & output);
    AlignWriter(const std::string & output, const BamHeader & header);
    ~AlignWriter();

    inline bool write(const Align & align) { return write(align.getBamPtr()); }
    bool write(const bam1_t * align);

private:
    // FIXME - move to BamHeader
    bam_header_t * buildBamHeader(const BamHeader & header);

};

} // namespace lifetechnologies
#endif // ALIGNWRITER_HPP
