/*
 *  Created on: 1-04-2011
 *  Author: Jonathan Manning
 *
 *  Latest revision:  $Revision: 77620 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-01-04 18:28:27 -0500 (Tue, 04 Jan 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#include <iostream>
#include <exception>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <log4cxx/logger.h>

#include "samita/sam/bam_metadata.hpp"

namespace lifetechnologies
{

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.sam.bam_metadata");
static const boost::regex BAMEXTENDEDCOMMENT("^@CO\t[RP]G:");

void RGExtended::parse(std::vector<std::string> const & fields)
{
    std::vector<std::string>::const_iterator field = fields.begin();
    ++field; // Skip initial RG record. FIXME - confirm same as this.ID

    for (;
         field != fields.end();
         ++field)
    {
        // Verify XX:value
        if((*field).find_first_of(":") != 2)
        {
            LOG4CXX_ERROR(g_log, "Attribute in RG Extended Comment without value? '" << (*field) << "'");
        }

        std::string const & id = (*field).substr(0,2);
        std::string const & value = (*field).substr(3);
        //LOG4CXX_DEBUG(g_log, "Parsing Extended header annotation: '" << id << "' = '" << value << "'");

        // try catch lexical_casts.
        try{
            if (id == "IX")
                IX = value;
            else if (id == "II")
                II = value;
            else if (id == "LD")
                LD = value;
            else if (id == "LT")
                LT = value;
            else if (id == "AT")
                AT = value;

            else if (id == "IA")
                IA = boost::lexical_cast<float>(value);
            else if (id == "IS")
                IS = boost::lexical_cast<float>(value);
            else if (id == "IN")
                IN = boost::lexical_cast<int32_t>(value);
            else if (id == "IM")
                IM = boost::lexical_cast<int32_t>(value);

            else if (id == "SP")
                SP = value;
            else if (id == "SD")
                SD = value;

            else if (id == "BX")
                BX = boost::lexical_cast<bool>(value);
            else if (id == "TN")
                TN = boost::lexical_cast<int32_t>(value);
            else if (id == "TX")
                TX = boost::lexical_cast<int32_t>(value);

            else if (id == "BY")
                BY = boost::lexical_cast<bool>(value);
            else if (id == "UN")
                UN = boost::lexical_cast<int32_t>(value);
            else if (id == "UX")
                UX = boost::lexical_cast<int32_t>(value);

            else if (id == "EC")
                EC = boost::lexical_cast<bool>(value);

            else if (id == "ER")
                ER = value;
            else if (id == "DE")
                DE = value;
            else if (id == "CO")
                CO = value;
            else if (id == "UU")
                UU = value;
            else if (id == "PN")
                PN = value;
            else if (id == "PJ")
                PJ = value;
            else if (id == "SO")
                SO = value;
            else if (id == "CU")
                CU = boost::lexical_cast<uint64_t>(value);
            else if (id == "CT")
                CT = boost::lexical_cast<uint64_t>(value);

            else
            {
                LOG4CXX_WARN(g_log, "bam metadata comment - ignoring unknown @CO RG attribute: " << *field);
                //throw invalid_input_record("bam metadata comment - unknown attribute:", *field);
            }
        } catch (boost::bad_lexical_cast & be) {
            LOG4CXX_FATAL(g_log, "bam metadata comment - failed to parse value: '" << *field << "': '" << id << "' = '" << value << "'");
        }
    }

    // Verify mandatory fields were set? -- Not yet.
}

bool RGExtended::hasMetadata() const
{
    /*
    // Debug assertions to help diagnose issues with specific errors
    assert(!LB.empty());
    assert(!LT.empty());
    assert(IA!=0);
    assert(IS!=0);
    assert(TX>0);
    assert(TN>0);
    //assert(LT!="frag" && (UX>0) && (UN>0));
    */
    if(LB.empty()) LOG4CXX_WARN(g_log, "Missing mandatory RG LB field (mandatory for LifeScope 2.0)");
    if(LT.empty()) LOG4CXX_WARN(g_log, "Missing mandatory @CO RG LT metadata field");
    if(TX<=0) LOG4CXX_WARN(g_log, "Missing mandatory @CO RG TX metadata field. (Must exist and be > 0.) " << TX);
    if(TN<=0) LOG4CXX_WARN(g_log, "Missing mandatory @CO RG TN metadata field. (Must exist and be > 0.) " << TN);
    if(LT != "Fragment")
    {
        if(IA<=0.0) LOG4CXX_WARN(g_log, "Missing conditional @CO RG IA metadata field. (Must exist and be non-zero.) IA=" << IA);
        if(IS<=0.0) LOG4CXX_WARN(g_log, "Missing conditional @CO RG IS metadata field. (Must exist and be non-zero.) IS=" << IS);
        if(UX<=0) LOG4CXX_WARN(g_log, "Missing conditional @CO RG UX metadata field. (Must exist and be > 0 for a paired run.) LT=" << LT << "UX=" << UX);
        if(UN<=0) LOG4CXX_WARN(g_log, "Missing conditional @CO RG UN metadata field. (Must exist and be > 0 for a paired run.) LT=" << LT << "UN=" << UN);
    } else {
        if(IA>0.0) LOG4CXX_WARN(g_log, "Conditional @CO RG IA metadata field specified for Fragment library. Ignoring. IA=" << IA);
        if(IS>0.0) LOG4CXX_WARN(g_log, "Conditional @CO RG IS metadata field specified for Fragment library. Ignoring. IS=" << IS);
    }

    // Verify mandatory fields
    return (
        //!LB.empty() && // RG field, not RGExt.
        !LT.empty() &&
        (TX>0) && (TN>0)
        && (LT=="Fragment" || ( (UX>0) && (UN>0) && (IA!=0.0) && (IS!=0.0) ) )  //FIXME - bad placeholder values... 0.0 could be valid */
        );
}


void PGExtended::parse(std::vector<std::string> const & fields)
{
    std::vector<std::string>::const_iterator field = fields.begin();
    field++; // Skip initial PG record. FIXME - confirm same as this.ID

    for (;
         field != fields.end();
         field++)
    {
        std::vector<std::string> annotation;
        string_util::tokenize(*field, ":", annotation);
        if(annotation.size() != 2) {
            LOG4CXX_ERROR(g_log, "Invalid PG Extended Comment: " << *field);
            return;
        }

        std::string & id    = annotation[0];
        std::string & value = annotation[1];

        if (id == "AS")
            AS = value;
        else if (id == "PS")
            PS = value;
        else if (id == "PN")
            PN = value;
        else if (id == "PV")
            PV = value;
        else if (id == "PM")
            PM = value;
        else
            LOG4CXX_WARN(g_log, "bam metadata comment - ignoring unknown @CO PG attribute: " << *field);
        //throw invalid_input_record("bam metadata comment - unknown attribute", *field);

    }
}

bool PGExtended::hasMetadata() const
{
    // True if any field has data. None are labeled Mandatory
    return (!AS.empty() || !PS.empty() || !PN.empty() || !PV.empty() || !PM.empty());
}



BamMetadata::BamMetadata(const BamHeader & header)
: BamHeader(header), m_foundExtended(false)
{
    // Parse Comments
    initialize();
    // Verify all RG records have been Upgraded...
    validate();
}

PGExtended & BamMetadata::getProgramGroupExtended(const std::string & id)
{
    //std::map<std::string, PGExtended>::const_iterator iter = m_rgid2rgext.find(id);
    for(std::vector<PGExtended>::iterator pgiter = m_pgext.begin();
        pgiter != m_pgext.end();
        pgiter++)
    {
        if( (*pgiter).ID == id)
        {
            PGExtended & pg = (*pgiter);
            return pg;
        }
    }

    // Otherwise, create new from template.
    //PG const & pg = getProgramGroup(id);
    std::vector<PG> const & allpg = getPrograms();
    for (std::vector<PG>::const_iterator pgiter = allpg.begin();
         pgiter!=allpg.end();
         ++pgiter)
    {
        PG const& pg = *pgiter;
        if(pg.ID == id)
        {
            PGExtended pgext(pg);
            m_pgext.push_back(pgext); // Adds a copy of pgext...
            return m_pgext.back(); // returns reference to just added element
        }
    }

    std::string msg = "Program Group Not Found: "  + id;
    throw invalid_input_record("Program Group Not Found: ", id);
}

RGExtended & BamMetadata::getReadGroupExtended(const std::string & id)
{
    std::map<std::string, RGExtended>::iterator iter = m_rgid2rgext.find(id);
    if (iter != m_rgid2rgext.end())
        return iter->second;

    RG const & match = getReadGroup(id);
    // potentially throws read_group_not_found

    // Build new rgext
    RGExtended rgext(match);
    m_rgid2rgext.insert(std::pair<std::string, RGExtended>(id, rgext));
    //m_rgid2rgext[id] = rgext;
    return m_rgid2rgext[id];
}


void BamMetadata::initialize()
{
    // Promote RG records to RGExtended
    //std::vector<RG> const & comments = this.getReadGroups();
    // Promote special PG record to PGExtended (PG ID=0 - instrument generated)
    //std::vector<PG> const & programs = this.getPrograms();

    typedef std::vector<std::string> CommentArray;
    //typedef std::vector<RG> RGArray;

    CommentArray const & comments = getComments();

    // Parse Comments into RG and PG extra records.
    for (CommentArray::const_iterator comment = comments.begin();
         comment != comments.end();
         comment++)
    {
        // boost regex
        boost::match_results<std::string::const_iterator> m;
        if(!boost::regex_search(*comment,
                                m,
                                BAMEXTENDEDCOMMENT,
                                boost::match_default | boost::match_continuous))
        {
            LOG4CXX_DEBUG(g_log, "Non Metadata comment: " << *comment);
            // Skip other comments.
            continue;
        }

        // tokenize matching records
        std::vector<std::string> fields;
        string_util::tokenize((*comment).substr(4), "\t", fields);
        // NB: We know from regex_match that field contains at least one tab
        std::string const& type = fields[0];
        if(type.compare(0,3,"RG:") == 0)
        {
            std::string id = type.substr(3);
            try
            {
                RGExtended & rgext = getReadGroupExtended(id);
                rgext.parse(fields);
                m_foundExtended = true;
            }
            catch (read_group_not_found const & e)
            {
                LOG4CXX_DEBUG(g_log, e.what());
                LOG4CXX_ERROR(g_log, "Extended Metadata Comment found without matching ReadGroup:" << *comment);
            }
        }
        else if (type.compare(0,3,"PG:") == 0)
        {
            std::string id = type.substr(3);
            // CO PG:
            try
            {
                PGExtended & pgext = getProgramGroupExtended(id);
                pgext.parse(fields);
                m_foundExtended = true;
            }
            catch (std::exception const & e)
            {
                LOG4CXX_DEBUG(g_log, e.what());
                LOG4CXX_ERROR(g_log, "Extended Metadata Comment found without matching ProgramGroup:" << *comment);
            }
        }
        else
        {
            LOG4CXX_ERROR(g_log, "Invalid BAM Metadata Comment Syntax" << *comment);
        }
    }
}

void BamMetadata::validate()
{
    if(!m_foundExtended)
        return;

    // We found something, so make sure we found everything.

    // Ensure all readgroups have matching comment, if one does, all should
    // Iterate through RG from parent, and check all RGs are present.
    std::vector<RG> const & myrg = getReadGroups();
    for (std::vector<RG>::const_iterator rgiter = myrg.begin();
         rgiter != myrg.end();
         rgiter++)
    {
        RGExtended & rg = getReadGroupExtended( (*rgiter).ID );
        if( !rg.hasMetadata() )
        {
            LOG4CXX_ERROR(g_log, "RG Extended Record: " << rg.ID << " has incomplete Metadata");
            m_foundExtended = false; // Well, we found something, but it was incomplete.
            return;
        }
    }

    // TODO: Check PG - Just check extended PG for full metadata
    for (std::vector<PGExtended>::const_iterator pgiter = m_pgext.begin();
         pgiter != m_pgext.end();
         pgiter++)
    {
        if(!(*pgiter).hasMetadata())
        {
            LOG4CXX_ERROR(g_log, "PG Extended Record: " << (*pgiter).ID << " has incomplete Metadata");
            m_foundExtended = false; // Well, we found something, but it was incomplete.
            return;
        }
    }

    LOG4CXX_INFO(g_log, "ALL BAM Metadata Processed Successfully");
    return;
}

// Overload getLibType to use extended header
LibraryType getLibType(RGExtended const& rg)
{
    if (rg.LT == "Fragment")
    {
        return LIBRARY_TYPE_FRAG;
    }
    else if (rg.LT == "MatePair")
    {
        return LIBRARY_TYPE_MP;
    }
    else if (rg.LT == "PairedEnd")
    {
        // fall back to LB/DS field to distinguish between RR and RRBC
        return getLibType(static_cast<RG const &>(rg));
        //return LIBRARY_TYPE_RR;
        // FIXME - use 2.0 headers to disambiguate
    }
    //else if (rg.LT == "PairedEnd")
    //{
    //    return LIBRARY_TYPE_RRBC;
    //}
    return LIBRARY_TYPE_NA;
}

std::ostream & operator<<(std::ostream &out, RGExtended const & rg)
{
    out << "@CO\tRG:" << rg.ID;
    // Mandatory fields
    out << "\tLT:" << rg.LT;
    out << "\tAT:" << rg.AT;
    out << "\tBX:" << rg.BX;
    out << "\tTN:" << rg.TN;
    out << "\tTX:" << rg.TX;
    if(rg.LT != "Fragment")
    {
        out << "\tBY:" << rg.BY;
        out << "\tUN:" << rg.UN;
        out << "\tUX:" << rg.UX;
        out << "\tIA:" << rg.IA;
        out << "\tIS:" << rg.IS;
        if(!rg.IN > 0) out << "\tIN:" << rg.IN;
        if(!rg.IM > 0) out << "\tIM:" << rg.IM;
    }
    if(!rg.SP.empty()) out << "\tSP:" << rg.SP;
    if(!rg.SD.empty()) out << "\tSD:" << rg.SD;
    if(!rg.SP.empty()) out << "\tSP:" << rg.SP;

    // Optional boolean field. - FIXME - suppress if not specified explicitly
    out << "\tEC:" << rg.EC;
    if(!rg.ER.empty()) out << "\tER:" << rg.ER;
    if(!rg.DE.empty()) out << "\tDE:" << rg.DE;
    if(!rg.CO.empty()) out << "\tCO:" << rg.CO;
    if(!rg.UU.empty()) out << "\tUU:" << rg.UU;
    if(!rg.PN.empty()) out << "\tPN:" << rg.PN;
    if(!rg.PJ.empty()) out << "\tPJ:" << rg.PJ;
    if(!rg.SO.empty()) out << "\tSO:" << rg.SO;
    out << std::endl; // Include endline - mandatory part of header

    if(rg.CU > 0) out << "\tCU:" << rg.CU;
    if(rg.CT > 0) out << "\tCT:" << rg.CT;

    // Output RG record using operator<< for base class
    out << dynamic_cast<RG const &>(rg);
    return out;
}

} //namespace lifetechnologies

