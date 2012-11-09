/*
 *	Created on: 1-04-2011
 *	Author: Jonathan Manning
 *
 *	Latest revision:  $Revision: 78911 $
 *	Last changed by:  $Author: manninjm $
 *	Last change date: $Date: 2011-01-04 18:28:27 -0500 (Tue, 04 Jan 2011) $
 *
 *	Copyright 2010 Life Technologies. All rights reserved.
 *	Use is subject to license terms.
 */

#ifndef BAM_METADATA_HPP_
#define BAM_METADATA_HPP_

#include <string>
#include <vector>
#include <stdint.h>

#include <samita/common/types.hpp>
#include <samita/sam/bam.hpp>

namespace lifetechnologies
{
    /*
     struct RGID {
     std::string name;
     };
     struct PGID {
     std::string name;
     };
     */

    /* Inherits from existing RG, provides extended per-RG data. */
    struct RGExtended: public RG
    {
            // Must have no-arg constructor to store in map
            RGExtended() :
                RG(),
                IA(0.0), IS(0.0),
                IN(0), IM(0),
                BX(false), TN(0), TX(0), BY(false), UN(0), UX(0),
                EC(false)
            {
                ;
            }

            RGExtended(RG const& parent) :
                RG(parent),
                IA(0.0), IS(0.0),
                IN(0), IM(0),
                BX(false), TN(0), TX(0), BY(false), UN(0), UX(0),
                EC(false)
            {
                ;
            }

            // Optional
            std::string IX; // IndexName
            std::string II; // IndexID
            std::string LD; // Library Details Description

            // Mandatory
            std::string LT; // LibraryType
            std::string AT; // ApplicationType

            float IA; // InsertAverage
            float IS; // Insert StdDev

            // Optional - conditional (required if paired)
            int32_t IN; // LibraryInsertSizeMinimum
            int32_t IM; // LibraryInsertSizeMaximum

            // Optional
            std::string SP; // SampleIdentifier
            std::string SD; // SequencingSampleDescription

            //Mandatory
            bool BX; // isBasePresent - tag1
            int32_t TN; // MinTrimmedReadLength - tag1
            int32_t TX; // NumColorCalls or NumBaseCalls - tag1

            bool BY; // isBasePresent - tag2
            int32_t UN; // MinTrimmedReadLength - tag2
            int32_t UX; // NumColorCalls or NumBaseCalls - tag2

            // Optional
            bool EC; // is ECC run (inferred)
            std::string ER; // ERCC
            std::string DE; // RunEndTime // DATE type?
            std::string CO; // Operator
            std::string UU; // LibraryIndexUUID
            std::string PN; // Application
            std::string PJ; // ProjectName
            std::string SO; // SampleOwner

            uint64_t CU; // InputReadCountUnfiltered
            uint64_t CT; // InputReadCountTotal

            void parse(std::vector<std::string> const & fields);
            bool hasMetadata() const;

    };

    /* Inherits from PG zero, stores instrument details */
    struct PGExtended: public PG
    {
            PGExtended(PG parent) :
                PG(parent)
            {
                ;
            }

            std::string AS; // AnalysisSoftware
            std::string PS; // InstrumentSerial
            //std::string PN; // InstrumentName // standard v1.3 header
            std::string PV; // InstrumentVendor
            std::string PM; // InstrumentModel

            void parse(std::vector<std::string> const & fields);
            bool hasMetadata() const;
    };

    /*!
     Class BamMetaData
     */
    class BamMetadata: public BamHeader
    {
            bool m_foundExtended;
            std::vector<RGExtended> m_rgext;
            std::map<std::string, RGExtended> m_rgid2rgext; // dictionary to make ReadGroup lookups faster
            std::vector<PGExtended> m_pgext;

        public:
            BamMetadata(const BamHeader & header);
             bool hasMetadata() const
            {
                return m_foundExtended;
            }
            PGExtended & getProgramGroupExtended(const std::string & id);
            RGExtended & getReadGroupExtended(const std::string & id);
        private:
            void initialize();
            void validate();
    };

    // Overload getLibType to use extended header
    LibraryType getLibType(RGExtended const& rg);

    std::ostream & operator<<(std::ostream &out, RGExtended const & rg);

} //namespace lifetechnologies

#endif //BAM_METADATA_HPP_
