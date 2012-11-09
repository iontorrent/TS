/*
 *  Created on: 12-21-2009
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 98824 $
 *  Last changed by:  $Author: utirams1 $
 *  Last change date: $Date: 2011-12-19 15:32:07 -0800 (Mon, 19 Dec 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef SAMITA_TYPES_HPP_
#define SAMITA_TYPES_HPP_

#include <cassert>
#include <boost/cstdint.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>

#include <samita/common/interval.hpp>
#include <samita/common/quality_value.hpp>
#include <samita/exception/exception.hpp>
#include <samita/align/cigar.hpp>

#define NO_MAP_QUAL     255
#define NO_OFFSET       0
#define NO_COLOR_MM     -1
#define INDEL_NO_RANGE  -1
#define INDEL_NO_AMBIGUITY -1
#define NO_NH           -1

namespace lifetechnologies {

/*!
   Strand of an alignment
*/
enum Strand
{
    FORWARD,    /*!< Forward strand */
    REVERSE ,    /*!< Reverse strand */
    STRAND_NA     /*!< non relevent strand */
};

/*!
  Type of a library
*/
enum LibraryType
{
    LIBRARY_TYPE_NA,      /*!< Unknown */
    LIBRARY_TYPE_MP,      /*!< Mate pair */
    LIBRARY_TYPE_RR,      /*!< Reverse read */
    LIBRARY_TYPE_RRBC,    /*!< Reverse read barcode */
    LIBRARY_TYPE_FRAG     /*!< Fragment */
};

/*!
  Type of a read
*/
enum ReadType
{
    READ_TYPE_NA,       /*!< Unknown */
    READ_TYPE_F3,       /*!< F3 */
    READ_TYPE_R3,       /*!< R3 */
    READ_TYPE_F5_P2,    /*!< F5 P2 */
    READ_TYPE_F5_BC     /*!< F5 barcode */
};

/*!
 Class representing statistics for a read group
 */
class RGStats
{
public:
    RGStats() :
        m_nTotalBases(0), m_nMappedBases(0), m_nTotalReads(0),
        m_nMappedReads(0), m_nMappedReadsPairedInSequencing(0), m_nMappedReadsProperlyPaired(0),
        m_pctMismatchedBases(0.0), m_avgQualityMappedBases(0.0), m_meanInsertSize(0.0),
        m_sdInsertSize(0.0), m_medianInsertSize(0.0), m_adMedianInsertSize(0.0)
    {
    }

    bool empty() {return (m_nMappedReads == 0);}

    void setTotalBases(size_t n) {m_nTotalBases = n;}
    size_t getTotalBases() const {return m_nTotalBases;}

    void setMappedBases(size_t n) {m_nMappedBases = n;}
    size_t getMappedBases() const {return m_nMappedBases;}

    void setTotalReads(size_t n) {m_nTotalReads = n;}
    size_t getTotalReads() const {return m_nTotalReads;}

    void setMappedReads(size_t n) {m_nMappedReads = n;}
    size_t getMappedReads() const {return m_nMappedReads;}

    void setMappedReadsPairedInSequencing(size_t n) {m_nMappedReadsPairedInSequencing = n;}
    size_t getMappedReadsPairedInSequencing() const {return m_nMappedReadsPairedInSequencing;}

    void setMappedReadsProperlyPaired(size_t n) {m_nMappedReadsProperlyPaired = n;}
    size_t getMappedReadsProperlyPaired() const {return m_nMappedReadsProperlyPaired;}

    void setPctMismatchedBases(double d) {m_pctMismatchedBases = d;}
    double getPctMismatchedBases() const {return m_pctMismatchedBases;}

    void setAvgQualityMappedBases(double d) {m_avgQualityMappedBases = d;}
    double getAvgQualityMappedBases() const {return m_avgQualityMappedBases;}

    void setMeanInsertSize(double d) {m_meanInsertSize = d;}
    double getMeanInsertSize() const {return m_meanInsertSize;}

    void setSdInsertSize(double d) {m_sdInsertSize = d;}
    double getSdInsertSize() const {return m_sdInsertSize;}

    void setMedianInsertSize(double d) {m_medianInsertSize = d;}
    double getMedianInsertSize() const {return m_medianInsertSize;}

    void setAdMedianInsertSize(double d) {m_adMedianInsertSize = d;}
    double getAdMedianInsertSize() const {return m_adMedianInsertSize;}

protected:
    size_t m_nTotalBases;
    size_t m_nMappedBases;
    size_t m_nTotalReads;
    size_t m_nMappedReads;
    size_t m_nMappedReadsPairedInSequencing;
    size_t m_nMappedReadsProperlyPaired;

    double m_pctMismatchedBases;
    double m_avgQualityMappedBases;
    double m_meanInsertSize;
    double m_sdInsertSize;
    double m_medianInsertSize;
    double m_adMedianInsertSize;
};

/*!
 Class for read group
 */
class RG
{
public:
    RG()
        : ID(), SM(), LB(), DS(), PU(), PI(), CN(), DT(), PL(), PG(), FO(), KS()
    {
    }
    RG(std::string id, std::string sm, std::string lb, std::string ds, std::string pu, std::string pi, std::string cn,
            std::string dt, std::string pl)
        : ID(id), SM(sm), LB(lb), DS(ds), PU(pu), PI(pi), CN(cn), DT(dt), PL(pl), PG(), FO(), KS()
    {
    }
    ~RG()
    {
    }
    std::string ID;
    std::string SM;
    std::string LB;
    std::string DS;
    std::string PU;
    std::string PI;
    std::string CN;
    std::string DT;
    std::string PL;
    std::string PG; // new in SAM 1.3
    std::string FO; // ion flow order
    std::string KS; //ion flow tag key
    bool operator==(RG const& rg) const
    {
        return ((ID == rg.ID) && (SM == rg.SM) && (LB == rg.LB) && (DS == rg.DS) && (PU == rg.PU) && (PI == rg.PI)
                && (CN == rg.CN) && (DT == rg.DT) && (PL == rg.PL) && (PG == rg.PG));
    }
    bool operator!=(RG const& rg) const
    {
        return !(*this == rg);
    }
    bool equivalent(RG const& rg) const
    {
        return ((SM == rg.SM) && (LB == rg.LB) && (PI == rg.PI) && (CN == rg.CN));
    }
    RGStats Stats;
};

/*!
 Class for sequence dictionary entry
 */
class SQ
{
public:
    std::string SN; /// Sequence Name
    int LN;         /// Sequence Length
    std::string AS; /// Assembly Name
    std::string M5; /// MD5 Sum
    std::string UR; /// URL
    std::string SP; /// Species

    bool operator==(SQ const& sq) const
    {
        // Note: All user supplied fields can vary. 
        // Some are optional, others can be non-equal but still equivalent
        // Commented lines below compare equal only if both are present.
        // FIXME - Change to log warnings.
        //if(!AS.empty() && !sq.AS.empty() && (AS != sq.AS)) return false;
        //if(!UR.empty() && !sq.UR.empty() && (UR != sq.AS)) return false;
        //if(!SP.empty() && !sq.SP.empty() && (SP != sq.AS)) return false;
        // MD5 sums MUST be equal if both present
        if(!M5.empty() && !sq.M5.empty() && (M5 != sq.M5)) return false;
        // And SN and LN must always be equal.
        return ((SN == sq.SN) && (LN == sq.LN));

        // Absolute equality...
        //return ((SN == sq.SN) && (LN == sq.LN) && (AS == sq.AS) && (M5 == sq.M5) /* && (UR == sq.UR)*/ && (SP == sq.SP));
    }
    bool operator!=(SQ const& sq) const
    {
        return !(operator==(sq));
    }

    SequenceInterval getInterval() const
    {
        //SequenceInterval s = SequenceInterval(SN, 0, LN);
        return SequenceInterval(SN, 1, LN);
    }

    SQ & merge(SQ const & sq)
    {
        // Like operator=, but prefer non-empty fields.
        assert(*this == sq); // Must be otherwise equivalent
        if(!sq.AS.empty()) AS = sq.AS;
        if(!sq.M5.empty()) M5 = sq.M5;
        if(!sq.UR.empty()) UR = sq.UR;
        if(!sq.SP.empty()) SP = sq.SP;
        return *this;
    }
};

/*!
 Class for program group
 */
class PG
{
public:
    std::string ID; /// Program Record Identifier
    std::string VN; /// Version Number
    std::string CL; /// Command Line
    std::string PN; /// Program Name
    bool operator==(PG const& pg) const
    {
        return ((ID == pg.ID) && (VN == pg.VN) && (CL == pg.CL) && (PN == pg.PN));
    }
    bool operator!=(PG const& pg) const
    {
        return !(operator==(pg));
    }

    /// Extended API
    std::string getCommandLine() const { return CL; }
    std::string getProgramVersion() const { return VN; }
    std::string getProgramName() const { return PN; }

};


struct BamCleanup : public std::unary_function<bam1_t *&, void>
{
    void operator()(bam1_t *& b) const {
        bam_destroy1(b);
    }
};

typedef boost::shared_ptr<bam1_t> Bam1Ptr;

/*!
  Align class
 */
class Align
{
    public:
        Align() : m_dataPtr(), m_fileID(-1), seqs(NULL), qvarr(NULL) {;}

        // Re-use shared_ptr - most efficient, but trusts that bam1_t isn't modified
        Align(Bam1Ptr const & bt) : m_dataPtr(bt), m_fileID(-1), seqs(NULL), qvarr(NULL) {;}

        // NB: This does deep copy of bam, otherwise we may delete pointer owned by another
        explicit Align(bam1_t * bt) : m_dataPtr(bam_dup1(bt), BamCleanup() ), m_fileID(-1), seqs(NULL), qvarr(NULL)  {;}

        /// Compiler generated operator= also re-uses existing dataPtr.

        /// The copy constructor forces a deep copy.
        Align(Align const& other)
            : m_dataPtr(other.m_dataPtr.get() ? bam_dup1(other.m_dataPtr.get()) : /*NULL*/ other.m_dataPtr.get(), BamCleanup()),
              m_fileID(other.m_fileID)
        {seqs=NULL; qvarr= NULL;}

        /// This should be unnecessary given implementation above.
        /// @deprecated
        Align clone()
        {
            Align ret(*this);
            ret.setDataPtr(bam_dup1(m_dataPtr.get())); // Force copy of bam1_t record
            return ret;
        }

        virtual ~Align() {
		if (seqs!=NULL) delete [] seqs; 
		if (qvarr != NULL) delete [] qvarr;
	}

        void setDataPtr(Bam1Ptr const & bt)
        {
            m_dataPtr = bt;
        }
        void setDataPtr(bam1_t *bt)
        {
            m_dataPtr.reset(bt, BamCleanup() );
        }
        bam1_t * getBamPtr() const {return m_dataPtr.get();}
        Bam1Ptr getSharedBamPtr() const {return m_dataPtr;}

        int32_t getFileId() const
        {
            return m_fileID;
        }
        void setFileId(int32_t id)
        {
            m_fileID = id;
        }
        /// @deprecated - This is incorrect!
        ReadType getReadType(LibraryType lib) const
        {
            if (!shouldHaveMate() || isFirstRead())
                // This isn't valid! F3 isn't always first,
                // and an unpaired read isn't always F3
                return READ_TYPE_F3;
            switch (lib)
            {
                case LIBRARY_TYPE_RR:
                    return READ_TYPE_F5_P2;
                case LIBRARY_TYPE_RRBC:
                    return READ_TYPE_F5_BC;
                default:
                    return READ_TYPE_R3; // LIBRARY_TYPE_MP
            }
        }

        /** \brief Return 1-based reference ID
         * Caution/Warning/FIXME: setRefId takes 0-based reference ID?!, but this returns 1-based
         */
        int32_t getRefId() const
        {
            assert(m_dataPtr);
            if (m_dataPtr->core.tid >= 0)
                return m_dataPtr->core.tid + 1;
            return -1;
        }
        /** \brief Assign Reference ID
         *
         * Caution: getRefId returns 1-based reference ID?!
         */
        void setRefId(int32_t id)
        {
            assert(m_dataPtr);
            m_dataPtr->core.tid = id;
        }
        int32_t getStart() const {assert(m_dataPtr); return m_dataPtr->core.pos + 1;}
        int32_t getEnd() const {assert(m_dataPtr); return bam_calend(&m_dataPtr->core, bam1_cigar(m_dataPtr)) + 1;}
        int32_t getQual() const {assert(m_dataPtr); return m_dataPtr->core.qual;} // comes from MAPQ field, applies to pair is this has a mate
        int32_t getMapQual() const  // comes from SM tag, applies to this read if this read has a mate.
        {
            assert(m_dataPtr);
            //assert(shouldHaveMate()); // This will be controversial!
            uint8_t *mq = bam_aux_get(m_dataPtr.get(), "SM");
            if (mq != NULL)
                return bam_aux2i(mq);
            return NO_MAP_QUAL;
        }
        int32_t getFlag() const {assert(m_dataPtr); return m_dataPtr->core.flag;}
        int32_t getInsertSize() const {assert(m_dataPtr); return m_dataPtr->core.isize;}
        int32_t getMateStart() const {assert(m_dataPtr); return m_dataPtr->core.mpos + 1;}
        int32_t getMateRefId() const
        {
            assert(m_dataPtr);
            if (m_dataPtr->core.mtid >= 0)
                return m_dataPtr->core.mtid + 1;
            return -1;
        }
        void setMateRefId(int32_t id)
        {
            assert(m_dataPtr);
            m_dataPtr->core.mtid = id;
        }
        std::string getName() const {assert(m_dataPtr); return bam1_qname(m_dataPtr);}
        std::string getReadGroupId() const
        {
            assert(m_dataPtr);
            uint8_t *aux = bam_aux_get(m_dataPtr.get(), "RG");
            if (aux)
            {
                return bam_aux2Z(aux);
            }
            return "";
        }
        int32_t getColorMismatches() const
        {
            assert(m_dataPtr);
            uint8_t *mm = bam_aux_get(m_dataPtr.get(), "CM");
            if (mm != NULL)
                return bam_aux2i(mm);
            return NO_COLOR_MM;
        }
        int32_t getMismatches() const
        {
            assert(m_dataPtr);
            uint8_t *mm = bam_aux_get(m_dataPtr.get(), "NM");
            if (mm != NULL)
                return bam_aux2i(mm);
            return NO_COLOR_MM;
        }

        std::pair<int32_t, int32_t> getIndelRange() const
        {
            assert(m_dataPtr);
            uint8_t *qv_aux = bam_aux_get(m_dataPtr.get(), "XW");
            int start = INDEL_NO_RANGE;
            int end = INDEL_NO_RANGE;
            char underscore;

            if (qv_aux)
            {
                std::istringstream sstrm(bam_aux2Z(qv_aux));
                sstrm >> start;
                sstrm >> underscore;
                sstrm >> end;
            }
            return std::pair<int, int>(start, end);
        }

        int getIndelAmbiguity() const
        {
           assert(m_dataPtr);
           uint8_t *amb = bam_aux_get(m_dataPtr.get(), "XA");
           if (amb != NULL)
               return bam_aux2i(amb);
           return INDEL_NO_AMBIGUITY;
        }

        bool isMappedUnique() const
        {
            assert(m_dataPtr);
            uint8_t *hits = bam_aux_get(m_dataPtr.get(), "NH");
            if (hits != NULL)
                return (bam_aux2i(hits) == 1);
            return false;
        }

        int32_t getReportedAlignments() const
        {
            assert(m_dataPtr);
            uint8_t *hits = bam_aux_get(m_dataPtr.get(), "NH");
            if (hits != NULL)
                return bam_aux2i(hits);
            return NO_NH;
        }

        bool shouldHaveMate() const {assert(m_dataPtr); return (m_dataPtr->core.flag & BAM_FPAIRED);}
        bool isFirstRead() const {assert(m_dataPtr); assert(shouldHaveMate()); return (m_dataPtr->core.flag & BAM_FREAD1);}
        bool isSecondRead() const {assert(m_dataPtr); assert(shouldHaveMate()); return (m_dataPtr->core.flag & BAM_FREAD2);}
        bool isPrimary() const {assert(m_dataPtr); return !(m_dataPtr->core.flag & BAM_FSECONDARY);}
        bool isProperPair() const {assert(m_dataPtr); assert(shouldHaveMate()); return (m_dataPtr->core.flag & BAM_FPROPER_PAIR);}
        bool isMapped() const {assert(m_dataPtr); return !isUnmapped();}
        bool isUnmapped() const {assert(m_dataPtr); return (m_dataPtr->core.flag & BAM_FUNMAP);}
        bool isMateMapped() const {assert(m_dataPtr); assert(shouldHaveMate()); return !isMateUnmapped();}
        bool isMateUnmapped() const {assert(m_dataPtr); assert(shouldHaveMate()); return (m_dataPtr->core.flag & BAM_FMUNMAP);}
        //bool isNRMate() const {assert(m_dataPtr); return (m_dataPtr->core.flag & BAM_FDUP);}
        bool isPcrDuplicate() const {assert(m_dataPtr); return (m_dataPtr->core.flag & BAM_FDUP);}
        bool isQcFail() const { assert(m_dataPtr); return (m_dataPtr->core.flag & BAM_FQCFAIL); }
        bool isNRMate() const { return !isPcrDuplicate(); }
        bool isMismatchedChr() const {
            assert(m_dataPtr);
            return (getRefId() != getMateRefId());
        }

        Strand getStrand() const
        {
            assert(m_dataPtr);
            if (m_dataPtr->core.flag & BAM_FREVERSE)
                return REVERSE;
            return FORWARD;
        }
        Strand getMateStrand() const
        {
            assert(m_dataPtr);
            if (m_dataPtr->core.flag & BAM_FMREVERSE)
                return REVERSE;
            return FORWARD;
        }

        Cigar getCigar() const
        {
            return Cigar(m_dataPtr.get()); // Cache constructed value
        }

        // New in 2.0 - setters - Modify bam record
        void setPcrDuplicate(const bool dup)
        {
            assert(m_dataPtr);
            if (dup) m_dataPtr->core.flag |= BAM_FDUP; // set bit
            else m_dataPtr->core.flag &= ~BAM_FDUP; // clear bit
        }

        /*! \brief Compute effect of soft clipping on query string coordinates
         * (soft clip doesn't affect start/end, but does appear in query string.)
         * Valid operations are [[H] [S]] M (Hard must be outside soft, which must be outside match)
         * @return bases between start and position 0 of query string or end and end of query string.
         * @deprecated - Use cigar operation directly to avoid constructing multiple Cigar objects
         */
        int getOffset() const
        {
            assert(m_dataPtr);
            // Now delegated to Cigar
            const Strand & strand = getStrand();
            if(strand == FORWARD) {
                return getCigar().getOffsetForward();
            } else if(strand == REVERSE) {
                return getCigar().getOffsetReverse();
            }
            return 0; // STRAND_NA
        }

        /*! \brief Compute effect of soft and hard clipping on coordinates
         * When offset is applied to either reference position or query position
         * the origin of sequencing can be identified.
         * @return offset of origin of sequencing from start or end
         * @deprecated - Use cigar operation directly to avoid constructing multiple Cigar objects
         */
        int getOriginOffset() const
        {
            assert(m_dataPtr);
            // Now delegated to Cigar
            const Strand & strand = getStrand();
            if(strand == FORWARD) {
                return getCigar().getOriginOffsetForward();
            } else if(strand == REVERSE) {
                return getCigar().getOriginOffsetReverse();
            }
            return 0; // Strand NA?
        }


        std::string getSeq() const
        {
            assert(m_dataPtr);
            uint8_t *bam_seq = bam1_seq(m_dataPtr);
            std::ostringstream seq;

            if (bam_seq)
            {
                size_t qlen = m_dataPtr->core.l_qseq;
                for (size_t i=0; i<qlen; i++)
                    seq << bam_nt16_rev_table[bam1_seqi(bam_seq, i)];
            }
            return seq.str();
        }
	char *getSeqArr() 
	{
	    if (seqs) return seqs;
	    assert(m_dataPtr);
            uint8_t *bam_seq = bam1_seq(m_dataPtr);

            if (bam_seq)
            {
                size_t qlen = m_dataPtr->core.l_qseq;
		seqs = new char[qlen+1];
                for (size_t i=0; i<qlen; i++)
                    seqs[i] =  bam_nt16_rev_table[bam1_seqi(bam_seq, i)];
            }
            return seqs;
	}
        QualityValueArray getBaseQvs() const
        {
            assert(m_dataPtr);
            uint8_t *bam_qual = bam1_qual(m_dataPtr);
            QualityValueArray qvs;

            if (bam_qual)
            {
                size_t qlen = m_dataPtr->core.l_qseq;
                qvs.reserve(qlen);
                for (size_t i=0; i<qlen; i++)
                    qvs.push_back(bam_qual[i]);
            }
            return qvs;
        }
	unsigned char *getBaseQVsArr(int &length) 
	{
	    if (qvarr) { length = qvl; return qvarr;}
            assert(m_dataPtr);
            uint8_t *bam_qual = bam1_qual(m_dataPtr);

            if (bam_qual)
            {
                size_t qlen = m_dataPtr->core.l_qseq;
		qvarr = new unsigned char[qlen+1];
		length = qvl = qlen;
                for (size_t i=0; i<qlen; i++)
                    qvarr[i] = bam_qual[i];
            }
            return qvarr;
	}

        std::string getColorStr() const
        {
            assert(m_dataPtr);
            uint8_t *aux = bam_aux_get(m_dataPtr.get(), "CS");

            if (aux != NULL)
                return bam_aux2Z(aux);
            return "";
        }
        std::vector<uint8_t> getColorArray() const
        {
            assert(m_dataPtr);
            uint8_t *aux = bam_aux_get(m_dataPtr.get(), "CS");
            std::vector<uint8_t> colors;

            if (aux)
            {
                std::string csStr = bam_aux2Z(aux);
                colors.reserve(csStr.length());
                for (std::string::const_iterator iter = csStr.begin(); iter !=csStr.end(); ++iter)
                    colors.push_back(*iter);
            }
            return colors;
        }

        QualityValueArray getColorQvs() const
        {
            assert(m_dataPtr);
            uint8_t *aux = bam_aux_get(m_dataPtr.get(), "CQ");

            if (aux)
            {
                std::string qvsStr = bam_aux2Z(aux);
                return asciiToQvs(qvsStr);
            }
            return QualityValueArray();
        }
        std::string getColorQvsStr() const
        {
            assert(m_dataPtr);
            uint8_t *aux = bam_aux_get(m_dataPtr.get(), "CQ");
            if (aux != NULL)
                return bam_aux2Z(aux);
            return "";
        }

	std::string getFlowSignals() const
        {
	    std::stringstream str ;
	    str << "FZ:B:S";
	    bool flowSigFound = false;
            assert(m_dataPtr);
	    const char *tag = "FZ";
            uint8_t *s;
	    const bam1_t *b = m_dataPtr.get();
	    s = bam1_aux(b);
	    int y = tag[0]<<8 | tag[1];

	    while (s < b->data + b->data_len) {
                uint8_t type, key[2];
		uint8_t sub_type = ' ';
                key[0] = s[0]; key[1] = s[1];
		int x = (int)s[0]<<8 | s[1];

                s += 2; type = *s; ++s;

                
                if (type == 'A') {  ++s; }
                else if (type == 'C') {  ++s; }
                else if (type == 'c') {  ++s; }
                else if (type == 'S') {  s += 2; }
                else if (type == 's') {  s += 2; }
                else if (type == 'I') {  s += 4; }
                else if (type == 'i') {  s += 4; }
                else if (type == 'f') {  s += 4; }
                else if (type == 'd') {  s += 8; }
                else if (type == 'Z' || type == 'H') { while (*s) s++; ++s; }
                else if (type == 'B') {
                        sub_type = *(s++);
                        int32_t n;
                        memcpy(&n, s, 4);
                        s += 4; // no point to the start of the array
                        for (int32_t i = 0; i < n; ++i) {
                               
                                if ('c' == sub_type || 'c' == sub_type) {  ++s; }
                                else if ('C' == sub_type) {  ++s; }
                                else if ('s' == sub_type) {  s += 2; }
                                else if ('S' == sub_type) {  str << ","; str << *(uint16_t*)s; s += 2; }
                                else if ('i' == sub_type) {  s += 4; }
                                else if ('I' == sub_type) {  s += 4; }
                                else if ('f' == sub_type) {  s += 4; }
                        }
                }
		if (x==y && type == 'B' && sub_type == 'S') {flowSigFound = true; break;}
        }
            if (flowSigFound)
                return str.str();
            return "";
        }

    protected:
        Bam1Ptr m_dataPtr;
        int32_t m_fileID;
	char *seqs;
	unsigned char *qvarr;
	int qvl;
};

/*!
  AlignUDT class
 */
template<class UserDataType = void*>
class AlignUDT : public Align
{
    public:
        AlignUDT() : Align(), m_refID(-1), m_start(-1), m_end(-1), m_qual(NO_MAP_QUAL), m_flag(0), m_insertSize(0) {}

        AlignUDT(AlignUDT const& other) : Align(other)
        {
            m_refID = other.getRefId();
            m_start = other.getStart();
            m_end = other.getEnd();
            m_qual = other.getQual();
            m_flag = other.getFlag();
            m_insertSize = other.getInsertSize();
            m_name = other.getName();
            m_rgID = other.getReadGroupId();
            m_userData = other.getUserData();
        }
        AlignUDT(Align const& other) : Align(other)
        {
            m_refID = other.getRefId();
            m_start = other.getStart();
            m_end = other.getEnd();
            m_qual = other.getQual();
            m_flag = other.getFlag();
            m_insertSize = other.getInsertSize();
            m_name = other.getName();
            m_rgID = other.getReadGroupId();
        }

        virtual ~AlignUDT() {}

        void setBamPtr(bam1_t const* bt)
        {
            Align ac(bt);
            m_refID = ac.getRefId();
            m_start = ac.getStart();
            m_end = ac.getEnd();
            m_qual = ac.getQual();
            m_flag = ac.getFlag();
            m_insertSize = ac.getInsertSize();
            m_name = ac.getName();
            m_rgID = ac.getReadGroupId();
            Align::setDataPtr(bt);
        }

        UserDataType getUserData() const {return m_userData;}
        void setUserData(UserDataType dt) {m_userData = dt;}

        void setName(std::string name) {m_name = name;}
        std::string getName() const {return m_name;}
        void setReadGroupId(std::string id) {m_rgID = id;}
        std::string getReadGroupId() const {return m_rgID;}
        void setRefId(int32_t id) {m_refID = id;}
        int32_t getRefId() const {return m_refID;}
        void setStart(int32_t start) {m_start = start;}
        int32_t getStart() const {return m_start;}
        void setEnd(int32_t end) {m_end = end;}
        int32_t getEnd() const {return m_end;}
        void setQual(int32_t qual) {m_qual = qual;}
        int32_t getQual() const {return m_qual;}
        void setFlag(int32_t flag) {m_flag = flag;}
        int32_t getFlag() const {return m_flag;}
        void setInsertSize(int32_t size) {m_insertSize = size;}
        int32_t getInsertSize() const {return m_insertSize;}
    private:
        int32_t m_refID;
        int32_t m_start;
        int32_t m_end;
        int32_t m_qual;
        int32_t m_flag;
        int32_t m_insertSize;
        std::string m_name;
        std::string m_rgID;

        UserDataType m_userData;
};

/*!
  << operator for outputting a formatted Align to an output stream
 */
inline std::ostream &operator<< (std::ostream &out, Align const& ac)
{
    out << ac.getName() << "\t";
    out << ac.getFlag() << "\t";
    int32_t tid = ac.getRefId();
    if (tid >= 0)
    {
        //out header.getSequence(tid - 1).ID << "\t";
        out << tid << "\t";
    }
    else
        out << "*" << "\t";
    out << ac.getStart() << "\t";
    out << ac.getQual() << "\t";
    out << ac.getCigar().toString() << "\t";

    int32_t mtid = ac.getMateRefId();
    if (mtid >= 0)
    {
        if(mtid == tid) 
            out << "=" << "\t";
        else
            out << mtid << "\t";
        //out header.getSequence(mtid - 1).ID << "\t";
    }
    else
        out << "*" << "\t";
    out << ac.getMateStart() << "\t";
    out << ac.getInsertSize() << "\t";

    out << ac.getSeq() << "\t";

    out << qvsToAscii(ac.getBaseQvs());

    bam1_t const* bamPtr = ac.getBamPtr();
    assert(bamPtr);
    uint8_t *data= bamPtr->data;
    uint8_t *s = bam1_aux(bamPtr);
    int data_len = bamPtr->data_len;
    while (s < (data + data_len))
    {
        uint8_t type;
        out << "\t";
        out << (char)s[0] << (char)s[1] << ":";
        s += 2; type = *s; ++s;
        if (type == 'A') { out << "A:" << *s; ++s;}
        else if (type == 'C') { out << "i:" << (int)*s; ++s;}
        else if (type == 'c') { out << "i:" << (int)*s; ++s;}
        else if (type == 'S') { out << "i:" << (int)*s; s += 2;}
        else if (type == 's') { out << "i:" << (int)*s; s += 2;}
        else if (type == 'I') {out << "i:" << (int)*s; s += 4;}
        else if (type == 'i') { out << "i:" << (int)*s; s += 4;}
        else if (type == 'i') { out << "i:" << (int)*s; s += 4;}
        else if (type == 'f') { out << "f:" << (float)*s; s += 4;}
        else if (type == 'd') { out << "d:" << (double)*s; s += 8;}
        else if (type == 'Z' || type == 'H') {out << type << ":"; while (*s) out << *s++; ++s;}
    }

    return out;
}

// utility functions implemented elsewhere

/*!
  Get the library type for the specified string.
  \param library name
  \return a LibraryType
 */
LibraryType getLibType(std::string const& library);

/*!
  Get the library type for the specified read group.
  \param rg input read group
  \return a LibraryType
 */
LibraryType getLibType(RG const& rg);

/*!
  Get the category for a AlignCore record.
  \param ac input AlignCore record
  \param lib_type input library type
  \param max
  \param min
  \return the category string
 */
std::string getCategory(Align const &ac, LibraryType lib_type, int32_t max, int32_t min);

} // namespace lifetechnologies

#endif //SAMITA_TYPES_HPP_
