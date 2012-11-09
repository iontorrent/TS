/*
 *  Created on: 10-28-2010
 *      Author: Jonathan Manning
 *
 *  Latest revision:  $Revision: 49984 $
 *  Last changed by:  $Author: edward_waugh $
 *  Last change date: $Date: 2010-10-01 14:54:43 -0400 (Fri, 01 Oct 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef PILEUP_TEST_HPP_
#define PILEUP_TEST_HPP_

#include <map>
#include <sstream>

#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>

#include "samita/align/align_reader.hpp"
#include "samita/common/quality_value.hpp"
#include "samita/pileup/pileup.hpp"
#include "samita/pileup/pileup_builder.hpp"
#include "samita/filter/filter.hpp"

using namespace std;
using namespace lifetechnologies;

class PileupTest: public CppUnit::TestFixture
{

            CPPUNIT_TEST_SUITE(PileupTest);
            CPPUNIT_TEST(pileupPositionTest);
            CPPUNIT_TEST(pileupIteratorTest);
            CPPUNIT_TEST(pileupTruncatedTest);
            CPPUNIT_TEST(pileupPathologicalTest);
            CPPUNIT_TEST_SUITE_END();
    public:

        void setUp()
        {
        }

        void tearDown()
        {
        }

        //*************************************************************
        // test basic iteration of Pileup records over a known input
        //*************************************************************
        void pileupIteratorTest()
        {
            const char* input = "data/test.sorted.bam";

            AlignReader sam(input);
            BamHeader & header = sam.getHeader();

            typedef PileupBuilder<AlignReader::iterator> UnfilteredPileup;

            UnfilteredPileup pb(sam.begin(), sam.end());
            int nExpectedPileup = 1431; // samtools pileup test.sorted.bam | wc -l
            int nPileup = 0;

            for(UnfilteredPileup::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++) {
                std::ostringstream oss;
                oss << (*plp)->getPileupStr(header) << std::endl;
                nPileup++;
            }

            CPPUNIT_ASSERT_EQUAL(nExpectedPileup, nPileup);
        }

        //*************************************************************
        // test deep coverage truncated file
        //*************************************************************
        void pileupTruncatedTest()
        {
            const char* input = "data/test.pileup2.bam";

            AlignReader sam(input);

            BamHeader & header = sam.getHeader();

            PileupBuilder<AlignReader::iterator> pb(sam.begin(), sam.end());
            int nExpectedPileup = 87; // samtools pileup input.bam | wc -l
            int nPileup = 0;

            //CPPUNIT_ASSERT(pb);

            for(PileupBuilder<AlignReader::iterator>::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++) {
                std::ostringstream oss;
                oss << (*plp)->getPileupStr(header) << std::endl;
                nPileup++;
            }

            CPPUNIT_ASSERT_EQUAL(nExpectedPileup, nPileup);
        }

        //*************************************************************
        // test pathological bam file with adjacent insertions and deletions
        //*************************************************************
        void pileupPathologicalTest()
        {
            const char* input = "data/test.pileup.bam";
            // Specifically crafted by JKB to break samtools-c

            AlignReader sam(input);

            BamHeader & header = sam.getHeader();

            PileupBuilder<AlignReader::iterator> pb(sam.begin(), sam.end());
            int nExpectedPileup = 14; // samtools pileup test.sorted.bam | wc -l == 12
            // samtools gets this one wrong, according to JK Bonfield
            int nPileup = 0;

            //CPPUNIT_ASSERT(pb);
            for(PileupBuilder<AlignReader::iterator>::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++) {
                std::ostringstream oss;
                oss << (*plp)->getPileupStr(header) << std::endl;
                nPileup++;
            }

            CPPUNIT_ASSERT_EQUAL(nExpectedPileup, nPileup);
        }

        //*************************************************************
        // test pathological ion BAM file - lots of CIGAR
        //*************************************************************
        void pileupIONTest()
        {
            const char* input = "data/Default.sorted.bam";

            int nExpectedPileup = 13; // samtools pileup test.sorted.bam | wc -l
            int nPileup = 0;

            AlignReader sam(input);
            BamHeader & header = sam.getHeader();
            PileupBuilder<AlignReader::iterator> pb(sam.begin(), sam.end());

            //CPPUNIT_ASSERT(pb);
            for(PileupBuilder<AlignReader::iterator>::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++) {
                std::stringstream oss;
                oss << (*plp)->getPileupStr(header) << std::endl;
                nPileup++;
            }

            CPPUNIT_ASSERT_EQUAL(nExpectedPileup, nPileup);
        }

        //*************************************************************
        // test if Pileup gets the correct coverage on a given position
        //*************************************************************
        void pileupPositionTest()
        {
            const char* input = "data/DH10B_6000.bam";

            AlignReader sam(input);
            BamHeader & header = sam.getHeader();

            bam_header_t * b_header = header.getRawHeader();
            CPPUNIT_ASSERT(b_header != NULL);
            StandardFilter filter;
            filter.setFilteringFlags(BAM_DEF_MASK); // Default samtools pileup flags, defined in bam.h
            typedef AlignReader::filter_iterator<StandardFilter> MyFilterIterator;
            PileupBuilder<MyFilterIterator> pb(MyFilterIterator(filter, sam.begin(), sam.end()), MyFilterIterator(filter, sam.end(), sam.end()));

            map <int32_t, int32_t> pileupcount;

            // FIXME - popen to samtools pileup directly?
            std::ifstream goodpileup("data/DH10B_6000.pileup.txt");
            CPPUNIT_ASSERT(goodpileup.is_open());
            int32_t pos, cov;
            std::string name, n, seq, qv;
            while(goodpileup >> name >> pos >> n >> cov >> seq >> qv)
            {
                pileupcount[pos] = cov;
            }
            //CPPUNIT_ASSERT(goodpileup.eof());
            goodpileup.close();

            for(PileupBuilder<MyFilterIterator>::pileup_iterator plp = pb.begin(); plp != pb.end(); plp++) {
                int32_t pos = (*plp)->pos();
                int32_t cov = getPileupCoverage(*((*plp).get()));
#ifdef SOLID_DEBUG
                if (pileupcount[pos] != cov) {
                    std::cerr << "at: " << pos << " expected coverage " << pileupcount[pos] << " actual cov " << cov << std::endl;
                    std::cerr << (*plp)->getPileupStr(header) << std::endl;
                    for(Pileup::iterator i = (*plp)->begin(); i != (*plp)->end(); ++i)
                    {
                        std::cerr << **i << std::endl; // Each PileupAlign
                    }
                    std::cerr << pb.getBuffer() << std::endl; // Full circular buffer
                }
#endif // SOLID_DEBUG
                CPPUNIT_ASSERT_EQUAL(pileupcount[pos], cov);

                // FIXME - compare complete strings
            }
        }

};

#endif //PILEUP_TEST_HPP

