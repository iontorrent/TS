/*
 *  Created on: 04-14-20010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 87775 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-04-28 10:46:53 -0700 (Thu, 28 Apr 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef ALIGN_TEST_HPP_
#define ALIGN_TEST_HPP_

#include "cppunit/TestFixture.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"
#include <samita/common/interval.hpp>
#include <samita/align/align_reader.hpp>
#include <lifetech/string/util.hpp>

using namespace std;
using namespace lifetechnologies;

class StrandFilter
{
    public:
      StrandFilter(){}
        bool operator() (Align const &a) const
        {
            return (a.getStrand() == FORWARD);
        }
};

class MapQualityFilter
{
    public:
      MapQualityFilter(int qual=25) : m_qual(qual) {}
        bool operator() (Align const &a) const
        {
            return (a.getQual() >= m_qual);
        }
        void setQual(int qual) {m_qual=qual;}
    private:
        int m_qual;
};

class AlignTest: public CppUnit::TestFixture
{
    public:
        static CppUnit::Test *suite()
        {
            CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("AlignTest");

            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignIterationTest", &AlignTest::AlignIterationTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("multiBamAlignIterationTest", &AlignTest::multiBamAlignIterationTest));

            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignValueTest", &AlignTest::AlignValueTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignIteration_filterTest", &AlignTest::AlignIterationFilterTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("alignExtendedIterationTest", &AlignTest::alignExtendedIterationTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignSelectTest", &AlignTest::AlignSelectTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignSelectIntervalTest", &AlignTest::AlignSelectIntervalTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("AlignSelectEmptyRangeTest", &AlignTest::AlignSelectEmptyRangeTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("multiBamAlignSelectTest", &AlignTest::multiBamAlignSelectTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("bamHeaderTest", &AlignTest::bamHeaderTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("bamStatsTest", &AlignTest::bamStatsTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("bamHeaderMergeTest", &AlignTest::bamHeaderMergeTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("buildIndexTest", &AlignTest::buildIndexTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("parseRegionTest", &AlignTest::parseRegionTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("libTypeTest", &AlignTest::libTypeTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("multiBamAlignMergeTest", &AlignTest::multiBamAlignMergeTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("multiBamPartialAlignSelectTest", &AlignTest::multiBamPartialAlignSelectTest));
            suiteOfTests->addTest(new CppUnit::TestCaller<AlignTest>("doubleDereferenceTest", &AlignTest::doubleDereferenceTest));
            return suiteOfTests;
        }

        void setUp()
        {
        }

        void tearDown()
        {
            remove("test.sorted.a.bam.bai");
            remove("test.sorted.b.bam.bai");
        }

        //*************************************************************
        // test basic iteration of Align records over a known input
        //*************************************************************
        void AlignIterationTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 85;

            AlignReader sam(input);
            size_t nRecords = 0;

            vector< Align > alignments;
            alignments.reserve(nExpectedRecords);

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                //cout << *iter << endl;
                alignments.push_back(*iter);
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
            CPPUNIT_ASSERT_EQUAL(alignments.size(), nRecords);
            alignments.clear();

            // lets just try again for good measure
            nRecords = 0;
            iter = sam.begin();
            while(iter != sam.end()) {
                //cout << *iter << endl;
                alignments.push_back(*iter);
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
            CPPUNIT_ASSERT_EQUAL(alignments.size(), nRecords);
            alignments.clear();
        }

        //*************************************************************
        // test basic iteration of Align records over a known input
        //   filter out reads that are on reverse strand
        //*************************************************************
        void AlignIterationFilterTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 47;
            StrandFilter filter;
            AlignReader sam;
            sam.open(input);
            size_t nRecords = 0;

            AlignReader::filter_iterator<StrandFilter> iter(filter, sam.begin(), sam.end());
            AlignReader::filter_iterator<StrandFilter> end(filter, sam.end(), sam.end());

            while(iter !=end) {
                //cout << *iter << endl;
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test fields of Align records over a known input
        //*************************************************************
        void AlignValueTest()
        {
            const char* input = "data/test.bam";

            AlignReader sam;
            sam.open(input);

            AlignReader::const_iterator iter = sam.begin();
            AlignReader::const_iterator end = sam.end();
            size_t nRecords = 0;
            while(iter != end) {
                Align const& ac = *iter;
                //cout << ac << endl;
                // verify values for some records
                if (nRecords == 0)
                {
                    CPPUNIT_ASSERT_EQUAL(131, ac.getFlag());
                    CPPUNIT_ASSERT_EQUAL(1, ac.getRefId());
                    CPPUNIT_ASSERT_EQUAL(245, ac.getStart());
                    CPPUNIT_ASSERT_EQUAL(255, ac.getMapQual());
                    CPPUNIT_ASSERT_EQUAL(std::string("23M1I1M"), ac.getCigar().toString());
                }
                ++iter;
                nRecords++;
            }
        }

        //*************************************************************
        // test basic iteration of AlignExtended records over a known input
        //*************************************************************
        void alignExtendedIterationTest()
        {
            const char* input = "data/test.bam";
            const size_t nExpectedRecords = 85;
            float pi = 3.1415;
            typedef AlignUDT<float> UserDefinedAlign;

            AlignReader sam(input);
            size_t nRecords = 0;
            vector< UserDefinedAlign > alignments;

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end())
            {
                UserDefinedAlign a(*iter);
                //cout << a << endl;
                a.setUserData(pi);
                alignments.push_back(a);
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);

            // verify user data
            for (vector< UserDefinedAlign >::iterator iter=alignments.begin(); iter!=alignments.end(); ++iter)
            {
                UserDefinedAlign &align = *iter;
                CPPUNIT_ASSERT_EQUAL(pi, align.getUserData());
            }

            //test copy of user create Align
            AlignUDT<double> alignSrc;
            const int MAPQ = 10;

            alignSrc.setName("foo");
            alignSrc.setReadGroupId("bar");
            alignSrc.setRefId(0);
            alignSrc.setStart(1);
            alignSrc.setEnd(2);
            alignSrc.setQual(MAPQ);
            alignSrc.setFlag(0);
            alignSrc.setInsertSize(2);

            AlignUDT<double> alignDst(alignSrc);
            CPPUNIT_ASSERT_EQUAL(MAPQ, alignDst.getQual());
            CPPUNIT_ASSERT(alignDst.getBamPtr() == NULL);
        }

        //*************************************************************
        // test basic iteration of Align records over multiple known inputs
        //*************************************************************
        void multiBamAlignIterationTest()
        {
            const char* input1 = "data/test.S1.sorted.bam";
            const char* input2 = "data/test.S2.sorted.bam";
            const size_t nExpectedRecordsS1 = 36;
            const size_t nExpectedRecordsS2 = 31;

            int32_t lastStart = 0, lastRefId = -1;

            AlignReader sam;
            sam.open(input1);
            sam.open(input2);
            size_t nRecords = 0;

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                Align const& ac = *iter;
                //cout << ac << endl;
                if(ac.getRefId() == lastRefId)
                    CPPUNIT_ASSERT(ac.getStart() >= lastStart);
                lastRefId = ac.getRefId();
                lastStart = ac.getStart();
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL((nExpectedRecordsS1 + nExpectedRecordsS2), nRecords);

            // close a file and iterate again
            sam.close(input1);
            nRecords = 0;
            lastStart = -1;
            iter = sam.begin();
            while(iter != sam.end()) {
                Align const& ac = *iter;
                //cout << ac << endl;
                if(ac.getRefId() == lastRefId)
                    CPPUNIT_ASSERT(ac.getStart() >= lastStart);
                lastRefId = ac.getRefId();
                lastStart = ac.getStart();
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecordsS2, nRecords);

            // close all files and iterate again
            sam.close(input2);
            nRecords = 0;
            iter = sam.begin();
            while(iter != sam.end()) {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL((size_t)0, nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over a known input using a select
        //*************************************************************
        void AlignSelectTest()
        {
            const char* input = "data/test.sorted.bam";
            size_t nRecords = 0;

            // verify range selection
            const size_t nExpectedRecords = 12;

            AlignReader sam1(input);
            CPPUNIT_ASSERT(sam1.select("1:1000-10000"));
            nRecords = 0;

            AlignReader::iterator iter1 = sam1.begin();
            while(iter1 != sam1.end()) {
                //cout << *iter1 << endl;
                ++iter1;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);

            // try a range that fails on select
            AlignReader sam2(input);            
            CPPUNIT_ASSERT(! sam2.select("12345") );
            nRecords = 0;

            AlignReader::iterator iter2 = sam2.begin();
            while(iter2 != sam2.end()) {
                ++iter2;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL((size_t)0, nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over a known input using a selection interval
        //*************************************************************
        void AlignSelectIntervalTest()
        {
            const char* input = "data/test.sorted.bam";
            size_t nRecords = 0;

            // verify range selection
            const size_t nExpectedRecords = 12;

            AlignReader sam1(input);
            SequenceInterval interval("1", 1000, 10000);
            CPPUNIT_ASSERT(sam1.select(interval));
            nRecords = 0;

            AlignReader::iterator iter1 = sam1.begin();
            while(iter1 != sam1.end()) {
                //cout << *iter << endl;
                ++iter1;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over a known input using
        // a select with an empty range
        //*************************************************************
        void AlignSelectEmptyRangeTest()
        {
            const char* input = "data/test.sorted.bam";
            size_t nRecords = 0;

            // verify empty range selection
            const size_t nExpectedRecords = 14;

            AlignReader sam;
            sam.open(input);
            CPPUNIT_ASSERT(sam.select("2"));
            nRecords = 0;

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                //cout << *iter << endl;
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over multiple known inputs
        // using a select
        //*************************************************************
        void multiBamAlignSelectTest()
        {
            const char* input1 = "data/test.S1.sorted.bam";
            const char* input2 = "data/test.S2.sorted.bam";
            size_t nRecords = 0;

            // verify empty range selection
            const size_t nExpectedRecords = 9;

            AlignReader sam;
            sam.open(input1);
            sam.open(input2);
            CPPUNIT_ASSERT(sam.select("2:46000-50000"));
            nRecords = 0;

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                //cout << *iter << endl;
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over a known input using a select
        //  where the index needs to be built
        //*************************************************************
        void buildIndexTest()
        {
            const char* inputBam = "align/missing_index.sorted.bam";
            const char* inputIndex = "align/missing_index.sorted.bam.bai";
            size_t nRecords = 0;

            // verify range selection
            const size_t nExpectedRecords = 12;

            AlignReader sam(inputBam);
            CPPUNIT_ASSERT(sam.select("1:1000-10000"));
            nRecords = 0;

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                //cout << *iter << endl;
                ++iter;
                nRecords++;
            }
            remove(inputIndex);
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // test values from header
        //*************************************************************
        void bamHeaderTest()
        {
            const char* input = "data/test.bam";

            AlignReader sam(input);
            BamHeader const& hdr = sam.getHeader();

            // verify values from header
            CPPUNIT_ASSERT_EQUAL(BAM_SO_COORDINATE, hdr.getSortOrder());
            CPPUNIT_ASSERT_EQUAL((size_t)25, hdr.getSequenceDictionary().size());
            CPPUNIT_ASSERT_EQUAL((size_t)3, hdr.getComments().size());

            RG rg = hdr.getReadGroup("S1");
            CPPUNIT_ASSERT_EQUAL(std::string("NA19240"), rg.SM);
        }

        //*************************************************************
        // test values from header
        //*************************************************************
        void bamStatsTest()
        {
            const char* inputBam = "data/test.bam";
            const char* inputBas = "data/test.bas";

            AlignReader sam(inputBam);
            BamHeader hdr = sam.getHeader();

            hdr.addBamStats(inputBas);

            RG rg = hdr.getReadGroup("S1");

            CPPUNIT_ASSERT_EQUAL((size_t)100, rg.Stats.getTotalBases());
            CPPUNIT_ASSERT_EQUAL((size_t)2, rg.Stats.getMappedBases());
            CPPUNIT_ASSERT_EQUAL((size_t)3, rg.Stats.getTotalReads());
            CPPUNIT_ASSERT_EQUAL((size_t)4, rg.Stats.getMappedReads());
            CPPUNIT_ASSERT_EQUAL((size_t)5, rg.Stats.getMappedReadsPairedInSequencing());
            CPPUNIT_ASSERT_EQUAL((size_t)6, rg.Stats.getMappedReadsProperlyPaired());
            CPPUNIT_ASSERT_EQUAL(1.0, rg.Stats.getPctMismatchedBases());
            CPPUNIT_ASSERT_EQUAL(2.0, rg.Stats.getAvgQualityMappedBases());
            CPPUNIT_ASSERT_EQUAL(3.0, rg.Stats.getMeanInsertSize());
            CPPUNIT_ASSERT_EQUAL(4.0, rg.Stats.getSdInsertSize());
            CPPUNIT_ASSERT_EQUAL(5.0, rg.Stats.getMedianInsertSize());
            CPPUNIT_ASSERT_EQUAL(6.0, rg.Stats.getAdMedianInsertSize());
        }

        //*************************************************************
        // test basic iteration of Align records over multiple known inputs
        // using a select
        //*************************************************************
        void bamHeaderMergeTest()
        {
            const char* input1 = "data/test.S1.sorted.bam";
            const char* input2 = "data/test.S2.sorted.bam";

            AlignReader sam;
            sam.open(input1);
            sam.open(input2);

            BamHeader const& hdr = sam.getHeader();

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                Align const& a = *iter;
                //cout << a << endl;
                int32_t refID = a.getRefId();
                string rgID = a.getReadGroupId();
                RG const& rg = hdr.getReadGroup(rgID);
                SQ const& sq = hdr.getSequence(refID);
                CPPUNIT_ASSERT_EQUAL(rg.ID, rgID);
                if (refID == 1)
                    CPPUNIT_ASSERT_EQUAL(247249719, sq.LN);
                else if (refID == 2)
                    CPPUNIT_ASSERT_EQUAL(242951149, sq.LN);
                else if (refID == 3)
                    CPPUNIT_ASSERT_EQUAL(199501827, sq.LN);
                else if (refID == 4)
                    CPPUNIT_ASSERT_EQUAL(191273063, sq.LN);
                ++iter;
            }
        }

        void parseRegionTest()
        {
            const char* input = "data/test.bam";

            AlignReader sam(input);
            BamHeader const& hdr = sam.getHeader();

            int tid, begin, end;
            string rgn;

            rgn = "1";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(0, tid);
            CPPUNIT_ASSERT_EQUAL(0, begin);
            CPPUNIT_ASSERT_EQUAL(247249719, end);

            rgn = "5:";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(4, tid);
            CPPUNIT_ASSERT_EQUAL(0, begin);
            CPPUNIT_ASSERT_EQUAL(180857866, end);

            rgn = "M";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(24, tid);
            CPPUNIT_ASSERT_EQUAL(0, begin);
            CPPUNIT_ASSERT_EQUAL(16571, end);

            rgn = "2:10-1000";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(1, tid);
            CPPUNIT_ASSERT_EQUAL(10, begin);
            CPPUNIT_ASSERT_EQUAL(1000, end);

            rgn = "2:10";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(1, tid);
            CPPUNIT_ASSERT_EQUAL(10, begin);
            CPPUNIT_ASSERT_EQUAL(242951149, end);

            rgn = "2:10-";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(1, tid);
            CPPUNIT_ASSERT_EQUAL(10, begin);
            CPPUNIT_ASSERT_EQUAL(242951149, end);

            rgn = "2:1,000-100,000";
            tid = begin = end = -1;
            hdr.getSequenceRegion(rgn.c_str(), tid, begin, end);
            CPPUNIT_ASSERT_EQUAL(1, tid);
            CPPUNIT_ASSERT_EQUAL(1000, begin);
            CPPUNIT_ASSERT_EQUAL(100000, end);
        }

        void libTypeTest()
        {
            string lib;

            lib = "foo-50F";
            CPPUNIT_ASSERT_EQUAL(LIBRARY_TYPE_FRAG, getLibType(lib));

            lib = "foo-10x25MP";
            CPPUNIT_ASSERT_EQUAL(LIBRARY_TYPE_MP, getLibType(lib));

            lib = "foo-10x25RR";
            CPPUNIT_ASSERT_EQUAL(LIBRARY_TYPE_RR, getLibType(lib));

            lib = "foo-10x25RRBC";
            CPPUNIT_ASSERT_EQUAL(LIBRARY_TYPE_RRBC, getLibType(lib));

            lib = "foo-10x25HI";
            CPPUNIT_ASSERT_EQUAL(LIBRARY_TYPE_NA, getLibType(lib));
        }

        void multiBamAlignMergeTest()
        {
            const char* input1 = "data/test.sorted.a.bam";
            const char* input2 = "data/test.sorted.b.bam";
            const size_t nExpectedRecords1 = 66;
            const size_t nExpectedRecords2 = 19;

            int32_t lastStart = 0, lastRefId = -1;

            AlignReader sam;
            sam.open(input1);
            sam.open(input2);
            size_t nRecords = 0;
            BamHeader const& hdr = sam.getHeader();

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                Align const& ac = *iter;
                //cout << ac << endl;
                if (ac.isMapped())
                {
                    int32_t refID = ac.getRefId();
                    if(lastRefId == refID)
                        CPPUNIT_ASSERT(ac.getStart() >= lastStart);
                    lastRefId = refID;
                    lastStart = ac.getStart();

                    BamReader const& bam = sam.getBamReader(ac.getFileId());
                    if ((refID == 1) || (refID == 2))
                        CPPUNIT_ASSERT_EQUAL(std::string(input1), bam.getFilename());
                    else if ((refID == 3) || (refID == 4))
                        CPPUNIT_ASSERT_EQUAL(std::string(input2), bam.getFilename());

                    SQ const& sq = hdr.getSequence(refID);
                    if (refID == 1)
                        CPPUNIT_ASSERT_EQUAL(247249719, sq.LN);
                    else if (refID == 2)
                        CPPUNIT_ASSERT_EQUAL(242951149, sq.LN);
                    else if (refID == 4)
                        CPPUNIT_ASSERT_EQUAL(199501827, sq.LN);
                    else if (refID == 3)
                        CPPUNIT_ASSERT_EQUAL(191273063, sq.LN);
                }
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL((nExpectedRecords1 + nExpectedRecords2), nRecords);
        }

        //*************************************************************
        // test basic iteration of Align records over multiple known inputs
        // using a select that only returns records from one file
        //*************************************************************
        void multiBamPartialAlignSelectTest()
        {
            const char* input1 = "data/test.sorted.a.bam";
            const char* input2 = "data/test.sorted.b.bam";
            size_t nRecords = 0;

            // verify empty range selection
            const size_t nExpectedRecords = 15;

            AlignReader sam;
            sam.open(input1);
            sam.open(input2);
            CPPUNIT_ASSERT(sam.select("4")); // Only 1 returns false
            nRecords = 0;
            
            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                ++iter;
                nRecords++;
            }
            CPPUNIT_ASSERT_EQUAL(nExpectedRecords, nRecords);
        }

        //*************************************************************
        // double iterator dereference test
        //    this is a regression of a previous bug
        //*************************************************************
        void doubleDereferenceTest()
        {
            const char* input = "data/test.bam";

            AlignReader sam(input);

            AlignReader::iterator iter = sam.begin();
            while(iter != sam.end()) {
                Align const &a1 = *iter;
                Align const &a2 = *iter;
                CPPUNIT_ASSERT_EQUAL(a1.getSeq(), a2.getSeq());
                ++iter;
            }
        }

};

#endif //ALIGN_TEST_HPP_
