/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <exception>

#include <extesting.h>
#include <comma_locale.h>
#include <test_suite.h>

#include <runtime_error.h>
#include <process_params.h>
#include <fileutils.h>
#include <time_counter.h>
#include <print_batches.h>

#include <CigarRoller.h>

#include <MDtag.h>
#include <recreate_ref.h>
#include <contalign.h>


// Actual test classes
class TestMdParse : public Test
{
public:
    TestMdParse ()
    :
    Test ("MDparse - parsing MD tag string (found in SAM and BAM files")
    {
    }
    bool process ()
    {
        const char* md_strings [] = {"6A10^TC0C4", "3CC^CCC0TGC^CC3^ATGC0ATGC5"};
        // const char* md_strings [] = {"6Z10^TC0C4"};

        for (unsigned idx = 0, testno = sizeof (md_strings) / sizeof (*md_strings); idx != testno; ++idx)
        {
            o_ << "\nTest " << idx << " of " << testno << std::endl;
            const char* md_string = md_strings [idx];
            o_ << "MD string is          : " << md_string << std::endl;
            MD md (md_string);
            o_ << "MD instance creatred  : " << md << std::endl;
            MDIterator mditer (md);
            o_ << "Iterating over md     : " << std::endl;
            unsigned p = 0;
            while (!mditer.done ())
            {
                const MD::Component curc = *mditer;
                o_ << "\t" << std::setw (3) << p++ << "\t" << curc << std::endl;
                ++mditer;
            }
        }
        return true;
    }
};

class TestRecreateRef : public Test
{
    struct RecrCase
    {
        const char* reference_;
        const char* query_;
        const char* cigarstr_;
        const char* mdstr_;
        RecrCase (const char* reference = NULL, const char* query = NULL, const char* cigarstr = NULL, const char* mdstr = NULL)
        :
        reference_ (reference),
        query_     (query),
        cigarstr_  (cigarstr),
        mdstr_     (mdstr)
        {
        }
    };

    bool try_recreate (const char* reference, const char* query, const char* cigarstr, const char* mdstr)
    {
        const unsigned RECR_BUFSZ = 100;

        char recreated [RECR_BUFSZ];

        unsigned qlen = strlen (query);
        o_ << "Query is              : " << query << "(" << qlen << " bp)" << std::endl;

        o_ << "Cigar string is       : " << cigarstr << std::endl;
        o_ << "MD string is          : " << mdstr << std::endl;

        CigarRoller cigar (cigarstr);
        std::string t;
        cigar.getCigarString (t);
        o_ << "Bootstrapped cigar    : " << t << std::endl;

        size_t recr_len = recreate_ref (query, qlen, &cigar, mdstr, recreated, RECR_BUFSZ-1);
        recreated [recr_len] = 0;
        o_ << "Recreated reference   : " << recreated << "(" << recr_len << " bp)" << std::endl;
        if (strcmp (recreated, reference) == 0)
            o_ << "    MATCH" << std::endl;
        else
        {
            o_ << "    MISMATCH" << std::endl;
            o_ << "Expected reference is : " << reference << "(" << strlen (reference) << " bp)" << std::endl;
        }
        return true;
    }
public:
    TestRecreateRef ()
    :
    Test ("RecRef - Reference sequence computing from BAM record")
    {
    }
    bool process ()
    {
        // ATGCAAATGCTGTGGCATGCCCGT
        // ||||||*|||||||||||++*|||
        // ATGCAACTGCTGTGGCAT--ACGT
        RecrCase cases [] = { 
            RecrCase ("ATGCAAAT", "ATGCCT", "4M2D2M", "4^AA0A1" ), 
            RecrCase ("ATGCAAATGCTGTGGCATGCCCGT", "ATGCAACTGCTGTGGCATACGT", "18M2D4M", "6A11^GC0C3"),
            RecrCase ("ATGCAAATGCTGTGGCATGCCCGT", "ATTTGCAACTGCTGTGGCATACGT", "2M2I16M2D4M", "6A11^GC0C3"),
            RecrCase ("ATGCAAATGCTGTGGCATGCCCGT", "ATTTGCAACTGCTGTGGCATACGT", "2M2I16M2D4M", "6A11^GCC3")
        };
        for (unsigned cidx = 0, csent = sizeof (cases) / sizeof (*cases); cidx != csent; ++ cidx)
        {
            const RecrCase& c = cases [cidx];
            o_ << "Test case " << cidx << " of " << csent - 1 << std::endl;
            try_recreate (c.reference_, c.query_, c.cigarstr_, c.mdstr_);
        }
        return true;
    }
};


class TestContAlign : public Test
{
public:
    TestContAlign ()
    :
    Test ("Contal - Context-Sensitive Nucleotide alignment")
    {
    }
    bool process ()
    {
        const unsigned max_batch_no = 100;
        BATCH batches [max_batch_no];

        // const char* xseq = "gaccgaaggagtagaaactttttcttcagcgaggcggccgagctgacgcaaacatgcagatctttgtgaagaccctcactggcaaaaccatcacccttgaggtcgagcccagtgacaccattgagaatgtcaaagccaaaattcaagacaaggagggtatcccacctgaccagcagcgtctgatatttgccggcaaacagctggaggatggccgcactctctcagactacaacatccagaaagagtccaccctgcacctggtgttgcgcctgcgaggtggcattattgagccttctctccgccagcttgcccagaaatacaactgcgacaagatgatctgccgcaagtgctatgctcgccttcaccctcgtgctgtcaactgccgcaagaagaagtgtggtcacaccaacaacctgcgtcccaagaagaaggtcaaataaggttgttctttccttgaagggcagcctcctgcccaggccccgtggccctggagcctcaataaagtgtccctttcattgactttgtaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        // unsigned x_seg_end = 551;
        // myassert (x_seg_end <= strlen (xseq));

        //const char xseq [] = "ACGCAAAAAATCGGCTGGTATTCGTGGCCAGAGAGTCGCGTCTAT";
        //const char yseq [] = "ATGCAAAAATCGCTGGTATTCGTGGCCAGAGAGTCCGGTCTAT";
        //const char xseq [] = "ACGCAAAAAATCGCTGAGTCG";
        //const char yseq [] = "ATGCAAAAATCGCTAGTCG";
        //const char xseq [] = "ACGCAAAAAATC";
        //const char yseq [] = "ATGCAAAAATC";
        // const char xseq [] = "TATGATATGTCTCCATACCCATTACAATCTCCAGCATTCCCCCTCAAACCTAAAGAAATATGTCTGATAAAAAGAGTTACTTTGATAGAATCGAATGAGGTT";
        // const char yseq [] = "TATGATATGTCTCCATACCCATTACAATCTCCAGCATTCCCCCTCAAACCTAAGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGTTT";

        // const char xseq [] = "GGACGAACCTGATCTCTTATACTAGTATCCTTAATCATTTTTATTGTCCCACAAACTAAACCTCCTTCGGACTCCTGCCTCACTCATTTGACACCAACCACCCAACTATCTATTAAACCTAGCCATGGCCATCCCCTTATGAGCGGGCGCAGTGATTATAGGCTTTCGCTCTAAGATTTAAAAAAAAAAAAAAAAAAAAAATGCC";
        // const char yseq [] = "GGACGAACCTGATCTCTTATACTAGTATCCTTAATCATTTTTATTGCCACAACTAACCTCCTCGGACTCCTGCCTCACTCATTTACACCAACCACCCAACTATCTATAAACCTAGCCATGGCCATCCCCTTATGAGCGGGCGCAGTGATTATAGGCTTTCGCTCTAAGATTAAAAATGCC";
        const char xseq [] = "GCGGGCGCAGTGATTATAGGCTTTCGCTCTAAGATTTAAAAAAAAAAAAAAAAAAAAAATGCC";
        const char yseq [] = "GCGGGCGCAGTGATTATAGGCTTTCGCTCTAAGATTAAAAATGCC";
        unsigned x_seg_beg = 0;
        unsigned x_seg_end = strlen (xseq);
        unsigned y_seg_beg = 0;
        unsigned y_seg_end = strlen (yseq);


        unsigned seg_len = std::max (x_seg_end - x_seg_beg, y_seg_end - y_seg_beg);

        unsigned width = 10;
        int width_right = -1;
        bool tobeg = true;
        bool toend = false;

        unsigned max_penalty = 12;

        unsigned max_seg_len = 1024;
        unsigned max_deviation = 20;
        unsigned max_query_len = 1024;

        // int gip = -3, gep = 0, match = 1, mism = -1;
        int gip = 5, gep = 2, match = 1, mism = -3;

        ContAlign aligner;

        unsigned max_size = (max_seg_len + 2 * max_deviation)*(1 + 2 * max_deviation);

        aligner.init (max_query_len, max_query_len, max_size, gip, gep, match, mism);
        aligner.set_trace (true);
        aligner.align_band (
                             xseq,      // xseq
                             x_seg_end - x_seg_beg, // xlen
                             yseq,      // yseq
                             y_seg_end - y_seg_beg, // ylen
                             x_seg_beg,  // xpos
                             y_seg_beg,  // ypos
                             seg_len,    // len
                             25, // std::min (max_penalty, seg_len), // width_left
                             false,      // unpack
                             5,         // width_right - forces to width_left
                             true,       // tobeg
                             true       // toend
                            );
        // int score = aligner_.get_last_score ();
        // backtrace (reverse fill the results array)
        int bno =  aligner.backtrace (
                             batches,  // BATCH *b_ptr
                             max_batch_no, // int max_cnt
                             false,               // bool reverse (default false)
                             0                   // width. ??should this be max_errors?
                            );
         std::cout << "Alignment done, " << bno << " batches" << std::endl;
         for (int i = 0; i < bno; ++i)
             std::cout << i << "\t" << batches [i] << std::endl;

         print_batches (xseq, strlen (xseq), false, yseq, strlen (yseq), false, batches, bno, std::cout, false);
         std::cout << std::endl;
         return true;
    }
};
class TestContAlignGIP : public Test
{
public:
    TestContAlignGIP ()
    :
    Test ("ContalGip - Context-Sensitive alignment with gap initiation scaling")
    {
    }
    bool process ()
    {
        const unsigned max_batch_no = 100;

        BATCH batches [max_batch_no];

        const char xseq [] = "CAGCCACAGGCTCCCAGACAT";
        const char yseq [] = "CAGCCACAGGTCCCAGACAT";
        unsigned x_seg_beg = 0;
        unsigned x_seg_end = strlen (xseq);
        unsigned y_seg_beg = 0;
        unsigned y_seg_end = strlen (yseq);


        unsigned seg_len = std::max (x_seg_end - x_seg_beg, y_seg_end - y_seg_beg);

        unsigned width = 10;
        int width_right = -1;
        bool tobeg = true;
        bool toend = false;

        unsigned max_penalty = 12;

        unsigned max_seg_len = 1024;
        unsigned max_deviation = 20;
        unsigned max_query_len = 1024;

        // int gip = -3, gep = 0, match = 1, mism = -1;
        int gip = 15, gep = 6, match = 1, mism = -9;

        ContAlign aligner;

        unsigned max_size = (max_seg_len + 2 * max_deviation)*(1 + 2 * max_deviation);

        aligner.init (max_query_len, max_query_len, max_size, gip, gep, match, mism);
        aligner.set_trace (true);
        aligner.set_scale (ContAlign::SCALE_GIP_GEP);
        aligner.align_band (
                             xseq,      // xseq
                             x_seg_end - x_seg_beg, // xlen
                             yseq,      // yseq
                             y_seg_end - y_seg_beg, // ylen
                             x_seg_beg,  // xpos
                             y_seg_beg,  // ypos
                             seg_len,    // len
                             25, // std::min (max_penalty, seg_len), // width_left
                             false,      // unpack
                             5,         // width_right - forces to width_left
                             true,       // tobeg
                             true       // toend
                            );
        // int score = aligner_.get_last_score ();
        // backtrace (reverse fill the results array)
        unsigned bno =  aligner.backtrace (
                             batches,  // BATCH *b_ptr
                             max_batch_no, // int max_cnt
                             false,               // bool reverse (default false)
                             0                   // width. ??should this be max_errors?
                            );
        std::cout << "Alignment done, " << bno << " batches" << std::endl;
        for (unsigned i = 0; i < bno; ++i)
            std::cout << i << "\t" << batches [i] << std::endl;

        print_batches (xseq, strlen (xseq), false, yseq, strlen (yseq), false, batches, bno, std::cout, false);
        std::cout << std::endl;
        return true;
    }
};

class SeqdataLibTest : public TestSuite
{
    TestMdParse        testMdParse;
    TestRecreateRef    testRecreateRef;
    // TestNalign         testNalign;
    TestContAlign      testContAlign;
    TestContAlignGIP   testContAlignGIP;

public:
    SeqdataLibTest (int argc, char* argv [], char* envp [] = NULL)
    :
    TestSuite (argc, argv, envp)
    {
    }
    void init ()
    {
        reg (testMdParse);
        reg (testRecreateRef);
        // reg (testNalign);
        reg (testContAlign);
        reg (testContAlignGIP);
    }
};

int main (int argc, char** argv)
{
    std::cerr.imbue (deccomma_locale);
    std::cout.imbue (deccomma_locale);
    ers.imbue (deccomma_locale);

    trclog.enable ();
    SeqdataLibTest test (argc, argv);
    bool retcode = !test.run ();
    std::cout << std::endl << "Press Enter to finish" << std::endl;
    getchar ();
    return retcode;

    return 0;
}

