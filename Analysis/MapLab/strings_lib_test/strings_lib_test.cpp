/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning (disable:4786)
#endif

#include <iostream>
#include <iomanip>
#include <time.h>
#include <string.h>

#include <alphabet.h>
#include <gap_cost_templ.h>
#include <align_templ.h>
#include <translator.h>
#include <gencode.h>
#include <full_history_tracer.h>
#include <convex_align.h>
#include <banded_convex_align.h>

#include "wmatrix.h"

#include <convex_align.h>
#include <fundamental_preprocessing.h>

using namespace genstr;

const char *nnseq1 = "ATGCAAATGCTGCCTGAATGTGGTCGTGTGAAAACTTTATTACGTAGCTGGCTGAGGAGTCCGCGCGGCGGGCTTTAGGGATGGATTCCTTGGAAGGCTCGAATAGAGAGAGATCGCTCGCGCGCGCGCCGCTGATTTGCTG";
const char *nnseq2 = "ATGCAAATGGCGCGCCTGCCTGAATGTGGTCGTGTGAAAACTTTATTACGTAGCTGGCTGAGGAGTCGCGGGCTTTAGGGATGGATTCCCGCGCGTTGGAAGGCTCGAATAGAGAGAGATCGCTCGCCGCTGATTTGCTG";

const char *nnseq1a = "ATGCAAACAATGCACC";
const char *nnseq2a = "ATGCAAAGGGGGGGGCAATGCACC";

const char *nnseq1b = "ATGCAAACAATGCACC";
const char *nnseq2b = "ATGCAAAGGCAATGCACC";

const char *aaseq1 = "MNLKVLKVIDGETVFAATGVVTGPPPIWMMRQAGRYLPEYRERRTGHLRTYLLK";
const char *aaseq2 = "MNLKSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYREHLERTYELK";

const char *aaseq1a = "MNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLKMNLKVLKVIDGETVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVTGPPPIETVREHLTYLLK";
const char *aaseq2a = "MNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELKMNLKSSTKLLKPVLKVIDGETVFPPPIWTVFAAFAAMRQAGRYLTGVVTGPPPIWMKVIDGETVPEYRERRTGHLRSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYTGVVPPIWMKVIDGEMMRQAGRYLPEYTGVVTGPPPIWMKVIDGETVREHLERTYELK";

const char* aaseq1b = "MNLKVRSTP";
const char* aaseq2b = "MNLKGGVRSTP";

const char *aaseq1c = "MNLKVLKVIDGETVFAATGVVTGPPPIWMMRQAGRYLPEYRERRDDFTYAVVTGHLRTYLLK";
const char *aaseq2c = "MNLKSSTKLLKPVLKVIDGETVFPPPIWMMRQAGRYLPEYREHLERTYELK";

const char *MATRIX_FILE_NAME = "blosum62";
const float GIP = (float) 4.0;
const float GEP = (float) 0.4;
const int DEFAULT_DEV = 10;

//static bool print_batches = false;
static bool print_batches = true;
static bool print_matrices = false;
//static bool print_matrices = true;

template <typename MatrixType, typename GapCostType, typename Trans1Type, typename Trans2Type, typename AlignerType>
void test_aligner (MatrixType& weight_matrix, GapCostType& gap_cost_evaluator1, GapCostType& gap_cost_evaluator2, Trans1Type& translator1, Trans2Type& translator2, AlignerType& aligner, const char* name, const char* seq1, const char* seq2)
{
    std::cout  << "Testing aligner: " << name << std::endl;
    aligner.configure (&weight_matrix, &gap_cost_evaluator1, &gap_cost_evaluator2, &translator1, &translator2);
    int l1 = strlen (seq1);
    int l2 = strlen (seq2);
    aligner.eval (seq1, l1, seq2, l2);
    Alignment* al = aligner.trace ();
    std::cout << "Aligned, max score " << aligner.weight () << " at " << aligner.bestPos1 () << ", " << aligner.bestPos2 () << ", " << al->size () << " batches" << std::endl;
    if (print_batches)
    {
        std::cout << "Batches: ";
        for (Alignment::iterator it = al->begin (); it != al->end (); it ++)
            std::cout << (*it).beg1 << ":" << (*it).beg2 << ":" << (*it).len << "  ";
        std::cout << std::endl;
    }
    if (print_matrices)
        aligner.print_trace_matrix (aligner.bestPos1 (), aligner.bestPos2 ());
}

template <typename AveType>
void test_nucleotide_alignments (const char* seq1, const char* seq2, const char* msg = NULL)
{
    std::cout << std::endl << "#### Nucleotide aligners";
    if (msg) std::cout << ". " << msg;
    std::cout << std::endl;

    AffineGapCost<AveType> gapEval (GIP, GEP);
    WeightMatrix<char, int, AveType> matrix;
    matrix.configure (nucleotides.symbols (), nucleotides.size (), NegUnitaryMatrix <int, 4>().values ());

    ConvexAlign <char, char, char, int, AveType, AveType> aligner;
    FullHistoryTracer <char, char, char, int, AveType, AveType> tracer;
    BandedConvexAlign <char, char, char, int, AveType, AveType> baligner;

    test_aligner (matrix, gapEval, gapEval, nn2num, nn2num, aligner,  "ConvexAlign",       seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, nn2num, nn2num, tracer,   "FullHistoryTracer", seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, nn2num, nn2num, baligner, "BandedConvexAlign", seq1, seq2);
}

template <typename AveType>
void test_aminoacid_alignments (const char* seq1, const char* seq2, const char* msg = NULL)
{
    std::cout << std::endl << "#### Protein aligners";
    if (msg) std::cout << ". " << msg;
    std::cout << std::endl;

    AffineGapCost<AveType> gapEval (GIP, GEP);
    WeightMatrix<char, int, AveType> matrix;
    readProtMatrix (MATRIX_FILE_NAME, matrix);

    ConvexAlign <char, char, char, int, AveType, AveType> aligner;
    FullHistoryTracer <char, char, char, int, AveType, AveType> tracer;
    BandedConvexAlign <char, char, char, int, AveType, AveType> baligner;

    test_aligner (matrix, gapEval, gapEval, aa2num, aa2num, aligner,  "ConvexAlign",       seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, aa2num, aa2num, tracer,   "FullHistoryTracer", seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, aa2num, aa2num, baligner, "BandedConvexAlign", seq1, seq2);
}

template <typename AveType>
void test_translated_alignments (const char* seq1, const char* seq2, const char* msg = NULL)
{
    std::cout << std::endl << "#### Dynamically translated nucleotide aligners";
    if (msg) std::cout << ". " << msg;
    std::cout << std::endl;

    AffineGapCost<AveType> gapEval (GIP, GEP);
    WeightMatrix<char, int, AveType> matrix;
    readProtMatrix (MATRIX_FILE_NAME, matrix);
    CompositeTranslator <char, char, char> ctr (nn2num, standardGeneticCode);

    ConvexAlign <char, char, char, int, AveType, AveType> aligner;
    FullHistoryTracer <char, char, char, int, AveType, AveType> tracer;
    BandedConvexAlign <char, char, char, int, AveType, AveType> baligner;

    test_aligner (matrix, gapEval, gapEval, ctr, ctr, aligner,  "ConvexAlign",       seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, ctr, ctr, tracer,   "FullHistoryTracer", seq1, seq2);
    test_aligner (matrix, gapEval, gapEval, ctr, ctr, baligner, "BandedConvexAlign", seq1, seq2);
}


#define ENOUGH_TIME 10
#define ENOUGH_CYCLES 2

template <typename AveType>
void convex_aa_performance_test (const char* seq1, const char* seq2)
{
    std::cout << "   Speed test - ConvexAlign, prot" << std::endl;
    int l1 = strlen (seq1);
    int l2 = strlen (seq2);

    AffineGapCost<int> gapEval (GIP, GEP);
    WeightMatrix<char, int, AveType> matrix;
    readProtMatrix (MATRIX_FILE_NAME, matrix);

    ConvexAlign <char, char, char, int, AveType, int> aligner;
    aligner.configure (&matrix, &gapEval, &gapEval, &aa2num, &aa2num);
    int cycle = 0;
    time_t st = time (NULL);
    time_t ct = st;
    while (!(ct - st > ENOUGH_TIME && cycle > ENOUGH_CYCLES))
    {
        aligner.eval (seq1, l1, seq2, l2);
        ct = time (NULL);
        if (ct != st)
        {
            double speed = ((double) cycle) / (ct - st);
            std::cout << "\r" << std::setprecision (2) << std::fixed << speed << " cycles/sec [cycle " << cycle << ", " << ct - st << " sec]" << std::flush;
        }
        cycle ++;
    }
    std::cout << std::endl;
}

void aa_performance_test (const char* seq1, const char* seq2)
{
    std::cout << "   Speed test - AlignOpt, prot" << std::endl;
    int l1 = strlen (seq1);
    int l2 = strlen (seq2);

    AffineGapCost<int> gapEval (GIP, GEP);
    WeightMatrix<char, int, float> matrix;
    readProtMatrix (MATRIX_FILE_NAME, matrix);

    ConvexAlign<char, char, char, int, float, int > aligner;
    aligner.configure (&matrix, &gapEval, &gapEval, &aa2num, &aa2num);
    int cycle = 0;
    time_t st = time (NULL);
    time_t ct = st;
    while (!(ct - st > ENOUGH_TIME && cycle > ENOUGH_CYCLES))
    {
        aligner.eval (seq1, l1, seq2, l2);
        ct = time (NULL);
        if (ct != st)
        {
            double speed = ((double) cycle) / (ct - st);
            std::cout << "\r" << std::setprecision (2) << std::fixed << speed << " cycles/sec [cycle " << cycle << ", " << ct - st << " sec]" << std::flush;
        }
        cycle ++;
    }
    std::cout << std::endl;
}

void test_convex_align (const char* seq1, const char* seq2)
{
    std::cout << std::endl << "### Optimized aligner" << std::endl;
    int l1 = strlen (seq1);
    int l2 = strlen (seq2);

    WeightMatrix <> matrix;
    //readProtMatrix (MATRIX_FILE_NAME, matrix);
    matrix.configure (nucleotides.symbols (), nucleotides.size (), NegUnitaryMatrix <int, 4>().values ());

    AffineGapCost<int> gapEval (normGapWeight (&matrix, GIP) , normGapWeight (&matrix, GEP));

    // CompositeTranslator <char, char, char> ctr (nn2num, standardGeneticCode);

    ConvexAlign <char, char, char, int, double, int> aligner;
    aligner.configure (&matrix, &gapEval, &gapEval, &nn2num, &nn2num);
    aligner.eval (seq1, l1, seq2, l2);
    Alignment* al = aligner.trace ();
    std::cout << "Scored, score = " << aligner.weight () << " at " << aligner.bestPos1 () << ", " << aligner.bestPos2 () << ", " << al->size () << " batches" << std::endl;
    if (print_batches)
    {
        std::cout << "Batches: ";
        for (Alignment::iterator it = al->begin (); it != al->end (); it ++)
            std::cout << (*it).beg1 << ":" << (*it).beg2 << ":" << (*it).len << "  ";
        std::cout << std::endl;
    }
    if (print_matrices)
        aligner.print_trace_matrix (aligner.bestPos1 (), aligner.bestPos2 ());

}

void test_unithread_searchers ()
{
    //test_nucleotide_alignments <int>    (nnseq1, nnseq2, "Integer gaps");
    test_nucleotide_alignments <float>  (nnseq1, nnseq2, "Float gaps");
    //test_nucleotide_alignments <float>  (nnseq1, nnseq2, "Double gaps");
    //test_translated_alignments <int>    (nnseq1, nnseq2, "Integer gaps");
    test_translated_alignments <float>  (nnseq1, nnseq2, "Float gaps");
    //test_translated_alignments <double> (nnseq1, nnseq2, "Double gaps");
    //test_aminoacid_alignments  <int>    (aaseq1, aaseq2, "Integer gaps");
    test_aminoacid_alignments  <float>  (aaseq1, aaseq2, "Float gaps");
    //test_aminoacid_alignments  <double> (aaseq1, aaseq2, "Double gaps");
    //convex_aa_performance_test <double> (aaseq1a, aaseq2a);
    //opt_aa_performance_test (aaseq1a, aaseq2a);

    //test_convex_align_o (nnseq1, nnseq2);
}


void test_fundamental_preprocessing_simple ()
{
    typedef PrefixTable < char, unsigned long > StringPreprocessor;

    char pattern [] = "ATGCAAATATGCAATTGCATGCATGCAATGC";
    unsigned long table [sizeof (pattern)-1];

    // StringPreprocessor::make ( (char*) pattern, (unsigned long) sizeof (pattern)-1, (unsigned long *) table);
    StringPreprocessor::make ( pattern, sizeof (pattern)-1, table);

    unsigned idx;
    for (idx = 0; idx != sizeof (pattern) - 1; ++idx)
        std::cout << " " << std::setw (2) << pattern [idx];
    std::cout << std::endl;
    for (idx = 0; idx != sizeof (pattern) - 1; ++idx)
        std::cout << " " << std::setw (2) << table [idx];
    std::cout << std::endl;

}

void test_pattern_matchers ()
{
    test_fundamental_preprocessing_simple ();
}

static void wait_key (bool advice_throw = false)
{
    if (advice_throw)
        std::cout << std::endl << "Unhandled exception caught. Press ENTER to terminate application and see output of the system error handler." << std::flush;
    else
        std::cout << std::endl << "Processing done. Press ENTER to terminate application" << std::flush;
    char c;
    std::cin.get (c);
}


int main (int argc, char* argv [])
{
    try
    {
        try
        {
            test_unithread_searchers ();
            test_pattern_matchers ();
        }
        catch (Rerror& e)
        {
            std::cout << "Error: " << (const char*) e << std::endl;
        }
    }
    catch (...)
    {
        wait_key (true);
        throw;
    }

    wait_key (false);
    return 0;
}
