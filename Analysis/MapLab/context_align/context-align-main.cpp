/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>
#include <csignal>

#define GCC_VERSION (__GNUC__ * 10000  + __GNUC_MINOR__ * 100  + __GNUC_PATCHLEVEL__)

#if GCC_VERSION > 40600
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include <SamFile.h>

#if GVV_VERSION > 40600
#pragma GCC diagnostic pop
#endif

#include <myassert.h>
#include <cmdline_s.h>
#include <resource.h>
#include <rerror.h>
#include <tracer.h>
#include <timer.h>
#include <fasta.h>
#include <fileutils.h>
#include <print_batches.h>

#include <recreate_ref.h>
#include <cigar_utils.h>
#include "context-align-params.h"

#include <alphabet.h>
#include <translator.h>
#include <banded_convex_align.h>

#include <nalign.h>

#include <contalign.h>

static const unsigned MAX_SEQ_LEN = 1600;
static const unsigned MAX_RSEQ_LEN = 2000; // just a fragment is used, no need for full chromosome :)
static const unsigned MAX_BAND_WIDTH = 60;
static const unsigned REPORT_IVAL = 2;

// signal handling
static bool interrupted = false;
static int signal_received = -1;

void int_handler (int signum)
{
    interrupted = true;
    signal_received = signum;
}


class BamProcessor
{
    const ContalignParams* p_;

    SamFile infile_;
    SamFile outfile_;
    SamFileHeader sam_header_;
    SamRecord rec_;

    std::ofstream logfile_;

    unsigned band_width_;

// #ifdef TEMPLALIGN
    typedef genstr::WeightMatrix <char, int, float> NucSubstMatrix;
    NucSubstMatrix matrix_;

    typedef genstr::AffineGapCost<float> GapCost;
    GapCost gap_cost_;

    typedef genstr::BandedConvexAlign <char, char, char, int, float, float> Aligner;
    Aligner taligner_;
// #else
    Align aligner_;
// #endif
    ContAlign contalign_;

    ulonglong limit_;
    ulonglong skip_;
    Timer timer_;

    static const unsigned REF_BUF_INCR = 1000;
    size_t ref_buffer_sz_;
    MemWrapper <char> ref_buffer_;

    MemWrapper <BATCH> batches_;
    static const unsigned max_batch_no_ = 1000;

    time_t    begtime_;
    ulonglong read_cnt_;
    ulonglong proc_cnt_;
    ulonglong toolongs_;
    ulonglong unaligned_cnt_;
    ulonglong fail_cnt_;
    ulonglong nomd_cnt_;
    ulonglong realigned_cnt_;
    ulonglong modified_cnt_;
    ulonglong pos_adjusted_cnt_;

    bool log_diff_;
    bool log_matr_;
    bool log_base_;

public:
    BamProcessor ()
    :
    ref_buffer_sz_ (0)
    {
    }
    bool init (const ContalignParams& p);
    bool process ();
    bool processRecord ();
    bool finalize (bool success = true);
    void print_stats (std::ostream& o, bool multiline = true) const;
};


bool BamProcessor::init (const ContalignParams& p)
{
    read_cnt_ = proc_cnt_ = toolongs_ = unaligned_cnt_ = fail_cnt_ = nomd_cnt_ = realigned_cnt_ = modified_cnt_ = pos_adjusted_cnt_ = 0;
    log_diff_ = log_matr_ = log_base_ = false;

    p_ = &p;

    if (!*p.inbam ())
        ers << "Input file name not specified" << Throw;

    limit_ = p.limit ();
    skip_ = p.skip ();

    infile_.OpenForRead (p.inbam ());
    if (!infile_.IsOpen ())
        ers << p.inbam () << ThrowEx (FileNotFoundRerror);

    bool index_ok = false;
    if (*p.bamidx ())
    {
        index_ok = infile_.ReadBamIndex (p.bamidx ());
        if (!index_ok)
            warn << "Unable to open specified BAM index: " << p.bamidx () << ". Default index will be attempted" <<  std::endl;
    }
    if (!index_ok)
    {
        try
        {
            index_ok = infile_.ReadBamIndex ();
        }
        catch (std::exception& e)
        {
            // for some reason not converted into return status by libStatGen
        }
        if (!index_ok)
            warn << "Unable to open default BAM index for " << p.inbam () << std::endl;
    }
    if (*p.refname () || p.refno () != -1)
    {
        if (!index_ok)
            ers << "Reference section specified, but the BAM index could not be open." << Throw;
        if (*p.refname ())
        {
            if (p.endpos () != 0)
            {
                infile_.SetReadSection (p.refname (), p.begpos (), p.endpos ());
                info << "Read section set : " << p.refname () << ": " << p.begpos () << "-" << p.endpos () << std::endl;
            }
            else
            {
                infile_.SetReadSection (p.refname ());
                info << "Read section set : " << p.refname () << std::endl;
            }
        }
        else
        {
            if (p.endpos () != 0)
            {
                info << "Read section set : ref# " << p.refno () << ": " << p.begpos () << "-" << p.endpos () << std::endl;
                infile_.SetReadSection (p.refno (), p.begpos (), p.endpos ());
            }
            else
            {
                info << "Read section set : ref# " << p.refno () << std::endl;
                infile_.SetReadSection (p.refno ());
            }
        }
    }
    if (*p.outbam ())
    {
        if (!p.overwrite () && file_exists (p.outbam ()))
            ers << "Output file " << p.outbam () << " exists. Use --ov key to allow overwriting" << Throw;
        outfile_.OpenForWrite (p.outbam ());
        if (!outfile_.IsOpen ())
            ers << "Unable to open output file " << p.outbam () << std::endl;
    }
    if (*p.logfname ())
    {
        if (!p.overwrite () && file_exists (p.logfname ()))
            ers << "Log file " << p.logfname () << " exists. Use --ov key to allow overwriting" << Throw;
        logfile_.open (p.logfname (), std::fstream::out);
        if (!logfile_.is_open ())
            ers << "Unable to open log file " << p.logfname () << std::endl;

        time_t t = time (NULL);
        logfile_ << "Context-aware realigner log\nStarted at " << asctime (localtime (&t)) << "\nParameters:\n";
        logfile_ << *(p.parameters_);
        logfile_ << std::endl;
        log_base_ = p.logging ("base");
        log_diff_ = p.logging ("diff");
        log_matr_ = p.logging ("matr");
    }
    band_width_ = p.bwid ();

    switch (p.algo ())
    {
        case ContalignParams::TEMPL:
        {
            matrix_.configure (genstr::nucleotides.symbols (), genstr::nucleotides.size (), genstr::NegUnitaryMatrix <int, 4>().values ());
            gap_cost_.configure (p.gip (), p.gep ());
            taligner_.configure (&matrix_, &gap_cost_, &gap_cost_, &genstr::nn2num, &genstr::nn2num);
        }
        break;
        case ContalignParams::PLAIN:
        {
            batches_.reset (max_batch_no_);
            aligner_.init (MAX_SEQ_LEN, MAX_SEQ_LEN*MAX_BAND_WIDTH, p.gip (), p.gep (), p.mat (), -p.mis ());
            if (log_matr_)
                aligner_.set_log (logfile_);
            if (p.debug () > 5)
                aligner_.set_trace (true);
        }
        break;
        case ContalignParams::POLY:
        {
            batches_.reset (max_batch_no_);
            contalign_.init (MAX_SEQ_LEN, MAX_RSEQ_LEN, MAX_SEQ_LEN*MAX_BAND_WIDTH, p.gip (), p.gep (), p.mat (), -p.mis ());
            if (log_matr_)
                contalign_.set_log (logfile_);
            if (p.debug () > 5)
                contalign_.set_trace (true);
        }
        break;
        default:
        {
            ers << "Alignment algorithm " << p.algostr () << " not yet supported" << Throw;
        }
    }
    timer_.reset (DEFAULT_REPORT_IVAL, 1);
    return true;
}

bool BamProcessor::finalize (bool success)
{
    if (outfile_.IsOpen ())
    {
        trclog << "Closing output file" << std::endl;
        outfile_.Close ();
    }
    if (logfile_.is_open ())
    {
        time_t t = time (NULL);
        print_stats (logfile_);
        logfile_ << "\nFinished " << (success ? "successfully" : "due to error") << " at " << asctime (localtime (&t)) << "\n";
        trclog << "Closing log file" << std::endl;
        logfile_.close ();
    }
    if (info.enabled ())
        print_stats (info.o_);
    return true;
}

void BamProcessor::print_stats (std::ostream& o, bool multiline) const
{
    time_t elapsed = time (NULL) - begtime_;
    if (multiline)
    {
        o << "\nRealignment statistics:";
        o << "\n" << read_cnt_ << " reacords read in " << elapsed << " sec";
        if (elapsed)
            o << " (" << std::setprecision (1) << std::fixed << double (read_cnt_) / elapsed << " rec/sec)";
        if (proc_cnt_ != read_cnt_)
            o << "\n" << proc_cnt_ << " processed";
        if (toolongs_)
            o << "\n" << toolongs_ << " skipped as too long (over " << MAX_SEQ_LEN << " bases)";
        if (unaligned_cnt_)
            o << "\n" << unaligned_cnt_ << " skipped due to being unaligned in the input file";
        if (nomd_cnt_)
            o << "\n" << nomd_cnt_ << " skipped because of lacking MD info";
        if (fail_cnt_)
            o << "\n" << fail_cnt_ << " failed to align";
        o << "\n" << realigned_cnt_ << " successfully relaigned";
        o << "\n" << modified_cnt_ << " produced alignment different from original";
        o << "\n" << pos_adjusted_cnt_ << " produced alignment at altered reference location";
        o << std::endl;
    }
    else
    {
        o << read_cnt_ << " read in " << elapsed << "s, ";
        if (proc_cnt_ != read_cnt_)
            o << proc_cnt_ << " proc, ";
        if (toolongs_ + unaligned_cnt_ + nomd_cnt_)
            o << toolongs_ + unaligned_cnt_ + nomd_cnt_ << " skip, ";
        if (fail_cnt_)
            o << fail_cnt_ << " fail, ";
        if (realigned_cnt_ != proc_cnt_)
            o << realigned_cnt_ << " align, ";
        o << modified_cnt_ << " mod,";
        o << pos_adjusted_cnt_  << " shift";
    }
}

bool BamProcessor::process ()
{
    if (!infile_.ReadHeader (sam_header_))
        ers << "Unable to read SAM header" << Throw;
    else
        info << "Header read" << std::endl;

    if (outfile_.IsOpen ())
    {
        if (!outfile_.WriteHeader (sam_header_))
            ers << "Unable to write header data" << Throw;
        else
            info << "Header written" << std::endl;
    }

    // set up signal handlers
    sighandler_t sighandler_int, sighandler_term, sighandler_hup;
    // set INT handler to int_handler if interrupting is not disabled allready
    if ((sighandler_int = signal (SIGINT, int_handler)) == SIG_IGN)
        signal (SIGINT, SIG_IGN), sighandler_int = NULL;
    // set HUP handler to nothing
    sighandler_hup = signal (SIGHUP, SIG_IGN);
    // set TERM handler to int_handler if terminating is not disabled allready
    if ((sighandler_term = signal (SIGTERM, int_handler)) == SIG_IGN)
        signal (SIGTERM, SIG_IGN), sighandler_term = NULL;

    begtime_ = time (NULL);
    while (!infile_.IsEOF () && !interrupted)
    {
        if (limit_ && proc_cnt_ >= limit_)
        {
            info << limit_ << " records processed. Limit reached." << std::endl;
            break;
        }

        if (read_cnt_ == skip_)
            timer_.mark ();

        infile_.ReadRecord (sam_header_, rec_);
        ++ read_cnt_;
        if (read_cnt_-1 >= skip_)
        {
            if (!processRecord ())
                ++ fail_cnt_;
            ++ proc_cnt_;
            if (outfile_.IsOpen ())
                outfile_.WriteRecord (sam_header_, rec_);
        }
        if (timer_ ())
        {
            info << "\r" << read_cnt_; 
            if (proc_cnt_ != read_cnt_)
                info << " rd " << proc_cnt_;
            info << " pr ";
            if (realigned_cnt_ != proc_cnt_)
                info <<  realigned_cnt_ << " al (" << (double (realigned_cnt_) * 100 / proc_cnt_) << "%) ";
            info << modified_cnt_ << " mod (" << (double (modified_cnt_) * 100 / proc_cnt_) << "%) ";
            if (pos_adjusted_cnt_)
                info << pos_adjusted_cnt_ << " sh (" << (double (pos_adjusted_cnt_) * 100 / modified_cnt_) << "% mod) ";
            info << "in " << timer_.tot_elapsed () << " sec (" << std::setprecision (3) << std::fixed << timer_.speed () << " r/s)" << std::flush;
        }
    }
    if (interrupted)
    {
        errlog << "\nProcessing interrupted by ";
        switch (signal_received)
        {
            case SIGTERM:
                errlog << "TERM signal";
                break;
            case SIGINT:
                errlog << "user's request";
                break;
            default:
                errlog << "receipt of signal " << signal_received;
        }
        errlog << std::endl;
    }

    // restore signal handlers
    if (sighandler_term)
        signal (SIGTERM, sighandler_term);
    if (sighandler_int)
        signal (SIGINT, sighandler_int);
    if (sighandler_hup)
        signal (SIGHUP, sighandler_hup);

    return 0;
}

bool BamProcessor::processRecord ()
{
    trclog << "\nProcessing record " << read_cnt_ << " - " << rec_.getReadName () << ", " << rec_.get0BasedUnclippedEnd () << "->" << rec_.getReadLength () << ", ref " << rec_.getReferenceName () << std::endl;
    const char* seq = rec_.getSequence ();
    unsigned position = rec_.get0BasedPosition ();
    unsigned new_position = position;
    bool reverse_match = (rec_.getFlag () & 0x10);

    Cigar* cigar_p = rec_.getCigarInfo ();
    if (!cigar_p->size ())  // can not recreate reference is cigar is missing. Keep record unaligned.
    {                       // TODO: allow to specify and load external reference
        ++ unaligned_cnt_;
        return true;
    }

    myassert (cigar_p);

    const String *mdval = rec_.getStringTag ("MD");
    if (!mdval) // can not recreate reference is MD tag is missing. Keep record as is.
    {
        warn << "No MD Tag for record " << proc_cnt_ << ". Skipping record." << std::endl;
        ++nomd_cnt_;
        return true; // record will be kept as-is.
    }
    std::string md_tag = mdval->c_str ();

    // find the non-clipped region
    uint32_t clean_len;
    EndClips clips;
    const char* clean_read = clip_seq (seq, *cigar_p, clean_len, clips);

    // find length needed for the reference
    // this reserves space enough for entire refference, including softclipped ends.
    unsigned ref_len = cigar_p->getExpectedReferenceBaseCount ();
    if (ref_buffer_sz_ < ref_len)
    {
        ref_buffer_sz_ = (1 + ref_len / REF_BUF_INCR) * REF_BUF_INCR;
        ref_buffer_.reset (ref_buffer_sz_);
    }
    if (clean_len > MAX_SEQ_LEN || ref_len > MAX_SEQ_LEN)
    {
        ++ toolongs_;
        return true;
    }

    // recreate reference by Query, Cigar, and MD tag. Do not include softclipped ends in the recreated sequence (use default last parameter)
    recreate_ref (seq, rec_.getReadLength (), cigar_p, md_tag.c_str (), ref_buffer_, ref_buffer_sz_);

    unsigned qry_ins; // extra bases in query     == width_left
    unsigned ref_ins; // extra bases in reference == width_right
    band_width (*cigar_p, qry_ins, ref_ins);

    if (log_matr_ || log_base_)
    {
        logfile_ << "Record " << read_cnt_ << ": " << rec_.getReadName () << "\n"
                 << "   sequence (" << rec_.getReadLength () << " bases)\n";
    }

    CigarRoller roller;
    int ref_shift = 0;  // shift of the new alignment position on refereance relative the original
    unsigned qry_off, ref_off; // offsets on the query and reference of the first non-clipped aligned bases
    double new_score = 0;

    switch (p_->algo ())
    {
        case ContalignParams::TEMPL:
        {
            // call aligner
            new_score = taligner_.eval (clean_read, clean_len, ref_buffer_, ref_len, 0, band_width_);
            // read traceback
            // TODO: convert directly to cigar
            genstr::Alignment* al = taligner_.trace ();
            // convert alignment to cigar
            ref_shift = roll_cigar (roller, *al, clean_len, clips, qry_off, ref_off);
        }
        break;
        case ContalignParams::PLAIN:
        {
            new_score = aligner_.align_band (
                clean_read,                     // xseq
                clean_len,                      // xlen
                ref_buffer_,                    // yseq
                ref_len,                        // ylen
                0,                              // xpos
                0,                              // ypos
                std::max (clean_len, ref_len),  // segment length
                qry_ins + band_width_,          // width_left
                false,                          // unpack
                ref_ins + band_width_,          // width_right - forces to width_left
                true,                           // to_beg
                true                            // to_end
                );
            unsigned bno = aligner_.backtrace (
                    batches_,      // BATCH buffer
                    max_batch_no_, // size of BATCH buffer
                    false,         // fill the BATCH array in reverse direction
                    ref_ins + band_width_ // width
                                    );
            // convert alignment to cigar
            ref_shift = roll_cigar (roller, batches_, bno, clean_len, clips, qry_off, ref_off);
        }
        break;
        case ContalignParams::POLY:
        {
            new_score = contalign_.align_band (
                clean_read,                     // xseq
                clean_len,                      // xlen
                ref_buffer_,                    // yseq
                ref_len,                        // ylen
                0,                              // xpos
                0,                              // ypos
                std::max (clean_len, ref_len),  // segment length
                qry_ins + band_width_,          // width_left
                false,                          // unpack
                ref_ins + band_width_,          // width_right - forces to width_left
                true,                           // to_beg
                true                            // to_end
                );
            unsigned bno = contalign_.backtrace (
                    batches_,      // BATCH buffer
                    max_batch_no_, // size of BATCH buffer
                    false,         // fill the BATCH array in reverse direction
                    ref_ins + band_width_ // width
                                    );
            // convert alignment to cigar
            ref_shift = roll_cigar (roller, batches_, bno, clean_len, clips, qry_off, ref_off);
        }
        break;
        default:
        break;
    }
    ++realigned_cnt_;
    // compare original and new cigar (and location)
    if (ref_shift || !(*cigar_p == roller))
    {
        // save original cigar and position for reporting
        std::string orig_cigar_str;
        rec_.getCigarInfo ()->getCigarString (orig_cigar_str);
        int32_t prior_pos = rec_.get0BasedPosition ();

        // replace cigar
        rec_.setCigar (roller);
        ++ modified_cnt_;
        // update pos_adjusted_cnt if position changed
        if (ref_shift != 0)
        {
            myassert (prior_pos + ref_shift >= 0);
            rec_.set0BasedPosition (prior_pos + ref_shift);
            ++ pos_adjusted_cnt_;
        }
        if (log_diff_)
        {
            const unsigned MAX_BATCH_PRINTED = 100;
            BATCH batches [MAX_BATCH_PRINTED];
            std::string new_cigar_str;
            unsigned bno;
            int swscore;

            rec_.getCigarInfo ()->getCigarString (new_cigar_str);
            if (!log_base_ && !log_matr_)
                logfile_ << "Record " << read_cnt_ << ": " << rec_.getReadName () << " (" << rec_.getReadLength () << " bases)\n";

            logfile_ << "   ORIG ALIGNMENT:" << std::right << std::setw (9) << prior_pos+1 << "->" <<  orig_cigar_str << "\n";
            bno = cigar_to_batches (orig_cigar_str, batches, MAX_BATCH_PRINTED);
            swscore = align_score (batches, bno, clean_read, ref_buffer_, p_->gip (), p_->gep (), p_->mat (), p_->mis ());
            print_batches (clean_read, clean_len, false, ref_buffer_, ref_len, false, batches, bno, logfile_, false, prior_pos + clips.soft_beg_, clips.soft_beg_, 0, 160);
            logfile_ << "\n     'classic' SW score is " << swscore << "\n";

            logfile_ << "   NEW ALIGNMENT:" << std::right << std::setw (9) << rec_.get1BasedPosition () << "->" <<  new_cigar_str << std::endl;
            bno = cigar_to_batches (new_cigar_str, batches, MAX_BATCH_PRINTED);
            swscore = align_score (batches, bno, clean_read + qry_off, ref_buffer_ + ref_off, p_->gip (), p_->gep (), p_->mat (), p_->mis ());
            print_batches (clean_read + qry_off, clean_len - qry_off, false, ref_buffer_ + ref_off, ref_len - ref_off, false, batches, bno, logfile_, false, prior_pos + clips.soft_beg_ + ref_off, clips.soft_beg_ + qry_off, 0, 160);
            logfile_ << "\n      'classic' SW score is " << swscore;
            logfile_ << "\n      alternate (context-aware) score is " << new_score << ", used bandwidth left: " << qry_ins + band_width_ << ", right: " << ref_ins + band_width_ << "\n" << std::endl;
        }
        else if (log_base_)
        {
            logfile_ << "Recomputed alignment differs from original:\n";
            logfile_ << "   ORIG ALIGNMENT:" << std::right << std::setw (9) << prior_pos+1 << "->" <<  orig_cigar_str << "\n";
            std::string new_cigar_str;
            rec_.getCigarInfo ()->getCigarString (new_cigar_str);
            logfile_ << "    NEW ALIGNMENT:" << std::right << std::setw (9) << rec_.get1BasedPosition () << "->" <<  new_cigar_str << "\n" << std::endl;
        }
    }
    else
    {
        if (log_base_)
        {
            logfile_ << "Recomputed alignment matches the original:\n";
            std::string orig_cigar_str;
            rec_.getCigarInfo ()->getCigarString (orig_cigar_str);
            int32_t prior_pos = rec_.get0BasedPosition ();
            logfile_ << "   " << std::right << std::setw (9) << prior_pos+1 << "->" <<  orig_cigar_str << "\n" << std::endl;
        }
    }
    return true;
}


int process (const ContalignParams& p)
{
    BamProcessor processor;
    processor.init (p);
    try
    {
        processor.process ();
    }
    catch (...)
    {
        errlog << "\nTerminated due to error\n";
        processor.finalize (false);
        throw;
    }
    processor.finalize ();
    return 0;
}

const char* description_str = "Tool for local realignment of BAM records using context-sensitive aligner";

static void wait_key (int errcode, bool advice_throw = false)
{
    if (advice_throw)
        std::cout << std::endl << "(DBG mode) Unhandled exception caught. Program will print trace and exit with code " << errcode << " after pressing ENTER." << std::flush;
    else
        std::cout << std::endl << "(DBG mode) Processing done. Program will exit with code " << errcode << " after pressing ENTER" << std::flush;
    char c;
    std::cin.get (c);
}


int main (int argc, char* argv [])
{
    int rv = -1;
    bool debug = true;
    try
    {
        ContalignParams p;
        set_logging_level (Logger::ERROR);
        if (!p.parseCmdline (argc, argv))
        {
            if (!p.help_mode ())
            {
                errlog << "Error processing parameters ";
                p.cmdline ()->reportErrors (errlog.o_);
                errlog << std::endl;
                rv = 2;
            }
            else
                rv = 0;
            debug = false;
        }
        else if (!p.process ())
        {
            errlog << "Parameters processing failed" << std::endl;
            rv = 2;
            debug = false;
        }
        else
        {
            debug = p.debug ();

            if (p.debug () >= 5)
                set_logging_level (Logger::TRACE);
            else if (p.debug () >= 4)
                set_logging_level (Logger::DEBUG);
            else if (p.verbose ())
                set_logging_level (Logger::INFO);
            else
                set_logging_level (Logger::WARNING);

            dbglog << "Program parameters:\n";
            p.parameters_->log (dbglog);
            dbglog << std::endl;

            rv = process (p);
        }
    }
    catch (Rerror& err)
    {
        errlog << "\nError: " << err << std::endl;
        rv = 1;
    }
    catch (std::exception& e)
    {
        if (debug)
        {
            wait_key (rv, true);
            throw;
        }
        errlog << "\nSystem exception: " << e.what () << " Run in Debug mode to see trace" << std::endl;
    }
    catch (...)
    {
        if (debug)
        {
            wait_key (rv, true);
            throw;
        }
        errlog << "\nUnhandled exception caught, run in Debug mode to see trace" << std::endl;
    }
    if (debug)
        wait_key (rv, false);
    return rv;
}


