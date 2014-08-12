/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <vector>
#include "OptArgs.h"
#include "Mask.h"
#include "NumericalComparison.h"
#include "Utils.h"
#include "H5File.h"
#include "H5Arma.h"

using namespace std;
using namespace arma;
/** 
 * Options about the two beadfind/separator results we're comparing.
 * query is the test results and gold is the current stable version by
 * convention.
 */
class SepCmpOpt {
public:
  SepCmpOpt() { 
    min_corr = 1.0;
    verbosity = 0;
    threshold_percent = .02;
  }
  string gold_dir, query_dir;
  double min_corr;
  double threshold_percent;
  string mode;
  int verbosity;
};

/**
 * Notes from a comparison, 
 */
class ComparisonMsg {

public:
  ComparisonMsg(const string &_name, double _min_corr, double _max_diff) : name(_name), 
                                                                    min_corr(_min_corr),
                                                                    max_diff(_max_diff) {
    equivalent = true;
  }

  void Append(const string &s) { msg += s; }
  void ToStr(ostream &o, int verbosity) {
    string status = "*FAILED*";
    if (equivalent) {
       status = "PASSED";
    }
    // 0 is no output
    if (verbosity >= 1 || !equivalent) {
      o << name << ":\t"  << status;
      o << endl;
      if (verbosity > 1) {
        if (cmp_names.size() > 0) {
          o << "   " << cmp_values[0].GetCount() << " entries." << endl;
        }
        for (size_t i = 0; i < cmp_names.size(); i++) {
          o << "   " << cmp_names[i] << "\t" << cmp_values[i].GetNumDiff() << "\t" << cmp_values[i].GetCorrelation() << endl;
        }
      }
    }
  }
    
  string name;
  string msg;
  bool equivalent;
  double min_corr;
  double max_diff;
  vector<string> cmp_names;
  vector<NumericalComparison<double> > cmp_values;
};

/* utility function for messages to console. */
int global_verbosity = 0;
void StatusMsg(const string &s) {
  if (global_verbosity > 0) {
    cout << s << endl;
  }
}

/* Is the query value significantly better than the gold. */
bool SigBetter(double query, double gold, double threshold_percent) {
  if (gold * (1 + threshold_percent) <= query) {
    return true;
  }
  return false;
}

/* Compare the masks from the two directories. */
void CompareMask(SepCmpOpt &sep, const string &mask_suffix,  ComparisonMsg &msg ) {
  string gold_file = sep.gold_dir + "/" + mask_suffix;
  string query_file = sep.query_dir + "/" + mask_suffix;
  Mask q_mask, g_mask;
  StatusMsg("Loading: " + gold_file);
  g_mask.LoadMaskAndAbortOnFailure(gold_file.c_str());
  StatusMsg("Loading: " + query_file);
  q_mask.LoadMaskAndAbortOnFailure(query_file.c_str());
  msg.name = "Beadfind Mask";
  msg.msg = "Results: ";
  msg.equivalent = true;
  if (g_mask.W() != q_mask.W() || g_mask.H() != g_mask.H()) {
    msg.equivalent = false;
    msg.Append("Masks are different sizes.");
    return;
  }
  msg.cmp_names.push_back("values");
  msg.cmp_names.push_back("library");
  msg.cmp_names.push_back("ignore");
  msg.cmp_names.push_back("empty");
  msg.cmp_values.resize(msg.cmp_names.size());
  size_t num_entries = g_mask.H() * g_mask.W();
  StatusMsg("Checking mask entries.");
  for (size_t i = 0; i < num_entries; i++) {
    short g = g_mask[i];
    short q = q_mask[i];
    msg.cmp_values[0].AddPair(g, q);
    msg.cmp_values[1].AddPair(g & MaskLib ? 1 : 0, q & MaskLib ? 1 : 0 );
    msg.cmp_values[2].AddPair(g & MaskIgnore ? 1 : 0, q & MaskIgnore ? 1 : 0);
    msg.cmp_values[3].AddPair(g & MaskEmpty ? 1 : 0, q & MaskEmpty ? 1 : 0);
  }
  
  for (size_t cmp_idx = 0; cmp_idx < msg.cmp_values.size(); cmp_idx++) {
    if (!msg.cmp_values[cmp_idx].CorrelationOk(msg.min_corr)) {
      msg.equivalent = false;
      msg.msg += " " + msg.cmp_names[cmp_idx] + " min: " + ToStr(msg.min_corr) + " got: " + ToStr(msg.cmp_values[cmp_idx].GetCorrelation());
    }   
  }
}

/* Compare the summary hdf5 values from the two directories. */
void CompareSummary(SepCmpOpt &opts, const string &summary_suffix, ComparisonMsg &msg) {
  string gold_file = opts.gold_dir + "/" + summary_suffix;
  string query_file = opts.query_dir + "/" + summary_suffix;
  
  Mat<float> gold_matrix, query_matrix;
  StatusMsg("Reading gold file: " + gold_file);
  H5Arma::ReadMatrix(gold_file + ":/separator/summary", gold_matrix);
  StatusMsg("Reading query file: " + query_file);
  H5Arma::ReadMatrix(query_file + ":/separator/summary", query_matrix);
  msg.name = "Summary Table";
  msg.msg = "Results:";
  msg.equivalent = true;
  if (gold_matrix.n_rows != query_matrix.n_rows) {
    msg.equivalent = false;
    msg.Append("Summary table are different sizes.");
    return;
  }
  msg.cmp_names.push_back("key");        //  0
  msg.cmp_names.push_back("t0");         //  1
  msg.cmp_names.push_back("snr");        //  2
  msg.cmp_names.push_back("mad");        //  3 
  msg.cmp_names.push_back("sd");         //  4
  msg.cmp_names.push_back("bf_metric");  //  5 
  msg.cmp_names.push_back("taub_a");     //  6 
  msg.cmp_names.push_back("taub_c");     //  7
  msg.cmp_names.push_back("taub_g");     //  8
  msg.cmp_names.push_back("taub_t");     //  9
  msg.cmp_names.push_back("peak_sig");   // 10 
  msg.cmp_names.push_back("flag");
  msg.cmp_names.push_back("good_live");
  msg.cmp_names.push_back("is_ref");
  msg.cmp_names.push_back("buffer_metric");
  msg.cmp_names.push_back("trace_sd");
  msg.cmp_values.resize(msg.cmp_names.size());
  size_t num_entries = gold_matrix.n_rows;
  StatusMsg("Checking summary entries.");
  for (size_t row_ix = 0; row_ix < num_entries; row_ix++) {
    for (size_t col_ix = 0; col_ix < msg.cmp_values.size(); col_ix++) {
      msg.cmp_values[col_ix].AddPair(gold_matrix.at(row_ix,col_ix), query_matrix.at(row_ix,col_ix));
    }
  }

  for (size_t cmp_idx = 0; cmp_idx < msg.cmp_values.size(); cmp_idx++) {
    if (!msg.cmp_values[cmp_idx].CorrelationOk(msg.min_corr)) {
      msg.equivalent = false;
      msg.msg += " " + msg.cmp_names[cmp_idx] + " min: " + ToStr(msg.min_corr) + " got: " + ToStr(msg.cmp_values[cmp_idx].GetCorrelation());
    }   
  }
}

/* some crumbs of documentation. */
void help_msg(ostream &o) {
  o << "SeparatorCmp - Program to compare separator results from different" << endl
    << "   versions of separator."
    << "options:" << endl
    << "   -h,--help        this message." << endl
    << "   -g,--gold-dir    trusted results to compare against [required]" << endl
    << "   -q,--query-dir   new results to check [required]" << endl
    << "   -c,--min-corr    minimum correlation to be considered equivalent [1.0]" << endl
    << "   -m,--mode        if 'research' output additional info ['exact']" << endl
    << "   --signficance    level for significantly better or worse in research mode [.02]" << endl
    << "   --verbosity      level of messages to print (higher is more verbose)" << endl
    ;
  exit(1);
}

/* Output some reseach statistics */
void OutputResearch(SepCmpOpt &opts, ComparisonMsg &mask_msg, ComparisonMsg &summary_msg) {
  const SampleStats<double> &gold_lib = mask_msg.cmp_values[1].GetXStats();
  const SampleStats<double> &query_lib = mask_msg.cmp_values[1].GetYStats();
  const SampleStats<double> &gold_ignore = mask_msg.cmp_values[2].GetXStats();
  const SampleStats<double> &query_ignore = mask_msg.cmp_values[2].GetYStats();
  const SampleStats<double> &gold_snr = summary_msg.cmp_values[2].GetXStats();
  const SampleStats<double> &query_snr = summary_msg.cmp_values[2].GetYStats();
  const SampleStats<double> &gold_mad = summary_msg.cmp_values[3].GetXStats();
  const SampleStats<double> &query_mad = summary_msg.cmp_values[3].GetYStats();
  const SampleStats<double> &gold_peak_sig = summary_msg.cmp_values[10].GetXStats();
  const SampleStats<double> &query_peak_sig = summary_msg.cmp_values[10].GetYStats();
  fprintf(stdout, "\nSeparator Results:\n");
  fprintf(stdout, "Name     Gold_Value     Query_Value      Change \n");
  fprintf(stdout, "----     ----------     -----------     --------\n");
  fprintf(stdout, "Lib      %10.2f      %10.2f      %6.2f%%\n", gold_lib.GetMean(), query_lib.GetMean(), query_lib.GetMean()/ gold_lib.GetMean() * 100.0f);
  fprintf(stdout, "Ignore   %10.2f      %10.2f      %6.2f%%\n", gold_ignore.GetMean(), query_ignore.GetMean(), query_ignore.GetMean()/ gold_ignore.GetMean() * 100.0f);
  fprintf(stdout, "SNR      %10.2f      %10.2f      %6.2f%%\n", gold_snr.GetMean(), query_snr.GetMean(), query_snr.GetMean()/ gold_snr.GetMean() * 100.0f);
  fprintf(stdout, "MAD      %10.2f      %10.2f      %6.2f%%\n", gold_mad.GetMean(), query_mad.GetMean(), query_mad.GetMean()/ gold_mad.GetMean() * 100.0f);
  fprintf(stdout, "Signal   %10.2f      %10.2f      %6.2f%%\n", gold_peak_sig.GetMean(), query_peak_sig.GetMean(), query_peak_sig.GetMean()/ gold_peak_sig.GetMean() * 100.0f);

  if ((SigBetter(query_lib.GetMean(), query_lib.GetMean(), opts.threshold_percent) && !SigBetter(gold_lib.GetMean(), query_lib.GetMean(), opts.threshold_percent)) || 
      (SigBetter(query_snr.GetMean(), query_snr.GetMean(), opts.threshold_percent) && !SigBetter(gold_snr.GetMean(), gold_snr.GetMean(), opts.threshold_percent)) ||
      (SigBetter(query_peak_sig.GetMean(), query_peak_sig.GetMean(), opts.threshold_percent) && !SigBetter(gold_peak_sig.GetMean(), gold_peak_sig.GetMean(), opts.threshold_percent)) ||
      (!SigBetter(query_mad.GetMean(), query_mad.GetMean(), opts.threshold_percent) && SigBetter(gold_mad.GetMean(), gold_mad.GetMean(), opts.threshold_percent))) {
    StatusMsg("Overall: **Better**");
  }
  else if ((!SigBetter(query_lib.GetMean(), query_lib.GetMean(), opts.threshold_percent) && SigBetter(gold_lib.GetMean(), query_lib.GetMean(), opts.threshold_percent)) || 
           (!SigBetter(query_snr.GetMean(), query_snr.GetMean(), opts.threshold_percent) && SigBetter(gold_snr.GetMean(), gold_snr.GetMean(), opts.threshold_percent)) ||
           (!SigBetter(query_peak_sig.GetMean(), query_peak_sig.GetMean(), opts.threshold_percent) && SigBetter(gold_peak_sig.GetMean(), gold_peak_sig.GetMean(), opts.threshold_percent)) ||
           (SigBetter(query_mad.GetMean(), query_mad.GetMean(), opts.threshold_percent) && !SigBetter(gold_mad.GetMean(), gold_mad.GetMean(), opts.threshold_percent))) {
    StatusMsg("Overall: **Worse**");
  }
  else {
    StatusMsg("Overall: **About Same**");
  }
  StatusMsg("");
}

/* Everybody's favorite function. */
int main (int argc, const char *argv[]) {
  const string mask_suffix = "separator.mask.bin";
  const string h5_suffix = "separator.h5";
  OptArgs o;
  bool all_ok = true;
  bool exit_help = false;
  SepCmpOpt opts;

  /* Get our command line options. */
  o.ParseCmdLine(argc, argv);
  o.GetOption(opts.gold_dir, "", 'g',"gold-dir");
  o.GetOption(opts.query_dir, "", 'q',"query-dir");
  o.GetOption(opts.min_corr, "1", 'c', "min-corr");
  o.GetOption(opts.verbosity, "1", '-', "verbosity");
  o.GetOption(opts.mode, "exact", 'm', "mode");
  o.GetOption(exit_help, "false", 'h', "help");
  
  global_verbosity = opts.verbosity;
  /* If something wrong or help, then help exit. */
  if (exit_help || opts.gold_dir.empty() || opts.query_dir.empty()) {
    help_msg(cout);
  }

  /* Check the masks. */
  ComparisonMsg mask_msg("mask_check", opts.min_corr, 0);
  mask_msg.min_corr = opts.min_corr;
  CompareMask(opts, mask_suffix, mask_msg);
  mask_msg.ToStr(cout,opts.verbosity);
  all_ok &= mask_msg.equivalent;

  /* Compare the summary tables. */
  ComparisonMsg summary_msg("summary_check", opts.min_corr, 0);
  mask_msg.min_corr = opts.min_corr;
  CompareSummary(opts, h5_suffix, summary_msg);
  summary_msg.ToStr(cout,opts.verbosity);
  all_ok &= summary_msg.equivalent;

  if (opts.mode == "research") {
    OutputResearch(opts, mask_msg, summary_msg);
  }
  /* If all was ok then return 0. */
  if (all_ok) {
    StatusMsg("Equivalent");
  }
  else {
    StatusMsg("Not Equivalent");
  }

  StatusMsg("Done.");
  return !all_ok;
}
