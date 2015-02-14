# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import division
import os.path
import sys
import math
import numpy
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proc_coverage import SamCoverage, result_div
from coverage_plot import plot_coverage
from csvdb import CSVDB

DATA = sys.argv[6]+'/data'
GENOME   = DATA + '/ercc.genome'
ERCC_CONC = DATA + '/ERCCconcentration.txt'

ERCC_CONC_COLS = {'ERCC ID': str, 'Pool1': float, 'Pool2': float, 'log2pool1': float, 'log2pool2': float}


def load_ercc_conc(filter = None, pool = 1):
  """
  Load the ERCC concentration data.
    filter determines which erccs are loaded
    pool determines which concentration values are loaded
  """

  if pool == 1: pool = 'Pool1'
  else:         pool = 'Pool2'

  ercc_conc_db = CSVDB(ERCC_CONC, types = ERCC_CONC_COLS)

  if filter is None:
    rows = ercc_conc_db.filter(cols=['ERCC ID', 'Pool1'])
  else:
    rows = ercc_conc_db.filter(
      cols=['ERCC ID', pool],
      rows=[('ERCC ID', filter)]
      )

  return rows

def dose_response(coverage, ercc_conc, counts, min_counts=0):
  """
  Generate the x and y values for the scatter plot and compute the
  dose/response curve from the coverage and concentration values.
  """
  
  x = []
  y = []
  N = 0

  for transcript in ercc_conc:
    ercc, conc = transcript
    if counts[ercc] >= min_counts:
      x.append(math.log(conc,2))
      y.append(math.log(counts[ercc],2))
      N += 1                        

  m, b, r, p, err = scipy.stats.linregress(x, y)

  return x, y, m, b, r, r**2, p, err, N


def plot_dose_response(x, y, m, b, r, r2, p, err, N, path):
  #todo: check for zero or null values in inputs, and
  #handle the error gracefully if so
  """
  Generate the dose/response plot.
  """
  plt.clf()
 
  stats_str =  """m   = %.4f
b   = %.2f
r   = %.2f
r^2 = %.2f
p   = %.2f
err = %.2f
N   = %d""" % (m, b, r, r2, p, err, N)

  xrange = (0.0, max(x) + (5 - max(x) % 5))
  yrange = (0.0, max(y) + (5 - max(y) % 5))

  plt.plot(xrange, m*numpy.array(xrange) + b)

  plt.plot(x, y, 'ro')
  plt.xlim(xrange)
  plt.ylim(yrange)
  plt.text(2, yrange[1] * .5, stats_str)
  plt.savefig(open(path, 'wb'), format='png')
  return


def generate_reports(outdir, coverage, dr, filtered_counts):

  dr_path = outdir+'dose_response.png'
  try:
    plot_dose_response(*(list(dr) + [dr_path]))
  except ValueError:
    raise

  dr_out = open(outdir+'dose_response.dat', 'w')
  dr_out.write("""m   = %.4f
b   = %.2f
r   = %.2f
r^2 = %.2f
p   = %.2f
err = %.2f
N   = %d""" % dr[2:])
  
  coverage_out = open(outdir+'coverage.dat', 'w')
  coverage.save_full_coverage(coverage_out)

  starts = coverage.start_site_iter()
  starts_out = open(outdir+'starts.dat', 'w')  
  coverage.save_full_coverage(starts_out, starts)
  

  report_components = {}
  report_components['dose_response_image'] = "<img src='dose_response.png' alt='dose response plot'>"
  proc_update = []
  results_divs = '<hr><div id="transcript_details">'
  for contig, starts, results in zip(coverage.coverage_iter(), coverage.start_site_iter(),
                                     coverage.proc_iter()):
    assert(contig[0] == starts[0] == results[0])
    template_vars = {}
    template_vars['contig_id'] = contig[0]
    try:
      template_vars['counts'] = filtered_counts[template_vars['contig_id']]
    except KeyError:
      template_vars['counts'] = 0
    if template_vars['counts'] == 0:
      continue
    template_vars['contig'] = contig[1]
    template_vars['starts']  = starts[1]

    proc_update.append('Processing'+template_vars['contig_id'])
    image_name = template_vars['contig_id'] + '.png'
    template_vars['image_path'] = image_name
    template_vars['length'] = results[1]['length']
    template_vars['coverage_depth'] = results[1]['contig_max'] - results[1]['contig_min']
    template_vars['contig_coverage'] = results[1]['contig_coverage']
    template_vars['contig_coverage_pct'] = results[1]['contig_coverage_pct']
    template_vars['start_site_coverage'] = results[1]['start_site_coverage']
    template_vars['start_site_coverage_pct'] = results[1]['start_site_coverage_pct']
    template_vars['unique_start_sites'] = results[1]['unique_start_sites']
    template_vars['start_site_complexity'] = results[1]['start_site_complexity']
    template_vars['contig_cv'] = results[1]['contig_cv']
    results_divs += (result_div % template_vars)
    plot_coverage(contig[1], starts[1], path = outdir+image_name)

  results_divs += '</div>'
  report_components['results_divs'] = results_divs
  report_components['updates'] = proc_update
  return report_components

if __name__=='__main__':

  coverage = SamCoverage(GENOME)

  if len(sys.argv) == 2: coverage.parse_sam(sys.argv[1])
  else:                  coverage.parse_sam(TEST_SAM)

  threshold = 1
  counts = {}
  for ercc, count in coverage.count_iter():
    if count >= threshold:
      counts[ercc] = count

  ercc_conc = load_ercc_conc(filter = counts.keys())

  dr = dose_response(coverage, ercc_conc)
  generate_reports(coverage, dr)
  
