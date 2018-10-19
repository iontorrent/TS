# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# This file is based on the 4.0 file 'ERCC_analysis_plugin.py' - renamed because too similar another file name

from __future__ import division
import os
import sys
sys.path.insert(0, sys.argv[6]+"/code")
from string import Template
from math import log
from proc_coverage import SamCoverage
from proc_ercc import load_ercc_conc, dose_response, generate_reports
from ercc_counter import write_output_counts_file
from jqplot_param_generator import chart_series_params, generate_trendline_points, generate_color_legend
from create_summary import create_summary_block

PLUGIN_NAME = sys.argv[4]
PLUGIN_DIR = sys.argv[6]
DATA = PLUGIN_DIR+'/data'
SRC = PLUGIN_DIR+'/code'
RAW_SAM_FILE = sys.argv[1] + '/tmap.sam'
FILTERED_SAM_FILE = sys.argv[1] + '/filtered.sam'
OUTPUT_DIR = sys.argv[1] + '/'
COUNTS_FILE = OUTPUT_DIR + 'ercc.counts'
COUNTS_URL = 'ercc.counts'
GENOME = DATA + '/' + sys.argv[12]
ERCC_CONC = DATA + '/ERCCconcentration.txt'
ERCC_CONC_COLS = {'ERCC ID': str, 'Pool1': float, 'Pool2': float, 'log2pool1': float, 'log2pool2': float}
try:
  MINIMUM_RSQUARED = float(sys.argv[5])
except ValueError: 
  MINIMUM_RSQUARED = 0.9
try:
  MINIMUM_COUNTS = int(sys.argv[7])
except ValueError: 
  MINIMUM_COUNTS = 10
try:
  ERCC_POOL_NBR = int(sys.argv[8])
except:
  ERCC_POOL_NBR = 1
try:
  FASTQ_FILE_CREATED = sys.argv[9]
except:
  FASTQ_FILE_CREATED = 'N'
try:
  BARCODE_ENTERED = sys.argv[10]
except:
  BARCODE_ENTERED = 'none found'
try:
  ONLY_FWD_READS = (sys.argv[11] == 'Y')
except:
  ONLY_FWD_READS = False

print >> sys.stderr, "ERCC genome subset: "+GENOME

data_to_display = True
msg_to_user = ""  
  
if FASTQ_FILE_CREATED != 'N':
  coverage = SamCoverage(GENOME)

  transcript_names = []
  transcript_sizes = []
  with open(GENOME) as ercc_genome:
      for line in ercc_genome:
          parsed_line = line.split()
          transcript_names.append(parsed_line[0])
          transcript_sizes.append(parsed_line[1])
  # write counts file and filtered sam file, return stats tuple
  counts, all_ercc_counts, total_counts, mean_mapqs = write_output_counts_file(RAW_SAM_FILE,FILTERED_SAM_FILE,COUNTS_FILE,transcript_names,ONLY_FWD_READS)
  if total_counts > 0:
    percent_total_counts_ercc = '%.2f' % (100 * (all_ercc_counts / total_counts))
    percent_total_counts_non_ercc = 100 - float(percent_total_counts_ercc)
  # GDM: changed to trap filtering out of all reads
  if len(counts) > 2:
    coverage.parse_sam(FILTERED_SAM_FILE)
    ercc_conc = load_ercc_conc(filter = counts.keys(), pool = ERCC_POOL_NBR)
    ercc_conc.sort()
    dr = dose_response(coverage,ercc_conc,counts,MINIMUM_COUNTS)
    trendline_points = generate_trendline_points(dr)
  else:
    msg_to_user = "Insufficient ERCC targets detected after applying filters to mapped reads."
    data_to_display = False
    
else:
  msg_to_user = "The barcode you entered, "+BARCODE_ENTERED+", is not found.  This is most likely to result when the barcode is typed incorrectly, or the run was originally processed with a Torrent Suite version less than 3.4."
  data_to_display = False

if data_to_display:
  try:
    report_components = generate_reports(OUTPUT_DIR, coverage, dr, counts)
  except ValueError:
    msg_to_user = "There does not appear to be any ERCC reads."
    data_to_display = False

if data_to_display:
  if (all_ercc_counts < 250):
    msg_to_user = "The total number of counts (with acceptable mapping quality), "+str(all_ercc_counts)+", is not sufficient for a reliable correlation to be calculated."
    data_to_display = False
  elif (dr[8]<3): 
    msg_to_user = "The number of transcripts detected, "+str(dr[8])+", is not sufficient for a reliable correlation to be calculated."
    data_to_display = False

if data_to_display:
  transcript_names_js, transcript_images_js, transcript_counts_js, transcript_sizes_js, series_params, ercc_points = chart_series_params(counts,transcript_sizes,transcript_names,  mean_mapqs,ercc_conc)
  color_legend = generate_color_legend()
  # GDM: changed to pass filepath name for summary file
  SUMMARY_FILE = OUTPUT_DIR+PLUGIN_NAME+'_block.html'
  msg_to_user, rsquared = create_summary_block(SUMMARY_FILE,dr,MINIMUM_RSQUARED)
  template = open(SRC + '/ercc_template.html')
  page_we_are_making = Template(template.read())
  if MINIMUM_COUNTS > 1:
      counts_msg = "Transcripts with fewer than "+str(MINIMUM_COUNTS)+" counts (below the dashed line) are not used in calculating R-squared."
  if BARCODE_ENTERED != 'none found' and BARCODE_ENTERED != '':
    barcode = '('+BARCODE_ENTERED+')'
  else:
    barcode = ''
  print page_we_are_making.substitute(dose_response_image = report_components['dose_response_image'],
                                    results_divs = report_components['results_divs'],
                                    ercc_points = ercc_points,
                                    percent_total_counts_non_ercc = percent_total_counts_non_ercc,
                                    percent_total_counts_ercc = percent_total_counts_ercc,
                                    all_ercc_counts = all_ercc_counts,
                                    rsquared = rsquared,
                                    slope = '%.2f' % (dr[2]),
                                    yintercept = '%.2f' % (dr[3]),
                                    N = str(dr[8]),
                                    trendline_points = trendline_points,
                                    color_legend = color_legend,
                                    counts_file = COUNTS_URL,
                                    msg_to_user = msg_to_user,
                                    series = series_params,
                                    plugin_name = PLUGIN_NAME,
                                    log2_min_counts = log(MINIMUM_COUNTS,2),
                                    counts_msg = counts_msg,
                                    barcode = barcode )
else:
  template = open(SRC + '/ercc_error_template.html')
  page_we_are_making = Template(template.read())
  print page_we_are_making.substitute(msg_to_user = msg_to_user)
