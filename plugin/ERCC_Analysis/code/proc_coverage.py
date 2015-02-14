# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Bed coverage took a loooong time and didn't work right (sorting issues?)
# This is a simple Python based coverage calculator

import numpy as np
import sys
import time

import ercc_seq_utils


# ------------------------------
# HTML Template
# ------------------------------

result_div = """
<div class="transcript">

<style>
  .toggle_tag {cursor:pointer}
</style>

<h2><span class="toggle_tag">[+]</span>%(contig_id)s (%(length)d bp)</h2>
<div class="transcript_table">
<table>
<tr><td>Reads</td><td>%(counts)d</td><td rowspan="6"><img src="%(image_path)s" alt="%(contig_id)s"/> </td></tr>
<tr class="alt"><td>Coverage Depth   </td><td>%(coverage_depth)d</td></tr>
<tr><td>Coverage</td><td>%(contig_coverage)d (%(contig_coverage_pct).2f%%)</td></tr>
<tr class="alt"><td>Start Sites</td><td>%(start_site_coverage)d (%(start_site_coverage_pct).2f%%)</td></tr>
<tr><td>Unique Start Sites</td><td>%(unique_start_sites)d (%(start_site_complexity).2f%%)</td></tr>
<tr class="alt"><td>Coverage CV</td><td>%(contig_cv).4f</td></tr>
</table>
</div>
</div>
"""

# ------------------------------
# Stat procedures
# ------------------------------

def proc_id(contig_id, contig, start_sites, count, results):
  results['contig_id'] = contig_id

def proc_counts(contig_id, contig, start_sites, count, results):
  results['counts'] = count

def proc_length(contig_id, contig, start_sites, count, results):
  results['length'] = len(contig)
  
def proc_coverage(contig_id, contig, start_sites, count, results):
  results['contig_coverage']     = len(np.where(contig)[0]) # return number of non-zero elements
  results['start_site_coverage'] = len(np.where(start_sites)[0]) 
  results['contig_coverage_pct']     = count and results['contig_coverage'] / float(len(contig)) * 100.0 or 0.0
  results['start_site_coverage_pct'] = count and results['start_site_coverage'] / float(len(contig)) * 100.0 or 0.0

def proc_unique_start_sites(contig_id, contig, start_sites, count, results):
  results['unique_start_sites'] = len(np.where(start_sites == 1)[0])
  results['start_site_complexity'] = count and results['unique_start_sites'] / float(results['start_site_coverage']) * 100.0 or 0.0
  
def proc_cv(contig_id, contig, start_sites, count, results):
  results['contig_cv']     = count and contig.std() / contig.mean() or 0.0
  results['start_site_cv'] = count and start_sites.std() / start_sites.mean() or 0.0

def proc_max(contig_id, contig, start_sites, count, results):
  results['contig_max']     = np.max(contig)
  results['start_site_max'] = np.max(start_sites)

def proc_min(contig_id, contig, start_sites, count, results):
  results['contig_min']     = np.min(contig)
  results['start_site_min'] = np.min(start_sites)

std_stats = [proc_id, proc_counts, proc_length, proc_coverage, proc_unique_start_sites, proc_cv, proc_max, proc_min]
stat_headers = {
  'contig_id': 'Contig',
  'counts': 'Counts',
  'length': 'Length',
  'contig_coverage': 'Coverage',
  'start_site_coverage': 'Start Sites',
  'contig_coverage_pct': 'Coverage (%)',
  'start_site_coverage_pct': 'Start Sites (%)',
  'unique_start_sites': 'Unique Start Sites',
  'start_site_complexity': 'Start Site Complexity',
  'contig_cv': 'Contig CV',
  'start_site_cv': 'Start Site CV',
  'contig_max': 'Max Coverage Depth',
  'start_site_max': 'Max Start Site Depth',
  'contig_min': 'Min Coverage Depth',
  'start_site_min': 'Min Start Site Depth',
  }

def bin_contig(contig, nbins=100):
  bins = np.zeros(nbins, dtype=np.int32)
  bin_size = np.ceil(len(contig) / float(nbins))
  max_stop = len(contig) - 1
  
  for i in range(nbins):
    start = int(i * bin_size)
    stop  = start + bin_size
    if start > max_stop: continue
    if stop >= max_stop: stop = max_stop
    
    if start == stop: bins[i] = contig[start]
    else:             bins[i] = np.max(contig[start:stop])

  return bins


# ------------------------------
# Sam File Processor
# ------------------------------
  
class SamCoverage:
  """
  Coverage/start site arrays and read counts for each contig in a genome.
  """

  def __init__(self, genome=None):
    self.contigs     = {} # id: coverage array
    self.start_sites = {} # id: start-site array
    self.counts      = {} # id: read count
    self.order       = []   # maintain the contig order for output

    if genome is not None:
      self.parse_genome(genome)
      
    return

  def parse_genome(self, genome):
    """
    Create the contigs for a genome file with lines:

      contig1 size
      contig2 size
      ...
      contigN size

    Each contig is represented by a np array with one integer for
    each base position.  Note that for human scale genomes, this will
    use approximately 24 GB of RAM (12 GB each for the contigs and
    start site arrays).
    """

    for contig, length in ercc_seq_utils.file_values(genome, skip_header=False):
      self.contigs[contig]     = np.zeros(int(length), np.int32)
      self.start_sites[contig] = np.zeros(int(length), np.int32)      
      self.counts[contig]  = 0
      self.order.append(contig)
      
    return

  def parse_sam(self, sam):
    """
    Read the records from a sam file, updating the coverage arrays
    and counts for each mapped feature.
    """

    for read, rname in ercc_seq_utils.sam_stream(sam):
      pos    = read.pos - 1 # convert to 0-based positions
      length = len(read.seq)

      # Ignore contigs that are not in the genome file
      if not self.contigs.has_key(rname): continue

      self.contigs[rname][pos:pos+length] += 1
      self.start_sites[rname][pos] += 1
      self.counts[rname] += 1

    return

  # ------------------------------
  # Iterators
  # ------------------------------

  # These iterators all return tuples of (contig_id, iter_value) using the contig order from the genome file.
  
  def count_iter(self):
    for contig_id in self.order:
      yield contig_id, self.counts[contig_id]
    return
    
  def coverage_iter(self):
    for contig_id in self.order:
      yield contig_id, self.contigs[contig_id]
    return
  
  def start_site_iter(self):
    for contig_id in self.order:
      yield contig_id, self.start_sites[contig_id]
    return
  
  def bin_iter(self, nbins=100, start_sites = False):
    """
    Return a binned version of each contig.
    """

    if start_sites: contigs = self.start_sites
    else:           contigs = self.contigs
      
    for contig_id in self.order:
      yield contig_id, bin_contig(contigs[contig_id], nbins)

    return

  def proc_iter(self, procs = std_stats):
    """
    stats is a list of procedures of the form:
      p(contig_id, coverage, start_sites, counts, results)

    results is a dict that each procedure can add arbitrary named
    results to.

    Procedures are processed in order, so later procs can rely on
    results from earlier procs
    """
    
    for contig_id in self.order:
      contig = self.contigs[contig_id]
      start_sites = self.start_sites[contig_id]
      counts = self.counts[contig_id]

      results = {}
      for proc in procs:
        proc(contig_id, contig, start_sites, counts, results)
      yield contig_id, results

    return


  # ------------------------------
  # Save Methods
  # ------------------------------

  def save_binned_coverage(self, nbins = 100, out=sys.stdout, start_sites = False):
    """
    For each contig, generate a binned array of coverage at each
    position:
      contig_id    1,2,3,1,1,3,4,...,5
      contig_id    1,2,3,1,1,3,4,...,5
    """

    for contig_id, bins in self.bin_iter(nbins, start_sites = start_sites):
      line = '%s\t%s\n' % (contig_id, ','.join(['%d' % v for v in bins]))
      out.write(line)
        
    return

  def save_full_coverage(self, out=sys.stdout, coverage_iter = None):
    """
    Save the full coverage array for each contig.
    """
    if coverage_iter is None:
      coverage_iter = self.coverage_iter()
      
    for contig_id, contig in coverage_iter:
      line = '%s\t%s\n' % (contig_id, ','.join(['%d' % v for v in contig]))
      out.write(line)

    return

  def save_counts(self, out=sys.stdout):
    """
    Print the counts for each contig:
      contig_id    read_count
      contig_id    read_count      
    """

    for contig_id in self.order:
      out.write('%s\t%d\n' % (contig_id, self.counts[contig_id]))

    return

  def save_stats(self, out=sys.stdout, stats = std_stats, headers = stat_headers):
    """
    Print a table of stats for the contigs:
      contig_id  length  positions_covered  percent_covered  cv  min_coverage  max_coverage  avg_coverage
    """

    out.write('\t'.join(headers[stat] for stat in headers))
      
    for contig_id, results in coverage.proc_iter(stats):
      out.write('\t'.join(results[stat] for stat in headers))
    
    return

  
