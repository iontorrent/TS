# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


def empty_ranges(contig):
  gaps = np.where(contig == 0)[0]

  if len(gaps) == 0: return []

  start = gaps[0]
  last  = start
  p = start
  
  ranges = []
  for p in gaps[1:]:
    if p == last + 1:
      last = p
    else:
      ranges.append((start, last))
      start = p
      last  = start

  if start != p:
    ranges.append((start, last))

  return ranges

def plot_coverage(contig, starts, display = False, path='coverage.png'):
  plt.clf()
  
  # Find the regions with no coverage
  ranges = empty_ranges(contig)

  # Get the labels and range for the x-axis
  xlim = len(contig) 
  x = np.arange(len(contig))
  
  # Set the figure size
  fig = plt.gcf()
  fig.set_size_inches(8, 2, forward=True)


  plt.subplot(121) 

  axStarts = plt.gca() 
  for loc, spine in axStarts.spines.iteritems():
    spine.set_linewidth(0.5)
    spine.set_antialiased(True)
  
  axStarts.set_xlim(0, xlim)
  axStarts.set_yscale('log')
  axStarts.grid(True,color='gray', aa=True)
  axStarts.axhline(1.0, color='gray', aa=True, linewidth=0.5)
  axStarts.bar(x, starts, linewidth=0, width=2, aa=True)
  plt.title('Start Points',fontsize='small')

  plt.subplot(122) 
  axContig = plt.gca() 
  # Plot the contig as a bar chart
  for loc, spine in axContig.spines.iteritems():
    spine.set_linewidth(0.5)
    spine.set_antialiased(True)
  
  axContig.set_xlim(0,xlim)
  axContig.grid(True, color='gray', aa=True)
  axContig.axvspan(len(contig), xlim, color='gray', alpha=.5)
  plt.title('Coverage',fontsize='small')

  # Add red bars to the x-axis where there is no coverage
  for start, stop in ranges:
    axContig.axvspan(start, stop, 0.0, 0.02, color='red',alpha='.5', zorder=10)
  axContig.bar(x, contig, linewidth=0, width=2, zorder=10, aa=True)
  if display:
    plt.show()
  else:
    plt.savefig(open(path, 'wb'), format='png')

  return

