#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
try:
    import argparse
except ImportError:
    sys.stderr.write( "Error: Can't import the python argparse package.\n" )
    sys.stderr.write( '       Perhaps you should do a "sudo apt-get install python-argparse".\n' )
    sys.exit( -1 )
from math import modf, floor
import time
import sys
import os
import h5py
import matplotlib
import scipy.io
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import json
import pylab
import matplotlib.pyplot as pyplot
import numpy
import png

def check_args(args):
    """See if we got our minimal set of valid arguments."""
    ok = True
    message = ""
    if not args.output_root  :
        ok = False
        message += "Didn't get directory for output (--output-root) "
    return([ok,message])

def quantile(x, q,  qtype = 7, issorted = False):
  """
  Args:
  x - input data
  q - quantile
  qtype - algorithm
  issorted- True if x already sorted.
  
  Compute quantiles from input array x given q.For median,
  specify q=0.5.
  
  References:
  http://reference.wolfram.com/mathematica/ref/Quantile.html
  http://wiki.r-project.org/rwiki/doku.php?id=rdoc:stats:quantile
  
  Author:
  Ernesto P.Adorio Ph.D.
  UP Extension Program in Pampanga, Clark Field.
  """
  if not issorted:
    y = sorted(x)
  else:
    y = x
  
  if not (1 <= qtype <= 9):
    return None  # error!
  
  # Parameters for the Hyndman and Fan algorithm
  abcd = [(0,   0, 1, 0), # inverse empirical distrib.function., R type 1
          (0.5, 0, 1, 0), # similar to type 1, averaged, R type 2
          (0.5, 0, 0, 0), # nearest order statistic,(SAS) R type 3
          
          (0,   0, 0, 1), # California linear interpolation, R type 4
          (0.5, 0, 0, 1), # hydrologists method, R type 5
          (0,   1, 0, 1), # mean-based estimate(Weibull method), (SPSS,Minitab), type 6
          (1,  -1, 0, 1), # mode-based method,(S, S-Plus), R type 7
          (1.0/3, 1.0/3, 0, 1), # median-unbiased ,  R type 8
          (3/8.0, 0.25, 0, 1)]   # normal-unbiased, R type 9.
  
  
  a, b, c, d = abcd[qtype-1]
  n = len(x)
  g, j = modf( a + (n+b) * q -1)
  if j < 0:
    return y[0]
  elif j >= n:
    return y[n-1]   # oct. 8, 2010 y[n]???!! uncaught  off by 1 error!!!
  
  j = int(floor(j))
  if g ==  0:
    return y[j]
  else:
    return y[j] + (y[j+1]- y[j])* (c + d * g)

def DrawMetric(metric,height,width,label,args,min_row,min_col,max_row,max_col,row_step,col_step,bound=True):

        if bound:
           (metric,vmax,vmin)=BoundMetric(metric,args)
        else:
           vmax=max(numpy.reshape(metric,(metric.size,1)))[0]
           vmin=min(numpy.reshape(metric,(metric.size,1)))[0]
        ree = numpy.reshape(metric, (height, width));
        pylab.figure(figsize=(12,8));
        pylab.jet();
        pylab.imshow(ree, vmin=vmin, vmax=vmax, origin='lower')
        pylab.colorbar();
        ax = pylab.gca()
        xtics = ax.get_xticks();
        ytics = ax.get_yticks();
        ax.set_xticklabels(map(str,(numpy.array(xtics) * col_step)))
        ax.set_yticklabels(map(str,(numpy.array(ytics) * row_step)))
        ax.set_xlabel('%d-%d (%d) wells' % (min_col, max_col, (max_col - min_col)))
        ax.set_ylabel('%d-%d (%d) wells' % (min_row, max_row, (max_row - min_row)))
        ax.set_title("%s %s" % (args.prefix,label ));
        ax.autoscale_view();
        file_name = "%s/%s_%s_spatial.png" % (args.output_root, args.prefix, label);
        print "file_name" + file_name
        pylab.savefig(file_name, bbox_inches="tight");
        pylab.close()



def DrawHist(metric,label,args):

  pylab.figure(figsize=(12,8))
  pylab.hist(metric[numpy.isfinite(metric)==True],bins=100)
  ax = pylab.gca()
  ax.set_title("%s %s" % (args.prefix,label ));
  ax.autoscale_view();
  file_name = "%s/%s_%s_histogram.png" % (args.output_root, args.prefix, label);
  pylab.savefig(file_name, bbox_inches="tight");
  pylab.close()


def GenerateSpeed(arr):

  dshape=arr.shape
  origarr=numpy.empty(dshape)
  dist=numpy.empty(dshape)
  timex=numpy.empty(dshape)
  for j in range(0,dshape[0]):
  #filter out zero estimates .
    for k in range(1,dshape[1]):
      origarr[j][k]=arr[j][k]
      if (arr[j][k]==0):
         arr[j][k]=max(arr[j][k-1],arr[j][k])
    #T0 monotonically increase from inlet to outlet.
    for k in range(dshape[1]-2,0,-1):
      if (arr[j][k]>arr[j][k+1]):
         arr[j][k]=arr[j][k+1]

    # Now identify areas where there is a clear time difference >.02 and make
    # velocity = distance over time.
    #
    for k in range(1,dshape[1]-2):
        # search for pixles before and after where the time difference is >1
        # and balance that time
        lefttime=0; kleft=k-1; kright=k+1;curtime=arr[j][k];
        while (((curtime-arr[j][kleft])<.5) & (kleft>0) & (origarr[j][kleft]>0)):
            kleft=kleft-1

        while ((kright<dshape[1]-1)&((arr[j][kright]-curtime)<.5) &(origarr[j][kright]>0)):
            kright=kright+1

        timex[j][k]=(arr[j][kright]-arr[j][kleft])
        dist[j][k]=kright-kleft;



  return (dist)



def BoundMetric(metric,args):
    metric_sorted = sorted(metric[numpy.isfinite(metric) == True])
    vmin=quantile(metric_sorted, args.min_quantile, issorted=True);
    vmax=quantile(metric_sorted, args.max_quantile, issorted=True);
    for m_ix in range(len(metric)) :
        if numpy.isfinite(metric[m_ix]) and metric[m_ix] < vmin :
            metric[m_ix] = vmin;
        if numpy.isfinite(metric[m_ix]) and metric[m_ix] > vmax :
            metric[m_ix] = vmax;
    return (metric,vmax,vmin)

def main():
    """Everybody's favorite function"""
    """Flow of program is to:
    1) Read in all the files
    2) Determine dimensions of final matrix
    3) For each column type specified:
       - Determine valid color range and mapping of value to color
       - Fill in data for image
       - Plot image
    """
    parser = argparse.ArgumentParser(description="Read series of spatial hdf5 files and create plots of the summaries.")
#    parser.add_argument("-f", "--file", help="h5 files to read for spatial data.", action='append', default=[]);
    parser.add_argument("-t", "--table", help="name of spatial table inside h5.", default="/spatial_table");
    parser.add_argument("--header", help="name of spatial header inside h5.", default="/spatial_header");
    parser.add_argument("--min_quantile", help="lower quantile to trim at.", default=.02);
    parser.add_argument("--max_quantile", help="upper quantile to trim at.", default=.98);
    parser.add_argument("-o","--output_root", help="name of directory to output results to.", default=None);
    parser.add_argument("-p","--prefix", help="prefix for plot titles.", default="");
    args_both = parser.parse_known_args();
    args = args_both[0];
    files = args_both[1];
    args_ok = check_args(args);
    if not args_ok[0] :
        sys.stderr.write("Error: %s\n" % args_ok[1]);
        sys.exit(-1);

    results={}
    # Load matrices
    header_json = "";
    matrices = [];
    min_row = 10000000;
    max_row = 0;
    min_col = 10000000;
    max_col = 0;
    header = "";
    for file_ix in range(len(files)) :
        print "Reading %s\n" % files[file_ix]
        h5_file = h5py.File(files[file_ix],'r')
        mat = h5_file[args.table].value;
        header = h5_file[args.header].value
        min_row = min(min_row, min(mat[:,0]));
        max_row = max(max_row, max(mat[:,0]));
        min_col = min(min_col, min(mat[:,2]));
        max_col = max(max_col, max(mat[:,2]));
        matrices.append(mat);
        h5_file.close();

#        mat[:,0] = mat[:,0] / h["row_step"]
#                            mat[:,2] = mat[:,2] / h["col_step"]
    header = json.loads(header)
    row_step = header["row_step"];
    col_step = header["col_step"];
    width = int((max_col - min_col)/ header["col_step"]) + 1
    height = int((max_row - min_row) / header["row_step"]) + 1                            
    for matrix in matrices :
        matrix[:,0] = (matrix[:,0] - min_row)/row_step
        matrix[:,2] = (matrix[:,2] - min_col)/col_step

    for metric_ix in range(4,matrices[0].shape[1]) :
        start_time = time.clock();
        print "Plotting metric %s\n" % header["headers"][metric_ix]
        metric = numpy.empty(width * height);
        metric[:] = float('NaN');
        # Load all the data from matrices into our ndarray
        current_time = time.clock();
        print "After init %d\n" % (current_time - start_time);
        for matrix in matrices :
            for row_ix in range(matrix.shape[0]) :
                metric[int(matrix[row_ix,0]) * width + int(matrix[row_ix,2])] = matrix[row_ix,metric_ix]

        current_time = time.clock();
        print "After matrix load %d\n" % (current_time - start_time);                
        # Clip the quantiles if desired
        #metric_sorted = sorted(metric[numpy.isfinite(metric) == True])
        #vmin=quantile(metric_sorted, args.min_quantile, issorted=True);
        #vmax=quantile(metric_sorted, args.max_quantile, issorted=True);
        #for m_ix in range(len(metric)) :
        #   if numpy.isfinite(metric[m_ix]) and metric[m_ix] < vmin :
        #      metric[m_ix] = vmin;
        #   if numpy.isfinite(metric[m_ix]) and metric[m_ix] > vmax :
        #      metric[m_ix] = vmax;
        #current_time = time.clock();
        #print "After filter %d\n" % (current_time - start_time);                
        # Create image
        DrawMetric(metric,height,width, header["headers"][metric_ix],args,min_row,min_col,max_row,max_col,row_step,col_step,True)
        if header["headers"][metric_ix]=='t0':
          try:
            speed=GenerateSpeed(numpy.reshape(metric, (height, width)))
            DrawMetric(speed,height,width,'Speed' ,args,min_row,min_col,max_row,max_col,row_step,col_step,False)
            DrawHist(speed.reshape(height*width,1),'Speed',args)
            filename = "%s/T0.npy" % (args.output_root);
            outfile=open(filename,'wb')
            numpy.save(outfile,numpy.reshape(metric,(height,width)))
            filename2 = "%s/speed.mat" % (args.output_root);
            scipy.io.savemat(filename2,mdict={'speed':speed})
            pylab.figure(figsize=(12,8));
            pylab.plot(speed[:,speed.shape[1]/2])
            ax = pylab.gca()
            ax.set_title("%s %s" % (args.prefix, "Speed Profile at Chip Center"));
            file_name = "%s/%s_%s_.png" % (args.output_root, args.prefix, "CrossSectionSpeed");
            pylab.savefig(file_name, bbox_inches="tight");
            pylab.close()

          except:
            print sys.exc_info()[0]
        if header["headers"][metric_ix]=='sig_clust_conf':
          vv=matrix[:,metric_ix]
          results['frac_low_9']=float(numpy.count_nonzero((vv>.5)&(vv<.9)))/(numpy.count_nonzero((vv>0.5))+.1)
          results['frac_low_85']=float(numpy.count_nonzero((vv>.5)&(vv<.85)))/(numpy.count_nonzero((vv>0.5))+.1)
          results['frac_low_8']=float(numpy.count_nonzero((vv>.5)&(vv<.8)))/(numpy.count_nonzero((vv>0.5))+.1)
          results['mean_conf']=float(numpy.mean(vv[vv>.5]))
          file_name_json = "%s/results.json" % (args.output_root)
          with open(file_name_json,'w') as f:
              json.dump({'separator_spatial':results},f)

        current_time = time.clock();
        print "After plot %d\n" % (current_time - start_time);                

   # for each column create plot
   
     # Fill in values
     # Create image
     # Write to disk
    
main()
