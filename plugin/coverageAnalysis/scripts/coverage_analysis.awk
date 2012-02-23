# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

BEGIN {
  IGNORECASE=1;
  genome += 0;
  if( genome <= 0 )
  {
    print "ERROR: Invalid genome length provided" > "/dev/stderr"
    exit 1;
  }
  if( showlevels == "" ) showlevels = 50;
  basereads += 0;
  plot0x += 0;
  no0x = (plot0x == 0);
  outx1 = (x1cover != "");
  maxdepth = 0;
  numpile = 0;
}

{
  if($3>maxdepth) {maxdepth=$3;}
  coverage[$3]++;
  numpile++;
}

END {
  # when bed file is used, samtools depth can give results for 0 coverage
  coverage[0] += genome - numpile;
  for( i = maxdepth; i >= 0; i-- )
  {
    coverage[i] += 0; # force numeric
    sum = i * coverage[i];
    cum_coverage[i] = coverage[i] + cum_coverage[i+1];
    sum_coverage[i] = sum + sum_coverage[i+1];
    sum_sq_coverage[i] = i*sum + sum_sq_coverage[i+1];
  }
  scl = 100/genome;
  spc = sum_coverage[0] == 0 ? 1 : 100 / sum_coverage[0];
  showcut = (showlevels > 0 && maxdepth > showlevels);
  if( showcut )
  {
    maxplot = showlevels + 1; # for extra bin
  }
  else
  {
    maxplot = maxdepth;
  }
  if( binsize == 0 )
  {
    if( maxplot >= 5000 ) binsize = 100;
    else if( maxplot >= 1000 ) binsize = 50;
    else if( maxplot >= 500 ) binsize = 25;
    else if( maxplot >= 200 ) binsize = 10;
    else if( maxplot >= 100 ) binsize = 5;
    else binsize = 1;
  }
  # no files produced if no data
  if( sum_coverage[binsize] > 0 )
  {
    print "read_depth\tcounts\tcum_counts\tpc_cum_counts\tnum_reads\tpc_cum_num_reads" > outfile ;
  }
  if( numpile == 0 )
  {
     outx1 = 0;
  }
  if( outx1 )
  {
    print "read_depth\tcounts\tcum_counts\tpc_cum_counts\tnorm_depth" > x1cover ;
  }  
  wrt = (showlevels > 0 || binsize > 1);
  bincov = 0;
  binsum = 0;
  abc = sum_coverage[0]/genome;
  for( i = 0; i <= maxplot; i++ )
  {
    cum = cum_coverage[i];
    sum = sum_coverage[i];
    if( outx1 )
    {
      print i"\t"coverage[i]"\t"cum"\t"cum*scl"\t"i/abc > x1cover ;
    }
    # skip x0 in binned plots unless requested
    if( i == 0 && no0x )
    {
      continue;
    }
    # for cutoff plots add extra last row with cumulative data
    if( showcut && i == maxplot )
    {
      print ">"i-1"x\t"cum"\t"cum"\t"cum*scl"\t"sum"\t"sum*spc > outfile ;
    }
    else if( wrt || coverage[i] )
    {
      bincov += coverage[i];
      binsum += i * coverage[i];
      if( i % binsize == 0 )
      {
        cum = cum_coverage[i-binsize+1];
        sum = sum_coverage[i-binsize+1];
        print i"x\t"bincov"\t"cum"\t"cum*scl"\t"binsum"\t"sum*spc > outfile ;
        bincov = 0;
        binsum = 0;
      }
    }
  }
  # generate output stats to STDOUT
  cum = cum_coverage[1];
  sum = sum_coverage[1];
  ave = cum > 0 ? sum/cum : 0;
  std = cum > 1 ? sqrt((cum*sum_sq_coverage[1] - sum*sum)/(cum*(cum-1))) : 0;
  p2m = int(0.2*abc+0.5);
  if( basereads > 0 )
  {
    printf "Total aligned base reads:     %.0f\n",basereads;
    printf "Total base reads on target:   %.0f\n",sum;
    printf "Percent base reads on target: %.2f%%\n",100*(sum/basereads);
  }
  else
  {
    printf "Total base reads on target:   %.0f\n",sum;
  }
  printf "Bases in targeted reference: %.0f\n",genome;
  printf "Bases covered (at least 1x): %.0f\n",numpile;
  printf "Average base coverage depth: %.2f\n",abc;
  printf "Uniformity of coverage:      %.2f%%\n",cum_coverage[p2m]*scl;
  printf "Maximum base read depth: %.0f\n",maxdepth;
  printf "Average base read depth: %.2f\n",ave;
  printf "Std.Dev base read depth: %.2f\n",std;
  printf "Target coverage at 1x:   %.3f%%\n",numpile*scl;
  printf "Target coverage at 10x:  %.3f%%\n",cum_coverage[10]*scl;
  printf "Target coverage at 20x:  %.3f%%\n",cum_coverage[20]*scl;
  printf "Target coverage at 50x:  %.3f%%\n",cum_coverage[50]*scl;
  printf "Target coverage at 100x: %.3f%%\n",cum_coverage[100]*scl;
}
