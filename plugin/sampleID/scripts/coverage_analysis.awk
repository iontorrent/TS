# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

BEGIN {
  IGNORECASE=1;
  genome += 0;
  if( genome <= 0 )
  {
    print "ERROR: Invalid genome length provided" > "/dev/stderr"
    exit 1;
  }
  maxdepth = 0;
  numpile = 0;
}

{
  $3 += 0; # cludge 'fix' for mawk bug
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
  abc = sum_coverage[0]/genome
  cum = cum_coverage[1];
  sum = sum_coverage[1];
  ave = cum > 0 ? sum/cum : 0;
  std = cum > 1 ? sqrt((cum*sum_sq_coverage[1] - sum*sum)/(cum*(cum-1))) : 0;
  p2m = int(0.2*abc+0.5);
  #printf "Base reads on target:    %.0f\n",sum;
  printf "Bases in target regions: %.0f\n",genome;
  #printf "Bases covered (at least 1x): %.0f\n",numpile;
  printf "Average base coverage depth: %.1f\n",abc;
  printf "Uniformity of coverage:      %.1f%%\n",cum_coverage[p2m]*scl;
  #printf "Maximum base read depth: %.0f\n",maxdepth;
  #printf "Average base read depth: %.1f\n",ave;
  #printf "Std.Dev base read depth: %.1f\n",std;
  printf "Coverage at 1x:   %.1f%%\n",numpile*scl;
  #printf "Coverage at 10x:  %.1f%%\n",cum_coverage[10]*scl;
  printf "Coverage at 20x:  %.1f%%\n",cum_coverage[20]*scl;
  #printf "Coverage at 50x:  %.1f%%\n",cum_coverage[50]*scl;
  printf "Coverage at 100x: %.1f%%\n",cum_coverage[100]*scl;
}
