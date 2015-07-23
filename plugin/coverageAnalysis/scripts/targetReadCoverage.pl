#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create table file (tsv) of bed target regions read coverage for BAM file. (Output to STDOUT.)
The coverage information is added to fields in the BED but assumes certain fields are present (e.g. from gcAnnoBed.pl).
Individual reads will first be assigned to individual targets, using limits specified by optional arguments.";
my $USAGE = "Usage:\n\t$CMD [options] <BAM file> <Annotated BED file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Print extra Log information to STDERR.
  -a Assign reads assuming Amplicon/AmpliSeq targets. (See -D/-U options below.)
     Default: Assign reads by maximum overlap with target. (Max. overlap also used to assign -a ties.)
  -b Output file is extended BED format (with BED coordinates and bedDetail track line).
     Default: Text table file (tab-separated text fields) with 1-based coordinates assumed field names.
  -d Ignore (PCR) Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -n Normalize read counts by the size of the targets. Affects what is output to the 'scoring' field (#9),
     which will be given the field ID 'norm_reads' or 'total_reads'.
  -E <int> End-to-end read proximity limit. If a read covers up to this disance from both ends of a region
     it will be counted as an end-to-end read. (Set to 0 if given as negative.) Default: 2.
  -C <num> Percentage Coverage threshold for 'full' target coverage read to be counted. When more than 0,
     specifiying this value causes fwd_cov and rev_cov counts to replace fwd_e2e and rev_e2e. Default: 0.
  -D <int> Downstream limit for matching read start to target end (appropriate for +/- strand mapping).
     This assignment parameter is only employed if the -a option is provided. Default: 1. Currently unused.
  -U <int> Upstream limit for matching read start to target end (appropriate for +/- strand mapping).
     This assignment parameter is only employed if the -a option is provided. Default: 30.
  -N <int> (Algorithm) Minimum Number of merged targets to use per samtools command. Default: 50.
     Forces more regions to be grouped than pysically overlapped within the -O option distance. The combination
     of -N, -O and -S option values affect run-time performance, depending mainly on the distribution of targets.
  -O <int> (Algorithm) Minimum merged region separation (Overhang). Default 10000.
     The minimum base distance between one merged group and the next. Because reads can overlap the ends of
     multiple (merged) target regions, this value should be at least 1000 to prevent a read being counted twice.
  -S <int> (Algorithm) Maximum merged region Separation. Default 1000000. (Ineffective if -N option is < 2.)
     Limits the spacing between grouped merged target regions to reduce number of off-target reads sampled.";

my $logopt = 0;
my $bedout = 0;
my $ampreads = 0;
my $dsLimit = 1;
my $usLimit = 30;
my $e2eLimit = 2;
my $tcovLimit = 0;
my $nondupreads = 0;
my $uniquereads = 0;
my $normreads = 0;

my $minNumMerge = 50;
my $endOvlp = 10000;
my $maxMrgSep = 1000000;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-a') {$ampreads = 1;}
  elsif($opt eq '-b') {$bedout = 1;}
  elsif($opt eq '-d') {$nondupreads = 1;}
  elsif($opt eq '-u') {$uniquereads = 1;}
  elsif($opt eq '-n') {$normreads = 1;}
  elsif($opt eq '-C') {$tcovLimit = int(shift);}
  elsif($opt eq '-D') {$dsLimit = int(shift);}
  elsif($opt eq '-E') {$e2eLimit = int(shift);}
  elsif($opt eq '-U') {$usLimit = int(shift);}
  elsif($opt eq '-N') {$minNumMerge = int(shift);}
  elsif($opt eq '-O') {$endOvlp = int(shift);}
  elsif($opt eq '-S') {$maxMrgSep = int(shift);}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else
  {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    print STDERR "$OPTIONS\n";
    exit 1;
  }
}
if( $help )
{
  print STDERR "$DESCR\n";
  print STDERR "$USAGE\n";
  print STDERR "$OPTIONS\n";
  exit 1;
}
elsif( scalar @ARGV != 2 )
{
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

my $bamfile = shift(@ARGV);
my $bedfile = shift(@ARGV);

$tcovLimit = 0 if( $tcovLimit < 0 );
$e2eLimit = 0 if( $e2eLimit < 0 );
$usLimit *= -1;  # more convenient for testing

my $usePcCov = ($tcovLimit > 0);
$tcovLimit *= 0.01;  # % to fraction
$maxMrgSep -= $endOvlp;

#--------- End command arg parsing ---------

my $samopt= ($nondupreads ? "-F 0x704" : "-F 0x304").($uniquereads ? " -q 1" : "");

# make quick test to see if this is a valid bam file
open( BAMTEST, "samtools view \"$bamfile\" 2> /dev/null |" ) || die "Cannot open BAM file '$bamfile'.";
unless(<BAMTEST>) {
  print STDERR "Error: BAM file missing, empty or badly formatted.\n";
  exit 1;
}
close(BAMTEST);

# read target regions from BED file
open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
my $headerLine = "track type=bedDetail";
if( !$bedout ) {
  $headerLine = sprintf "%s\t%s\t%s\t%s\t%s\t%s\toverlaps\t%s\t%s\tfwd_reads\trev_reads",
    "contig_id", "contig_srt", "contig_end", "region_id", "gene_id", "gc_count",
    ($usePcCov ? "fwd_cov\trev_cov" : "fwd_e2e\trev_e2e"), ($normreads ? "norm_reads" : "total_reads");
}

my ($bchr,$bend,$bname,$bgene,$bgc,$mrgSrt,$mrgEnd,$padend);
my @targSrts = ();
my @targEnds = ();
my @targNames = ();
my @targGenes = ();
my @targGCs = ();
my $bsrt = 0;
my $lastChr = "";

$| = 1;  # autoflush

# read the bedfile for header and the first region to avoid extra tests in main loop
while(<BEDFILE>) {
  chomp;
  ($bchr,$bsrt,$bend,$bname,$bgene,$bgc) = split('\t',$_);
  next if( $bchr !~ /\S/ );
  if( $bchr =~ /^track / ) {
    if( $bedout ) {
      print "$_\n";
      $headerLine = "";
    } elsif( m/ionVersion=([\S\.]*)/ ) {
      my $tsv = $1;
      if( $tsv =~ m/^(\d+\.\d+)(\..*)?$/ ) {
        $tsv = $1;
        $headerLine =~ s/gene_id/attributes/ if( $tsv >= 4 );
      }
    }

    next;
  }
  ++$bsrt;
  $padend = $bend+$endOvlp;
  $lastChr = $bchr;
  @targSrts = ($bsrt);
  @targEnds = ($bend+0);
  @targNames = ($bname);
  @targGenes = ($bgene);
  @targGCs = ($bgc);
  $mrgSrt = $bsrt;
  $mrgEnd = $padend;
  last;
}
unless( $bsrt ) {
  print STDERR "WARNING: targets file $bedfile had no effective targets.\n";
  exit 0;
}
# output header if not done so already
if( $headerLine ne "" ) {
  print "$headerLine\n";
  $headerLine = "";
}

# Check for BAI file existence as this will cause the individual samtools command calls to fail
my $nobai = 0;
unless( -e "${bamfile}.bai" ) {
  $nobai = 1;
  print STDERR "WARNING: Required BAM index (BAI) file not found. Coverage for all targets assigned as 0 reads.\n";
}

# Outer loop is to allow the last line read to be used as the next merged region and to group mutiple merged regions
my $badviews = 0;
$bsrt = 0;  # indicates last read if there was only one!
while(1) {
  # For performance, the BAM file will be read for individual targets
  # The BED file is assumed to well ordered, etc., and have overlapping targets
  # To assign reads to more specific targets, the merged target must be used and the unmerged targets collated
  while(<BEDFILE>) {
    chomp;
    ($bchr,$bsrt,$bend,$bname,$bgene,$bgc) = split('\t',$_);
    next if( $bchr !~ /\S/ );
    if( $bchr =~ /^track / ) {
      print STDERR "WARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n";
      last;
    }
    ++$bsrt;
    $padend = $bend+$endOvlp;
    last if( $bchr ne $lastChr || $bsrt > $mrgEnd );
    $mrgEnd = $padend if( $padend > $mrgEnd );
    push( @targSrts, $bsrt );
    push( @targEnds, $bend+0 );
    push( @targNames, $bname );
    push( @targGenes, $bgene );
    push( @targGCs, $bgc );
    $bsrt = 0;	# indicates last read already in list
  }
  # expand the number of targets if too few
  my $nTrgs = scalar(@targSrts);
  if( $bsrt && $nTrgs < $minNumMerge && $lastChr eq $bchr && $bsrt-$mrgEnd < $maxMrgSep ) {
    $mrgEnd = $padend;
    push( @targSrts, $bsrt );
    push( @targEnds, $bend+0 );
    push( @targNames, $bname );
    push( @targGenes, $bgene );
    push( @targGCs, $bgc );
    $bsrt = 0;
    next;
  }
  my (@targFwdReads,@targRevReads,@targFwdE2E,@targRevE2E,@targOvpReads);
  unless( $nobai ) {
    # process BAM reads covering these merged targets
    open( MAPPINGS, "samtools view $samopt \"$bamfile\" \"$lastChr:$mrgSrt-$mrgEnd\" 2> /dev/null |" )
      || die "Failed to pipe reads from $bamfile for regions in $bedfile\n";
    my $firstRegion = 0, $lastEnd = $targEnds[$nTrgs-1];
    while( <MAPPINGS> ) {
      next if(/^@/);
      if(/^.bam_parse_region/) {
        ++$badviews;
        next;
      }
      my ($rid,$flag,$chrid,$srt,$scr,$cig) = split('\t',$_);
      $srt += 0;
      last if( $srt > $lastEnd );  # skip remaining off-target reads in merge buffer
      my $end = $srt-1;
      my $rev = $flag & 16;
      while( $cig =~ s/^(\d+)(.)// ) {
        $end += $1 if( $2 eq "M" || $2 eq "D" || $2 eq "X" || $2 eq "=" );
      }
      my ($bestTn,$bestPrm) = (-1,0);
      my ($maxOvlp,$bestPrmDist) = (0,$usLimit);
      my ($maxEndDist,$bestTrgLen);
      for( my $tn = $firstRegion; $tn < $nTrgs; ++$tn ) {
        # safe to looking when read end is prior to start of target
        my $tSrt = $targSrts[$tn];
        last if( $end < $tSrt );
        # no match if read start is after target end, but ends can overlap therefore need to carry on searching
        my $tEnd = $targEnds[$tn];
        if( $srt > $tEnd ) {
          # adjust start of list for further reads if no earlier target end found
          $firstRegion = $tn+1 if( $maxOvlp < 0 );
          next;
        }
        my $trgLen = $tEnd - $tSrt;
        # record a hit for any read overlap
        ++$targOvpReads[$tn];
        $dSrt = $srt - $tSrt;
        $dEnd = $tEnd - $end;
        # test if this can be assigned using expected read starts
        if( $ampreads ) {
          # favor target with least distance BEFORE primer if within range of priming
          my $ddSrt = $rev ? $dEnd : $dSrt;
          if( $ddSrt < 0 && $ddSrt >= $bestPrmDist ) {
            # force this as best choice if mostly likely start so far, else use ovlp to split ties
            $maxOvlp = 0 if( $ddSrt > $bestPrmDist );
            $bestPrm = 1;
            $bestPrmDist = $ddSrt;
          } elsif( $bestPrm ) {
            # ignore this target if a suitable target start has been seen already
            next;
          }
        }
        # save region number for max overlap
        $dSrt = 0 if( $dSrt < 0 );
        $dEnd = 0 if( $dEnd < 0 );
        $tSrt = $tEnd - $tSrt - $dSrt - $dEnd; # actually 1 less than overlap
        # in case of a tie, keep the most 3' match for backwards-compatibility to old 3.6 version
        if( $tSrt >= $maxOvlp ) {
          $maxOvlp = $tSrt;
          $bestTn = $tn;
          $maxEndDist = $dSrt > $dEnd ? $dSrt : $dEnd;
          $bestTrgLen = $trgLen;
        }
      }
      unless( $bestTn < 0 ) {
        if( $rev ) {
          ++$targRevReads[$bestTn];
          if( $usePcCov ) {
            ++$targRevE2E[$bestTn] if( ($maxOvlp+1)/$bestTrgLen >= $tcovLimit );
          } else {
            ++$targRevE2E[$bestTn] if( $maxEndDist <= $e2eLimit );
          }
        } else {
          ++$targFwdReads[$bestTn];
          if( $usePcCov ) {
            ++$targFwdE2E[$bestTn] if( ($maxOvlp+1)/$bestTrgLen >= $tcovLimit );
          } else {
            ++$targFwdE2E[$bestTn] if( $maxEndDist <= $e2eLimit );
          }
        }
      }
    }
    close( MAPPINGS );
  }
  # output assigned coverage for this group of targets
  for( my $i = 0; $i < $nTrgs; ++$i ) {
    my $tLen = $targEnds[$i]-$targSrts[$i]+1;
    --$targSrts[$i] if( $bedout );
    print "$lastChr\t$targSrts[$i]\t$targEnds[$i]\t$targNames[$i]\t$targGenes[$i]\t$targGCs[$i]\t";
    printf "%d\t%d\t%d\t", $targOvpReads[$i], $targFwdE2E[$i], $targRevE2E[$i];
    my $nreads = $targFwdReads[$i]+$targRevReads[$i];
    if( $normreads ) {
      $nreads = $tLen > 0 ? $nreads / $tLen : 0;
      printf "%.3f\t", $nreads;
    } else {
      printf "%.0f\t", $nreads;
    }
    printf "%d\t%d\n", $targFwdReads[$i], $targRevReads[$i];
  }
  # make last region start of new merged set
  last unless( $bsrt );
  $lastChr = $bchr;
  @targSrts = ($bsrt);
  @targEnds = ($bend+0);
  @targNames = ($bname);
  @targGenes = ($bgene);
  @targGCs = ($bgc);
  $mrgSrt = $bsrt;
  $mrgEnd = $padend;
  $bsrt = 0;  # indicate no pending reads
}
close(BEDFILE);
print STDERR "WARNING: $badviews target regions were not located by contig name. Targets file unsuitable.\n" if( $badviews );

# ----------------END-------------------
