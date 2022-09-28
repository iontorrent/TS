#!/usr/bin/perl

# GDM: Reduced version of phoenix_qc_plate.pl script used in PhoenixQC plugin
# Outputs just the BED files for full racks and barcode -> rack mapping (barcode_panel.tsv)
# Only e2e reads is used for analysis - being the typical last call to generate the files
# This script can also be run to detect if the target tca(s) was a Phoenix plate QC run.
# - if not then no output will be generated - test barcode_panel.tsv exists

die "$0 <tca_dir> [<tca_dir2> ...]\n" if @ARGV<1;

$showMapping = 0;
$allRackMaps = 1;

#for (@ARGV) { push @tcas,glob($_); } #this generates a meaningless "Attempt to free unreferenced scalar" on exit
for (@ARGV) { push @tcas,(split/\n/,`ls -d $_`); } 

#parse TCA outputs#
for $tca (@tcas)
  {
  if (!$bedparsed)
    {for $bed (split/\n/, `ls $tca/local_beds/*.bed`)
      {if ($bed=~/p(\d+)\.gc\.bed$/) {$panel=$1; $type='bed'}
      elsif ($bed=~/p(\d+)\.anti\.gc\.bed$/) {$panel=$1; $type='antibed'}
      parsebed($bed);
      }
    }

  for $amplicon (split/\n/,`cut -f2 $tca/*.bcmatrix.xls|tail -n+2`)
    {push @amplicons, $amplicon if defined $pool{$amplicon} && ! defined $amplicons{$amplicon};
    $amplicons{$amplicon}++;
    }

  for $bcfile (split/\n/, `ls $tca/*/*.amplicon.cov.xls`)
    {@_ = split /\/+/, $bcfile;
    $bc = $_[-2];

#allow multiple runs be combined#
    if(@_>=5 && $_[-5]=~/_(\d+)$/) {$bc="$1$bc";}

    open in, $bcfile;
    while (<in>)
      {next if /#contig_id/;
      @_ = split /\t/;
      ($amplicon,$f,$r) = @_[3,7,8];

#make sure not to count repeatedly from different tca for a barcode-amplicon combination
      next if defined $tca{$bc}{$amplicon} && $tca{$bc}{$amplicon} ne $tca;

      $hm{$bc}{$amplicon} += $f+$r;
      $tca{$bc}{$amplicon} = $tca;
      }
    close in;
    }
  }

@panels = sort keys %picklist;
@racks = sort keys %rackcount;
@bcs = sort keys %hm;

#when there is no coverageAnalysis#
@amplicons=sort keys %pool if @amplicons==0;

#count plexies and ensure panel0 has antibed populated
for $panel (@panels)
  {
  # GDM: Only create p0 panel bed files - plexies count removed
  next unless( $panel );
  next unless( $panel =~ m/^0\./ || $allRackMaps );
  ($rk = $panel) =~ s/^0\.//;
  open out, ">$rk.bed";
  print out $header;
  for $amplicon (@amplicons) 
    {if ($type{$panel}{$amplicon} eq 'bed')
      {
      $bed{$amplicon}=~s/QcPool=p[^;]+/QcPool=p$panel/ if $panel>=1;
      print out $bed{$amplicon},"\n";
      }
    else {$type{$panel}{$amplicon}='antibed';}
    }
  close out;
  } 

# GDM: modified so empty file does not get created if there are no barcode -> rack mappings
$nrackmaps = 0;
for $bc (@bcs)
  {
#try to match panel to barcodes instead of trusting run plan#
  %match=(); undef $maxmatch;
  for $panel (@panels)
    {$total=0; $totalhm=0;
    for $amplicon (keys %{$hm{$bc}})
      {next if $type{$panel}{$amplicon} ne 'bed';
      $totalhm += $hm{$bc}{$amplicon};
      $total++;
      }
    next if $total==0;
    $threshold = $totalhm/$total*0.2;
    for $amplicon (keys %{$hm{$bc}})
      {if ($type{$panel}{$amplicon} eq 'bed' && $hm{$bc}{$amplicon}>=$threshold || $type{$panel}{$amplicon} eq 'antibed' && $hm{$bc}{$amplicon}<$threshold) {$match{$panel}++}
      else {$match{$panel}--}
      }
    $maxmatch=$match{$panel} if $maxmatch<$match{$panel} || !defined $maxmatch;
    } 
  for $panel (@panels) { if($match{$panel}==$maxmatch) {$panel{$bc}=$panel}} #seems inefficient, but to avoid preferential assignment to the first panel.
  $panel = $panel{$bc};
# GDM: Only output for the barcodes maping to the p0 panel - output modified to just barcode<tab>rack<crt>
  next unless( $panel );
  next unless( $panel =~ s/^0\.// || $allRackMaps );
  open out,">barcode_panel.tsv" if( ++$nrackmaps == 1 );
  $bc =~ s/^\d*//;
  print out "$bc\t$panel\n";
  print STDERR "$bc -> $panel\n" if( $showMapping );
  }
close(out) if( $nrackmaps );

sub parsebed()
  {open in, $_[0];
  while (<in>)
    {$header=$_ if /^track/;

    next if !/;Tube=/; #avoid messing up panel setup by non-QC BED files

    @_ = split /\t/;
    ($chr,$start,$end,$amplicon) = @_[0..3];
    ($rack) = /Rack=([^\s\;=]+)[;\s]/i; $rack="Rack" if !defined $rack;
    ($gene) = /GENE_ID=([^\s\;=\_]+?)[;\s\_]/i;
    ($pool,$row,$col) = /Pool=(\d+);Tube=(\w)0*(\d+)/i;
#This only address the non-unique amplicon issue for BED file ...#
    if ( ! defined $bed{$amplicon} )
      {$bed{$amplicon} = join"\t",@_[0..3],'.',"GENE_ID=$gene;Pool=$pool;Tube=$row$col;Rack=$rack;QcPool=p0;QcPoolType=bed";
      }
    elsif ( $bed{$amplicon}!~/GENE_ID=[^\s\;]*?$gene[\s\;\,]/ || $bed{$amplicon}!~/Rack=[^\s\;]*?$rack[\s\;\,]/ )
      {($racktemp) = $bed{$amplicon}=~/Rack=([^\s\;]+?)[;\s]/i;
      ($genetemp) = $bed{$amplicon}=~/GENE_ID=([^\s\;]+?)[;\s]/i;
      ($pooltemp,$rowcoltemp) = $bed{$amplicon}=~/Pool=([^\s\;]+?);Tube=([^\s\;]+)[;\s]/i;
      $bed{$amplicon} = join"\t",@_[0..3],'.',"GENE_ID=$genetemp,$gene;Pool=$pooltemp,$pool;Tube=$rowcoltemp,$row$col;Rack=$racktemp,$rack;QcPool=p0;QcPoolType=bed";
      }
    $csv{$amplicon}{"Gene_ID=$gene;Rack=$rack;Pool=$pool;Row=$row;Col=$col"}++;

    $chr{$amplicon} = $chr;
    $start{$amplicon} = $start;
    $end{$amplicon} = $end;
    $rackcount{$rack}++;
    $rack{$amplicon} = $rack;
    $row{$amplicon} = $row;
    $col{$amplicon} = $col;
    $gene{$amplicon} = $gene;
    $pool{$amplicon} = $pool;

    while (/QcPool=p(\d+);QcPoolType=(\w+)/g) #make sure not to overwrite type call by filenames
      {($panel,$type) = ($1,$2);
      $panel="0$panel" if $panel>0 && $panel<10 && $panel!~/^0/;
      $panel="0.$rack" if $panel eq '0';
      $type{$panel}{$amplicon} = $type;
      $picklist{$panel}{$row}{$col}{$type}++;
      }

    }
  close in;
  }

