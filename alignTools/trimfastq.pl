#!/usr/bin/perl -w
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use warnings;

use Getopt::Long;

my $In = "";
my $Out = "";
my $Length = "";
my $help = 0;

my $result = GetOptions("i|input=s" => \$In,
			"o|out=s" => \$Out,
			"l|length=s" => \$Length
    );


&usage() if(
    $help || $In eq "" || $Out eq ""  || $Length eq "" || (!$result)
    );


sub usage () {
    print STDERR << "EOF";
  usage: [-i -o -l]
      -i                     : fastq input file
      -o                     : truncated fastq output file
      -l                     : length to which reads are trimmed
EOF
exit(1);
}

open(IN,"<$In");
open(OUT,">$Out");

my $bead;
my $read;
my $plus;
my $qual;

while($bead = <IN>){
    chomp $bead;
    $read = <IN>;
    chomp $read;
    $plus = <IN>;
    chomp $plus;
    $qual = <IN>;
    chomp $qual;


    print OUT $bead,"\n";
    print OUT substr($read,0,$Length),"\n";
    print OUT $plus,"\n";
    print OUT substr($qual,0,$Length),"\n";
}
close(IN);
close(OUT);



