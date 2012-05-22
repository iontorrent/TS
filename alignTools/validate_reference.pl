#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use Getopt::Long;

our $opt = {
  "fastaFile"         => undef,
  "dos"               => undef,
  "mac"               => undef,
  "autoFix"           => undef,
  "perLine"           => undef,
  "upperCaseOnly"     => undef,
  "limit"             => 4294967295,
  "onlyN"             => undef,
  "noIUPAC"           => undef,
  "help"              => undef,
};

GetOptions(
  "f|fasta|fasta-file=s"    => \$opt->{"fastaFile"},
  "a|auto-fix"              => \$opt->{"autoFix"},
  "p|per-line=s"            => \$opt->{"perLine"},
  "l|limit=s"               => \$opt->{"limit"},
  "u|upper-case-only"       => \$opt->{"upperCaseOnly"},
  "d|dos"                   => \$opt->{"dos"},
  "m|mac"                   => \$opt->{"mac"},
  "o|only-n"                => \$opt->{"onlyN"},
  "n|no-iupac"              => \$opt->{"noIUPAC"},
  "h|help"                  => \$opt->{"help"},
);
if ($opt->{"help"}) {
  &usage();
  exit(0);
}

# args checking
my $badArgs = 0;
if (!defined($opt->{"fastaFile"})) {
  $badArgs = 1;
  print STDERR "FATAL ERROR: must specify input fasta file with -f or --fasta or --fasta-file option !\n";
}
if (defined $opt->{"perLine"} && ($opt->{"perLine"}=~/\D/||$opt->{"perLine"}>65535)) {
  $badArgs = 1;
  print STDERR "FATAL ERROR: --per-line must be a positive integer no larger than 65535 !\n";
}
if ($badArgs) {
  &usage();
  exit(1);
}

sub usage () {
  print STDERR << "EOF";

usage: $0
  Required args:
    -f,--fasta,--fasta-file my.fasta   : Single fasta file of genome sequence(s)
  Optional args:
    -a,--auto-fix                      : Attemp to fix errors, otherwise default to die at the first error
    -p,--per-line length               : Require specific length per line, default to first non-header line of each sequence
    -l,--limit maximum                 : maximal total reference length in base pairs, default to 4294967295
    -u,--upper-case-only               : Require converting lower case base symbols to upper case
    -d,--dos                           : Allow DOS style line end;
    -m,--mac                           : Allow MAC Classic line end;
    -n,--no-iupac                      : Only allow [acgtACGT] characters;
    -o,--only-n                        : Only allow [acgtnACGTN] characters;
    -h,--help                          : This help message
EOF
}

our %error;
sub error
 {my ($errmsg, $errtype) = @_;
 if ($opt->{"autoFix"}) {warn $errmsg if $error{$errtype}++==0;}
 else {die "FATAL ERROR: $errmsg" if $errtype!=13}
 }

sub myprint {if ($opt->{"autoFix"}) {print @_}}

my $fastaFile = $opt->{"fastaFile"};
my ($perline,$seq,$n,$line,$diff,$total,$lasttotal,$name,$emptyline);
my %count;

open (in, $fastaFile) || die "FATAL ERROR: Cannot open $fastaFile: $!\n";
while (<in>)
{if (! defined $opt->{"dos"} && s/\r\n/\n/) {error("DOS style line end found at input line ".($line+1)." !\n",1);}
s/\r*\n+//;
if (! defined $opt->{"mac"} && s/\r/\n/g) {error("MAC Classic line end found at input line ".($line+1)." !\n",1);}
if (!/\S/) 
 {$line++;
 $emptyline++;
 next;
 }
my $temp = $_;
for(split/[\r\n]+/, $temp)
 {$line++;
 error("Empty line found at input line ".($line-$emptyline)." !\n",3) if $emptyline;
 if (/^>(.*)/) 
  {myprint "$seq\n" if $seq;
  if($name && $total==$lasttotal) {error("Sequence '$name' is empty !",4);}
  $lasttotal = $total;
  ($name,$seq,$diff,$perline) = ($1,'','',$opt->{"perLine"});
  my $origname = $name;
  if ($name=~s/^[\s\*\=]+//g) {error("Sequence name '$origname' starts with a white space, or an asterisk or equal sign at line $line !\n",4);}
#  if ($name=~s/\:/_/g) {error("Sequence name '$origname' contains a colon at line $line, which should be replaced by an underscore to avoid breaking samtools mpileup!\n",14)}
  if ($name=~/\W/) {error("Sequence name '$name' contains a non-alphanumeric character at line $line !\n",13)}
  $emptyline = '';
  }
 else
  {
#  error("No fasta header found at line 1 !\n",0) if $line==1;
#
  die "FATAL ERROR: No fasta header found at line 1 !\n" if $line==1;
  $emptyline = '';
  error($diff,6) if $diff;
  s/\s//g && error("White space found at input line $line !\n",7);
  $total+=length($_);
  $total <= $opt->{"limit"} || die "FATAL ERROR: Exceeding $opt->{limit} base pairs limit at input line $line !\n";
  s/[^ACGTNRYSWKMBDHVacgtnryswkmbdhv]/N/g && error("Non-IUPAC characters found at input line $line !\n",8);
  $opt->{"noIUPAC"} && /[^acgtACGT]/ && die "FATAL ERROR: Characters other than [acgtACGT] found at input line $line, cannot proceed !\n";
  $opt->{"onlyN"} && s/[^acgtnACGTN]/N/g && error("Characters other than [acgtnACGTN] found at input line $line !\n",9);
  $opt->{"upperCaseOnly"} && tr/acgtnryswkmbdhv/ACGTNRYSWKMBDHV/ && error("Lower case characters found at input line $line !\n",10);
  $perline = $opt->{"perLine"} if defined $opt->{"perLine"};
  if ($perline) {$diff = "Uneven sequence line length - input line $line is not $perline base pairs long !\n" if length($_)!=$perline}
  else {$perline = length($_)}
  if ($perline > 65535) 
   {error("Exceeding 65535 bases per line at input line $line !\n",11);
   $perline = 70;
   }
  $seq .= $_;
  if ($seq && $name) # empty sequences are ignored
   {split /\s+/, $name;
   if (++$count{$_[0]}>1)
    {error("Sequence named '$_[0]' is duplicated !\n",12);
    my $count = $count{$_[0]};
    $name .= "_$count" if $name!~s/(\s)/_$count\1/;
    }
   myprint ">$name\n";
   $name = '';
   }
  $n = 0;
  while (length($seq)>=($n+1)*$perline)
   {myprint substr($seq,$n*$perline,$perline),"\n";
   $n++;
   }
  $seq = substr($seq,$n*$perline) if $n;
  }
 }
}
close in;
myprint "$seq\n" if $seq; 
$total >0 || die "FATAL ERROR: No valid bases found !\n";
if($name && $total==$lasttotal) {error("Sequence '$name' is empty !",4);}
