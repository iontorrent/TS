#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use warnings;
use Getopt::Long;
use JSON;
use Text::ParseWords;
use LWP;
use File::Path;

if( $#ARGV != 3 ) {
  usage();
  exit(1);
}

my $opt = {
  "id"    =>  $ARGV[0],
  "directory"  =>  $ARGV[1],
  "bedFile"  =>  $ARGV[2],
  "metaFile"  =>  $ARGV[3]
};

# Global Variables
my $outPrefix = $opt->{"directory"};
my $mergeDelim = ",";
my $skipErrs = 1;
my $maxTotalOverlapWarnings = 50;
my $ref = "";
my $refFile = undef;
my $bedName = undef;
my $ua = LWP::UserAgent->new;
my $apiUrl = "http://localhost/rundb/api/v1/";

# Find reference using TS API
if( $opt->{"metaFile"} ) {
  open(MFILE,"<",$opt->{"metaFile"}) or die "FATAL ERROR: Cannot open $opt->{'metaFile'}: $!\n";
  my $line = <MFILE>;
  chomp $line;
  my $json = decode_json($line);
  $ref = $json->{"reference"};
  # Query API to get reference path
  my $url = $apiUrl."referencegenome/?format=json&short_name=".$ref;
  my $resp = $ua->get($url);
  if( !$resp->is_success ) {
    error("FATAL DEV ERROR: Couldn't access '$url' to obtain reference genome file path.",1,1);
  }
  $resp = decode_json($resp->content);
  $refFile = $resp->{"objects"}[0]->{"reference_path"}."/".$ref.".fasta.fai";
  if( ! -e $refFile ) {
    error("FATAL ERROR: '$refFile' does not exist.",1,1);
    $refFile = undef;
  }
}

# Process bed file name
foreach( split(/\//,$opt->{"bedFile"}) ) {
  $bedName = $_;
}

# Be sure file name does not have illegal characters
if( $bedName =~ m/([^a-zA-Z0-9._-])/g ) {
  error("Error: Illegal character '$1' found in file name. Only alphanumeric characters [a-z,A-Z,0-9], ".
    "periods [.], hyphens [-], and underscores [_] are allowed in file names. Please correct and try again.",1,1);
}

# Process bed file extension
if( $bedName !~ m/\.bed$/ ) {
  error("Error: File name does not expected BED file type suffix. Check file format and rename file to end in '.bed'.",1,1);
}

# Check if file already exists
my $url = $apiUrl."content/?format=json&publisher_name=BED&path__endswith=/".$bedName;
my $response = $ua->get($url);
if( !$response->is_success ) {
  error("FATAL ERROR: Couldn't access '$url'.",1,1);
}
my $resp = decode_json($response->content);
if( $resp->{"meta"}{"total_count"} > 0 ) {
  error("Error: The file $bedName already exists. Please rename your file.",1,1);
}

# Make directories if non-existent
mkpath("$opt->{'directory'}/$ref/unmerged/plain/");
mkpath("$opt->{'directory'}/$ref/unmerged/detail/");
mkpath("$opt->{'directory'}/$ref/merged/plain/");
mkpath("$opt->{'directory'}/$ref/merged/detail/");


open(BFILE,"<",$opt->{"bedFile"}) or die "FATAL ERROR: Cannot open " . $opt->{"bedFile"} . ": $!\n";
open(FFILE,"<",$refFile) or die "FATAL ERROR: Cannot open " . $refFile . ": $!\n";
open(PBED,">","$opt->{'directory'}/$ref/unmerged/plain/$bedName")
  or die "FATAL ERROR: Cannot write to '$opt->{'directory'}/$ref/unmerged/plain/$bedName': $!\n";
open(DBED,">","$opt->{'directory'}/$ref/unmerged/detail/$bedName")
  or die "FATAL ERROR: Cannot write to '$opt->{'directory'}/$ref/unmerged/detail/$bedName': $!\n";
open(MPBED,">","$opt->{'directory'}/$ref/merged/plain/$bedName")
  or die "FATAL ERROR: Cannot write to '$opt->{'directory'}/$ref/merged/plain/$bedName': $!\n";
open(MDBED,">","$opt->{'directory'}/$ref/merged/detail/$bedName")
  or die "FATAL ERROR: Cannot write to '$opt->{'directory'}/$ref/merged/detail/$bedName': $!\n";

# Read in fai file information
my (%faiConts,%refIndex);
my @refOrder; # Order listed in reference file; follow when sorting validated files
my $refIdx = 0;
while( <FFILE> ) {
  chomp;
  my @elements = split /\t/;
  $faiConts{$elements[0]} = $elements[1];
  $refIndex{$elements[0]} = ++$refIdx;
  push(@refOrder,$elements[0]);
}
close FFILE;


### Begin validation ###
my ($nonEmpty,$inputLine,$isDetail,$lineNum,$trackCols,$lineMapIdx);
my (@lineNumbers,@plainLines,@detailLines,@plainMLines);
my (%lineMap,%contigSeen);
my ($tsVersion,$warnNumFields,$warnDOS,$emptyColVal) = (0,0,0,0);
my ($warnRefOrder,$warnSrtOrder,$warnEndOrder) = (0,0,0);
my ($col1Warn,$colIDWarn,$colNameAuto1,$colNameAuto2) = (0,0,0,0);
my $totalOverlapWarnings = 0;

my %stats = ("totRgns" => 0, "bigRgn" => 0, "bigRgnStr" => "",
    "smallRgn" => undef, "smallRgnStr" => "", "sumRgns" => 0);
my %mStats = ("totRgns" => 0, "bigRgn" => 0, "bigRgnStr" => "",
    "smallRgn" => undef, "smallRgnStr" => "", "sumRgns" => 0);
my @prebfile = <BFILE>;
my @bfile = ();

# Check for DOS or MAC line endings
for( my $i = 0; $i <= $#prebfile; $i++ ) {
  my $lineNum = $i+1;
  my $line = $prebfile[$i];
  if( $line =~ s/\r\n// ) {
    #error("Warning: DOS line ending(s) found. Correcting.",1,0) unless $warnDOS;
    $warnDOS = 1;
    push(@bfile,$line);
  } elsif( $line =~ s/\r/\n/g ) {
    #error("Warning: MAC line ending(s) found. Correcting.",1,0);
    push(@bfile,split(/\n/,$line));
  }
  else {
    chomp $line;
    push(@bfile,$line);
  }
}

my $lastContig = "";
my ($lastRefIdx,$lastSrt,$lastSrt2,$lastEnd) = (0,0,0,0);
my $regionNum = 1; # Used for final column (region#) if input is not detailed

for( my $i = 0; $i <= $#bfile; $i++ ) {
  $lineNum = $i+1;
  $inputLine = $bfile[$i];
  # Empty line
  if( $inputLine !~ m/\S/ ) {
    error("Warning @ line $lineNum: Empty line. Correcting.",1,0);
    next;
  }
  chomp $inputLine;
  # Line type
  if( $inputLine =~ m/^\s*track\s/ ) { # Track line
    # Check that this is the first line
    # If not, send warning and don't process other lines
    if( $lineNum != 1 ) {
      # Send out warning message and end processing
      error("Warning @ line $lineNum: Track line only allowed on first line. Ignoring all lines after this.",1,0);
      last;
    }
    
    # White space at start of line
    $inputLine = chkTrack($inputLine,$lineNum);
    # If invalid, replace track line with invalid track message
    if( !defined $inputLine ) {
      $inputLine = "Invalid track was here";
    }
    # Check if "type=bedDetail" exists
    if( $inputLine =~ m/type=bedDetail/ ) {
      $isDetail = 1;
      print DBED $inputLine."\n";
      print MDBED $inputLine."\n";
    }
    else {
      # Add type=bedDetail to detailed
      print PBED $inputLine."\n";
      print DBED $inputLine." type=bedDetail\n";
      print MDBED $inputLine." type=bedDetail\n";
    }
    if( $inputLine =~ m/torrentSuiteVersion=/ ) {
      $inputLine =~ m/torrentSuiteVersion=([\S\.]*)/;
      my $tsv = $1;
      if( $tsv =~ m/^(\d+\.\d+)(\..*)?$/ ) {
        $tsVersion = $1;
        error("Reading TS version format '$tsVersion'.",1,0);
      } else {
        error("Error @ line $lineNum: Illegal version format '$tsv'. Defaulting to 3.6.",1,0);
        $tsVersion = 3.6;
      }
    }
    if( $tsVersion >= 3.6 && $isDetail != 1 ) {
      error("Error @ line $lineNum: type=bedDetail is required in track line with torrentSuiteVersion=$tsVersion.",1,1);
    }
  }
  else { # Region line
    # Check if a track line must be created for detailed files
    if( $lineNum == 1 ) {
      print DBED "track type=bedDetail\n";
      print MDBED "track type=bedDetail\n";
    }
    my @lineConts = split(/\t/,$inputLine);
    my $numCols = $#lineConts+1;
    
    # Stricter format check for torrentSuiteVersion
    if( $tsVersion >= 3.6 && $numCols < 6 ) {
      error("Error @ line $lineNum: Too few columns($numCols). At least 6 tab-separated fields are required with torrentSuiteVersion=$tsVersion.",1,1);
    }
    # Check column numbers based on detailed or not
    if( $isDetail && $numCols < 5 ) {
      error("Error @ line $lineNum: Too few columns($numCols). ".
        "Expected at least 5 (3 required + 2 additional for detailed format).",1,$skipErrs);
      next;
    } elsif( $numCols < 3 ) {
      error("Error @ line $lineNum: Too few columns ($numCols). ".
        "Expected at least 3 columns.",1,$skipErrs);
      next;
    } elsif( $numCols < 4 && $warnNumFields == 0 ) {
      error("Warning @ line $lineNum: Fewer than recommended number of standard BED columns ($numCols). ".
        "No Region IDs (or Gene Symbols) are specified. This may impede the utility of applications using this file.",1,0);
      $warnNumFields = 1;
    } elsif( !$isDetail && $warnNumFields == 0 ) {
      error("Warning @ line $lineNum: Standard BED format (w/o type=bedDetail) does not allow for the Gene Symbol of Auxillary field columns. ".
        "This may impede the utility of applications using this file.",1,0);
      $warnNumFields = 1;
    }
    # Check expected track number of columns
    if( !defined $trackCols ) {
      $trackCols = $numCols;
    } elsif( $numCols != $trackCols ) {
      error("Error @ line $lineNum: Inconsistent number of columns. ".
        "Expected $trackCols but found $numCols instead.",1,$skipErrs);
      next;
    }
    # check for empty fields
    for( my $col = 0; $col < $trackCols; ++$col ) {
      my $val = $lineConts[$col];
      $val =~ s/\s//g;
      if( $val eq "" ) {
        $lineConts[$col] = '.';
        if( ++$emptyColVal <= 4 ) {
          my $ncol = $col+1;
          my $exMsg = ($emptyColVal == 4) ? " This is the last warning for this issue." : "";
          error("Warning @ line $lineNum: Column $ncol is blank or empty. Correcting to '.'.$exMsg",1,0);
        }
      }
    }
    # Validate and fill in empty columns
    my ($col1,$col2,$col3,$col4,$col5,$col6,$col7,$col8);
    if( $isDetail ) {
      $col1 = chkCol1($lineConts[0],$lineNum,\$col1Warn);
      $col2 = chkCol2($col1,$lineConts[1],$lineNum);
      $col3 = chkCol3($col1,$lineConts[2],$lineNum);
      if( $trackCols == 5 ) {
        if( $colNameAuto1 == 0 ) {
          error("Warning @ line $lineNum: Missing Region Name field (4). Names will be automatically assigned as <col1>:<col2>-<col3>.",1,0);
          $colNameAuto1 = 1;
        }
        $col4 = ($col2 < $col3) ? $col1.":".($col2+1)."-".$col3 : $col1.":".$col2."-".$col3;
      } else {
        $col4 = chkColID($lineConts[3],$lineNum,4);
      }
      $col5 = ($trackCols <= 6) ? 0 : chkCol5($lineConts[4],$lineNum);
      $col6 = ($trackCols <= 7) ? "+" : chkCol6($lineConts[5],$lineNum);
      # Take GENE_ID from auxilary for $tsVersion >= 3.6 from last column
      if( $tsVersion >= 3.6 ) {
        #$col7 = chkColAux($lineConts[$trackCols-1],$lineNum,$trackCols);
        #$col8 = chkColID($lineConts[$trackCols-2],$lineNum,$trackCols-1);
        # (N-1)th column kept as old auxiliary output column 7
        $col7 = chkColAux($lineConts[$trackCols-1],$lineNum,$trackCols-1);
        $col8 = getNameValue($col7,"GENE_ID","");
        if( $col8 eq "" ) {
          $col8 = "GENE#".$regionNum;
          ++$regionNum;
          if( $colNameAuto2 < 4 ) {
            my $msg = (++$colNameAuto2 >= 4) ? " Further warnings for auto-naming will not be issued." : "";
            error("Warning @ line $lineNum: Missing GENE_ID value in auxillary field ($trackCols). ID automatically assigned as $col8.$msg",1,0);
          }
        } else {
          $col8 = chkColID($col8,$lineNum,$trackCols);
        }
      } else {
        $col7 = chkColAux($lineConts[$trackCols-2],$lineNum,$trackCols-1);
        $col8 = chkColID($lineConts[$trackCols-1],$lineNum,$trackCols);
      }
    } else {
      $col1 = chkCol1($lineConts[0],$lineNum,\$col1Warn);
      $col2 = chkCol2($col1,$lineConts[1],$lineNum);
      $col3 = chkCol3($col1,$lineConts[2],$lineNum);
      if( $trackCols == 3 ) {
        if( $colNameAuto1 == 0 ) {
          error("Warning @ line $lineNum: Missing Region Name field (4). Names will be automatically assigned as <col1>:<col2>-<col3>.",1,0);
          $colNameAuto1 = 1;
        }
        $col4 = ($col2 < $col3) ? $col1.":".($col2+1)."-".$col3 : $col1.":".$col2."-".$col3;
      } else {
        $col4 = chkColID($lineConts[3],$lineNum,4);
      }
      ###$col4 = ($trackCols == 3) ? $col1.":".($col2+1)."-".$col3 : chkColID($lineConts[3],$lineNum,4);
      $col5 = ($trackCols <= 4) ? 0 : chkCol5($lineConts[4],$lineNum);
      $col6 = ($trackCols <= 5) ? "+" : chkCol6($lineConts[5],$lineNum);
      $col7 = 0;
      $col8 = "region#".$regionNum;
      ++$regionNum;
      if( $colNameAuto2 == 0 ) {
        error("Warning @ line $lineNum: Missing Region ID fields automatically assigned as 'region#<N>'.",1,0);
        $colNameAuto2 = 1;
      }
    }
    
    # Check for start > stop
    if( $col2 > $col3 ) {
      error("Error @ line $lineNum: Start point is higher than end point: $col2 > $col3.",1,$skipErrs);
      next;
    }
    
    # Store data line for later merging/sorting
    my $newLine = join("\t",($col1,$col2,$col3));
    push(@plainMLines,$newLine);
    $newLine = join("\t",($newLine,$col4,$col5,$col6));
    push(@plainLines,$newLine);
    $newLine = join("\t",($newLine,$col7,$col8));
    push(@detailLines,$newLine);
    $lineMap{$newLine} = $lineMapIdx++;
    push(@lineNumbers,$lineNum);
    $nonEmpty = 1;
  }
  
}# End for loop through BED file

# If not Non-empty
if( !$nonEmpty ) {
  error("Error: No valid regions found in file.",1,1);
}


# Sort and print all data
sortPrint(\%lineMap,\@plainLines,\@detailLines,\@plainMLines,\@lineNumbers,$isDetail);


# Print out stats
print "\n******UNMERGED******\n";
print "Total number of regions: $stats{'totRgns'}\n";
print "Size of biggest region: $stats{'bigRgn'} bp => '$stats{'bigRgnStr'}'\n";
print "Size of smallest region: $stats{'smallRgn'} bp => '$stats{'smallRgnStr'}'\n";
if( $stats{"sumRgns"} > 1000000 ) {
  print "Sum of regions: " . sprintf("%.2f",$stats{"sumRgns"}/1000000) . " Mbp\n";
}
elsif( $stats{"sumRgns"} > 1000 ) {
  print "Sum of regions: " . sprintf("%.2f",$stats{"sumRgns"}/1000) . " Kbp\n";
}
else {
  print "Sum of regions: $stats{'sumRgns'} bp\n";
}
print "********************\n";

print "\n*******MERGED*******\n";
print "Total number of regions: $mStats{'totRgns'}\n";
print "Size of biggest region: $mStats{'bigRgn'} bp => '$mStats{'bigRgnStr'}'\n";
print "Size of smallest region: $mStats{'smallRgn'} bp => '$mStats{'smallRgnStr'}'\n";
if( $mStats{"sumRgns"} > 1000000 ) {
  print "Sum of regions: " . sprintf("%.2f",$mStats{"sumRgns"}/1000000) . " Mbp\n";
}
elsif( $mStats{"sumRgns"} > 1000 ) {
  print "Sum of regions: " . sprintf("%.2f",$mStats{"sumRgns"}/1000) . " Kbp\n";
}
else {
  print "Sum of regions: $mStats{'sumRgns'} bp\n";
}
print "********************\n";

# Close file handles
close BFILE;
close PBED;
close DBED;
close MPBED;
close MDBED;


sub usage {
  print STDERR << "EOF";

usage: $0 <ID> <DIRECTORY> <UPLOAD_BED> <META_FILE>
   Required args:
      <ID>              : Process ID number
      <DIRECTORY>       : Given directory from Publisher
      <UPLOAD_BED>      : BED file to validate
      <META_FILE>       : Metadata JSON file

EOF
}

## General error message output ##
sub error {
  my ($errMsg,$errNum,$exit) = @_;
  print STDERR "$errMsg\n";
  registerError($errMsg, $exit) if( $opt->{'id'} ne "test" );
  if( $exit ) {
    # Attempt to close and unlink bed files
    close BFILE;
    close PBED && unlink "$opt->{'directory'}/$ref/unmerged/plain/$bedName";
    close DBED && unlink "$opt->{'directory'}/$ref/unmerged/detail/$bedName";
    close MPBED && unlink "$opt->{'directory'}/$ref/merged/plain/$bedName";
    close MDBED && unlink "$opt->{'directory'}/$ref/merged/detail/$bedName";
    rmtree("$opt->{'directory'}/$ref/");
    exit(0);
  }
}

sub registerError {
  my ($errMsg,$error) = @_;
  
  # First post message to log
  my $req = HTTP::Request->new(POST => 'http://localhost/rundb/api/v1/log/');
  $req->content_type('application/json');
  $req->content('{"upload":"/rundb/api/v1/contentupload/'.$opt->{'id'}.'/", "text":"'.$errMsg.'"}');
  print $req->as_string."\n";

  my $response = $ua->request($req);
  print $response->as_string."\n";
  
  if( $error ) {
    # Now put status=Error to contentupload
    $req = HTTP::Request->new(PUT => "http://localhost/rundb/api/v1/contentupload/$opt->{'id'}/");
    $req->content_type('application/json');
    $req->content('{"status":"Error"}');
    print $req->as_string."\n";
  
    $response = $ua->request($req);
    print $response->as_string."\n";
  }
}

sub chkTrack {
  my ($line,$lineNum) = @_;
  
  # Leading white space
  if( $line =~ s/^\s+// ) {
    error("Warning @ line $lineNum: Track line has leading whitespace. Correcting.",1,0);
  }
  # Trailing white space
  if( $line =~ s/\s+$// ) {
  #  error("Warning @ line $lineNum: Track line has trailing whitespace. Correcting.",1,0);
  }
  # Tab separators (or in quotes!)
  if( $line =~ s/\t/ /g ) {
    error("Warning @ line $lineNum: Track line contains tab characters. Correcting to spaces.",1,0);
  }
  # Split contents
  my @lineConts = quotewords('\s+',1,$line);
  unless( @lineConts ) {
    error("Error @ line $lineNum: track line incorrectly formatted with quotes.",1,$skipErrs);
  }
  # Cycle through line objects
  foreach my $obj (@lineConts) {
    next if $obj eq "track";
    
    # Correct key=value format
    my ($key,$val) = split(/=/,$obj,2);
    if( !defined $val ) {
      error("Error @ line $lineNum: track element expected [key]=[value] format.",1,$skipErrs);
      return undef;
    }
    # No white space without quotes
    if( $val =~ m/\s/ && (substr($val,0,1) ne '"' || substr($val,-1,1) ne '"') ) {
      error("Error @ line $lineNum: track element cannot have white space without surrounding quotes.",1,$skipErrs);
      return undef;
    }
    # Quotes only at ends
    my $tmp = substr($val,1,length($val)-2);
    if( $tmp =~ m/"/ ) {
      error("Error @ line $lineNum: quotes only allowed at beginning or end of track element.",1,$skipErrs);
      return undef;
    }
  }
  return $line;
}

sub chkCol1 {
  my ($obj,$lineNum,$col1Warn) = @_;
  # White space
  if( $obj =~ s/\s+//g ) {
    error("Warning @ line $lineNum, column 1: Whitespace found. Correcting.",1,0);
  }
  # Non-letter/number
  # Allow dots, underscores, hyphens, colons, and plus signs to pass but warn
  if( $obj =~ m/[^a-zA-Z0-9_.-]/ ) {
    # Check if bad characters exist
    my $errmsg = "";
    # Check if warning characters exist
    if( $obj =~ m/$mergeDelim/ ) {
      $errmsg = "Warning @ line $lineNum, column 1: merge delimiter character '$mergeDelim' used in '$obj'.";
      error($errmsg,1,0);
    } elsif( !$col1Warn ) {
      $errmsg = "Warning @ line $lineNum, column 1: non-alphanumeric, underscore, dot or hyphen character(s) used in '$obj'.";
      error($errmsg,1,0);
      $col1Warn = 1;
    }
  }
  # Reference fai
  if( !exists $faiConts{$obj} ) {
    error("Error @ line $lineNum, column 1: No match found in reference.",1,$skipErrs);
    return undef;
  }  
  # check for re-ordering warnings
  if( $obj ne $lastContig ) {
    my $refIdx = $refIndex{$obj};
    if( $lastContig ne "" && $refIdx < $lastRefIdx && !$warnRefOrder ) {
      ++$warnRefOrder;
      error("Warning @ line $lineNum, column 1: Contigs are not in same order as seen in reference. Correcting, with no further warnings.",1,0);
    }
    if( defined($contigSeen{$obj}) ) {
      error("Warning @ line $lineNum, column 1: Contigs are not in contiguous order: $obj first read @ line $contigSeen{$obj}. Correcting.",1,0);
    } else {
      $contigSeen{$obj} = $lineNum;
    }
    $lastContig = $obj;
    $lastRefIdx = $refIdx;
    $lastSrt = $lastEnd = 0;
  }
  return $obj;
}

sub chkCol2 {
  my ($chrom,$obj,$lineNum) = @_;
  # White space
  if( $obj =~ s/\s+//g ) {
    error("Warning @ line $lineNum, column 2: Whitespace found. Correcting.",1,0);
  }
  # Negative
  if( $obj =~ m/^-/ ) {
    error("Error @ line $lineNum, column 2: Negative integer found.",1,$skipErrs);
    return undef;
  }
  # Non-integer
  if( $obj =~ m/[^0-9]/ ) {
    error("Error @ line $lineNum, column 2: Non-integer found.",1,$skipErrs);
    return undef;
  }
  # Start too high
  if( $obj >= $faiConts{$chrom} ) {
    error("Error @ line $lineNum, column 2: Start point too high.",1,$skipErrs);
    return undef;
  }
  # Out of order
  if( $obj < $lastSrt && !$warnSrtOrder ) {
    error("Warning @ line $lineNum, column 2: Start positions not in increasing order. Correcting, with no further warnings.",1,0);
    ++$warnSrtOrder;
  }
  $lastSrt2 = $lastSrt;
  $lastSrt = $obj;
  return $obj;
}

sub chkCol3 {
  my ($chrom,$obj,$lineNum) = @_;
  # White space
  if( $obj =~ s/\s+//g ) {
    error("Warning @ line $lineNum, column 3: Whitespace found. Correcting.",1,0);
  }
  # Negative
  if( $obj =~ m/^-/ ) {
    error("Error @ line $lineNum, column 3: Negative integer found.",1,$skipErrs);
    return undef;
  }
  # Non-integer
  if( $obj =~ m/[^0-9]/ ) {
    error("Error @ line $lineNum, column 3: Non-integer found.",1,$skipErrs);
    return undef;
  }
  # End too high
  if( $obj > $faiConts{$chrom} ) {
    error("Error @ line $lineNum, column 3: End point too high.",1,$skipErrs);
    return undef;
  }
  # Out of order
  if( $obj < $lastEnd && $lastSrt == $lastSrt2 && !$warnEndOrder ) {
    error("Warning @ line $lineNum, column 3: End positions not in increasing order (at same start). Correcting, with no further warnings.",1,0);
    ++$warnEndOrder;
  }
  $lastEnd = $obj;
  return $obj;
}
sub chkCol5 {
  my ($obj,$lineNum) = @_;
  # White space
  if( $obj =~ s/\s+//g ) {
    error("Warning @ line $lineNum, column 5: Whitespace found. Correcting.",1,0);
  }
  # Number in range [0,1000]
  if( $obj !~ m/^\d+$/ || $obj < 0 || $obj > 1000 ) {
    error("Error @ line $lineNum, column 5: Expected a number in range [0,1000].",1,$skipErrs);
    return undef;
  }
  return $obj;
}

sub chkCol6 {
  my ($obj,$lineNum) = @_;
  # White space
  if( $obj =~ s/\s+//g ) {
    error("Warning @ line $lineNum, column 6: Whitespace found. Correcting.",1,0);
  }
  # + or - strand
  if( $obj ne "+" && $obj ne "-" ) {
    error("Error @ line $lineNum, column 6: Strand information expected.",1,$skipErrs);
    return undef;
  }
  return $obj;
}

sub chkColID {
  my ($obj,$lineNum,$colNum) = @_;
  # White space
  if( $obj =~ s/^\s+// ) {
    error("Warning @ line $lineNum, column $colNum: leading whitespace found. Correcting.",1,0);
  }
  if( $obj =~ s/\s+$// ) {
    error("Warning @ line $lineNum, column $colNum: trailing whitespace found. Correcting.",1,0);
  }
  if( $obj =~ m/[^a-zA-Z0-9_.-]/ ) {
    # Check if bad characters exist
    my $errmsg = "";
    # Check if warning characters exist
    if( $obj =~ m/$mergeDelim/ ) {
      $errmsg = "Warning @ line $lineNum, column $colNum: merge delimiter character '$mergeDelim' used in '$obj'.";
      error($errmsg,1,0);
    } elsif( !$colIDWarn ) {
      $errmsg = "Warning @ line $lineNum, column $colNum: non-alphanumeric, underscore, dot or hyphen character(s) used in '$obj'.";
      error($errmsg,1,0);
      $colIDWarn = 1;
    }
  }
  return $obj;
}

sub chkColAux {
  my ($obj,$lineNum,$colNum) = @_;
  # Trim only
  if( $obj =~ s/^\s+// ) {
    error("Warning @ line $lineNum, column $colNum: leading whitespace found. Correcting.",1,0);
  }
  if( $obj =~ s/\s+$// ) {
    error("Warning @ line $lineNum, column $colNum: trailing whitespace found. Correcting.",1,0);
  }
  if( $obj =~ m/$mergeDelim/ ) {
    error("Warning @ line $lineNum, column $colNum: merge delimiter character '$mergeDelim' used in '$obj'.",1,0);
  }
  return $obj;
}

sub getNameValue {
  # Return the value of $name in $query assuming format ...name=value;... or ... x.y = "a;b" ;...
  # Both comma and semicolon are allowed as equivalant separators (to allow concatonation of fields).
  # Or return $defval if $name not found.
  my ($query,$name,$defval) = @_;
  $name = quotemeta($name);
  if( $query =~ m/^(.*;|.*,)?\s*$name\s*=\s*(".*?"|'.*?'|.*?)\s*((;|,).*)?$/ )
  {
    # remove outer paired quotes
    $defval = $2;
    if( $defval =~ m/^"(.*)"$/ ) {
      $defval = $1;
    } elsif( $defval =~ m/^'(.*)'$/ ) {
      $defval = $1;
    }
  }
  return $defval;
}

sub sortPrint {
  my ($lineMap,$plainLines,$detailLines,$plainMLines,$lineNumbers,$isDetail) = @_;
  # Sort
  my (@tmpOrder,@finalOrder);
  foreach my $chrom(@refOrder) {
    $chrom = quotemeta($chrom);
    @tmpOrder = grep { $_ =~ m/^$chrom\t/ } @$detailLines;
    push(@finalOrder, (
      map { $lineMap->{$_->[0]} }
      sort { $a->[1] <=> $b->[1] || $a->[2] <=> $b->[2] }
      map { [$_,(split(/\t/,$_))[1,2]] } @tmpOrder )
    );
  }
  
  # Clear temp array
  @tmpOrder = ();

  # Check for any identical regions
  my $numOvlps = ckOverlapRegions(\@finalOrder,$detailLines,$lineNumbers);
  if( $numOvlps > 0 ) {
    error("Warning: Detected $numOvlps overlaps between regions.",1,0);
  }

  # Print out final order
  # Merge overlapping regions
  my (@prevInfo,$prevPLine,$prevDLine,@currInfo);
  foreach my $idx (@finalOrder) {
    print PBED $plainLines->[$idx];
    print DBED $detailLines->[$idx];
    if( $idx != $finalOrder[$#finalOrder] ) {
      print PBED "\n";
      print DBED "\n";
    }
    print "$idx:  $detailLines->[$idx]\n" if( $opt->{'id'} ne "test" );

    # Unmerged stats check
    @currInfo = split(/\t/,$detailLines->[$idx]);
    my $currLen = $currInfo[2]-$currInfo[1]-1;
    ckStats(0,$detailLines->[$idx],$currLen);
    
    # Check for merge
    if( !defined $prevPLine ) { # First line
      $prevPLine = $plainMLines->[$idx];
      $prevDLine = $detailLines->[$idx];
      @prevInfo = @currInfo;
    }
    # Do not merge if diff. chr. OR same chr. and previous_end <= current_start
    elsif( $prevInfo[0] ne $currInfo[0] || $prevInfo[2] <= $currInfo[1] ) {
      print MPBED $prevPLine."\n";
      print MDBED $prevDLine."\n";
      my $len = $prevInfo[2]-$prevInfo[1]-1;
      ckStats(1,$prevDLine,$len);
      $prevPLine = $plainMLines->[$idx];
      $prevDLine = $detailLines->[$idx];
      @prevInfo = @currInfo;
    }
    # Merge new lines (do not print yet)
    else {
      if( $prevInfo[2] < $currInfo[2] ) {
        $prevInfo[2] = $currInfo[2];
      }
      # Check if making comma-delimited field is necessary
      if( $prevInfo[3] ne $currInfo[3] || $prevInfo[3] !~ m/$currInfo[3]/ ) {
        $prevInfo[3] .= $mergeDelim . $currInfo[3];
      }
      if( $prevInfo[4] ne $currInfo[4] || $prevInfo[4] !~ m/$currInfo[4]/ ) {
        $prevInfo[4] .= $mergeDelim . $currInfo[4];
      }
      if( $prevInfo[5] ne $currInfo[5] ) {
        $prevInfo[5] .= $mergeDelim . $currInfo[5];
      }
      if( $prevInfo[6] ne $currInfo[6] || $prevInfo[6] !~ m/$currInfo[6]/ ) {
        $prevInfo[6] .= $mergeDelim . $currInfo[6];
      }
      if( $prevInfo[7] ne $currInfo[7] || $prevInfo[7] !~ m/$currInfo[7]/ ) {
        $prevInfo[7] .= $mergeDelim . $currInfo[7];
      }
      $prevPLine = join("\t",@prevInfo[0..2]);
      $prevDLine = join("\t",@prevInfo);
    }
  }
  # Final line
  print MPBED $prevPLine;
  print MDBED $prevDLine;
  my $len = $prevInfo[2]-$prevInfo[1]-1;
  ckStats(1,$prevDLine,$len);
}

sub ckIdenticalRegions {
  my ($finalOrder,$bedLines) = @_;
  my $prevChrom = "";
  my $prevStart = -1;
  my $prevEnd = -1;
  my $simRegions = 1;
  my @currInfo = ();
  foreach my $idx(@{$finalOrder}) {
    @currInfo = split(/\t/,$bedLines->[$idx]);
    if( $currInfo[0] eq $prevChrom &&
      $currInfo[1] == $prevStart &&
      $currInfo[2] == $prevEnd ) {
      
      # Identical region found
      # Increment region counter
      ++$simRegions;
    } elsif( $simRegions != 1 ) {
      # Similar regions found before
      # Send warning to user
      error("Warning: $simRegions identical regions found for '$prevChrom $prevStart $prevEnd'.'",1,0);
      # Reset
      $simRegions = 1;
    }
    $prevChrom = $currInfo[0];
    $prevStart = $currInfo[1];
    $prevEnd = $currInfo[2];
  }
  # Check if we ended within an exact region
  if( $simRegions != 1 ) {
    # Similar regions found before
    # Send warning to user
    error("Warning: $simRegions identical regions found for '$prevChrom $prevStart $prevEnd'.'",1,0);
  }
}

# assumes regions are ordered by chrom, start, end
# also checks for duplicate reion names (BED column 4)
sub ckOverlapRegions {
  my ($finalOrder,$bedLines,$lineNumbers) = @_;
  my $maxWarns = 3;    # max. overlap warnings per test region
  my $ovlpThres = 0.5; # warn if one region is 50% ovelapped by another
  my $numLines = scalar(@$finalOrder);
  my $numOvlps = 0;
  for( my $lidx1 = 0; $lidx1 < $numLines; ++$lidx1 ) {
    my $idx1 = $finalOrder->[$lidx1];
    my $lineNum1 = $lineNumbers->[$idx1];
    my ($chr1,$srt1,$end1,$rid1) = split(/\t/,$bedLines->[$idx1]);
    my $len1 = $end1-$srt1;
    my $numWarns = 0;
    for( my $lidx2 = $lidx1+1; $lidx2 < $numLines; ++$lidx2 ) {
      my $idx2 = $finalOrder->[$lidx2];
      my $lineNum2 = $lineNumbers->[$idx2];
      my ($chr2,$srt2,$end2,$rid2) = split(/\t/,$bedLines->[$idx2]);
      if( $rid1 eq $rid2 ) {
        error("Warning @ line $lineNum1: Region has same Name ('$rid1') as region @ line $lineNum2.",1,0);
      }
      last if( $chr2 ne $chr1 );
      last if( $end1 <= $srt2 );  # takes into account 0-base srt, 1-base end
      ++$numOvlps;
      next if( $totalOverlapWarnings >= $maxTotalOverlapWarnings );
      next if( $numWarns >= $maxWarns );
      if( $srt1 == $srt2 && $end1 <= $end2 ) {
        if( $end1 == $end2 ) {
          error("Warning @ line $lineNum1: Region $chr1:$srt1-$end1 ('$rid1') is identical to region @ line $lineNum2 ('$rid2').",1,0);
        } else {
          error("Warning @ line $lineNum1: Region $chr1:$srt1-$end1 ('$rid1') is completely overlapped by region @ line: $lineNum2 $srt2-$end2 ('$rid2').",1,0);
        }
        ++$numWarns;
        ++$totalOverlapWarnings;
      } elsif( $end2 <= $end1 ) {
        error("Warning @ line $lineNum1: Region $chr1:$srt1-$end1 ('$rid1') completely overlaps region @ line $lineNum2: $srt2-$end2 ('$rid2').",1,0);
        ++$numWarns;
        ++$totalOverlapWarnings;
      } else {
        my $len2 = $end2-$srt2;
        my $ovlp = $end1-$srt2;  # must be > 0
        if( $ovlp/$len1 >= $ovlpThres ) {
          $ovlp = int(0.5+100*$ovlp/$len1).'%';
          error("Warning @ line $lineNum1: Region $chr1:$srt1-$end1 ('$rid1') is $ovlp overlapped by region @ line $lineNum2: $srt2-$end2 ('$rid2').",1,0);
          ++$numWarns;
          ++$totalOverlapWarnings;
        } elsif( $ovlp/$len2 >= $ovlpThres ) {
          $ovlp = int(0.5+100*$ovlp/$len2).'%';
          error("Warning @ line $lineNum1: Region $chr1:$srt1-$end1 ('$rid1') overlaps $ovlp of region @ line $lineNum2: $srt2-$end2 ('$rid2').",1,0);
          ++$numWarns;
          ++$totalOverlapWarnings;
        }
      }
      if( $numWarns >= $maxWarns ) {
        error("Warning: Ignoring subsequent overlap warnings for region @ line $lineNum1.",1,0);
      }
      if( $totalOverlapWarnings >= $maxTotalOverlapWarnings ) {
        error("Warning: Reached limit on region overlap warnings ($maxTotalOverlapWarnings). Omitting subsequent warnings.",1,0);
      }
    }
  }
  return $numOvlps;
}

sub ckStats {
  my ($merge,$str,$len) = @_;
  if( $merge ) {
    $mStats{"totRgns"}++;
    if( $mStats{"bigRgn"} < $len ) {
      $mStats{"bigRgn"} = $len;
      $mStats{"bigRgnStr"} = $str;
    }
    if( !defined $mStats{"smallRgn"} || $mStats{"smallRgn"} > $len ) {
      $len = ($len > 0) ? $len : 0; # Can't have negative lengths
      $mStats{"smallRgn"} = $len;
      $mStats{"smallRgnStr"} = $str;
    }
    $mStats{"sumRgns"} += $len;
  } else {
    $stats{"totRgns"}++;
    if( $stats{"bigRgn"} < $len ) {
      $stats{"bigRgn"} = $len;
      $stats{"bigRgnStr"} = $str;
    }
    if( !defined $stats{"smallRgn"} ||  $stats{"smallRgn"} > $len ) {
      $len = ($len > 0) ? $len : 0; # Can't have negative lengths
      $stats{"smallRgn"} = $len;
      $stats{"smallRgnStr"} = $str;
    }
    $stats{"sumRgns"} += $len;
  }
}

