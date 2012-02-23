# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
package Check;
use strict;
use warnings;

our $VERSION = '1.00';
use base 'Exporter';
our @EXPORT = qw();

sub DosEndings {
    my $err = 0;
    my $file = shift;
    my $type = `file '$file'`;
    if($type =~ /CRLF/ || $type =~ /[^a-zA-Z]CR[^a-zA-Z]/) {
        warn "$file: has DOS line endings\n";
        $err = 1;
    }
    return $err;
}

sub Copyright {
    my $err = 0;
    my $file = shift;
    # Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

    if($file =~ /\.h(\.in)?$/ || $file =~ /\.cpp(\.in)?$/) {
        my $firstLine = `head -n1 $file`;
        if($firstLine !~ /^\/\* Copyright \(C\) \d+ Ion Torrent Systems, Inc\. All Rights Reserved \*\/\s*$/) {
            warn "$file: missing or illformed copyright line\n";
            $err = 1;
        }
    }
    elsif($file =~ /((\.sh)|(\.pl)|(\.pm)|(\.py)|(CMakeLists.txt))(\.in)?$/) {
        my $firstLine = `head -n2 $file | grep Copyright`;
        if($firstLine !~ /^\# Copyright \(C\) \d+ Ion Torrent Systems, Inc\. All Rights Reserved\s*$/) {
            warn "$file: missing or illformed copyright line\n";
            $err = 1;
        }
    }

    return $err;
}

1;
