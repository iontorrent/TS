#!/usr/bin/perl -w
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use warnings;
use FileHandle;
use File::Basename;

use lib dirname($0);
use Check;

sub checkDefines {
    my $file = shift;
    my $err = 0;
    my $defineName = $file;
    $defineName =~ s/.*\///;
    $defineName =~ tr/a-z/A-Z/;
    $defineName =~ s/\./\_/g;
    $defineName =~ s/_IN$//;
    $defineName =~ s/\-/\_/;
    my $fh = new FileHandle("<$file");
    if($fh) {
        my $foundIfNotDef = 0;
        my $foundDefine = 0;
        my $foundEndif = 0;
        while(<$fh>) {
            $foundIfNotDef = 1 if /^\#ifndef $defineName\s*$/;
            $foundDefine = 1 if /^\#define $defineName\s*$/;
            $foundEndif = 1 if /^\#endif\s*\/\/\s*$defineName\s*$/;
        }
        $fh->close();
        if(!($foundIfNotDef && $foundDefine && $foundEndif)){
            warn "$file: did not find expected include protection (eg '#ifndef $defineName', '#endif // $defineName')\n";
            $err = 1;
        }
    } else {
        warn "$file: unable to open '$file' for reading\n";
        $err = 1;
    }
    return $err;
}

# Main
my @files = @ARGV;
my $err = 0;
foreach my $file (@files) {
    $err = Check::DosEndings($file) || $err;
    $err = Check::Copyright($file) || $err;

    if($file =~ /\.h(\.in)?$/) {
        $err = checkDefines($file) || $err;
    }
}
exit $err;

