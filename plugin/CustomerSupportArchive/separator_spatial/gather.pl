#!/usr/bin/perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

$name = shift(@ARGV);
printHeader(STDOUT);
print "<body>\n";
printNav(STDOUT);
print '    <div class="container">';

# Spatial Plots
while ($png = shift(@ARGV)) {
    print "<img src='$png'><br\>";
}
print "</div>\n";
print "</body></html>\n";

open (OUT, ">separator_spatial_block.html") or die "Can't open separator_block_html\n";
printHeader(OUT);
print '    <div class="container">';
printf OUT " <img src='%s_snr_spatial.png' width=400>\n",  $name;
printf OUT " <img src='%s_bf_metric_spatial.png' width=400>\n",  $name;
printf OUT " <img src='%s_sig_clust_conf_spatial.png' width=400>\n",  $name;
print '    </div>';
print OUT "</body></html>\n";
sub printPlainThumbnail() {
    my($fh,$img,$caption,$color,$width) = @_;
    print $fh "<div class='thumbnail'>\n";
    printf $fh "  <img src='%s' width='%s'>\n",$$img, $$width;
    print $fh "  <div class='caption'>\n";
    if ($$color ne "normal") {
	printf $fh "<center><table><tr><td bgcolor='#%s'><h5 >%s</h5></td></tr></table></center>\n",$$color, $$caption;
    }
    else {
	printf $fh "<center><h5>%s</h5></center>\n", $$caption;
    }
    print $fh "  </div>\n";
    print $fh "</div>\n";
}


sub printThumbnailLightbox() {
    my($fh,$img,$id,$verbage,$caption) = @_;
    print $fh "<div class='thumbnail'>\n";
    printf $fh "  <a data-toggle='lightbox' href='#%s'><img src='%s'></a>\n",$$id,$$img;
    printf $fh "  <div class='lightbox' id='%s' style='display: none;'>\n",$$id;
    print $fh "<div class='lightbox-content'>\n";
    print $fh "<center>\n";
    printf $fh "<img src='%s'></img>\n", $$img;
    printf $fh "<p><h5>%s</h5></p>\n",$$verbage;
    print $fh "</center>\n";
    print $fh "</div>\n";
    print $fh "  </div>\n";
    print $fh "  <div class='caption'>\n";
    printf $fh "<center><h5>%s</h5></center>\n",$$caption;
    print $fh "  </div>\n";
    print $fh "</div>\n";
}

sub printHeader {
    my ($fh) = shift(@_);
    print $fh <<ENDHEADER;
    <!DOCTYPE html>
	<html lang="en">
	<head>
	<meta charset="utf-8">
ENDHEADER
printf $fh ("<title>Separator Spatial Plugin</title>");
    print $fh <<ENDHEADER2;
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta name="description" content="Diagnostics from trying to classify wells">
	<meta name="author" content="charles.sugnet@lifetech.com">

	<link href="assets/bootstrap/css/bootstrap.css" rel="stylesheet">
	<link href="assets/bootstrap/css/bootstrap-lightbox.css" rel="stylesheet">
	<style>
	body {
	    padding-top: 60px; /* 60px to make the container go all the way to the bottom of the topbar */
    }
    </style>
	<link href="assets/bootstrap/css/bootstrap-responsive.css" rel="stylesheet">

	<!--[if lt IE 9]>
	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
	<![endif]-->

	<link rel="shortcut icon" href="assets/bootstrap/ico/favicon.ico">
	<link rel="apple-touch-icon-precomposed" sizes="144x144" href="assets/bootstrap/ico/apple-touch-icon-144-precomposed.png">
	<link rel="apple-touch-icon-precomposed" sizes="114x114" href="assets/bootstrap/ico/apple-touch-icon-114-precomposed.png">
	<link rel="apple-touch-icon-precomposed" sizes="72x72" href="assets/bootstrap/ico/apple-touch-icon-72-precomposed.png">
	<link rel="apple-touch-icon-precomposed" href="assets/bootstrap/ico/apple-touch-icon-57-precomposed.png">
	</head>
ENDHEADER2
}

sub printJS {
    my $fh = shift(@_);
print $fh <<ENDJS;
    <script src="http://code.jquery.com/jquery-latest.js"></script>
    <script src="assets/bootstrap/js/bootstrap.js"></script>
    <script src="assets/bootstrap/js/bootstrap-lightbox.js"></script>
    <script type='text/javascript'>
ENDJS
print $fh '$(document).ready(function () {';
print $fh 'if ($("[rel=tooltip]").length) {';
print $fh '$("[rel=tooltip]").tooltip();';
print $fh '}';
print $fh 'if ($("[rel=popover]").length) {';
print $fh '$("[rel=popover]").popover();';
print $fh '}';
print $fh '})';
print $fh '</script>';
}

sub printNav {
    $fh = shift(@_);
    print $fh <<ENDNAV;
    <div class="navbar navbar-inverse navbar-fixed-top" color="#165E29">
      <div class="navbar-inner">
	<div class="container">
	  <a class="btn btn-navbar" data-toggle="collapse" data-target=".nav-collapse">
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
	  </a>
	  <a class="brand" href="#">Separator Spatial Plugin</a>
	  <div class="nav-collapse collapse">
ENDNAV
print $fh <<ENDNAV;
	  </div>
	</div>
      </div>
    </div>
ENDNAV
}
