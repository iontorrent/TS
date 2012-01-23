<?php
/* 
** Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
**
** Creates the system statistics support archive file
*/

/* Gather the following files and zip them
 *
 * /opt/sge/iontorrent/spool/master/messages
 * /var/log/ion/crawl.log
 * /var/log/ion/iarchive.log
 * /var/log/ion/serve.log
 * /var/log/ion/tsconf.log*
 * /var/log/ion/tsconfig_install.log*
 * /usr/share/ion-tsconfig/mint-config/*
 * /var/log/kern.log*
 * /var/log/syslog*
 * /var/log/apache2/access.log*
 * /var/log/apache2/error.log*
 * /var/log/postgresql/postgresql-8.4-main.log
 * 
 * Permission Issues:
 * /home/ionadmin/.bash_history
 *
 * Winblows Support Requirements:
 * Filenames with an underscore need to be converted to IBM850 encoding.
 * No forward slash directory delimiters.  Thats why all the files live in
 * top level.  Any help with this would be much appreciated.
 */
 
//Turn on output buffering.  This is how we ensure we don't send errant bytes
//along with http.
ob_start();

// We need the Dell service tag for the archive filename
// If not, then use the hostname
$fname = "/etc/torrentserver/tsconf.conf";
$servicetag = "";
if (file_exists($fname)){
    $cmd = "grep '^serialnumber:' ".$fname." | cut -f2 -d:";
    $servicetag = trim(shell_exec($cmd));
}
if ($servicetag === ""){
    $servicetag = gethostname();
}

$file_name = "/tmp/".$servicetag."_systemStats.zip";

// Remove existing zip file, if it exists
if (is_readable($file_name)) {
    unlink ($file_name);
}

$zip = new ZipArchive();
if ($zip->open($file_name, ZIPARCHIVE::CREATE) !== TRUE) {
    exit("Cannot open <$file_name>\n");
}

// celery files 
foreach (glob("/var/log/ion/celery_*.log") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// ionCrawler files
foreach (glob("/var/log/ion/crawl.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// inoPlugin files 
foreach (glob("/var/log/ion/ionPlugin.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// jobserver files 
foreach (glob("/var/log/ion/jobserver.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// crawler files 
foreach (glob("/var/log/ion/iarchive.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// install/config files 
foreach (glob("/var/log/ion/tsconfig_install.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// install/config files 
foreach (glob("/var/log/ion/tsconf.log") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
// install/config files 
foreach (glob("/usr/share/ion-tsconfig/mint-config/*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
foreach (glob("/var/log/kern.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
foreach (glob("/var/log/syslog*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
foreach (glob("/var/log/apache2/access.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
foreach (glob("/var/log/apache2/error.log*") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}
foreach (glob("/var/log/postgresql/postgresql-8.4-main.log") as $srcfile){
    if (is_readable($srcfile)) {
        $zip->addFile($srcfile,basename($srcfile));
    }
}

# Configuration files
$srcfile="/etc/vsftpd.conf";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}
$srcfile="/etc/torrentserver/tsconf.conf";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}
$srcfile="/etc/network/interfaces";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}
$srcfile="/etc/apache2/httpd.conf";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}
# needs special permission to read
$srcfile="/etc/postgresql/8.4/main/pg_hba.conf";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}

# TODO: dump sql schema
shell_exec("pg_dump -s iondb --username=ion > /tmp/dbschema.sql");
$srcfile="/tmp/dbschema.sql";
if (is_readable($srcfile)) {
    $zip->addFile($srcfile,basename($srcfile));
}


$srcfile = "/tmp/stats_sys.txt";
if (is_readable($srcfile)) {
    $in_charset = iconv_get_encoding($srcfile);
    $out_charset = "IBM850";
    $zip->addFile(iconv($in_charset,$out_charset,$srcfile),basename($srcfile));
}

$zip->close();

header("Content-type: application/zip");
header("Content-Disposition: attachment; filename=". basename($file_name));
header("Content-Length: ".filesize($file_name));

ob_end_clean();
readfile($file_name);

unlink($file_name);
?>
