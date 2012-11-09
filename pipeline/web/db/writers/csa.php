<?php
/* 
** Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
**
** Creates the dataset analysis support archive file
**
*/

//Turn on output buffering.  This is how we ensure we don't send errant bytes
//along with http.
ob_start();

if(file_exists("./expMeta.dat")){
    include ("./parsefiles.php");
    $metadata = parseMeta("./expMeta.dat");
}else{
    include ("../parsefiles.php");
    $metadata = parseMeta("../expMeta.dat");
}

if ($metadata) {
    $rawdataname = $metadata[0][1];
    $analysisname = $metadata[2][1];
}
else {
    $rawdataname = "unknownDataset";
    $analysisname = getcwd();
}

//---   parse processParameters.txt
$procparams = parse_to_keys("./processParameters.txt");
if ($procparams) {
    $rawdatadir = $procparams['dataDirectory'];
}
else {
    $rawdatadir = "";
}

	
//---	Generate a pdf of the report to include
//get the url for this page without the querystring
$page_url = "http://localhost". parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH) . "?no_header=True";

//replace Default_Report with csa.php
$default_report = str_replace("csa.php","Default_Report.php",$page_url);

//build the command to be ran, wkhtmltopdf has to be in the correct path
$pdf_string = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q --margin-top 15 --header-spacing 5 --header-left " '. $analysisname .' - [date]" --header-right "Page [page] of [toPage]" --header-font-size 9 --disable-internal-links --disable-external-links ' . $default_report . ' report.pdf';
//run the command
exec($pdf_string);

// <rawdataname>_<analysisname>.support.zip
$zipfilename = $rawdataname."_".$analysisname.".support.zip";

// Remove existing zip file, if it exists
unlink ($zipfilename);

//Initialize new zip archive file
$zip = new ZipArchive();
if ($zip->open($zipfilename, ZIPARCHIVE::CREATE) !== TRUE) {
    exit("Cannot open <$zipfilename>\n");
}

//Define the local (cwd) files to archive
$files_to_archive = array(
        'processParameters.txt',
        'uploadStatus',
        'version.txt',
        'ReportLog.html',
        'sysinfo.txt',
        'separator.trace.txt',
        'separator.traces.txt',
        'separator.bftraces.txt',
        'report.pdf',
        'drmaa_stdout.txt',
        'drmaa_stdout_block.txt',
        'drmaa_stderr_block.txt',
		'DefaultTFs.conf',
		'alignmentQC_out.txt');
        
$files_to_archive = array_merge ($files_to_archive, glob("avgNukeTrace_*.txt"));
$files_to_archive = array_merge($files_to_archive,glob("NucStep/*"));
$files_to_archive = array_merge($files_to_archive,glob("dcOffset/*"));
//For barcode analysis, there are additional alignmentQC_out.txt files to gather
$files_to_archive = array_merge($files_to_archive,glob("bc_files/alignmentQC_out_*.txt"));

//Add files to the zip archive file
foreach ($files_to_archive as $srcfile){
    if (is_readable($srcfile)) {
		$zip->addFile($srcfile);
    }
}

//TS versions later than 1.5 provide the pgm logs in a zip file
if (is_readable('pgm_logs.zip')) {
	# open local zip file
	$lzip = new ZipArchive;
	if ($lzip->open('pgm_logs.zip')) {
		for ($i = 0; $i < $lzip->numFiles; $i++) {
			$zfile = $lzip->getNameIndex($i);
			$contents = $lzip->getFromName($zfile);
			#and add it to the zip archive
			$zip->addFromString($zfile,$contents);
		}
		$lzip->close();
	}
}
//if pgm zip does not exist, look in raw data directory
else {
	//Define the rawdataset files to archive
	$files_to_archive = array(
		'explog_final.txt',
		'InitLog.txt',
		'RawInit.txt',
		'RawInit.jpg',
		'InitValsW3.txt',
		'InitValsW2.txt',
		'Controller',
		'debug');
	//Define path to rawdata and append to files in list
	//Add files to the zip archive file
	foreach ($files_to_archive as $srcfile){
		$srcfile = implode("/",array($rawdatadir,$srcfile));
		if (is_readable($srcfile)) {
			$zip->addFile($srcfile);
		}
	}
}

//Finalize zip archive file
$zip->close();

if (file_exists($zipfilename)) {
	header("Content-type: application/zip");
	header("Content-Disposition: attachment; filename=". $zipfilename);
	header("Content-Length: ".filesize($zipfilename));
	ob_end_clean();
	readfile($zipfilename);
} else {
    print "<html>";
    print "<head><title>Archive Error</title></head>";
    print "<br>";
    print "<b>Error creating customer support archive file.</b>";
    print "<br>";
    print "Could not write file $zipfilename.";
    print "<br>";
    print "</html>";
	ob_end_flush();
}
?>
