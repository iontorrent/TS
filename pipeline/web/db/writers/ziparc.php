<?php
/* 
** Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
**
** Creates the archive file of files with matching extension
** called with ziparc.php?stem=<extension>
** Typically called from Default_Report.php (format_whole.php)
*/

//---	Include PHP parser
include ("parsefiles.php"); 

//---   parse expMeta.dat
$metadata = parseMeta("./expMeta.dat");
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

//---	Get extension of files to archive
$filestem = $_GET["stem"];

//---	Create name of archive file
// <rawdataname>_<analysisname>.<filestem>.zip
$zipfilename = $rawdataname."_".$analysisname.".".$filestem.".zip";

// Remove existing archive file, if it exists
if (file_exists($zipfilename)) {
	unlink ($zipfilename);
}

//Initialize new archive file
$zip = new ZipArchive();
if ($zip->open($zipfilename, ZIPARCHIVE::CREATE) !== TRUE) {
    error_log("Cannot open <".$zipfilename.">\n",0);
	exit(1);
}

//Define the list of local (cwd) files to archive
$files_to_archive = glob("./*.". $filestem);

//---	Special exceptions	---//
// if its BAM include the index files also
if (strcmp ($filestem, 'bam') == 0){
	$files_to_archive = array_merge ($files_to_archive, glob("./*.bam.bai"));
}
//If its sff, look for and remove tf.sff
if (strcmp ($filestem, 'sff') == 0) {
	foreach ($files_to_archive as $i => $value) {
		$pos = strpos($value,'tf.sff');
		if ($pos > 0) {
			unset($files_to_archive[$i]);
		}
	}
}

if (count($files_to_archive) == 0) {
	$zip->addFromString('ERROR.txt', 'No '.$filestem.' files exist.');
} else {
	//Add files to the zip archive file
	foreach ($files_to_archive as $srcfile){
		if (is_readable($srcfile)) {
			//$zip->addFile($srcfile,basename($srcfile));
			$in_charset = iconv_get_encoding($srcfile);
			$out_charset = "IBM850";
			$zip->addFile(iconv($in_charset,$out_charset,$srcfile),basename($srcfile));
		}
	}
}

//Finalize zip archive file
$zip->close();

if (file_exists($zipfilename)) {
	header("Content-type: application/zip");
	header("Content-Disposition: attachment; filename=". $zipfilename);
	header("Content-Length: " . filesize($zipfilename));

	readfile($zipfilename);
	unlink ($zipfilename);
} else {
    print "<html>";
    print "<head><title>Archive Error</title></head>";
    print "<b>Error creating archive of $filestem files.</b>";
    print "<br>";
    print "Could not write file $zipfilename.";
    print "<br>";
    print "</html>";
}

?>

