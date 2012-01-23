<?php
/* 
** Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
**
** Gets the customer support archive file
*/

$file_zip = glob("*.support.zip");
$file_name = $file_zip[0];

header("Content-type: application/zip");
header("Content-Disposition: attachment; filename=". $file_name);

readfile($file_name);
?>
