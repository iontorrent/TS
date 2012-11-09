<?php
if ($xml = simplexml_load_string($_POST["xml"])) {
	$siteName = $xml->Site[0]->Name;
	$htmlName = $xml->Site[0]->HTMLName;
	$htmlData = $xml->Site[0]->HTMLData;
	$filePath = "files/".$siteName."_".$htmlName;
	$htmlHandle = fopen($filePath, "w");
	fwrite($htmlHandle, $htmlData);
	fclose($htmlHandle);
	echo "</br>Thanks.</br>";
} else {
	trigger_error('Error reading XML string',E_USER_ERROR);
}
?>

