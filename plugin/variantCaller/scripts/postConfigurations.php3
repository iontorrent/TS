<?php

$configuration_values = $_POST['configuration_values'];
$filename = "/results/plugins/scratch/variantCallerConfigurations.txt";
if (count($configuration_values) == 0) {
	unlink($filename);	
}
else {
	file_put_contents($filename, json_encode($configuration_values));
	try {
		chmod($filename, 0777);
	} catch(Exception $e) {}
}
?>
