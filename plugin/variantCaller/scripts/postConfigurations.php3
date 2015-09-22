<?php

$configuration_values = $_POST['configuration_values'];
if (count(configuration_values) == 0) {
	unlink("/results/plugins/scratch/variantCallerConfigurations.txt");	
}
else {
	file_put_contents("/results/plugins/scratch/variantCallerConfigurations.txt", json_encode($configuration_values));
}
?>
