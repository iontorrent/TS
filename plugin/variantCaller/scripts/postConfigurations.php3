<?php

$configuration_values = $_POST['configuration_values'];
file_put_contents("/results/plugins/scratch/variantCallerConfigurations.txt", json_encode($configuration_values));
?>
