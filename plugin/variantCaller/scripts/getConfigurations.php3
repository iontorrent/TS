<?php
if (file_get_contents("/results/plugins/scratch/variantCallerConfigurations.txt") == "null") {
	print "";
}
else {
	print file_get_contents("/results/plugins/scratch/variantCallerConfigurations.txt");
}
?>
