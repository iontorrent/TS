#!/usr/bin/perl -w
while(<>){
     chomp;
     @a=split /:/;
     $t = $a[0];
     if($t eq "Number of Amplicons") {$t="Number_of_amplicons";}
     if($t eq "Median Functional Molecular Coverage per Amplicon") {$t="Median_funcfam_coverage_per_amplicon";}
     if($t eq "Uniformity of Molecular Coverage for all Amplicons") {$t="Fam_uniformity_of_amplicon_coverage";}
     if($t eq "Percentage of Amplicons larger than 0.8x Median Functional Molecular Coverage") {$t="80% Uniformity";}
     if($t eq "Median Total Molecular Coverage per Amplicon") {$t="Median_allfam_coverage_per_amplicon";}
     if($t eq "Percentage of Reads with Perfect Molecular Tags") {$t="Strict Rate";}
     if($t eq "Median Functional Molecular Loss due to Strand Bias per Amplicon") {$t="Median_fam_loss_by_strand_bias";}
     if($t eq "Median Percentage of Functional Molecules out of Total Molecules per Amplicon") {$t="Funtional molecules %";}
     if($t eq "Median Reads per Functional Molecule") {$t="Median_average_fam_size";}
     if($t eq "Median Percentage of Reads Contributed to Functional Molecules per Amplicon") {$t="Median_conversion_rate";}
     if($t eq "Median Limit of Detection (LOD) per Amplicon") {$t="LOD";}
     if($t eq "Percentage of Amplicons Below 5% LOD") {$t="Amplicons_with_5per_lod";}
     if($t eq "Percentage of Amplicons Below 1% LOD") {$t="Amplicons_with_1per_lod";}
     if($t eq "Percentage of Amplicons Below 0.5% LOD") {$t="Amplicons_with_05per_lod";}
     if($t eq "Percentage of Amplicons Around 0.1% LOD") {$t="Amplicons_with_01per_lod";}
     print "$t:\t$a[1]\n";
}
