#!/usr/bin/perl -w
$index = -2;
$lod_i = 0;
open(IN,"$ARGV[0]");
while (<IN>){
   $index++;
   if($index==-1){next;}
   @a=split;
   $fam[$index] = $a[5];
   if($a[6] ne "N/A"){
   $lod[$lod_i] = $a[6];
    $lod_i = $lod_i+1;
   }
   $all_fam_cov[$index]=$a[9]+$a[10]+$a[11];
   $strict_rate[$index]=$a[7];
   $loss[$index]=$a[8];
   $rate[$index]=$a[12];
   $fam_size[$index]=$a[13];
}
close(IN);
$min_var = 3;
open(IN,"$ARGV[1]");
while(<IN>){
    @a = split ;
    $seq = $a[0];
    if ($seq =~ m/snp_min_var_coverage/){
           @b = split /,/,$a[1];
           $min_var = $b[0];
    }
}
close(IN);
$med = int($index/2);
$m = $index+1;
@sorted = sort {$a <=> $b} @fam;
print "Number of Amplicons: $m\n";
print "Median Functional Molecular Coverage per Amplicon: $sorted[$med]\n";
$u50=0;
$u80=0;
$size = @fam;
$v1 = 0.5*$sorted[$med] ;
$v2 = 2.0*$sorted[$med];
$v3 = 0.8*$sorted[$med];
$l1 = 0.05;
$l2 = 0.01;
$l3 = 0.005;
$l4 = 0.002;
$n1 = 0;
$n2 = 0;
$n3 = 0;
$n4 = 0;
foreach $ii (@fam){
     if($ii>=$v1 && $ii <=$v2){$u50=$u50+1;}
     if($ii>=$v3 ){$u80=$u80+1;}
}
foreach $ii (@lod){
	if($ii<$l1){$n1=$n1+1;}
	if($ii<$l2){$n2=$n2+1;}
	if($ii<$l3){$n3=$n3+1;}
	if($ii<$l4){$n4=$n4+1;}
}
print "Uniformity of Molecular Coverage for all Amplicons: ";
printf("%5.2f%%\n",$u50/$size*100);
print "Percentage of Amplicons larger than 0.8x Median Functional Molecular Coverage: ";
printf("%5.2f%%\n",$u80/$size*100);

@sorted = sort {$a <=> $b} @all_fam_cov;
print "Median Total Molecular Coverage per Amplicon: $sorted[$med]\n";

@sorted = sort {$a <=> $b} @strict_rate;
print "Percentage of Reads with Perfect Molecular Tags: ";
printf("%5.2f%%\n",$sorted[$med]*100);

@sorted = sort {$a <=> $b} @loss;
print "Median Functional Molecular Loss due to Strand Bias per Amplicon: ";
printf("%5.2f%%\n",$sorted[$med]*100);
print "Median Percentage of Functional Molecules out of Total Molecules per Amplicon: ";
printf("%5.2f%%\n",100-$sorted[$med]*100);

@sorted = sort {$a <=> $b} @fam_size;
print "Median Reads per Functional Molecule: $sorted[$med]\n";
@sorted = sort {$a <=> $b} @rate;
print "Median Percentage of Reads Contributed to Functional Molecules per Amplicon: ";
printf("%5.2f%%\n",$sorted[$med]);

@sorted = sort {$a <=> $b} @lod;
$s = @lod;
$med = int($s/2);
print "Median Limit of Detection (LOD) per Amplicon: ";
printf("%5.2f%%\n",$sorted[$med]*100);
print "Percentage of Amplicons Below 5% LOD: ";
printf("%5.2f%%\n",$n1/$size*100);
print "Percentage of Amplicons Below 1% LOD: ";
printf("%5.2f%%\n",$n2/$size*100);
print "Percentage of Amplicons Below 0.5% LOD: ";
printf("%5.2f%%\n",$n3/$size*100);
print "Percentage of Amplicons Around 0.1% LOD: ";
printf("%5.2f%%\n",$n4/$size*100);
