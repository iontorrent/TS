#!/usr/bin/perl -w
$ind = -1;
$total = 0;
$infile = $ARGV[0]."/".$ARGV[1].".amplicon.cov.xls";
$outfile = $ARGV[0]."/".$ARGV[1].".mol.cov.xls";
open(IN,"$infile");
while(<IN>){
    @a=split;
    $ind++;
    if($ind==0){next;}
    $total = $total+1;
    if(!$q{$a[5]}){$q{$a[5]}=1;}
    else{
        $q{$a[5]}=$q{$a[5]}+1;
    }
}
close(IN);
open(OUT,">$outfile");
print OUT "mol_depth\tamp_cov\tamp_cum_cov\tnorm_mol_depth\tpc_amp_cum_cov\n";
$re=0;
foreach $ii (sort{$a <=> $b} keys(%q)){
   $s = $total-$re;
   $per = $q{$ii}/$total;
   $sper = $s/$total*100;
   print OUT "$ii\t$q{$ii}\t$s";
   printf OUT ("\t%.4f\t%.2f\n",$per,$sper);
   $re = $re+$q{$ii};
}
close(OUT);
