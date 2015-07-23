Some notes for compiling annotations in /results/plugins/scratch/RNASeq_2step/hg19/annotations

1. Newer annotations/hg19/gene.gtf was downloaded from 
ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz

2. An example of constructing an alternative rRNA interval file:
samtools view -H $BAM > annotations/hg19/rRNA.interval
grep 'gene_type "rRNA"' annotations/hg19/gene.gtf|perl -ne '($name)=/gene_name "(.+?)"/;split/\t/; print "$_[0]\t$_[3]\t$_[4]\t$_[6]\t$name\n"'|uniq >> annotations/hg19/rRNA.interval

3. Generate refFlat using the same gtf annotation from 1 above
###
##- Convert gtf into refFlat
##- The gtfToGenePred software is available at http://hgdownload.cse.ucsc.edu/admin/exe/
##- http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/gtfToGenePred

gtfToGenePred -ignoreGroupsWithoutExons -genePredExt gencode.v19.annotation.gtf tmp

awk 'BEGIN{FS="\t"};{print $12"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' tmp > gencode.v19.annotation.refflat
3. Generate refFlat using the same gtf annotation from 1 above
###
##- Convert gtf into refFlat
##- The gtfToGenePred software is available at http://hgdownload.cse.ucsc.edu/admin/exe/
##- http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/gtfToGenePred

gtfToGenePred -ignoreGroupsWithoutExons -genePredExt gencode.v19.annotation.gtf tmp

awk 'BEGIN{FS="\t"};{print $12"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' tmp > gencode.v19.annotation.refflat
3. Generate refFlat using the same gtf annotation from 1 above
###
##- Convert gtf into refFlat
##- The gtfToGenePred software is available at http://hgdownload.cse.ucsc.edu/admin/exe/
##- http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/gtfToGenePred

gtfToGenePred -ignoreGroupsWithoutExons -genePredExt gencode.v19.annotation.gtf tmp

awk 'BEGIN{FS="\t"};{print $12"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' tmp > gencode.v19.annotation.refflat

