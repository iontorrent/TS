# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# Run SFFTrim on the library sff in a given directory. 
# Run R code to create scatterplots of flow vs. score.

dir=$1
xdb=TACGTACGTCTGAGCATCGATCGATGTACAGC 
P1=ATCACCGACTGCCCATAGAGAGGCTGAGAC 
rcode=`sed s/sh$/R/ <<< $0`

(
	cd $dir
	in_sff=`ls R_201*.sff | egrep -v tf.sff$`
	time SFFTrim -a $P1 -c 10 -f $xdb -p -i $in_sff -o trim.10.sff > trim.10.out
	time SFFTrim -a $P1 -c  4 -f $xdb -p -i $in_sff -o trim.04.sff > trim.04.out
	R CMD BATCH $rcode
)


