# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# This script handles optionally creating different plots at the same time.
# A single script is used to avoid having to re-read the data and making similar transformations.
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"rna_rep",args[1])
nFileOut <- ifelse(is.na(args[2]),"rna_rep",args[2])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

rcov <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")

# test for property
a <- rcov$total_reads
ndata <- length(a)
if( ndata < 2 )
{
  write(sprintf("ERROR: No coverage property field found in data file %s\n",nFileIn),stderr())
  q(status=1)
}
bars = c( length(a[a<10]), length(a[10<=a & a<100]), length(a[100<=a & a<1000]), length(a[1000<=a & a<10000]), length(a[10000<=a & a<100000]), length(a[a>=100000]) )

xnames=c("<10","10-100","100-1K","1K-10K","10K-100K","100K+")
ymax = max(bars)+1

png(nFileOut,width=400,height=200)
par(mfrow=c(1,1),bty = 'n',mar=(c(4,4,2,0)+0.3))

barplot( t(bars), names.arg=xnames, ylim=c(0,ymax), yaxs='i', col="grey", cex.names=0.8, cex.axis=0.9,
  xlab="Binned numbers of reads", ylab="Number of Amplicons", main="Representation Overview" )

#a <- rcov$fwd_cov + rcov$rev_cov
#pass = c( length(a[a<10]), length(a[10<=a & a<100]), length(a[100<=a & a<1000]), length(a[1000<=a & a<10000]), length(a[10000<=a & a<100000]), length(a[a>=100000]) )
#legendTitle <- c("Drop Out (0 Assigned Reads)", "Fail (Assigned Reads < 0.2x mean)", "Pass", "Pass/Over (Assigned Reads >= 5x mean)" )
#lcols=c("grey","white")
#barplot( t(cbind(pass,bars-pass)),
#  names.arg=xnames, beside=FALSE, ylim=c(0,ymax), xaxs='i', yaxs='i', col=lcols,
#  xlab="Binned numbers of reads", ylab="Number of Amplicons", main="Representation Overview" )
#space=0, , cex.main=1.6, cex.lab=1.4 )

q()

