# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# This script handles optionally creating different plots at the same time.
# A single script is used to avoid having to re-read the data and making similar transformations.
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"rna_rep",args[1])
nFileOut <- ifelse(is.na(args[2]),"rna_rep",args[2])
readType <- ifelse(is.na(args[3]),"Amplicons",args[3])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

col_bkgd = "#F5F5F5"
col_plot = "#2D4782"
col_title = "#999999"
col_frame = "#DBDBDB"
col_line = "#D6D6D6"

rcov <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")

# test for property
a <- rcov$total_reads
ndata <- length(a)
if( ndata < 2 )
{
  write(sprintf("ERROR: plot_rna_rep.R: No coverage property field found in data file %s\n",nFileIn),stderr())
  q(status=1)
}
bars = c( length(a[a<10]), length(a[10<=a & a<100]), length(a[100<=a & a<1000]), length(a[1000<=a & a<10000]), length(a[10000<=a & a<100000]), length(a[a>=100000]) )

xnames=c("<10","10-100","100-1K","1K-10K","10K-100K","100K+")
ymax = max(bars)+1

png(nFileOut,width=400,height=200,bg=col_bkgd)
par(mfrow=c(1,1),bty = 'n',mar=(c(4,4,2,0)+0.3))

barplot( t(bars), names.arg=xnames, ylim=c(0,ymax), yaxs='i', col=col_plot, cex.names=0.8, cex.axis=0.9,
  xlab="Binned numbers of reads", ylab=sprintf("Number of %s",readType) )

title(main="Representation Overview",col.main=col_title)
box(which="plot",lty="solid",col=col_frame)

q()

