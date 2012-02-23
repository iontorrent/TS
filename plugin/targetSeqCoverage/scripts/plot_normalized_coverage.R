# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"coverage_depth.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"coverage_depth.png",args[2])

if( !file.exists(nFileIn) )
{
     write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
     q()
}
bcov <- read.table(nFileIn, header=TRUE)

# check to avoid no coverage cases
ndata <- length(bcov$pc_cum_counts)
if( ndata < 2 )
{
     write(sprintf("No data output to %s\n",nFileOut),stderr())
     q()
}

# avoid very high x0 plot skew
ymax <- 1.0
xmax <- 1.0

png(nFileOut,width=800,height=800)
par(mfrow=c(1,1),bty = 'n',mar=(c(5,5,4,5)+0.1))

plot( NULL, NULL, xaxs='i', yaxs='i', ylim=c(0,ymax), xlim=c(0,xmax), xlab="", ylab="" )

abline(v=(seq(0.1,1,0.1)), col="grey", lty="dotted", lwd=2 )
abline(h=(seq(0.1,1,0.1)), col="grey", lty="dotted", lwd=2)

par(new=TRUE)
plot( bcov$norm_depth, 0.01*bcov$pc_cum_counts, type='l', xaxs = 'i', yaxs = 'i', ylim=c(0,ymax), xlim=c(0,xmax),
     xlab="Normalized Coverage", ylab="Fraction of Bases Covered", main="Normalized Target Coverage" )

q()

