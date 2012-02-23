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
ndata <- length(bcov$counts)
if( ndata < 2 )
{
    write(sprintf("No data output to %s\n",nFileOut),stderr())
    q()
}

# avoid very high x0 plot skew
ymax <- max(bcov$counts)
title <- "Target Coverage"
if( bcov$counts[1] == ymax )
{
    ymax = max(bcov$counts[2:ndata])
    title <- sprintf("Target Coverage\n(Bases covered at 0 read depth = %d)",bcov$counts[1])
}
b = 10^(as.integer(log10(ymax)))
ymax = b * as.integer(1 + ymax/b)
xmax <- max(bcov$read_depth)

png(nFileOut,width=800,height=800)
par(mfrow=c(1,1),bty = 'n',mar=(c(5,5,4,5)+0.1))

plot( NULL, NULL, xaxs='i', yaxs='i', ylim=c(0,ymax), xlim=c(0,xmax), xlab="", ylab="" )

ydata = bcov$pc_cum_counts * 0.996 * ymax / 100
abline(h=(1:20)*(ymax/20), col="lightblue", lty="dotted", lwd=2)

par(new=TRUE)
plot( bcov$read_depth, bcov$counts, type='o', xaxs = 'i', yaxs = 'i', ylim=c(0,ymax), xlim=c(0,xmax),
     xlab="Read Depth", ylab="Bases Covered at Read Depth", main=title )
     
lines( 0:xmax, ydata, type='l', col="blue", lwd=2 )
ryax=(0:10)*(ymax/10)
rlab=paste(((0:10)*10),"%",sep="")
axis(4,at=ryax,labels=rlab,tck=-0.02,las = 1,col = "blue")
mtext("Bases Covered at >= Read Depth",4,padj=5)

q()

