# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"coverage_bin_depth.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"coverage_bin_depth.png",args[2])
option <- ifelse(is.na(args[3]),"",args[3])

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

png(nFileOut,width=800,height=800)
par(mfrow=c(1,1),bty = 'n',mar=(c(5,5,4,5)+0.1))
if( option == 1 )
{
     ymax = as.integer(max(bcov$num_reads))+1
     par(yaxt='n')
     bdata <- barplot( bcov$num_reads, names.arg=bcov$read_depth, ylim=c(0,ymax),
          xlab="Read Depth", ylab="Reads", main="Distribution of Target Reads vs. Read Depth" )
     ydata = bcov$pc_cum_num_reads * 0.996 * ymax / 100
     lines( bdata, ydata, type='o', col="blue", lwd=2 )
     par(yaxt='s')
     ystep = ifelse(ymax >= 50, 5, ifelse(ymax >= 20, 2, 1))
     ryax=(0:as.integer((ymax+ystep)/ystep))*ystep
     rlab=paste(ryax,"%",sep="")
     axis(2,at=ryax,labels=rlab,tck=-0.02,las = 1)
     ryax=(0:10)*(ymax/10)
     rlab=paste(((0:10)*10),"%",sep="")
     axis(4,at=ryax,labels=rlab,tck=-0.02,las = 1,col = "blue")
     mtext("Cumulative Reads",4,padj=5)
} else
{
     ymax <- max(bcov$counts)
     b = 10^(as.integer(log10(ymax)))
     ymax = b * as.integer(1 + ymax/b)
     # add xaxs='i' to remove gap between bars and axis: seems difficult to set to nice spacing like used by barp()
     bdata <- barplot( bcov$counts, names.arg=bcov$read_depth, ylim=c(0,ymax),
          xlab="Read Depth", ylab="Bases Covered at Read Depth", main="Target Coverage" )
     ydata = bcov$pc_cum_counts * 0.996 * ymax / 100
     lines( bdata, ydata, type='o', col="blue", lwd=2 )
     ryax=(0:10)*(ymax/10)
     rlab=paste(((0:10)*10),"%",sep="")
     axis(4,at=ryax,labels=rlab,tck=-0.02,las = 1,col = "blue")
     mtext("Bases Covered at >= Read Depth",4,padj=5)
}
q()

