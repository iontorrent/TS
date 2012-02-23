# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)
nFileIn <- ifelse(is.na(args[1]),"on_target.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"on_target.png",args[2])
plotGC <- ifelse(is.na(args[3]),0,as.integer(args[3]))

# read in data
if( !file.exists(nFileIn) )
{
     write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
     q()
}
mydata <- read.table(nFileIn, header=TRUE)
chr.unique = unique(mydata$chrom_id)

# define multiplot layout
numplot = length(chr.unique)
if( numplot == 0 ) {
     write(sprintf("No data output to %s\n",nFileOut),stderr())
     q()
}
ncol = floor(sqrt(numplot))
nrow = ceiling(numplot/ncol)
if( ncol == 2 && nrow == 4 )
{
     ncol = 3
     nrow = 3
}

wd = ifelse(ncol==1,640,ifelse(ncol<4,ncol*320,1280))
ht = ifelse(nrow==1,320,ifelse(nrow<6,nrow*160,960))
png(nFileOut,width=wd,height=ht)
par(mfrow=c(nrow,ncol))

# this gives common height based of maximum hits across all chromosomes
ymax <- max(mydata$norm_reads)
if( ymax > 0 )
{
     b <- 10^(as.integer(log10(ymax))-1)
     ymax <- b * (as.integer(ymax/b)+1)
} else {
     ymax = 1
}

par(bty = 'n',mar=c(2,3,4,1),oma = c(0,0,3,0))
for( chr in chr.unique )
{
     sel = (mydata$chrom_id == chr)
     pos <- mydata$start_pos[sel]
     cnt <- mydata$norm_reads[sel]
     cov <- 0.01*mydata$pc_coverage[sel]*cnt

     bdata = barplot( t(cbind(cov,cnt-cov)), beside=FALSE, col=c("grey","red"),
          names.arg=pos, ylim=c(0,ymax), xaxs = 'i', yaxs = 'i', #border = NA, space=0,
          axisnames = FALSE, main=chr, ann=FALSE, xaxt='n' )

    if( plotGC == 1 )
    {
        ydata = mydata$pc_gc_reads[sel] * ymax / 100
        lines( bdata, ydata, type='p', col="blue", lwd=2, pch=3 )
    }
}
mtext("Reads per Target Base", outer = TRUE, cex = 1.5)

q()
