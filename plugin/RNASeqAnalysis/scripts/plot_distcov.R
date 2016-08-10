# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)
nargs = length(args)
if( nargs < 2 ) {
  write("ERROR: plot_distcov.R requires an output (PNG) file name and at least one input file to process\n",stderr())
  q(status=1)
}
nFileOut <- args[1]

col_bkgd = "#E5E5E5"
col_plot = "#4580B6"
col_frame = "#CCCCCC"
col_title = "#999999"
col_line = "#D6D6D6"
col_grid = "#FFFFFF"
col_fitline = "goldenrod"

nplot <- nargs-1
if( nplot > 1 ) {
  hgt <- 600
  mrg <- 6
  # shaded color sets are hard to distinguish, most qualative brewer sets limited to 8 or 12
  nr <- if( nplot > 6 ) 7 else nplot
  colors <- c(rainbow(nr),"#3F3F3F",brewer.pal(12,"Paired"),brewer.pal(8,"Set2"),brewer.pal(8,"Dark2"))
} else {
  hgt <- 700
  mrg <- 0.5
  colors <- "black"
}
lgd <- 0.005 - 0.001 * nplot

# get max y across all files: may as well validate here
ymax <- 0
lnames <- character(0)
for( i in 1:nplot) {
  nFileIn <- args[i+1]
  if( !file.exists(nFileIn) ) {
    write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
    q(status=1)
  }
  rcov <- try( read.table(nFileIn, skip=10, header=TRUE, sep="\t", as.is=T, comment.char="#"), T )
  ndata <- if( class(rcov) == "try-error") 0 else length(rcov$normalized_position)
  if( ndata < 2 ) {
    if( nplot == 1 ) {
      write(sprintf("ERROR: No transcript coverage data in '%s'\n",nFileIn),stderr())
      q(status=1)
    }
    write(sprintf("WARNING: No transcript coverage data in '%s'\n",nFileIn),stderr())
    next
  }
  ym <- max(rcov$All_Reads.normalized_coverage)
  if( ym > ymax ) { ymax = ym }
  lnames <- c(lnames,basename(dirname(nFileIn)))
}
ymax <- (ymax+0.0001)*1.01

# loop again for making the plots
usedcols <- c()
firstPlot = TRUE
for( i in 1:nplot) {
  nFileIn <- args[i+1]
  rcov <- try( read.table(nFileIn, skip=10, header=TRUE, sep="\t", as.is=T, comment.char="#"), T )
  ndata <- if( class(rcov) == "try-error") 0 else length(rcov$normalized_position)
  if( ndata < 2 ) next
  a <- rcov$normalized_position
  b <- rcov$All_Reads.normalized_coverage
  if( firstPlot ) {
    png(nFileOut,width=700,height=hgt)
    par(mfrow=c(1,1),mar=c(3.5,3.5,3,mrg),cex=2)
    plot( a, b, type="n", xlab="", ylab="", ylim=c(0,ymax) )
    rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col=col_bkgd)
    grid( col="white", lwd=1, lty=1 )
    title(main="Normalized Transcript Coverage")
    mtext("Normalized Coverage",side=2,line=2.4,cex=1.8)
    mtext("Normalized Distance Along Transcript",side=1,line=2.2,cex=1.8)
    box()
    firstPlot = FALSE
  }
  lines( a, b, type="l", col=colors[i], lwd=2 )
  usedcols <- c(usedcols,colors[i])
}
if( nplot > 1 ) {
  legend( legend=lnames, xpd=T, x="topright",cex=0.6,inset=c(-0.38,lgd),fill=usedcols,bty="n",x.intersp=0.5 )
}
q()

