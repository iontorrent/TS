# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"isopdfplot.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"isopdfplot.png",args[2])
title    <- ifelse(is.na(args[3]),"Frequency Plot",args[3])
minFPKM  <- as.numeric(ifelse(is.na(args[4]),0.03,args[4]))

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

col_bkgd = "#E5E5E5"

# read in matrix file and check expected format
data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols = ncol(data)
if( ncols < 2 ) {
  write(sprintf("ERROR: Expected at least 2 columns of data, including row ID field from barcode x isoform FPKM matrix file %s\n",nFileIn),stderr())
  q(status=1)
}
nrows = nrow(data)
if( nrows < 1 ) {
  write(sprintf("ERROR: Expected at least 1 row of data plus header line barcode x isoform FPKM matrix file %s\n",nFileIn),stderr())
  q(status=1)
}

# Remove first column and any (annotation) columns after barcode data and take log2(x+1) of counts
data <- data[,-1,drop=FALSE]
data <- log10(data+1)
minLog <- log10(minFPKM+1)

nplot <- ncol(data)
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

lnames <- colnames(data)

xmax <- max(data)
nbrks <- as.integer(xmax*10)+1

ymax <- 0
for( i in 1:nplot) {
  h <- hist(data[[i]],breaks=nbrks,plot=F)
  b <- h$breaks[-length(h$breaks)]
  ym <- max( h$counts[b>=minLog] )
  if( ym > ymax ) { ymax = ym }
}
ymax <- (ymax+0.0001)*1.01

png(nFileOut,width=700,height=hgt)
par(mfrow=c(1,1),mar=(c(3.5,2,3,mrg)),cex=2)
plot( data[[1]], type="n", xlim=c(minLog,xmax), ylim=c(0,ymax), yaxt='n', xlab="", col=colors[1], main="" )
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col=col_bkgd)
grid( col="white", lwd=1, lty=1 )
title(main=title)
mtext(sprintf("Frequency of Isoforms with FPKM >= %g",minFPKM),side=2,line=0.6,cex=1.8)
mtext("Representation : log10(FPKM+1)",side=1,line=2,cex=1.8)
box()

for( i in 1:nplot ) {
  # plot histogram heights at breaks, except the last
  h <- hist(data[[i]],breaks=nbrks,plot=F)
  b <- h$breaks[-length(h$breaks)]
  c <- h$counts
  # remove points below threshold. Note: hist does a good job of removing breaks at end where there are beyond the max of data
  c[b<minLog] <- NA
  lines(b,c,col=colors[i],lwd=2)
}
if( nplot > 1 ) {
  legend( legend=lnames, xpd=T, x="topright",cex=0.6,inset=c(-0.346,lgd),fill=colors,bty="n",x.intersp=0.5 )
}

q()
