# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"pdfplot.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"pdfplot.png",args[2])
nBarcode <- ifelse(is.na(args[3]),0,args[3])
title    <- ifelse(is.na(args[4]),"Frequency Plot",args[4])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

col_bkgd = "#E5E5E5"

# read in matrix file and check expected format
data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols = ncol(data)
if( ncols < 2 ) {
  write(sprintf("ERROR: Expected at least 2 columns of data, including row ID field from bcmatrix file %s\n",nFileIn),stderr())
  q(status=1)
}
nrows = nrow(data)
if( nrows < 1 ) {
  write(sprintf("ERROR: Expected at least 1 row of data plus header line bcmatrix file %s\n",nFileIn),stderr())
  q(status=1)
}

# Remove first column and any (annotation) columns after barcode data
data <- data[,-1,drop=FALSE]
if( nBarcode > 0 ) {
  data <- data[,1:nBarcode,drop=FALSE]
}

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

# values of 0 go to -inf and are ignored for histogram, but wanted here for normalization
#totals <- colSums(data)
data <- log10(data)

xmax = as.integer(max(data)+0.95)
if( xmax < 1 ) xmax = 1

# historam plot is made using int(log10) intevals, e.g. with intervals as if ticks were one linear values
# e.g. 1..10,20..100,200..1000,2000..10000
# for consistency here, the minimum histogram width is 1..10, with the limit always being an integer power of 10
#brks <- c(0)
#tcks <- log10(2:10)
#for( i in 1:xmax-1 ) {
#  brks <- c(brks,tcks+i)
#}
#write(brks,stderr())

# binning divisons of 3 or 4 work best at low read counts and gives answer closest to PDF
div <- 3
brks <- (0:(xmax*div))/div

ymax <- 0
for( i in 1:nplot) {
  h <- hist(data[[i]],breaks=brks,right=F,plot=F)
  ym <- max( h$counts )
  if( ym > ymax ) { ymax = ym }
}
ymax <- (ymax+0.001)*1.05

png(nFileOut,width=700,height=hgt)
par(mfrow=c(1,1),mar=(c(3.5,2,3,mrg)),cex=2)
plot( data[[1]], type="n", xlim=c(0,xmax), ylim=c(0,ymax), yaxt='n', xlab="", col=colors[1], main="" )
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col=col_bkgd)
grid( col="white", lwd=1, lty=1 )
title(main=title)
mtext("Frequency of Genes with N Reads (N > 0)",side=2,line=0.6,cex=1.8)
mtext("Number of Reads : log10(N)",side=1,line=2,cex=1.8)
box()

for( i in 1:nplot) {
  h <- hist(data[[i]],breaks=brks,right=F,plot=F)
  b <- h$breaks[-length(h$breaks)]
  lines(b,h$counts,col=colors[i],lwd=2)
}
if( nplot > 1 ) {
  legend( legend=lnames, xpd=T, x="topright",cex=0.6,inset=c(-0.346,lgd),fill=colors,bty="n",x.intersp=0.5 )
}

q()
