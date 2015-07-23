# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"hbar.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"hbar.png",args[2])
title    <- ifelse(is.na(args[3]),"Horizontal Bar Chart",args[3])
xtitle   <- ifelse(is.na(args[4]),"Counts",args[4])
dscale   <- as.numeric(ifelse(is.na(args[5]),"1",args[5]))

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols = ncol(data)
if( ncols < 2 ) {
  write(sprintf("ERROR: Expected at least 2 columns of data, including row ID field from data file %s\n",nFileIn),stderr())
  q(status=1)
}
nrows <- nrow(data)
if( nrows < 1 ) {
  write(sprintf("ERROR: Expected at least 1 row of data (plus header line) in data file %s\n",nFileIn),stderr())
  q(status=1)
}

# grab row names and strip, convert to matrix and reverse column order (for barplot)
lnames <- data[[1]]
data <- as.matrix(data[-1])
ncols <- ncol(data)
data <- data[,ncols:1,drop=F]

# scale data for output
if( dscale > 0 ) {
  data <- data * dscale
}

# default colors plus rainbow if more than typical 5
colors <- c("#F1B556","#FFCFEF","#636767","#81A8C9","#A476A4")
if( nrows > length(colors) ) {
  colors <- c( colors, rainbow(length(colors)-nrows) )
}

# attempt to make better legend with equally spaced text (by adding spaces)
#frm <- sprintf("%%-%ds",5+max(nchar(lnames)))
#lnames <- sprintf(frm,lnames)
lnames <- sprintf("%s    ",lnames)

# adjust plot height when less than 8 barcodes
hgt <- if( ncols < 8 ) 70*ncols+140 else 700
lyo <- if( ncols < 8 ) 0.37-0.04*ncols else 0.05
# account for missing space with only 1 bar
if( ncols <= 1 ) {
  hgt <- 200
  lyo <- 0.8
}

png(nFileOut,width=700,height=hgt)
par(mar=c(5,9,6,2))
barplot( data, names=colnames(data), horiz=T, las=1, xlab=xtitle, col=colors, border=NA, cex.lab=1.3,
  legend=lnames, args.legend=list(x="top",horiz=T,cex=1.3,inset=c(0,-lyo),bty="n",x.intersp=0.5) )
mtext(title,line=3,cex=2.3,font=2)

q()

