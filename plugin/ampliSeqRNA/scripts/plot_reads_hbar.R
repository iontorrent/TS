# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"hbar.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"hbar.png",args[2])
title    <- ifelse(is.na(args[3]),"Horizontal Bar Chart",args[3])
xtitle   <- ifelse(is.na(args[4]),"Counts",args[4])
dscale   <- as.numeric(ifelse(is.na(args[5]),"1",args[5]))
fields   <- ifelse(is.na(args[6]),"6,5,4,7,3",args[6])

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
  write(sprintf("ERROR: Expected at least 1 row of data (after header line) in data file %s\n",nFileIn),stderr())
  q(status=1)
}

# grab axis names and extract desired columns data, default fields assume:
# Barcode,Sample,Total,Mapped,On-Target,Assigned,ERCC  ->  Assigned,On-Target,Mapped,ERCC,Total
pnames <- data[[1]]
if( fields != "" ) {
  columns = as.numeric(unlist(strsplit(fields,",")))
  data <- data[,columns]
} else {
  data <- as.numeric(data[-1])
}
# scale data for output
if( dscale > 0 ) {
  data <- data * dscale
}

# get upper axis assuming scaled
xmax <- as.integer(max(data[,2])+0.95)

# subtract lists to get differences to get:
lnames <- c("Valid","Filtered","Off Target","ERCC","Unmapped")
# total => unmapped = total - mapped
data[,5] <- data[,5] - data[,3]
# mapped => off-target = mapped - on-target - ercc
data[,3] <- data[,3] - data[,2] - data[,4]
# on-target => filtered = on-target - assigned
data[,2] <- data[,2] - data[,1]

# for this plot rows/columns need to be transposed
data <- t(data)
nrows <- nrow(data)
ncols <- ncol(data)

# finally swap order of rows for barplot
# note: converts table back to vector when only 1 column!!!
if( ncols > 1 ) {
  data <- data[,ncols:1]
  pnames <- pnames[ncols:1]
}

# default colors plus rainbow if more than typical 5
colors <- c("#F1B556","#FFCFEF","#636767","#81A8C9","#A476A4")
if( nrows > length(colors) ) {
  colors <- c( colors, rainbow(length(colors)-nrows) )
}

png(nFileOut,width=700,height=700)
par(mar=c(5,9,6,2))
barplot( data, names=pnames, xlim=c(0,xmax), horiz=T, las=1, xlab=xtitle, col=colors, border=NA, cex.lab=1.3,
  legend=lnames, args.legend=list(x="top",horiz=T,cex=1.3,inset=c(0,-0.05),bty="n",x.intersp=0.5) )
mtext(title,line=3,cex=2.3,font=2)

q()

