# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"hbar.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"hbar.png",args[2])
title    <- ifelse(is.na(args[3]),"Horizontal Bar Chart",args[3])
xtitle   <- ifelse(is.na(args[4]),"Counts",args[4])
dscale   <- as.numeric(ifelse(is.na(args[5]),"1",args[5]))
fields   <- ifelse(is.na(args[6]),"4,3",args[6])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s",nFileIn),stderr())
  q(status=1)
}

data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols = ncol(data)
if( ncols < 2 ) {
  write(sprintf("ERROR: Expected at least 2 columns of data, including row ID field from data file %s",nFileIn),stderr())
  q(status=1)
}
nrows <- nrow(data)
if( nrows < 1 ) {
  write(sprintf("ERROR: Expected at least 1 rows of data (plus header line) in data file %s",nFileIn),stderr())
  q(status=1)
}

# grab axis names and extract desired columns data
pnames <- data[[1]]
if( fields != "" ) {
  columns = as.numeric(unlist(strsplit(fields,",")))
  data <- data[,columns,drop=F]
} else {
  data <- data[,-1,drop=F]
}
# grab field names for legend and strip out dots added in
#lnames <- colnames(data)
#lnames <- gsub("[.]"," ",lnames)
lnames <- c("Mapped Reads","Unmapped Reads")

# scale data for output
if( dscale > 0 ) {
  data <- data * dscale
}

# get upper axis assuming scaled
xmax <- as.integer(max(data[,2])+0.95)

# subtract aligned reads from total for barplot()
data[,2] <- data[,2] - data[,1]

# for this plot rows/columns need to be transposed
data <- t(data)
nrows <- nrow(data)
ncols <- ncol(data)

# finally swap order of rows for barplot
data <- data[,ncols:1,drop=F]
pnames <- pnames[ncols:1]

# default colors plus rainbow if more than typical 5
colors <- c("#F1B556","#FFCFEF","#636767","#81A8C9","#A476A4")
if( nrows > length(colors) ) {
  colors <- c( colors, rainbow(length(colors)-nrows) )
}

# adjust plot height when less than 8 barcodes
hgt <- if( ncols < 8 ) 70*ncols+140 else 700
if( ncols <= 1 ) 200
lyo <- if( ncols < 7 ) c(0.72,0.29,0.17,0.11,0.07,0.06)[ncols] else 0.05

png(nFileOut,width=700,height=hgt)
par(mar=c(5,9,6,2))
barplot( data, names=pnames, xlim=c(0,xmax), horiz=T, las=1, xlab=xtitle, col=colors, border=NA, cex.lab=1.3,
  legend=lnames, args.legend=list(x="top",horiz=T,cex=1.3,inset=c(0,-lyo),bty="n",x.intersp=0.5) )
mtext(title,line=3,cex=2.3,font=2)

q()

