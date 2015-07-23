# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"piechart.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"piechart.png",args[2])
title    <- ifelse(is.na(args[3]),"Pie Chart",args[3])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
nrows <- nrow(data)
if( nrows < 2 ) {
  write(sprintf("ERROR: Expected at least 2 rows of data (plus header line) in data file %s\n",nFileIn),stderr())
  q(status=1)
}

colors <- c("#F1B556","#FFCFEF","#636767","#81A8C9","#A476A4")

if( nrows > length(colors) ) {
  colors <- c( colors, rainbow(length(colors)-nrows) )
}

slices <- as.numeric(data$Reads)
lbls <- data$Feature_Name

pcts <- round(slices/sum(slices)*100,1)
lbls <- paste(lbls, pcts)
lbls <- paste(lbls,"%",sep="")

png(nFileOut,width=700,height=700)
par(mar=c(1,2.4,1,3.5),cex=1.5)
pie( slices, labels=lbls, main="", col=colors, border=NA )
#mtext(title,side=3,line=0,cex=2.5)

q()

