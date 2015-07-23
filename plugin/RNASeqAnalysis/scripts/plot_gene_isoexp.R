# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"geneisoexp",args[1])
nFileOut <- ifelse(is.na(args[2]),"geneisoexp",args[2])
title    <- ifelse(is.na(args[3]),"Gene Isoform Expression",args[3])
maxisofm <- as.numeric(ifelse(is.na(args[4]),0,args[4]))

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols <- ncol(data)
if( ncols != 3 ) {
  write(sprintf("ERROR: Input error. Expected 3 columns of data data file %s\n",nFileIn),stderr())
  q(status=1)
}
nrows <- nrow(data)
if( nrows < 2 ) {
  write(sprintf("ERROR: Expected at least 2 rows of data (plus header line) in data file %s\n",nFileIn),stderr())
  q(status=1)
}

col_bkgd = "#E5E5E5"

# remove data for genes with more than maximum limit (if provided)
maxiso <- max(data$num_isoforms)
if( maxisofm > 0 && maxiso > maxisofm ) {
  maxiso <- maxisofm
  data <- data[ drop=F, data$num_isoforms <= maxisofm, ]
}

# add records for missing isoform counts (so x-axis look linear integers, as expected)
missing <- setdiff( 1:maxiso, data$num_isoforms )
if( length(missing) > 0 ) {
  data <- rbind( data, data.frame(gene_id='',num_isoforms=missing,isoforms_detected=0) )
}

outptsz <- 1 - maxiso/200
if( outptsz < 0.3 ) outptsz <- 0.3
xaxs <- if( maxiso < 15 ) 'r' else 'i'

# create boxplot with ggplot2 styling - unortunately boxplot() has to be called twice to get bkgd grid (?)
png(nFileOut,width=700,height=700)
par(mfrow=c(1,1),mar=c(4,4,3,1),cex=1.5)
boxplot( isoforms_detected~num_isoforms, data=data, ylim=c(0,maxiso), xlim=c(1,maxiso), yaxt='n', xaxt='n', xaxs=xaxs )
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col=col_bkgd)
glines <- seq(0,maxiso,5)
axis(2,at=glines,las=1) 
abline( h=glines, v=glines, col="white", lwd=1, lty=1 )
par(new=TRUE)
boxplot( isoforms_detected~num_isoforms, data=data, col="white", ylim=c(0,maxiso), xlim=c(1,maxiso), yaxt='n', xaxs=xaxs, outcex=outptsz )

# add expectation line, with 0's virtual genes where a particular number of isoforms is missing
expt <- 1:maxiso
if( length(missing) > 0 ) expt[missing] <- 0
lines(1:maxiso,expt,type="p",pch=20,col='darkgrey')

# add the labels
title(main=title,cex.main=1.8);
mtext("Annotated Isoforms Per Gene",side=1,line=2.7,cex=1.8)
mtext("Isoforms Expressed Per Gene",side=2,line=2.7,cex=1.8)

q()

