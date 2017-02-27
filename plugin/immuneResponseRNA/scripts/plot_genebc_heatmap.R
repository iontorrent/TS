# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
# This script creates a default heatmap from a given matrix file (table with no row/col headers).

options(warn=1)
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"heatmap.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"heatmap.png",args[2])
r_util_fun <- ifelse(is.na(args[3]),"utilitFunctioins.R",args[3])
title    <- ifelse(is.na(args[4]),"Heatmap",args[4])
keytitle <- ifelse(is.na(args[5]),"Value",args[5])
maxgenes <- as.numeric(ifelse(is.na(args[6]),"-1",args[6]))
rdthresh <- as.numeric(ifelse(is.na(args[7]),"10000",args[7]))
minrpm   <- as.numeric(ifelse(is.na(args[8]),"100",args[8]))
nBarcode <- as.numeric(ifelse(is.na(args[9]),"0",args[9]))

source(r_util_fun)

if( !file.exists(nFileIn) ) {
    write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
    q(status=1)
}


# read in matrix file and check expected format
data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 1 data column plus row ids in data file %s\n",nFileIn),stderr())
    q(status=1)
}
nrows <- nrow(data)
if( nrows < 1 ) {
    write(sprintf("ERROR: Expected at least 1 row of data plus header line in data file %s\n",nFileIn),stderr())
    q(status=1)
}
if( maxgenes < 0 || maxgenes > nrows ) { maxgenes = nrows }

# grab row names and strip extra columns
lnames <- data[[1]]
data <- data[,-c(1,2),drop=FALSE]
if( nBarcode > 0 ) {
    data <- data[,1:nBarcode,drop=FALSE]
}
data <- as.matrix(data)
ncols <- ncol(data)
if( ncols < 1 ) {
    write(sprintf("ERROR: Expected at least 1 data column after removing annotation columns in %s\n",nFileIn),stderr())
    q(status=1)
}

# remove columns whose sums are less than the counts threshold
trds <- apply(data,2,sum)
gidx <- trds >= rdthresh
data <- data[,gidx]
ocols <- ncols
ncols <- ncol(data)
if( ncols < ocols ) {
    write(sprintf("Warning: Gene heatmap plot excluded %d barcodes as having less than %d aligned reads.",ocols-ncols,rdthresh),stderr())
}

# convert to RPM reads
#- add 0.01 to the total reads to accomodate barcode/sample with 0 total reads
trds <- apply(data,2,sum) + 0.01
data <- sweep(data,2,1000000/trds,'*')

novmin <- sum(apply(data,1,max) >= minrpm)
if( novmin < maxgenes ) {
    write(sprintf("Warning: Gene heatmap plot reduced number of genes from %d to %d genes with more than %d RPM",maxgenes,novmin,minrpm),stderr())
    maxgenes <- novmin
}

# determine best n genes by maxmima across columns
covar <- function(x) {
    if( max(x) < minrpm ) { return(0) }
    m <- mean(x,na.rm=TRUE)
    if( m <= 0 ) { m = 1 }
    sd(x,na.rm=TRUE)/m
}
kidx <- order(apply(data,1,covar),decreasing=T)[1:maxgenes]
lnames <- lnames[kidx]
data <- data[kidx,]

# convert to log10(x+1)
data <- log10(data+1)

# color threshold, fixed at 0 for lowest value
ncolors = 200
#pbreaks = ncolors+1
maxd <- max(data)
if( maxd <= 0 ) { maxd = 1 }
pbreaks <- seq( 0, maxd, maxd/ncolors )

#colors=heat.colors(ncolors)
#colors=colorRampPalette(c("blue","cyan","white","yellow","red"))(ncolors)
colors <- colorRampPalette(rev(brewer.pal(name="RdBu",n=8)))

invalid <- function (x) 
{
    if (missing(x) || is.null(x) || length(x) == 0) 
        return(TRUE)
    if (is.list(x)) 
        return(all(sapply(x, invalid)))
    else if (is.vector(x)) 
        return(all(is.na(x)))
    else return(FALSE)
}

#----------------------------------------------------------------

# view needs to scale by number of rows but keeping titles areas at same absolute sizes
# E.g. at 900 height want lhei=c(1.4,5,0.25,0.9)
#wid <- 200+50*ncols
wid <- 800
hgt <- 200 + 12 * maxgenes
a <- 900*1.4/hgt
c <- 900*0.25/hgt
d <- 900*0.9/hgt
b <- 7.55-a-c-d

png(nFileOut,width=wid,height=hgt)
heatmap_2( data, col=colors, main=title, symkey=FALSE,
    lmat=rbind(c(0,3,0),c(2,1,0),c(0,0,0),c(0,4,0)), lwid=c(1,5,0.2), lhei=c(a,b,c,d),
    density.info="none", trace="none", breaks=pbreaks, key.abs=TRUE, labRow=lnames,
    key.xlab = "", key.title = keytitle, cexRow=1.5, cexCol=1.5, margins=c(8,9)
)
dev.off()

q()
