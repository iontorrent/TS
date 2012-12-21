# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"coverage_overview",args[1])
nFileOut <- ifelse(is.na(args[2]),"coverage_overview",args[2])

allContigLabels <- TRUE

if( !file.exists(nFileIn) )
{
  write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}
bcov <- read.table(nFileIn, header=TRUE, as.is=TRUE)

# check to avoid no coverage cases
ndata <- length(bcov$reads)
if( ndata < 2 )
{
  write(sprintf("No data output to %s\n",nFileOut),stderr())
  q(status=1)
}

xmax = ndata
title <- "Coverage Overview"

# show where changes in labels occur (e.g. each chromosome start/end)
useLabels <- c()
useTicks <- c()
lastChr <- ''
lastTick <- 0
for( i in 1:ndata )
{
  chr = bcov$contigs[i]
  # take back end of bin id spanning contigs
  n <- regexpr('-',chr)
  while( n > 0 ) {
    chr <- substring(chr,n+1)
    n <- regexpr('-',chr)
  }
  if( lastChr != chr ) {
    lastTick = i
    lastChr = chr
    useLabels <- append(useLabels,chr)
    useTicks <- append(useTicks,i)
  }
}
# force R to draw last part of axis
if( lastTick < ndata )
{
  useLabels <- append(useLabels,'')
  useTicks <- append(useTicks,ndata)
}

png(nFileOut,width=800,height=200)
par(mfrow=c(1,1),bty = 'n',mar=(c(4,4,1,0.2)+0.3))

yaxisTitle <- "Log10(Base Reads)"
logd = log10(1+bcov$reads)
ymax = as.integer(max(logd))+1
plot( logd, xaxs = 'i', yaxs = 'i', type='o', pch='.', ylim=c(0,ymax), xlim=c(1,xmax), 
  xaxt='n', xlab="Reference Position", ylab=yaxisTitle, main=title )

if( allContigLabels && length(useTicks) > 10 ) {
  axis(1, at=useTicks, lab=useLabels, cex.axis=0.8, las=2)
} else {
  axis(1, at=useTicks, lab=useLabels, cex.axis=0.9)
}

q()

