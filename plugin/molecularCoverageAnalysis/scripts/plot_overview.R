# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"coverage_overview",args[1])
nFileOut <- ifelse(is.na(args[2]),"coverage_overview",args[2])

allContigLabels <- TRUE

col_bkgd = "#F5F5F5"
col_plot = "#2D4782"
col_title = "#999999"
col_frame = "#DBDBDB"
col_line = "#D6D6D6"

if( !file.exists(nFileIn) )
{
  write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}
bcov <- read.table(nFileIn, header=TRUE, as.is=TRUE, sep="\t", comment.char="")

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
chr <- ''
lastChr <- ''
lastChrDraw <- ''
lastTick <- 0
maxLen <- 0
numChrs <- 0
for( i in 1:ndata )
{
  # take back end of bin id spanning contigs, except for first
  chr <- bcov$contigs[i]
  if( i == 1 ) {
    chr <- sub("(--.*)","",chr)
  } else {
    chr <- sub("(.*--)","",chr)
  }
  if( lastChr != chr ) {
    numChrs <- numChrs + 1
    lastChr = chr
    if( i - lastTick >= 10 || lastTick == 0 ) {
      lastChrDraw = chr
      lastTick = i
      useLabels <- append(useLabels,chr)
      useTicks <- append(useTicks,i)
      if( nchar(chr) > maxLen ) {
        maxLen = nchar(chr)
      }
    }
  }
}
# force R to draw last part of axis (if missing)
if( lastTick < ndata )
{
  if( ndata - lastTick >= 8 && lastChrDraw != chr ) {
    useLabels <- append(useLabels,chr)
    if( nchar(chr) > maxLen ) {
      maxLen = nchar(chr)
    }
  } else {
    useLabels <- append(useLabels,'')
  }
  useTicks <- append(useTicks,ndata)
}
if( numChrs < 2 ) {
  useLabels[1] <- ''
  useTicks <- append(useTicks,ndata/2)
  useLabels <- append(useLabels,lastChr)
  maxLen <- 5
}
if( maxLen > 12 ) {
  maxLen <- 12
}

png(nFileOut,width=800,height=200,bg=col_bkgd)
par(mfrow=c(1,1),bty = 'n',mar=(c(3+maxLen/6,4,1,0.2)+0.3))

yaxisTitle <- "Log10(Base Reads)"
logd = log10(1+bcov$reads)
ymax = as.integer(max(logd))+1
plot( logd, xaxs = 'i', yaxs = 'i', type='o', pch='.', ylim=c(0,ymax), xlim=c(1,xmax),
  lwd=2, col=col_plot, xaxt='n', xlab="Reference Location", ylab=yaxisTitle )
title(main=title,col.main=col_title)

if( allContigLabels && length(useTicks) > 10 ) {
  axis(1, at=useTicks, lab=useLabels, cex.axis=0.8, las=2)
} else {
  axis(1, at=useTicks, lab=useLabels, cex.axis=0.9)
}

box(which="plot",lty="solid",col=col_frame)

q()

