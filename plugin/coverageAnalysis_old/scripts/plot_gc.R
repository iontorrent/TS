# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# This script handles optionally creating different plots at the same time.
# A single script is used to avoid having to re-read the data and making similar transformations.
args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"fine_coverage.xls",args[1])

# current options:
# "a" for amplicon reads (vs. base coverage)
# "f" or "F" output pass/fail (>=0.2 of Mean) counts vs. binned gc plot: "F" for 4-color plot
# "g" or "G" output cov vs. gc plot: "G" for log axis
# "k" or "K" output pass/fail (>=0.2 of Mean) counts vs. target length plot: "K" for 4-color plot
# "l" or "L" output cov vs. length plot: "L" for log axis
option <- ifelse(is.na(args[2]),"",args[2])

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

# output size is reduced to fit pefectly into display widget
picWidth = 1.5*798
picHeight= 1.5*264

rcov <- read.table(nFileIn, header=TRUE, as.is=TRUE, sep="\t", comment.char="")

# test the type of the field in case the wrong type of targets are provided
if( grepl("a",option) ) {
  yprop <- rcov$total_reads
  if( is.null(yprop) ) yprop <- rcov$depth
  legoff = 0.89
} else {
  yprop <- rcov$depth
  if( is.null(yprop) ) yprop <- rcov$total_reads
  legoff = 0.86
}
if( is.null(yprop) ) {
  write(sprintf("ERROR: Cannot locate fields 'depth' or 'total_reads' in data file %s\n",nFileOut),stderr())
  q(status=1)
}

# test for property
ndata <- length(yprop)
if( ndata < 2 )
{
  write(sprintf("ERROR: No coverage property field found in data file %s\n",nFileIn),stderr())
  q(status=1)
}

tlen <- rcov$contig_end - rcov$contig_srt + 1
pcgc <- 100 * rcov$gc / tlen
aver <- mean(yprop)
av02 <- 0.2 * aver

# Create GC vs. coverage scatter plot
if( grepl("g",option,ignore.case=TRUE) ) {
  nFileOut <- sub(".xls$",".gc.png",nFileIn)
  png(nFileOut,width=picWidth,height=picHeight)
  par(mar=(c(4,4,2,0.2)+0.3))

  if( grepl("a",option) ) {
    title <- "Number of Amplicon Reads vs. Amplicon GC Content"
    xaxisTitle <- "Amplicon C/G Base Content (%)"
    yaxisTitle <- "Amplicon Reads"
    legendTitle <- "Mean Read Count"
  } else {
    title <- "Number of Target Base Read Depth vs. Target C/G Content"
    xaxisTitle <- "Target C/G Base Content (%)"
    yaxisTitle <- "Base Read Depth"
    legendTitle <- "Mean Read Depth"
  }
  if( grepl("G",option) ) {
    yaxisTitle <- sprintf("Log10(%s)", yaxisTitle)
    ydata <- log10(1+yprop)
    laver <- log10(1+aver)
    lav02 <- log10(1+av02)
  } else {
    ydata <- yprop
    laver <- aver
    lav02 <- av02
  }
  plot( pcgc, ydata, pch=4, xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  abline(h=laver,col="green")
  abline(h=lav02,col="red")
  legend(legoff*max(pcgc), max(ydata), legend=c(legendTitle,"0.2 x Mean"), cex=1.2, bty="n", col=c("green","red"), lty=1)
}
  
# Create GC vs. length scatter plot
if( grepl("l",option,ignore.case=TRUE) ) {
  nFileOut <- sub(".xls$",".len.png",nFileIn)
  png(nFileOut,width=picWidth,height=picHeight)
  par(mar=(c(4,4,2,0.2)+0.3))
  if( grepl("a",option) ) {
    title <- "Number of Amplicon Reads vs. Amplicon Length"
    xaxisTitle <- "Amplicon Length"
    yaxisTitle <- "Amplicon Reads"
    legendTitle <- "Mean Read Count"
  } else {
    title <- "Number of Target Base Read Depth vs. Target Length"
    xaxisTitle <- "Target Length"
    yaxisTitle <- "Base Read Depth"
    legendTitle <- "Mean Read Depth"
  }
  if( grepl("L",option) ) {
    yaxisTitle <- sprintf("Log10(%s)", yaxisTitle)
    ydata <- log10(1+yprop)
    laver <- log10(1+aver)
    lav02 <- log10(1+av02)
  } else {
    ydata <- yprop
    laver <- aver
    lav02 <- av02
  }
  plot( tlen, ydata, pch=4, xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  abline(h=laver,col="green")
  abline(h=lav02,col="red")
  legend(legoff*max(tlen), max(ydata), legend=c(legendTitle,"0.2 x Mean"), cex=1.2, bty="n", col=c("green","red"), lty=1)
}
  
# Create GC vs. pass/fail plot
if( grepl("f",option,ignore.case=TRUE) ) {
  nFileOut <- sub(".xls$",".fedora.png",nFileIn)
  png(nFileOut,width=picWidth,height=picHeight)
  par(mar=(c(4,4,2,0.2)+0.3))
  lcols = c("red","green")
  if( grepl("a",option) ) {
    title <- "Amplicon Representation vs. Amplicon C/G Content"
    xaxisTitle <- "Amplicon C/G Base Content (%)"
    yaxisTitle <- "Number of Amplicons"
    legendTitle <- c("Fail (Assigned reads < 0.2x mean)","Pass")
  } else {
    title <- "Target Representation vs. Target C/G Content"
    xaxisTitle <- "Target C/G Base Content (%)"
    yaxisTitle <- "Number of Targets"
    legendTitle <- c("Fail (Base reads < 0.2x mean)","Pass")
  }
  bgc <- as.integer(pcgc+0.5)
  xmin <- min(bgc)
  xmax <- max(bgc)
  xdata <- xmin:xmax
  yall <- as.vector(table(factor(bgc,levels=xdata)))
  ylow  <- as.vector(table(factor(bgc[yprop < av02],levels=xdata)))
  ymax = max(yall)+1
  if( grepl("F",option) ) {
    ybad <- as.vector(table(factor(bgc[yprop == 0],levels=xdata)))
    yhigh <- as.vector(table(factor(bgc[yprop < 5*aver],levels=xdata)))
#    yunder <- as.vector(table(factor(bgc[yprop < 0.5*aver],levels=xdata)))
#    yover <- as.vector(table(factor(bgc[yprop < 2*aver],levels=xdata)))
#    legendTitle <- c("Fail/None (No Assigned Reads)", "Fail (AR < 0.2x mean)", "Pass/Low (0.2x <= AR < 0.5x mean)",
#      "Pass/Norm (0.5x <= AR < 2x mean)", "Pass/High (2x <= AR < 5x)", "Pass/Over (AR >= 5x mean)" )
#    lcols=rainbow(6)
#    barplot( t(cbind(ybad,ylow-ybad,yunder-ylow,yover-yunder,yhigh-yover,yall-yhigh)),
#      names.arg=xdata, beside=FALSE, ylim=c(0,ymax), space=0, xaxs='i', yaxs='i', col=lcols,
#      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
    legendTitle <- c("Drop Out (0 Assigned Reads)", "Fail (Assigned Reads < 0.2x mean)", "Pass", "Pass/Over (Assigned Reads >= 5x mean)" )
    lcols=c("red","orange","darkgreen","cyan")
    barplot( t(cbind(ybad,ylow-ybad,yhigh-ylow,yall-yhigh)),
      names.arg=xdata, beside=FALSE, ylim=c(0,ymax), space=0, xaxs='i', yaxs='i', col=lcols,
      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  } else {
    barplot( t(cbind(ylow,yall-ylow)), beside=FALSE, ylim=c(0,ymax), space=0,
      names.arg=xdata, xaxs='i', yaxs='i', col=lcols,
      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  }
  legend("topright", rev(legendTitle), cex=1.2, bty="n", fill=rev(lcols))
} 

# Create length vs. pass/fail plot
if( grepl("k",option,ignore.case=TRUE) ) {
  nFileOut <- sub(".xls$",".fedlen.png",nFileIn)
  png(nFileOut,width=picWidth,height=picHeight)
  par(mar=(c(4,4,2,0.2)+0.3))
  lcols = c("red","green")
  if( grepl("a",option) ) {
    title <- "Amplicon Representation vs. Amplicon Length"
    xaxisTitle <- "Amplicon Length (%)"
    yaxisTitle <- "Number of Amplicons"
    legendTitle <- c("Fail (Assigned reads < 0.2x mean)","Pass")
  } else {
    title <- "Target Representation vs. Target Length"
    xaxisTitle <- "Target Length (%)"
    yaxisTitle <- "Number of Targets"
    legendTitle <- c("Fail (Base reads < 0.2x mean)","Pass")
  }
  xmin <- min(tlen)
  xmax <- max(tlen)
  xdata <- xmin:xmax
  yall <- as.vector(table(factor(tlen,levels=xdata)))
  ylow  <- as.vector(table(factor(tlen[yprop < av02],levels=xdata)))
  ymax = max(yall)+1
  if( grepl("K",option) ) {
    ybad <- as.vector(table(factor(tlen[yprop == 0],levels=xdata)))
    yhigh <- as.vector(table(factor(tlen[yprop < 5*aver],levels=xdata)))
#    yunder <- as.vector(table(factor(tlen[yprop < 0.5*aver],levels=xdata)))
#    yover <- as.vector(table(factor(tlen[yprop < 2*aver],levels=xdata)))
#    legendTitle <- c("Fail/None (No Assigned Reads)", "Fail (AR < 0.2x mean)", "Pass/Low (0.2x <= AR < 0.5x mean)",
#      "Pass/Norm (0.5x <= AR < 2x mean)", "Pass/High (2x <= AR < 5x)", "Pass/Over (AR >= 5x mean)" )
#    lcols=rainbow(6)
#    barplot( t(cbind(ybad,ylow-ybad,yunder-ylow,yover-yunder,yhigh-yover,yall-yhigh)),
#      names.arg=xdata, beside=FALSE, ylim=c(0,ymax), space=0, xaxs='i', yaxs='i', col=lcols,
#      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
    legendTitle <- c("Drop Out (0 Assigned Reads)", "Fail (Assigned Reads < 0.2x mean)", "Pass", "Pass/Over (Assigned Reads >= 5x mean)" )
    lcols=c("red","orange","darkgreen","cyan")
    barplot( t(cbind(ybad,ylow-ybad,yhigh-ylow,yall-yhigh)),
      names.arg=xdata, beside=FALSE, ylim=c(0,ymax), space=0, xaxs='i', yaxs='i', col=lcols,
      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  } else {
    barplot( t(cbind(ylow,yall-ylow)), beside=FALSE, ylim=c(0,ymax), space=0,
      names.arg=xdata, xaxs='i', yaxs='i', col=lcols,
      xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
  }
  legend("topright", rev(legendTitle), cex=1.2, bty="n", fill=rev(lcols))
} 

q()

