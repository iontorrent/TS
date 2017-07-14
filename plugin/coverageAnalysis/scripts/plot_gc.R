# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# This script handles optionally creating different plots at the same time.
# A single script is used to avoid having to re-read the data and making similar transformations.
library(stringr)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"fine_coverage.xls",args[1])

# current options:
# "a" for amplicon reads (vs. base coverage)
# "f" or "F" output pass/fail (>=0.2 of Mean) counts vs. binned gc plot: "F" for 4-color plot
# "g" or "G" output cov vs. gc plot: "G" for log axis
# "k" or "K" output pass/fail (>=0.2 of Mean) counts vs. target length plot: "K" for 4-color plot
# "l" or "L" output cov vs. length plot: "L" for log axis
# "p" or "P" output mead reads vs. pool ID: "P" for log axis
option <- ifelse(is.na(args[2]),"",args[2])
nFileOutRoot <- ifelse(is.na(args[3]),nFileIn,args[3])

nFileOutRoot <- sub(".xls$","",nFileOutRoot)

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

# output size is reduced to fit pefectly into display widget
picWidth = 1.5*798
picHeight= 1.5*264

rcov <- read.table(nFileIn, header=TRUE, as.is=TRUE, sep="\t", comment.char="", quote="")

# test the type of the field in case the wrong type of targets are provided
if( grepl("a",option) ) {
  yprop <- rcov$total_reads
  if( is.null(yprop) ) yprop <- rcov$depth
  legoff = 0.89
} else {
  yprop <- rcov$depth
  if( is.null(yprop) ) yprop <- rcov$ave_basereads
  if( is.null(yprop) ) yprop <- rcov$total_reads
  legoff = 0.86
}
if( is.null(yprop) ) {
  write(sprintf("ERROR: Cannot locate fields 'depth', 'ave_basereads' or 'total_reads' in data file %s\n",nFileIn),stderr())
  q(status=1)
}

# test for property
ndata <- length(yprop)
if( ndata < 1 )
{
  write(sprintf("ERROR: No coverage property field found in data file %s\n",nFileIn),stderr())
  q(status=1)
}
if( ndata < 2 )
{
  write("WARNING: Skipping representation plot generation for only a single target region\n",stderr())
  q(status=0)
}

tlen <- rcov$contig_end - rcov$contig_srt + 1
pcgc <- 100 * rcov$gc / tlen
aver <- mean(yprop)
av02 <- 0.2 * aver

# test for presence of NVP attributes field
NVP <- rcov$attributes
if( is.null(NVP) ) NVP <- rcov$gene_id
haveNVP <- !is.null(NVP)

get_ymax <- function(ydata) {
  ymax = max(ydata)
  ykp = ymax
  if( ymax < 0 ) return(0)
  # exception for specific integer limits 
  if( ymax == as.integer(ymax) && ymax > 6 && ymax < 10 ) {
    return(ymax + (ymax %% 2))
  }
  blog = 10^(as.integer(log10(ymax+1)))
  sft = 1
  rng = ymax/blog
  if( rng < 1.0001 ) {
    return(ymax)  # ~ integer log10
  } else if( rng < 2 ) {
    sft = 0.2
  } else if( rng < 5 ) {
    sft = 0.5
  }
  blog = blog * sft
  ymax = blog*as.integer((ymax+blog)/blog)
  #write(sprintf("ymax %.4f (%.4f -> %.1f) -> %.2f",ykp,rng,sft,ymax),stderr())
  return(ymax)
}

# Create reads vs. GC scatter plot
if( grepl("g",option,ignore.case=TRUE) ) {
  nFileOut <- paste(nFileOutRoot,".gc.png",sep="")
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
  ymax = get_ymax(ydata)
  plot( pcgc, ydata, pch=4, xlab=xaxisTitle, ylab=yaxisTitle, main=title, ylim=c(0,ymax), cex.main=1.6, cex.lab=1.4 )
  abline(h=laver,col="green")
  abline(h=lav02,col="red")
  legend(legoff*max(pcgc), max(ymax), legend=c(legendTitle,"0.2 x Mean"), cex=1.2, bty="n", col=c("green","red"), lty=1)
}
  
# Create reads vs. length scatter plot
if( grepl("l",option,ignore.case=TRUE) ) {
  nFileOut <- paste(nFileOutRoot,".ln.png",sep="")
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
  ymax = get_ymax(ydata)
  plot( tlen, ydata, pch=4, xlab=xaxisTitle, ylab=yaxisTitle, main=title, ylim=c(0,ymax), cex.main=1.6, cex.lab=1.4 )
  abline(h=laver,col="green")
  abline(h=lav02,col="red")
  legend(legoff*max(tlen), max(ymax), legend=c(legendTitle,"0.2 x Mean"), cex=1.2, bty="n", col=c("green","red"), lty=1)
}
  
# Create pass/fail vs. GC plot
if( grepl("f",option,ignore.case=TRUE) ) {
  nFileOut <- paste(nFileOutRoot,".gc_rep.png",sep="")
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
  ymax = get_ymax(yall)
  #ymax = max(yall)+1
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

# Create pass/fail vs. length plot
if( grepl("k",option,ignore.case=TRUE) ) {
  nFileOut <- paste(nFileOutRoot,".ln_rep.png",sep="")
  png(nFileOut,width=picWidth,height=picHeight)
  par(mar=(c(4,4,2,0.2)+0.3))
  lcols = c("red","green")
  if( grepl("a",option) ) {
    title <- "Amplicon Representation vs. Amplicon Length"
    xaxisTitle <- "Amplicon Length"
    yaxisTitle <- "Number of Amplicons"
    legendTitle <- c("Fail (Assigned reads < 0.2x mean)","Pass")
  } else {
    title <- "Target Representation vs. Target Length"
    xaxisTitle <- "Target Length"
    yaxisTitle <- "Number of Targets"
    legendTitle <- c("Fail (Base reads < 0.2x mean)","Pass")
  }
  xmin <- min(tlen)
  xmax <- max(tlen)
  xdata <- xmin:xmax
  yall <- as.vector(table(factor(tlen,levels=xdata)))
  ylow  <- as.vector(table(factor(tlen[yprop < av02],levels=xdata)))
  ymax = get_ymax(yall)
  #ymax = max(yall)+1
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

# Create Pool representation plots
if( haveNVP && grepl("p",option,ignore.case=TRUE) ) {
  # extract pool information from NVP strings
  pools <- toupper(paste(NVP,";",sep=""))
  pools <- str_extract(pools,"POOL=.*?;")
  # only create plot if pooling info is there for at least one target
  if( sum(!is.na(pools)) > 0 ) {
    # replace no POOL key targets with '?' as pool
    pools[is.na(pools)] <- "?"
    # this assumes values of NVP's are merged using '&'
    pools <- sub("POOL=(.*?);","\\1",pools)
    # treat merged regions as also assigned to multiple pools (or even the same pools)
    pools <- gsub("&",",",pools)
    # add extra "," to list since R ignores last emply value in strsplit()
    spools <- paste(pools,",",sep="")
    spools <- strsplit(spools,",")
    # (again) replace empty values with '?'
    spools <- sapply(spools,function(x){x[x==""]<-"?";x})
    # assume targets belonging to multiple pools have coverage divided evenly between pools
    npools <- sapply(spools,length)
    cov <- yprop / npools
    # expand list and duplicate averaged values to create factors/values pairs
    pools <- unlist(spools)
    cov <- rep(cov,npools)
    # check if pools (factors) should be converted to numeric for better plot axis
    poolns <- suppressWarnings(as.numeric(pools))
    if( sum(is.na(poolns)) == 0 ) pools = poolns
    # get average target coverage vs. pool
    dt <- aggregate(cov~pools,data.frame(cov,pools),mean)
    # make plot of average cov vs. pool IF there is more than one Pool
    if( length(dt$pools) > 1 ) {
      xaxisTitle <- "Primer Pool"
      if( grepl("a",option) ) {
        title <- "Average Amplicon Reads by Primer Pool"
        yaxisTitle <- "Mean Reads per Amplicon"
      } else {
        title <- "Average Target Base Reads by Primer Pool"
        yaxisTitle <- "Mean Base Reads per Target"
      }
      if( grepl("P",option) ) {
        dt$cov <- log10(1+dt$cov)
        yaxisTitle <- sprintf("Log10(%s)", yaxisTitle)
      }
      nFileOut <- paste(nFileOutRoot,".pool.png",sep="")
      png(nFileOut,width=picWidth,height=picHeight)
      par(mar=(c(4,4,2,0.2)+0.3))
      ymax = get_ymax(dt$cov)
      barplot( dt$cov, names.arg=dt$pools, yaxs='i', ylim=c(0,ymax), 
        xlab=xaxisTitle, ylab=yaxisTitle, main=title, cex.main=1.6, cex.lab=1.4 )
    } else {
      write("- Coverage distribution by Primer Pool omitted for all targets in a single pool.\n",stderr())
    }
  } 
}
  
q()

