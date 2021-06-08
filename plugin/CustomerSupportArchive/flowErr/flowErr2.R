library(rhdf5,lib.loc=libDir)
library(torrentR)
library(rjson)

system(sprintf("mkdir -p %s",plotDir))

depthThreshold   <- 0.005
sdThreshold <- 4
flowErrThreshold <- 5
baseErrThreshold <- 3
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
flowCycleLength <- nchar(floworder)

x <- h5dump(errorSummaryH5,bit64conversion='double')

if(!is.null(x$per_read_group)) {
  prefix <- strsplit(names(x$per_read_group)[1],".",fixed=TRUE)[[1]][1]
  nomatchIndex <- names(x$per_read_group)==sprintf("%s.nomatch",prefix)
  x$all_read_group = combineGroups(x$per_read_group,which(!nomatchIndex))
  LogErrors=errSummary(x$all_read_group$per_flow$error_data)
  if(unfiltered==1) {
    LogErrors0 <- fromJSON(paste(readLines("results.json"),collapse=""))
    system("rm results.json")
    if(untrimmed==1) {
      LogErrors1 <- list("unfiltered_untrimmed"=LogErrors)
      LogErrors2 <- c(LogErrors0,LogErrors1)  
      write.summary.to.json(LogErrors2)
    } else {
      LogErrors1 <- list("unfiltered_trimmed"=LogErrors)
      LogErrors2 <- c(LogErrors0,LogErrors1) 
      write.summary.to.json(LogErrors2)
    }
  } else {
    write.summary.to.json(LogErrors)
  }

  if((!is.null(x$all_read_group$per_flow)) && (!all(x$all_read_group$per_flow$error_data==0))) {
    nFlow <- nrow(x$all_read_group$per_flow$error_data)
    flowBase <- strsplit(paste(rep(floworder,ceiling(nFlow/nchar(floworder))),collapse=""),"")[[1]]

    for(type in c("err","ins","del","sub")) {
      png(plotFile <- sprintf("%s/%s.per_flow.%s.png",plotDir,analysisName,type),width=400,height=400)
      par(mai=c(1,0.8,0.8,1))
      flowLim <- errVsFlow(x$all_read_group$per_flow$error_data,type,depthThreshold,sdThreshold,flowBase,flowErrThreshold,flowCycleLength,denominatorType="depth_span")
      title(sub=sprintf("Denominator = read depth"))
      dev.off()
      #system(sprintf("eog %s",plotFile))
    }

    for(type in c("err","ins","del","sub")) {
      png(plotFile <- sprintf("%s/%s.per_incorporating_flow.%s.png",plotDir,analysisName,type),width=400,height=400)
      par(mai=c(1,0.8,0.8,1))
      flowLim <- errVsFlow(x$all_read_group$per_flow$error_data,type,depthThreshold,sdThreshold,flowBase,flowErrThreshold,flowCycleLength,denominatorType="depth")
      title(sub=sprintf("Denominator uses only _incorporating_ flows"))
      dev.off()
      #system(sprintf("eog %s",plotFile))
    }
  }

  if(!is.null(x$all_read_group$per_base)) {
    png(plotFile <- sprintf("%s/%s.per_base.png",plotDir,analysisName),width=400,height=400)
    par(mai=c(1,0.8,0.8,1))
    totalErr <- errVsBase(x$all_read_group$per_base$error_data,depthThreshold,baseErrThreshold)
    dev.off()
    #system(sprintf("eog %s",plotFile))
  }

  if(!is.null(x$all_read_group$per_hp)) {
    png(plotFile <- sprintf("%s/%s.per_hp.err_distribution.png",plotDir,analysisName),width=400,height=400)
    par(mai=c(1,0.8,0.8,1.3))
    plotErrDistribution(x$all_read_group$per_hp,analysisName)
    dev.off()
    #system(sprintf("eog %s",plotFile))
    for(errorType in c("over-under","total")) {
      png(plotFile <- sprintf("%s/%s.per_hp.%s.error.png",plotDir,analysisName,errorType),width=400,height=400)
      par(mai=c(1,0.8,0.8,1.3))
      plotHpErr(x$all_read_group$per_hp,analysisName,errorType=errorType)
      dev.off()
      #system(sprintf("eog %s",plotFile))
      png(plotFile <- sprintf("%s/%s.per_hp.%s.phred.png",plotDir,analysisName,errorType),width=400,height=400)
      par(mai=c(1,0.8,0.8,1.3))
      plotHpErr(x$all_read_group$per_hp,analysisName,phredScale=TRUE,errorType=errorType)
      dev.off()
      #system(sprintf("eog %s",plotFile))
    }
  }
}

if(!is.null(x$per_region)) {
  # Trim off any entirely-blank regions to the top or right of the spatial map
  # This is useful if the chip dimensions that were applied were too large for the actual chip used
  r <- getRegionCenters(x$per_region)
  nonEmptyMask <- unlist(lapply(x$per_region,function(z){sum(z$hp_count)>0}))
  lastX <- length(unique(r$xBlock)) + 1 - which(rev(unlist(lapply(split(nonEmptyMask,r$xBlock),any))))[1]
  lastY <- length(unique(r$yBlock)) + 1 - which(rev(unlist(lapply(split(nonEmptyMask,r$yBlock),any))))[1]
  toRetain <- (r$xBlock <= lastX) & (r$yBlock <= lastY)
  per_region <- x$per_region[toRetain]

  d1 <- median(unlist(lapply(per_region,function(z){z$region_dim[1]})))
  d2 <- median(unlist(lapply(per_region,function(z){z$region_dim[2]})))
  dimString <- sprintf("%dx%d",d1,d2)
  r <- getRegionCenters(per_region)
  depth <- matrix(unlist(lapply(per_region,function(z){z$n_aligned}))[r$order],ncol=r$nRow)
  depth[depth==0] <- NA
  err   <- 100*matrix(unlist(lapply(per_region,function(z){z$n_err    }))[r$order],ncol=r$nRow)/depth
  depth <- 1e-6*depth
  
  png(plotFile <- sprintf("%s/%s.per_base.regional.aligned.dynamic.png",plotDir,analysisName),width=400,height=550)
  par(mai=c(1,0.8,0.8,1))
  depthLimLower <- quantile(depth,prob=0.005,na.rm=TRUE)
  depthLimUpper <- quantile(depth,prob=0.995,na.rm=TRUE)
  imageWithHist(depth,zlim=c(depthLimLower,depthLimUpper),header=analysisName,col=jet.colors(256))
  title(sprintf("Aligned Mb per %s chunk (%d Mb overall)",dimString,round(sum(depth,na.rm=TRUE),0)))
  dev.off()
  #system(sprintf("eog %s",plotFile))
  
  png(plotFile <- sprintf("%s/%s.per_base.regional.error.fixed.png",plotDir,analysisName),width=400,height=550)
  par(mai=c(1,0.8,0.8,1))
  imageWithHist(err,zlim=c(0,baseErrThreshold),header=analysisName,col=jet.colors(256))
  title(sprintf("Error Rate in %s chunks (%1.3f%% overall)",dimString,round(totalErr,3)))
  dev.off()
  #system(sprintf("eog %s",plotFile))
  
  png(plotFile <- sprintf("%s/%s.per_base.regional.error.dynamic.png",plotDir,analysisName),width=400,height=550)
  par(mai=c(1,0.8,0.8,1))
  errLimLower <- quantile(err,prob=0.005,na.rm=TRUE)
  errLimUpper <- max(quantile(err,prob=0.995,na.rm=TRUE),baseErrThreshold)
  imageWithHist(err,zlim=c(errLimLower,errLimUpper),header=analysisName,col=jet.colors(256))
  title(sprintf("Error Rate in %s chunks (%1.3f%% overall)",dimString,round(totalErr,3)))
  dev.off()
  #system(sprintf("eog %s",plotFile))

  # plot of cumulative depth vs cumulative error rate, orderd by regional accuracy
  depth <- matrix(unlist(lapply(per_region,function(z){z$n_aligned}))[r$order],ncol=r$nRow)
  err   <- matrix(unlist(lapply(per_region,function(z){z$n_err    }))[r$order],ncol=r$nRow)
  keepMask <- depth > 0
  depth <- depth[keepMask]
  err   <- err[keepMask]
  myOrder <- order(err/depth)
  err   <- err[myOrder]
  depth <- depth[myOrder]
  throughputDivisor <- 1e6
  throughputUnit <- "Mb"
  depth <- depth/throughputDivisor
  err <- err/throughputDivisor
  lowerPercentileIndex <- sum(cumsum(depth) < 0.01*sum(depth))
  if(lowerPercentileIndex < 1){
    lowerPercentileIndex <- 1
  }
  ylim <- (cumsum(err)/cumsum(depth))[c(lowerPercentileIndex,length(depth))]*100
  png(plotFile <- sprintf("%s/%s.spatial_filter.png",plotDir,analysisName),width=400,height=450)
  plot(cumsum(depth),cumsum(err)/cumsum(depth)*100,xlab=sprintf("Throughput (%s)",throughputUnit),ylab="Error Rate (%)",las=3,type="l",ylim=ylim,lwd=3)
  points(sum(depth),sum(err)/sum(depth)*100,pch=21,bg="red",cex=2)
  title(sprintf("Impact of spatial error rate filtering\nFull chip has %1.0f%s with %1.2f error rate",round(sum(depth)),throughputUnit,round(sum(err)/sum(depth)*100,2)))
  grid(lwd=2)
  dev.off()
  #system(sprintf("eog %s",plotFile))

  hp_err_allflow   <- lapply(per_region,function(z){apply(z$hp_err,  1,sum)})[r$order]
  hp_count_allflow <- lapply(per_region,function(z){apply(z$hp_count,1,sum)})[r$order]
  for(k in 1:length(hp_err_allflow[[1]])) {
    thisErr <- unlist(lapply(hp_err_allflow,function(z){z[k]}))
    thisCount <- unlist(lapply(hp_count_allflow,function(z){z[k]}))
    err <- 100*matrix(thisErr/thisCount,ncol=r$nRow)
    png(plotFile <- sprintf("%s/%s.%dmer_error.regional.png",plotDir,analysisName,k-1),width=400,height=550)
    par(mai=c(1,0.8,0.8,1))
    errLimLower <- quantile(err,prob=0.005,na.rm=TRUE)
    errLimUpper <- quantile(err,prob=0.995,na.rm=TRUE)
    imageWithHist(err,zlim=c(errLimLower,errLimUpper),header=analysisName,col=jet.colors(256))
    title(sprintf("%dmer error rate (%1.3f %% overall)",k-1,round(100*sum(as.numeric(thisErr))/sum(as.numeric(thisCount)),3)))
    dev.off()
    #system(sprintf("eog %s",plotFile))
  }

  hp_err <- lapply(per_region,function(z){z$hp_err/z$hp_count    })[r$order]
  for(k in 1:nrow(hp_err[[1]])) {
    err_kmer <- 100*array(unlist(lapply(hp_err,function(z){z[k,]})),c(ncol(hp_err[[1]]),r$nCol,r$nRow))
    if (k<4){
     LogErrors$regionalhp[[k]]<-err_kmer;
    }
    errLimLower <- quantile(err_kmer,prob=0.005,na.rm=TRUE)
    errLimUpper <- max(quantile(c(apply(err_kmer,c(2,3),quantile,prob=0.98,na.rm=TRUE)),prob=0.99,na.rm=TRUE),baseErrThreshold)
    flowRange <- which(apply(err_kmer,1,function(z){sum(!is.na(z))})>0)
    if(length(flowRange) > 0) {
      for(flow in flowRange[flowRange <= max(flowLim)]) {
        png(plotFile <- sprintf("%s/%s.per_flow.flow%04d.regional.%dmer.png",plotDir,analysisName,flow,k-1),width=400,height=550)
        par(mai=c(1,0.8,0.8,1))
        imageWithHist(err_kmer[flow,,],zlim=c(errLimLower,errLimUpper),header=analysisName,col=jet.colors(256))
        title(sprintf("%dmer Error Rate (%%) in %s chunks (flow %04d)%s",k-1,dimString,flow,flowBase[flow]))
        dev.off()
      }
      system(sprintf("convert -delay 10 -loop 100 %s/%s.per_flow.flow*.regional.%dmer.png %s/%s.per_flow.movie.regional.%dmer.gif",plotDir,analysisName,k-1,plotDir,analysisName,k-1))
      system(sprintf("rm -rf %s/%s.per_flow.flow*.regional.%dmer.png",plotDir,analysisName,k-1))
    }
  }

  # Plot of kmer error rate vs flow
  nHP   <- nrow(per_region[[1]]$hp_count)
  nFlow <- ncol(per_region[[1]]$hp_count)
  numerator   <- matrix(apply(matrix(unlist(lapply(per_region,function(z){z$hp_err})),  nrow=nHP*nFlow),1,sum),nrow=nHP)
  denominator <- matrix(apply(matrix(unlist(lapply(per_region,function(z){z$hp_count})),nrow=nHP*nFlow),1,sum),nrow=nHP)
  for(k in 1:nrow(numerator)) {
    png(plotFile <- sprintf("%s/%s.per_flow.%dmer_err.png",plotDir,analysisName,k-1),width=400,height=400)
    par(mai=c(1,0.8,0.8,1))
    errVsFlow(cbind(numerator[k,],denominator[k,]),"err",depthThreshold,sdThreshold,flowBase,flowErrThreshold,flowCycleLength)
    title(sub=sprintf("%dmer Error Rate vs Flow",k-1))
    dev.off()
    #system(sprintf("eog %s",plotFile))
  }
  h5save(LogErrors,file="results.h5")
}
