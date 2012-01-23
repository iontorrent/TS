findMixedReads <- function(
  dataDir,
  libKeySeq,
  mixResults,
  keySnrThreshold=3
) {
  if(.Platform$file.sep != rev(unlist(strsplit(dataDir,"")))[1])
    dataDir <- sprintf("%s%s",dataDir,.Platform$file.sep)

  bfMaskFile <- paste(dataDir,"bfmask.bin",sep="")
  wellFile   <- paste(dataDir,"1.wells",sep="")
  bf         <- readBeadFindMask(bfMaskFile)
  libIndex   <- which(bf$maskLib & !(bf$maskIgnore | bf$maskWashout))
  flowRange  <- 1:max(mixResults$mixFlows)
  cat(sprintf("Reading %d flows for %d wells from %s\n",max(flowRange),length(libIndex),wellFile))
  x          <- readWells(wellFile,col=bf$col[libIndex],row=bf$row[libIndex],flow=flowRange)
  keyStat    <- keyStats(x$signal,libKeySeq,x$flowOrder)
  goodSignal <- (keyStat$key_snr > keySnrThreshold)
  goodSignal[is.na(goodSignal)] <- FALSE

  xNorm  <- normalizeIonogram(x$signal,keySeq=libKeySeq,flowOrder=x$flowOrder)$normalized[goodSignal,]
  merEst <- matrix(NA,nrow=sum(goodSignal),ncol=length(mixResults$mixFlows))
  for(j in 1:length(mixResults$mixFlows)) {
    cat(sprintf("  Estimating mixture probabilities in flow %d\n",mixResults$mixFlows[j]))
    sig <- xNorm[,mixResults$mixFlows[j]]
    sigProb <- sigPosterior(sig,mixResults$sigDist)
    merEst[,j] <- apply(sigProb,2,which.max)
  }
  mixMask <- apply(merEst==2,1,sum)!=0

  return(list(
    col=x$col[goodSignal][mixMask],
    row=x$row[goodSignal][mixMask]
  ))
}

libMixtureAnalysis <- function(
  dataDir,
  libKeySeq,
  plotDir=NA,
  setName=NA,
  nSample=10000,
  densityBandwidth=0.05,
  keySnrThreshold=3,
  minLibNumber=5000,
  plotType=c("none","png","bitmap"),
  ret=TRUE
) {

  plotType <- match.arg(plotType)

  # Set random seed for reproducible sampling
  set.seed(0)

  # Add trailing file separators if not already present
  if(.Platform$file.sep != rev(unlist(strsplit(dataDir,"")))[1])
    dataDir <- sprintf("%s%s",dataDir,.Platform$file.sep)
  if(plotType != "none") {
    if(.Platform$file.sep != rev(unlist(strsplit(plotDir,"")))[1])
      plotDir <- sprintf("%s%s",plotDir,.Platform$file.sep)
  }

  # If setName wasn't specified use the base name of the data directory
  if(is.na(setName))
    setName <- basename(dataDir)
          
  # Load bead find mask, determine loading 
  bfMaskFile <- paste(dataDir,"bfmask.bin",sep="")
  bf         <- readBeadFindMask(bfMaskFile)
  pctLoaded  <- mean(bf$maskBead)
  pctLive    <- mean(bf$maskLive[bf$maskBead==1])
        
  # Read wells once to determine flow order and number of flows
  wellFile   <- paste(dataDir,"1.wells",sep="")
  cat(sprintf("Reading key flows from %s\n",wellFile))
  cat(sprintf("Loading is %0.1f%%\n",100*pctLoaded))
  cat(sprintf("Live:Loaded is %0.1f%%\n",100*pctLive))
  x <- readWells(wellFile,col=0,row=0)
  flowOrder <- x$flowOrder
  keyFlowLen <- length(seqToFlow(libKeySeq,paste(rep(flowOrder,nchar(libKeySeq)),collapse=""),finishAtSeqEnd=TRUE))
  mixEstimationFlows <- keyFlowLen+(1:4)
        
  if(x$nFlow < max(mixEstimationFlows)) {
    warning(sprintf("Too few flows to continue, need %d but found %d\n",max(mixEstimationFlows),x$nFlow))
    return(NULL)
  } else {
    # Determine the keypass beads for lib
    index <- list()
    temp <- findLibReads(bf,wellFile,keyFlowLen,libKeySeq,nSample=5*nSample,keySnrThreshold=keySnrThreshold)
    index$lib <- temp$index
    keyPassPct <- temp$keyPassPct
    cat(sprintf("KeyPass:Lib is %0.1f%%\n",100*keyPassPct))

    # Load flow data for library
    flowRange <- 1:max(mixEstimationFlows,keyFlowLen)
    keyPassNorm <- keyPass <- keyPassSample <- list()
    # Load all flows for the library wells
    aggregatedInfiniteMask <- numeric()
    if(length(index$lib) <= minLibNumber) {
      warning(sprintf("Too few library reads in %s, need %d but found %d\n",dataDir,minLibNumber,length(index$lib)))
      return(NULL)
    } else {
      thisSample <- min(nSample,length(index$lib))
      cat(sprintf("Loading %d lib beads\n",thisSample))
      keyPassSample$lib <- sample(index$lib,thisSample)
      keyPass$lib       <- readWells(wellFile,col=bf$col[keyPassSample$lib],row=bf$row[keyPassSample$lib],flow=flowRange)

      # Hack around a bug whereby some signal estimates are infinite
      libInfiniteMask <- apply(!is.finite(keyPass$lib$signal),1,any)
      aggregatedInfiniteMask <- c(aggregatedInfiniteMask,libInfiniteMask)
      if(any(libInfiniteMask)) {
        keyPassSample$lib <- keyPassSample$lib[!libInfiniteMask]
        keyPass$lib       <- readWells(wellFile,col=bf$col[keyPassSample$lib],row=bf$row[keyPassSample$lib],flow=flowRange)
      }
      keyPassNorm$lib   <- normalizeIonogram(keyPass$lib$signal,keySeq=libKeySeq,flowOrder=flowOrder)$normalized

      pctInfinite <- mean(aggregatedInfiniteMask)
      cat(sprintf("pctInfinite is %2.1f\n",100*pctInfinite))
    
      # Estimate N-mer signal distributions
      sigDistBase <- estimateNmerSignal(
        keyPass$lib$signal[,1:keyFlowLen],
        libKeySeq,
        x$flowOrder,
        densityBandwidth
      )
              
      # Fit N-mer signal distributions to data
      merEst <- matrix(NA,nrow=nrow(keyPassNorm$lib),ncol=length(mixEstimationFlows))
      merWeight <- merDens <- as.list(1:length(mixEstimationFlows))
      for(j in 1:length(mixEstimationFlows)) {
        sig <- keyPassNorm$lib[,mixEstimationFlows[j]]
        result <- sigFit(sig,sigDistBase,updateType="oneMer")
        #sigDist        <- result$sigDist
        #result <- sigFit(sig,sigDist,updateType="twoMer")
        merEst[,j]     <- result$merEst
        merWeight[[j]] <- result$merWeight
        merDens[[j]]   <- result$merDens
        sigDist        <- result$sigDist
      }
      clonalIndex <- apply(merEst==2,1,sum)==0
      pctClonal   <- mean(clonalIndex[!is.na(clonalIndex)])
      pctOutlier  <- mean(is.na(clonalIndex))
      clonalIndex[is.na(clonalIndex)] <- FALSE
      cat(sprintf("pctClonal is %0.1f%%\n",100*pctClonal))
      cat(sprintf("pctOutlier is %0.1f%%\n",100*pctOutlier))
    
      # Plots of the flows used to estimate mixing
      if(plotType != "none") {
        plotHelper(sprintf("%smixEstHist.png",plotDir),height=400,width=1200)
        op <- par(
          mfrow=c(1,4),
          omd=c(0.2,0.95,0.2,0.8),
          omi=c(0.5,0.2,0.3,0.2),
          mar=c(1,1,1,1),
          cex.axis=0.8
        )
        xlim <- c(0,3.5)
        ylim <- c(0,4)
        for(j in 1:length(mixEstimationFlows)) {
          plot(xlim,ylim,xlim=xlim,ylim=ylim,xlab="Normalized Signal",type="n")
          sig <- keyPassNorm$lib[,mixEstimationFlows[j]]
          sig <- sig[sig < xlim[2]]
          histFit <- hist(sig,breaks=seq(floor(min(sig)),ceiling(max(sig)),by=0.05),plot=FALSE)
          for(histBinCounter in which(histFit$breaks < xlim[2])) {
            polygon(
              histFit$breaks[histBinCounter+c(0,1,1,0)],
              rep(c(0,histFit$density[histBinCounter]),c(2,2)),
              col="darkgrey",border="darkgrey"
            )
          }
          title(sprintf("flow %d, %1.1f%% mixed",mixEstimationFlows[j],100*merWeight[[j]][2]),cex.main=1.3)
          lines(sigDist$x, merDens[[j]],  col="blue",lwd=2)
          legendText <- c(
            sprintf("Fit to Lib",nrow(keyPassNorm$lib)),
            sprintf("%6d Lib Reads",nrow(keyPassNorm$lib))
          )
          legendCol  <- c(
            "blue",
            "darkgrey"
          )
          if(j==length(mixEstimationFlows))
            legend("topright",legendText,lwd=2,col=legendCol,bty="n")
        }
        mtext(sprintf("%s, key-norm signal density, estimated %0.1f%% clonal",setName,100*pctClonal),outer=TRUE,line=0.5,cex=1.4)
        plot_version(line=2.5)
        dev.off()
      }
    
      if(ret) {
        return(list(
          pctLoaded     = pctLoaded,
          pctLive       = pctLive,
          keyPassLib    = keyPassPct,
          pctClonal     = pctClonal,
          pctOutlier    = pctOutlier,
          pctInfinite   = pctInfinite,
          mixFlows      = mixEstimationFlows,
          sigDist       = sigDist
        ))
      }
    }
  }
}


estimateNmerSignal <- function(sig,keySeq,flowOrder,densityBandwidth,xlim=c(0,4),nX=1024) {

  # Determine key 0mers and 1mers
  mer <- seqToFlow(keySeq,flowOrder,finishAtSeqEnd=TRUE)
  zeroMerMask <- mer==0
  oneMerMask  <- mer==1
  if(any(oneMerMask))
    oneMerMask[max(which(oneMerMask))] <- FALSE

  if(!any(zeroMerMask)) {
    warning("No zero-mers in the key flows, cannot do empirical estimation of distribution\n")
    return(NULL)
  } else if(!any(oneMerMask)) {
    warning("No one-mers in the key flows, cannot do empirical estimation of distribution\n")
    return(NULL)
  } else {
    scale.zero <- matrix(rep(apply(sig[,oneMerMask],1,mean),sum(zeroMerMask)),ncol=sum(zeroMerMask))
    scale.zero <- pmax(scale.zero,1e-6)
    dens.zero  <- density(sig[,zeroMerMask]/scale.zero,bw=densityBandwidth,from=xlim[1],to=xlim[2],n=nX)
    xVals <- dens.zero$x

    scale.one <- matrix(NA,nrow=nrow(sig),ncol=sum(oneMerMask))
    for(i in 1:sum(oneMerMask))
      scale.one[,i] <- apply(sig[,which(oneMerMask)[-i]],1,mean)
    one.scaled <- sig[,oneMerMask]/scale.one
    dens.one  <- density(one.scaled,bw=densityBandwidth,from=xlim[1],to=xlim[2],n=nX)
    dens.one$y[1] <- 0
    dens.one$y[nX] <- 0
    center.one <- median(c(one.scaled))

    p.mixMer <- approx(dens.one$x-(0.4*center.one),dens.one$y,xVals,rule=2)$y
    p.2Mer   <- approx(dens.one$x+(1*center.one),dens.one$y,xVals,rule=2)$y
    p.3Mer   <- approx(dens.one$x+(2*center.one),dens.one$y,xVals,rule=2)$y
    return(list(
      x = xVals,
      y0 = dens.zero$y,
      yM = p.mixMer,
      y1 = dens.one$y,
      y2 = p.2Mer,
      y3 = p.3Mer
    ))
  }
}



sigPosterior <- function(sig,sigDist,nullProb=0.02) {
  p0 <- approx(sigDist$x,sigDist$y0,sig,rule=2)$y
  pM <- approx(sigDist$x,sigDist$yM,sig,rule=2)$y
  p1 <- approx(sigDist$x,sigDist$y1,sig,rule=2)$y
  p2 <- approx(sigDist$x,sigDist$y2,sig,rule=2)$y
  p3 <- approx(sigDist$x,sigDist$y3,sig,rule=2)$y
  pN <- rep(nullProb,length(sig))
  ret <- rbind(p0,pM,p1,p2,p3,pN)
  rownames(ret) <- c("p0","pM","p1","p2","p3","pN")
  retScale <- matrix(rep(apply(ret,2,sum),6),nrow=6,byrow=TRUE)
  return(ret/retScale)
}

sigFit <- function(sig,sigDist,newPlot=FALSE,doPlot=FALSE,maxEmIterations=10,updateType=c("weight","oneMer","twoMer")) {
  updateType <- match.arg(updateType)
  xlim <- c(0,4)
  ylim <- c(0,4)
  outlierMask <- sig>xlim[2]
  sig <- sig[!outlierMask]
  if(doPlot & newPlot) {
    plot(xlim,ylim,xlim=xlim,ylim=ylim,xlab="Normalized Signal",type="n")
    histFit <- hist(sig,breaks=seq(floor(min(sig)),ceiling(max(sig)),by=0.05),plot=FALSE)
    for(histBinCounter in which(histFit$breaks < xlim[2])) {
      polygon(
        histFit$breaks[histBinCounter+c(0,1,1,0)],
        rep(c(0,histFit$density[histBinCounter]),c(2,2)),
        col="darkgrey",border="darkgrey"
      )
    }
  }
  merWeight    <- rep(0.2,5)
  oneMerMean   <- sum(sigDist$x*sigDist$y1)/sum(sigDist$y1)
  twoMerMean   <- sum(sigDist$x*sigDist$y2)/sum(sigDist$y2)
  for(emCounter in 1:maxEmIterations) {
    sigProb <- sigPosterior(sig,sigDist)

    merEst    <- apply(sigProb,2,which.max)
    newMerWeight <- rep(0,nrow(sigProb))
    names(newMerWeight) <- as.character(1:nrow(sigProb))
    merTable <- table(merEst)
    newMerWeight[names(merTable)] <- merTable
    newMerWeight <- newMerWeight[1:5]/sum(newMerWeight[1:5])

    if(updateType=="oneMer") {
      newOneMerMean   <- sum(sigProb[3,] * sig)/sum(sigProb[3,])
      newTwoMerMean   <- 2*newOneMerMean
      sigDist$yM <- setNewMean(sigDist$x,sigDist$yM,newOneMerMean*0.6)
      sigDist$y1 <- setNewMean(sigDist$x,sigDist$y1,newOneMerMean)
      sigDist$y2 <- setNewMean(sigDist$x,sigDist$y2,newOneMerMean*2)
      sigDist$y3 <- setNewMean(sigDist$x,sigDist$y3,newOneMerMean*3)
    } else if(updateType=="twoMer") {
      newOneMerMean   <- sum(sigProb[3,] * sig)/sum(sigProb[3,])
      newTwoMerMean   <- sum(sigProb[4,] * sig)/sum(sigProb[4,])
      sigDist$y2 <- setNewMean(sigDist$x,sigDist$y2,newTwoMerMean)
      sigDist$y3 <- setNewMean(sigDist$x,sigDist$y3,newOneMerMean + 2*(newTwoMerMean-newOneMerMean))
    }
  
    if(doPlot) {
      merDens <- sigDist$y0 * newMerWeight[1] + sigDist$yM * newMerWeight[2] + sigDist$y1 * newMerWeight[3] + sigDist$y2 * newMerWeight[4] + sigDist$y3 * newMerWeight[5]
      lines(sigDist$x, merDens)
    }

    if(all(abs(newMerWeight - merWeight) < 0.005) & all(abs(newOneMerMean - oneMerMean) < 0.005) & all(abs(newTwoMerMean - twoMerMean) < 0.005)) {
      break
    } else {
      merWeight    <- newMerWeight
      oneMerMean   <- newOneMerMean
      twoMerMean   <- newTwoMerMean
    }
  }

  merDens <- sigDist$y0 * merWeight[1] + sigDist$yM * merWeight[2] + sigDist$y1 * merWeight[3] + sigDist$y2 * merWeight[4] + sigDist$y3 * merWeight[5]
  returnedMerEst <- rep(NA,length(outlierMask)) 
  returnedMerEst[!outlierMask] <- merEst
  return(list(
    merEst    = returnedMerEst,
    merWeight = newMerWeight,
    merDens   = merDens,
    sigDist   = sigDist
  ))
}
  
setNewMean <- function(x,y,mu) {
  xStep <- x[2]-x[1]
  nX <- length(x)
  distSum <- sum(y)
  curMu <- sum(x*y)/distSum
  nShift <- round((curMu-mu)/xStep)
  if(nShift > 0)
    y <- c(y[-(1:nShift)],rep(0,nShift))
  else if(nShift < 0)
    y <- c(rep(0,-nShift),y[-((nX+nShift+1):nX)])
  y <- y + (distSum-sum(y))/length(y)
  return(y)
}

