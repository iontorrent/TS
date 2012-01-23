plotIonogram <- function(
  signal,
  flowOrder,
  flow,
  plotType=c("raw","norm","cafie"),
  cf=NA,
  ie=NA,
  dr=NA,
  keySeq=NA,
  trueSeq=NA,
  headerBase="",
  ...
) {
  plotType <- match.arg(plotType)

  if(plotType=="raw") {
    header <- paste(headerBase,"Raw Signal",sep=" ")
    plotIonogramBase(signal,flowOrder,flow,header=header,...)
  } else if(plotType=="norm") {
    if(is.na(keySeq))
      stop("keySeq must be specified for normalization to happen\n")
    header <- paste(headerBase,"Normalized Signal",sep=" ")
    normalized <- normalizeIonogram(signal,keySeq,flowOrder)$normalized
    plotIonogramBase(normalized,flowOrder,flow,header=header,...)
  } else {
    if(is.na(trueSeq) | is.na(keySeq))
      stop("Both trueSeq and keySeq must be specified for CAFIE to work\n")
    if(is.na(cf) || is.na(ie) || is.na(dr)) {
      param <- findBestCafie(signal,flowOrder,trueSeq,keySeq)
      if(is.na(cf))
        cf <- param$carryForward
      if(is.na(ie))
        ie <- param$incompleteExtension
      if(is.na(dr))
        dr <- param$droop
    }
    result <- correctCafie(signal,flowOrder,keySeq,cf,ie,dr)
    paramHeader <- sprintf("(cf,ie,dr) = (%1.4f,%1.4f,%1.4f)",cf,ie,dr)
    header <- paste(headerBase,"Corrected Signal;",paramHeader,sep=" ")
    plotIonogramBase(result$corrected,flowOrder,flow,header=header,...)
  }
}

plotIonogramBase <- function(
  signal,
  flowOrder,
  flow,
  flowRange=NA,
  signalLim=NA,
  flowsPerWindow=50,
  header="",
  ...
) {
  # If signal is a vector, convert to a matrix
  if(is.null(dim(signal)))
    signal <- matrix(signal,1,length(signal))
  nFlow <- ncol(signal)

  # Cycle flowOrder if necessary
  if(nchar(flowOrder) < nFlow)
    flowOrder <- substring(paste(rep(flowOrder,ceiling(nFlow/nchar(flowOrder))),collapse=""),1,nFlow)

  nucToNum <- c(1:4,1:4)
  names(nucToNum) <- c("A","C","G","T","a","c","g","t")
  nucToNum <- nucToNum[unlist(strsplit(flowOrder,""))]
  nucColorSet <- c("green","blue","black","red")
  nucColors <- nucColorSet[nucToNum]

  # Restrict to the requested flow range
  if(any(is.na(flowRange))) {
    flowRange <- range(flow)
    flowMask  <- rep(TRUE,length(flow))
  } else {
    flowMask <- is.element(flowRange,flow)
    if(!any(flowMask))
      stop("Requested flowRange included none of the available flows")
    flow <- flowRange[flowMask]
    flowRange <- range(flow)
  }
  signal <- signal[,flowMask]
  nucColors <- nucColors[flow]
  nucToNum <- nucToNum[flow]

  # If signal is a vector, convert to a matrix
  if(is.null(dim(signal)))
    signal <- matrix(signal,1,length(signal))

  # Determine the yRange for plots
  if(any(is.na(signalLim)))
    signalLim <- range(c(signal,1))

  n <- max(flowRange)
  nWindows <- ceiling(n/flowsPerWindow)
  op <- par(
    mfrow=c(nWindows,1),
    omd=c(0.1,0.95,0.1,0.8),
    omi=c(0.2,0.2,0.2,0.2),
    mar=c(1,1,1,1),
    cex.axis=0.8
  )
  flowMin <- min(flowRange)
  nWells <- nrow(signal)
  for(i in 0:(nWindows-1)) {
    index <- i*flowsPerWindow + 1:flowsPerWindow
    plot(range(flowMin-1+index),signalLim,type="n",xlab="",ylab="",...)
    if(i==0)
      legend("topleft",c("A","C","G","T"),pch=21:24,pt.bg=nucColorSet,cex=1.4,bty="n")
    index <- which(is.element(flow,index))
    if(nWells==1) {
      for(j in index)
        lines(rep(flow[j],2),c(0,signal[1,j]),lwd=2,col=nucColors[j])
      points(flow[index],signal[1,index],pch=20+nucToNum[index],bg=nucColors[index])
    } else {
      for(j in 1:nWells)
        lines(flow[index],signal[j,index],lty=2,lwd=0.5,col="darkgrey")
      signalSummary <- apply(signal[,index],2,quantile,probs=c(0.10,0.50,0.90))
      lines(flow[index],signalSummary[2,],lty=1,lwd=3,pch="*",type="b")
      points(flow[index],signalSummary[2,],pch=20+nucToNum[index],bg=nucColors[index])
      lines(flow[index],signalSummary[1,],lty=2,col="red")
      lines(flow[index],signalSummary[3,],lty=2,col="red")
    }
  }
  mtext(header,outer=TRUE,line=-0.3)
}
