write.summary.to.json<-function(summary){
jsonResultsFile<-"results.json"
write(toJSON(summary),file=jsonResultsFile)
}


errSummary <- function(z) {
  z <- t(z)
  if(nrow(z)==6 || nrow(z)==7) {
    if(nrow(z)==6)
      rownames(z) <- c("ins","del","sub","align_start","align_stop","depth")
    else
      rownames(z) <- c("ins","del","sub","no_call","align_start","align_stop","depth")
    err <- apply(z[c("ins","del","sub"),],2,sum)
    z1 <- cumsum(c(z["align_start",],0))
    z2 <- cumsum(c(0,z["align_stop",]))
    depth_span <- (z1-z2)[-length(z1)]
    return(list(
      ins=z["ins",],
      del=z["del",],
      sub=z["sub",],
      align_start=z["align_start",],
      align_stop=z["align_stop",],
      err=err,
      depth=z["depth",],
      depth_span=depth_span
    ))
  } else if(nrow(z)==2) {
    return(list(
      err=z[1,],
      depth=z[2,]
    ))
  } else {
    stop("Unexpected input format")
  }
}

errPlot <- function(x,plotType=c("accuracy","err","ins","del","sub"),denominatorType=c("depth_span","depth"),addDepth=FALSE,newPlot=TRUE,doPlot=TRUE,...) {
  plotType <- match.arg(plotType)
  denominatorType <- match.arg(denominatorType)
  z <- errSummary(x)
  if(denominatorType=="depth_span" && !is.null(z$depth_span))
    denominator <- z$depth_span
  else
    denominator <- z$depth
  xlim <- c(1,length(z$err))
  if(plotType == "accuracy") {
    yval <- 100-100*z$err/denominator
    ylab <- "Accuracy (%)"
  } else if(plotType == "err") {
    yval <- 100*z$err/denominator
    ylab <- "Error Rate (%)"
  } else if(plotType == "ins") {
    yval <- 100*z$ins/denominator
    ylab <- "Insertion Rate (%)"
  } else if(plotType == "del") {
    yval <- 100*z$del/denominator
    ylab <- "Deletion Rate (%)"
  } else if(plotType == "sub") {
    yval <- 100*z$sub/denominator
    ylab <- "Substitution Rate (%)"
  }
  ylim <- range(yval,na.rm=TRUE)
  if(doPlot & newPlot) {
    plot(xlim,ylim,xlab="position",ylab="Accuracy (%)",type="n",las=1)
    grid()
  }
  if(addDepth) {
    yaxp <- par("yaxp")
    yaxp.lim <- yaxp[1:2]
    yaxp.vals <- seq(yaxp[1],yaxp[2],length=1+yaxp[3])
    if(is.null(z$depth_span))
      depth_range <- range(z$depth)
    else
      depth_range <- range(c(z$depth,z$depth_span))
    yaxp.labels <- as.character(round(seq(min(depth_range),max(depth_range),length=length(yaxp.vals))/1e3))
    axis(4,at=yaxp.vals,lab=yaxp.labels,las=3,col.axis="grey")
    lines(1:length(z$depth),yaxp.lim[1] + (yaxp.lim[2]-yaxp.lim[1])*(z$depth     /max(depth_range)),col="grey",lwd=1)
    if(!is.null(z$depth_span))
      lines(1:length(z$depth),yaxp.lim[1] + (yaxp.lim[2]-yaxp.lim[1])*(z$depth_span/max(depth_range)),col="grey",lwd=1)
    mtext("Coverage Depth (1000's)",side=4,col="grey",line=3)
  }
  if(doPlot)
    lines(1:length(yval),yval,...)
  return(list(yval=yval,ylab=ylab,depth=denominator))
}

combineGroups <- function(z,toCombine) {
  if(length(toCombine)==0) {
    stop("Nothing to combine")
  } else if(length(toCombine)==1) {
    return(z[[toCombine]])
  } else {
    ret <- z[[toCombine[1]]]
    for(i in 2:length(toCombine)) {
      if(!is.null(ret$per_hp)) {
        ret$per_hp$A <- ret$per_hp$A + z[[toCombine[i]]]$per_hp$A
        ret$per_hp$C <- ret$per_hp$C + z[[toCombine[i]]]$per_hp$C
        ret$per_hp$G <- ret$per_hp$G + z[[toCombine[i]]]$per_hp$G
        ret$per_hp$T <- ret$per_hp$T + z[[toCombine[i]]]$per_hp$T
      }
      if(!is.null(ret$per_flow))
        ret$per_flow$error_data <- ret$per_flow$error_data +z[[toCombine[i]]]$per_flow$error_data
      if(!is.null(ret$per_base))
        ret$per_base$error_data <- ret$per_base$error_data +z[[toCombine[i]]]$per_base$error_data
    }
    return(ret)
  }
}

errVsFlow <- function(data,type,depthThreshold,sdThreshold,flowBase,errThreshold,flowCycleLength=0,denominatorType=c("depth_span","depth")) {
  denominatorType <- match.arg(denominatorType)

  nFlow <- nrow(data)
  flowBase <- rep(flowBase,ceiling(nFlow/length(flowBase)))[1:nFlow]
  flowDelay <- rep(0,nFlow)
  for(nuc in unique(flowBase))
    flowDelay[which(flowBase==nuc)[-1]] <- diff(which(flowBase==nuc))
  flowDelayPch <- rep(24,nFlow)
  flowDelayPch[flowDelay < 7] <- 23
  flowDelayPch[flowDelay < 6] <- 22
  flowDelayPch[flowDelay < 4] <- 21
  flowDelayPch[flowDelay < 3] <- 25
  
  myCol <- c("green","blue","black","red")
  names(myCol) <- c("A","C","G","T")
  result <- errPlot(data,plotType=type,doPlot=FALSE,denominatorType=denominatorType)
  dataFlows <- result$depth/max(result$depth)>depthThreshold
  errLim <- range(c(0,errThreshold,result$yval[dataFlows]),na.rm=TRUE)
  flowLim <- range(which(dataFlows))
  plot(flowLim,errLim,xlab="Flow",ylab=result$ylab,type="n",las=1)
  result <- errPlot(data,plotType=type,newPlot=FALSE,addDepth=TRUE,pch=flowDelayPch,bg=myCol[flowBase],type="p",denominatorType=denominatorType)
  grid()

  # Determine outliers, set outlier limit dependent on position in flow cycle
  iqr <- function(x) {quantile(x,c(0.75,0.25),na.rm=TRUE) %*% c(1,-1)}
  if(flowCycleLength==0) {
    yLimit <- c(median(result$yval[dataFlows]) + sdThreshold * iqr(result$yval[dataFlows]))
  } else {
    temp <- split(result$yval[dataFlows],rep(1:flowCycleLength,ceiling(length(result$yval)/flowCycleLength))[1:length(result$yval)][dataFlows])
    yLimit <- unlist(lapply(temp,median)) + sdThreshold * unlist(lapply(temp,iqr))
    yLimit <- rep(yLimit,ceiling(length(result$yval)/flowCycleLength))[1:length(result$yval)]
  }
  badFlow <- (result$yval > yLimit & dataFlows)
  if (any(badFlow)) {
    for(i in which(badFlow)) {
      text(i,result$yval[i]-0.002,as.character(i),adj=c(0.5,1))
    }
  }

  title(analysisName)
  legend(flowLim[1]+0.6*(flowLim[2]-flowLim[1]),max(errLim),names(myCol),lwd=3,col=myCol,title="nuc",cex=0.7)
  legend(flowLim[1]+0.8*(flowLim[2]-flowLim[1]),max(errLim),c("7 or more","6","4 or 5","3","2 or fewer"),pch=c(24:21,25),title="flow delay",cex=0.7)
  return(flowLim)
}

plotErrDistribution <- function(z,analysisName="") {
  maxHp <- min(9,nrow(z$A)-1)
  z$All <- matrix(as.double(z$A) + as.double(z$C) + as.double(z$G) + as.double(z$T),nrow=nrow(z$A))
  fourNucs <- c("A","C","G","T")
  allNucs  <- c(fourNucs,"All")
  n <- nrow(z$All)
  errBaseCount <- abs(matrix(1:n,n,n) - matrix(1:n,n,n,byrow=TRUE))
  nBaseOver    <- lapply(z[allNucs],function(zz){apply(zz*lower.tri(zz)*errBaseCount,2,sum)})
  nBaseUnder   <- lapply(z[allNucs],function(zz){apply(zz*upper.tri(zz)*errBaseCount,2,sum)})
  nBaseCorrect <- lapply(z[allNucs],function(zz){diag(zz)*0:(n-1)})

  nBaseError <- sum(nBaseOver$All + nBaseUnder$All)
  pctOver  <- lapply(nBaseOver, function(zz){100*zz/nBaseError})
  pctUnder <- lapply(nBaseUnder,function(zz){100*zz/nBaseError})

  myColors <- c("green","blue","brown","red","black")
  myLwd    <- c(rep(2,4),3)
  myLty    <- 5:1
  names(myColors) <- allNucs
  names(myLwd) <- allNucs
  names(myLty) <- allNucs

  ylim <- range(unlist(pctOver[fourNucs]),-1*unlist(pctUnder[fourNucs]))
  plot(c(0,maxHp),ylim,type="n",xlab="HP Length",ylab="Percentage of Erroneous Bases (%)")
  grid(lwd=2)
  abline(h=0,lwd=2)
  for(nuc in fourNucs) {
    lines(0:maxHp,pctOver[[nuc]][1:(maxHp+1)],    col=myColors[nuc],type="l",lty=myLty[nuc], lwd=2)
    lines(1:maxHp,-1*pctUnder[[nuc]][2:(maxHp+1)],col=myColors[nuc],type="l",lty=myLty[nuc], lwd=2)
    lines(0:maxHp,pctOver[[nuc]][1:(maxHp+1)],     bg=myColors[nuc],type="p",pch=24)
    lines(1:maxHp,-1*pctUnder[[nuc]][2:(maxHp+1)], bg=myColors[nuc],type="p",pch=25)
  }
  legendText <- paste(c(round(sum(unlist(pctOver[fourNucs]))),round(sum(unlist(pctUnder[fourNucs])))),c("% Overcall","% Undercall"),sep="")
  legend("top",inset=0.01,legendText,pch=c(24,25),pt.bg="grey",cex=0.8)
  legendText <- paste(round(unlist(lapply(pctOver[fourNucs],sum))+unlist(lapply(pctUnder[fourNucs],sum))),"% ",fourNucs,sep="")
  legend("topright",inset=0.01,legendText,col=myColors[fourNucs],lty=myLty[fourNucs],lwd=2,cex=0.8)
  
  title(main="Characterization of Erroneous Bases")
  title(sub=analysisName,cex.sub=0.7)
}

plotHpErr <- function(z,analysisName="",addDepth=TRUE,phredScale=FALSE,errorType=c("total","over-under")) {
  errorType <- match.arg(errorType)
  maxNuc <- min(9,nrow(z$A)-1)
  z$All <- matrix(as.double(z$A) + as.double(z$C) + as.double(z$G) + as.double(z$T),nrow=nrow(z$A))
  fourNucs <- c("A","C","G","T")
  allNucs  <- c(fourNucs,"All")
  hpErr   <- lapply(z[allNucs],function(zz){1-(diag(zz)/apply(zz,2,sum))})
  hpOver  <- lapply(z[allNucs],function(zz){apply(zz*lower.tri(zz),2,sum)/apply(zz,2,sum)})
  hpUnder <- lapply(z[allNucs],function(zz){apply(zz*upper.tri(zz),2,sum)/apply(zz,2,sum)})
  myColors <- c("green","blue","brown","red","black")
  myLwd    <- c(rep(2,4),3)
  names(myColors) <- allNucs
  names(myLwd) <- allNucs
  if(phredScale) {
    plot(c(0,maxNuc),c(0,35),type="n",xlab="HP Length",ylab="Empirical Per-HP Phred Score")
    if(errorType == "over-under") {
      for(nuc in fourNucs) {
        lines(0:maxNuc,-10*log10(hpOver[[nuc]][1:(maxNuc+1)]), col=myColors[nuc],type="l",lty=1)
        lines(0:maxNuc,-10*log10(hpUnder[[nuc]][1:(maxNuc+1)]),col=myColors[nuc],type="l",lty=2)
        lines(0:maxNuc,-10*log10(hpOver[[nuc]][1:(maxNuc+1)]),  bg=myColors[nuc],type="p",pch=24)
        lines(0:maxNuc,-10*log10(hpUnder[[nuc]][1:(maxNuc+1)]), bg=myColors[nuc],type="p",pch=25)
      }
      legend("top",inset=0.01,c("Overcall","Undercall"),pch=c(24,25),pt.bg="grey",lty=c(1,2))
      legend("topright",inset=0.01,fourNucs,fill=myColors[fourNucs])
    } else {
      for(nuc in allNucs)
        lines(0:maxNuc,-10*log10(hpErr[[nuc]][1:(maxNuc+1)]),lwd=myLwd[nuc],col=myColors[nuc],type="b")
      legend("topright",inset=0.01,allNucs,fill=myColors[allNucs])
    }
  } else {
    plot(c(0,maxNuc),c(0,50),type="n",xlab="HP Length",ylab="Per-HP Error Rate (%)")
    if(errorType == "over-under") {
      for(nuc in fourNucs) {
        lines(0:maxNuc,100*hpOver[[nuc]][1:(maxNuc+1)], col=myColors[nuc],type="l",lty=1)
        lines(0:maxNuc,100*hpUnder[[nuc]][1:(maxNuc+1)],col=myColors[nuc],type="l",lty=2)
        lines(0:maxNuc,100*hpOver[[nuc]][1:(maxNuc+1)],  bg=myColors[nuc],type="p",pch=24)
        lines(0:maxNuc,100*hpUnder[[nuc]][1:(maxNuc+1)], bg=myColors[nuc],type="p",pch=25)
      }
      legend("top",inset=0.01,c("Overcall","Undercall"),pch=c(24,25),pt.bg="grey",lty=c(1,2))
      legend("topleft",inset=0.01,fourNucs,fill=myColors[fourNucs])
    } else {
      for(nuc in allNucs)
        lines(0:maxNuc,100*hpErr[[nuc]][1:(maxNuc+1)],lwd=myLwd[nuc],col=myColors[nuc],type="b")
      legend("topleft",inset=0.01,allNucs,fill=myColors[allNucs])
    }
  }
  grid()
  if(addDepth) {
    yaxp <- par("yaxp")
    yaxp.lim <- yaxp[1:2]
    yaxp.vals <- seq(yaxp[1],yaxp[2],length=1+yaxp[3])
    hpDepth <- lapply(z[c("A","C","G","T")],function(zz){log10(pmax(apply(zz,2,sum),1))})
    depth_range <- range(unlist(hpDepth),na.rm=TRUE)
    yaxp.labels <- as.character(round((10^seq(min(depth_range),max(depth_range),length=length(yaxp.vals)))/1e3))
    axis(4,at=yaxp.vals,lab=yaxp.labels,las=2,col.axis="grey")
    for(nuc in c("A","C","G","T"))
      lines(0:maxNuc, yaxp.lim[1] + (yaxp.lim[2]-yaxp.lim[1])*(hpDepth[[nuc]][1:(maxNuc+1)]/max(depth_range)),col=myColors[nuc],lty=3)
    mtext("Depth (1000's)",side=4,col="grey",line=3)
  }
  title(sub=analysisName,cex.sub=0.8)
}

errVsBase <- function(data,depthThreshold,errThreshold) {
  plotTypes <- c("err","ins","del","sub")
  result <- as.list(plotTypes)
  names(result) <- plotTypes
  for(type in plotTypes)
    result[[type]] <- errPlot(data,plotType=type,doPlot=FALSE)
  totalErr <- sum(result$err$yval*result$err$depth,na.rm=TRUE)/sum(result$err$depth,na.rm=TRUE)
  dataPositions <- result$err$depth/max(result$err$depth)>depthThreshold
  errLim <- range(c(0,errThreshold,result$err$yval[dataPositions]),na.rm=TRUE)
  posLim <- range(which(dataPositions))
  plot(posLim,errLim,xlab="Base Position",ylab="Error Rate",type="n",las=1)
  grid()
  myColors <- c("black","red","blue","green")
  for(i in 1:length(plotTypes)) {
    type <- plotTypes[i]
    result <- errPlot(data,plotType=type,newPlot=FALSE,addDepth=TRUE,lwd=2,col=myColors[i],type="l")
  }
  title(sprintf("%s\nTotal Erorr Rate = %1.1f%%",analysisName,round(totalErr,1)))
  legend("topright",inset=0.01,plotTypes,fill=myColors)
  return(totalErr)
}

getRegionCenters <- function(z) {
  temp <- lapply(z,function(zz){zz$region_origin + 0.5*zz$region_dim})
  temp <- matrix(unlist(temp),nrow=2)
  xValNames <- as.character(sort(unique(temp[1,])))
  xVal <- 1:length(xValNames)
  names(xVal) <- xValNames
  yValNames <- as.character(sort(unique(temp[2,])))
  yVal <- 1:length(yValNames)
  names(yVal) <- yValNames
  nCol <- length(xValNames)
  nRow <- length(yValNames)
  xBlock <- xVal[as.character(temp[1,])]
  yBlock <- yVal[as.character(temp[2,])]
  names(xBlock) <- NULL
  names(yBlock) <- NULL
  regionIndex <- (yBlock-1)*nCol + xBlock
  return(list(
    nCol=nCol,
    nRow=nRow,
    xCoord=as.numeric(xValNames),
    yCoord=as.numeric(yValNames),
    order=order(regionIndex),
    xBlock=xBlock,
    yBlock=yBlock
  ))
}

getRegionErrSummary <- function(z,depthThreshold,dataType="per_flow",valueType=c("accuracy","err","ins","del","sub","depth"),summaryType=c("weighted.mean","max","sum")) {
  valueType <- match.arg(valueType)
  summaryType <- match.arg(summaryType)

  regionError <- function(zz,valueType,summaryType,depthThreshold) {
    temp <- errSummary(zz$error_data)
    dataFlows <- temp$depth/max(temp$depth)>depthThreshold
    temp <- lapply(temp,function(x){x[dataFlows]})
    if(valueType == "accuracy") {
      val <- 1-temp$err/temp$depth
    } else if(valueType == "err") {
      val <- temp$err/temp$depth
    } else if(valueType == "ins") {
      val <- temp$ins/temp$depth
    } else if(valueType == "del") {
      val <- temp$del/temp$depth
    } else if(valueType == "sub") {
      val <- temp$sub/temp$depth
    } else if(valueType == "depth") {
      val <- temp$depth
    }
    if(summaryType == "weighted.mean") {
      res <- weighted.mean(val,temp$depth/sum(temp$depth))
    } else if(summaryType == "max") {
      res <- max(val)
    } else if(summaryType == "sum") {
      res <- sum(val)
    }
    return(res)
  }
  
  return(unlist(lapply(z,regionError,depthThreshold=depthThreshold,valueType=valueType,summaryType=summaryType)))
}
