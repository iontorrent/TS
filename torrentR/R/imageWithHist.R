imageWithHist <- function(z,zlim=NA,header="",histLim=NA,xaxt="n",yaxt="n",nHistBar=100,cex.header=1,col=rgb(rep(0,256),seq(0,1,length=256),seq(1,0,length=256)),...) {
  # Image plot with a histogram

  nColor <- length(col)

  # Possible trimming of values
  if(!any(is.na(zlim))) {
    if(length(zlim) != 2)
      stop("zlim must be of length 2")
    zlim <- sort(zlim)
  } else {
    zlim <- quantile(z,probs=c(0.02,0.98),na.rm=TRUE)
  }
  z[z < min(zlim)] <- min(zlim)
  z[z > max(zlim)] <- max(zlim)

  # Plot setup
  par(
    omd=c(0.1,0.95,0.1,0.8),
    omi=c(0.2,0.2,0.4,0.2),
    mar=c(1,2,1,1),
    cex.axis=0.8
  )
  layout(matrix(1:2,2,1),heights=c(2,1))

  # The image plot
  image(z,zlim=zlim,col=col,xaxt=xaxt,yaxt=yaxt,...)

  # The histogram
  histVal <- z[!is.na(z)]
  histVal <- histVal[histVal >= min(zlim) & histVal <= max(zlim)]
  zRange <- zlim %*% c(-1,1)
  if(any(is.na(histLim)))
    histLim <- range(histVal)
  histBar <- nHistBar
  temp <- hist(
    histVal,
    breaks=seq(zlim[1]-0.01*zRange,zlim[2]+0.01*zRange,length=histBar+1),
    plot=FALSE
  )
  plot(
    histLim,
    range(temp$counts),
    yaxt="n",
    ylab="",
    xlab="",
    main="",
    xlim=zlim,
    type="n"
  )
  colorIndex <- 1+floor(nColor*((1:histBar)-1)/histBar)
  barHalfWidth <- (temp$breaks[2]-temp$breaks[1])/2
  for(i in 1:histBar) {
    xlim <- temp$breaks[i] + barHalfWidth * c(-1,1)
    polygon(xlim[c(1,2,2,1)],c(rep(0,2),rep(temp$counts[i],2)),col=col[colorIndex[i]],border=col[colorIndex[i]])
  }
  if(header != "")
    mtext(header,outer=TRUE,line=0,cex=cex.header)
}

