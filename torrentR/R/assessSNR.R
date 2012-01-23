snrStats <- function(trueSeq,flowOrder,keySeq,signal,cf=NA,ie=NA,dr=NA) {

  # Determine true homopolymer sequence
  trueHP <- seqToFlow(trueSeq,flowOrder)
  nHP <- sum(!is.na(trueHP))
  hpType <- unique(trueHP[!is.na(trueHP)])

  # Cafie correction
  if(is.na(cf) | is.na(ie) | is.na(dr)) {
    cafie  <- findBestCafie(signal,flowOrder,trueSeq,keySeq)
    cf <- cafie$carryForward
    ie <- cafie$incompleteExtension
    dr <- cafie$droop
  }
  corrected <- correctCafie(signal,flowOrder,keySeq,cf,ie,dr)$corrected

  # Determine typical signals
  medSig <- apply(corrected,2,median)
  sdSig  <- apply(corrected,2,sdFromIQR)

  # Linear fits to 0-mer & 1-mer
  fit.med <- fit.sd  <- as.list(0:1)
  names(fit.med) <- names(fit.sd) <- c("0","1")
  for(mer in 0:1) {
    flow <- which(trueHP==mer & !is.na(trueHP))
    med  <- medSig[flow]
    sd   <- sdSig[flow]
    fit.med[[as.character(mer)]] <- rlm(med ~ flow)
    fit.sd[[as.character(mer)]]  <- rlm(sd  ~ flow)
  }

  return(list(
    trueHP  = trueHP,
    medSig  = medSig[1:nHP],
    sdSig   = sdSig[1:nHP],
    fit.med = fit.med,
    fit.sd  = fit.sd,
    cf      = cf,
    ie      = ie,
    dr      = dr
  ))
}

snrPlotMedSig <- function(
  z,
  xlab="Flow",
  ylab="Median Corrected Signal",
  pchOffset=21,
  colOffset=2,
  cex=3,
  cex.axis=1.5,
  cex.lab=1.5,
  cex.legend=1.5,
  ...
) {
  xRange <- 1:length(z$trueHP)
  plot(
    xRange,
    z$medSig,
    xlab=xlab,
    ylab=ylab,
    pch=pchOffset+z$trueHP,
    bg =colOffset+z$trueHP,
    cex=cex,
    cex.axis=cex.axis,
    cex.lab=cex.lab,
    ...
  )
  hpType <- unique(z$trueHP[!is.na(z$trueHP)])
  legend("topleft",paste(hpType,"-mer",sep=""),bty="n",pch=pchOffset+hpType,pt.bg=colOffset+hpType,cex=cex.legend)
  abline(h=0:3,lwd=2,lty=2,col="darkgrey")
  abline(z$fit.med[["0"]]$coeff,lwd=2)
  abline(z$fit.med[["1"]]$coeff,lwd=2)
}

snrPlotSdSig <- function(
  z,
  xlab="Flow",
  ylab="SD(Corrected Signal)",
  pchOffset=21,
  colOffset=2,
  cex=3,
  cex.axis=1.5,
  cex.lab=1.5,
  cex.legend=1.5,
  ...
) {
  xRange <- 1:length(z$trueHP)
  plot(
    xRange,
    z$sdSig,
    xlab=xlab,
    ylab=ylab,
    pch=pchOffset+z$trueHP,
    bg =colOffset+z$trueHP,
    cex=cex,
    cex.axis=cex.axis,
    cex.lab=cex.lab,
    ...
  )
  hpType <- unique(z$trueHP[!is.na(z$trueHP)])
  legend("topleft",paste(hpType,"-mer",sep=""),bty="n",pch=pchOffset+hpType,pt.bg=colOffset+hpType,cex=cex.legend)
  abline(z$fit.sd[["0"]]$coeff,lwd=2)
  abline(z$fit.sd[["1"]]$coeff,lwd=2)
}

oneMerSNR <- function(
  z,
  xlab="Flow",
  ylab="1-mer SNR (Corrected Signal)",
  ylim=NA,
  main="1-mer SNR vs position",
  cex=3,
  cex.axis=1.5,
  cex.lab=1.5,
  cex.legend=1.5,
  doPlot=TRUE,
  ...
) {
  xRange <- 1:length(z$trueHP)
  myData <- data.frame(flow=xRange)
  med.0 <- predict(z$fit.med[["0"]],myData)
  med.1 <- predict(z$fit.med[["1"]],myData)
  sd.0  <- predict(z$fit.sd[["0"]],myData)
  sd.1  <- predict(z$fit.sd[["1"]],myData)
  sig   <- med.1-med.0
  noise <- sqrt(predict(z$fit.sd[["1"]],myData)^2+predict(z$fit.sd[["0"]],myData)^2)
  snr   <- sig/noise

  if(any(is.na(ylim))) {
    ylim <- range(c(0,snr))
  }
  if(doPlot) {
    plot(
      xRange,
      snr,
      type="l",
      ylim=ylim,
      xlab=xlab,
      ylab=ylab,
      main=main,
      cex=cex,
      cex.axis=cex.axis,
      cex.lab=cex.lab,
      ...
    )
  } else {
    return(snr)
  }
}
