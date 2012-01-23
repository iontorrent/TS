bkgModel <- function(
  dat,
  bkgWell,
  fitWell,
  maxFlow=NA,
  sigma_guess=2.5,
  t0_guess=23,
  dntp_uM=50.0
) {

  bkgData <- dat$signal[bkgWell,]
  if(is.null(dim(bkgData)))
    bkgData <- matrix(bkgData,nrow=1)
  bkg <- fitBkgTrace(bkgData,dat$nFrame,dat$nFlow)

  sigData <- dat$signal[fitWell,]
  if(is.null(dim(sigData)))
    sigData <- matrix(sigData,nrow=1)
  if(is.na(maxFlow))
    maxFlow <- dat$nFlow-1
  val <- .Call("bkgModel",
    sigData,
    bkg,
    dat$nFrame,
    dat$nFlow,
    "TCAG",
    maxFlow,
    sigma_guess,
    t0_guess,
    dntp_uM,
    PACKAGE="torrentR"
  )
  val
}
