calcHypothesesDistances <- function(
  signal,
  cf,
  ie,
  dr,
  flowOrder,
  hypotheses,
  startFlow = 0,
  normalize=0,
  verbose = 0
) {

  if(nchar(flowOrder) < length(signal))
    flowOrder <- substring(paste(rep(flowOrder,ceiling(length(signal)/nchar(flowOrder))),collapse=""),1,length(signal))

  val <- .Call("calcHypothesesDistances", signal, cf, ie, dr, flowOrder, hypotheses, startFlow, normalize, verbose, PACKAGE="torrentR")

  return(val)
}
