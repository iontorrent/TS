phaseSolve <- function(
  signal,
  flowOrder,
  cf,
  ie,
  dr,
  hpScale              = 1,
  conc                 = diag(4),
  droopType            = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
  maxAdvances          = 2,
  nIterations          = 3,
  residualScale        = TRUE,
  residualScaleMinFlow = -1,
  residualScaleMaxFlow = -1,
  extraTaps            = 0,
  debugBaseCall        = FALSE
) {

  droopType  <- match.arg(droopType)

  if(!is.matrix(signal))
    signal <- matrix(signal,nrow=1)

  if((length(hpScale) != 1) & (length(hpScale) != 4))
    stop("hpScale must be of length 1 or 4\n");

  val <- .Call("phaseSolve", signal, flowOrder, conc, cf, ie, dr, hpScale, droopType, maxAdvances, nIterations, residualScale, residualScaleMinFlow, residualScaleMaxFlow, extraTaps, debugBaseCall, PACKAGE="torrentR")

  return(val)
}
