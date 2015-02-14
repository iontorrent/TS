normalizeRead <- function(
		signal,
		prediction,
		method=c("adaptive", "gain", "pid"),
		windowSize=0,
		numSteps=0,
		startFlow=0,
        endFlow=0
) {
    method <- match.arg(method)
	
	if(!is.matrix(signal))
		signal <- matrix(signal,nrow=1)
	if(!is.matrix(prediction))
		prediction <- matrix(prediction,nrow=1)
    
    if (ncol(signal) != ncol(prediction))
      stop("Error in nomarizeRead: signal and prediction must have the same number of columns.")
    if (nrow(signal) != nrow(prediction))
	  stop("Error in nomarizeRead: signal and prediction must have the same number of columns.")
    if(endFlow > startFlow)
	  stop("Error in nomarizeRead: endFlow needs to be larger than startFlow.")
    if(endFlow > ncol(signal))
	  stop("Error in nomarizeRead: endFlow must not exceed the number of flows.")
  
  val <- .Call("normalizeRead", signal, prediction, method, windowSize, numSteps, startFlow, endFlow, PACKAGE="torrentR")
  return(val)
}