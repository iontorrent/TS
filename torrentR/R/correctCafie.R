correctCafie <- function(measured,flowOrder,keySeq,caf,ie,droop) {

	# Compute the keyFlow by comparing keySeq with flowOrder
	keyFlow <- seqToFlow(keySeq,flowOrder)
        nKeyFlow <- length(keyFlow)-1

	# If measured is a vector then transform it to a matrix
	if(is.null(dim(measured)))
	    measured <- matrix(measured,1,length(measured))

	if(ncol(measured) > nchar(flowOrder)) {
	    stop("flowOrder not long enough for supplied data")
	} else if(ncol(measured) < nchar(flowOrder)) {
	    flowOrder <- substr(flowOrder,1,ncol(measured))
	}
      
	val <- .Call("correctCafie",
          measured,
          flowOrder,
          keyFlow,
          nKeyFlow,
          caf,
          ie,
          droop,
          PACKAGE="torrentR"
        )
	val
}
