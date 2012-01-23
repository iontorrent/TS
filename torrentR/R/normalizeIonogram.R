normalizeIonogram <- function(measured,keySeq,flowOrder) {

	# Compute the keyFlow by comparing keySeq with flowOrder
	keyFlow <- seqToFlow(keySeq,flowOrder,finishAtSeqEnd=TRUE)
        nKeyFlow <- length(keyFlow)-1

	# If measured is not a matrix, make it so
	if(is.null(dim(measured)))
	  measured <- matrix(measured,1,length(measured))

	val <- .Call("normalizeIonogram",
          measured,
          keyFlow,
          nKeyFlow,
          flowOrder,
          PACKAGE="torrentR"
        )
	val
}
