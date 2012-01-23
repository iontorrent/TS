keyStats <- function(measured,keySeq,flowOrder,sdFudge=0) {
  # Compute the keyFlow by comparing keySeq with flowOrder
  keyFlow  <- seqToFlow(keySeq,flowOrder,finishAtSeqEnd=TRUE)
  nKeyFlow <- length(keyFlow)-1
  keyFlow  <- keyFlow[1:nKeyFlow]

  # If measured is not a matrix, make it so
  if(is.null(dim(measured)))
    measured <- matrix(measured,1,length(measured))

  # trim measured to just the key
  measured <- measured[,1:nKeyFlow]

  val <- .Call("keyStats",
    measured, keyFlow,
    PACKAGE="torrentR"
  )

  return(val)
}
