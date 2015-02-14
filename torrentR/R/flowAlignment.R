flowAlignment <- function(
		targetSeq,
		querySeq,
		scaledRes,
		flowOrder,
		startFlow
) {
  
  if (length(targetSeq) != length(querySeq))
    stop("Error in flowAlignment: targetSeq and querySeq must have the same number of elements.")
  if(!is.matrix(scaledRes))
    scaledRes <- matrix(scaledRes,nrow=1)
  if (nrow(scaledRes) != length(targetSeq))
    stop("Error in flowAlignment: nrows(scaledRes) must be the same as the number of elements in targetSeq.")
  if (ncol(scaledRes) != nchar(flowOrder))
	stop("Error in flowAlignment: ncol(scaledRes) must be the length of flowOrder.")


  # Remove non-base characters from basestrings
  targetSeq <- gsub("[^ACGT]", "", targetSeq)
  querySeq  <- gsub("[^ACGT]", "",  querySeq)

  val <- .Call("flowAlignment", targetSeq, querySeq, scaledRes, flowOrder, startFlow, PACKAGE="torrentR")
  return(val)
	
}