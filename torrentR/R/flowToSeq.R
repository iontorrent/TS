flowToSeq <- function(flowVals,flowOrder) {

  # ensure flowVals is a matrix
  if(!is.matrix(flowVals))
    flowVals <- matrix(flowVals,nrow=1)

  # determine nFlow
  nFlow <- ncol(flowVals)

  # Compute flowOrder
  f <- strsplit(flowOrder,"")[[1]]
  f <- rep(f,ceiling(nFlow/length(f)))[1:nFlow]

  return(apply(flowVals,1,function(x){paste(rep(f,x),collapse="")}))

}
