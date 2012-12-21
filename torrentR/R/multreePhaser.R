multreePhaser <- function(
  signal,
  activeUntilFlow,
  numFlows,
  flowOrders,
  PhaseParameters,
  keySeq="TCAG",
  basecaller=c("treephaser-swan", "treephaser-adaptive"),
  verbose=0
) {

  basecaller <- match.arg(basecaller)
  
  # --- Checking consistency of input arguments ---
  if(!is.matrix(signal)){
    signal <- matrix(signal,nrow=1)
    print("Converting signal to Matrix with nrow=1")
  }
  if(!is.matrix(PhaseParameters)) {
    PhaseParameters <- matrix(PhaseParameters, nrow=1)
    print("Converting PhaseParameters to Matrix with nrow=1")
  }
  
  if((nrow(signal) %% nrow(PhaseParameters)) > 0)
    stop ("Error: The number of rows in <signal> must be divisable by the number of rows in <PhaseParameters>")
    
  if(nrow(signal) != length(activeUntilFlow))
    stop ("Error: The length of <activeUntilFlow> must be equal to the number of rows in <signal>")
  
  if (length(numFlows) == 1)
    numFlows <- rep(numFlows[1], nrow(PhaseParameters))
    
  if (length(flowOrders) == 1)
    flowOrders <- rep(flowOrders[1], nrow(PhaseParameters))
    
  if(length(numFlows) != length(flowOrders))
    stop ("Error: The length of <numFlows> needs to be be equal to the length of <flowOrders>")
  
  if(length(flowOrders) != nrow(PhaseParameters))
    stop ("Error: The length of <flowOrders> needs to be the same as the number of rows in <PhaseParameters>")

  if(ncol(PhaseParameters) != 2)
    stop ("Error: <PhaseParameters> needs to have 2 columns <carry forward, incomplete extension>")
  # ---
    
  # Expand flow order and determine key flows
  keyFlows <- vector(mode="list")
  numKeyFlows <- vector(length=length(numFlows))
  for (i in 1:length(numFlows)) {
    if(nchar(flowOrders[i]) < numFlows[i])
      flowOrders[i] <- substring(paste(rep(flowOrders[i],ceiling(numFlows[i]/nchar(flowOrders[i]))),collapse=""),1,numFlows[i])
      
    if(keySeq=="") {
      keyFlow <- numeric()
    } else {
      keyFlow <- seqToFlow(keySeq,flowOrders[i],finishAtSeqEnd=TRUE)
    }

    name <- paste("keys",i,sep="")
    keyFlows[[name]] <- keyFlow
    numKeyFlows[i] <- length(keyFlow)
  }
  
  keyFlowMatrix <- mat.or.vec(length(numFlows), max(numKeyFlows))
  for (i in 1:length(numFlows)) {
    keyFlowMatrix[i,1:numKeyFlows[i]] <- keyFlows[[paste("keys",i,sep="")]]
  }
  
  # --- Finally call C++ function ---
  
  val <- .Call("multreePhaser", signal, activeUntilFlow, flowOrders, numFlows, PhaseParameters, keyFlowMatrix,
                numKeyFlows, basecaller, verbose, PACKAGE="torrentR")

  return(val)
}
