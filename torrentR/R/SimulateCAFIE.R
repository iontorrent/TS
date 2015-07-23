SimulateCAFIE <- function(
  seq,
  flowOrder,
  cf,
  ie,
  dr,
  nflows,
  hpScale      = 1,
  simModel     = c("treePhaserSim","CafieSolver","DPPhaseSim","PhaseSim"),
  hpSignal     = 0:11,
  sigMult      = 1,
  conc         = diag(4),
  maxAdvances  = 2,
  droopType    = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
  extraTaps    = 0,
  getStates    = 0,
  diagonalStates = 0,
  RecalModelFile="",
  RecalModelThreshold=4,
  xval=NA,
  yval=NA
) {

  if(nflows<1)
    stop ("Error in SimulateCAFIE: argument <nflows> must be greater than zero.")
    
  simModel <- match.arg(simModel)
  if ((diagonalStates > 0) && (simModel != "treePhaserSim"))
    stop ("Error in SimulateCAFIE: Can only simulate diagonal states using treePhaserSim.")
  
  if(nchar(flowOrder) < nflows){
    flowOrder <- substring(paste(rep(flowOrder,ceiling(nflows/nchar(flowOrder))),collapse=""),1,nflows)
  } else if (nchar(flowOrder) > nflows) {
    flowOrder <- substring(flowOrder, 1, nflows)
  }

  if((length(hpScale) != 1) & (length(hpScale) != 4))
    stop("hpScale must be of length 1 or 4\n");
  

  if(!is.matrix(cf))
	cf <- matrix(cf,nrow=1)
  size_ok_cf <- all(dim(cf)==c(1,1)) || all(dim(cf)==c(1,length(seq))) || all(dim(cf)==c(nflows,nflows))
  if (!size_ok_cf)
    stop("Error in SimulateCAFIE: Lenght of <cf> must be 1, equal to length(seq), or a nflowsXnflows matrix (DPPhaseSim).")

  if(!is.matrix(ie))
	ie <- matrix(ie,nrow=1)
  if ( any(dim(ie) != dim(cf)) )
    stop("Error in SimulateCAFIE: Dimension of <ie> must be equal to length of cf.")

  if(!is.matrix(dr))
	dr <- matrix(dr,nrow=1)
  if ( any(dim(dr) != dim(cf)) )
	stop("Error in SimulateCAFIE: Dimension of <dr> must be equal to length of cf.")

  
  if (RecalModelFile != "") {
    if (simModel != "treePhaserSim")
      stop("Error in SimulateCAFIE: Recalibration only works with simModel <treePhaserSim>")
    if (any(is.na(xval)) | (length(xval) != length(seq)))
      stop("Error in SimulateCAFIE: If recalibration file is provided <xval> and <yval> need to be provided for every read.")
    if (any(is.na(yval)) | (length(yval) != length(seq)))
      stop("Error in SimulateCAFIE: If recalibration file is provided <xval> and <yval> need to be provided for every read.")
  }

  if (simModel == "treePhaserSim") {
    val <- .Call("treePhaserSim", seq, flowOrder, cf, ie, dr, nflows, getStates, diagonalStates, RecalModelFile, RecalModelThreshold, xval, yval, PACKAGE="torrentR")
  } else if(simModel == "CafieSolver") {
    val <- .Call("SimulateCAFIE", seq, flowOrder, cf, ie, dr, nflows, hpSignal, sigMult, PACKAGE="torrentR")
  } else if(simModel == "DPPhaseSim") {
	  val <- .Call("DPPhaseSim", seq, flowOrder, cf, ie, dr, nflows, getStates, conc, PACKAGE="torrentR")
  } else {
    droopType <- match.arg(droopType)
    val <- .Call("phaseSimulator", seq, flowOrder, conc, cf, ie, dr, hpScale, nflows, maxAdvances, droopType, extraTaps, PACKAGE="torrentR")
  }

  return(val)
}
