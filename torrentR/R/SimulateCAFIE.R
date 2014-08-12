SimulateCAFIE <- function(
  seq,
  flowOrder,
  cf,
  ie,
  dr,
  nflows,
  hpScale      = 1,
  simModel     = c("treePhaserSim","CafieSolver","PhaseSim"),
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
  
  if ((length(cf) != 1) && (length(cf) != length(seq)))
    stop("Error in SimulateCAFIE: Lenght of <cf> must be 1 or equal to length(seq)")
  if (length(ie) != length(cf))
    stop("Error in SimulateCAFIE: Length of <ie> must be equal to length of cf.")
  if (length(dr) != length(cf))
    stop("Error in SimulateCAFIE: Length of <dr> must be equal to length of cf.")
  
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
  } else {
    droopType <- match.arg(droopType)
    val <- .Call("phaseSimulator", seq, flowOrder, conc, cf, ie, dr, hpScale, nflows, maxAdvances, droopType, extraTaps, PACKAGE="torrentR")
  }

  return(val)
}
