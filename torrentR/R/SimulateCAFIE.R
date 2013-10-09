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
  terminatorChemistryRun = 0
) {

  if(nflows<1)
    stop ("Error in SimulateCAFIE: argument <nflows> must be greater than zero.")
    
  simModel <- match.arg(simModel)
  if ((terminatorChemistryRun > 0) && (simModel != "treePhaserSim"))
    stop ("Error in SimulateCAFIE: Can only simulate a terminator run using treePhaserSim.")
  
  if(nchar(flowOrder) < nflows){
    flowOrder <- substring(paste(rep(flowOrder,ceiling(nflows/nchar(flowOrder))),collapse=""),1,nflows)
  } else if (nchar(flowOrder) > nflows) {
    flowOrder <- substring(flowOrder, 1, nflows)
  }

  if((length(hpScale) != 1) & (length(hpScale) != 4))
    stop("hpScale must be of length 1 or 4\n");

  if (simModel == "treePhaserSim") {
    val <- .Call("treePhaserSim", seq, flowOrder, cf, ie, dr, nflows, getStates, terminatorChemistryRun, PACKAGE="torrentR")
  } else if(simModel == "CafieSolver") {
    val <- .Call("SimulateCAFIE", seq, flowOrder, cf, ie, dr, nflows, hpSignal, sigMult, PACKAGE="torrentR")
  } else {
    droopType <- match.arg(droopType)
    val <- .Call("phaseSimulator", seq, flowOrder, conc, cf, ie, dr, hpScale, nflows, maxAdvances, droopType, extraTaps, PACKAGE="torrentR")
  }

  return(val)
}
