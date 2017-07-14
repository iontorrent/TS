#hyperparameter models for buffering change over time

AdjustEmptyToBeadRatio<-function(
              etbR,
              NucModifyRatio,
              RatioDrift,
              flow,fitTauE                         
){

	val <- .Call("AdjustEmptyToBeadRatioForFlowR",
	      etbR,NucModifyRatio,RatioDrift,flow,fitTauE,
          PACKAGE="torrentR"
        )
  return(val)
}

TauBFromLinearModel<-function(
                etbR,tau_R_m,tau_R_o
){

	val <- .Call("ComputeTauBfromEmptyUsingRegionLinearModelR",
	      etbR,tau_R_m,tau_R_o,
          PACKAGE="torrentR"
        )
  return(val)
}

TauBFromLinearModelUsingTauE<-function(
                etbR,tauE
){

	val <- .Call("ComputeTauBfromEmptyUsingRegionLinearModelUsingTauER",
	      etbR,tauE,
          PACKAGE="torrentR"
        )
  return(val)
}

CheckIfFittingTauE <- function(logFile, nrLinesToParse=500) {
  fitTauE = FALSE
  if (file.exists(logFile)){
    tmp = readLines(logFile,nrLinesToParse)
    for (i in 1:length(tmp)){
      if ((length(grep("Command line",tmp[i])) >0 ) & (length(grep("--fitting-taue true",tmp[i])) >0)){
        fitTauE = TRUE
        break
      }
    }
    return (fitTauE)
  }
  else
    print(sprintf("%s could not be found",logFile))
}

