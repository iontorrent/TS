#hyperparameter models for buffering change over time

AdjustEmptyToBeadRatio<-function(
              etbR,
              NucModifyRatio,
              RatioDrift,
              flow
){

	val <- .Call("AdjustEmptyToBeadRatioForFlowR",
	      etbR,NucModifyRatio,RatioDrift,flow,
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
