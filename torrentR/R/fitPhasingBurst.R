fitPhasingBurst <- function(
		signal,
		flowOrder,
		readSequence,
		phase,
		motifFlow,
		maxEvalFlow,
		maxSimFlow
) {
	
	if(!is.matrix(signal))
		signal <- matrix(signal,nrow=1)
	if(!is.matrix(phase))
		phase <- matrix(phase,nrow=1)
	
	if(nchar(flowOrder) < ncol(signal)){
		flowOrder <- substring(paste(rep(flowOrder,ceiling(ncol(signal)/nchar(flowOrder))),collapse=""),1,ncol(signal))
	} else if (nchar(flowOrder) > ncol(signal)) {
		flowOrder <- substring(flowOrder, 1, ncol(signal))
	}
	
	if (length(readSequence) != nrow(signal))
		stop ("Error in FitIePhasingBurst: length of readSequence needs to be equal to the number of rows in signal.")
	if (nrow(phase) != nrow(signal))
		stop ("Error in FitIePhasingBurst: signal and phase need the same number of rows.")
	if (ncol(phase) != 3)
		stop ("Error in FitIePhasingBurst: phase needs 3 columns for cf, ie, dr.")
	if (length(motifFlow) != nrow(signal))
		stop ("Error in FitIePhasingBurst: length of motifFlow needs to be equal to the number of rows in signal.")
	if (length(maxEvalFlow) != nrow(signal))
		stop ("Error in FitIePhasingBurst: length of maxEvalFlow needs to be equal to the number of rows in signal.")
	
# Cpp code for this function in torrenR/src/treephaser.cpp
	val <- .Call("FitPhasingBurst", signal, flowOrder, readSequence, phase, motifFlow, maxEvalFlow, maxSimFlow, PACKAGE="torrentR")
	return(val)
}