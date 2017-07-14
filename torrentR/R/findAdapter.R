# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# RcppExport SEXP findAdapter(SEXP Rsequence, SEXP Rsignal, SEXP RstartFlow, SEXP Rphasing,
#                            SEXP RflowCycle, SEXP Radapter, SEXP RtrimMethod)

findAdapter <- function(
		sequence,
		signal,
		phasing,
		scaledResidual=NA,
		flowOrder="TACGTACGTCTGAGCATCGATCGATGTACAGC",
		adapter="ATCACCGACTGCCCATAGAGAGGCTGAGAC",
		trimMethod=1
) {
	if(!is.matrix(signal))
		signal <- matrix(signal,nrow=1)
	if(!is.matrix(phasing))
		phasing <- matrix(phasing,nrow=1)
	if(!is.matrix(scaledResidual))
		scaledResidual <- matrix(scaledResidual,nrow=1)
	
	if(nchar(flowOrder) < ncol(signal))
		flowOrder <- substring(paste(rep(flowOrder,ceiling(ncol(signal)/nchar(flowOrder))),collapse=""),1,ncol(signal))
	
	val <- .Call("findAdapter", sequence, signal, phasing, scaledResidual, flowOrder, adapter, trimMethod, PACKAGE="torrentR")
	
	return(val)
}