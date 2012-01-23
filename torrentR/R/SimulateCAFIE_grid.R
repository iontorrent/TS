SimulateCAFIE_grid <- function(seq, flowOrder, cfMin, cfMax, ieMin, ieMax, dr, nFlow, nBin=50, header="") {

  cfRange <- seq(cfMin,cfMax,length=nBin)
  ieRange <- seq(ieMin,ieMax,length=nBin)
  sigMat <- rep(list(matrix(NA,nBin,nBin)),nFlow)
  for(cfCounter in 1:length(cfRange)) {
    for(ieCounter in 1:length(ieRange)) {
      sim = SimulateCAFIE(seq, flowOrder, cfRange[cfCounter], ieRange[ieCounter], dr, nFlow+20)
      for(pos in 1:nFlow) {
        sigMat[[pos]][cfCounter,ieCounter] <- sim[pos]
      }
    }
  }

  for(pos in 1:nFlow) {
    imageWithHist(sigMat[[pos]],header=sprintf("%s flow %02d",header,pos),nHistBar=50)
  }
}
