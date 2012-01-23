sdFromIQR <- function(x,na.rm=FALSE) {
  # Estimate sigma from IQR, for robustness
  iqrToSigma <- 1/(qnorm(0.75)*2)
  iqr <- (c(-1,1) %*% quantile(x,c(0.25,0.75),na.rm=na.rm))
  return(iqr*iqrToSigma)
}
