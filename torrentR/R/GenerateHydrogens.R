
GenerateRedHydrogensFromNucRise<-function(
    nucRise, sub_steps,
    deltaFrame, my_start, 
    maxConc, 
    amplitude, copies,
    krate, kmax, diffusion
){

	val <- .Call("CalculateCumulativeIncorporationHydrogensR",
	      nucRise,sub_steps,
        deltaFrame, my_start,
        maxConc, 
        amplitude,copies, 
        krate,kmax,diffusion,
          PACKAGE="torrentR"
        )
  return(val)
}

SimplifyGenerateRedHydrogensFromNucRise<-function(
    nucRise, sub_steps,
    deltaFrame, my_start, 
    maxConc, 
    amplitude, copies,
    krate, kmax, diffusion
){

	val <- .Call("SimplifyCalculateCumulativeIncorporationHydrogensR",
	      nucRise,sub_steps,
        deltaFrame, my_start,
        maxConc, 
        amplitude,copies, 
        krate,kmax,diffusion,
          PACKAGE="torrentR"
        )
  return(val)
}



ComplexGenerateRedHydrogensFromNucRise<-function(
    nucRise, sub_steps,
    deltaFrame, my_start, 
    maxConc, 
    amplitude, copies,
    krate, kmax, diffusion
){

	val <- .Call("ComplexCalculateCumulativeIncorporationHydrogensR",
	      nucRise,sub_steps,
        deltaFrame, my_start,
        maxConc, 
        amplitude,copies, 
        krate,kmax,diffusion,
          PACKAGE="torrentR"
        )
  return(val)
}

CalculateNucRise<-function(
  timeFrame, sub_steps,
  maxConc, t_mid_nuc, sigma,nuc_span=100
){
	val <- .Call("CalculateNucRiseR",
        timeFrame, sub_steps,
        maxConc, t_mid_nuc, sigma, nuc_span,
          PACKAGE="torrentR"
        )
  return(val)
}

CalculateNucRiseSpline<-function(
  timeFrame, sub_steps,
  maxConc, t_mid_nuc, sigma, tangent_zero=0,tangent_one=0
){
	val <- .Call("CalculateNucRiseSplineR",
        timeFrame, sub_steps,
        maxConc, t_mid_nuc, sigma, tangent_zero,tangent_one,
          PACKAGE="torrentR"
        )
  return(val)
}

CalculateNucRiseSigma<-function(
  timeFrame, sub_steps,
  maxConc, t_mid_nuc, sigma
){
	val <- .Call("CalculateNucRiseSigmaR",
        timeFrame, sub_steps,
        maxConc, t_mid_nuc, sigma,
          PACKAGE="torrentR"
        )
  return(val)
}

CalculateNucRiseMeasured<-function(
  timeFrame, sub_steps,
  maxConc, t_mid_nuc, sigma
){
	val <- .Call("CalculateNucRiseMeasuredR",
        timeFrame, sub_steps,
        maxConc, t_mid_nuc, sigma,
          PACKAGE="torrentR"
        )
  return(val)
}

ExportHiddenBkgParameters<-function(){
#dummy function to provide parameters that are hidden by the background model
  list(C=50,sens=1.256 * 2/100000)
}
