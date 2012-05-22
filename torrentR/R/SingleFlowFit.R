
SingleFlowFit<-function(
    observed,
    nucRise, sub_steps,
    deltaFrame, my_start, 
    Astart, Kmultstart,
    maxConc, 
    amplitude, copies,
    krate, kmax, diffusion, sens, tauB
){

	val <- .Call("SingleFlowFitR",
        observed,
	      nucRise,sub_steps,
        deltaFrame, my_start,
        Astart, Kmultstart,
        maxConc, 
        amplitude,copies, 
        krate,kmax,diffusion, sens, tauB,
          PACKAGE="torrentR"
        )
  return(val)
}
