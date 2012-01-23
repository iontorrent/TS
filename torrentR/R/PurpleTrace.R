#three functions doing "hydrogen ion accounting"
#Purple does Red & Blue combined
#Red computes the trace for newly generated "red" hydrogen in the well
#note: red hydrogen is cumulative
#Blue computes the trace tracking the background measured in an empty well (above the well) "blue" hydrogen
#tauB = buffering*conductance in the well
#etbR = ratio of tauE/tauB - tauE = buffering*conductance in an empty well measuring the "blue" hydrogen
PurpleTrace<-function(
              BlueHydrogen,
              RedHydrogen,
              DeltaFrame,
              tauB, etbR
){

	val <- .Call("PurpleSolveTotalTraceR",
	      BlueHydrogen,RedHydrogen,DeltaFrame,tauB,etbR,
          PACKAGE="torrentR"
        )
  return(val)
}

RedTrace<-function(
              RedHydrogen, Istart,
              DeltaFrame,
              tauB
){

	val <- .Call("RedSolveHydrogenFlowInWellR",
	      RedHydrogen,Istart,DeltaFrame,tauB,
          PACKAGE="torrentR"
        )
  return(val)
}

BlueTrace<-function(
              BlueHydrogen,
              DeltaFrame,
              tauB, etbR
){

	val <- .Call("BlueSolveBackgroundTraceR",
	      BlueHydrogen,DeltaFrame,tauB,etbR,
          PACKAGE="torrentR"
        )
  return(val)
}
