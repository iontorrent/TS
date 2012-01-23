
CalculateEmphasisVector<-function(
  emphasisParams, hpLength,
  timeFrame, framesPerPoint,
  timeCenter, amplitudeMultiplier, emphasisWidth, emphasisAmplitude
){
	val <- .Call("CalculateEmphasisVectorR",
        emphasisParams, hpLength, 
        timeFrame, framesPerPoint,
        timeCenter, amplitudeMultiplier, emphasisWidth, emphasisAmplitude,
          PACKAGE="torrentR"
        )
  return(val)
}
