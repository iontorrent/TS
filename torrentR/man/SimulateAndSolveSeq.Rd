\name{SimulateAndSolveSeq}
\alias{SimulateAndSolveSeq}
\title{
  Simulates random base sequences given phase and noise paramters.
}
\description{
  The function SimulateAndSolveSeq Simulates random base sequences given phase and noise
  paramters. The return value is a matric of size [10, numBases+nchar(keySeq)] containing
  the fraction of reads with "row"th error occuring before or at base position "column".
}
\usage{
  SimulateAndSolveSeq <- function(
  numBases,
  numFlows,
  numWells,
  noiseSigma,
  PhaseParameters,
  flowOrder = "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  keySeq = "TCAG",
  noNegativeSignal=TRUE,
  plotFigure=TRUE)
}
\arguments{
  \item{numBases}{
    Number of bases in a random sequence
  }
  \item{numFlows}{
    Number of flows to be simulated
  }
  \item{numWells}{
    Number of sequences to be simulated
  }
  \item{noiseSigma}{
    Standard deviation of white gaussian noise that is added to the signal.
  }
  \item{PhaseParameters}{
    Phase parameters, vector of length 3 <CF, IE, Droop>
  }
   \item{flowOrder}{
     The flow order. Default: Samba "TACGTACGTCTGAGCATCGATCGATGTACAGC"
  }
  \item{keySeq}{
    The known key sequence at the start of the read. Default: "TCAG"
  }
  \item{noNegativeSignal}{
    If TRUE (default) no negative values can occur in the signal.
  }
  \item{plotFigure}{
    If TRUE (default) the function plots a figure.
  }
}
\value{
  The return value is list containing
  1) cumulativeErrorPos
    A matric of size [10, numBases+nchar(keySeq)] containing the fraction of reads 
    with "row"th error occuring before or at base position "column".
  2) meanQ17length
  3) meanQ20length
  4) meanQ30length
  5) meanQ47length
}
\examples{ 
	\dontrun{
        CumulativeErrorRate <- SimulateAndSolveSeq(250, 400, 1000, 0.08, c(0.007, 0.006, 0.001))
}
}
\author{
  Christian Koller, Nov. 12, 2012
}
