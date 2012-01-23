\name{flowToSeq}
\alias{flowToSeq}
\title{
  Convert flow-space values into DNA seqeunce
}
\description{
  Given a matrix of flow values and a flow order, returns the sequence under ideal circumstances - i.e. in the absence of phase errors or signal decay.
}
\usage{
  flowToSeq(flowVals,flowOrder)
  
}
\arguments{
  \item{flowVals}{
    Matrix of flow values with one row per sequence and one column per flow.  Values will be rounded to nearest integer.
  }
  \item{flowOrder}{
    A DNA string specifying the flow sequence.  Will be cycled as necessary.
  }
}
\value{
  The return value is a character vector representing the DNA sequence under ideal circumstances.
}
\author{
  Simon Cawley
}
\examples{
mySeq1     <- "TCAGCTTGTAACAGGTCAGTTACCGTCCGTCCACGCCGCCGCG"
mySeq2     <- "TCAGGCAATCAACTGGCGAAACTGGAACCGATTGTTTCGGTA"
flowOrder <- "TACG"
flowVals1  <- seqToFlow(mySeq1,flowOrder,nFlow=16)
flowVals2  <- seqToFlow(mySeq2,flowOrder,nFlow=16)
flowToSeq(rbind(flowVals1,flowVals2),flowOrder)
}
\seealso{
  \code{\link{seqToFlow}},
}
