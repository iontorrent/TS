\name{seqToFlow}
\alias{seqToFlow}
\title{
  Convert DNA sequence into flow-space representation
}
\description{
  Given a DNA sequence and a flow order, returns the flow-space signal that would be observed under ideal circumstances - i.e. in the absence of phase errors or signal decay.
}
\usage{
  seqToFlow(sequence,flowOrder,nFlow=NA,finishAtSeqEnd=FALSE,flowOffset=0)
  
}
\arguments{
  \item{sequence}{
    The DNA string to be transformed.  Any non [ACGT] characters will be silently ignored.
  }
  \item{flowOrder}{
    A DNA string specifying the flow sequence.  If nFlow is specified the flowOrder will be cycled as necessary.
  }
  \item{nFlow}{
    The maximum number of flows to return.  If not specified, the number of flows returned will
    be equal to the length of flowOrder.
  }
  \item{finishAtSeqEnd}{
    If true, returned flow sequence will end if/when the supplied template completes,
    otherwise the returned sequence will continue out to the number of flows.
  }
  \item{flowOffset}{
    If positive, specifies a number of initial flows that will be skipped, the corresponding flow
    values will be set to NA.  Designed to work with the value stored in the ZF tag in Ion BAM files,
    returned as flowOffset by readIonBam()
  }
}
\value{
  The return value is an integer vector representing the flow values expected under ideal circumstances.
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
}
\seealso{
  \code{\link{flowToSeq}},
  \code{\link{readIonBam}},
}
