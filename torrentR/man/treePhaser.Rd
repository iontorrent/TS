\name{treePhaser}
\alias{treePhaser}
\title{
  Estimate DNA sequence from flow data.
}
\description{
  Given observed flow values and phasing parameters for the process that generated
  them, returns an esimate of the underlying DNA sequence.
}
\usage{
  treePhaser(
    signal,
    flowOrder,
    cf,
    ie,
    dr,
    keySeq      = "",
    basecaller  = c("treephaser-swan", "dp-treephaser", "treephaser-adaptive", "treephaser-solve"),
    diagonalStates=0,
    RecalModelFile="",
    RecalModelThreshold=4,
    xval=NA,
    yval=NA
  )
}
\arguments{
  \item{signal}{
    The matrix of observed signal values, one row per read and one column per flow.
  }
  \item{flowOrder}{
    The flow cycle - for example "TACG".
  }
  \item{cf,ie,dr}{
    Estimates for cf, ie and dr.
  }
  \item{keySeq}{
    The known key sequence at the start of the read - will be used
    for key normalization.  If not specified no key normalization will
    be performed.
  }
  \item{basecaller}{
    The variant of treePhaser to be used - options are 
    "treephaser-swan"      Sliding window adaptive normalization (default)
    "treephaser-adaptive"  Adaptive normalization
    "dp-treephaser"        Solve with older normalization method
    "treephaser-solve"     Solve without doing any normalization
  }
  \item{diagonalStates}{
    Switch to enable a diagonal state progression model.
  }
  \item{RecalModelFile}{
    Filename of a HP recalibration file to be loaded.
  }
  \item{RecalModelThreshold}{
    Lower (inclusive) threshold for model HP recalibration. (default 4)
  }
  \item{xval}{
    x coordinates of the wells. Required when RecalModelFile is provided.
  }
  \item{yval}{
    y coordinates of the wells. Required when RecalModelFile is provided.
  }
}
\value{
  The return value is a list with the following elements.
  \item{seq}{
    The estimated sequence.
  }
  \item{predicted}{
    The flow values predicted from the estimated sequence.
  }
  \item{residual}{
    The flow residuals (observed flow values minus predicted).
  }
  \item{hpFlow}{
    The estimated number of bases per flow.
  }
}
\seealso{
  \code{\link{SimulateCAFIE}},
}
\examples{
key   <-"TCAG"
mySeq <- paste(key,"CGCCAGGCGTTGAAGATACGCAGCGGGGCAAGCTATCCAAGGCTTCGG",sep="")
flow  <- "TACGTACGTCTGAGCATCGATCGATGTACAGC"
cf    <- 0.01
ie    <- 0.005
dr    <- 0.001
nflow <- 100
data.sim <- SimulateCAFIE(mySeq,flow,cf,ie,dr,nflow)
data.sol <- treePhaser(data.sim$sig,flow,cf,ie,dr,key)
}
\author{
  Simon Cawley
}
