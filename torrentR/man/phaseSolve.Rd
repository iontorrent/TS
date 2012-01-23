\name{phaseSolve}
\alias{phaseSolve}
\title{
  Estimate DNA sequence from flow data.
}
\description{
  Given observed flow values and phasing parameters for the process that generated
  them, returns an esimate of the underlying DNA sequence.
}
\usage{
  phaseSolve(
    signal,
    flowOrder,
    cf,
    ie,
    dr,
    hpScale              = 1,
    conc                 = diag(4),
    droopType            = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
    maxAdvances          = 2,
    nIterations          = 3,
    residualScale        = TRUE,
    residualScaleMinFlow = -1,
    residualScaleMaxFlow = -1,
    extraTaps            = 0
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
    Estimates for cf, ie and dr.  Can be scalars, if vectors then values will be cycled over flows.
  }
  \item{hpScale}{
    HpScaling factor - incorporation signals for an HP of length h will be modeled as h*hpScale^(h-1).  Can be of length 1 or 4, in the
    case of the latter it is interpreted as a vector of per-nuc values in the order A,C,G,T.
  }
  \item{conc}{
    The 4x4 nucleotide concentration matrix.  Column and row order is ACGT.  The value in
    row i and colum j is the amount of nucleotide j that is present when flowing nucleotide i.
    The default is to use the identity matrix.
  }
  \item{droopType}{
    The droop model used - can be either "ONLY_WHEN_INCORPORATING" (the default) or "EVERY_FLOW".
  }
  \item{maxAdvances}{
    The maximum number of homopolymer stretches that can be extended in a single flow.
  }
  \item{nIterations}{
    The maximum number of iterations of seqeunce estimation.
  }
  \item{residualScale}{
    if true, then enables signal rescaling based on residuals after each iteration of sequence estimation.
  }
  \item{residualScaleMinFlow,residualScaleMaxFlow}{
    The first and last flows to use for residual scaling.  0-based.  If set to -1 (the default) then
    whatever is the current default in the underlying C++ code will be used.
  }
  \item{extraTaps}{
    Controls the amount of extra flows to apply after each nuc flow.  The idea is to model situations where
    extra flows are applied to try drive to complete extension, though signal isn't actually collected on these
    flows.
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
  \item{multiplier}{
    If residualScale is set to TRUE then the multipliers used for each iteration are returned in this slot.  The
    multiplier used are returned in reverse order - the multiplier used for the last iteration is returned first.
  }
}
\seealso{
  \code{\link{SimulateCAFIE}}, \code{\link{phaseFit}},
}
\author{
  Simon Cawley
}
