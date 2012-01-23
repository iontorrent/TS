\name{phaseFit}
\alias{phaseFit}
\title{
  Estimate phase-related parameters from flow data.
}
\description{
  Given observed flow values and the assumed-true underlying sequence, returns esimates
  of phase-related parameters for one of a few different types of model.
}
\usage{
  phaseFit(
    trueSeq,
    sig,
    flowOrder,
    cf                 = 0.0100,
    ie                 = 0.0050,
    dr                 = 0.0015,
    hpScale            = 1,
    conc               = diag(4),
    maxAdvances        = 2,
    maxIter            = 30,
    droopType          = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
    fitType            = c("CfIeDr","CfIeDrHpScale","HpScale","CfIeDrHpScale4","HpScale4","CfIe","NucContam","CfIe4","NucContamIe"),
    resType            = c("SQUARED","ABSOLUTE","GEMAN_MCCLURE"),
    resSummary         = c("MEAN","MEDIAN","MEAN_OF_MEDIAN"),
    ignoreHPs          = FALSE,
    flowWeight         = NULL,
    maxErr             = 1,
    extraTaps          = 0
  )
}
\arguments{
  \item{trueSeq}{
    The assumed-known sequence underlying the signal values.  Can be either a vector of DNA strings or a matrix
    of flow values, one row per read and one column per flow.
  }
  \item{sig}{
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
    Estimate for the 4x4 nucleotide concentration matrix.  Column and row order is ACGT.  The value in
    row i and colum j is the amount of nucleotide j that is present when flowing nucleotide i.
    The default is to use the identity matrix.
  }
  \item{maxAdvances}{
    The maximum number of homopolymer stretches that can be extended in a single flow.
  }
  \item{maxIter}{
    The maximum number of iterations in the LevMar fit used to optimized phase parameter estimates.
  }
  \item{droopType}{
    The droop model used - can be either "ONLY_WHEN_INCORPORATING" (the default) or "EVERY_FLOW".
  }
  \item{fitType}{
    The phase model to use.  Available models are:
    \itemize{
      \item "CfIe" - fit a single carry-forward and a single incomplete extension parameter.
      \item "CfIeDr" - fit one carry-forward, one incomplete extension and one droop parameter.
      \item "CfIeDrHpScale" - fit one carry-forward, one incomplete extension, one droop and one hpScale parameter.
      \item "HpScale" - fit one hpScale parameter.
      \item "CfIeDrHpScale4" - fit one carry-forward, one incomplete extension, one droop and four nuc-specific hpScale parameters.
      \item "HpScale4" - fit four nuc-specific hpScale parameters.
      \item "NucContam" - fit the 12 off-diagonal elements of a nucleotide contamination matrix.
      \item "CfIe4" - fit a single carry-forward and 4 nucleotide-specific incomplete extension parameters.
      \item "NucContamIe" - fit a single incomplete extension parameter and the 12 off-diagonal elements of a nucleotide contamination matrix.
    }
  }
  \item{resType}{
    The type of residuals to use when evaluating the fit.  Options are:
    \itemize{
      \item "SQUARED" - r^2
      \item "ABSOLUTE" - |r|
      \item "GEMAN_MCCLURE" - x^2 / (2*(1+x^2))
    }
  }
  \item{resSummary}{
    The way transformed residuals should be summarized whene valuating the model fit.  Options are:
    \itemize{
      \item "MEAN" - the weighted mean residual across all reads and flows.
      \item "MEDIAN" - the median residual for all reads and flows with positive weight.
      \item "MEAN_OF_MEDIAN" - the average of the per-read median residuals.
    }
  }
  \item{ignoreHPs}{
    If set to true then flows involving a homopolymer stretch of more than 1 will be ignored during the fit.
  }
  \item{flowWeight}{
    A vector of weights in the range [0,1] to allow for down-weighting or ignoring certain flows during the fit.
    Length must be equal to number of flows.  Default is to set weight to 1 for every flow.
  }
  \item{maxErr}{
    The max rounding error before we start ignoring the read.  This only applies when the seuqnce is not
    set explicitly but is determined by rounding the signal.  The first time the difference between a signal
    and the nearest int is larger than this value, the weights for the rest of the read and for the 4 flows before
    are set to zero.
  }
  \item{extraTaps}{
    Controls the amount of extra flows to apply after each nuc flow.  The idea is to model situations where
    extra flows are applied to try drive to complete extension, though signal isn't actually collected on these
    flows.
  }
}
\value{
  The return value is a list with the following slots.
  \item{nIter}{
    The number of LevMar iterations to reach convergence.
  }
  \item{param.*}{
    Any fitted parameters returned will be in slots prefixed by "param."
  }
  \item{residualSummarized}{
    The optimum summarized residual value that was obtained by the LevMar fit.
  }
  \item{residualRaw}{
    A matrix of residuals (observed signal minus fitted) corresponding to the the fitted parameters.  One row
    per read and one column per flow.
  }
  \item{residualWeighted}{
    As above, but residuals are weighted as in the LevMar fit.
  }
}
\seealso{
  \code{\link{SimulateCAFIE}}, \code{\link{phaseSolve}},
}
\author{
  Simon Cawley
}
