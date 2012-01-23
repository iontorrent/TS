\name{bkgModel}
\alias{bkgModel}
\title{
  Fit the background model to raw dat data
}
\description{
  Fits the background model to raw Ion Torrent dat files to produce estimates of
  incorporation signals, along with a lot of additional parameters and diagnostics.
}
\usage{
  bkgModel(
    dat,
    bkgWell,
    fitWell,
    maxFlow=NA,
    sigma_guess=2.5,
    t0_guess=23,
    tau_guess=25.0,
    dntp_uM=50.0
  )
}
\arguments{
  \item{dat}{
    A list with dat data, such as can be obtained with \code{\link{readDatCollection}}.  The list must contain the following names elements
    \itemize{
      \item signal: A matrix of signal values, each row is a well and each column is a frame.  Flows are stored consecutively, so the number of columns should be equal to the number of frames times the number of flows.
      \item nFlow: The total number of flows.
      \item nFrame: The number of frames per flow.
    }
  }
  \item{bkgWell}{
    Used to specify the rows corresponding to the empty wells in the dat$signal matrix.
    Can be either a vector of Booleans of length equal to the number of wells, or a vector
    of integers in the range [1,nrow(dat$signal)].
  }
  \item{fitWell}{
    Used to specify the wells to which the background model should be applied.  As with bkgWell,
    this vector can be Boolean or integer.
  }
  \item{maxFlow}{
    Limit the fit to flows less than or equal to maxFlow (1-indexed).  By default all available
    flows will be fit.
  }
  \item{sigma_guess}{
    Initial sigma_guess estimate supplied to the background model.
  }
  \item{t0_guess}{
    Initial t0_guess estimate supplied to the background model.
  }
  \item{tau_guess}{
    Initial tau_guess estimate supplied to the background model.
  }
  \item{dntp_uM}{
    Initial dntp_uM estimate supplied to the background model.
  }
}
\value{
  The return value is a list with estimates and parameters from the model fit.  Full documentation
  of all the values is beyond the current scope of this man page, but here's a start:
  \item{sig}{
    Matrix of estimated incorporation signals, dimension is (nWell,nFlow).
  }
  \item{bkg}{
    Vector with estimated background trace supplied to the background model, length is (nFrame*nFlow).
  }
  \item{nFitFrame}{
    The number of frames modeled by the background model.  The model compresses some frames to the
    fitted background and signal that it returns have fewer frames than are passed in.
  }
  \item{fitFg}{
    Matrix of fitted dat-level data, dimension is (nWell,nFitFrame).
  }
  \item{fitBg}{
    Vector with fitted background trace returned by the background model, length is (nFitFrame*nFlow)
  }
  \item{offset}{
    Matrix of fitted offset values, dimension is (nWell*nFlow)
  }
  \item{tau}{
    Matrix of fitted tau values, dimension is (nWell*4) - i.e. one per nucleotide per well
  }
  \item{nFlowBatch}{
    The model is fit in batches, typically 8 flows per batch.  nFlowBatch is the number of flow batches that were fitted.
  }
  \item{R,dt,P,gain,tshift}{
    Matrices of fitted parameters, dimensions are (nWell,nFlowBatch) - i.e. one per well per flow batch.
  }
  \item{krate,d,mR,oR}{
    Matrices of fitted parameters, dimensions are (4,nFlowBatch) - i.e. one per nucleotide per flow batch.
  }
  \item{sigma}{
    Vector of fitted sigma values, length is nFlowBatch.
  }
  \item{sens,C}{
    Fitted scalar parameter values.
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readDatCollection}},
}
\examples{\dontrun{
analysisDir <- "/results2/analysis/output/IonWest/B6_60_trunk_r5928_SEC_2118"
minCol <- 300
maxCol <- 349
minRow <- 300
maxRow <- 349
maxFlow <- 16

# Read 16 flows from raw dats
dat <- readDatCollection(analysisDir=analysisDir,minCol=minCol,maxCol=maxCol,minRow=minRow,maxRow=maxRow,maxFlow=maxFlow)

# Read the beadfind mask
bfFile   <- sprintf("\%s/bfmask.bin",analysisDir)
bf   <- readBeadFindMask(bfFile)

# Determine the empty wells in the slice that was loaded
regionMask <- bf$col >= minCol & bf$col <= maxCol & bf$row >= minRow & bf$row <= maxRow
emptyMask <- bf$maskBead[regionMask]==0

# Fit the background model on the first 20 non-empty wells
bgFit <- bkgModel(dat,emptyMask,which(!emptyMask)[1:20])

# Insepct what's avaialble to play with in the returned object
str(bgFit)
}}
