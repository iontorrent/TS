\name{SimulateCAFIE}
\alias{SimulateCAFIE}
\title{
  Simulate incorporation signal as a funciton of sequence and phasing parameters
}
\description{
  Given a DNA seqeunce and parameters describing the phasing process, returns
  simulated incorporate signal and associated quantities.  The function calls one
  of two different underlying models and some of the inputs are relevant for one
  and not the other - as indicated in the details below.
}
\usage{
  SimulateCAFIE(
    seq,
    flowOrder,
    cf,
    ie,
    dr,
    nflows,
    hpScale      = 1,
    simModel     = c("treePhaserSim","CafieSolver","PhaseSim"),
    hpSignal     = 0:7,
    sigMult      = 1,
    conc         = diag(4),
    maxAdvances  = 2,
    droopType    = c("ONLY_WHEN_INCORPORATING","EVERY_FLOW"),
    extraTaps    = 0,
    getStates    = 0,
    diagonalStates = 0,
    RecalModelFile="",
    RecalModelThreshold=4,
    xval=NA,
    yval=NA
  )
}
\arguments{
  \item{seq}{
    The DNA sequence for which to simulate.
  }
  \item{flowOrder}{
    The flow cycle - for example "TACG".
  }
  \item{cf,ie,dr}{
    Estimates for cf, ie and dr.  For simModel="CafieSolver" these must be be scalars.  For
    simModel="PhaseSim" they can be scalars or vectors.  If vectors, then values will be
    cycled over flows.
  }
  \item{nflows}{
    The number of flows to model.
  }
  \item{hpScale}{
    HpScaling factor - incorporation signals for an HP of length h will be modeled as h*hpScale^(h-1).  Can be of length 1 or 4, in the
    case of the latter it is interpreted as a vector of per-nuc values in the order A,C,G,T.
  }
  \item{simModel}{
    The simulation engine to use - can be either "CafieSolver" or "PhaseSim" or "treePhaserSim".  In the case of the
    first, simulation is performed by CafieSolver::SimulateCAFIE(), for the second PhaseSim::simulate() is used.
  }
  \item{hpSignal}{
    The homopolymer response - only used when simModel="CafieSolver".  Can be used to model
    nonlinear homopolymer response.  The first element gives the expected response for a 0mer,
    the second for a 1mer, and so on.  The number of elements should be equal to MAX_MER as
    defined in CafieSolver.h (or a warning will be delivered).
  }
  \item{sigMult}{
    Value by which to multiply all incorporation signals - only used when simModel="CafieSolver".
  }
  \item{conc}{
    The 4x4 nucleotide concentration matrix to use.  Column and row order is ACGT.  The value in
    row i and colum j is the amount of nucleotide j that is present when flowing nucleotide i.
    The default is to use the identity matrix.
  }
  \item{maxAdvances}{
    The maximum number of homopolymer stretches that can be extended in a single flow.
  }
  \item{droopType}{
    The droop model used - can be either "ONLY_WHEN_INCORPORATING" (the default) or "EVERY_FLOW".
  }
  \item{extraTaps}{
    Controls the amount of extra flows to apply after each nuc flow.  The idea is to model situations where
    extra flows are applied to try drive to complete extension, though signal isn't actually collected on these
    flows.
  }
    \item{getStates}{
    Advanced simulation that also logs the state of each incorporating homopolymer.
  }
    \item{diagonalStates}{
    Switch to enable a diagonal state progression model (only with treephaser simulator).
  }
  \item{RecalModelFile}{
    Filename of a HP recalibration file to be loaded. (only with treephaser simulator)
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
  The return value is a list, some of whose element depend on which simulation model is being used.
  \item{sig}{
    The simulated incorporation signal
  }
  \item{hpWeight}{
    Only returned when simModel="PhaseSim".  Returns a matrix recording the proportion of live templates in
    each state for each flow of the simulation.  One row per flow and one column for each state, where the
    first state is the null state (no incorporation) and the last state is full template incorporation.
  }
  \item{droopWeight}{
    Only returned when simModel="PhaseSim".  Identifies the proporation of drooped templates at each flow.
  }
}
\seealso{
  \code{\link{phaseFit}}, \code{\link{phaseSolve}},
}
\author{
  Stuart Davidson and Simon Cawley
}
