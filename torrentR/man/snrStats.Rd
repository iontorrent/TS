\name{snrStats}
\alias{snrStats}
\title{
  Signal-to-noise analysis for wells with known sequence.
}
\description{
  Given a matrix of signal values and a true sequence associated with them, this function computes a set of summary statistics relating to typical
  signals and noise of 0-mers and 1-mers as a function of flow.  The returned data are in a form suitable for passing to other related functions
  for plotting and further analysis.

  After determining the expected ideal flow given the true sequence, the raw data are key-normalized and CAFIE-corrected.  CAFIE-estimation is
  done with findBestCafie unless the CAFIE parameters are explicitly supplied.

  The cafie-corrected signal is then analyzed separately for flows contining known 0-mers and known 1-mers.  For both 0-mers and 1-mers two robust
  linear regresisons are performed, regressing the median and sd of signal on flow, and the fitted models are part of what is returned by the
  function.
}
\usage{
snrStats(trueSeq,flowOrder,keySeq,signal,cf=NA,ie=NA,dr=NA)
}
\arguments{
  \item{trueSeq}{
    Character string representing the assumed-known sequence for all the wells being analyzed.
  }
  \item{flowOrder}{
    Character string representing the nucleotide flow order.
  }
  \item{keySeq}{
    Character string representing the key sequence for all the wells being analyzed.
  }
  \item{signal}{
    The matrix of raw signal values to be analyzed.  One row per well, one column per flow.
  }
  \item{cf,ie,dr}{
    Optionally a set of pre-computed carry-forward, incomplete extension and droop estimates can
    be supplied.  If none are supplied they are estimated on-the-fly using findBestCafie.
  }
}
\value{
  snrStats() returns a list with elements listed below.  The list is suitable for passing on to the functions snrPlotMedSig, snrPlotSdSig and oneMerSNR.
  \item{trueHP}{
    The ideal expected signal (assuming no CAFIE or droop) based on the known sequence and the flow order.
  }
  \item{medSig}{
    Vector with the median signal for each flow.
  }
  \item{sdSig}{
    Vector with the standard deviation of signal for each flow
  }
  \item{fit.med}{
    A list of two elements named "0" and "1", each of which is the robust linear model fit returned by rlm() for 0-mer and 1-mer median signals respectively.
  }
  \item{fit.sd}{
    Like fit.med, but with robust linear model fits for standard deviations of signals.
  }
  \item{cf,ie,dr}{
    The carry-forward, incomplete-extension and droop values used in CAFIE-correction.
  }
}
\author{
  Simon Cawley
}
