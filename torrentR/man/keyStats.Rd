\name{keyStats}
\alias{keyStats}
\title{
  Compute summmary statistics based on key flows
}
\description{
  Given a matrix of raw signal intensities, keyStats() computes for each well summary metrics based on the key flows incuding the median and standard deviation of 0-mers and 1-mers in the key, as well as the total signal, noise and signal-to-noise ratio when discriminating between 1-mers and 0-mers.  
}
\usage{
  keyStats(measured,keySeq,flowOrder,sdFudge=0)
}
\arguments{
  \item{measured}{
    A matrix of signal intensities, one row per well and one column per flow.
  }
  \item{keySeq}{
    A character string representing the key sequence.
  }
  \item{flowOrder}{
    A character string representing the nucleotide flow order.
  }
  \item{sdFudge}{
    A fudge-factor to allow for avoidance of divide-by-zero issues when computing key SNR.  Any key standard deviation that is less than sdFudge is replaced by sdFudge.  Default value for sdFudge is 0.  The last 1-mer in the key is always ignored as its signal will be some mix of key and library sequence.
  }
}
\value{
  keyStats() returns a list whose elements are:
  \item{key_1_med, key_0_med}{
    Median value over the 1-mers,0-mers in the key flows.
  }
  \item{key_1_sd, key_0_sd}{
    Standard deviation over the 1-mers,0-mers in the key flows.
  }
  \item{key_sig}{
    Key signal - the difference between the median 1-mer and the median 0-mer
  }
  \item{key_sd}{
    Key standard deviation - the pooled standard deviation for the difference between 1-mers and 0-mers.  In other words, sqrt(key.0.sd^2 + key.1.sd^2)
  }
  \item{key_snr}{
    Key signal-to-noise ratio - the ratio of key_sig to key_sd.  If sdFudge is  positive then any key_sd values less than sdFudge are set to sdFudge.
  }
}
\author{
  Simon Cawley
}
