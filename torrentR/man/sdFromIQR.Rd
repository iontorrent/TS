\name{sdFromIQR}
\alias{sdFromIQR}
\title{
  Robust estimate of SD, using Inter-Quartile Range
}
\description{
  A quick-and-dirty robust estimator of standard deviation from the Inter-Quartile Range of a collection of values.
  Computes the IQR and scales it by a factor that would return an estimate of the SD if the data were Gaussian.
}
\usage{
  sdFromIQR(x,na.rm=FALSE)
}
\arguments{
  \item{x}{
    The data
  }
  \item{na.rm}{
    If set the TRUE then NA values are ignored.  Default is FALSE.
  }
}
\value{
  An estimate of the Standard Deviation
}
\author{
  Simon Cawley
}
\examples{
sdFromIQR(rnorm(1000))
}
