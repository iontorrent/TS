\name{normalizeIonogram}
\alias{normalizeIonogram}
\title{
  Key-normalized wells data
}
\description{
  Scales all flows of each well such that the average of the key 1-mer signals is equal to 1.  This
  function assumes that all wells being normlized have the same key sequence.  It can handle being
  supplied with just a subset of all flows so long as all of the key flows are present, otherwise
  unexpected things may happen.
}
\usage{
  normalizeIonogram(
    measured,
    keySeq,
    flowOrder
  )
}
\arguments{
  \item{measured}{
    A matrix of raw signal values, a row for each well and a column for each flow.  It is fine
    to supply a subset of flows so long as all of the key flows are present, otherwise unexpected
    things may happen.
  }
  \item{keySeq}{
    The key sequence against which to normalize.
  }
  \item{flowOrder}{
    A single characer string specifying the flow order.  This string needs to be long enough to cover
    the key flows.
  }
}
\value{
  The return value is a list with a single element named normalized.
  \item{normalized}{
    A numeric vector of the key-normalized values, dimensions are the same
    as the measured vector supplied as input.
  }
}
\author{
  Simon Cawley
}
\examples{
  raw  <- matrix(rnorm(5000),nrow=100,ncol=50)
  norm <- normalizeIonogram(raw,"ATCG","TACGTACGTACG")
}
