\name{plot_version}
\alias{plot_version}
\title{
  Add the torrentR version number to a plot
}
\description{
  Makes a call to mtext to write the torrentR version id to the current
  plotting device.
}
\usage{
  plot_version(
    side=1,
    line=4,
    adj=1,
    ...
  )
}
\arguments{
  \item{side}{
    The side of the plot to which the version string is added (1=bottom, 2=left, 3=top, 4=right).
  }
  \item{line}{
    The margin line to which text is added.  Starts at 0, counting outward.
  }
  \item{adj}{
    Text justification.  For strings parallel to the axes, 'adj = 0' means
    left or bottom alignment, and 'adj = 1' means right or top alignment.
  }
  \item{...}{
    Any additional arguments are passed right through to mtext().
  }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{torrentR_version}},
}
\examples{
  plot(rnorm(100))
  plot_version()
}
