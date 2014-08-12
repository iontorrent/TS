\name{tnormalizeRead}
\alias{normalizeRead}
\title{
  Normalize reads.
}
\description{
  Normalize (key normalized) reads w.r.t a predicted signal.
}
\usage{
normalizeRead(
		signal,
		prediction,
		method=c("adaptive", "gain", "pid"),
		windowSize=0,
		numSteps=0,
		startFlow=0,
        endFlow=0
)
}
\arguments{
  \item{signal}{
    The matrix of observed signal values, one row per read and one column per flow.
  }
  \item{prediction}{
    The matrix of predicted signal values, one row per read and one column per flow.
  }
  \item{method}{
    Selecting the normalization method to be used:
    adaptive: Fitting a windowed additive and multiplicative offset.
    gain:     Fitting a multiplicative offset between startFlow and EndFlow.
    pid:      Fitting a multiplicative offset through a PID controller.
  }
  \item{windowSize}{
    Window size for adaptive normalization. If zero, the built-in default of the c++ code is used.
  }
  \item{numSteps}{
    The number of steps for adaptive normalization. One step computes one window 
    and interpolates between the last and current window. If 0, the number of
    steps is set to num_flows/windowSize.
  }
  \item{startFlow}{
    Flow at which to start gain of PID normalization.
  }
  \item{endFlow}{
    Flow at which to end gain or PID normalization. (Open interval)
  }
}
\value{
  The return value is a list with the following elements.
  \item{method}{
    String indicating the normalization method used.
  }
  \item{normalized}{
    Matrix containing the normalized signal values.
  }
}
\author{
  Christian Koller
}
